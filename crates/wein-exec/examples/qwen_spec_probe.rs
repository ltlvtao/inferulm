//! MTP 投机解码的 exec 级性能读数（不进 battery）：R ∈ {1,4,16} 并发请求、
//! 各生成 32 token，spec_k ∈ {0,3}——墙钟 tokens/s + 接受率。
//!
//! 这是"投机到底有没有赚"的唯一产物：主前向每步 1 次（批 R 行），投机链
//! 每请求每步 k 次草稿前向（批 1 行）+ 1 次验证前向（链 cnt = k+1 行）+ 2 次
//! 25.2MB 递推状态拷入/拷回。若加速比 < 1，脚本同时打印每步分解（主前向 /
//! k 次草稿 / 验证含拷贝），决定下一段先修哪。
//!
//! 用法：`cargo run --release -p wein-exec --example qwen_spec_probe`
//! （门控：无 CUDA 设备或无模型目录 → 打印 SKIP 退出；一次只跑一个进程）
use std::time::Instant;

use wein_engine::Executor;
use wein_exec::Qwen35Executor;
use wein_model::{model_dir, HfTokenizer};
use wein_sched::StepPlan;

const N: usize = 32; // 每请求生成 token 数
const K: u32 = 3; // 投机草稿数

fn main() {
    let Some(dir) = model_dir() else {
        eprintln!("SKIP（模型目录不在场：WEIN_MODEL_DIR 或 ~/.cache/wein/Qwen3.5-4B）");
        return;
    };
    let mut ex = match Qwen35Executor::open(&dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("SKIP（Qwen35Executor 装配失败：{e}）");
            return;
        }
    };
    if !ex.has_mtp() {
        eprintln!("SKIP（checkpoint 无 MTP 草稿层权重 mtp.*）");
        return;
    }
    ex.set_radix(false); // 形状可预测（前缀命中会改 prefill 批形状）
    let tok = HfTokenizer::open(&dir).unwrap();
    let base = tok.encode(
        "The quick brown fox jumps over the lazy dog while the sun sets behind the distant \
         mountains and the river flows quietly through the ancient forest where birds sing \
         their evening songs before night falls over the sleeping village, and the farmers \
         walk home along the narrow road with lanterns in their hands, talking about the \
         harvest and the long winter that will soon cover the fields with snow and ice \
         while the children run ahead through the cold air laughing at the first flakes",
    );
    assert!(base.len() >= 64, "prompt 须 ≥ 64 token（实得 {}）", base.len());

    // 每请求一份不同 prompt（长度 32..=32+R）——避免同前缀/同步长偏置
    let prompt = |i: usize| -> Vec<u32> {
        let mut q = base.clone();
        q.truncate(32 + i);
        let l = q.len();
        q.rotate_left((i * 5) % l);
        q
    };

    // 预热：tokens/s 表只测稳态（cuBLASLt heuristic/JIT 已在 open 期落缓存，
    // 先走两个配置各 2 步确认访存/页表就位）
    for k in [0u32, K] {
        warmup(&mut ex, &prompt, k);
    }

    println!(
        "MTP 投机解码探针（Qwen3.5-4B / {}）：每请求 {N} token，k={K}\n",
        ex.device_name()
    );
    println!(
        "{:>3}  {:>7}  {:>9}  {:>9}  {:>8}  {:>9}  {:>8}",
        "R", "spec_k", "tok/s", "加速比", "步数", "接受率", "出token/步"
    );
    let mut rows: Vec<(usize, u32, f64, f64, f64)> = Vec::new();
    for r in [1usize, 4, 16] {
        let mut tps = Vec::new();
        for k in [0u32, K] {
            let m = run(&mut ex, &prompt, r, k);
            tps.push(m);
        }
        let (p_tps, s_tps) = (tps[0], tps[1]);
        let speedup = s_tps.0 / p_tps.0;
        println!(
            "{:>3}  {:>7}  {:>9.1}  {:>9.2}×  {:>8}  {:>8.1}%  {:>8.2}",
            r,
            0,
            p_tps.0,
            1.0,
            p_tps.1,
            0.0,
            1.0
        );
        println!(
            "{:>3}  {:>7}  {:>9.1}  {:>9.2}×  {:>8}  {:>8.1}%  {:>8.2}",
            r, K, s_tps.0, speedup, s_tps.1, s_tps.2 * 100.0, s_tps.3
        );
        rows.push((r, K, s_tps.0 / p_tps.0, s_tps.2, s_tps.1 as f64));
    }

    // 每步耗时分解（R=1）：主前向 1 行 vs k 次草稿前向 vs 验证（链 4 行 +
    // 25.2MB 拷入/拷回）。这是"加速比 < 1 时先修哪"的判据
    decompose(&mut ex, &prompt);

    let worst = rows.iter().map(|x| x.2).fold(f64::INFINITY, f64::min);
    println!(
        "\n结论：最小加速比 {worst:.2}×（R={}，k={K}）；接受率 {}",
        rows.iter().min_by(|a, b| a.2.total_cmp(&b.2)).map(|x| x.0).unwrap_or(0),
        rows.iter()
            .map(|x| format!("R{} {:.0}%", x.0, x.3 * 100.0))
            .collect::<Vec<_>>()
            .join(" / ")
    );
}

fn warmup(ex: &mut Qwen35Executor, prompt: &dyn Fn(usize) -> Vec<u32>, k: u32) {
    for i in 0..2usize {
        let id = 900 + i as u64;
        let q = prompt(i);
        let l = q.len();
        ex.admitted(id, &q);
        ex.set_spec(id, k);
        let _ = ex.step(&StepPlan { prefill: vec![(id, l)], decode: vec![id], batch: vec![] });
        for _ in 0..2 {
            let _ = ex.step(&StepPlan { prefill: vec![], decode: vec![id], batch: vec![] });
        }
    }
    ex.retain(&[]);
    ex.spec_stats_reset();
}

/// 一次配置的读数：(tokens/s, 步数, 接受率, 平均每步出 token)
fn run(
    ex: &mut Qwen35Executor,
    prompt: &dyn Fn(usize) -> Vec<u32>,
    r: usize,
    k: u32,
) -> (f64, u64, f64, f64) {
    let ids: Vec<u64> = (1..=r as u64).collect();
    let prompts: Vec<Vec<u32>> = (0..r).map(prompt).collect();
    for (i, id) in ids.iter().enumerate() {
        ex.admitted(*id, &prompts[i]);
    }
    if k > 0 {
        for id in &ids {
            ex.set_spec(*id, k);
        }
    }
    ex.spec_stats_reset();
    // chunked prefill（32 token/请求/步——R=16 时整段会超 T_MAX=512 行），
    // 尾段带采样位（出首 token）
    let mut emitted = vec![0usize; r];
    let mut off = vec![0usize; r];
    loop {
        let mut prefill = Vec::new();
        let mut decode = Vec::new();
        for (i, id) in ids.iter().enumerate() {
            if off[i] < prompts[i].len() {
                let take = 32.min(prompts[i].len() - off[i]);
                off[i] += take;
                prefill.push((*id, take));
                if off[i] == prompts[i].len() {
                    decode.push(*id);
                    emitted[i] += 1; // prefill 末位采样
                }
            }
        }
        if prefill.is_empty() {
            break;
        }
        let _ = ex.step(&StepPlan { prefill, decode, batch: vec![] });
    }
    let mut steps = 0u64;
    let t0 = Instant::now();
    while emitted.iter().any(|&n| n < N) {
        let out = ex.step(&StepPlan { prefill: vec![], decode: ids.clone(), batch: ids.clone() });
        for &(id, _) in &out {
            if let Ok(i) = ids.binary_search(&id) {
                emitted[i] += 1;
            }
        }
        steps += 1;
        assert!(steps < 4 * N as u64 + 8, "投机未收敛（emitted={emitted:?}）");
    }
    let dt = t0.elapsed().as_secs_f64();
    let total: usize = emitted.iter().sum();
    let (_rounds, drafts, accepted, _fb) = ex.spec_stats(); // 本轮（预热后已清零）
    let acc = if drafts > 0 { accepted as f64 / drafts as f64 } else { 0.0 };
    let per_step = total as f64 / steps as f64;
    ex.retain(&[]);
    (total as f64 / dt, steps, acc, per_step)
}

fn decompose(ex: &mut Qwen35Executor, prompt: &dyn Fn(usize) -> Vec<u32>) {
    let id = 500u64;
    let q = prompt(0);
    let l = q.len();
    ex.admitted(id, &q);
    ex.set_spec(id, K);
    let _ = ex.step(&StepPlan { prefill: vec![(id, l)], decode: vec![id], batch: vec![] });
    let reps = 12;
    let mut t_plain = 0.0f64;
    let mut t_draft = 0.0f64;
    let mut t_verify = 0.0f64;
    for _ in 0..reps {
        // 普通 decode（1 行主前向 + argmax）
        let t = Instant::now();
        let _ = ex.step(&StepPlan { prefill: vec![], decode: vec![id], batch: vec![] });
        t_plain += t.elapsed().as_secs_f64();
        // 草稿：k 次 MTP 前向（每次含 sync 回读）
        let t = Instant::now();
        let d = ex.spec_draft(id, K);
        t_draft += t.elapsed().as_secs_f64();
        assert_eq!(d.len(), K as usize, "草稿数须 = k");
        // 验证：链 cnt=4 行 eager 前向 + 落定行 25.2MB 拷回（+ 可能的拷入）
        let t = Instant::now();
        let r = ex.spec_verify(id, &d).expect("链行分配失败");
        t_verify += t.elapsed().as_secs_f64();
        let _ = r;
    }
    ex.retain(&[]);
    let ms = |x: f64| x / reps as f64 * 1e3;
    let total = ms(t_plain) + ms(t_draft) + ms(t_verify);
    println!(
        "\n每步耗时分解（R=1，k={K}，{reps} 轮均值）：主前向 {:.2} ms ｜ {K} 次草稿 {:.2} ms ｜ 验证（链 {} 行 + 25.2MB 状态拷）{:.2} ms ｜ 合计 {:.2} ms/投机步",
        ms(t_plain),
        ms(t_draft),
        K + 1,
        ms(t_verify),
        total
    );
    println!(
        "  对照：普通 decode 步 = {:.2} ms/token ⇒ 投机步 {:.2} ms 出 1..={} token",
        ms(t_plain),
        total,
        K + 1
    );
}
