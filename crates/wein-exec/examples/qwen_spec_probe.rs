//! MTP 投机解码的 exec 级性能读数（不进 battery）：R ∈ {1,4,16} 并发请求、
//! 各生成 32 token。四张表：
//!
//! [1] 固定 k 对照（spec_k ∈ {0,3}，策略强制关）——"投机到底有没有赚"；
//! [2] R=16 的 k 扫描（k ∈ {1,2,3}）——找拐点 + 校准自适应阈值；
//! [3] 步内分解（主前向 / 草稿 / 验证前向 / 落定 / 上传）+ 行账（链行分配、
//!     拷入次数、free_rows 长度）——换行零拷贝之后落定应只剩簿记；
//! [4] 自适应 k（策略开，k 由 EMA 自选）——打印实际 k 分布与切换次数。
//!
//! 用法：`cargo run --release -p wein-exec --example qwen_spec_probe`
//! 段级读数：`WEIN_STEP_PROFILE=1 cargo run --release ...`（[3] 里打段表）
//! 固定 k 读数：`WEIN_SPEC_ADAPTIVE=0 ...`（策略默认开，见 [4]）
//! 门控：无 CUDA 设备或无模型目录 → 打印 SKIP 退出；一次只跑一个进程
use std::time::Instant;

use wein_engine::Executor;
use wein_exec::Qwen35Executor;
use wein_model::{model_dir, HfTokenizer};
use wein_sched::StepPlan;

const N: usize = 32; // 每请求生成 token 数
const K: u32 = 3; // 固定档投机草稿数（[1]/[2]/[3] 用）

/// 一次配置的读数。除 tok/s 外都取自 `spec_stats_reset` 之后的累计
struct Meas {
    tps: f64,
    steps: u64,
    /// 接受率 = accepted / drafts
    acc: f64,
    tok_per_step: f64,
    /// 自适应 k：本轮结束时活跃请求的 k 分布 + 切换次数
    hist: Vec<(u32, usize)>,
    switches: u64,
    /// 行账：(链行分配次数, 拷入次数, 池行数, next_row, free_rows 长度)
    chain_allocs: u64,
    copyins: u64,
    row_cap: usize,
    next_row: u32,
    free_rows: usize,
    chain_max: u32,
    /// 段级报告（`WEIN_STEP_PROFILE` 未开 = None）——上传段读数从表头取
    prof: Option<String>,
    draft_ms: f64,
    fwd_ms: f64,
    fin_ms: f64,
}

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

    println!(
        "MTP 投机解码探针（Qwen3.5-4B / {}）：每请求 {N} token\n",
        ex.device_name()
    );

    // 预热：cuBLASLt heuristic/JIT 已在 open 期落缓存，先各走 2 步确认
    // 访存/页表/常驻上传缓冲就位（否则第一张表吃冷启动）
    ex.set_spec_policy(false);
    for k in [0u32, K] {
        warmup(&mut ex, &prompt, k);
    }

    // ── [1] 固定 k 对照 ──
    println!(
        "[1] 固定 k 对照（策略关）：\n{:>3}  {:>7}  {:>9}  {:>8}  {:>9}  {:>9}  {:>8}",
        "R", "spec_k", "tok/s", "加速比", "步数", "接受率", "出token/步"
    );
    let mut plain_tps: Vec<(usize, f64)> = Vec::new();
    let mut fixed: Vec<(usize, u32, Meas)> = Vec::new();
    for r in [1usize, 4, 16] {
        let p = run(&mut ex, &prompt, r, 0);
        println!(
            "{:>3}  {:>7}  {:>9.1}  {:>8}  {:>8}  {:>8}  {:>8.2}",
            r, 0, p.tps, "1.00×", p.steps, "-", 1.0
        );
        plain_tps.push((r, p.tps));
        let s = run(&mut ex, &prompt, r, K);
        println!(
            "{:>3}  {:>7}  {:>9.1}  {:>8.2}×  {:>8}  {:>7.1}%  {:>8.2}",
            r,
            K,
            s.tps,
            s.tps / p.tps,
            s.steps,
            s.acc * 100.0,
            s.tok_per_step
        );
        fixed.push((r, K, s));
    }
    let plain_of = |r: usize| plain_tps.iter().find(|x| x.0 == r).map(|x| x.1).unwrap_or(f64::NAN);

    // ── [2] R=16 的 k 扫描 ──
    // 接受率随 k 掉、草稿成本随 k 线性涨——拐点即自适应阈值的校准源
    let base16 = plain_of(16);
    println!("\n[2] R=16 的 k 扫描（基线 {base16:.1} tok/s）");
    println!("{:>3}  {:>9}  {:>8}  {:>8}  {:>9}  {:>8}", "k", "tok/s", "加速比", "步数", "接受率", "出token/步");
    for k in [1u32, 2, K] {
        let m = run(&mut ex, &prompt, 16, k);
        println!(
            "{:>3}  {:>9.1}  {:>8.2}×  {:>8}  {:>7.1}%  {:>8.2}",
            k,
            m.tps,
            m.tps / base16,
            m.steps,
            m.acc * 100.0,
            m.tok_per_step
        );
    }

    // ── [3] 步内分解 + 行账 ──
    decompose(&mut ex, &prompt);

    // ── [4] 自适应 k ──
    // 策略开（默认）：`set_spec(id, 3)` 只是上界，冷启动 k=2，之后由 EMA
    // 在 1/2/3/0 之间自选；基线仍取同 R 的 spec 关 tok/s
    ex.set_spec_policy(true);
    println!("\n[4] 自适应 k（策略开；阈值 EMA≥0.45→3 / ≥0.30→2 / <0.30→1 / 连续 3 轮<0.15→0）");
    println!(
        "{:>3}  {:>9}  {:>8}  {:>8}  {:>9}  {:>8}  {:>9}  {:>16}",
        "R", "tok/s", "加速比", "步数", "接受率", "出token/步", "k 切换", "实际 k 分布"
    );
    for r in [1usize, 4, 16] {
        let m = run(&mut ex, &prompt, r, K);
        let p = plain_of(r);
        println!(
            "{:>3}  {:>9.1}  {:>8.2}×  {:>8}  {:>7.1}%  {:>8.2}  {:>9}  {:>16}",
            r,
            m.tps,
            m.tps / p,
            m.steps,
            m.acc * 100.0,
            m.tok_per_step,
            m.switches,
            hist_str(&m.hist)
        );
    }
    ex.set_spec_policy(true);

    let worst_ratio = fixed
        .iter()
        .map(|(r, _, m)| m.tps / plain_of(*r))
        .fold(f64::INFINITY, f64::min);
    println!(
        "\n结论：固定 k={K} 最小加速比 {worst_ratio:.2}×；接受率 {}",
        fixed.iter().map(|x| format!("R{} {:.0}%", x.0, x.2.acc * 100.0)).collect::<Vec<_>>().join(" / ")
    );
}

fn hist_str(h: &[(u32, usize)]) -> String {
    h.iter().map(|(k, c)| format!("k{k}×{c}")).collect::<Vec<_>>().join(" ")
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

/// 一次配置的读数。k>0 时开投机（策略关 = 固定 k；策略开 = 上界 k）
fn run(ex: &mut Qwen35Executor, prompt: &dyn Fn(usize) -> Vec<u32>, r: usize, k: u32) -> Meas {
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
    ex.step_profile_reset();
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
    ex.spec_stats_reset();
    ex.step_profile_reset();
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
    let (rounds, drafts, accepted, _fb) = ex.spec_stats();
    let (hist, switches) = ex.spec_k_stats();
    let (chain_allocs, copyins, row_cap, next_row, free_rows) = ex.spec_row_stats();
    let (draft_ms, fwd_ms, fin_ms) = ex.spec_timing_ms();
    let chain_max = ex.spec_chain_batch_max();
    let prof = ex.step_profile_report();
    let acc = if drafts > 0 { accepted as f64 / drafts as f64 } else { 0.0 };
    let _ = rounds;
    ex.retain(&[]);
    Meas {
        tps: total as f64 / dt,
        steps,
        acc,
        tok_per_step: total as f64 / steps as f64,
        hist,
        switches,
        chain_allocs,
        copyins,
        row_cap,
        next_row,
        free_rows,
        chain_max,
        prof,
        draft_ms: draft_ms / steps as f64,
        fwd_ms: fwd_ms / steps as f64,
        fin_ms: fin_ms / steps as f64,
    }
}

/// [3] 步内耗时分解：主前向 / 草稿（全部链合计）/ 验证前向 / 落定 / 上传
fn decompose(ex: &mut Qwen35Executor, prompt: &dyn Fn(usize) -> Vec<u32>) {
    println!("\n[3] 步内耗时分解（每步均值；`落定` 已是换行零拷贝，只剩簿记）");
    println!(
        "{:>3}  {:>7}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>8}",
        "R", "spec_k", "主前向", "草稿", "验证前向", "落定", "上传", "合计ms", "出token/步"
    );
    for r in [1usize, 16] {
        // 同 R 的普通 decode 步（spec 关，1 token/步）——投机步的对照组
        let plain_ms = plain_step_ms(ex, prompt, r);
        for k in [0u32, K] {
            if k == 0 {
                println!(
                    "{:>3}  {:>7}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>8}",
                    r, 0, fmt_ms(plain_ms), "-", "-", "-", "-", fmt_ms(plain_ms), "1.00"
                );
                continue;
            }
            let m = run(ex, prompt, r, k);
            let sum = m.draft_ms + m.fwd_ms + m.fin_ms;
            println!(
                "{:>3}  {:>7}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>8}",
                r,
                k,
                fmt_ms(plain_ms), // 对照基准（同为 R 行的普通 decode）
                fmt_ms(m.draft_ms),
                fmt_ms(m.fwd_ms),
                fmt_ms(m.fin_ms),
                fmt_up(m.prof.as_deref()),
                fmt_ms(sum),
                format!("{:.2}", m.tok_per_step)
            );
            // 行账：链行分配 / 拷入 是每步摊销；next_row 与 free_rows 长度
            // 在稳态不涨 = 换行零拷贝不泄漏（多轮用例 qs13 在测试里钉）
            println!(
                "     行账：链行分配 {}/步，拷入 {}（{:.2}/步），链数上限 {}，池行 {}（next_row {}，free_rows {}）",
                m.chain_allocs,
                m.copyins,
                m.copyins as f64 / m.steps as f64,
                m.chain_max,
                m.row_cap,
                m.next_row,
                m.free_rows
            );
            if let Some(rep) = &m.prof {
                println!("--- R={r} spec_k={k} 段级分解（WEIN_STEP_PROFILE；仅主/验证那条 eager 前向）---\n{rep}");
            }
        }
    }
    println!(
        "  说明：k=0 行只有主前向（R 行 eager decode，走图档）；k={K} 行 = 草稿（k 次 MTP\n  \
         前向，批 = 链数）+ 验证前向（R×{} 行，eager）+ 落定（换行零拷贝：无 memcpy，\n  \
         只剩 seq/行表簿记）。上传段 = eager 混合批的常驻缓冲前缀 H2D（`WEIN_STEP_PROFILE`\n  \
         表头；图档的 `fx_*` 固定表不受影响）",
        K + 1
    );
}

/// 同 R 的普通 decode 步（spec 关）每步毫秒——投机步的对照组
fn plain_step_ms(ex: &mut Qwen35Executor, prompt: &dyn Fn(usize) -> Vec<u32>, r: usize) -> f64 {
    let ids: Vec<u64> = (1..=r as u64).collect();
    let prompts: Vec<Vec<u32>> = (0..r).map(prompt).collect();
    for (i, id) in ids.iter().enumerate() {
        ex.admitted(*id, &prompts[i]);
    }
    for i in 0..r {
        let (id, l) = (ids[i], prompts[i].len());
        let _ = ex.step(&StepPlan { prefill: vec![(id, l)], decode: vec![id], batch: vec![] });
    }
    let reps = 8;
    let t0 = Instant::now();
    for _ in 0..reps {
        let _ = ex.step(&StepPlan { prefill: vec![], decode: ids.clone(), batch: ids.clone() });
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    ex.retain(&[]);
    ms
}

fn fmt_ms(x: f64) -> String {
    format!("{x:.2}")
}

/// 从段级报告表头取"host：上传 x + 提交 y"里的上传毫秒（未开插桩 = "-"）
fn fmt_up(rep: Option<&str>) -> String {
    let Some(r) = rep else { return "-".into() };
    match r.find("上传 ").map(|i| &r[i + "上传 ".len()..]) {
        Some(rest) => match rest.split_whitespace().next().and_then(|x| x.parse::<f64>().ok()) {
            Some(v) => format!("{v:.2}"),
            None => "-".into(),
        },
        None => "-".into(),
    }
}
