//! 长上下文实测（W2 B3）：chunked prefill + decode，含 radix 前缀命中对照
//! 与"分段喂 == 单发"的一致性自证。
//!
//!   WEIN_CTX_LEN=4096 WEIN_KV_PAGES=512 \
//!   cargo run --release -p wein-exec --example qwen_longctx_probe -- [n] [gen]
//!
//! 三轮（同一段 3000 token 文本）：
//! ① 冷跑：chunk=256 逐 chunk prefill → 生成 gen 个 token（TTFT + 段表 +
//!    贪心 token 序列 = 基准）；
//! ② 同 prompt 再来一次（radix 快照命中）：命中 token 数 + TTFT 对照；
//! ③ 同 prompt 换 chunk=512（分段形状变化）：生成序列须与 ① **逐 token 同**
//!    ——分段只改批形，不改每行算术序。
//!
//! GPU 独占运行（单执行器 ~15GB，长上下文再 +KV 池/快照）。
use std::path::Path;
use std::time::Instant;

use wein_engine::Executor;
use wein_exec::Qwen35Executor;
use wein_model::{model_dir, HfTokenizer};
use wein_sched::StepPlan;

struct Turn {
    ttft_ms: f64,
    gen_ms: Vec<f64>,
    tokens: Vec<u32>,
    hit: usize,
    steps: usize,
}

fn main() {
    let Some(dir) = model_dir() else {
        eprintln!("SKIP（模型目录不在场）");
        return;
    };
    // 须先于 open；已设（如 =per 逐步档）则尊重外部值
    if std::env::var("WEIN_STEP_PROFILE").is_err() {
        std::env::set_var("WEIN_STEP_PROFILE", "1");
    }
    let args: Vec<usize> = std::env::args().skip(1).filter_map(|t| t.parse().ok()).collect();
    let n = args.first().copied().unwrap_or(3000);
    let gen = args.get(1).copied().unwrap_or(32);
    let envn = |k: &str, d: usize| -> usize {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    };
    let chunk = envn("WEIN_CHUNK", 256);
    let chunk2 = envn("WEIN_CHUNK2", 512);

    let mut ex = match Qwen35Executor::open(Path::new(&dir)) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("SKIP（装配失败：{e}）");
            return;
        }
    };
    println!(
        "长上下文探针（{}）：ctx_limit={} token；KV 池 = {} 页 × {} = {} token；快照栅格 {}",
        ex.device_name(),
        ex.ctx_limit(),
        wein_exec::kv_pages(),
        wein_exec::kv_block(),
        wein_exec::kv_pages() as usize * wein_exec::kv_block() as usize,
        std::env::var("WEIN_SNAP_STRIDE").unwrap_or_else(|_| "16".into())
    );
    let tok = HfTokenizer::open(&dir).unwrap();
    let base = tok.encode(
        "The quick brown fox jumps over the lazy dog while the sun sets behind the distant \
         mountains and the river flows quietly through the ancient forest where birds sing \
         their evening songs before night falls over the sleeping village, and the farmers \
         walk home along the narrow road with lanterns in their hands, talking about the \
         harvest and the long winter that will soon cover the fields with snow and ice \
         while the children run ahead through the cold air laughing at the first flakes \
         of the season, and the old bell in the church tower rings across the valley as \
         the lights come on one by one in the windows of the houses along the main street",
    );
    assert!(base.len() >= 64, "底稿太短：{}", base.len());
    let mut prompt: Vec<u32> = Vec::with_capacity(n);
    while prompt.len() < n {
        prompt.extend_from_slice(&base);
    }
    prompt.truncate(n);

    // A/B 模式（WEIN_AB=1，配 WEIN_RADIX=0 保两轮皆冷）：v3 关/开交替冷跑，
    // 同进程对照 TTFT/段表/贪心分叉（attn_long 每次调用现读 WEIN_ATTN_V3）
    if envn("WEIN_AB", 0) == 1 {
        // 暖场（gemm 调优/L2/模块惰性都在首轮；短 prompt 一发，结果不入表）
        let warm = &prompt[..256.min(prompt.len())];
        std::env::set_var("WEIN_ATTN_V3", "0");
        let _ = run_turn(&mut ex, 99, warm, 1, 256);
        let a = run_turn(&mut ex, 1, &prompt, gen, chunk);
        std::env::set_var("WEIN_ATTN_V3", "1");
        let b = run_turn(&mut ex, 2, &prompt, gen, chunk);
        let div = a.tokens.iter().zip(b.tokens.iter()).filter(|(x, y)| x != y).count();
        let p50a = p50(&a.gen_ms);
        let p50b = p50(&b.gen_ms);
        println!("\n| 臂 | WEIN_ATTN_V3 | 前缀命中 | TTFT ms | ITL p50 ms |");
        println!("|---|---|---|---|---|");
        println!("| 旧（在线 n8r4） | 0 | {} | {:.1} | {:.2} |", a.hit, a.ttft_ms, p50a);
        println!("| 新（v3） | 1 | {} | {:.1} | {:.2} |", b.hit, b.ttft_ms, p50b);
        println!(
            "\n贪心 {}t 分叉数：{}/{}（v3 是容差档，分叉允许存在，如实报数）",
            gen, div, a.tokens.len().min(b.tokens.len())
        );
        return;
    }

    // 冷跑（chunk）、第二次（radix 命中）、第三次（另一个 chunk 粒度）
    let t1 = run_turn(&mut ex, 1, &prompt, gen, chunk);
    let t2 = run_turn(&mut ex, 2, &prompt, gen, chunk);
    let t3 = run_turn(&mut ex, 3, &prompt, gen, chunk2);

    println!("\n| 轮 | chunk | 前缀命中 | prefill 步数 | TTFT ms | 生成 {:?} 均值 ms | ITL p50 ms |",
        gen);
    for (i, t) in [&t1, &t2, &t3].iter().enumerate() {
        let c = if i == 2 { chunk2 } else { chunk };
        let g = if t.gen_ms.is_empty() { 0.0 } else { t.gen_ms.iter().sum::<f64>() / t.gen_ms.len() as f64 };
        println!("| {} | {} | {} | {} | {:.1} | {:.2} | {:.2} |",
            i + 1, c, t.hit, t.steps, t.ttft_ms, g, p50(&t.gen_ms));
    }
    // 全长打印（不同进程间逐 token 对拍：贪心序列一致性是用户可见的正确性）
    println!("\n① tokens({}): {:?}", t1.tokens.len(), t1.tokens);
    println!("② tokens({}): {:?}", t2.tokens.len(), t2.tokens);
    println!("③ tokens({}): {:?}", t3.tokens.len(), t3.tokens);
    println!(
        "一致性：② ≡ ① {}（radix 命中恢复位同）／③ ≡ ① {}（分段粒度无关）",
        t2.tokens == t1.tokens,
        t3.tokens == t1.tokens
    );
    if t2.tokens != t1.tokens || t3.tokens != t1.tokens {
        std::process::exit(1);
    }
}

/// 单请求 decode ITL 的 p50（长上下文基线列：毫秒）
fn p50(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn run_turn(ex: &mut Qwen35Executor, id: u64, prompt: &[u32], gen: usize, chunk: usize) -> Turn {
    // 先清场：上一轮的请求行/块全部归还（快照按块移交 radix）
    ex.retain(&[]);
    ex.step_profile_reset();
    let t0 = Instant::now();
    let hit = ex.admitted(id, prompt);
    let mut off = hit;
    let mut steps = 0usize;
    let mut tokens = Vec::with_capacity(gen);
    let mut ttft = f64::NAN;
    // 逐 chunk 吃到 prompt 尾；末 chunk 与 decode 同拍取样（prefill 末位出 token）
    while off < prompt.len() {
        let c = chunk.min(prompt.len() - off);
        off += c;
        let last = off == prompt.len();
        let out = ex.step(&StepPlan {
            prefill: vec![(id, c)],
            decode: if last { vec![id] } else { vec![] },
            batch: vec![],
        });
        steps += 1;
        if last {
            ttft = t0.elapsed().as_secs_f64() * 1e3;
            if let Some(&(_, t)) = out.first() {
                tokens.push(t);
            }
        }
        let fails = ex.take_failures();
        if !fails.is_empty() {
            panic!("step 失败：{fails:?}");
        }
    }
    // 段表口径 = **prefill 段**（TTFT 点快照；decode 步的 gemm/attn 极小，
    // 混进来会稀释 chunk 间差异——chunk 扫描表按此对齐）
    let rep = ex.step_profile_report().unwrap_or_default();
    ex.step_profile_reset();
    let prefill_steps = steps;
    // decode 余下 gen-1 个 token
    let mut gen_ms = Vec::new();
    while tokens.len() < gen {
        let tk = Instant::now();
        let out = ex.step(&StepPlan { prefill: vec![], decode: vec![id], batch: vec![id] });
        gen_ms.push(tk.elapsed().as_secs_f64() * 1e3);
        steps += 1;
        let fails = ex.take_failures();
        if !fails.is_empty() {
            panic!("decode 失败：{fails:?}");
        }
        match out.first() {
            Some(&(_, t)) => tokens.push(t),
            None => break,
        }
    }
    // decode 段表（长上下文 decode 账定位：attn/delta/gemm 在 6k ctx 的单价）
    let rep_d = ex.step_profile_report().unwrap_or_default();
    println!("\n===== 请求 {id}：decode 段表（{} 步 @ ctx {}）=====", gen_ms.len(), prompt.len());
    println!("{rep_d}");
    println!("\n===== 请求 {id}：n={} chunk={chunk} 命中 {hit}｜TTFT {:.1} ms（prefill {prefill_steps} 步）=====",
        prompt.len(), ttft);
    println!("{rep}");
    Turn { ttft_ms: ttft, gen_ms, tokens, hit, steps: prefill_steps }
}
