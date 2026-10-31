//! 长上下文刀：单行长 prefill chunk 的 GPU 段分解 + TTFT（`WEIN_STEP_PROFILE`
//! 自激活——无须调用方记得）。
//!
//!   cargo run --release -p wein-exec --example qwen_prefill_profile -- [n ...]
//!
//! 每 n：admit 一个 prompt=n 的请求 → 单步吃满 n 个 prefill token（一行
//! × n token）→ 重复 R 次（默认 5），逐次墙钟 = 该 n 的 TTFT（GPU 侧，含
//! 提交/读回）。段表按 R 次累计给校正值/步（= /n 次 prefill 段）。
//!
//! 读数须知：单行 span 的 prefill 在 deltanet 上只起 32 个 block（同请求
//! 连续行合成一个 span），延迟主导——段表里 `delta` 段即该几何的账。
//! GPU 独占运行（单执行器 ~15GB）。
use std::path::Path;
use std::time::Instant;

use wein_engine::Executor;
use wein_exec::Qwen35Executor;
use wein_model::model_dir;
use wein_sched::StepPlan;

fn main() {
    let Some(dir) = model_dir() else {
        eprintln!("SKIP（模型目录不在场：WEIN_MODEL_DIR 或 ~/.cache/wein/Qwen3.5-4B）");
        return;
    };
    std::env::set_var("WEIN_STEP_PROFILE", "1"); // 须先于 open
    let ns: Vec<usize> = {
        let a: Vec<usize> = std::env::args()
            .skip(1)
            .filter_map(|t| t.parse().ok())
            .collect();
        if a.is_empty() { vec![250] } else { a }
    };
    let reps: usize =
        std::env::var("WEIN_PREFILL_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let mut ex = match Qwen35Executor::open(Path::new(&dir)) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("SKIP（Qwen35Executor 装配失败：{e}）");
            return;
        }
    };

    for n in ns {
        // prompt：真实 token（首 token 非 BOS 也无妨——这里量的是步时延）
        let prompt: Vec<u32> = (0..n as u32).map(|i| 1000 + (i * 37) % 2000).collect();
        // 冷启一步（模块/图/选中形状的启发式缓存）不计入
        ex.admitted(1, &prompt);
        ex.step(&StepPlan { prefill: vec![(1, n)], decode: vec![1], batch: vec![1] });
        ex.retain(&[]);
        let mut walls = Vec::new();
        for r in 0..reps {
            let id = 100 + r as u64;
            ex.admitted(id, &prompt);
            // 追平步段尾即采样位（plan.decode 含 id ⇒ 最后一个 prefill 行出 token）
            let plan = StepPlan { prefill: vec![(id, n)], decode: vec![id], batch: vec![id] };
            ex.step_profile_reset();
            let t0 = Instant::now();
            let out = ex.step(&plan);
            let wall = t0.elapsed().as_secs_f64() * 1e3;
            assert_eq!(out.len(), 1, "prefill 步每请求出一个 token");
            walls.push(wall);
            ex.retain(&[]);
        }
        let mut sorted = walls.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = sorted[sorted.len() / 2];
        println!(
            "\n===== n={n}（{reps} 次：TTFT wall 中位 {med:.2} ms；全部 {:?}）=====",
            walls.iter().map(|w| (w * 100.0).round() / 100.0).collect::<Vec<_>>()
        );
        println!("{}", ex.step_profile_report().expect("插桩已由 example 激活"));
        println!("（段表为 {reps} 次累计的每步值；单次 = 全程 n 个 token 的账）");
    }
}
