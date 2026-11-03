//! 满负荷账定位：m 行纯 decode 步的 GPU 段分解（WEIN_STEP_PROFILE 激活，
//! 自设环境变量后 open——无须调用方记得）。
//!
//!   cargo run --release -p wein-exec --example qwen_step_profile -- [m ...]
//!
//! 每 m：admit → 追平（prompt=3，m×3 ≤ T_MAX）→ N 步纯 decode → 段表
//! （校正列已按打点数扣事件记录成本）。m 列表默认 4 16 64 100 128；
//! N 默认 20（WEIN_PROF_STEPS）。GPU 独占运行（单执行器 ~15GB）。
//!
//! 读数须知（BASELINE §7）：m 小时步本身 host 受限（cuBLASLt 提交 ~25ms
//! 与 GPU 重叠），段内含饥饿间隙——段排名参考、绝对值看大 m（m≥64 时
//! host 提交 < GPU 段和，GPU-bound 成立，段表干净）。
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
    let ms: Vec<usize> = {
        let a: Vec<usize> = std::env::args()
            .skip(1)
            .filter_map(|t| t.parse().ok())
            .collect();
        if a.is_empty() { vec![4, 16, 64, 100, 128] } else { a }
    };
    let steps: usize =
        std::env::var("WEIN_PROF_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let mut ex = match Qwen35Executor::open(Path::new(&dir)) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("SKIP（Qwen35Executor 装配失败：{e}）");
            return;
        }
    };
    let prompt = vec![1u32, 2, 3]; // 追平 t = m×3 ≤ T_MAX@128

    for m in ms {
        let ids: Vec<u64> = (1..=m as u64).collect();
        for &id in &ids {
            ex.admitted(id, &prompt);
        }
        let plan = StepPlan {
            prefill: ids.iter().map(|&i| (i, prompt.len())).collect(),
            decode: ids.clone(),
            batch: vec![],
        };
        assert_eq!(ex.step(&plan).len(), m, "追平步各出一个 token");
        ex.step_profile_reset();
        let t0 = Instant::now();
        for _ in 0..steps {
            let out =
                ex.step(&StepPlan { prefill: vec![], decode: ids.clone(), batch: vec![] });
            assert_eq!(out.len(), m, "纯 decode 每请求每步一个 token");
        }
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;
        ex.retain(&[]); // 行回收——下一档 m 重用状态池
        println!("\n===== m={m}（{steps} 步，wall(含插桩) {wall_ms:.2} ms/步）=====");
        println!("{}", ex.step_profile_report().expect("插桩已由 example 激活"));
    }
    // 调优审计：每形状 chosen/argmin/best_ms（布局 A/B 的选型判别——
    // gemm 段漂移须先排除跨进程择优翻转）。末尾打——记录在首次调用后才有。
    if std::env::var("WEIN_TUNE_DUMP").is_ok() {
        for r in ex.gemm_tune_report() {
            if r.candidates > 0 {
                println!("tune m={} n={} k={} wt={} chosen={} argmin={} best={:.4}ms ws={}", r.m, r.n, r.k, r.wt, r.chosen, r.argmin, r.best_ms, r.ws);
            }
        }
    }
}
