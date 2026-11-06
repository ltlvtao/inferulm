//! 发射地图探针：m 行纯 decode 稳态步的**裸 kernel 序**（无 WEIN_STEP_PROFILE
//! 事件打点，只留给 nsys/ncu 抓）——用途是与 `nsys profile --stats=true`
//! 的 `cuda_gpu_kern_sum` 相除，得到"每步发数 / 每发 µs"。
//!
//!   ./target/release/examples/qwen_launch_map <m> <steps>
//!
//! 追平（prompt=3）1 步之后跑 `<steps>` 步纯 decode 再退出。因为 nsys 汇总
//! 覆盖整个进程（含追平那一步），发射地图口径用**两次不同 steps 相减**：
//! 同一二进制、同一 m，steps=A 与 steps=B 的实例数/总时差 ÷ (B−A) 即稳态
//! 每步值（追平步与初始化在两次里逐字相同，被减掉）。
use std::path::Path;

use wein_engine::Executor;
use wein_exec::Qwen35Executor;
use wein_model::model_dir;
use wein_sched::StepPlan;

fn main() {
    let Some(dir) = model_dir() else {
        eprintln!("SKIP（模型目录不在场）");
        return;
    };
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|t| t.parse().ok()).collect();
    let m = a.first().copied().unwrap_or(100);
    let steps = a.get(1).copied().unwrap_or(20);
    let mut ex = match Qwen35Executor::open(Path::new(&dir)) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("SKIP（装配失败：{e}）");
            return;
        }
    };
    let prompt = vec![1u32, 2, 3];
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
    for _ in 0..steps {
        let out = ex.step(&StepPlan { prefill: vec![], decode: ids.clone(), batch: vec![] });
        assert_eq!(out.len(), m, "纯 decode 每请求每步一个 token");
    }
    println!("done m={m} steps={steps}");
}
