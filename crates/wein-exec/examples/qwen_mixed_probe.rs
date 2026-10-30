//! 混合步账户探针：d 行 decode 搭乘 p_total 个 prefill token 的步墙钟
//! （p99 尖峰定位——满负荷波次切换时统一预算把 ~400 prefill token 与
//! ~100 decode 行塞进同一步，T≈512 在 m=512 下 gemm 转计算受限）。
//!
//!   cargo run --release -p wein-exec --example qwen_mixed_probe -- [d:p ...]
//!
//! 每 d:p：d 个常驻 decode 请求（prompt=3，首步追平）+ 每迭代新 admit
//! k=ceil(p/17) 个 prompt=p 的请求做一步混合前向（T=d+p ≤ T_MAX），
//! 5 迭代各自墙钟（首迭代含 cuBLASLt 启发式/模块冷载——与后续差即
//! novel-shape 的 host 债）。GPU 独占运行。
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
    // "d:p" 列表；默认覆盖满负荷波次切换的 T 谱
    let cfgs: Vec<(usize, usize)> = {
        let a: Vec<String> = std::env::args().skip(1).collect();
        if a.is_empty() {
            vec![(100, 0), (100, 56), (100, 112), (100, 168), (100, 256), (100, 344), (100, 412), (0, 512)]
        } else {
            a.iter()
                .filter_map(|t| t.split_once(':').and_then(|(d, p)| Some((d.parse().ok()?, p.parse().ok()?))))
                .collect()
        }
    };
    let mut ex = match Qwen35Executor::open(Path::new(&dir)) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("SKIP（Qwen35Executor 装配失败：{e}）");
            return;
        }
    };
    let prompt = vec![1u32, 2, 3];
    let mut next_id: u64 = 1;

    for (d, p) in cfgs {
        // 常驻 decode 请求：追平后每步 1 token
        let decoders: Vec<u64> = (next_id..next_id + d as u64).collect();
        next_id += d as u64;
        for &id in &decoders {
            ex.admitted(id, &prompt);
        }
        ex.step(&StepPlan {
            prefill: decoders.iter().map(|&i| (i, prompt.len())).collect(),
            decode: decoders.clone(),
            batch: vec![],
        });
        let k = if p == 0 { 0 } else { p.div_ceil(17) }; // ≤27 新请求：行容量 127 内
        let per = if p == 0 { 0 } else { p / k };
        let mut walls = Vec::new();
        for _ in 0..5 {
            let fresh: Vec<u64> = (next_id..next_id + k as u64).collect();
            next_id += k as u64;
            for &id in &fresh {
                ex.admitted(id, &vec![1u32; per]);
            }
            let t0 = Instant::now();
            let mut batch = decoders.clone();
            batch.extend_from_slice(&fresh); // 追平者同步入 decode 采样（调度器语义）
            let out = ex.step(&StepPlan {
                prefill: fresh.iter().map(|&i| (i, per)).collect(),
                decode: batch,
                batch: vec![],
            });
            walls.push(t0.elapsed().as_secs_f64() * 1e3);
            assert_eq!(out.len(), d + k, "混合步：decode 全行 + 追平者各一个 token");
            ex.retain(&decoders); // 新请求离场——行回收，下一迭代重来
        }
        let t = d + p;
        let med = walls[1..].iter().sum::<f64>() / (walls.len() - 1) as f64;
        println!(
            "T={t:3} (d={d:3} + prefill {p:3} in {k:2} 请求): {:>6.2} ms 中位(warm) | 首发 {:.2} | 各迭代 {:.2?}",
            med,
            walls[0],
            walls
        );
        ex.retain(&[]); // 常驻 decode 请求也离场——下一档从头建行
    }
}
