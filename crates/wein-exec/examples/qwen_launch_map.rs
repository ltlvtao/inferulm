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
//!
//! `WEIN_GRAPH_BUCKETS=1,2,4,…,128` 改走 FULL_DECODE 图档（生产路径）：
//! 纯 decode 步即桶图回放，nsys 里能看到图内 kernel 与节点间空隙。
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
    let graph = std::env::var("WEIN_GRAPH_BUCKETS")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse::<u32>().ok()).collect::<Vec<_>>());
    let opened = match graph {
        Some(b) => Qwen35Executor::open_graph(Path::new(&dir), b),
        None => Qwen35Executor::open(Path::new(&dir)),
    };
    let mut ex = match opened {
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
    let first = ex.step(&plan);
    assert_eq!(first.len(), m, "追平步各出一个 token");
    // 对拍口径（WEIN_IDS_HASH=1）：把发出的 token id 逐排序累积成一个
    // FNV-1a 哈希打在末尾——同一输入下两次运行哈希相等 ⇔ 全序列位级同
    let hash = std::env::var("WEIN_IDS_HASH").is_ok();
    let mut h: u64 = 0xcbf29ce484222325;
    if let Ok(path) = std::env::var("WEIN_IDS_DUMP") {
        let _ = std::fs::write(&path, "");
    }
    let mut feed = |out: &[(u64, u32)]| {
        if let Ok(path) = std::env::var("WEIN_IDS_DUMP") {
            let mut v: Vec<(u64, u32)> = out.to_vec();
            v.sort_unstable();
            let line: String = v.iter().map(|(r, t)| format!("{r}:{t} ")).collect();
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(&path) {
                let _ = writeln!(f, "{line}");
            }
        }
        if !hash {
            return;
        }
        let mut v: Vec<(u64, u32)> = out.to_vec();
        v.sort_unstable();
        for (r, t) in v {
            for b in r.to_le_bytes().iter().chain(t.to_le_bytes().iter()) {
                h ^= *b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
    };
    feed(&first);
    let t0 = std::time::Instant::now();
    let mut done = 0usize;
    for _ in 0..steps {
        let out = ex.step(&StepPlan { prefill: vec![], decode: ids.clone(), batch: vec![] });
        // 到 EOS 的请求退场 ⇒ 行数掉档（图档会切小桶）——稳态口径到此为止
        assert!(out.len() <= m, "纯 decode 每请求每步至多一个 token");
        feed(&out);
        done += 1;
        if out.len() < m {
            eprintln!("（第 {done} 步有请求 EOS 退场：{m} → {}，口径在此截断）", out.len());
            break;
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / done as f64;
    println!(
        "done m={m} steps={done} wall={ms:.3}ms/step tps={:.1}{}",
        m as f64 * 1e3 / ms,
        if hash { format!(" ids_hash={h:016x}") } else { String::new() }
    );
}
