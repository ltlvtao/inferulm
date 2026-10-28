//! W2-C 图档验收（qg 系列）——Qwen35Executor FULL_DECODE（真权重）。
//!
//! toy 侧机制（桶共享单输入存储 / sink 吸收 / 图回放 ≡ 同形状直发）由
//! w 系钉死；这里钉 32 层真模型装配：纯 decode 下 span/gather/采样行表
//! 是桶形的纯函数——固化上传一次，每步只覆写 ids/pos/seqs/rid/bt 五缓冲；
//! 设备端 argmax 烤进图尾（采样不回 logits 全词表）。
//!
//! 纪律（q 系 + w 系叠加）：单执行器 ≈15GB 显存，4090 装不下两个——
//! 对照执行器**先跑先放**串行；图捕获窗口与并行线程隐式同步互斥——
//! guard 全程持有。
//!
//! - qg1 · 图回放 ≡ 直发：图档 generate 与 eager 逐 token 相等（m 恒 1
//!        → 桶 1 精确命中）+ 图路径计数 ≥ decode 步数
//! - qg2 · sink 吸收：3 请求走桶 4（1 行 padding）vs eager 4 请求（第 4
//!        个 = 第 3 个克隆，补齐 m=4 形状位）——前 3 行 token 逐步相等
//! - qg3 · 步时延金丝雀：单请求 decode 20 步中位，图档 < eager（图档先
//!        测——热身漂移偏向后测的 eager，通过才是真收益）
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use wein_engine::Executor;
use wein_exec::Qwen35Executor;
use wein_model::{model_dir, HfTokenizer};
use wein_sched::StepPlan;

/// 显存 + 捕获窗口双重串行锁（q 系语义 + w 系捕获语义）
fn guard() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

fn mdir() -> Option<std::path::PathBuf> {
    let d = model_dir();
    if d.is_none() {
        eprintln!("SKIP（模型目录不在场：WEIN_MODEL_DIR 或 ~/.cache/wein/Qwen3.5-4B）");
    }
    d
}

fn exec(dir: &std::path::Path) -> Option<Qwen35Executor> {
    match Qwen35Executor::open(dir) {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("SKIP（Qwen35Executor 装配失败：{e}）");
            None
        }
    }
}

/// 图档执行器（FULL_DECODE 桶集合；NONE 档走 `open`）
fn graph_exec(dir: &std::path::Path, buckets: &[u32]) -> Option<Qwen35Executor> {
    match Qwen35Executor::open_graph(dir, buckets.to_vec()) {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("SKIP（图档装配失败：{e}）");
            None
        }
    }
}

/// qg1 · 图回放 ≡ 直发：同权重同 prompt，图档（桶{1}，m 恒 1 精确命中）
/// generate 与 eager 逐 token 相等——32 层 + 递推状态 + 分页 KV 的整条
/// kernel 序在图路径上零漂移
#[test]
fn qg1_replay_matches_eager() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let tok = HfTokenizer::open(&dir).unwrap();
    let prompt = tok.encode("The capital of France is");
    // 对照先跑先放（两份 15GB 显存装不下）
    let eager = {
        let Some(mut e) = exec(&dir) else { return };
        e.generate(&prompt, 10)
    };
    let Some(mut g) = graph_exec(&dir, &[1]) else { return };
    let got = g.generate(&prompt, 10);
    assert_eq!(got.len(), 10);
    assert_eq!(got, eager, "图档与 eager 须逐 token 相等");
    // 生成合同：首 token 在追平段尾采样，其后 max_tokens-1 个纯 decode 步
    assert_eq!(g.graph_replays(), 9, "9 个 decode 步全部走桶 1 回放");
    eprintln!("qg1 图档 10 token 逐位对齐 eager（回放 {} 次）", g.graph_replays());
}

/// qg2 · sink 吸收：图档 3 请求纯 decode（m=3 → 桶 4，1 行 padding）vs
/// eager 4 请求（第 4 个 = 第 3 个的克隆：同 prompt 同 pos，补齐 m=4 形状
/// 位）。sink 行的 KV/递推垃圾只落自己的页与状态行——前 3 行 token 逐步
/// 相等
#[test]
fn qg2_sink_row_absorbs_padding() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let tok = HfTokenizer::open(&dir).unwrap();
    let ps = [
        tok.encode("The capital of France is"),
        tok.encode("Water boils at a temperature of"),
        tok.encode("The largest planet is"),
    ];
    // 运行一（eager）：4 请求 = 3 真 + 1 克隆。**追平按请求逐个走**——
    // 两侧真实行全程同批形状（m 依赖的 GEMM algo 差异会从 catchup 注入
    // 微漂移，q8 的坑；克隆行的独走步不触真实行状态）
    let eager_steps = {
        let Some(mut e) = exec(&dir) else { return };
        let all = [&ps[0], &ps[1], &ps[2], &ps[2]];
        for (k, p) in all.iter().enumerate() {
            e.admitted(k as u64 + 1, p);
        }
        for (i, p) in all.iter().enumerate() {
            let id = i as u64 + 1;
            let plan =
                StepPlan { prefill: vec![(id, p.len())], decode: vec![id], batch: vec![] };
            assert_eq!(e.step(&plan).len(), 1);
        }
        let mut out = Vec::new();
        for _ in 0..8 {
            let o = e.step(&StepPlan { prefill: vec![], decode: vec![1, 2, 3, 4], batch: vec![] });
            assert_eq!(o.len(), 4);
            out.push(o);
        }
        out
    }; // eager 先放（显存归还）
    // 运行二（图档）：3 请求走桶 4（追平同样逐请求走，与运行一逐形对齐）
    let Some(mut g) = graph_exec(&dir, &[4]) else { return };
    for (k, p) in ps.iter().enumerate() {
        g.admitted(k as u64 + 1, p);
    }
    for (i, p) in ps.iter().enumerate() {
        let id = i as u64 + 1;
        let plan = StepPlan { prefill: vec![(id, p.len())], decode: vec![id], batch: vec![] };
        assert_eq!(g.step(&plan).len(), 1);
    }
    assert_eq!(g.graph_replays(), 0, "追平步是 prefill——必须 eager");
    for (k, et) in eager_steps.iter().enumerate() {
        let og = g.step(&StepPlan { prefill: vec![], decode: vec![1, 2, 3], batch: vec![] });
        assert_eq!(og.len(), 3);
        for i in 0..3 {
            assert_eq!(og[i], et[i], "步 {k} 请求 {}：sink 行泄漏进真请求", i + 1);
        }
    }
    assert_eq!(g.graph_replays(), 8, "8 个纯 decode 步全走桶 4 回放");
    eprintln!("qg2 sink 吸收：m=3 → 桶 4，8 步前 3 行与 eager m=4 零漂移");
}

/// qg3 · 步时延金丝雀：单请求 decode 20 步中位步时延。大模型步 GEMM
/// 主导，图收益 = launch 债 + 每步 14 次分配/同步上传的扣除——图档先测
/// （热身漂移偏向后测的 eager），仍须严格更低
#[test]
fn qg3_graph_step_latency_canary() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let prompt = HfTokenizer::open(&dir).unwrap().encode("The capital of France is");
    let med = |ex: &mut Qwen35Executor| -> f64 {
        ex.admitted(1, &prompt);
        let plan = StepPlan { prefill: vec![(1, prompt.len())], decode: vec![1], batch: vec![] };
        assert_eq!(ex.step(&plan).len(), 1);
        let mut us = Vec::new();
        for _ in 0..20 {
            let t0 = Instant::now();
            let p = StepPlan { prefill: vec![], decode: vec![1], batch: vec![] };
            assert_eq!(ex.step(&p).len(), 1);
            us.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        us.sort_by(|a, b| a.total_cmp(b));
        us[us.len() / 2]
    };
    let Some(mut g) = graph_exec(&dir, &[1]) else { return };
    let mg = med(&mut g);
    drop(g); // 先放再开 eager
    let Some(mut e) = exec(&dir) else { return };
    let me = med(&mut e);
    eprintln!(
        "qg3 单请求 decode 步时延：图档 {mg:.0}µs vs eager {me:.0}µs（收益 {:.1}%）",
        (me - mg) / me * 100.0
    );
    assert!(mg < me, "图档步时延 {mg:.0}µs 须低于 eager {me:.0}µs");
}
