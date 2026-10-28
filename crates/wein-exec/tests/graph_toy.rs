//! W2-B 图档位验收（w 系列）——ToyExecutor FULL_DECODE（02-architecture §7）。
//!
//! 机制（gr 系已钉）：桶集合共用**单输入存储**（常驻 DevBuf，设备指针烤进
//! 图节点，数据每步 upload_on 整段覆写）；纯 decode 步路由到 ≥m 的最小桶
//! 图回放；批尾不足一桶的行由 **sink 行**（块表末行 + 专属页）吸收——
//! 吸收行把垃圾写进自己的页，不触真请求任何状态。混合步（含 prefill）
//! 走 eager 直发。
//!
//! 数值合同（m 依赖坑，q8 教训）：cuBLASLt 按 m 选 algo，跨形状对拍无效；
//! 唯一干净不变量 = **图回放 ≡ 同形状直发（位一致）**。
//!
//! - w1 · 图回放 ≡ 同形状直发位一致：单请求逐 token（m 恒 1 → 桶 1 精确
//!        命中），图档与 eager 两执行器 logits 逐位相等
//! - w2 · sink 吸收：3 请求走桶 4（1 行 padding）vs eager 4 请求（第 4
//!        个 = 第 3 个的克隆，补齐同形状 m=4）——前 3 行 token 逐步相等
//! - w3 · 步时延金丝雀：批 4 纯 decode 50 步，图档中位步时延 < eager
//!
//! 引擎集成（w4）在独立二进制 graph_engine.rs：spawn_with 的引擎步在
//! worker 线程上做隐式同步，会击穿并行线程的捕获态（gr 系真发现）——
//! 捕获窗口与跨线程同步必须进程隔离，不能与本文件同进程并行。
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use wein_engine::Executor;
use wein_exec::{argmax, ToyConfig, ToyExecutor};
use wein_sched::StepPlan;

/// 捕获串行锁（gr 系同款）：THREAD_LOCAL 捕获下，并行线程的隐式同步
/// （legacy stream）会击穿捕获态——本文件三个测试全串行
fn guard() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

fn open(cfg: ToyConfig) -> Option<ToyExecutor> {
    match ToyExecutor::open(cfg) {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("SKIP（exec 会话不可用：{e}）");
            None
        }
    }
}

fn exec() -> Option<ToyExecutor> {
    open(ToyConfig::default())
}

/// 图档执行器：FULL_DECODE 桶集合（空桶集 = NONE，x 系保持 eager）
fn graph_exec(buckets: &[u32]) -> Option<ToyExecutor> {
    open(ToyConfig { graph_buckets: buckets.to_vec(), ..Default::default() })
}

/// 位一致断言（图合同是位级，不是容差级）
fn assert_bits_eq(a: &[f32], b: &[f32], ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}：logits 维度不一致");
    for i in 0..a.len() {
        assert_eq!(
            a[i].to_bits(),
            b[i].to_bits(),
            "{ctx}：logits[{i}] 位不一致（{} vs {}）",
            a[i],
            b[i]
        );
    }
}

/// w1 · 图回放 ≡ 同形状直发位一致：单请求 m 恒 1（桶 1 精确命中，无
/// padding），图档每步回放 vs eager 同形状直发——logits 逐位相等。
/// 位级合同是最强的"图是录制不是重写"证明（gr1 的 exec 版）
#[test]
fn w1_replay_bitwise_matches_eager() {
    let _g = guard();
    let Some(mut a) = graph_exec(&[1]) else { return };
    let Some(mut b) = exec() else { return };
    let prompt: Vec<u32> = vec![11, 7, 3, 19, 4];
    a.admitted(1, &prompt);
    b.admitted(1, &prompt);

    // 追平：逐 token 单行前向（图档侧每步 = 桶 1 回放）
    let (mut la, mut lb) = (Vec::new(), Vec::new());
    let mut pos = 0u32;
    for &tok in &prompt {
        la = a.forward_one(1, tok, pos);
        lb = b.forward_one(1, tok, pos);
        pos += 1;
    }
    assert_bits_eq(&la, &lb, "追平位");
    assert!(
        a.graph_replays() >= prompt.len() as u64,
        "单行步必须走桶 1 回放（实测 {} 次）",
        a.graph_replays()
    );

    // decode：贪心喂回，流级位一致
    for k in 0..8 {
        let (ta, tb) = (argmax(&la), argmax(&lb));
        assert_eq!(ta, tb, "decode 步 {k}：贪心决策漂移");
        la = a.forward_one(1, ta, pos);
        lb = b.forward_one(1, tb, pos);
        pos += 1;
        assert_bits_eq(&la, &lb, &format!("decode 步 {k}"));
    }
    let n = a.graph_replays();
    eprintln!("w1 位一致：追平 5 + decode 8 步，图回放 {n} 次全部位对齐");
}

/// w2 · sink 吸收：图档 3 请求纯 decode（m=3 → 桶 4，1 行 padding）vs
/// eager 4 请求——第 4 个是第 3 个的克隆（同 prompt 同 pos），把 eager
/// 补到同形状 m=4。sink 行只写自己的页：前 3 行 token 逐步相等
#[test]
fn w2_sink_row_absorbs_padding() {
    let _g = guard();
    let Some(mut g) = graph_exec(&[4]) else { return };
    let Some(mut e) = exec() else { return };
    let ps: [Vec<u32>; 3] = [vec![3, 8, 5], vec![1, 2, 3, 4], vec![9, 6]];
    for (k, p) in ps.iter().enumerate() {
        let id = k as u64 + 1;
        g.admitted(id, p);
        e.admitted(id, p);
    }
    e.admitted(4, &ps[2]); // eager 第 4 行 = 克隆请求（同形状位，自己的页）

    // 追平步：两侧都 eager（图档 prefill 非空 → 不入图）
    let pg = StepPlan {
        prefill: (1..=3u64).map(|i| (i, ps[i as usize - 1].len())).collect(),
        decode: vec![1, 2, 3],
        batch: vec![],
    };
    let pe = StepPlan {
        prefill: (1..=4u64)
            .map(|i| (i, if i == 4 { ps[2].len() } else { ps[i as usize - 1].len() }))
            .collect(),
        decode: vec![1, 2, 3, 4],
        batch: vec![],
    };
    assert_eq!(g.step(&pg).len(), 3);
    assert_eq!(e.step(&pe).len(), 4);
    assert_eq!(g.graph_replays(), 0, "追平步是 prefill——必须 eager");

    let before = g.graph_replays();
    for k in 0..16u32 {
        let og = g.step(&StepPlan { prefill: vec![], decode: vec![1, 2, 3], batch: vec![] });
        let oe = e.step(&StepPlan { prefill: vec![], decode: vec![1, 2, 3, 4], batch: vec![] });
        assert_eq!(og.len(), 3);
        assert_eq!(oe.len(), 4);
        for i in 0..3 {
            assert_eq!(og[i], oe[i], "步 {k} 请求 {}：sink 行泄漏进真请求", i + 1);
        }
    }
    let replays = g.graph_replays() - before;
    assert_eq!(replays, 16, "16 个纯 decode 步全部走桶 4 回放（1 行 padding）");
    eprintln!("w2 sink 吸收：m=3 → 桶 4，{replays} 步前 3 行与 eager m=4 零漂移");
}

/// w3 · 步时延金丝雀：批 4 纯 decode 50 步中位步时延，图档 < eager
/// （eager 步 = 5 次同步上传 + 16 kernel launch + 回读；图档步 = 5 次
/// 异步覆写 + 1 次图 launch——launch 债是 toy 小 kernel 的主导项）
#[test]
fn w3_graph_step_latency_canary() {
    let _g = guard();
    let Some(mut g) = graph_exec(&[1, 2, 4, 8]) else { return };
    let Some(mut e) = exec() else { return };
    let med = |ex: &mut ToyExecutor| -> f64 {
        let ids: Vec<u64> = (1..=4).collect();
        let prompt = vec![5u32, 6, 7, 8];
        for &i in &ids {
            ex.admitted(i, &prompt);
        }
        let plan = StepPlan {
            prefill: ids.iter().map(|&i| (i, prompt.len())).collect(),
            decode: ids.clone(),
            batch: vec![],
        };
        assert_eq!(ex.step(&plan).len(), 4);
        let mut us = Vec::new();
        for _ in 0..50 {
            let t0 = Instant::now();
            let p = StepPlan { prefill: vec![], decode: ids.clone(), batch: vec![] };
            assert_eq!(ex.step(&p).len(), 4);
            us.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        us.sort_by(|a, b| a.total_cmp(b));
        us[us.len() / 2]
    };
    let me = med(&mut e);
    let mg = med(&mut g);
    eprintln!("w3 批 4 decode 步时延：eager 中位 {me:.0}µs vs 图档 {mg:.0}µs（{:.2}×）", me / mg);
    assert!(mg < me, "图档步时延 {mg:.0}µs 须低于 eager {me:.0}µs");
}
