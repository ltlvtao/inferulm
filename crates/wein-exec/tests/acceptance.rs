//! wein-exec 验收（x 系列）——exec 层第一个真身。
//!
//! ToyExecutor：DeviceSession（探测→画像→OpTable resolve→流）+ 单块 toy
//! 模型（embed/rms/rope/KV 分页/paged attn/silu×up/残差/lm_head——M0 算子
//! 清单全走真路径，GEMM 走 cuBLASLt）+ BlockPool 页记账 + host 贪心。
//!
//! 全部 GPU 门控（无卡环境 SKIP——与 o/g 系列同式）。
//!
//! - x1 · 会话成立：探测→resolve→权重上传→页池零占用
//! - x2 · 前向数值诚实：设备 logits vs 独立 CPU 镜像（容差——堆叠 GEMM 的
//!        累加序在厂商库内，位对齐不可达；golden 两模式的 exec 版）
//! - x3 · 生成确定性：两个全新执行器同 prompt 同输出（逐 token）
//! - x4 · 位置敏感：rope 位置必须影响 logits（前向不是查表）
//! - x5 · KV 跨页：40 token prompt（块 16）→ 3 页，块表记账与 BlockPool 一致
//! - x6 · 引擎集成：spawn_with 组装，事件流/Length/取消与直跑零漂移
//! - x7 · 步时延金丝雀：批 4 纯 decode 中位步时延（L3 对照 launch 4.69µs，
//!        BASELINE §7）
//! - x8 · 块表行回收：600 次 admit 走过 512 行上限（battery 抓到的长寿性缺陷回归钉）
use std::time::{Duration, Instant};

use wein_engine::{CoreConfig, EngineCore, EngineEvent, Executor, FinishReason, SamplingParams};
use wein_exec::{argmax, reference, ToyConfig, ToyExecutor};
use wein_sched::StepPlan;

fn exec() -> Option<ToyExecutor> {
    match ToyExecutor::open(ToyConfig::default()) {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("SKIP（exec 会话不可用：{e}）");
            None
        }
    }
}

/// x1 · 会话成立：整条装配链（探测→画像→十槽 resolve→权重上传→页池）
/// 一次跑通，开局状态干净
#[test]
fn x1_session_open() {
    let Some(ex) = exec() else { return };
    assert_eq!(ex.pages_used(), 0, "开局零页占用");
    eprintln!(
        "toy exec 就绪：{}，页 {} × 块 {} × 头积 {}，词表 {}",
        ex.device_name(),
        ToyConfig::default().num_pages,
        ToyConfig::default().block_size,
        ToyConfig::default().kv_heads * ToyConfig::default().head_dim,
        ToyConfig::default().vocab,
    );
}

/// x2 · 前向数值诚实：同一 prompt 逐步前向，末位 logits 对独立 CPU 镜像
/// （reference::toy_logits——bf16 存储点逐位对齐设备、GEMM f64 累加）。
/// 容差 = 3% 相对 + 0.05 绝对底（堆叠 6 个 GEMM 的 TC f32 累加 vs f64）
#[test]
fn x2_forward_matches_cpu_mirror() {
    let Some(mut ex) = exec() else { return };
    let cfg = ToyConfig::default();
    let prompt: Vec<u32> = vec![3, 17, 5, 9, 2];
    ex.admitted(1, &prompt);
    let mut logits = Vec::new();
    for pos in 0..prompt.len() as u32 {
        logits = ex.forward_one(1, prompt[pos as usize], pos);
    }
    let expect = reference::toy_logits(&prompt, &cfg);
    assert_eq!(logits.len(), expect.len(), "logits 维度 = 词表");
    let mut worst = 0f64;
    for i in 0..expect.len() {
        let (g, r) = (logits[i] as f64, expect[i] as f64);
        let tol = (0.03 * r.abs()).max(0.05);
        let err = (g - r).abs();
        assert!(err <= tol, "logits[{i}] 偏差 {err} 超容差 {tol}（got {g} vs 镜像 {r}）");
        worst = worst.max(err / tol);
    }
    eprintln!("x2 最差元素容差占用 {:.0}%", worst * 100.0);
    // 贪心决策一致（镜像不只数值近，决策同）
    assert_eq!(argmax(&logits), argmax(&expect), "设备/镜像贪心决策须一致");
}

/// x3 · 生成确定性：两个全新执行器（各自 resolve/heuristic）同 prompt
/// 逐 token 相同——不可复现的 algo 选择会在这里现形（g4 的 exec 版）
#[test]
fn x3_generate_deterministic() {
    let Some(mut a) = exec() else { return };
    let Some(mut b) = exec() else { return };
    let (ta, tb) = (a.generate(&[9, 4, 4], 16), b.generate(&[9, 4, 4], 16));
    assert_eq!(ta, tb, "两个全新执行器输出必须逐 token 相同");
    assert_eq!(ta.len(), 16);
    eprintln!("x3 生成序列：{ta:?}");
}

/// x4 · 位置敏感：同 token 在 pos=5（有 5 token 上下文）与 pos=0（孤立）
/// 的 logits 必须不同——rope/分页 KV/因果 attention 任一接错都会撞这条
#[test]
fn x4_position_matters() {
    let Some(mut ex) = exec() else { return };
    let ctx: Vec<u32> = vec![1, 2, 3, 4, 5];
    ex.admitted(1, &ctx);
    for pos in 0..ctx.len() as u32 {
        ex.forward_one(1, ctx[pos as usize], pos);
    }
    let with_ctx = ex.forward_one(1, 7, 5);
    // 同一执行器另一请求（新行新页），同 token 落 pos=0：因果窗口只有自己
    ex.admitted(2, &[]);
    let isolated = ex.forward_one(2, 7, 0);
    assert_ne!(with_ctx, isolated, "rope 位置必须影响 logits");
}

/// x5 · KV 跨页：prompt 40（块 16）+ 生成 3 → 写到位置 41 → 42 槽 → 3 页；
/// 块表记账与 BlockPool 占用一致（跨页寻址在 o11/o12 已位对齐，这里钉
/// exec 侧页管理正确接上了）
#[test]
fn x5_kv_pages_across_blocks() {
    let Some(mut ex) = exec() else { return };
    let cfg = ToyConfig::default();
    let prompt: Vec<u32> = (0..40).collect();
    let toks = ex.generate(&prompt, 3);
    assert_eq!(toks.len(), 3);
    assert_eq!(ex.pages_of(1), 3, "42 槽 / 块 16 = 3 页");
    assert_eq!(ex.pages_used(), 3, "BlockPool 占用与块表一致");
    assert!(toks.iter().all(|&t| t < cfg.vocab), "token 在词表内");
}

/// x6 · 引擎集成：EngineCore::spawn_with 组装真执行器（依赖规则：组装在
/// 调用方，engine 不依赖 exec）。事件流语义与直跑路径零漂移；取消 → Aborted
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn x6_engine_integration() {
    let Some(ex) = exec() else { return };
    let (handle, stats) =
        EngineCore::spawn_with(CoreConfig::scripted(vec![], Duration::ZERO), Box::new(ex));
    let prompt = vec![3u32, 17, 5, 9, 2];
    let (_id, mut rx) = handle.submit(prompt.clone(), 8, SamplingParams::default()).expect("engine alive");
    let mut toks = Vec::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            EngineEvent::Token(t) => toks.push(t),
            EngineEvent::Finished(FinishReason::Length) => break,
            EngineEvent::Finished(r) => panic!("8 token 内不该以 {r:?} 终结"),
            EngineEvent::Accepted => {}
        }
    }
    assert_eq!(toks.len(), 8, "max_tokens=8 → 8 个 Token 事件");
    // 直跑对照：调度器驱动的批组成与直跑一致（追平段同形状）→ 同输出
    let Some(mut direct) = exec() else { return };
    assert_eq!(toks, direct.generate(&prompt, 8), "引擎路径与直跑路径输出须一致");
    assert_eq!(stats.finished_count(), 1);

    // 取消：确认在跑（首个 token）后 cancel → Aborted；在飞一步的 token 允许到达。
    // multi_thread flavor 必须：step_delay=0 时 sleep(ZERO) 立即就绪不 yield，
    // 单线程 runtime 里引擎任务会独占线程自由跑——toy 的 max_pos=256 上限
    // 会被冲破（真发现：exec 的位置上限是 caller 合同，max_tokens 须 ≤ 上限）
    let (id2, mut rx2) = handle.submit(vec![1, 2, 3], 1_000, SamplingParams::default()).expect("engine alive");
    loop {
        match rx2.recv().await {
            Some(EngineEvent::Token(_)) => break,
            Some(EngineEvent::Finished(_)) => panic!("1000 上限不该先终结"),
            Some(EngineEvent::Accepted) => {}
            None => panic!("事件流提前关闭"),
        }
    }
    handle.cancel(id2);
    loop {
        match rx2.recv().await {
            Some(EngineEvent::Finished(FinishReason::Aborted)) => break,
            Some(EngineEvent::Token(_)) => {}
            other => panic!("取消后意外事件：{other:?}"),
        }
    }
}

/// x7 · 步时延金丝雀：批 4 纯 decode 50 步，中位步时延（对照 BASELINE §7
/// launch 4.69µs——一步 = 上传 + 16 launch + GEMM×7 + 回读，量级差距即
/// exec 层优化的账）。50ms 是 Sanity 上限不是性能承诺
#[test]
fn x7_decode_step_latency_canary() {
    let Some(mut ex) = exec() else { return };
    let ids: Vec<u64> = (1..=4).collect();
    let prompt = vec![5u32, 6, 7, 8];
    for &id in &ids {
        ex.admitted(id, &prompt);
    }
    // 追平步：prefill 4 + decode 同列（catch-up 采样）
    let plan = StepPlan {
        prefill: ids.iter().map(|&i| (i, prompt.len())).collect(),
        decode: ids.clone(),
        batch: vec![],
    };
    assert_eq!(ex.step(&plan).len(), 4);
    let mut us = Vec::new();
    for _ in 0..50 {
        let t0 = Instant::now();
        let plan = StepPlan { prefill: vec![], decode: ids.clone(), batch: vec![] };
        let out = ex.step(&plan);
        us.push(t0.elapsed().as_secs_f64() * 1e6);
        assert_eq!(out.len(), 4, "纯 decode 每请求每步一个 token");
    }
    us.sort_by(|a, b| a.total_cmp(b));
    let med = us[us.len() / 2];
    eprintln!(
        "x7 批 4 decode 步时延：中位 {med:.0}µs（p10 {:.0} / p90 {:.0}）",
        us[5],
        us[us.len() - 5]
    );
    assert!(med < 50_000.0, "步时延 {med:.0}µs 超 Sanity 上限 50ms");
}

/// x8 · 块表行回收：600 次串行 admit/step/retain 走过 max_rows=512 上限
/// ——battery run2/3 全空抓到的长寿性缺陷（行只增不收，第 513 次 admit
/// 炸 engine 任务，后续所有流瞬关）的回归钉
#[test]
fn x8_row_recycling_longevity() {
    let Some(mut ex) = exec() else { return };
    let prompt = [7u32, 11];
    for k in 1u64..=600 {
        ex.admitted(k, &prompt);
        let plan = StepPlan {
            prefill: vec![(k, prompt.len())],
            decode: vec![k],
            batch: vec![],
        };
        let out = ex.step(&plan);
        assert_eq!(out.len(), 1, "第 {k} 次 admit 后仍能采样");
        // 只保留当前请求——前一个在此回收（行 + 页）
        ex.retain(&[k]);
    }
    assert_eq!(ex.pages_used(), 1, "串行模式下任意时刻只占一个请求的页");
}
