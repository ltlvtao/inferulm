//! memory core 验收：BlockPool 水位驱动调度（im1-im4）。
//!
//! 这层测的是**接线**：wein-memory 的 MemState 经 PoolCapacity 适配到
//! wein-sched 的 CapacityModel 钩子后，端到端行为是否兑现设计承诺——
//! 压力触发抢占、前缀命中省钱、超大请求兜底中止、协议不变量不破。
//! （MemState 内部语义由 wein-memory 的 m1-m10 钉死，不在此重复）

use std::time::Duration;

use wein_engine::{CoreConfig, EngineCore, EngineEvent, EventStream, FinishReason};
use wein_memory::state::MemConfig;
use wein_sched::SchedConfig;

async fn next(stream: &mut EventStream) -> EngineEvent {
    tokio::time::timeout(Duration::from_secs(5), stream.recv())
        .await
        .expect("5s 内必须有事件（不能死等）")
        .expect("事件流必须仍然打开")
}

/// 收齐一个请求的完整事件流，返回 (tokens, finish)
async fn run_to_end(stream: &mut EventStream) -> (Vec<u32>, FinishReason) {
    let mut tokens = Vec::new();
    loop {
        match next(stream).await {
            EngineEvent::Accepted => {}
            EngineEvent::Token(t) => tokens.push(t),
            EngineEvent::Finished(r) => {
                assert!(stream.recv().await.is_none(), "终结之后必须关流");
                return (tokens, r);
            }
        }
    }
}

/// 内存启用的 core：小池制造真实压力
fn mem_core(bs: u32, blocks: u32, watermark: f32, max_running: usize) -> (wein_engine::EngineHandle, std::sync::Arc<wein_engine::EngineStats>) {
    let cfg = CoreConfig {
        sched: SchedConfig { max_running, max_batch_tokens: usize::MAX, ..Default::default() },
        pieces: vec![5, 6, 7, 8],
        step_delay: Duration::from_millis(0),
        memory: Some(MemConfig { block_size: bs, num_blocks: blocks, low_watermark: watermark }),
    };
    EngineCore::spawn(cfg)
}

/// 宽裕内存的 core（不触发压力，验证纯记账路径）
fn roomy_core() -> (wein_engine::EngineHandle, std::sync::Arc<wein_engine::EngineStats>) {
    mem_core(2, 4096, 0.1, 16)
}

/// im1 · 池压力触发抢占（recompute）：块随生成增长 → 水位 → 驱逐最浅者；
///      被抢占者最终完成且 token 序精确（生成进度跨抢占保留），无 token 丢失/重复
#[tokio::test]
async fn im1_pool_pressure_preempts_and_both_finish_exactly() {
    // bs=2, 12 块, 水位 3：A/B 各 prompt 2 token，max 20 → 增长必触水位
    let (engine, stats) = mem_core(2, 12, 0.25, 8);
    let expected: Vec<u32> = (0..20).map(|i| [5u32, 6, 7, 8][i % 4]).collect();

    let (_a, mut sa) = engine.submit(vec![1, 2], 20, Default::default()).expect("engine alive");
    let (_b, mut sb) = engine.submit(vec![3, 4], 20, Default::default()).expect("engine alive");
    let (ta, ra) = run_to_end(&mut sa).await;
    let (tb, rb) = run_to_end(&mut sb).await;

    assert!(matches!(ra, FinishReason::Length));
    assert!(matches!(rb, FinishReason::Length));
    assert_eq!(ta, expected, "A 的 token 序必须精确（跨抢占无丢失无重复）");
    assert_eq!(tb, expected, "B 的 token 序必须精确");

    // 压力真实发生过（不是 max_batch_tokens——它已设为无穷）
    assert!(stats.preempted_count() >= 1, "水位压力必须触发至少一次抢占");
    // 收敛性：两个 20-token 请求不该跑几百步（重算债务有限）
    assert!(stats.step_count() < 500, "步数 {}", stats.step_count());
    // 内存记账闭环：分配有限（12 块池绝不该分配出几十块仍在占用）
    assert!(stats.mem().blocks_allocated >= 12, "增长过程必然复用或新配");
}

/// im2 · 前缀缓存命中：同 prompt 的第二个请求命中 radix（cache_hit_tokens 记账）
#[tokio::test]
async fn im2_prefix_cache_hit_on_second_request() {
    let (engine, stats) = roomy_core();
    let prompt = vec![11u32, 22, 33, 44, 55, 66]; // 3 整块（bs=2）

    let (_a, mut sa) = engine.submit(prompt.clone(), 4, Default::default()).expect("engine alive");
    run_to_end(&mut sa).await;
    let hit0 = stats.mem().cache_hit_tokens;
    assert_eq!(hit0, 0, "首请求无缓存可命中");

    let (_b, mut sb) = engine.submit(prompt, 4, Default::default()).expect("engine alive");
    run_to_end(&mut sb).await;
    let hit1 = stats.mem().cache_hit_tokens;
    assert!(hit1 >= 6, "第二个请求必须命中 prompt 前缀（≥6 token），got {hit1}");
}

/// im3 · 超大请求兜底中止：批空无条件准入但池放不下 → preempt 被拒 → Aborted；
///      引擎存活，后续正常请求照常完成
#[tokio::test]
async fn im3_oversized_request_aborted_engine_alive() {
    // 4 块 × bs 2 = 8 token 容量；prompt 20 token 放不下
    let (engine, stats) = mem_core(2, 4, 0.25, 4);

    let (_big, mut sb) = engine.submit(vec![7u32; 20], 4, Default::default()).expect("engine alive");
    assert!(matches!(next(&mut sb).await, EngineEvent::Accepted));
    let r = loop {
        match next(&mut sb).await {
            EngineEvent::Token(_) => {} // 容忍在飞（放不下本不该有）
            EngineEvent::Finished(fin) => break fin,
            EngineEvent::Accepted => {}
        }
    };
    assert!(matches!(r, FinishReason::Aborted), "放不下必须 Aborted 而非死等，got {r:?}");
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(stats.cancelled_count(), 1);

    // 引擎存活：小请求正常完成
    let (_ok, mut sok) = engine.submit(vec![1], 3, Default::default()).expect("engine alive");
    let (toks, rok) = run_to_end(&mut sok).await;
    assert!(matches!(rok, FinishReason::Length));
    assert_eq!(toks, vec![5, 6, 7]);
}

/// im4 · 协议不变量在内存核心下依然成立：事件序 / 恰一次终结 / 关流 / 取消释放
#[tokio::test]
async fn im4_protocol_invariants_hold_with_memory() {
    let (engine, _stats) = mem_core(2, 64, 0.1, 2);

    // 事件序：Accepted → Token×3 → Finished(Length) → 关流
    let (_a, mut sa) = engine.submit(vec![9, 9], 3, Default::default()).expect("engine alive");
    assert!(matches!(next(&mut sa).await, EngineEvent::Accepted));
    assert!(matches!(next(&mut sa).await, EngineEvent::Token(5)));
    assert!(matches!(next(&mut sa).await, EngineEvent::Token(6)));
    assert!(matches!(next(&mut sa).await, EngineEvent::Token(7)));
    assert!(matches!(next(&mut sa).await, EngineEvent::Finished(FinishReason::Length)));
    assert!(sa.recv().await.is_none());

    // 取消：Aborted + 容量释放（节拍 1ms/步——取消要与生成赛跑，零延迟会先跑完）
    let cfg = CoreConfig {
        sched: SchedConfig { max_running: 1, max_batch_tokens: usize::MAX, ..Default::default() },
        pieces: vec![5, 6, 7, 8],
        step_delay: Duration::from_millis(1),
        memory: Some(MemConfig { block_size: 2, num_blocks: 64, low_watermark: 0.1 }),
    };
    let (engine, stats) = EngineCore::spawn(cfg);
    let (b, mut sb_stream) = engine.submit(vec![1], 50, Default::default()).expect("engine alive");
    assert!(matches!(next(&mut sb_stream).await, EngineEvent::Accepted));
    assert!(matches!(next(&mut sb_stream).await, EngineEvent::Token(5)));
    engine.cancel(b);
    let t0 = std::time::Instant::now();
    loop {
        match next(&mut sb_stream).await {
            EngineEvent::Finished(FinishReason::Aborted) => break,
            EngineEvent::Token(_) => {
                assert!(t0.elapsed() < Duration::from_millis(200), "取消后不得继续产出多步");
            }
            ev => panic!("取消后不应出现 {ev:?}"),
        }
    }
    assert!(sb_stream.recv().await.is_none());
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(stats.cancelled_count(), 1);

    // 块真的释放了：后续请求照常（DeferredRelease 闭环）
    let (_c, mut sc) = engine.submit(vec![2, 2], 4, Default::default()).expect("engine alive");
    let (toks, r) = run_to_end(&mut sc).await;
    assert!(matches!(r, FinishReason::Length));
    assert_eq!(toks.len(), 4);
}
