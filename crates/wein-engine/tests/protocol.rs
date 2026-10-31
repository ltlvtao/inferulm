//! engine core 协议验收：事件序不变量。
//! 这把尺子同样适用于 MockEngine——任何引擎实现都必须满足：
//! Accepted ≤ 首 Token < Finished，token 序无丢失，终结恰一次且随后关流。

use std::time::{Duration, Instant};

use wein_engine::{CoreConfig, EngineCore, EngineEvent, EventStream, FinishReason, SamplingParams};

async fn next(stream: &mut EventStream) -> EngineEvent {
    tokio::time::timeout(Duration::from_secs(5), stream.recv())
        .await
        .expect("5s 内必须有事件（不能死等）")
        .expect("事件流必须仍然打开")
}

/// e1 · 事件序不变量：Accepted → Token×N（依序）→ Finished(Length) 恰一次 → 关流；
///      关流不等于引擎死亡——后续请求照常服务
#[tokio::test]
async fn e1_event_order_tokens_then_single_finish_then_close() {
    let (engine, stats) =
        EngineCore::spawn(CoreConfig::scripted(vec![5, 6, 7], Duration::from_millis(1)));
    let (_id, mut s) = engine.submit(vec![1, 2, 3], 3, Default::default()).expect("engine alive");

    assert!(matches!(next(&mut s).await, EngineEvent::Accepted));
    assert!(matches!(next(&mut s).await, EngineEvent::Token(5)));
    assert!(matches!(next(&mut s).await, EngineEvent::Token(6)));
    assert!(matches!(next(&mut s).await, EngineEvent::Token(7)));
    assert!(matches!(next(&mut s).await, EngineEvent::Finished(FinishReason::Length)));

    // 终结后事件流关闭（sender 已释放），且不再有任何事件
    assert!(s.recv().await.is_none(), "Finished 之后必须关流");
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(stats.finished_count(), 1);

    // 引擎仍活着：新请求照常
    let (_id2, mut s2) = engine.submit(vec![1], 1, Default::default()).expect("engine alive");
    assert!(matches!(next(&mut s2).await, EngineEvent::Accepted));
    assert!(matches!(next(&mut s2).await, EngineEvent::Token(5)));
    assert!(matches!(next(&mut s2).await, EngineEvent::Finished(FinishReason::Length)));
}

/// e2 · 取消运行中请求：收到 Aborted，且容量真的被释放
///     （max_running=1：A 不释放，B 永远进不了批——用超时把饿死变成显式失败）
#[tokio::test]
async fn e2_cancel_running_yields_aborted_and_frees_capacity() {
    let cfg = CoreConfig {
        sched: wein_sched::SchedConfig { max_running: 1, ..Default::default() },
        pieces: vec![0, 1],
        step_delay: Duration::from_millis(1),
        memory: None,
    };
    let (engine, stats) = EngineCore::spawn(cfg);

    let (a, mut sa) = engine.submit(vec![1], 50, Default::default()).expect("engine alive");
    // 等 A 真正产出第一个 token（已在批中运行）再取消
    assert!(matches!(next(&mut sa).await, EngineEvent::Accepted));
    assert!(matches!(next(&mut sa).await, EngineEvent::Token(0)));
    engine.cancel(a);
    // 取消在步边界生效：在飞的那一步可能还有 ≤1 步的 token 已在途中，
    // 容忍间隔 token，但必须在有界时间内收到 Aborted 且之后无任何事件
    let t0 = Instant::now();
    loop {
        match next(&mut sa).await {
            EngineEvent::Finished(FinishReason::Aborted) => break,
            EngineEvent::Token(_) => {
                assert!(t0.elapsed() < Duration::from_millis(200), "取消后不得继续产出多步");
            }
            ev => panic!("取消后不应出现 {ev:?}"),
        }
    }
    assert!(sa.recv().await.is_none(), "Aborted 之后必须关流");
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(stats.cancelled_count(), 1, "取消必须被引擎记账");

    // B 能完成 ⇒ A 的容量确实释放了
    let (_b, mut sb) = engine.submit(vec![1], 2, Default::default()).expect("engine alive");
    assert!(matches!(next(&mut sb).await, EngineEvent::Accepted));
    assert!(matches!(next(&mut sb).await, EngineEvent::Token(0)));
    assert!(matches!(next(&mut sb).await, EngineEvent::Token(1)));
    assert!(matches!(next(&mut sb).await, EngineEvent::Finished(FinishReason::Length)));
}

/// e3 · max_tokens=0：Accepted 后立即 Finished(Length)，不占批不耗步
#[tokio::test]
async fn e3_max_tokens_zero_finishes_immediately() {
    let (engine, _stats) =
        EngineCore::spawn(CoreConfig::scripted(vec![0], Duration::from_millis(50)));
    let t0 = Instant::now();
    let (_id, mut s) = engine.submit(vec![1, 2], 0, Default::default()).expect("engine alive");
    assert!(matches!(next(&mut s).await, EngineEvent::Accepted));
    assert!(matches!(next(&mut s).await, EngineEvent::Finished(FinishReason::Length)));
    assert!(t0.elapsed() < Duration::from_millis(100), "max=0 不得等待任何一步执行");
}

/// e4 · stop_token_ids（W5）：采样命中停止词表（chat eos）→ 该 token 不外吐、
///      Finished(Stop) 恰一次后关流；记账走 finished 而非 cancelled；
///      空表请求行为与旧全同（Length 路径零漂移）
#[tokio::test]
async fn e4_stop_token_ids_finishes_stop_without_emitting() {
    let (engine, stats) = EngineCore::spawn(CoreConfig::scripted(
        vec![5, 6, 7, 9, 8],
        Duration::from_millis(1),
    ));
    let mut sp = SamplingParams::default();
    sp.stop_token_ids = vec![9, 8];
    let (_id, mut s) = engine.submit(vec![1], 10, sp).expect("engine alive");

    assert!(matches!(next(&mut s).await, EngineEvent::Accepted));
    assert!(matches!(next(&mut s).await, EngineEvent::Token(5)));
    assert!(matches!(next(&mut s).await, EngineEvent::Token(6)));
    assert!(matches!(next(&mut s).await, EngineEvent::Token(7)));
    // 9 命中停止表：不外吐，直接终结
    assert!(matches!(next(&mut s).await, EngineEvent::Finished(FinishReason::Stop)));
    assert!(s.recv().await.is_none(), "Finished 之后必须关流");
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(stats.finished_count(), 1, "eos 停止是 finished 不是 cancelled");
    assert_eq!(stats.cancelled_count(), 0);

    // 空 stop 表零漂变：Length 路径原样
    let (_, mut s2) = engine.submit(vec![1], 2, Default::default()).expect("engine alive");
    assert!(matches!(next(&mut s2).await, EngineEvent::Accepted));
    assert!(matches!(next(&mut s2).await, EngineEvent::Token(5)));
    assert!(matches!(next(&mut s2).await, EngineEvent::Token(6)));
    assert!(matches!(next(&mut s2).await, EngineEvent::Finished(FinishReason::Length)));
}
