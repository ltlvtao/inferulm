//! w4 · 引擎集成（独立二进制，见 graph_toy.rs 模块注释）：spawn_with 组装
//! FULL_DECODE 图档执行器，事件流与 eager 引擎/直跑三方零漂移。单请求
//! m 恒 1 → 桶 1 精确命中（无 padding），位级合同在引擎路径上成立
use std::time::Duration;

use wein_engine::{CoreConfig, EngineCore, EngineEvent, FinishReason, SamplingParams};
use wein_exec::{ToyConfig, ToyExecutor};

fn exec(cfg: ToyConfig) -> Option<ToyExecutor> {
    match ToyExecutor::open(cfg) {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("SKIP（exec 会话不可用：{e}）");
            None
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn w4_engine_integration() {
    let Some(ex) = exec(ToyConfig {
        graph_buckets: vec![1, 2, 4, 8],
        ..Default::default()
    }) else {
        return;
    };
    // 引擎步在 worker 线程做隐式同步——open 期捕获已在测试线程完成，
    // 此后本进程不再有捕获窗口（与 graph_toy.rs 并行测试进程隔离）
    let (handle, stats) =
        EngineCore::spawn_with(CoreConfig::scripted(vec![], Duration::ZERO), Box::new(ex));
    let prompt = vec![3u32, 17, 5, 9, 2];
    let (_id, mut rx) = handle.submit(prompt.clone(), 10, SamplingParams::default());
    let mut toks = Vec::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            EngineEvent::Token(t) => toks.push(t),
            EngineEvent::Finished(FinishReason::Length) => break,
            EngineEvent::Finished(r) => panic!("10 token 内不该以 {r:?} 终结"),
            EngineEvent::Accepted => {}
        }
    }
    assert_eq!(toks.len(), 10, "max_tokens=10 → 10 个 Token 事件");

    // eager 直跑对照：追平段同形状 → decode 段位级等价（图回放 ≡ 直发）
    let Some(mut direct) = exec(ToyConfig::default()) else { return };
    assert_eq!(toks, direct.generate(&prompt, 10), "图档引擎路径与 eager 直跑零漂移");
    assert_eq!(stats.finished_count(), 1);
    eprintln!("w4 引擎集成：图档事件流 10 token，与 eager 直跑零漂移");
}
