use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::mpsc::UnboundedReceiver;
use tokio::task::JoinHandle;
use tokio::time::sleep;

use crate::handle::EngineHandle;
use crate::protocol::{EngineCommand, EngineEvent, FinishReason};

/// 可编程的引擎替身：验收测试与 bench 都驱动它。
/// 故意做小——它只需要"按脚本吐 token + 可被取消 + 状态可断言"。
#[derive(Clone, Debug)]
pub struct MockScript {
    /// 模拟 prefill：首个 token 前的延迟（TTFT 下界）
    pub prefill_delay: Duration,
    /// 每个 token 的间隔（TPOT 下界）
    pub per_token_delay: Duration,
    /// 依序产出的 token id（循环使用）
    pub pieces: Vec<u32>,
    /// 在产出第 N 个 token 之前"自然停止"（FinishReason::Stop）；None = 跑满 max_tokens
    pub stop_at: Option<u32>,
}

impl Default for MockScript {
    fn default() -> Self {
        Self {
            prefill_delay: Duration::from_millis(2),
            per_token_delay: Duration::from_millis(1),
            pieces: vec![0, 1, 2, 3, 4, 5, 6, 7],
            stop_at: None,
        }
    }
}

/// 供断言用的引擎侧状态（取消集合 + 接受计数）。
#[derive(Default)]
pub struct MockState {
    cancelled: Mutex<HashSet<u64>>,
    accepted: AtomicUsize,
}

impl MockState {
    pub fn is_cancelled(&self, id: u64) -> bool {
        self.cancelled.lock().unwrap().contains(&id)
    }

    pub fn cancelled_count(&self) -> usize {
        self.cancelled.lock().unwrap().len()
    }

    pub fn accepted_count(&self) -> usize {
        self.accepted.load(Ordering::Relaxed)
    }
}

pub struct MockEngine;

impl MockEngine {
    /// 启动 mock 引擎任务，返回 (serve 用的句柄, 断言用的状态)。
    pub fn spawn(script: MockScript) -> (EngineHandle, Arc<MockState>) {
        let (cmd_tx, cmd_rx) = tokio::sync::mpsc::unbounded_channel();
        let state = Arc::new(MockState::default());
        tokio::spawn(Self::run(script, cmd_rx, state.clone()));
        (EngineHandle::new(cmd_tx), state)
    }

    async fn run(script: MockScript, mut cmd_rx: UnboundedReceiver<EngineCommand>, state: Arc<MockState>) {
        let mut inflight: HashMap<u64, JoinHandle<()>> = HashMap::new();
        while let Some(cmd) = cmd_rx.recv().await {
            match cmd {
                EngineCommand::Submit(req) => {
                    state.accepted.fetch_add(1, Ordering::Relaxed);
                    let _ = req.respond_to.send(EngineEvent::Accepted);
                    let stop_at = script.stop_at;
                    let (pre_delay, tok_delay) = (script.prefill_delay, script.per_token_delay);
                    let pieces = script.pieces.clone();
                    let h = tokio::spawn(async move {
                        sleep(pre_delay).await;
                        for k in 0..req.max_tokens {
                            if stop_at == Some(k) {
                                let _ = req.respond_to.send(EngineEvent::Finished(FinishReason::Stop));
                                return;
                            }
                            sleep(tok_delay).await;
                            let id = pieces[(k as usize) % pieces.len().max(1)];
                            if req.respond_to.send(EngineEvent::Token(id)).is_err() {
                                return; // 接收端已放弃
                            }
                        }
                        let _ = req.respond_to.send(EngineEvent::Finished(FinishReason::Length));
                    });
                    inflight.insert(req.id, h);
                }
                EngineCommand::Cancel(id) => {
                    if let Some(h) = inflight.remove(&id) {
                        h.abort();
                    }
                    state.cancelled.lock().unwrap().insert(id);
                }
            }
        }
        // 引擎关闭：终止全部在飞请求
        for (_, h) in inflight {
            h.abort();
        }
    }
}
