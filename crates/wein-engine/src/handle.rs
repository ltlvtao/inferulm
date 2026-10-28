use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

use crate::protocol::{EngineCommand, EngineRequest, EngineEvent, RequestId, SamplingParams};

pub type EventStream = UnboundedReceiver<EngineEvent>;

/// serve 侧持有的引擎句柄：提交与取消是仅有的两个入口。
/// 真实 engine core 上线后此接口不变（协议先于实现——TDD 外→内的锚点）。
#[derive(Clone)]
pub struct EngineHandle {
    cmd_tx: UnboundedSender<EngineCommand>,
    next_id: Arc<AtomicU64>,
}

impl EngineHandle {
    pub fn new(cmd_tx: UnboundedSender<EngineCommand>) -> Self {
        Self { cmd_tx, next_id: Arc::new(AtomicU64::new(1)) }
    }

    /// 提交生成请求，返回 (请求 id, 事件流)。
    pub fn submit(
        &self,
        prompt_token_ids: Vec<u32>,
        max_tokens: u32,
        sampling: SamplingParams,
    ) -> (RequestId, EventStream) {
        let (tx, rx) = mpsc::unbounded_channel();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let _ = self.cmd_tx.send(EngineCommand::Submit(EngineRequest {
            id,
            prompt_token_ids,
            max_tokens,
            sampling,
            respond_to: tx,
        }));
        (id, rx)
    }

    pub fn cancel(&self, id: RequestId) {
        let _ = self.cmd_tx.send(EngineCommand::Cancel(id));
    }
}
