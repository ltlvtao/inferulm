use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

use crate::core::EngineStats;
use crate::protocol::{EngineCommand, EngineRequest, EngineEvent, RequestId, SamplingParams};

pub type EventStream = UnboundedReceiver<EngineEvent>;

/// 提交失败：引擎任务已退出（命令通道关闭）。
///
/// 这是"服务假死"的唯一合法信号——引擎任务 panic/退出后 `cmd_rx` 随之 drop，
/// 此后再提交必然失败。调用方（serve）必须把它映射成 5xx，**不许**静默吞掉
/// 当 200 空响应处理（那正是旧实现里 `let _ =` 造成的假死）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineClosed;

impl std::fmt::Display for EngineClosed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "引擎任务已退出（命令通道关闭）")
    }
}

impl std::error::Error for EngineClosed {}

/// serve 侧持有的引擎句柄：提交与取消是仅有的两个入口。
/// 真实 engine core 上线后此接口不变（协议先于实现——TDD 外→内的锚点）。
#[derive(Clone)]
pub struct EngineHandle {
    cmd_tx: UnboundedSender<EngineCommand>,
    next_id: Arc<AtomicU64>,
    /// 执行器上报的上下文窗口（serve 前置校验用；`usize::MAX` = 无上限/未知）
    ctx_limit: Arc<AtomicUsize>,
    /// 引擎侧统计（EngineCore 装配时挂上；MockEngine 也挂一份以便 /metrics 有数）
    stats: Option<Arc<EngineStats>>,
}

impl EngineHandle {
    pub fn new(cmd_tx: UnboundedSender<EngineCommand>) -> Self {
        Self {
            cmd_tx,
            next_id: Arc::new(AtomicU64::new(1)),
            ctx_limit: Arc::new(AtomicUsize::new(usize::MAX)),
            stats: None,
        }
    }

    /// 执行器上报的上下文窗口（token 数）。serve 用它做"prompt + max_tokens
    /// 越界"的前置校验——不在 serve 里硬编码任何模型参数。
    pub fn with_ctx_limit(mut self, n: usize) -> Self {
        self.ctx_limit = Arc::new(AtomicUsize::new(n));
        self
    }

    pub fn with_stats(mut self, stats: Arc<EngineStats>) -> Self {
        self.stats = Some(stats);
        self
    }

    pub fn ctx_limit(&self) -> usize {
        match &self.stats {
            Some(s) => s.ctx_limit(),
            None => self.ctx_limit.load(Ordering::Relaxed),
        }
    }

    pub fn stats(&self) -> Option<&Arc<EngineStats>> {
        self.stats.as_ref()
    }

    /// 引擎任务是否还活着：命令通道关闭 = 任务已退出（panic 或正常结束）。
    /// `/health` 据此报 503，submit 也据此在 send 前先判一次（拿到更明确的错误）。
    pub fn is_alive(&self) -> bool {
        !self.cmd_tx.is_closed()
    }

    /// 提交生成请求，返回 (请求 id, 事件流)。
    ///
    /// 通道已关闭（引擎任务不在）→ `Err(EngineClosed)`：调用方必须回 503，
    /// 绝不许当成"提交成功但内容为空"（服务假死的根因）。
    pub fn submit(
        &self,
        prompt_token_ids: Vec<u32>,
        max_tokens: u32,
        sampling: SamplingParams,
    ) -> Result<(RequestId, EventStream), EngineClosed> {
        let (tx, rx) = mpsc::unbounded_channel();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.cmd_tx
            .send(EngineCommand::Submit(EngineRequest {
                id,
                prompt_token_ids,
                max_tokens,
                sampling,
                respond_to: tx,
            }))
            .map_err(|_| EngineClosed)?;
        Ok((id, rx))
    }

    /// 取消：引擎已死时无事可做（在飞事件流随之关闭）——best-effort 语义不变
    pub fn cancel(&self, id: RequestId) {
        let _ = self.cmd_tx.send(EngineCommand::Cancel(id));
    }

    /// 前缀缓存清空（`/reset_prefix_cache`）：best-effort 同 cancel——命令经
    /// 通道按序处理，先于其后到达的 submit 生效
    pub fn clear_prefix_cache(&self) {
        let _ = self.cmd_tx.send(EngineCommand::ClearPrefixCache);
    }
}
