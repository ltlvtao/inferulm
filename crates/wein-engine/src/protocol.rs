use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

pub type RequestId = u64;

/// 采样参数。M0 直通引擎（mock 忽略）；真实实现中采样在 exec 图尾部（02-architecture §8）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_p: f64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self { temperature: 1.0, top_p: 1.0 }
    }
}

/// serve → engine 的提交载荷。事件经 respond_to 单向回传（设计总则：单向数据流）。
#[derive(Debug)]
pub struct EngineRequest {
    pub id: RequestId,
    pub prompt_token_ids: Vec<u32>,
    pub max_tokens: u32,
    pub sampling: SamplingParams,
    pub respond_to: mpsc::UnboundedSender<EngineEvent>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// 命中停止条件（自然停止或 stop 参数）
    Stop,
    /// 达到 max_tokens
    Length,
    /// 被取消（客户端断开等）
    Aborted,
}

#[derive(Debug, Clone)]
pub enum EngineEvent {
    /// 已被引擎接受
    Accepted,
    /// 产出一个 token
    Token(u32),
    /// 请求终结
    Finished(FinishReason),
}

/// serve → engine 的全部命令（仅提交与取消两个入口）。
pub enum EngineCommand {
    Submit(EngineRequest),
    Cancel(RequestId),
}
