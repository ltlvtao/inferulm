use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

pub type RequestId = u64;

/// 采样参数。M0 直通引擎（mock 忽略）；真实实现中采样在 exec 图尾部（02-architecture §8）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_p: f64,
    /// 停止词表（chat eos 等，W5）：采样命中即 Finished(Stop)，该 token 不外吐
    /// （vLLM stop_token_ids 语义）。检查发生在 engine core 的 token 发射处
    pub stop_token_ids: Vec<u32>,
}

impl Default for SamplingParams {
    /// 引擎协议的默认 = **greedy**（temperature 0 ⇒ 等价 argmax，与"设备端
    /// 只有 argmax"时代逐位一致）。OpenAI 兼容层的默认是 temperature = 1.0
    /// （见 wein-serve 的 `validate_common`）：客户端不填温度即按分布采样。
    /// 两个默认不同是刻意的——协议默认不许悄悄改变既有引擎集成的输出。
    fn default() -> Self {
        Self { temperature: 0.0, top_p: 1.0, stop_token_ids: Vec::new() }
    }
}

impl SamplingParams {
    /// serve 侧校验过的 temperature/top_p → 执行器采样参数。
    ///
    /// `top_k` 缺省取 64：serve 的 `SamplingParams` 没有 top_k 字段（OpenAI
    /// 兼容层也不透传），而设备端采样器只支持 k ≤ 64 的 top-k 路径。取满
    /// 64 是"最宽的可支持集合"，**不是**全词表——全词表 top-p 要对 248K
    /// 项降序累加（等价全排序，见 `Executor::set_sampling` 的注释），不做。
    /// 需要别的 k 请走 `Executor::set_sampling` 直连（测试/内部调用）。
    pub fn sample_params(&self) -> SampleParams {
        SampleParams {
            temperature: self.temperature,
            top_k: 64,
            top_p: self.top_p,
            seed: None,
        }
    }
}

/// 设备端采样器的逐请求参数（exec 图尾 `sample_rows` 的合同）。
///
/// `Default` = **greedy**（temperature ≤ 0 → 走原 argmax 分支，逐位不变）。
/// `top_k` 是**必须**的：设备端只有 top-k（k ≤ 64）路径——全词表采样
/// （`top_k = 0`，无论 top_p 取多少）要 248K 项全排序，执行器直接拒。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SampleParams {
    /// ≤ 0（含 NaN）→ 等价 argmax
    pub temperature: f64,
    /// 1..=64（执行器夹到 ≤ 64）；0 = 全词表 → 不支持
    pub top_k: u32,
    /// (0,1]；执行器按"累加概率 ≥ top_p 即止、至少留 1 个"截断
    pub top_p: f64,
    /// None → 执行器用全局递增计数器派生（不同请求互不相关，同 seed 可复现）
    pub seed: Option<u64>,
}

impl Default for SampleParams {
    fn default() -> Self {
        // 默认 greedy：与"exec 只会 argmax"的历史行为逐位一致
        Self { temperature: 0.0, top_k: 0, top_p: 1.0, seed: None }
    }
}

impl SampleParams {
    pub fn greedy() -> Self {
        Self::default()
    }
    /// temp ≤ 0（含 NaN）→ greedy 路径（不进口采样 kernel）
    pub fn is_greedy(&self) -> bool {
        !(self.temperature > 0.0)
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
    /// **容量类**执行失败（页池/状态行/位置上限——请求根本装不下，见
    /// `Executor::take_failures`）。2026-11-08 从 `Length` 拆出：Agent 压测里
    /// 16 会话撞 KV 池时的对外表现是 `finish_reason: "length"` + **零 token**，
    /// 客户端会把它当成"被截断、可以重试更短"，而真相是服务端容量不足
    /// （要扩容或排队）——错误语义错位与"假死"同类，必须可判别
    Capacity,
    /// 被取消（客户端断开等）
    Aborted,
    /// 请求级执行失败（执行器拒绝/真异常：vocab 越界、采样参数不可满足等）。
    ///
    /// OpenAI 协议里没有 error 这个 finish_reason——serve 侧的兼容取舍见
    /// `wein-serve` 的 `finish_reason_str`：body 里仍写 `"error"`，并额外给
    /// `wein_finish` 判别字段/响应头，客户端不会把它误读成正常结束。
    Error,
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

/// serve → engine 的全部命令（提交/取消 + 前缀缓存清空）。
pub enum EngineCommand {
    Submit(EngineRequest),
    Cancel(RequestId),
    /// `/reset_prefix_cache`：驱逐全部可逐前缀（在飞请求的 pin 块保留）。
    /// 经命令通道按序处理，先于其后到达的 Submit 生效
    ClearPrefixCache,
}
