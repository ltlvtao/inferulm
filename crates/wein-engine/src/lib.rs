//! wein-engine：engine ↔ serve 的边界层。
//!
//! M0 的 TDD 从最外层（serve）开始，本 crate 目前只包含：
//! - 边界协议（protocol）：请求/事件/命令——真实 engine core 将来实现同一协议
//! - EngineHandle：serve 侧唯一入口
//! - MockEngine：可编程引擎替身，验收测试与 bench 共用
//!
//! 真实 engine core（调度器 / KV / exec 线程）由 W4 按 design/02-architecture.md 填充，
//! 届时 MockEngine 退役为测试专用。
mod handle;
mod mock;
mod protocol;

pub use handle::{EngineHandle, EventStream};
pub use mock::{MockEngine, MockScript, MockState};
pub use protocol::{EngineCommand, EngineEvent, EngineRequest, FinishReason, RequestId, SamplingParams};
