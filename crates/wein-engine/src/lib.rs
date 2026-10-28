//! wein-engine：engine ↔ serve 的边界层 + engine core 组装。
//!
//! - 边界协议（protocol）：请求/事件/命令——所有引擎实现共用的唯一协议
//! - EngineHandle：serve 侧唯一入口
//! - EngineCore：真实 engine core 第一步——wein-sched 调度器 + 可插拔执行器
//! - MockEngine：可编程引擎替身（验收测试与 bench 的对照组）
mod core;
mod handle;
mod mock;
mod protocol;

pub use core::{CoreConfig, EngineCore, EngineStats, Executor, ScriptedExecutor};
pub use handle::{EngineHandle, EventStream};
pub use mock::{MockEngine, MockScript, MockState};
pub use protocol::{EngineCommand, EngineEvent, EngineRequest, FinishReason, RequestId, SamplingParams};
