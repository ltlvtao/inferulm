//! wein-serve：OpenAI-compatible HTTP/SSE 层（M0 对外的唯一承诺接口，design/01-strategy Q8）。
//!
//! TDD 外→内的最外层：全部行为由 tests/ 下的验收用例钉死，
//! 引擎侧只依赖 wein-engine 的边界协议（MockEngine 可替换为真实 engine core）。
mod api;
mod detok;
mod openai;

pub use detok::{IncrementalDetok, MockTokenizer, Tokenizer};
pub use openai::ChatMessage;

use std::net::SocketAddr;
use std::sync::Arc;

use wein_engine::EngineHandle;

pub struct ServeState {
    pub model_id: String,
    pub tokenizer: Arc<dyn Tokenizer>,
    pub engine: EngineHandle,
}

/// 在给定地址启动 serving，返回实际绑定地址（测试传 127.0.0.1:0）。
/// CLI 与测试共用此入口——保证被测的就是被部署的。
pub async fn spawn_server(state: ServeState, addr: SocketAddr) -> std::io::Result<SocketAddr> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    let app = api::router(Arc::new(state));
    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            tracing::error!("server exited: {e}");
        }
    });
    Ok(bound)
}
