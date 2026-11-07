//! wein-serve：OpenAI-compatible HTTP/SSE 层（M0 对外的唯一承诺接口，design/01-strategy Q8）。
//!
//! TDD 外→内的最外层：全部行为由 tests/ 下的验收用例钉死，
//! 引擎侧只依赖 wein-engine 的边界协议（MockEngine 可替换为真实 engine core）。
//!
//! 可用性（任务书第二件）：`/health`、`/metrics`、X-Request-Id、panic→500、
//! 提交失败 503、SIGTERM 优雅退出（`Server::shutdown`）。
mod api;
mod detok;
mod metrics;
mod marker;
mod mw;
mod reasoning;
mod openai;
mod tools;

pub use detok::{IncrementalDetok, MockTokenizer, Tokenizer};
pub use metrics::{Inflight, Metrics};
pub use openai::ChatMessage;
pub use reasoning::{ReasoningSplitter, Seg};
pub use tools::{parser_enabled, Piece, ToolCall, ToolParser};

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::watch;
use wein_engine::EngineHandle;

pub struct ServeState {
    pub model_id: String,
    pub tokenizer: Arc<dyn Tokenizer>,
    pub engine: EngineHandle,
    /// serve 侧请求/token 计数（`/metrics` 的另一半来自 `EngineStats`）
    pub metrics: Arc<Metrics>,
    /// 在飞请求登记簿（drain 时报告未完成的 request_id）
    pub inflight: Arc<Inflight>,
    /// 优雅退出进行中：新请求一律 503 + `Connection: close`
    pub draining: Arc<AtomicBool>,
    /// 客户端没给 `X-Request-Id` 时的生成序号
    pub req_seq: Arc<AtomicU64>,
    /// 输出侧思维链解析（`</think>` 切分，vLLM `--reasoning-parser` 同位）。
    /// 默认关——只有「模板以 `<think>` 收尾」的模型该开（CLI `--reasoning-parser auto`
    /// 按模板探测）；关时全文进 `content`，行为与旧版完全一致
    pub reasoning: bool,
}

impl ServeState {
    pub fn new(
        model_id: impl Into<String>,
        tokenizer: Arc<dyn Tokenizer>,
        engine: EngineHandle,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            tokenizer,
            engine,
            metrics: Arc::new(Metrics::default()),
            inflight: Arc::new(Inflight::default()),
            draining: Arc::new(AtomicBool::new(false)),
            req_seq: Arc::new(AtomicU64::new(1)),
            reasoning: false,
        }
    }
}

/// 可优雅退出的服务句柄（CLI 的 SIGINT/SIGTERM 与测试共用）。
pub struct Server {
    pub addr: SocketAddr,
    shutdown: watch::Sender<bool>,
    draining: Arc<AtomicBool>,
    inflight: Arc<Inflight>,
    task: tokio::task::JoinHandle<()>,
}

impl Server {
    /// 触发优雅退出：①置 draining（新请求 503）②停止 accept
    /// ③等在飞请求收完（上限 `timeout`）④超时则 abort 并返回未完成的
    /// （引擎 request id, 客户端 request id）。
    ///
    /// 返回空表 = 干净收完（CLI 退出码 0）。
    pub async fn shutdown(self, timeout: Duration) -> Vec<(u64, String)> {
        // 先置 draining：已建立的 keep-alive 连接上后续请求会拿到 503 + close
        self.draining.store(true, Ordering::SeqCst);
        let _ = self.shutdown.send(true);
        let Server { inflight, mut task, .. } = self;
        match tokio::time::timeout(timeout, &mut task).await {
            Ok(_) => Vec::new(),
            Err(_) => {
                let left = inflight.snapshot();
                task.abort();
                left
            }
        }
    }
}

/// 绑定 + 起服务任务。返回（实际地址，任务，停止开关）。
async fn bind_and_serve(
    state: Arc<ServeState>,
    addr: SocketAddr,
) -> std::io::Result<(SocketAddr, tokio::task::JoinHandle<()>, watch::Sender<bool>)> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    let app = api::router(state);
    let (tx, mut rx) = watch::channel(false);
    let task = tokio::spawn(async move {
        let shut = async move {
            while rx.changed().await.is_ok() {
                if *rx.borrow() {
                    return;
                }
            }
        };
        if let Err(e) = axum::serve(listener, app).with_graceful_shutdown(shut).await {
            tracing::error!("server exited: {e}");
        }
    });
    Ok((bound, task, tx))
}

/// 在给定地址启动 serving，返回实际绑定地址（测试传 127.0.0.1:0）。
/// CLI 与测试共用此入口——保证被测的就是被部署的。
///
/// 便利面：不暴露退出把手（进程自然退出即可）。需要 SIGTERM drain 的调用方
/// 用 [`spawn_server_guarded`]。
pub async fn spawn_server(state: ServeState, addr: SocketAddr) -> std::io::Result<SocketAddr> {
    let (bound, task, tx) = bind_and_serve(Arc::new(state), addr).await?;
    // 通道 sender 挪进守护任务里（永不置 true）——否则 drop 即触发优雅退出
    tokio::spawn(async move {
        let _keep = tx;
        let _ = task.await;
    });
    Ok(bound)
}

/// 同上，但返回可 `shutdown()` 的句柄（SIGTERM/SIGINT 优雅退出用）。
pub async fn spawn_server_guarded(
    state: ServeState,
    addr: SocketAddr,
) -> std::io::Result<Server> {
    let draining = state.draining.clone();
    let inflight = state.inflight.clone();
    let (bound, task, shutdown) = bind_and_serve(Arc::new(state), addr).await?;
    Ok(Server { addr: bound, shutdown, draining, inflight, task })
}
