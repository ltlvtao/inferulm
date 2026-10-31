//! 错误语义补齐：panic → 500 JSON、`JsonRejection` → OpenAI 错误体。
//!
//! 旧实现里 serde 失败走 axum 原生纯文本（415/422），panic 直接断连接——
//! 客户端拿不到 OpenAI 风格的 `{"error": …}`，也没法区分"服务崩了"和"请求写错了"。
use std::panic::AssertUnwindSafe;

use axum::extract::rejection::JsonRejection;
use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;
use futures::FutureExt;

use crate::openai::openai_error;

/// 捕获 handler 内 panic（CatchPanic 手写版：依赖里没有 tower-http）。
///
/// 边界：只覆盖 handler 本体——SSE 流体的 panic（在 body 里执行）不在此列，
/// 那种情况由引擎侧的"不 panic + 失败帧"（`StepError`）兜底。
pub async fn catch_panic(req: Request, next: Next) -> Response {
    match AssertUnwindSafe(next.run(req)).catch_unwind().await {
        Ok(resp) => resp,
        Err(_payload) => {
            // panic 载荷多为 &str/String；打印后与请求一起记日志
            tracing::error!("handler panic 已被捕获，返回 500");
            openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "internal server error (handler panicked)",
            )
        }
    }
}

/// `JsonRejection` → OpenAI 错误体：**保留原状态码**（400/415/422），
/// body 换成 OpenAI 的 `{"error": …}` 形状，`type` 统一 `invalid_request_error`。
pub fn json_rejection(rej: JsonRejection) -> Response {
    let status = rej.status();
    openai_error(status, "invalid_request_error", rej.body_text())
}
