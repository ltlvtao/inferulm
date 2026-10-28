//! OpenAI wire format：请求 DTO 与错误体。响应用 serde_json::json! 构造（结构多态，M0 不值得全量建模）。
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use wein_engine::FinishReason;

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: Option<bool>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    /// OpenAI 语义：string 或 string[]（≤4 个）
    pub stop: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    /// OpenAI 语义：string 或 string[]，M0 只取第一个
    pub prompt: Value,
    pub stream: Option<bool>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub stop: Option<Value>,
}

/// OpenAI 风格错误体（非 2xx 一律走这里）。
pub fn openai_error(status: StatusCode, err_type: &str, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({
            "error": { "message": message.into(), "type": err_type, "param": null, "code": null }
        })),
    )
        .into_response()
}

pub fn parse_stop(v: &Option<Value>) -> Result<Vec<String>, String> {
    match v {
        None | Some(Value::Null) => Ok(vec![]),
        Some(Value::String(s)) => Ok(vec![s.clone()]),
        Some(Value::Array(a)) => {
            let mut out = Vec::new();
            for item in a {
                match item {
                    Value::String(s) => out.push(s.clone()),
                    _ => return Err("'stop' array must contain only strings".into()),
                }
            }
            Ok(out)
        }
        Some(_) => Err("'stop' must be a string or an array of strings".into()),
    }
}

pub fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub fn finish_reason_str(r: FinishReason) -> &'static str {
    match r {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        // OpenAI 协议无 aborted；客户端断开时响应本就无人收
        FinishReason::Aborted => "stop",
    }
}
