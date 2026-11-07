//! OpenAI wire format：请求 DTO 与错误体。响应用 serde_json::json! 构造（结构多态，M0 不值得全量建模）。
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use wein_engine::FinishReason;

/// chat 消息。**`content` 可空**（OpenAI 合同：带 `tool_calls` 的 assistant
/// 消息 content 常为 null）；`tool_calls` / `tool_call_id` / `reasoning_content`
/// 原样透传给 chat 模板（模板按 `message.tool_calls[i].function.{name,arguments}`
/// 回放历史调用、按 `message.reasoning_content` 回放思维链）。
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
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
    /// 函数调用（OpenAI tools 数组，原样喂模板 `tools` 变量）。
    /// `Some` ⇒ 本次响应启用工具调用解析（见 `tools.rs`）
    #[serde(default)]
    pub tools: Option<Value>,
    /// "none"/"auto"/{"type":"function",...}——模板未消费，仅记录合法性
    #[serde(default)]
    pub tool_choice: Option<Value>,
    /// vLLM 扩展：模板顶层 kwarg（如 `enable_thinking: false`）
    #[serde(default)]
    pub chat_template_kwargs: Option<Value>,
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
        // **兼容性取舍**：OpenAI 规范的 finish_reason 只有 stop/length/
        // content_filter/tool_calls，没有 "error"。把"引擎侧异常终结"
        // （`Done{Aborted}`、执行器可恢复失败 `StepError::Invalid`）报成
        // stop/length 会让客户端把**截断/失败**当成正常完成——这正是旧实现
        // 的可用性缺陷。故这里输出规范的超集值 "error"，并在非流式 body
        // 顶层附 `wein_finish`（"aborted"/"error"）、流式在终止帧前先发
        // `event: error`，客户端可无歧义判别（`tests/recovery.rs` 钉死）。
        FinishReason::Aborted | FinishReason::Error => "error",
    }
}

/// 非流式 body 的可判别字段值（`wein_finish`）：只有异常收尾才 Some。
pub fn wein_finish(r: FinishReason) -> Option<&'static str> {
    match r {
        FinishReason::Aborted => Some("aborted"),
        FinishReason::Error => Some("error"),
        FinishReason::Stop | FinishReason::Length => None,
    }
}
