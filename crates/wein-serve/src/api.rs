//! OpenAI-compatible handlers。生成逻辑统一为 `generation()` 流，
//! 非流式与 SSE 是它的两个适配器——同一语义，两副皮（避免双轨，设计总则 §0-6）。
use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedReceiver;

use crate::detok::{IncrementalDetok, Tokenizer};
use crate::openai::{finish_reason_str, now, openai_error, parse_stop, ChatCompletionRequest, CompletionRequest};
use crate::ServeState;
use wein_engine::{EngineEvent, EngineHandle, FinishReason, RequestId, SamplingParams};

pub fn router(state: Arc<ServeState>) -> Router {
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .fallback(not_found)
        .with_state(state)
}

// ---------- 生成核心：引擎事件 → 统一内部流 ----------

/// 客户端断开（handler/stream 被 drop）→ drop 时取消引擎请求；自然结束则 disarm。
struct CancelOnDrop {
    engine: EngineHandle,
    id: RequestId,
    armed: bool,
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.engine.cancel(self.id);
        }
    }
}

enum GenOut {
    Text(String),
    Done { reason: FinishReason, tokens: u32 },
}

fn generation(
    engine: EngineHandle,
    tokenizer: Arc<dyn Tokenizer>,
    id: RequestId,
    mut events: UnboundedReceiver<EngineEvent>,
    stops: Vec<String>,
) -> impl Stream<Item = GenOut> + Send {
    async_stream::stream! {
        let mut guard = CancelOnDrop { engine, id, armed: true };
        let mut detok = IncrementalDetok::new(tokenizer);
        // stop 前缀可能跨 delta：扣住 (最长 stop - 1) 字节，避免吐出去又收不回
        let hold = stops.iter().map(|s| s.len()).max().unwrap_or(0).saturating_sub(1);
        let mut text = String::new();
        let mut yielded = 0usize;
        let mut tokens = 0u32;
        while let Some(ev) = events.recv().await {
            match ev {
                EngineEvent::Accepted => {}
                EngineEvent::Token(t) => {
                    tokens += 1;
                    let delta = detok.push(t);
                    text.push_str(&delta);
                    if let Some(i) = stop_index(&text, &stops) {
                        if i > yielded {
                            yield GenOut::Text(text[yielded..i].to_string());
                        }
                        // 保持 armed：drop 时取消引擎（引擎还在生成）
                        yield GenOut::Done { reason: FinishReason::Stop, tokens };
                        return;
                    }
                    let mut safe = text.len().saturating_sub(hold);
                    while safe > yielded && !text.is_char_boundary(safe) {
                        safe -= 1;
                    }
                    if safe > yielded {
                        yield GenOut::Text(text[yielded..safe].to_string());
                        yielded = safe;
                    }
                }
                EngineEvent::Finished(r) => {
                    detok.finish();
                    guard.armed = false;
                    if text.len() > yielded {
                        yield GenOut::Text(text[yielded..].to_string());
                    }
                    yield GenOut::Done { reason: r, tokens };
                    return;
                }
            }
        }
        // 引擎事件流意外关闭：按 Aborted 收尾（guard 已 disarm 语义由 Drop 处理）
        guard.armed = false;
        yield GenOut::Done { reason: FinishReason::Aborted, tokens };
    }
}

fn stop_index(text: &str, stops: &[String]) -> Option<usize> {
    stops.iter().filter_map(|s| text.find(s)).min()
}

// ---------- 校验 ----------

struct ValidatedGen {
    stops: Vec<String>,
    max_tokens: u32,
    sampling: SamplingParams,
}

fn validate_common(
    temperature: Option<f64>,
    top_p: Option<f64>,
    stop: &Option<Value>,
    max_tokens: Option<u32>,
) -> Result<ValidatedGen, Response> {
    let temperature = temperature.unwrap_or(1.0);
    if !(0.0..=2.0).contains(&temperature) {
        return Err(openai_error(StatusCode::BAD_REQUEST, "invalid_request_error", "temperature must be between 0 and 2"));
    }
    let top_p = top_p.unwrap_or(1.0);
    if !(0.0..=1.0).contains(&top_p) {
        return Err(openai_error(StatusCode::BAD_REQUEST, "invalid_request_error", "top_p must be between 0 and 1"));
    }
    let stops = parse_stop(stop)
        .map_err(|m| openai_error(StatusCode::BAD_REQUEST, "invalid_request_error", m))?;
    // M0 默认 16：未设上限的请求不至于无限流（真实引擎接入后改为模型上下文上限）
    let max_tokens = max_tokens.unwrap_or(16);
    Ok(ValidatedGen {
        stops,
        max_tokens,
        sampling: SamplingParams { temperature, top_p, stop_token_ids: Vec::new() },
    })
}

// ---------- handlers ----------

async fn models(State(st): State<Arc<ServeState>>) -> Response {
    Json(json!({
        "object": "list",
        "data": [{ "id": st.model_id, "object": "model", "created": now(), "owned_by": "wein" }]
    }))
        .into_response()
}

async fn not_found() -> Response {
    openai_error(StatusCode::NOT_FOUND, "invalid_request_error", "unknown route")
}

fn chunk(id: &str, created: u64, model: &str, delta: Value, finish: Option<&str>) -> String {
    json!({
        "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish, "logprobs": null }]
    })
    .to_string()
}

async fn chat_completions(State(st): State<Arc<ServeState>>, Json(req): Json<ChatCompletionRequest>) -> Response {
    if req.model != st.model_id {
        return openai_error(StatusCode::NOT_FOUND, "invalid_request_error", format!("The model `{}` does not exist", req.model));
    }
    let v = match validate_common(req.temperature, req.top_p, &req.stop, req.max_tokens) {
        Ok(v) => v,
        Err(r) => return r,
    };
    if req.messages.is_empty() {
        return openai_error(StatusCode::BAD_REQUEST, "invalid_request_error", "'messages' must not be empty");
    }
    // chat 模板（W5 真身）：tokenizer 有模板即渲染（模型侧 raise → 400）；
    // 无模板（mock/未接）回退占位包装——既有验收即 parity
    let prompt_ids = match st.tokenizer.render_chat_template(&req.messages) {
        Some(Ok(text)) => st.tokenizer.encode(&text),
        Some(Err(msg)) => {
            return openai_error(StatusCode::BAD_REQUEST, "invalid_request_error", msg)
        }
        None => {
            let prompt: String =
                req.messages.iter().map(|m| format!("{}: {}\n", m.role, m.content)).collect();
            st.tokenizer.encode(&prompt)
        }
    };
    let prompt_tokens = prompt_ids.len();

    // eos 停止（vLLM 默认）：命中即 Finished(Stop) 且不外吐
    let mut sampling = v.sampling;
    if let Some(eos) = st.tokenizer.eos_id() {
        sampling.stop_token_ids.push(eos);
    }
    let (id, events) = st.engine.submit(prompt_ids, v.max_tokens, sampling);
    let resp_id = format!("chatcmpl-{id}");
    let created = now();
    let model = st.model_id.clone();
    let gen = generation(st.engine.clone(), st.tokenizer.clone(), id, events, v.stops);

    if req.stream.unwrap_or(false) {
        let rid = resp_id.clone();
        let first = Event::default().data(chunk(&rid, created, &model, json!({"role": "assistant", "content": ""}), None));
        let body = futures::stream::once(async move { Ok::<_, Infallible>(first) })
            .chain(gen.map(move |o| {
                Ok(match o {
                    GenOut::Text(t) => Event::default().data(chunk(&resp_id, created, &model, json!({ "content": t }), None)),
                    GenOut::Done { reason, .. } => Event::default().data(chunk(&resp_id, created, &model, json!({}), Some(finish_reason_str(reason)))),
                })
            }))
            .chain(futures::stream::once(async { Ok(Event::default().data("[DONE]")) }));
        Sse::new(body).keep_alive(KeepAlive::default()).into_response()
    } else {
        let mut content = String::new();
        let (mut reason, mut tokens) = (FinishReason::Length, 0u32);
        let mut gen = Box::pin(gen);
        while let Some(o) = gen.next().await {
            match o {
                GenOut::Text(t) => content.push_str(&t),
                GenOut::Done { reason: r, tokens: n } => {
                    reason = r;
                    tokens = n;
                }
            }
        }
        Json(json!({
            "id": resp_id, "object": "chat.completion", "created": created, "model": model,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": content },
                "finish_reason": finish_reason_str(reason),
                "logprobs": null,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": tokens,
                "total_tokens": prompt_tokens + tokens as usize,
            }
        }))
            .into_response()
    }
}

async fn completions(State(st): State<Arc<ServeState>>, Json(req): Json<CompletionRequest>) -> Response {
    if req.model != st.model_id {
        return openai_error(StatusCode::NOT_FOUND, "invalid_request_error", format!("The model `{}` does not exist", req.model));
    }
    let v = match validate_common(req.temperature, req.top_p, &req.stop, req.max_tokens) {
        Ok(v) => v,
        Err(r) => return r,
    };
    let prompt_text = match &req.prompt {
        Value::String(s) => s.clone(),
        Value::Array(a) => match a.first() {
            Some(Value::String(s)) => s.clone(),
            _ => return openai_error(StatusCode::BAD_REQUEST, "invalid_request_error", "'prompt' array must contain only strings"),
        },
        _ => return openai_error(StatusCode::BAD_REQUEST, "invalid_request_error", "'prompt' must be a string or an array of strings"),
    };
    let prompt_ids = st.tokenizer.encode(&prompt_text);
    let prompt_tokens = prompt_ids.len();

    // eos 停止与 chat 端点同一默认（vLLM：原始续写也停于 eos）
    let mut sampling = v.sampling;
    if let Some(eos) = st.tokenizer.eos_id() {
        sampling.stop_token_ids.push(eos);
    }
    let (id, events) = st.engine.submit(prompt_ids, v.max_tokens, sampling);
    let resp_id = format!("cmpl-{id}");
    let created = now();
    let model = st.model_id.clone();
    let gen = generation(st.engine.clone(), st.tokenizer.clone(), id, events, v.stops);

    if req.stream.unwrap_or(false) {
        let body = gen.map(move |o| {
            Ok::<_, Infallible>(match o {
                GenOut::Text(t) => Event::default().data(json!({
                    "id": resp_id, "object": "text_completion", "created": created, "model": model,
                    "choices": [{ "index": 0, "text": t, "finish_reason": null }]
                }).to_string()),
                GenOut::Done { reason, .. } => Event::default().data(json!({
                    "id": resp_id, "object": "text_completion", "created": created, "model": model,
                    "choices": [{ "index": 0, "text": "", "finish_reason": finish_reason_str(reason) }]
                }).to_string()),
            })
        })
        .chain(futures::stream::once(async { Ok(Event::default().data("[DONE]")) }));
        Sse::new(body).keep_alive(KeepAlive::default()).into_response()
    } else {
        let mut content = String::new();
        let (mut reason, mut tokens) = (FinishReason::Length, 0u32);
        let mut gen = Box::pin(gen);
        while let Some(o) = gen.next().await {
            match o {
                GenOut::Text(t) => content.push_str(&t),
                GenOut::Done { reason: r, tokens: n } => {
                    reason = r;
                    tokens = n;
                }
            }
        }
        Json(json!({
            "id": resp_id, "object": "text_completion", "created": created, "model": model,
            "choices": [{ "index": 0, "text": content, "logprobs": null, "finish_reason": finish_reason_str(reason) }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": tokens,
                "total_tokens": prompt_tokens + tokens as usize,
            }
        }))
            .into_response()
    }
}
