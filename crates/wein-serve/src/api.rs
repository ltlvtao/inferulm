//! OpenAI-compatible handlers。生成逻辑统一为 `generation()` 流，
//! 非流式与 SSE 是它的两个适配器——同一语义，两副皮（避免双轨，设计总则 §0-6）。
//!
//! 可用性刀（任务书第二件）：越界/容量/引擎故障一律走**协议错误**——
//! 前置 400（`ctx_limit` 由执行器上报）、提交失败 503（绝不吞成 200 空响应）、
//! 执行器失败以 `finish_reason: "error"` + 可判别字段 `wein_finish` 收尾。
use std::convert::Infallible;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedReceiver;

use crate::detok::{IncrementalDetok, Tokenizer};
use crate::mw;
use crate::openai::{
    finish_reason_str, now, openai_error, parse_stop, wein_finish, ChatCompletionRequest,
    CompletionRequest,
};
use crate::reasoning::{ReasoningSplitter, Seg};
use crate::tools::{call_delta, parser_enabled, Piece, ToolParser};
use crate::ServeState;
use wein_engine::{EngineEvent, EngineHandle, FinishReason, RequestId, SamplingParams};

/// 执行器未上报上下文窗口（`usize::MAX`）时的缺省生成长度。
/// 真实执行器（qwen/toy）都上报 `ctx_limit`，这条只服务替身/无窗口引擎。
const DEFAULT_MAX_TOKENS: u32 = 256;

pub fn router(state: Arc<ServeState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/reset_prefix_cache", post(reset_prefix_cache))
        .fallback(not_found)
        // 最外层：handler panic → 500 JSON（手写 CatchPanic，依赖里没有 tower-http）
        .layer(axum::middleware::from_fn(mw::catch_panic))
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

/// 生成流的横切上下文：日志（request_id/耗时）与指标（tokens/结果分类）。
struct GenCtx {
    request_id: String,
    model: String,
    prompt_tokens: usize,
    st: Arc<ServeState>,
}

fn generation(
    engine: EngineHandle,
    tokenizer: Arc<dyn Tokenizer>,
    id: RequestId,
    mut events: UnboundedReceiver<EngineEvent>,
    stops: Vec<String>,
    ctx: GenCtx,
) -> impl Stream<Item = GenOut> + Send {
    async_stream::stream! {
        let mut guard = CancelOnDrop { engine, id, armed: true };
        // 在飞登记：流结束或客户端断开（本流被 drop）时注销
        let mut inflight = ctx.st.inflight.entry(id, ctx.request_id.clone());
        let mut detok = IncrementalDetok::new(tokenizer);
        // stop 前缀可能跨 delta：扣住 (最长 stop - 1) 字节，避免吐出去又收不回
        let hold = stops.iter().map(|s| s.len()).max().unwrap_or(0).saturating_sub(1);
        let mut text = String::new();
        let mut yielded = 0usize;
        let mut tokens = 0u32;
        let t0 = Instant::now();
        let mut ttft_ms: Option<u64> = None;
        let mut outcome: Option<FinishReason> = None;
        while let Some(ev) = events.recv().await {
            match ev {
                EngineEvent::Accepted => {}
                EngineEvent::Token(t) => {
                    tokens += 1;
                    ctx.st.metrics.add_completion_tokens(1);
                    if ttft_ms.is_none() {
                        let ms = t0.elapsed().as_millis() as u64;
                        ttft_ms = Some(ms);
                        tracing::info!(
                            request_id = %ctx.request_id, model = %ctx.model,
                            prompt_tokens = ctx.prompt_tokens, ttft_ms = ms,
                            "first token"
                        );
                    }
                    let delta = detok.push(t);
                    text.push_str(&delta);
                    if let Some(i) = stop_index(&text, &stops) {
                        if i > yielded {
                            yield GenOut::Text(text[yielded..i].to_string());
                        }
                        // 命中 stop 串：后续文本全部丢弃（yielded 推到底，
                        // 收尾时不会再吐含 stop 的尾巴）
                        yielded = text.len();
                        // 保持 armed：drop 时取消引擎（引擎还在生成）
                        outcome = Some(FinishReason::Stop);
                        break;
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
                    outcome = Some(r);
                    break;
                }
            }
        }
        if text.len() > yielded {
            yield GenOut::Text(text[yielded..].to_string());
        }
        // 事件流意外关闭（引擎任务消失）→ Aborted：guard 已无意义，置 disarm
        let reason = match outcome {
            Some(r) => r,
            None => {
                guard.armed = false;
                FinishReason::Aborted
            }
        };
        // 结果四分类（与 /metrics 的 wein_requests_total 一一对应）
        match reason {
            FinishReason::Stop | FinishReason::Length => {
                ctx.st.metrics.finished.fetch_add(1, Ordering::Relaxed);
            }
            FinishReason::Aborted => {
                ctx.st.metrics.cancelled.fetch_add(1, Ordering::Relaxed);
            }
            FinishReason::Error | FinishReason::Capacity => {
                ctx.st.metrics.failed.fetch_add(1, Ordering::Relaxed);
            }
        }
        let total_ms = t0.elapsed().as_millis() as u64;
        let finish = finish_reason_str(reason);
        tracing::info!(
            request_id = %ctx.request_id, model = %ctx.model,
            prompt_tokens = ctx.prompt_tokens, completion_tokens = tokens,
            ttft_ms = ttft_ms.unwrap_or(0), total_ms, finish_reason = finish,
            "request finished"
        );
        inflight.finish();
        yield GenOut::Done { reason, tokens };
    }
}

fn stop_index(text: &str, stops: &[String]) -> Option<usize> {
    stops.iter().filter_map(|s| text.find(s)).min()
}

// ---------- 请求标识 ----------

/// 接受客户端 `X-Request-Id`（去空白、截断到 128 字符），没给就生成。
fn request_id(st: &ServeState, headers: &HeaderMap) -> String {
    if let Some(v) = headers.get("x-request-id").and_then(|v| v.to_str().ok()) {
        let s: String = v.trim().chars().take(128).collect();
        if !s.is_empty() {
            return s;
        }
    }
    format!("req-{}", st.req_seq.fetch_add(1, Ordering::Relaxed))
}

/// 把 request_id 回写到响应头（错误响应也带，便于客户端对账）
fn with_rid(mut resp: Response, rid: &str) -> Response {
    if let Ok(v) = HeaderValue::from_str(rid) {
        resp.headers_mut().insert("x-request-id", v);
    }
    resp
}

fn draining_response(request_id_hint: &str) -> Response {
    tracing::warn!(request_id = %request_id_hint, "请求到达时服务正在退出：503");
    let mut resp = openai_error(
        StatusCode::SERVICE_UNAVAILABLE,
        "server_error",
        "server is shutting down (draining); retry on another instance",
    );
    resp.headers_mut()
        .insert(header::CONNECTION, HeaderValue::from_static("close"));
    resp
}

// ---------- 校验 ----------

struct ValidatedGen {
    stops: Vec<String>,
    sampling: SamplingParams,
}

fn validate_common(
    temperature: Option<f64>,
    top_p: Option<f64>,
    stop: &Option<Value>,
) -> Result<ValidatedGen, Response> {
    // 缺省 **greedy**（temperature = 0）：请求里显式给 temperature > 0 才按
    // 分布采样（设备端 top-k ≤ 64 的 top-p 采样器）。确定性是既有的服务
    // 合同（q11/q12 与 CLI 的续写可复现），故不跟 OpenAI 的 1.0 缺省
    let temperature = temperature.unwrap_or(0.0);
    if !(0.0..=2.0).contains(&temperature) {
        return Err(openai_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "temperature must be between 0 and 2",
        ));
    }
    let top_p = top_p.unwrap_or(1.0);
    if !(0.0..=1.0).contains(&top_p) {
        return Err(openai_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "top_p must be between 0 and 1",
        ));
    }
    let stops = parse_stop(stop)
        .map_err(|m| openai_error(StatusCode::BAD_REQUEST, "invalid_request_error", m))?;
    Ok(ValidatedGen {
        stops,
        sampling: SamplingParams { temperature, top_p, stop_token_ids: Vec::new() },
    })
}

/// `max_tokens` 缺省与越界前置校验。
///
/// - 缺省 = **剩余上下文**：`ctx_limit - prompt_tokens - 1`（留 1 token 余量，
///   生成的第 1 个 token 必须落在窗口内）。旧实现的硬编码 16 已删除——那让
///   "不写 max_tokens" 的客户端莫名只拿到 16 个 token。
/// - `ctx_limit` 由执行器经 `Executor::ctx_limit()` → `EngineStats` →
///   `EngineHandle::ctx_limit()` 上报（**serve 不硬编码任何模型参数**）。
/// - `prompt + max_tokens > ctx_limit` → 400（旧实现越界即引擎 panic：在飞
///   请求全部 Aborted、后续请求全部 200 空内容 = 服务假死）
fn resolve_max_tokens(
    st: &ServeState,
    requested: Option<u32>,
    prompt_tokens: usize,
) -> Result<u32, Response> {
    let ctx = st.engine.ctx_limit();
    let bounded = ctx != usize::MAX;
    let max_tokens = match requested {
        Some(m) => m,
        None if bounded => ctx.saturating_sub(prompt_tokens + 1).min(u32::MAX as usize) as u32,
        None => DEFAULT_MAX_TOKENS,
    };
    if bounded && prompt_tokens + max_tokens as usize > ctx {
        return Err(openai_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "This model's maximum context length is {ctx} tokens. However, your request \
                 has {prompt_tokens} prompt tokens and max_tokens={max_tokens}, which sum to \
                 {} tokens. Please reduce the length.",
                prompt_tokens + max_tokens as usize
            ),
        ));
    }
    Ok(max_tokens)
}

struct Prepared {
    max_tokens: u32,
    sampling: SamplingParams,
    stops: Vec<String>,
}

fn prepare(
    st: &ServeState,
    rid: &str,
    prompt_ids: Vec<u32>,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    stop: &Option<Value>,
) -> Result<(Vec<u32>, usize, Prepared), Response> {
    let v = validate_common(temperature, top_p, stop)?;
    let prompt_tokens = prompt_ids.len();
    let max_tokens = resolve_max_tokens(st, max_tokens, prompt_tokens)?;
    // eos 停止（vLLM 默认）：命中即 Finished(Stop) 且不外吐
    let mut sampling = v.sampling;
    if let Some(eos) = st.tokenizer.eos_id() {
        sampling.stop_token_ids.push(eos);
    }
    tracing::debug!(request_id = %rid, prompt_tokens, max_tokens, "request prepared");
    Ok((prompt_ids, prompt_tokens, Prepared { max_tokens, sampling, stops: v.stops }))
}

// ---------- 健康与指标 ----------

async fn health(State(st): State<Arc<ServeState>>) -> Response {
    let steps = st.engine.stats().map(|s| s.step_count()).unwrap_or(0);
    let draining = st.draining.load(Ordering::SeqCst);
    if st.engine.is_alive() {
        Json(json!({
            "status": "ok",
            "steps": steps,
            "draining": draining,
            "inflight": st.inflight.len(),
        }))
        .into_response()
    } else {
        // 引擎任务死亡 = 服务已不能生成任何东西：必须是 503（探针据此摘流量）
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "error",
                "error": "engine task exited",
                "steps": steps,
            })),
        )
            .into_response()
    }
}

async fn metrics(State(st): State<Arc<ServeState>>) -> Response {
    let text = st.metrics.render(st.engine.stats().map(|a| a.as_ref()));
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")], text).into_response()
}

/// 前缀缓存清空（对齐 vLLM 的 dev 端点，长上下文探针的冷跑依赖它）。
/// 命令经引擎通道按序处理——先于其后到达的 Submit 生效，无需等待回执
async fn reset_prefix_cache(State(st): State<Arc<ServeState>>) -> Response {
    if !st.engine.is_alive() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"success": false, "error": "engine task exited"})),
        )
            .into_response();
    }
    st.engine.clear_prefix_cache();
    Json(json!({"success": true})).into_response()
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

/// 线格式：chat 用 `delta`/`chat.completion(.chunk)`，completions 用
/// `text`/`text_completion(.chunk)`（流式对象的 chunk 后缀是 OpenAI 合同）
#[derive(Clone, Copy, PartialEq, Eq)]
enum Wire {
    Chat,
    Text,
}

impl Wire {
    fn stream_object(self) -> &'static str {
        match self {
            Wire::Chat => "chat.completion.chunk",
            Wire::Text => "text_completion.chunk",
        }
    }
    fn body_object(self) -> &'static str {
        match self {
            Wire::Chat => "chat.completion",
            Wire::Text => "text_completion",
        }
    }
    fn id_prefix(self) -> &'static str {
        match self {
            Wire::Chat => "chatcmpl",
            Wire::Text => "cmpl",
        }
    }
}

fn chat_chunk(id: &str, created: u64, model: &str, delta: Value, finish: Option<&str>) -> String {
    json!({
        "id": id, "object": Wire::Chat.stream_object(), "created": created, "model": model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish, "logprobs": null }]
    })
    .to_string()
}

fn text_chunk(id: &str, created: u64, model: &str, text: &str, finish: Option<&str>) -> String {
    json!({
        "id": id, "object": Wire::Text.stream_object(), "created": created, "model": model,
        "choices": [{ "index": 0, "text": text, "finish_reason": finish }]
    })
    .to_string()
}

/// 切分片 → SSE 帧：`Reasoning` 走 `delta.reasoning_content`（vLLM 同名字段），
/// 正文经工具解析器后再下发（**顺序不可颠倒**：推理段里常出现 `<tool_call>` 的
/// 格式示例，先过工具解析器会造出假调用）
fn seg_events(
    rid: &str,
    created: u64,
    model: &str,
    segs: Vec<Seg>,
    tp: Option<&mut ToolParser>,
    n: &mut usize,
) -> Vec<Event> {
    let mut evs = Vec::new();
    let mut tp = tp;
    for seg in segs {
        match seg {
            Seg::Reasoning(s) if s.is_empty() => {}
            Seg::Reasoning(s) => evs.push(Event::default().data(chat_chunk(
                rid,
                created,
                model,
                json!({ "reasoning_content": s }),
                None,
            ))),
            Seg::Content(s) if s.is_empty() => {}
            Seg::Content(s) => match tp.as_deref_mut() {
                Some(par) => evs.extend(tool_events(rid, created, model, par.feed(&s), n)),
                None => evs.push(Event::default().data(chat_chunk(
                    rid,
                    created,
                    model,
                    json!({ "content": s }),
                    None,
                ))),
            },
        }
    }
    evs
}

/// 解析器输出的一片 → SSE 帧（调用片转 OpenAI `tool_calls` 增量，`index` 递增）
fn tool_events(rid: &str, created: u64, model: &str, ps: Vec<Piece>, n: &mut usize) -> Vec<Event> {
    let mut evs = Vec::new();
    for p in ps {
        match p {
            Piece::Content(s) if s.is_empty() => {}
            Piece::Content(s) => evs.push(Event::default().data(chat_chunk(
                rid,
                created,
                model,
                json!({ "content": s }),
                None,
            ))),
            Piece::Call(c) => {
                evs.push(Event::default().data(chat_chunk(
                    rid,
                    created,
                    model,
                    call_delta(*n, &c),
                    None,
                )));
                *n += 1;
            }
        }
    }
    evs
}

/// 异常收尾的 SSE 帧：先一帧 `event: error`（可判别），再一帧带
/// `finish_reason = "error"` 的终止块。
fn error_frames(rid: &str, reason: FinishReason) -> Vec<Event> {
    let Some(kind) = wein_finish(reason) else { return vec![] };
    vec![
        Event::default().event("error").data(
            json!({
                "error": {
                    "message": format!("generation ended abnormally: {kind}"),
                    "type": "server_error",
                    "param": null,
                    "code": kind,
                }
            })
            .to_string(),
        ),
        Event::default().data(
            json!({
                "id": rid, "object": "error", "finish_reason": finish_reason_str(reason),
                "wein_finish": kind,
            })
            .to_string(),
        ),
    ]
}

/// 提交 + 组装响应（两条路由共用：同一语义，两副皮）。
async fn dispatch(
    st: &Arc<ServeState>,
    wire: Wire,
    rid: String,
    prompt_ids: Vec<u32>,
    prompt_tokens: usize,
    p: Prepared,
    stream: bool,
    // `Some(tools)` = 把生成流按工具调用解析（OpenAI `tool_calls` 形状）
    tools: Option<Value>,
    // 输出侧思维链切分（`</think>`）：推理进 `reasoning_content`
    reasoning: bool,
) -> Response {
    let (id, events) = match st.engine.submit(prompt_ids, p.max_tokens, p.sampling) {
        Ok(v) => v,
        Err(e) => {
            // 引擎任务已退出（channel 关闭）——旧实现把它吞成 200 空内容（假死）。
            // 这里是唯一的合法信号：必须 503。
            st.metrics.failed.fetch_add(1, Ordering::Relaxed);
            tracing::error!(request_id = %rid, error = %e, "engine submit failed");
            return openai_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                format!("engine unavailable: {e}"),
            );
        }
    };
    st.metrics.accepted.fetch_add(1, Ordering::Relaxed);
    st.metrics.add_prompt_tokens(prompt_tokens);
    let resp_id = format!("{}-{rid}", wire.id_prefix());
    let created = now();
    let model = st.model_id.clone();
    let ctx = GenCtx {
        request_id: rid.clone(),
        model: model.clone(),
        prompt_tokens,
        st: st.clone(),
    };
    let gen = generation(st.engine.clone(), st.tokenizer.clone(), id, events, p.stops, ctx);

    if stream {
        let first = match wire {
            Wire::Chat => Event::default().data(chat_chunk(
                &resp_id,
                created,
                &model,
                json!({"role": "assistant", "content": ""}),
                None,
            )),
            Wire::Text => Event::default().data(text_chunk(&resp_id, created, &model, "", None)),
        };
        let rid_s = resp_id.clone();
        let model_s = model.clone();
        let mut tp = tools.map(|t| ToolParser::new(resp_id.clone(), t));
        let mut rp = reasoning.then(ReasoningSplitter::new);
        let body = async_stream::stream! {
            yield Ok::<Event, Infallible>(first);
            let mut n_call = 0usize;
            let mut gen = Box::pin(gen);
            while let Some(o) = gen.next().await {
                match o {
                    GenOut::Text(t) => {
                        if wire == Wire::Chat && (rp.is_some() || tp.is_some()) {
                            let segs = match rp.as_mut() {
                                Some(r) => r.feed(&t),
                                None => vec![Seg::Content(t)],
                            };
                            for ev in seg_events(&rid_s, created, &model_s, segs, tp.as_mut(), &mut n_call) {
                                yield Ok(ev);
                            }
                            continue;
                        }
                        let ev = match wire {
                            Wire::Chat => Event::default().data(chat_chunk(
                                &rid_s, created, &model_s, json!({ "content": t }), None,
                            )),
                            Wire::Text => Event::default().data(text_chunk(
                                &rid_s, created, &model_s, &t, None,
                            )),
                        };
                        yield Ok(ev);
                    }
                    GenOut::Done { reason, .. } => {
                        // 截断/结束前先把解析器残留吐干净（未闭合的思考→推理尾巴，
                        // 未闭合的调用→正文，都不丢字）
                        if wire == Wire::Chat {
                            let segs = match rp.as_mut() {
                                Some(r) => r.finish(),
                                None => vec![],
                            };
                            for ev in seg_events(&rid_s, created, &model_s, segs, tp.as_mut(), &mut n_call) {
                                yield Ok(ev);
                            }
                        }
                        if let Some(par) = tp.as_mut() {
                            for ev in tool_events(&rid_s, created, &model_s, par.finish(), &mut n_call) {
                                yield Ok(ev);
                            }
                        }
                        for ev in error_frames(&rid_s, reason) {
                            yield Ok(ev);
                        }
                        // 有调用且正常停 ⇒ OpenAI 合同 finish_reason = "tool_calls"
                        let fin = if n_call > 0 && matches!(reason, FinishReason::Stop) {
                            "tool_calls"
                        } else {
                            finish_reason_str(reason)
                        };
                        let ev = match wire {
                            Wire::Chat => Event::default().data(chat_chunk(
                                &rid_s, created, &model_s, json!({}), Some(fin),
                            )),
                            Wire::Text => Event::default().data(text_chunk(
                                &rid_s, created, &model_s, "", Some(fin),
                            )),
                        };
                        yield Ok(ev);
                    }
                }
            }
            yield Ok(Event::default().data("[DONE]"));
        };
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
        // 思维链切分（非流式）：同一状态机整段过一遍 ⇒ 与流式同一语义
        let mut reasoning_text = String::new();
        if reasoning {
            let mut rs = ReasoningSplitter::new();
            let mut segs = rs.feed(&content);
            segs.extend(rs.finish());
            content.clear();
            for seg in segs {
                match seg {
                    Seg::Reasoning(s) => reasoning_text.push_str(&s),
                    Seg::Content(s) => content.push_str(&s),
                }
            }
        }
        // 工具调用解析（非流式）：整段文本过同一个状态机 ⇒ 与流式同一语义
        let mut calls: Vec<Value> = Vec::new();
        if let Some(t) = tools {
            let mut par = ToolParser::new(resp_id.clone(), t);
            let mut ps = par.feed(&content);
            ps.extend(par.finish());
            let mut text = String::new();
            for piece in ps {
                match piece {
                    Piece::Content(s) => text.push_str(&s),
                    Piece::Call(c) => { calls.push(call_delta(calls.len(), &c)); }
                }
            }
            content = text;
        }
        let fin = if !calls.is_empty() && matches!(reason, FinishReason::Stop) {
            "tool_calls"
        } else {
            finish_reason_str(reason)
        };
        let usage = json!({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": tokens,
            "total_tokens": prompt_tokens + tokens as usize,
        });
        let choices = match wire {
            Wire::Chat => {
                // 只有调用没有正文时 content 必须是 null（OpenAI 合同）
                let mut message = json!({ "role": "assistant" });
                // 只有调用时 content 必须是 null；`</think>` 后残留的空白也算"没有正文"
                message["content"] = if content.trim().is_empty() && !calls.is_empty() {
                    Value::Null
                } else {
                    json!(content)
                };
                if !reasoning_text.is_empty() {
                    message["reasoning_content"] = json!(reasoning_text);
                }
                if !calls.is_empty() {
                    // 非流式的 tool_calls 不带 index（流式才有）
                    let flat: Vec<Value> = calls
                        .iter()
                        .map(|c| {
                            let mut c = c["tool_calls"][0].clone();
                            c.as_object_mut().map(|o| o.remove("index"));
                            c
                        })
                        .collect();
                    message["tool_calls"] = Value::Array(flat);
                }
                json!([{
                    "index": 0, "message": message, "finish_reason": fin, "logprobs": null,
                }])
            }
            Wire::Text => json!([{
                "index": 0, "text": content, "logprobs": null, "finish_reason": fin,
            }]),
        };
        let mut body = json!({
            "id": resp_id, "object": wire.body_object(), "created": created, "model": model,
            "choices": choices,
            "usage": usage,
        });
        // OpenAI 规范里没有 "error" 这个 finish_reason——兼容性取舍：异常收尾
        // 用 finish_reason:"error"，并在 body 顶层追加 `wein_finish`
        // （"aborted"/"error"）让客户端可判别（`recovery.rs` 钉死）
        if let Some(kind) = wein_finish(reason) {
            body["wein_finish"] = json!(kind);
        }
        Json(body).into_response()
    }
}

async fn chat_completions(
    State(st): State<Arc<ServeState>>,
    headers: HeaderMap,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let rid = request_id(&st, &headers);
    if st.draining.load(Ordering::SeqCst) {
        return with_rid(draining_response(&rid), &rid);
    }
    let req = match payload {
        Ok(Json(r)) => r,
        Err(rej) => {
            st.metrics.failed.fetch_add(1, Ordering::Relaxed);
            return with_rid(mw::json_rejection(rej), &rid);
        }
    };
    if req.model != st.model_id {
        return with_rid(
            openai_error(
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                format!("The model `{}` does not exist", req.model),
            ),
            &rid,
        );
    }
    if req.messages.is_empty() {
        return with_rid(
            openai_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "'messages' must not be empty",
            ),
            &rid,
        );
    }
    // 工具调用（tools）：`tools` 必须是数组（vLLM 同约束，坏形状 400 而非
    // 拿去做 prompt）；`tool_choice` 只认 none/auto/required/函数对象。
    // `none` ⇒ 不把工具描述注入 prompt（模型看不到就不会调），也不做解析。
    if let Some(t) = &req.tools {
        if !t.is_array() {
            return with_rid(
                openai_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "'tools' must be an array of function definitions",
                ),
                &rid,
            );
        }
    }
    let tool_choice_none = match &req.tool_choice {
        None => false,
        Some(Value::String(s)) if s == "none" => true,
        Some(Value::String(s)) if s == "auto" || s == "required" => false,
        Some(Value::Object(_)) => false,
        Some(_) => {
            return with_rid(
                openai_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "'tool_choice' must be \"none\", \"auto\", \"required\" or a function object",
                ),
                &rid,
            )
        }
    };

    // chat 模板（W5 真身）：tokenizer 有模板即渲染（模型侧 raise → 400）；
    // 无模板（mock/未接）回退占位包装——既有验收即 parity。
    // 模板顶层 kwargs：`tools`（函数调用，vLLM 同语义）+ `chat_template_kwargs`
    // 直通（`enable_thinking` 等）；两者都只在模板侧消费，serve 不做解释
    let mut extra: Vec<(String, Value)> = Vec::new();
    if let Some(t) = req.tools.as_ref().filter(|_| !tool_choice_none) {
        extra.push(("tools".into(), t.clone()));
    }
    if let Some(k) = &req.chat_template_kwargs {
        match k.as_object() {
            Some(obj) => {
                for (name, v) in obj {
                    extra.push((name.clone(), v.clone()));
                }
            }
            None => {
                return with_rid(
                    openai_error(
                        StatusCode::BAD_REQUEST,
                        "invalid_request_error",
                        "'chat_template_kwargs' must be an object",
                    ),
                    &rid,
                )
            }
        }
    }
    let prompt_ids = match st.tokenizer.render_chat_template_with(&req.messages, &extra) {
        Some(Ok(text)) => st.tokenizer.encode(&text),
        Some(Err(msg)) => {
            return with_rid(
                openai_error(StatusCode::BAD_REQUEST, "invalid_request_error", msg),
                &rid,
            )
        }
        None => {
            let prompt: String = req
                .messages
                .iter()
                .map(|m| format!("{}: {}\n", m.role, m.content.as_deref().unwrap_or("")))
                .collect();
            st.tokenizer.encode(&prompt)
        }
    };
    let (prompt_ids, prompt_tokens, p) = match prepare(
        &st,
        &rid,
        prompt_ids,
        req.max_tokens,
        req.temperature,
        req.top_p,
        &req.stop,
    ) {
        Ok(v) => v,
        Err(r) => return with_rid(r, &rid),
    };
    let tool_schema =
        req.tools.clone().filter(|_| !tool_choice_none && parser_enabled());
    // 输出侧思维链：服务端开关（CLI --reasoning-parser 探测模板）+ 本次请求没关思考。
    // `enable_thinking: false` 时提示里已是闭合的 `<think>\n\n</think>`，输出不再有
    // `</think>`——此时若仍切分，全部正文会被误判成推理。
    let thinking_off = req
        .chat_template_kwargs
        .as_ref()
        .and_then(|k| k.get("enable_thinking"))
        .map(|v| v == &Value::Bool(false))
        .unwrap_or(false);
    let reasoning = st.reasoning && !thinking_off;
    let resp = dispatch(
        &st,
        Wire::Chat,
        rid.clone(),
        prompt_ids,
        prompt_tokens,
        p,
        req.stream.unwrap_or(false),
        tool_schema,
        reasoning,
    )
    .await;
    with_rid(resp, &rid)
}

async fn completions(
    State(st): State<Arc<ServeState>>,
    headers: HeaderMap,
    payload: Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    let rid = request_id(&st, &headers);
    if st.draining.load(Ordering::SeqCst) {
        return with_rid(draining_response(&rid), &rid);
    }
    let req = match payload {
        Ok(Json(r)) => r,
        Err(rej) => {
            st.metrics.failed.fetch_add(1, Ordering::Relaxed);
            return with_rid(mw::json_rejection(rej), &rid);
        }
    };
    if req.model != st.model_id {
        return with_rid(
            openai_error(
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                format!("The model `{}` does not exist", req.model),
            ),
            &rid,
        );
    }
    // prompt 数组**不再静默丢**：M0 一次只服务一个 prompt，数组必须明确拒绝
    // （旧实现取第一个 = 客户端拿到"只有一半"的答案还不知道）
    let prompt_text = match &req.prompt {
        Value::String(s) => s.clone(),
        Value::Array(_) => {
            return with_rid(
                openai_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "'prompt' arrays are not supported; send one prompt per request",
                ),
                &rid,
            )
        }
        _ => {
            return with_rid(
                openai_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "'prompt' must be a string",
                ),
                &rid,
            )
        }
    };
    let prompt_ids = st.tokenizer.encode(&prompt_text);
    let (prompt_ids, prompt_tokens, p) = match prepare(
        &st,
        &rid,
        prompt_ids,
        req.max_tokens,
        req.temperature,
        req.top_p,
        &req.stop,
    ) {
        Ok(v) => v,
        Err(r) => return with_rid(r, &rid),
    };
    let resp = dispatch(
        &st,
        Wire::Text,
        rid.clone(),
        prompt_ids,
        prompt_tokens,
        p,
        req.stream.unwrap_or(false),
        None,
        false,
    )
    .await;
    with_rid(resp, &rid)
}
