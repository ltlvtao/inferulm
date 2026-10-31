//! 错误语义回归：**每条失败路径都必须是 OpenAI 错误体**（任务书 §2）。
//!
//! 旧实现只有 400/404 是 JSON；serde 失败走 axum 原生纯文本（415/422），
//! handler panic 直接断连接。这里逐个钉死状态码 + body 形状 + request id。
mod common;

use std::time::Duration;

use serde_json::{json, Value};

use common::{client, default_tok, start, MockScript};

async fn post_raw(base: &str, path: &str, body: &str, ct: Option<&str>) -> reqwest::Response {
    let mut req = client()
        .post(format!("{base}{path}"))
        .timeout(Duration::from_secs(10))
        .body(body.to_string());
    if let Some(ct) = ct {
        req = req.header("content-type", ct);
    }
    req.send().await.expect("send failed")
}

async fn assert_openai_error(resp: reqwest::Response, want: u16, hint: &str) -> Value {
    assert_eq!(resp.status(), want, "{hint}: 状态码");
    assert!(
        resp.headers().contains_key("x-request-id"),
        "{hint}: 错误响应也要回写 X-Request-Id"
    );
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(ct.starts_with("application/json"), "{hint}: content-type = {ct}");
    let v: Value = resp.json().await.expect("错误体必须是 JSON");
    assert!(v["error"].is_object(), "{hint}: body = {v}");
    assert!(v["error"]["message"].is_string(), "{hint}: error.message 必须存在");
    v
}

/// e1 · 非法 JSON → **400**（axum 原生行为也是 400，但 body 换成 OpenAI 形状）
#[tokio::test]
async fn e1_invalid_json_returns_400_json_body() {
    let ts = start(MockScript::default(), default_tok()).await;
    let resp =
        post_raw(&ts.base, "/v1/chat/completions", "{\"model\": \"mock-model\", ", Some("application/json"))
            .await;
    let v = assert_openai_error(resp, 400, "非法 JSON").await;
    assert_eq!(v["error"]["type"], "invalid_request_error");
}

/// e2 · 缺字段（messages 不在）→ **422**（保留原状态码，body 换 JSON）
#[tokio::test]
async fn e2_missing_field_returns_422_json_body() {
    let ts = start(MockScript::default(), default_tok()).await;
    let resp = post_raw(
        &ts.base,
        "/v1/chat/completions",
        "{\"model\": \"mock-model\", \"max_tokens\": 2}",
        Some("application/json"),
    )
    .await;
    let v = assert_openai_error(resp, 422, "缺 messages").await;
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert!(v["error"]["message"].as_str().unwrap().contains("messages"));
}

/// e3 · 缺 content-type → **415**（保留状态码，body 换 JSON）
#[tokio::test]
async fn e3_missing_content_type_returns_415_json_body() {
    let ts = start(MockScript::default(), default_tok()).await;
    let body = json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2,
    })
    .to_string();
    let resp = post_raw(&ts.base, "/v1/chat/completions", &body, None).await;
    assert_openai_error(resp, 415, "缺 content-type").await;
}

/// e4 · model 不存在 → **404**
#[tokio::test]
async fn e4_unknown_model_returns_404() {
    let ts = start(MockScript::default(), default_tok()).await;
    let body = json!({
        "model": "nope",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2,
    });
    let resp = client()
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap();
    let v = assert_openai_error(resp, 404, "错 model").await;
    assert!(v["error"]["message"].as_str().unwrap().contains("nope"));
}

/// e5 · `/v1/completions` 的 `prompt` 数组 → **400**（明确不支持，不再静默取第一个）
#[tokio::test]
async fn e5_prompt_array_returns_400() {
    let ts = start(MockScript::default(), default_tok()).await;
    let body = json!({
        "model": "mock-model",
        "prompt": ["once upon a time", "and then"],
        "max_tokens": 2,
    });
    let resp = client()
        .post(format!("{}/v1/completions", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap();
    let v = assert_openai_error(resp, 400, "prompt 数组").await;
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(msg.contains("array"), "message = {msg:?}");
    // 单元素数组同样拒绝（一致的语义，不做"只有一个就放过"的特殊分支）
    let body = json!({"model": "mock-model", "prompt": ["x"], "max_tokens": 2});
    let resp = client()
        .post(format!("{}/v1/completions", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

/// e6 · 客户端给的 X-Request-Id 回写到响应头与 body id（错误响应也回写）
#[tokio::test]
async fn e6_client_request_id_is_echoed() {
    let ts = start(MockScript::default(), default_tok()).await;
    let body = json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2,
    });
    let resp = client()
        .post(format!("{}/v1/chat/completions", ts.base))
        .header("x-request-id", "trace-abc-123")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.headers()["x-request-id"], "trace-abc-123");
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["id"], "chatcmpl-trace-abc-123");

    // 没给就生成（非空）
    let resp = client()
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap();
    let rid = resp.headers()["x-request-id"].to_str().unwrap().to_string();
    assert!(!rid.is_empty());
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["id"], format!("chatcmpl-{rid}"));
}

/// e7 · 流式 completions 的 object 是 `text_completion.chunk`（旧实现错写成
/// `text_completion`——客户端按 OpenAI SDK 解析会拿不到 chunk 语义）
#[tokio::test]
async fn e7_completions_stream_object_is_chunk() {
    let ts = start(MockScript::default(), default_tok()).await;
    let frames = common::post_sse(
        &format!("{}/v1/completions", ts.base),
        &json!({"model": "mock-model", "prompt": "once upon", "max_tokens": 2}),
    )
    .await;
    let mut seen = 0;
    for f in &frames {
        if f == "[DONE]" {
            continue;
        }
        let v: Value = serde_json::from_str(f).unwrap();
        assert_eq!(v["object"], "text_completion.chunk", "frame = {f}");
        seen += 1;
    }
    assert!(seen > 0);
}
