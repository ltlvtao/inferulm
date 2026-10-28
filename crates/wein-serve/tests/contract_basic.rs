//! A 组 · 基础契约：models / 非流式生成 / 错误码边界。
mod common;

use common::*;
use serde_json::{json, Value};
use wein_engine::MockScript;

#[tokio::test]
async fn models_list_contains_served_model() {
    let ts = start(MockScript::default(), default_tok()).await;
    let resp = reqwest::Client::new()
        .get(format!("{}/v1/models", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "mock-model");
    assert_eq!(body["data"][0]["object"], "model");
}

#[tokio::test]
async fn chat_completion_non_stream() {
    let script = MockScript { pieces: vec![0, 1, 2], ..Default::default() };
    let ts = start(script, default_tok()).await;
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&chat_body(3))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(body["choices"][0]["message"]["content"], "Hello world");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(body["usage"]["completion_tokens"], 3);
    assert!(body["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert_eq!(
        body["usage"]["total_tokens"].as_u64().unwrap(),
        body["usage"]["prompt_tokens"].as_u64().unwrap() + 3
    );
}

#[tokio::test]
async fn completions_legacy_endpoint() {
    let script = MockScript { pieces: vec![0, 1], ..Default::default() };
    let ts = start(script, default_tok()).await;
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/completions", ts.base))
        .json(&json!({"model": "mock-model", "prompt": "once upon", "max_tokens": 2}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["choices"][0]["text"], "Hello wo");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
}

#[tokio::test]
async fn unknown_model_returns_404_with_openai_error_body() {
    let ts = start(MockScript::default(), default_tok()).await;
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&json!({"model": "nope", "messages": [{"role": "user", "content": "x"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(body["error"]["message"].as_str().unwrap().contains("nope"));
}

#[tokio::test]
async fn out_of_range_temperature_returns_400() {
    let ts = start(MockScript::default(), default_tok()).await;
    let mut body = chat_body(2);
    body["temperature"] = json!(3.0);
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let b: Value = resp.json().await.unwrap();
    assert_eq!(b["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn bad_stop_type_returns_400() {
    let ts = start(MockScript::default(), default_tok()).await;
    let mut body = chat_body(2);
    body["stop"] = json!(42);
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn max_tokens_zero_returns_empty_immediately() {
    let script = MockScript { per_token_delay: std::time::Duration::from_millis(50), ..Default::default() };
    let ts = start(script, default_tok()).await;
    let t0 = tokio::time::Instant::now();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&chat_body(0))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(body["usage"]["completion_tokens"], 0);
    assert!(t0.elapsed() < std::time::Duration::from_millis(100));
}

#[tokio::test]
async fn unknown_route_returns_json_404() {
    let ts = start(MockScript::default(), default_tok()).await;
    let resp = reqwest::Client::new()
        .get(format!("{}/v1/nope", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body: Value = resp.json().await.unwrap();
    assert!(body["error"].is_object());
}
