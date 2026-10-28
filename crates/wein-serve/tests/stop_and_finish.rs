//! A 组 · 停止语义：自然停止、stop 参数截断 + 引擎取消、finish_reason 正确性。
mod common;

use common::*;
use serde_json::{json, Value};
use wein_engine::MockScript;
use std::time::Duration;

#[tokio::test]
async fn natural_stop_yields_finish_reason_stop() {
    // 引擎在第 2 个 token 前自然停止（模拟 EOS）
    let script = MockScript { pieces: vec![0, 1, 2, 3], stop_at: Some(2), ..Default::default() };
    let ts = start(script, default_tok()).await;
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&chat_body(10))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["choices"][0]["message"]["content"], "Hello wo");
    assert_eq!(body["usage"]["completion_tokens"], 2);
}

#[tokio::test]
async fn stop_param_truncates_content_and_cancels_engine() {
    let tok = MockTokenizer::from_texts(&["Alpha", " beta", " gamma", " done"]);
    let script = MockScript { pieces: vec![0, 1, 2, 3], ..Default::default() };
    let ts = start(script, tok).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 10,
            "stop": ["gamma"],
        }))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(content.starts_with("Alpha beta"), "content = {content:?}");
    assert!(!content.contains("gamma"), "stop 串必须被截掉");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");

    // serve 层发现 stop 后必须取消引擎（否则引擎继续白算 7 个 token）
    eventually(Duration::from_secs(2), || ts.mock.cancelled_count() >= 1).await;
}

#[tokio::test]
async fn stop_as_string_form_also_works() {
    let tok = MockTokenizer::from_texts(&["aa", "bb", "cc"]);
    let script = MockScript { pieces: vec![0, 1, 2], ..Default::default() };
    let ts = start(script, tok).await;
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 3,
            "stop": "cc",
        }))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "aabb");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn stop_prefix_held_back_until_disambiguated() {
    // stop 前缀跨 delta：pieces 逐字符吐出，stop="end"
    // 到位前的 "e"（"The e..."）不能提前吐出后又收回
    let tok = MockTokenizer::from_texts(&["x ", "e", "nd", "stuff"]);
    let script = MockScript { pieces: vec![0, 1, 2, 3], ..Default::default() };
    let ts = start(script, tok).await;
    let frames = post_sse(
        &format!("{}/v1/chat/completions", ts.base),
        &json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 4,
            "stop": ["end"],
            "stream": true,
        }),
    )
    .await;
    assert_eq!(content_of(&frames), "x ");
    let finish: Value = serde_json::from_str(&frames[frames.len() - 2]).unwrap();
    assert_eq!(finish["choices"][0]["finish_reason"], "stop");
}
