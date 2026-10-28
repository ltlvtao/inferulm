//! A 组 · 流式契约：SSE 形状、UTF-8 边界（增量 detok 的核心验收点）。
mod common;

use common::*;
use serde_json::Value;
use wein_engine::MockScript;

#[tokio::test]
async fn stream_shape_role_first_done_last() {
    let script = MockScript { pieces: vec![0, 1, 2], ..Default::default() };
    let ts = start(script, default_tok()).await;
    let frames = post_sse(&format!("{}/v1/chat/completions", ts.base), &chat_body(3)).await;

    let last = frames.last().unwrap();
    assert_eq!(last, "[DONE]", "SSE 必须以 [DONE] 收尾");

    let first: Value = serde_json::from_str(&frames[0]).unwrap();
    assert_eq!(first["object"], "chat.completion.chunk");
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant", "首帧是 role delta");
    assert_eq!(first["choices"][0]["finish_reason"], Value::Null);

    // 倒数第二帧 = finish 帧：空 delta + finish_reason
    let finish: Value = serde_json::from_str(&frames[frames.len() - 2]).unwrap();
    assert_eq!(finish["choices"][0]["finish_reason"], "length");
    assert!(finish["choices"][0]["delta"].as_object().unwrap().is_empty());

    assert_eq!(content_of(&frames), "Hello world");
}

#[tokio::test]
async fn every_frame_is_valid_json_with_seq_ids() {
    let ts = start(MockScript::default(), default_tok()).await;
    let frames = post_sse(&format!("{}/v1/chat/completions", ts.base), &chat_body(4)).await;
    let mut ids = Vec::new();
    for f in &frames {
        if f == "[DONE]" {
            continue;
        }
        let v: Value = serde_json::from_str(f).expect("每个 data 帧都必须是合法 JSON");
        ids.push(v["id"].as_str().unwrap().to_string());
    }
    assert!(ids.len() >= 3);
    assert!(ids.windows(2).all(|w| w[0] == w[1]), "同一条流内 id 必须一致");
}

/// 跨 token 的多字节字符（"你"=e4bda0、"好"=e5a5bd）被刻意切在字节中间：
/// 任何一帧都不允许出现半个字符或替换符，拼接结果必须是完整"你好"。
#[tokio::test]
async fn utf8_multibyte_char_never_split_across_frames() {
    let tok = MockTokenizer::from_bytes(vec![
        vec![0xe4, 0xbd],       // "你" 的前两字节
        vec![0xa0, 0xe5, 0xa5], // "你" 尾字节 + "好" 前两字节
        vec![0xbd],             // "好" 尾字节
    ]);
    let script = MockScript { pieces: vec![0, 1, 2], ..Default::default() };
    let ts = start(script, tok).await;
    let frames = post_sse(&format!("{}/v1/chat/completions", ts.base), &chat_body(3)).await;

    for f in &frames {
        assert!(!f.contains('\u{fffd}'), "不允许出现替换符: {f}");
        if f != "[DONE]" {
            let v: Value = serde_json::from_str(f).unwrap();
            if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
                assert!(!c.is_empty() || v["choices"][0]["delta"].get("role").is_some(), "不允许空 content 帧");
            }
        }
    }
    assert_eq!(content_of(&frames), "你好");
}

#[tokio::test]
async fn legacy_completions_stream_ends_with_done() {
    let ts = start(MockScript::default(), default_tok()).await;
    let frames = post_sse(
        &format!("{}/v1/completions", ts.base),
        &serde_json::json!({"model": "mock-model", "prompt": "x", "max_tokens": 2, "stream": true}),
    )
    .await;
    assert_eq!(frames.last().unwrap(), "[DONE]");
    let mut text = String::new();
    for f in &frames {
        if f == "[DONE]" {
            continue;
        }
        let v: Value = serde_json::from_str(f).unwrap();
        if let Some(t) = v["choices"][0]["text"].as_str() {
            text.push_str(t);
        }
    }
    assert_eq!(text, "Hello wo");
}
