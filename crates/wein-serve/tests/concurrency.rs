//! B 组 · 并发正确性：多流不串扰、慢消费者不阻塞他人、风暴后状态干净。
//! （注：mock 引擎对所有请求产出相同内容，"串扰"靠逐流精确比对 + 完整性判定覆盖；
//!   真实引擎接入后补 per-request 差异化输出的串扰用例。）
mod common;

use common::*;
use futures::StreamExt;
use serde_json::Value;
use wein_engine::MockScript;
use std::time::{Duration, Instant};

fn expected_content(n_tokens: usize) -> String {
    const TEXTS: [&str; 8] = ["Hello", " wo", "rld", "!", " The", " end", ".", " again"];
    (0..n_tokens).map(|i| TEXTS[i % 8]).collect()
}

#[tokio::test]
async fn concurrent_streams_all_complete_with_exact_content() {
    let script = MockScript {
        per_token_delay: Duration::from_millis(1),
        pieces: (0..8).collect(),
        ..Default::default()
    };
    let ts = start(script, default_tok()).await;
    let url = format!("{}/v1/chat/completions", ts.base);

    const N: usize = 100;
    const TOKENS: usize = 10;
    let t0 = Instant::now();
    let mut handles = Vec::new();
    for i in 0..N {
        let url = url.clone();
        handles.push(tokio::spawn(async move {
            let mut body = chat_body(TOKENS as u32);
            body["stream"] = serde_json::json!(true);
            body["messages"] = serde_json::json!([{"role": "user", "content": format!("prompt-{i}")}]);
            post_sse(&url, &body).await
        }));
    }
    for (i, h) in handles.into_iter().enumerate() {
        let frames = tokio::time::timeout(Duration::from_secs(30), h).await.expect("no timeout").unwrap();
        assert_eq!(frames.last().unwrap(), "[DONE]", "流 {i} 必须正常收尾");
        let content = content_of(&frames);
        assert_eq!(content, expected_content(TOKENS), "流 {i} 内容精确匹配");
        let finish: Value = serde_json::from_str(&frames[frames.len() - 2]).unwrap();
        assert_eq!(finish["choices"][0]["finish_reason"], "length", "流 {i}");
    }
    assert!(t0.elapsed() < Duration::from_secs(30));
    assert_eq!(ts.mock.accepted_count(), N, "引擎必须恰好接受 {N} 个请求");
}

#[tokio::test]
async fn slow_consumer_does_not_block_fast_ones() {
    let script = MockScript {
        per_token_delay: Duration::from_millis(2),
        pieces: (0..8).collect(),
        ..Default::default()
    };
    let ts = start(script, default_tok()).await;
    let url = format!("{}/v1/chat/completions", ts.base);

    // 慢消费者：每帧之间停 80ms
    let slow_url = url.clone();
    let slow = tokio::spawn(async move {
        let client = reqwest::Client::new();
        let resp = client
            .post(&slow_url)
            .json(&{
                let mut b = chat_body(8);
                b["stream"] = serde_json::json!(true);
                b
            })
            .send()
            .await
            .unwrap();
        let mut stream = resp.bytes_stream();
        let mut got = 0;
        while let Some(c) = stream.next().await {
            c.unwrap();
            got += 1;
            tokio::time::sleep(Duration::from_millis(80)).await;
        }
        got
    });

    // 快消费者们必须在慢消费者读完全程之前就完成
    tokio::time::sleep(Duration::from_millis(50)).await; // 确保慢的先建立
    let t0 = Instant::now();
    let mut fast = Vec::new();
    for _ in 0..20 {
        let url = url.clone();
        fast.push(tokio::spawn(async move {
            let mut b = chat_body(8);
            b["stream"] = serde_json::json!(true);
            post_sse(&url, &b).await
        }));
    }
    for h in fast {
        tokio::time::timeout(Duration::from_secs(5), h).await.expect("fast consumers must not be blocked").unwrap();
    }
    assert!(t0.elapsed() < Duration::from_secs(4), "快消费者组耗时 {:?}", t0.elapsed());

    let slow_frames = tokio::time::timeout(Duration::from_secs(15), slow).await.unwrap().unwrap();
    assert!(slow_frames > 0);
}

#[tokio::test]
async fn server_state_clean_after_concurrent_load() {
    let ts = start(MockScript::default(), default_tok()).await;
    let url = format!("{}/v1/chat/completions", ts.base);
    let mut handles = Vec::new();
    for _ in 0..50 {
        let url = url.clone();
        handles.push(tokio::spawn(async move {
            let mut b = chat_body(4);
            b["stream"] = serde_json::json!(true);
            post_sse(&url, &b).await
        }));
    }
    for h in handles {
        tokio::time::timeout(Duration::from_secs(30), h).await.unwrap().unwrap();
    }
    // 一轮并发后：单个请求依旧精确
    let resp = reqwest::Client::new().post(&url).json(&chat_body(3)).send().await.unwrap();
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "Hello world");
    assert_eq!(ts.mock.cancelled_count(), 0, "全部自然结束，不应有取消");
}
