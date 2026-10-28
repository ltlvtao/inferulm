//! D 组 · 协议锚兑现：serve 验收换真实 EngineCore（wein-sched 调度器 + 脚本执行器）原样复跑。
//! 断言与 A/B 组同源——换引擎后行为零漂移，这就是 TDD 外→内的验收闭环：
//! serve 层的全部语义只依赖边界协议，与引擎实现无关。
mod common;

use common::*;
use futures::StreamExt;
use serde_json::{json, Value};
use std::time::{Duration, Instant};
use wein_engine::CoreConfig;

fn core(pieces: Vec<u32>, step_delay_ms: u64) -> CoreConfig {
    CoreConfig::scripted(pieces, Duration::from_millis(step_delay_ms))
}

/// p1 · 非流式：内容精确、usage 正确、finish=length（对照 chat_completion_non_stream）
#[tokio::test]
async fn p1_chat_non_stream_content_usage_and_finish() {
    let ts = start_core(core(vec![0, 1, 2], 1), default_tok()).await;
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&chat_body(3))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "Hello world");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert_eq!(body["usage"]["completion_tokens"], 3);
}

/// p2 · 流式形状：role 首、[DONE] 尾、finish 帧居次尾（对照 stream_shape_role_first_done_last）
#[tokio::test]
async fn p2_stream_shape_role_first_done_last() {
    let ts = start_core(core(vec![0, 1, 2], 1), default_tok()).await;
    let frames = post_sse(&format!("{}/v1/chat/completions", ts.base), &chat_body(3)).await;

    assert_eq!(frames.last().unwrap(), "[DONE]");
    let first: Value = serde_json::from_str(&frames[0]).unwrap();
    assert_eq!(first["object"], "chat.completion.chunk");
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
    let finish: Value = serde_json::from_str(&frames[frames.len() - 2]).unwrap();
    assert_eq!(finish["choices"][0]["finish_reason"], "length");
    assert_eq!(content_of(&frames), "Hello world");
}

/// p3 · UTF-8 多字节字符跨 token 切割：任何一帧不得出现半个字符（对照 utf8_multibyte_char_never_split_across_frames）
#[tokio::test]
async fn p3_utf8_multibyte_never_split_across_frames() {
    let tok = MockTokenizer::from_bytes(vec![
        vec![0xe4, 0xbd],       // "你" 的前两字节
        vec![0xa0, 0xe5, 0xa5], // "你" 尾字节 + "好" 前两字节
        vec![0xbd],             // "好" 尾字节
    ]);
    let ts = start_core(core(vec![0, 1, 2], 1), tok).await;
    let frames = post_sse(&format!("{}/v1/chat/completions", ts.base), &chat_body(3)).await;
    for f in &frames {
        assert!(!f.contains('\u{fffd}'), "不允许出现替换符: {f}");
    }
    assert_eq!(content_of(&frames), "你好");
}

/// p4 · stop 参数：内容截断 + finish=stop + 引擎被取消（对照 stop_param_truncates_content_and_cancels_engine）
#[tokio::test]
async fn p4_stop_param_truncates_and_cancels_engine() {
    let tok = MockTokenizer::from_texts(&["Alpha", " beta", " gamma", " done"]);
    let ts = start_core(core(vec![0, 1, 2, 3], 1), tok).await;
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

    eventually(Duration::from_secs(2), || ts.stats.cancelled_count() >= 1).await;
}

/// p5 · 断连：客户端断开 → 引擎取消，服务仍健康（对照 client_disconnect_cancels_engine_request）
#[tokio::test]
async fn p5_disconnect_cancels_and_server_stays_healthy() {
    let ts = start_core(default_core(), default_tok()).await;
    let mut body = chat_body(200); // 长生成，确保断开时仍在跑
    body["stream"] = json!(true);
    {
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/v1/chat/completions", ts.base))
            .json(&body)
            .send()
            .await
            .unwrap();
        let mut stream = resp.bytes_stream();
        let _ = stream.next().await; // 读到首帧后立刻断开
        drop(stream);
    }

    eventually(Duration::from_secs(2), || ts.stats.cancelled_count() >= 1).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&chat_body(2))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "Hello wo");
}

fn expected_content(n_tokens: usize) -> String {
    const TEXTS: [&str; 8] = ["Hello", " wo", "rld", "!", " The", " end", ".", " again"];
    (0..n_tokens).map(|i| TEXTS[i % 8]).collect()
}

/// p6 · 100 并发逐流精确内容 + 引擎恰好接受 100（对照 concurrent_streams_all_complete_with_exact_content）
#[tokio::test]
async fn p6_concurrent_streams_exact_content_and_accepted() {
    let ts = start_core(default_core(), default_tok()).await;
    let url = format!("{}/v1/chat/completions", ts.base);

    const N: usize = 100;
    const TOKENS: usize = 10;
    let t0 = Instant::now();
    let mut handles = Vec::new();
    for i in 0..N {
        let url = url.clone();
        handles.push(tokio::spawn(async move {
            let mut body = chat_body(TOKENS as u32);
            body["stream"] = json!(true);
            body["messages"] = json!([{"role": "user", "content": format!("prompt-{i}")}]);
            post_sse(&url, &body).await
        }));
    }
    for (i, h) in handles.into_iter().enumerate() {
        let frames =
            tokio::time::timeout(Duration::from_secs(30), h).await.expect("no timeout").unwrap();
        assert_eq!(frames.last().unwrap(), "[DONE]", "流 {i} 必须正常收尾");
        assert_eq!(content_of(&frames), expected_content(TOKENS), "流 {i} 内容精确匹配");
    }
    assert!(t0.elapsed() < Duration::from_secs(30));
    assert_eq!(ts.stats.accepted_count(), N, "引擎必须恰好接受 {N} 个请求");
}

/// p7 · max_tokens=0：立即空响应（对照 max_tokens_zero_returns_empty_immediately）
#[tokio::test]
async fn p7_max_tokens_zero_empty_immediate() {
    let ts = start_core(core(vec![0], 50), default_tok()).await;
    let t0 = Instant::now();
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
    assert!(t0.elapsed() < Duration::from_millis(100));
}

/// p8 · 取消风暴：30 连断后引擎记账干净、服务健康、无残留取消（DeferredRelease 压力面）
#[tokio::test]
async fn p8_abort_storm_then_healthy_and_clean() {
    let ts = start_core(default_core(), default_tok()).await;
    let url = format!("{}/v1/chat/completions", ts.base);
    let body = {
        let mut b = chat_body(300);
        b["stream"] = json!(true);
        b
    };

    let mut handles = Vec::new();
    for _ in 0..30 {
        let url = url.clone();
        let body = body.clone();
        handles.push(tokio::spawn(async move {
            let client = reqwest::Client::new();
            let resp = client.post(&url).json(&body).send().await.unwrap();
            let mut stream = resp.bytes_stream();
            let _ = stream.next().await;
            drop(stream);
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
    eventually(Duration::from_secs(3), || ts.stats.cancelled_count() >= 30).await;

    // 风暴之后：正常请求完整且精确
    let resp = reqwest::Client::new().post(&url).json(&chat_body(3)).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "Hello world");
}
