//! B 组 · 断连与 abort 语义：客户端断开必须转化为引擎取消，且服务不因此劣化。
mod common;

use common::*;
use futures::StreamExt;
use serde_json::Value;
use wein_engine::MockScript;
use std::time::Duration;

async fn open_stream_and_drop_after_first_frame(url: &str, body: &Value) {
    let client = reqwest::Client::new();
    let resp = client.post(url).json(body).send().await.unwrap();
    let mut stream = resp.bytes_stream();
    // 读到第一帧（证明已进入生成）后立刻断开（resp 已 move 进 stream）
    let _ = stream.next().await;
    drop(stream);
}

#[tokio::test]
async fn client_disconnect_cancels_engine_request() {
    let script = MockScript {
        per_token_delay: Duration::from_millis(5),
        pieces: (0..8).collect(),
        ..Default::default()
    };
    let ts = start(script, default_tok()).await;
    open_stream_and_drop_after_first_frame(
        &format!("{}/v1/chat/completions", ts.base),
        &{
            let mut b = chat_body(200); // 长生成，确保断开时仍在跑
            b["stream"] = serde_json::json!(true);
            b
        },
    )
    .await;

    eventually(Duration::from_secs(2), || ts.mock.cancelled_count() >= 1).await;

    // 断连后服务必须仍然健康
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&chat_body(2))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "Hello wo");
}

#[tokio::test]
async fn abort_storm_then_still_healthy() {
    let script = MockScript {
        per_token_delay: Duration::from_millis(5),
        pieces: (0..8).collect(),
        ..Default::default()
    };
    let ts = start(script, default_tok()).await;
    let url = format!("{}/v1/chat/completions", ts.base);
    let body = {
        let mut b = chat_body(300);
        b["stream"] = serde_json::json!(true);
        b
    };

    let mut handles = Vec::new();
    for _ in 0..30 {
        let url = url.clone();
        let body = body.clone();
        handles.push(tokio::spawn(async move {
            open_stream_and_drop_after_first_frame(&url, &body).await;
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    eventually(Duration::from_secs(3), || ts.mock.cancelled_count() >= 30).await;

    // 风暴之后：正常请求仍完整
    let resp = reqwest::Client::new().post(&url).json(&chat_body(3)).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "Hello world");
}
