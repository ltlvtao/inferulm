//! 验收测试公共设施：起真实 HTTP 服务（临时端口）+ MockEngine，reqwest 做客户端。
//! 被测的就是被部署的（spawn_server 与 CLI 同入口）。
#![allow(dead_code)] // 每个测试二进制只用其中一部分

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use serde_json::Value;
use wein_engine::{CoreConfig, EngineCore, EngineStats, MockEngine, MockScript, MockState};
use wein_serve::{spawn_server, ServeState};

pub use wein_serve::MockTokenizer;

pub struct TestServer {
    pub base: String,
    pub mock: Arc<MockState>,
}

pub async fn start(script: MockScript, tok: MockTokenizer) -> TestServer {
    let (engine, mock) = MockEngine::spawn(script);
    let state = ServeState {
        model_id: "mock-model".into(),
        tokenizer: Arc::new(tok),
        engine,
    };
    let addr = spawn_server(state, "127.0.0.1:0".parse().unwrap()).await.unwrap();
    TestServer { base: format!("http://{addr}"), mock }
}

/// 默认词表：id 0..7 → 8 个片段（MockScript 默认 pieces 也是 0..7 循环）
pub fn default_tok() -> MockTokenizer {
    MockTokenizer::from_texts(&["Hello", " wo", "rld", "!", " The", " end", ".", " again"])
}

pub fn chat_body(max_tokens: u32) -> Value {
    serde_json::json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": max_tokens,
    })
}

/// 发起流式请求（自动置 stream=true——这是本函数的语义，调用方无须记得设），
/// 读回全部 SSE data 帧（按序）。帧内容必须整体是合法 UTF-8——
/// 这本身就是一条断言：多字节字符绝不能被切成半个吐出去。
pub async fn post_sse(url: &str, body: &Value) -> Vec<String> {
    let mut body = body.clone();
    body["stream"] = serde_json::json!(true);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .unwrap();
    let resp = client.post(url).json(&body).send().await.expect("send failed");
    assert_eq!(resp.status(), 200, "stream request must be 200");
    let ct = resp.headers()["content-type"].to_str().unwrap().to_string();
    assert!(ct.starts_with("text/event-stream"), "content-type = {ct}");
    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    let mut frames = Vec::new();
    while let Some(chunk) = stream.next().await {
        buf.extend_from_slice(&chunk.expect("chunk failed"));
        while let Some(pos) = find_sub(&buf, b"\n\n") {
            let frame: Vec<u8> = buf.drain(..pos + 2).collect();
            for line in frame.split(|b| *b == b'\n') {
                if let Some(data) = line.strip_prefix(b"data: ") {
                    frames.push(String::from_utf8(data.to_vec()).expect("SSE frame must be valid UTF-8"));
                }
            }
        }
    }
    frames
}

pub fn find_sub(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// 轮询断言：条件须在 timeout 内成立。
pub async fn eventually(timeout: Duration, f: impl Fn() -> bool) {
    let deadline = tokio::time::Instant::now() + timeout;
    while tokio::time::Instant::now() < deadline {
        if f() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("condition not met within {timeout:?}");
}

/// 从 SSE 帧序列提取正文文本（content delta 拼接）。
pub fn content_of(frames: &[String]) -> String {
    let mut out = String::new();
    for f in frames {
        if f == "[DONE]" {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<Value>(f) {
            if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
                out.push_str(c);
            }
        }
    }
    out
}

pub struct TestCore {
    pub base: String,
    pub stats: Arc<EngineStats>,
}

/// 与 start() 同构，但引擎换成 EngineCore（wein-sched 调度器 + 脚本执行器）：
/// serve 层验收对引擎实现完全无感知——协议锚的兑现。
pub async fn start_core(cfg: CoreConfig, tok: MockTokenizer) -> TestCore {
    let (engine, stats) = EngineCore::spawn(cfg);
    let state = ServeState {
        model_id: "mock-model".into(),
        tokenizer: Arc::new(tok),
        engine,
    };
    let addr = spawn_server(state, "127.0.0.1:0".parse().unwrap()).await.unwrap();
    TestCore { base: format!("http://{addr}"), stats }
}

/// 与 MockScript::default() 输出对齐的 core 配置（pieces 0..7 循环，1ms 步进）
pub fn default_core() -> CoreConfig {
    CoreConfig::scripted(vec![0, 1, 2, 3, 4, 5, 6, 7], Duration::from_millis(1))
}
