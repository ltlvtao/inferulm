//! 输出侧思维链解析（`</think>` 切分）serve 契约——rs 系列。
//!
//! 语义与 vLLM `--reasoning-parser deepseek_r1/qwen3` 同：第一次出现的 `</think>`
//! 之前的文本进 `reasoning_content`（流式是 `delta.reasoning_content`），标记本身
//! 外发。**关闭时行为与旧版逐字一致**（全文进 content）。
mod common;

use std::sync::Arc;

use common::{client, post_sse};
use serde_json::{json, Value};
use wein_engine::{MockEngine, MockScript};
use wein_serve::{spawn_server, ServeState};

/// 模型输出：推理 → `</think>` → 正文
const THINKING: [&str; 4] = ["我在想：", "先看天气</thi", "nk>", "\n\n北京晴，-3°C"];

/// 推理里**讨论**了工具调用格式，正文里才是真调用
const THINK_WITH_TOOL_TALK: [&str; 6] = [
    "格式是 ",
    "<tool_call>\n<function=说明>\n</function>\n</tool_call>",
    " 这样的。</think>\n",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\n北京",
    "\n</parameter>\n</function>\n</tool_call>",
    "",
];

async fn start(pieces: &[&str], reasoning: bool) -> String {
    let script = MockScript {
        pieces: (0..pieces.len() as u32).collect(),
        stop_at: Some(pieces.len() as u32),
        ..Default::default()
    };
    let (engine, _mock) = MockEngine::spawn(script);
    let tok = wein_serve::MockTokenizer::from_texts(pieces);
    let mut state = ServeState::new("mock-model", Arc::new(tok), engine);
    state.reasoning = reasoning;
    let addr = spawn_server(state, "127.0.0.1:0".parse().unwrap()).await.unwrap();
    format!("http://{addr}")
}

fn body(extra: Value) -> Value {
    let mut b = json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "北京天气"}],
        "max_tokens": 32,
    });
    if let (Some(o), Some(e)) = (b.as_object_mut(), extra.as_object()) {
        for (k, v) in e {
            o.insert(k.clone(), v.clone());
        }
    }
    b
}

/// rs1 · 开关关（默认）：全文进 content，行为与旧版逐字一致（零回归护栏）
#[tokio::test]
async fn rs1_off_keeps_everything_in_content() {
    let base = start(&THINKING, false).await;
    let v: Value = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&body(json!({})))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let m = &v["choices"][0]["message"];
    assert_eq!(m["content"], json!("我在想：先看天气</think>\n\n北京晴，-3°C"));
    assert!(m.get("reasoning_content").is_none());
}

/// rs2 · 开关开（非流式）：推理与正文分离，标记外发（不出现在任何字段里）
#[tokio::test]
async fn rs2_on_splits_nonstream() {
    let base = start(&THINKING, true).await;
    let v: Value = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&body(json!({})))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let m = &v["choices"][0]["message"];
    assert_eq!(m["reasoning_content"], json!("我在想：先看天气"));
    assert_eq!(m["content"], json!("\n\n北京晴，-3°C"));
    assert!(!serde_json::to_string(&m).unwrap().contains("</think>"));
}

/// rs3 · 开关开（流式）：先 reasoning 片、后 content 片，标记一帧都不外泄
#[tokio::test]
async fn rs3_on_streams_reasoning_then_content() {
    let base = start(&THINKING, true).await;
    let frames = post_sse(&format!("{base}/v1/chat/completions"), &body(json!({}))).await;
    for f in &frames {
        assert!(!f.contains("</think>") && !f.contains("</thi"), "标记外泄：{f}");
    }
    let mut reason = String::new();
    let mut content = String::new();
    let mut seen_content = false;
    for f in frames.iter().filter(|f| f.starts_with('{')) {
        let d = &serde_json::from_str::<Value>(f).unwrap()["choices"][0]["delta"];
        if let Some(r) = d.get("reasoning_content").and_then(|x| x.as_str()) {
            assert!(!seen_content, "推理片必须全部出现在正文片之前");
            reason.push_str(r);
        }
        if let Some(c) = d.get("content").and_then(|x| x.as_str()) {
            // 首个帧是 `{"role":"assistant","content":""}` 的角色帧，不算正文
            if !c.is_empty() {
                seen_content = true;
                content.push_str(c);
            }
        }
    }
    assert_eq!(reason, "我在想：先看天气");
    assert_eq!(content, "\n\n北京晴，-3°C");
}

/// rs4 · 思维链 + 工具调用：推理段的「格式示例」不得造出假调用，正文的真调用照常解析
#[tokio::test]
async fn rs4_reasoning_tool_example_is_not_a_call() {
    let base = start(&THINK_WITH_TOOL_TALK, true).await;
    let v: Value = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&body(json!({"tools": [{"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]})))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let m = &v["choices"][0]["message"];
    let calls = m["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 1, "只应解析正文里的那一个调用：{m}");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(calls[0]["function"]["arguments"], json!("{\"city\":\"北京\"}"));
    assert!(m["reasoning_content"].as_str().unwrap().contains("<function=说明>"));
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
}

/// rs5 · `enable_thinking: false` 的请求不切分（提示里已是闭合的 think，
///      输出不含 `</think>`——若仍切分会把全部正文误判成推理）
#[tokio::test]
async fn rs5_thinking_off_request_not_split() {
    let base = start(&["北京晴，-3°C"], true).await;
    let v: Value = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&body(json!({"chat_template_kwargs": {"enable_thinking": false}})))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let m = &v["choices"][0]["message"];
    assert_eq!(m["content"], json!("北京晴，-3°C"));
    assert!(m.get("reasoning_content").is_none());
}

/// rs7 · 只有调用没有正文（`</think>` 后只剩空白）：content 必须是 null（OpenAI 合同）
#[tokio::test]
async fn rs7_call_only_content_null() {
    let base = start(&THINK_WITH_TOOL_TALK, true).await;
    let v: Value = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&body(json!({"tools": [{"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]})))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], Value::Null);
}

/// rs6 · 被 max_tokens 截断在推理中途：不丢字（全部进 reasoning_content）
#[tokio::test]
async fn rs6_truncated_in_reasoning_keeps_text() {
    let base = start(&["想", "想", "想"], true).await;
    let v: Value = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&body(json!({})))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let m = &v["choices"][0]["message"];
    assert_eq!(m["reasoning_content"], json!("想想想"));
    assert_eq!(m["content"], json!(""));
}
