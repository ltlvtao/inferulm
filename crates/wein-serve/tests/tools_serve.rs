//! 工具调用（function calling）serve 契约——ts 系列。
//!
//! 三件事被钉住：
//! 1. **标准 OpenAI 请求体不再 422**：`tools` / `tool_choice` / `chat_template_kwargs`、
//!    assistant 的 `tool_calls`、`tool` 角色、`content: null` 全都能收。
//! 2. **模型吐的 `<tool_call>` XML 转成 OpenAI `tool_calls`**（流式与非流式同语义），
//!    标记本身绝不进 `content`；`finish_reason` = "tool_calls"。
//! 3. **未请求 tools 就不解析**（标记原样留在 content——旧行为，零回归）。
mod common;

use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use common::{client, post_sse};
use serde_json::{json, Value};
use wein_engine::{EngineCore, EngineHandle, Executor, MockEngine, MockScript};
use wein_serve::{spawn_server, ChatMessage, ServeState, Tokenizer};

/// 模型输出：一句正文 + 一个 add(a=1,b=2) 调用（4 个 piece）
const PIECES: [&str; 4] = [
    "let me ",
    "<tool_call>\n<function=add>\n<parameter=a>\n",
    "1",
    "\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call>",
];

/// 工具声明（参数定类型用：a/b 是 integer）
fn tools() -> Value {
    json!([{"type": "function", "function": {
        "name": "add", "description": "Add two ints",
        "parameters": {"type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"]}}}])
}

/// 模板 stub：把收到的 kwargs 名字拼进 prompt（`prompt_tokens` 即「注入没注入」的判据）
struct KwTok {
    seen: Arc<Mutex<Vec<String>>>,
}

impl Tokenizer for KwTok {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|b| b as u32).collect()
    }
    fn piece_bytes(&self, id: u32) -> Cow<'_, [u8]> {
        Cow::Owned(PIECES.get(id as usize).map(|s| s.as_bytes().to_vec()).unwrap_or(b"?".to_vec()))
    }
    fn render_chat_template_with(
        &self,
        m: &[ChatMessage],
        extra: &[(String, Value)],
    ) -> Option<Result<String, String>> {
        let mut names: Vec<String> = extra.iter().map(|(k, _)| k.clone()).collect();
        names.sort();
        *self.seen.lock().unwrap() = names.clone();
        let kw = if names.is_empty() { String::new() } else { format!("<{}>", names.join(",")) };
        // 模板真身会读 tool_calls/tool 角色；stub 只验证它们没被 DTO 吞掉
        let msgs = m
            .iter()
            .map(|x| {
                format!(
                    "[{}:{}:{}]",
                    x.role,
                    x.content.as_deref().unwrap_or("null"),
                    x.tool_calls.as_ref().map(|c| c.as_array().map(|a| a.len()).unwrap_or(0)).unwrap_or(0)
                )
            })
            .collect::<String>();
        Some(Ok(format!("{kw}{msgs}")))
    }
}

async fn start(seen: Arc<Mutex<Vec<String>>>) -> String {
    let script = MockScript {
        pieces: vec![0, 1, 2, 3],
        stop_at: Some(4),
        ..Default::default()
    };
    let (engine, _mock) = MockEngine::spawn(script);
    let state = ServeState::new("mock-model", Arc::new(KwTok { seen }), engine);
    let addr = spawn_server(state, "127.0.0.1:0".parse().unwrap()).await.unwrap();
    format!("http://{addr}")
}

fn body(tools_on: bool) -> Value {
    let mut b = json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "add 1 and 2"}],
        "max_tokens": 16,
    });
    if tools_on {
        b["tools"] = tools();
    }
    b
}

/// ts1 · 非流式：XML → tool_calls，content 只剩正文，finish_reason = tool_calls
#[tokio::test]
async fn ts1_nonstream_parses_tool_call() {
    let base = start(Arc::default()).await;
    let v: Value = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&body(true))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let msg = &v["choices"][0]["message"];
    assert_eq!(msg["content"], json!("let me "));
    assert_eq!(msg["tool_calls"][0]["type"], "function");
    assert_eq!(msg["tool_calls"][0]["function"]["name"], "add");
    let args: Value = serde_json::from_str(msg["tool_calls"][0]["function"]["arguments"].as_str().unwrap())
        .unwrap();
    assert_eq!(args, json!({"a": 1, "b": 2}));
    assert!(msg["tool_calls"][0].get("index").is_none(), "非流式不带 index");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
}

/// ts2 · 流式：正文片 + 调用片按序，标记一帧都不外泄，终止帧 finish_reason 对齐
#[tokio::test]
async fn ts2_stream_parses_tool_call() {
    let base = start(Arc::default()).await;
    let frames = post_sse(&format!("{base}/v1/chat/completions"), &body(true)).await;
    let last = &frames[frames.len() - 2];
    let tail: Value = serde_json::from_str(last).unwrap();
    assert_eq!(tail["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(frames.last().unwrap(), "[DONE]");
    for f in &frames {
        assert!(!f.contains("<tool_call>"), "调用标记外泄：{f}");
        assert!(!f.contains("<function="), "调用标记外泄：{f}");
    }
    // 正文片在校验之外还要按序：先 role 帧，再 content，最后 tool_calls
    let mut saw_content = false;
    let mut saw_call = false;
    for f in frames.iter().take(frames.len() - 2) {
        let v: Value = serde_json::from_str(f).unwrap();
        let d = &v["choices"][0]["delta"];
        if d["content"] == json!("let me ") {
            assert!(!saw_call, "正文片必须在调用片之前");
            saw_content = true;
        }
        if let Some(calls) = d.get("tool_calls") {
            assert!(saw_content, "调用片缺前置正文");
            assert_eq!(calls[0]["index"], json!(0));
            assert_eq!(calls[0]["function"]["name"], json!("add"));
            saw_call = true;
        }
    }
    assert!(saw_content && saw_call);
}

/// ts3 · 未给 tools：不解析（标记原样落在 content，旧行为零回归）
#[tokio::test]
async fn ts3_no_tools_request_keeps_raw_text() {
    let base = start(Arc::default()).await;
    let v: Value = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&body(false))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let msg = &v["choices"][0]["message"];
    assert!(msg.get("tool_calls").is_none());
    assert!(msg["content"].as_str().unwrap().contains("<tool_call>"));
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

/// ts4 · 完整 OpenAI 请求体（历史 tool_calls + tool 角色 + content:null）不再 422，
///      且这些字段确实进了模板（stub 把它们拼进 prompt ⇒ prompt_tokens 变大）
#[tokio::test]
async fn ts4_full_openai_request_body_accepted() {
    let seen = Arc::new(Mutex::new(vec![]));
    let base = start(seen.clone()).await;
    let mut b = body(true);
    b["messages"] = json!([
        {"role": "user", "content": "add 1 and 2"},
        {"role": "assistant", "content": null, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "add", "arguments": "{\"a\":1,\"b\":2}"}}]},
        {"role": "tool", "content": "3", "tool_call_id": "call_1"}
    ]);
    let resp = client().post(format!("{base}/v1/chat/completions")).json(&b).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let v: Value = resp.json().await.unwrap();
    // stub 渲染串里 tool_calls 计数 = 1（不是 0）⇒ DTO 没吞字段
    assert_eq!(v["usage"]["prompt_tokens"].as_u64().unwrap(), "[tools][user:add 1 and 2:0][assistant:null:1][tool:3:0]".len() as u64);
    assert_eq!(*seen.lock().unwrap(), vec!["tools".to_string()]);
}

/// ts5 · tool_choice="none"：工具描述不注入模板（模型看不到就不会调）
#[tokio::test]
async fn ts5_tool_choice_none_skips_injection() {
    let seen = Arc::new(Mutex::new(vec![]));
    let base = start(seen.clone()).await;
    let mut b = body(true);
    b["tool_choice"] = json!("none");
    let v: Value = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&b)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(*seen.lock().unwrap(), Vec::<String>::new());
    assert_eq!(v["usage"]["prompt_tokens"].as_u64().unwrap(), "[user:add 1 and 2:0]".len() as u64);
    // 不注入 ⇒ 也不解析
    assert!(v["choices"][0]["message"].get("tool_calls").is_none());
}

/// ts6 · 坏形状一律 400（而不是拿去做 prompt）
#[tokio::test]
async fn ts6_bad_tool_shapes_400() {
    let base = start(Arc::default()).await;
    let mut b = body(true);
    b["tools"] = json!({"oops": 1});
    let r = client().post(format!("{base}/v1/chat/completions")).json(&b).send().await.unwrap();
    assert_eq!(r.status(), 400);
    let v: Value = r.json().await.unwrap();
    assert!(v["error"]["message"].as_str().unwrap().contains("'tools' must be an array"));

    let mut b = body(true);
    b["tool_choice"] = json!("banana");
    let r = client().post(format!("{base}/v1/chat/completions")).json(&b).send().await.unwrap();
    assert_eq!(r.status(), 400);
}

/// ts7 · chat_template_kwargs 非对象 → 400
#[tokio::test]
async fn ts7_bad_template_kwargs_400() {
    let base = start(Arc::default()).await;
    let mut b = body(false);
    b["chat_template_kwargs"] = json!("oops");
    let r = client().post(format!("{base}/v1/chat/completions")).json(&b).send().await.unwrap();
    assert_eq!(r.status(), 400);
    let v: Value = r.json().await.unwrap();
    assert!(v["error"]["message"].as_str().unwrap().contains("chat_template_kwargs"));
}

/// 未用到的导入兜底（EngineCore/Executor/EngineHandle 供将来换真引擎的用例）
#[allow(dead_code)]
fn _types(_: EngineCore, _: EngineHandle, _: &dyn Executor) {}
