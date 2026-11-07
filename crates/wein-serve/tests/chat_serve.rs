//! W5-A chat 模板 serve 契约（cs 系列）——`/v1/chat/completions` 的模板真身。
//!
//! 模板能力经 Tokenizer trait 的 `render_chat_template` 钩子注入（桥接在
//! cli 层，§2 单向）；无模板（mock）回退占位包装——既有 24 用例即 parity。
//! eos 停止走引擎 `stop_token_ids`（W5-B）：serve 把 `eos_id` 递给引擎，
//! 命中即 Finished(Stop)、该 token 不外吐（vLLM 语义）。
mod common;

use std::borrow::Cow;
use std::sync::Arc;

use common::client;
use serde_json::{json, Value};
use wein_engine::EngineCore;
use wein_serve::{spawn_server, ChatMessage, ServeState, Tokenizer};

/// 模板化 stub：render 输出确定形态 `[role]content`（与占位
/// `role: content\n` 长度不同——prompt_tokens 即「走没走模板」的判据）；
/// encode 按 ASCII 字节、piece 按 'a'+id 映射（pieces 0..7 → "abcdefg…"）
struct TplTok {
    eos: Option<u32>,
    fail: bool,
}

impl Tokenizer for TplTok {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.as_bytes().iter().map(|b| *b as u32).collect()
    }
    fn piece_bytes(&self, id: u32) -> Cow<'_, [u8]> {
        Cow::Owned(vec![b'a' + id as u8])
    }
    fn render_chat_template(&self, m: &[ChatMessage]) -> Option<Result<String, String>> {
        if self.fail {
            return Some(Err("No user query found in messages.".into()));
        }
        Some(Ok(m
            .iter()
            .map(|m| format!("[{}]{}", m.role, m.content.as_deref().unwrap_or("")))
            .collect()))
    }
    fn eos_id(&self) -> Option<u32> {
        self.eos
    }
}

/// EngineCore（脚本执行器 pieces 0..7）+ 自定义 tokenizer 起服务
async fn start_tok(tok: TplTok) -> String {
    let (engine, _stats) = EngineCore::spawn(common::default_core());
    let state = ServeState::new(
        "mock-model",
        Arc::new(tok),
        engine,
    );
    let addr = spawn_server(state, "127.0.0.1:0".parse().unwrap()).await.unwrap();
    format!("http://{addr}")
}

/// cs1 · 模板进引擎：prompt_tokens == 模板渲染串的 token 数
///      （`[user]hello` = 11；占位 `user: hello\n` = 12——判据即在此）
#[tokio::test]
async fn cs1_template_output_is_the_engine_prompt() {
    let base = start_tok(TplTok { eos: None, fail: false }).await;
    let resp = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 3,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["usage"]["prompt_tokens"].as_u64().unwrap(), 11, "须走模板而非占位");
    assert_eq!(v["choices"][0]["message"]["content"].as_str().unwrap(), "abc");
    assert_eq!(v["choices"][0]["finish_reason"], "length");
}

/// cs2 · 模板拒绝 → 400 且错误信息透传（模型侧 raise_exception 语义）
#[tokio::test]
async fn cs2_template_rejection_maps_to_400() {
    let base = start_tok(TplTok { eos: None, fail: true }).await;
    let resp = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "mock-model",
            "messages": [{"role": "assistant", "content": "hi"}],
            "max_tokens": 3,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let v: Value = resp.json().await.unwrap();
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(msg.contains("No user query found"), "got {msg:?}");
}

/// cs3 · eos 停止：eos_id=7 递给引擎，pieces 0..7 第 8 个 token 命中 →
///      不外吐（content 到 "abcdefg" 为止）、finish_reason=stop、
///      completion_tokens=7（max_tokens=16 内提前停）
#[tokio::test]
async fn cs3_eos_stops_generation_without_emitting() {
    let base = start_tok(TplTok { eos: Some(7), fail: false }).await;
    let resp = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["choices"][0]["message"]["content"].as_str().unwrap(), "abcdefg");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["completion_tokens"].as_u64().unwrap(), 7);
}

/// cs4 · /v1/completions 同样吃 eos 停止（原始续写端点默认停于 eos——
///      vLLM 同款；无 eos 的 tokenizer 行为不变由既有套件 parity 钉）
#[tokio::test]
async fn cs4_completions_also_stop_at_eos() {
    let base = start_tok(TplTok { eos: Some(7), fail: false }).await;
    let resp = client()
        .post(format!("{base}/v1/completions"))
        .json(&json!({
            "model": "mock-model",
            "prompt": "The capital of France is",
            "max_tokens": 16,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["choices"][0]["text"].as_str().unwrap(), "abcdefg");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}
