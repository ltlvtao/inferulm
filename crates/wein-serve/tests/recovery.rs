//! 恢复类回归：**毒请求不许带走服务**。
//!
//! 旧实现的三条链路（任务书 §1/§2）：
//! ① `prompt + max_tokens > MAX_POS` → exec `assert!` → EngineCore 任务 panic
//!    → 在飞流 `Done{Aborted}`、`cmd_rx` 随任务 drop → 此后 `submit` 的 `Err`
//!    被 `let _ =` 吞掉 → **后续请求全部 200 + 空内容**（服务假死）。
//! ② 页池耗尽 / vocab 越界同样 panic。
//!
//! 本文件用**注入式执行器**（记录 `StepError` 而不是 panic）钉死：
//! - 越界请求在 serve 前置校验处拿到 400（`ctx_limit` 由执行器上报）；
//! - 执行器级失败以 `finish_reason:"error"`（Invalid）/ `"length"`（Capacity）
//!   收尾，并在非流式 body 带 `wein_finish`；
//! - **其后的正常请求仍拿到正确 token**（假死回归的判据）。
mod common;

use std::borrow::Cow;
use std::time::Duration;

use serde_json::{json, Value};
use wein_engine::{Executor, StepError};
use wein_sched::{ReqId, StepPlan};
use wein_serve::{ChatMessage, MockTokenizer, Tokenizer};

use common::{client, default_core};

/// 毒标记：prompt 里出现这些 token 就模拟执行器的两类失败
const POISON_INVALID: u32 = 999; // 词表越界 → StepError::Invalid → finish_reason "error"
const POISON_CAPACITY: u32 = 998; // 页池耗尽 → StepError::Capacity → finish_reason "length"

/// 注入式执行器：真身是 `ScriptedExecutor`，只在准入时识别毒 token 并记
/// `StepError`（**不 panic**——这正是被测的合同）
struct PoisonExec {
    inner: wein_engine::ScriptedExecutor,
    failures: Vec<(ReqId, StepError)>,
}

impl PoisonExec {
    fn new(pieces: Vec<u32>) -> Self {
        Self { inner: wein_engine::ScriptedExecutor::new(pieces), failures: Vec::new() }
    }
}

impl Executor for PoisonExec {
    fn step(&mut self, plan: &StepPlan) -> Vec<(ReqId, u32)> {
        self.inner.step(plan)
    }
    fn retain(&mut self, active: &[ReqId]) {
        self.inner.retain(active)
    }
    fn admitted(&mut self, id: ReqId, prompt: &[u32]) -> usize {
        if prompt.contains(&POISON_INVALID) {
            self.failures
                .push((id, StepError::Invalid(format!("token 越界（vocab {POISON_INVALID}）"))));
            return 0;
        }
        if prompt.contains(&POISON_CAPACITY) {
            self.failures.push((id, StepError::Capacity("页池耗尽（4 页，请求需 9 页）".into())));
            return 0;
        }
        self.inner.admitted(id, prompt)
    }
    fn ctx_limit(&self) -> usize {
        // 小窗口便于构造"越界"（真身是 MAX_POS）
        8
    }
    fn take_failures(&mut self) -> Vec<(ReqId, StepError)> {
        std::mem::take(&mut self.failures)
    }
}

/// 毒性 tokenizer：文本含 "poison-invalid"/"poison-capacity" 时产出毒 token，
/// 其余走 MockTokenizer 的确定性切分
struct PoisonTok {
    base: MockTokenizer,
}

impl Tokenizer for PoisonTok {
    fn encode(&self, text: &str) -> Vec<u32> {
        if text.contains("poison-invalid") {
            return vec![POISON_INVALID];
        }
        if text.contains("poison-capacity") {
            return vec![POISON_CAPACITY];
        }
        self.base.encode(text)
    }
    fn piece_bytes(&self, id: u32) -> Cow<'_, [u8]> {
        self.base.piece_bytes(id)
    }
    fn render_chat_template(&self, _messages: &[ChatMessage]) -> Option<Result<String, String>> {
        None
    }
    fn eos_id(&self) -> Option<u32> {
        None
    }
}

fn tok() -> PoisonTok {
    PoisonTok { base: MockTokenizer::from_texts(&["a", "b", "c", "d"]) }
}

fn body_with(content: &str, max_tokens: u32) -> Value {
    json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    })
}

async fn start_poison() -> common::TestCore {
    common::start_core_with(default_core(), PoisonExec::new(vec![0, 1, 2, 3]), tok()).await
}

/// 用 reqwest 拿原始 SSE 帧（含 `event:` 行）——`post_sse` 只留 data 行
async fn sse_raw(base: &str, body: &Value) -> Vec<String> {
    let mut body = body.clone();
    body["stream"] = json!(true);
    let resp = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .no_proxy()
        .build()
        .unwrap()
        .post(format!("{base}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send failed");
    assert_eq!(resp.status(), 200);
    resp.text().await.unwrap().split("\n\n").map(|s| s.to_string()).collect()
}

/// r1 · 越界（prompt + max_tokens > ctx_limit）→ **400**，且错误体是 OpenAI 形状。
/// 旧实现：`assert!(prompt.len()+max_tokens <= MAX_POS)` → 引擎任务 panic。
#[tokio::test]
async fn r1_oversized_request_is_400_not_panic() {
    let ts = start_poison().await;
    // ctx_limit = 8：36 字节 → 9 token，再加 max_tokens 2 → 11 > 8
    let body = body_with("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 2);
    let resp = client()
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400, "越界必须是协议错误");
    assert!(resp.headers().contains_key("x-request-id"), "错误响应也要回写 request id");
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["error"]["type"], "invalid_request_error");
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(msg.contains("maximum context length"), "message = {msg:?}");
    // 400 不经过引擎：accepted 计数保持 0
    assert_eq!(ts.stats.accepted_count(), 0);
}

/// r2 · **假死回归钉死**：毒请求（vocab 越界）以 error 收尾，紧接着的正常
/// 请求必须拿到正确 token（旧实现此后全部 200 + 空内容）。
#[tokio::test]
async fn r2_invalid_request_errors_then_next_request_succeeds() {
    let ts = start_poison().await;
    let poisoned = client()
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&body_with("poison-invalid", 2))
        .send()
        .await
        .unwrap();
    assert_eq!(poisoned.status(), 200, "已在生成的请求无法改状态码：以 finish_reason 表达");
    let v: Value = poisoned.json().await.unwrap();
    assert_eq!(v["choices"][0]["finish_reason"], "error", "aborted/invalid 不再伪装成 stop");
    assert_eq!(v["wein_finish"], "error", "非流式 body 必须有可判别字段");

    // 假死判据：下一条正常请求仍走通且内容正确（pieces 0..3 = "abc"）
    let resp = client()
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&body_with("hi", 3))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "引擎任务必须还活着（不许假死）");
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "abc");
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert!(v.get("wein_finish").is_none(), "正常结束不带 wein_finish");
    // 引擎侧确实死了这条请求、而不是整个引擎：accepted 2（毒 + 正常）
    assert_eq!(ts.stats.accepted_count(), 2);
}

/// r3 · 容量类失败（页池耗尽）→ `finish_reason:"length"`（可恢复，别的请求不受影响）
#[tokio::test]
async fn r3_capacity_failure_ends_as_length_and_server_survives() {
    let ts = start_poison().await;
    let resp = client()
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&body_with("poison-capacity", 2))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["choices"][0]["finish_reason"], "length", "容量类 = length（OpenAI 可解释）");
    // 容量类是"可解释的截断"：finish_reason 已经是 length，不再额外打
    // wein_finish（那是 aborted/error 的判别字段）。对照表见汇报。
    assert!(v.get("wein_finish").is_none(), "capacity → length，无额外判别字段");

    let resp = client()
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&body_with("hi", 2))
        .send()
        .await
        .unwrap();
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "ab");
}

/// r4 · 流式：异常收尾先发 `event: error` 帧，再发 `finish_reason:"error"`
/// 的终止块，最后 `[DONE]`（客户端可无歧义判别，不会把截断当正常结束）。
#[tokio::test]
async fn r4_stream_error_frame_before_finish() {
    let ts = start_poison().await;
    let frames = sse_raw(&ts.base, &body_with("poison-invalid", 2)).await;
    let pos_err = frames
        .iter()
        .position(|f| f.contains("event: error"))
        .expect("必须有一帧 event: error");
    let pos_done = frames
        .iter()
        .position(|f| f.contains("\"finish_reason\":\"error\""))
        .expect("终止块 finish_reason 必须是 error");
    assert!(pos_err < pos_done, "event: error 必须在终止块之前");
    let last = frames.iter().rev().find(|f| !f.trim().is_empty()).unwrap();
    assert_eq!(last.trim(), "data: [DONE]");
    // 之后仍能正常流式生成
    let frames = common::post_sse(&format!("{}/v1/chat/completions", ts.base), &body_with("hi", 2)).await;
    assert_eq!(common::content_of(&frames), "ab");
}
