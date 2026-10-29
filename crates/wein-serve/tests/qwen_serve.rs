//! q11 · 真模型 serve 端到端（I1 收官面）：Qwen35Executor 经 spawn_with 组进
//! EngineCore，HfTokenizer 包成 serve Tokenizer，真 HTTP 请求吐真文本。
//! q12 · chat 端点模板真身（W5）：/v1/chat/completions 经真 chat 模板进引擎。
//! CLI `--exec qwen` 的组装与此完全同构——本文件钉住该组合的合同；
//! 延迟/吞吐账归 battery L3q（bench-reports/BASELINE.md）。
mod common;

use std::borrow::Cow;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use serde_json::json;
use wein_engine::{CoreConfig, EngineCore};
use wein_exec::Qwen35Executor;
use wein_model::{model_dir, ChatMsg, ChatTemplate, HfTokenizer};
use wein_serve::{spawn_server, ChatMessage, ServeState, Tokenizer};

/// 显存串行锁：单执行器 ≈15GB，4090 装不下两个——q 系同款纪律
fn guard() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

/// HfTokenizer(+ChatTemplate) → serve Tokenizer 桥（与 wein-cli 的 HfServing 同构）
struct HfServing {
    tok: HfTokenizer,
    tpl: Option<ChatTemplate>,
}
impl Tokenizer for HfServing {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.tok.encode(text)
    }
    fn piece_bytes(&self, id: u32) -> Cow<'_, [u8]> {
        Cow::Owned(self.tok.decode(&[id]).into_bytes())
    }
    fn render_chat_template(&self, messages: &[ChatMessage]) -> Option<Result<String, String>> {
        self.tpl.as_ref().map(|t| {
            t.render(
                &messages
                    .iter()
                    .map(|m| ChatMsg { role: m.role.clone(), content: m.content.clone() })
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| e.to_string())
        })
    }
    fn eos_id(&self) -> Option<u32> {
        Some(self.tok.eos_id())
    }
}

/// q11 · "The capital of France is" → HTTP 响应含 "Paris"：
/// tokenizer 边界（encode/piece_bytes 增量 detok）+ 引擎调度 + 真权重前向
/// 全链路一杆子捅到底——serve 层任何一环接错都产不出 Paris
#[tokio::test]
async fn q11_real_model_http_serves_paris() {
    let _g = guard();
    let Some(dir) = model_dir() else {
        eprintln!("SKIP（模型目录不在场：WEIN_MODEL_DIR 或 ~/.cache/wein/Qwen3.5-4B）");
        return;
    };
    let Ok(ex) = Qwen35Executor::open(&dir) else {
        eprintln!("SKIP（Qwen35Executor 装配失败）");
        return;
    };
    let Ok(hf) = HfTokenizer::open(&dir) else {
        eprintln!("SKIP（tokenizer.json 缺失）");
        return;
    };
    // 引擎自跑真前向——脚本件/节拍不参与（与 CLI --exec qwen 同构）
    let (engine, _stats) =
        EngineCore::spawn_with(CoreConfig::scripted(vec![], Duration::ZERO), Box::new(ex));
    let state = ServeState {
        model_id: "Qwen3.5-4B".into(),
        tokenizer: Arc::new(HfServing { tok: hf, tpl: None }),
        engine,
    };
    let addr = spawn_server(state, "127.0.0.1:0".parse().unwrap()).await.unwrap();
    let base = format!("http://{addr}");

    // 走 /v1/completions（原始 prompt 直编码）；/v1/chat/completions 的
    // chat 模板是 W5 交付项（现为占位包装），不在 I1 验收面
    let body = json!({
        "model": "Qwen3.5-4B",
        "prompt": "The capital of France is",
        "max_tokens": 12,
    });
    let resp = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .unwrap()
        .post(format!("{base}/v1/completions"))
        .json(&body)
        .send()
        .await
        .expect("send failed");
    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(v["object"], "text_completion");
    let content = v["choices"][0]["text"].as_str().unwrap().to_string();
    eprintln!("q11 续写：{content:?}");
    assert!(content.contains("Paris"), "续写须含 Paris（got {content:?}）");
    assert_eq!(v["choices"][0]["finish_reason"], "length", "12 token 内不触及 eos");
    assert_eq!(v["usage"]["completion_tokens"].as_u64().unwrap(), 12);
    assert!(v["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
}

/// q12 · chat 端点模板真身（W5）：/v1/chat/completions 的 prompt 必须 ==
/// encode(ChatTemplate 渲染串) 的精确 token 数——模板→tokenizer→引擎全链
/// 一杆对齐（差一个 token 都是走了占位/错模板）；贪心续写确定性 +
/// eos 停止接线（12 token 到不了 eos → length，与 q11 同判据）
#[tokio::test]
async fn q12_chat_endpoint_template_end_to_end() {
    let _g = guard();
    let Some(dir) = model_dir() else {
        eprintln!("SKIP（模型目录不在场）");
        return;
    };
    let Ok(ex) = Qwen35Executor::open(&dir) else {
        eprintln!("SKIP（Qwen35Executor 装配失败）");
        return;
    };
    let Ok(hf) = HfTokenizer::open(&dir) else {
        eprintln!("SKIP（tokenizer.json 缺失）");
        return;
    };
    let Ok(tpl) = ChatTemplate::open(&dir) else {
        eprintln!("SKIP（chat 模板不在场）");
        return;
    };
    // 期望值独立计算：模板渲染 → encode 的 token 数
    let rendered = tpl
        .render(&[ChatMsg { role: "user".into(), content: "The capital of France is".into() }])
        .unwrap();
    let want = hf.encode(&rendered).len();

    let (engine, _stats) =
        EngineCore::spawn_with(CoreConfig::scripted(vec![], Duration::ZERO), Box::new(ex));
    let state = ServeState {
        model_id: "Qwen3.5-4B".into(),
        tokenizer: Arc::new(HfServing { tok: hf, tpl: Some(tpl) }),
        engine,
    };
    let addr = spawn_server(state, "127.0.0.1:0".parse().unwrap()).await.unwrap();
    let base = format!("http://{addr}");

    let body = json!({
        "model": "Qwen3.5-4B",
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "max_tokens": 12,
    });
    let resp = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .unwrap()
        .post(format!("{base}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send failed");
    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(
        v["usage"]["prompt_tokens"].as_u64().unwrap(),
        want as u64,
        "chat prompt 须精确等于模板渲染串的 token 数（got {}，渲染 {:?}）",
        v["usage"]["prompt_tokens"],
        rendered
    );
    let content = v["choices"][0]["message"]["content"].as_str().unwrap().to_string();
    eprintln!("q12 chat 续写（模板尾 <think> 后贪心）：{content:?}");
    assert!(!content.is_empty(), "贪心续写必须非空");
    assert_eq!(v["choices"][0]["finish_reason"], "length", "12 token 内不触及 eos");
    assert_eq!(v["usage"]["completion_tokens"].as_u64().unwrap(), 12);
}
