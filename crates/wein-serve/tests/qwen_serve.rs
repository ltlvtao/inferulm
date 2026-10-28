//! q11 · 真模型 serve 端到端（I1 收官面）：Qwen35Executor 经 spawn_with 组进
//! EngineCore，HfTokenizer 包成 serve Tokenizer，真 HTTP 请求吐真文本。
//! CLI `--exec qwen` 的组装与此完全同构——本条钉住该组合的合同；
//! 延迟/吞吐账归 battery L3q（bench-reports/BASELINE.md）。
mod common;

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use wein_engine::{CoreConfig, EngineCore};
use wein_exec::Qwen35Executor;
use wein_model::{model_dir, HfTokenizer};
use wein_serve::{spawn_server, ServeState, Tokenizer};

/// HfTokenizer → serve Tokenizer 桥（与 wein-cli 的 HfServing 同构）
struct HfServing(HfTokenizer);
impl Tokenizer for HfServing {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.0.encode(text)
    }
    fn piece_bytes(&self, id: u32) -> Cow<'_, [u8]> {
        Cow::Owned(self.0.decode(&[id]).into_bytes())
    }
}

/// q11 · "The capital of France is" → HTTP 响应含 "Paris"：
/// tokenizer 边界（encode/piece_bytes 增量 detok）+ 引擎调度 + 真权重前向
/// 全链路一杆子捅到底——serve 层任何一环接错都产不出 Paris
#[tokio::test]
async fn q11_real_model_http_serves_paris() {
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
        tokenizer: Arc::new(HfServing(hf)),
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
