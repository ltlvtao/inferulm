//! `/health` 与 `/metrics`（任务书 §3）：探针与观测面。
//!
//! - `/health`：引擎活着 = 200 + `{"status":"ok","steps":N}`；引擎任务没了 = 503
//! - `/metrics`：Prometheus 文本，数据源 = serve 侧计数器（请求/token）
//!   + `EngineStats`（steps/preempted/batch_rows/prefix_hit_tokens）
mod common;

use serde_json::{json, Value};

use common::{chat_body, client, default_core, default_tok, start, start_core, MockScript};

/// `name value` 形式的单值指标（无 label）
fn scalar(text: &str, name: &str) -> Option<f64> {
    text.lines()
        .find(|l| l.starts_with(&format!("{name} ")))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
}

/// `name{label="value"} x` 形式的带标签指标
fn labeled(text: &str, name: &str, label: &str, value: &str) -> Option<f64> {
    let needle = format!("{name}{{{label}=\"{value}\"}} ");
    text.lines()
        .find(|l| l.starts_with(&needle))
        .and_then(|l| l.split_whitespace().last())
        .and_then(|v| v.parse().ok())
}

#[tokio::test]
async fn h1_health_reports_ok_and_steps() {
    let ts = start_core(default_core(), default_tok()).await;
    let resp = client().get(format!("{}/health", ts.base)).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["status"], "ok");
    assert!(v["steps"].is_u64(), "steps 必须是数字：{v}");
    assert_eq!(v["draining"], false);

    // 跑一个请求后 steps 单调增（真引擎走了前向步）
    let resp = client()
        .post(format!("{}/v1/chat/completions", ts.base))
        .json(&chat_body(2))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    common::eventually(std::time::Duration::from_secs(2), || ts.stats.step_count() >= 1).await;
    let v: Value = client()
        .get(format!("{}/health", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(v["steps"].as_u64().unwrap() >= 1);
}

#[tokio::test]
async fn h2_metrics_reflect_two_requests() {
    let ts = start_core(default_core(), default_tok()).await;
    let mut prompt_sum = 0u64;
    let mut completion_sum = 0u64;
    for _ in 0..2 {
        let resp = client()
            .post(format!("{}/v1/chat/completions", ts.base))
            .json(&chat_body(2))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let v: Value = resp.json().await.unwrap();
        prompt_sum += v["usage"]["prompt_tokens"].as_u64().unwrap();
        completion_sum += v["usage"]["completion_tokens"].as_u64().unwrap();
    }
    common::eventually(std::time::Duration::from_secs(2), || ts.stats.finished_count() >= 2).await;

    let resp = client().get(format!("{}/metrics", ts.base)).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let ct = resp.headers()["content-type"].to_str().unwrap().to_string();
    assert!(ct.starts_with("text/plain"), "content-type = {ct}");
    let text = resp.text().await.unwrap();

    assert_eq!(labeled(&text, "wein_requests_total", "result", "accepted"), Some(2.0));
    assert_eq!(labeled(&text, "wein_requests_total", "result", "finished"), Some(2.0));
    assert_eq!(labeled(&text, "wein_requests_total", "result", "failed"), Some(0.0));
    assert_eq!(labeled(&text, "wein_tokens_total", "kind", "prompt"), Some(prompt_sum as f64));
    assert_eq!(
        labeled(&text, "wein_tokens_total", "kind", "completion"),
        Some(completion_sum as f64)
    );
    assert!(scalar(&text, "wein_steps_total").unwrap() >= 1.0, "steps 要进指标");
    assert!(scalar(&text, "wein_batch_rows").is_some(), "batch_rows gauge 必须在场");
    assert!(scalar(&text, "wein_preempted_total").is_some());
    assert!(
        scalar(&text, "wein_prefix_hit_tokens_total").is_some(),
        "prefix_hit_tokens_total 必须在场（拿不到就是 0）"
    );
}

/// 引擎任务死亡（命令通道关闭）→ `/health` 503、提交 503。
/// 旧实现：`submit` 的错误被 `let _ =` 吞掉 → 200 + 空内容（服务假死）。
#[tokio::test]
async fn h3_dead_engine_health_and_submit_are_503() {
    // 造一个"任务已退出"的引擎：channel 的收端立即丢弃
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    drop(rx);
    let engine = wein_engine::EngineHandle::new(tx);
    assert!(!engine.is_alive());
    let state = wein_serve::ServeState::new(
        "mock-model",
        std::sync::Arc::new(default_tok()),
        engine,
    );
    let addr = wein_serve::spawn_server(state, "127.0.0.1:0".parse().unwrap()).await.unwrap();
    let base = format!("http://{addr}");

    let resp = client().get(format!("{base}/health")).send().await.unwrap();
    assert_eq!(resp.status(), 503, "引擎死了 /health 必须 503");
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["status"], "error");

    let resp = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 503, "提交失败必须 503（不许假死成 200 空内容）");
    let v: Value = resp.json().await.unwrap();
    assert!(v["error"].is_object());
    assert_eq!(v["error"]["type"], "server_error");

    // /metrics 仍可读（failed 计数 +1）
    let text = client().get(format!("{base}/metrics")).send().await.unwrap().text().await.unwrap();
    assert_eq!(labeled(&text, "wein_requests_total", "result", "failed"), Some(1.0));
}

/// `/v1/models` 仍是模型清单（bench 探针已改 /health，但路由不能删——客户端在用）
#[tokio::test]
async fn h4_models_route_still_serves_list() {
    let ts = start(MockScript::default(), default_tok()).await;
    let v: Value = client()
        .get(format!("{}/v1/models", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["data"][0]["id"], "mock-model");
}
