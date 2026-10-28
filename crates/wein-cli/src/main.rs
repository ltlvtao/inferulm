//! `wein serve` 入口。M0 只有 mock 引擎；真实引擎接入后此处按 HardwareProfile 组装。
//! 参数手写解析（M0 够用），接真实模型时换 clap。
use std::sync::Arc;
use std::time::Duration;

use wein_engine::{MockEngine, MockScript};
use wein_serve::{spawn_server, MockTokenizer, ServeState};

fn arg(args: &[String], k: &str, d: &str) -> String {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| d.to_string())
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args: Vec<String> = std::env::args().collect();
    let port: u16 = arg(&args, "--port", "8080").parse().unwrap();
    let model = arg(&args, "--model", "mock-model");
    let prefill_ms: u64 = arg(&args, "--prefill-delay-ms", "2").parse().unwrap();
    let token_ms: u64 = arg(&args, "--token-delay-ms", "1").parse().unwrap();

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async move {
        let texts: Vec<&str> = vec![
            "The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", ". ",
        ];
        let tok = MockTokenizer::from_texts(&texts);
        let pieces: Vec<u32> = (0..texts.len() as u32).collect();
        let script = MockScript {
            prefill_delay: Duration::from_millis(prefill_ms),
            per_token_delay: Duration::from_millis(token_ms),
            pieces,
            stop_at: None,
        };
        let (engine, _mock) = MockEngine::spawn(script);
        let state = ServeState { model_id: model.clone(), tokenizer: Arc::new(tok), engine };
        let addr = spawn_server(state, format!("0.0.0.0:{port}").parse().unwrap())
            .await
            .expect("bind failed");
        println!("wein (mock) listening on http://{addr}  model={model}");
        tokio::signal::ctrl_c().await.ok();
    });
}
