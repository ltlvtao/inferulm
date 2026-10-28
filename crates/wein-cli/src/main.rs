//! `wein serve` 入口。`--engine mock`（协议对照）或 `--engine core`（wein-sched 调度器
//! + 脚本执行器）；真实 exec/模型接入后此处按 HardwareProfile 组装。
//! 参数手写解析（M0 够用），接真实模型时换 clap。
use std::sync::Arc;
use std::time::Duration;

use wein_engine::{CoreConfig, EngineCore, EngineHandle, MockEngine, MockScript};
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
    let engine_kind = arg(&args, "--engine", "mock");
    let prefill_ms: u64 = arg(&args, "--prefill-delay-ms", "2").parse().unwrap();
    let token_ms: u64 = arg(&args, "--token-delay-ms", "1").parse().unwrap();

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async move {
        let texts: Vec<&str> = vec![
            "The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", ". ",
        ];
        let tok = MockTokenizer::from_texts(&texts);
        let pieces: Vec<u32> = (0..texts.len() as u32).collect();
        // core 引擎：稳态 decode 每步恰 1 token，步延迟即 token 节拍；
        // prefill 在首几步内完成，无需单独延迟参数。
        // --memory：启用 BlockPool 水位 + radix 前缀缓存（L2m 归因电池；
        // 块容量按 bench 负载给宽裕值——量的是记账开销，不制造抢占噪声）
        let engine: EngineHandle = match engine_kind.as_str() {
            "core" => {
                let mut cfg = CoreConfig::scripted(pieces, Duration::from_millis(token_ms));
                if args.iter().any(|a| a == "--memory") {
                    cfg.memory = Some(wein_memory::state::MemConfig {
                        block_size: 16,
                        num_blocks: 8192,
                        low_watermark: 0.1,
                    });
                }
                EngineCore::spawn(cfg).0
            }
            _ => MockEngine::spawn(MockScript {
                prefill_delay: Duration::from_millis(prefill_ms),
                per_token_delay: Duration::from_millis(token_ms),
                pieces,
                stop_at: None,
            })
            .0,
        };
        let state = ServeState { model_id: model.clone(), tokenizer: Arc::new(tok), engine };
        let addr = spawn_server(state, format!("0.0.0.0:{port}").parse().unwrap())
            .await
            .expect("bind failed");
        let mem_tag = if args.iter().any(|a| a == "--memory") { "+memory" } else { "" };
        println!("wein ({engine_kind}{mem_tag}) listening on http://{addr}  model={model}");
        tokio::signal::ctrl_c().await.ok();
    });
}
