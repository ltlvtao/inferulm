//! `wein serve` 入口。`--engine mock`（协议对照）或 `--engine core`（wein-sched 调度器
//! + 执行器）；`--exec toy`（toy 模型真 GPU 步）/ `--exec qwen`（Qwen3.5-4B 真权重，
//! I1 竖切）在此组装。参数手写解析（M0 够用），接更多模型时换 clap。
use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use wein_engine::{CoreConfig, EngineCore, EngineHandle, MockEngine, MockScript};
use wein_serve::{spawn_server, MockTokenizer, ServeState, Tokenizer};

fn arg(args: &[String], k: &str, d: &str) -> String {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| d.to_string())
}

/// HfTokenizer → serve Tokenizer 桥（组装在 cli 层——serve 不依赖 model，§2 单向）。
/// 组合合同由 q11（wein-serve tests）钉住，此处与之间构
struct HfServing(wein_model::HfTokenizer);
impl Tokenizer for HfServing {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.0.encode(text)
    }
    fn piece_bytes(&self, id: u32) -> Cow<'_, [u8]> {
        Cow::Owned(self.0.decode(&[id]).into_bytes())
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args: Vec<String> = std::env::args().collect();
    let port: u16 = arg(&args, "--port", "8080").parse().unwrap();
    let model = arg(&args, "--model", "mock-model");
    let engine_kind = arg(&args, "--engine", "mock");
    let exec_kind = arg(&args, "--exec", "scripted");
    let prefill_ms: u64 = arg(&args, "--prefill-delay-ms", "2").parse().unwrap();
    let token_ms: u64 = arg(&args, "--token-delay-ms", "1").parse().unwrap();

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async move {
        let texts: Vec<&str> = vec![
            "The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", ". ",
        ];
        let mock_tok = MockTokenizer::from_texts(&texts);
        let pieces: Vec<u32> = (0..texts.len() as u32).collect();
        // core 引擎：稳态 decode 每步恰 1 token，步延迟即 token 节拍；
        // prefill 在首几步内完成，无需单独延迟参数。
        // --memory：启用 BlockPool 水位 + radix 前缀缓存（L2m 归因电池；
        // 块容量按 bench 负载给宽裕值——量的是记账开销，不制造抢占噪声）
        //
        // --exec qwen：Qwen3.5-4B 真权重（I1 竖切）——exec 即节拍器
        // （脚本件/步延迟不参与），--model 给模型目录（safetensors+tokenizer）
        let use_qwen = engine_kind == "core" && exec_kind == "qwen";
        let (engine, tok): (EngineHandle, Arc<dyn Tokenizer>) = if use_qwen {
            let dir = PathBuf::from(&model);
            let ex = wein_exec::Qwen35Executor::open(&dir).unwrap_or_else(|e| {
                eprintln!("qwen exec 打开失败（--model 需为模型目录）：{e}");
                std::process::exit(1)
            });
            let hf = wein_model::HfTokenizer::open(&dir).unwrap_or_else(|e| {
                eprintln!("tokenizer 打开失败：{e}");
                std::process::exit(1)
            });
            let engine = EngineCore::spawn_with(
                CoreConfig::scripted(vec![], Duration::ZERO),
                Box::new(ex),
            )
            .0;
            (engine, Arc::new(HfServing(hf)))
        } else {
            let engine: EngineHandle = match engine_kind.as_str() {
                "core" => {
                    let mut cfg =
                        CoreConfig::scripted(pieces.clone(), Duration::from_millis(token_ms));
                    if args.iter().any(|a| a == "--memory") {
                        cfg.memory = Some(wein_memory::state::MemConfig {
                            block_size: 16,
                            num_blocks: 8192,
                            low_watermark: 0.1,
                        });
                    }
                    if exec_kind == "toy" {
                        // 真 exec：DeviceSession + toy 模型（L3 归因电池）。
                        // 组装在 cli 层——engine 不依赖 exec（§2 依赖单向）
                        let toy = wein_exec::ToyExecutor::open(wein_exec::ToyConfig::default())
                            .unwrap_or_else(|e| {
                                eprintln!("toy exec 打开失败：{e}");
                                std::process::exit(1)
                            });
                        EngineCore::spawn_with(cfg, Box::new(toy)).0
                    } else {
                        EngineCore::spawn(cfg).0
                    }
                }
                _ => MockEngine::spawn(MockScript {
                    prefill_delay: Duration::from_millis(prefill_ms),
                    per_token_delay: Duration::from_millis(token_ms),
                    pieces,
                    stop_at: None,
                })
                .0,
            };
            (engine, Arc::new(mock_tok))
        };
        let state = ServeState { model_id: model.clone(), tokenizer: tok, engine };
        let addr = spawn_server(state, format!("0.0.0.0:{port}").parse().unwrap())
            .await
            .expect("bind failed");
        let mem_tag = if args.iter().any(|a| a == "--memory") { "+memory" } else { "" };
        let exec_tag = if engine_kind == "core" && exec_kind == "toy" {
            "+toy"
        } else if use_qwen {
            "+qwen"
        } else {
            ""
        };
        println!("wein ({engine_kind}{mem_tag}{exec_tag}) listening on http://{addr}  model={model}");
        tokio::signal::ctrl_c().await.ok();
    });
}
