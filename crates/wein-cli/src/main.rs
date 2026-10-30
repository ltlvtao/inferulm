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

/// 可选数值旗标（缺省 None）——覆盖类配置（如 --mix-cap）
fn arg_opt(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned()
}

/// HfTokenizer + ChatTemplate → serve Tokenizer 桥（组装在 cli 层——
/// serve 不依赖 model，§2 单向）。组合合同由 q11/q12（wein-serve tests）
/// 钉住，此处与之间构
struct HfServing {
    tok: wein_model::HfTokenizer,
    tpl: Option<wein_model::ChatTemplate>,
}
impl Tokenizer for HfServing {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.tok.encode(text)
    }
    fn piece_bytes(&self, id: u32) -> Cow<'_, [u8]> {
        Cow::Owned(self.tok.decode(&[id]).into_bytes())
    }
    fn render_chat_template(
        &self,
        messages: &[wein_serve::ChatMessage],
    ) -> Option<Result<String, String>> {
        // 模板在场即真身；模型目录无模板 → None（serve 回退占位）
        self.tpl.as_ref().map(|t| {
            t.render(
                &messages
                    .iter()
                    .map(|m| wein_model::ChatMsg { role: m.role.clone(), content: m.content.clone() })
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| e.to_string())
        })
    }
    fn eos_id(&self) -> Option<u32> {
        Some(self.tok.eos_id())
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
        // --graph：FULL_DECODE 图档（W2 §7）——纯 decode 步走桶图回放
        let use_qwen = engine_kind == "core" && exec_kind == "qwen";
        let use_graph = args.iter().any(|a| a == "--graph");
        // qwen 桶集盖满行容量（满负荷 c=100+ 纯 decode 步不回落 eager）；
        // toy 沿用 {1..32}（L3 档 c=100 图档不进 battery，测试自定桶集）。
        // WEIN_GRAPH_BUCKETS=1,2,4,… 可覆盖（批扩展曲线/battery 变体）
        let graph_buckets: Vec<u32> = match std::env::var("WEIN_GRAPH_BUCKETS") {
            Ok(s) => s.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
            Err(_) => {
                if use_qwen {
                    vec![1, 2, 4, 8, 16, 32, 64, 100, 128]
                } else {
                    vec![1, 2, 4, 8, 16, 32]
                }
            }
        };
        let (engine, tok): (EngineHandle, Arc<dyn Tokenizer>) = if use_qwen {
            let dir = PathBuf::from(&model);
            let ex = if use_graph {
                wein_exec::Qwen35Executor::open_graph(&dir, graph_buckets.clone()).unwrap_or_else(
                    |e| {
                        eprintln!("qwen 图档打开失败（--model 需为模型目录）：{e}");
                        std::process::exit(1)
                    },
                )
            } else {
                wein_exec::Qwen35Executor::open(&dir).unwrap_or_else(|e| {
                    eprintln!("qwen exec 打开失败（--model 需为模型目录）：{e}");
                    std::process::exit(1)
                })
            };
            let hf = wein_model::HfTokenizer::open(&dir).unwrap_or_else(|e| {
                eprintln!("tokenizer 打开失败：{e}");
                std::process::exit(1)
            });
            // chat 模板（W5）：在场即接（--graph 与否无关）
            let tpl = match wein_model::ChatTemplate::open(&dir) {
                Ok(t) => Some(t),
                Err(e) => {
                    eprintln!("chat 模板不在场（chat 端点回退占位包装）：{e}");
                    None
                }
            };
            let engine = EngineCore::spawn_with(
                {
                    // --mix-cap N：混合步封顶覆盖（battery 调参用；默认走
                    // SchedConfig::default 的 192）
                    let mut cfg = CoreConfig::scripted(vec![], Duration::ZERO);
                    if let Some(v) = arg_opt(&args, "--mix-cap") {
                        cfg.sched.mix_cap = v.parse().unwrap_or_else(|_| {
                            eprintln!("--mix-cap 须为非负整数：{v}");
                            std::process::exit(1)
                        });
                    }
                    // --chunk-size N：单请求单步 prefill 上限（同上，调参用）
                    if let Some(v) = arg_opt(&args, "--chunk-size") {
                        cfg.sched.chunk_size = v.parse().unwrap_or_else(|_| {
                            eprintln!("--chunk-size 须为正整数：{v}");
                            std::process::exit(1)
                        });
                    }
                    cfg
                },
                Box::new(ex),
            )
            .0;
            (engine, Arc::new(HfServing { tok: hf, tpl }))
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
                        let mut tcfg = wein_exec::ToyConfig::default();
                        if use_graph {
                            tcfg.graph_buckets = graph_buckets.clone();
                        }
                        let toy = wein_exec::ToyExecutor::open(tcfg).unwrap_or_else(|e| {
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
        let graph_tag = if use_graph { "+graph" } else { "" };
        let exec_tag = if engine_kind == "core" && exec_kind == "toy" {
            format!("+toy{graph_tag}")
        } else if use_qwen {
            format!("+qwen{graph_tag}")
        } else {
            graph_tag.to_string()
        };
        println!("wein ({engine_kind}{mem_tag}{exec_tag}) listening on http://{addr}  model={model}");
        tokio::signal::ctrl_c().await.ok();
    });
}
