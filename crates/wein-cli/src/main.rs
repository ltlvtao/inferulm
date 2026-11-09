//! `wein serve` 入口。`--engine mock`（协议对照）或 `--engine core`（wein-sched 调度器
//! + 执行器）；`--exec toy`（toy 模型真 GPU 步）/ `--exec qwen`（Qwen3.5-4B 真权重，
//! I1 竖切）在此组装。参数手写解析（M0 够用），接更多模型时换 clap。
use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use wein_engine::{CoreConfig, EngineCore, EngineHandle, MockEngine, MockScript};
use wein_serve::{spawn_server_guarded, MockTokenizer, ServeState, Tokenizer};

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
        self.render_chat_template_with(messages, &[])
    }
    fn render_chat_template_with(
        &self,
        messages: &[wein_serve::ChatMessage],
        extra: &[(String, serde_json::Value)],
    ) -> Option<Result<String, String>> {
        // 模板在场即真身；模型目录无模板 → None（serve 回退占位）
        self.tpl.as_ref().map(|t| {
            t.render_with(
                &messages
                    .iter()
                    .map(|m| wein_model::ChatMsg {
                        role: m.role.clone(),
                        content: m.content.clone(),
                        tool_calls: m.tool_calls.clone(),
                        reasoning_content: m.reasoning_content.clone(),
                    })
                    .collect::<Vec<_>>(),
                extra,
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
    // SIGINT/SIGTERM 后的 drain 上限：等完在飞请求再退，超时强制退出并列出未完成
    let drain_ms: u64 = arg(&args, "--drain-timeout-ms", "5000").parse().unwrap_or(5000);

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
        // 上下文容量（W2 B2）：`--ctx-len` / `--kv-pages` / `--snap-stride`
        // 覆盖环境变量兜底（WEIN_CTX_LEN / WEIN_KV_PAGES / WEIN_SNAP_STRIDE），
        // 默认 256/512/16 = 原编译期常量。exec 的池、块表宽度、rope 表、
        // serve 的 `ctx_limit` 全部读这一份
        {
            let mut c = wein_exec::CtxConfig::from_env();
            let mut take = |flag: &str, dst: &mut usize, name: &str| {
                if let Some(v) = arg_opt(&args, flag) {
                    *dst = v.parse().unwrap_or_else(|_| {
                        eprintln!("{flag} 须为正整数：{v}");
                        std::process::exit(1)
                    });
                }
                let _ = name;
            };
            take("--ctx-len", &mut c.max_pos, "max_pos");
            take("--kv-pages", &mut c.kv_pages, "kv_pages");
            take("--snap-stride", &mut c.snap_stride, "snap_stride");
            take("--snap-slots", &mut c.snap_slots, "snap_slots");
            // --t-max N：批前向行上限显式覆盖（未给 = 按 --ctx-len 自动档：
            // ≥512、封顶 8192）。大块 prefill（chunk > 512）必须抬它——
            // exec 的激活/输入表全按 t_max 定尺
            take("--t-max", &mut c.t_max, "t_max");
            wein_exec::ctx::set(c).unwrap_or_else(|e| {
                eprintln!("上下文容量配置无效：{e}");
                std::process::exit(1)
            });
        }
        let mut warmup_len = 0usize;
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
            let mut warmup_len_inner = warmup_len;
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
                    // --mix-budget-ms F：混合步时间预算（0=关；与 --mix-cap
                    // 双闸门取更小——battery/Agent 调参用）
                    if let Some(v) = arg_opt(&args, "--mix-budget-ms") {
                        cfg.sched.mix_budget_ms = v.parse().unwrap_or_else(|_| {
                            eprintln!("--mix-budget-ms 须为非负数：{v}");
                            std::process::exit(1);
                        });
                    }
                    if let Some(v) = arg_opt(&args, "--mix-budget-rows") {
                        cfg.sched.mix_budget_rows = v.parse().unwrap_or_else(|_| {
                            eprintln!("--mix-budget-rows 须为非负整数：{v}");
                            std::process::exit(1);
                        });
                    }
                    // --chunk-size N：单请求单步 prefill 上限（同上，调参用）
                    if let Some(v) = arg_opt(&args, "--chunk-size") {
                        cfg.sched.chunk_size = v.parse().unwrap_or_else(|_| {
                            eprintln!("--chunk-size 须为正整数：{v}");
                            std::process::exit(1)
                        });
                    }
                    // token 预算默认接 t_max（默认档 = 512，与 SchedConfig
                    // 原值相同⇒默认路径逐字节不变；--ctx-len 8192 时自动
                    // 抬到 8192，chunk > 512 不被预算钳死）。--token-budget
                    // 显式覆盖（battery 调参用）
                    cfg.sched.token_budget = wein_exec::ctx::t_max();
                    if let Some(v) = arg_opt(&args, "--token-budget") {
                        cfg.sched.token_budget = v.parse().unwrap_or_else(|_| {
                            eprintln!("--token-budget 须为正整数：{v}");
                            std::process::exit(1);
                        });
                    }
                    // --memory：BlockPool 水位准入 + radix 前缀缓存（真模型路径
                    // 与 toy/scripted 同一套尺子）。块几何取自执行器自己的
                    // 页池常量（KV 页 16 token × 512 页）——不在 cli 里抄魔数。
                    // 溢出策略走既有 `PoolCapacity`：先 preempt（recompute），
                    // 批内仅剩自己才 abort（engine/core.rs 的 forced 路径）
                    if args.iter().any(|a| a == "--memory") {
                        cfg.memory = Some(wein_memory::state::MemConfig {
                            block_size: wein_exec::kv_block(),
                            num_blocks: wein_exec::kv_pages(),
                            low_watermark: 0.1,
                        });
                    }
                    // 启动预热的步长（= 调度器实际会用的满 chunk，见下）。
                    // 钳到 ctx/KV 池余量内：chunk ≥ ctx 时预热请求会被容量
                    // 校验当场打回（warmup 静默 0ms，冷跑全款付 Lt 首调——
                    // chunk8192/ctx8192 档实测）
                    let ctx_cap = wein_exec::ctx::max_pos() as usize;
                    let kv_cap =
                        wein_exec::kv_pages() as usize * wein_exec::kv_block() as usize;
                    warmup_len_inner = cfg
                        .sched
                        .chunk_size
                        .min(cfg.sched.token_budget)
                        .min(ctx_cap.saturating_sub(64))
                        .min(kv_cap.saturating_sub(64));
                    cfg
                },
                Box::new(ex),
            )
            .0;
            warmup_len = warmup_len_inner;
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
        let ctx_limit = engine.ctx_limit();
        // 启动预热（qwen；--no-warmup 可关，对照测量用）：进程首步 prefill
        // 的一次性成本 ~137ms（cuBLASLt workspace/algo 初始化、lm_head 首
        // 调、模块惰性页入——6009 冷跑首轮实测 676 vs 540）。暖一发满
        // chunk 即覆盖全部 m 族（1913-after-2048 实测零税——初始化成本
        // 按 (n,k) 摊，不按 m 重付），带 4 个 decode 步顺带暖采样路径；
        // 快照/KV 随后整池清空。vLLM 同款 startup warmup。
        if use_qwen && warmup_len > 0 && !args.iter().any(|a| a == "--no-warmup") {
            let t0 = std::time::Instant::now();
            let ids: Vec<u32> =
                (0..warmup_len as u32).map(|i| 1000 + (i * 37) % 2000).collect();
            match engine.submit(ids, 4, Default::default()) {
                Ok((_, mut stream)) => {
                    while let Some(ev) = stream.recv().await {
                        if matches!(ev, wein_engine::EngineEvent::Finished(_)) {
                            break;
                        }
                    }
                    engine.clear_prefix_cache();
                    println!(
                        "warmup: {}ms（{} token prefill + 4 decode；前缀缓存已清）",
                        t0.elapsed().as_millis(),
                        warmup_len
                    );
                }
                Err(_) => eprintln!("warmup 提交失败（引擎已关闭）——跳过"),
            }
        }
        // 输出侧思维链解析（vLLM --reasoning-parser 同位）：
        // auto = 渲染一条探针消息，生成提示以**未闭合**的 `<think>` 收尾就开
        // （开思考的模型先吐推理再吐 `</think>`，不切分则推理混在 content 里）
        let rp_mode = arg(&args, "--reasoning-parser", "auto");
        let reasoning = match rp_mode.as_str() {
            "none" | "off" => false,
            "qwen3" | "deepseek_r1" | "on" => true,
            _ => {
                let probe = [wein_serve::ChatMessage {
                    role: "user".into(),
                    content: Some("hi".into()),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                    reasoning_content: None,
                }];
                match tok.render_chat_template_with(&probe, &[]) {
                    Some(Ok(s)) => match s.rfind("<think>") {
                        Some(i) => !s[i..].contains("</think>"),
                        None => false,
                    },
                    _ => false,
                }
            }
        };
        let mut state = ServeState::new(model.clone(), tok, engine);
        state.reasoning = reasoning;
        let server = spawn_server_guarded(state, format!("0.0.0.0:{port}").parse().unwrap())
            .await
            .expect("bind failed");
        let addr = server.addr;
        let mem_tag = if args.iter().any(|a| a == "--memory") { "+memory" } else { "" };
        let graph_tag = if use_graph { "+graph" } else { "" };
        let exec_tag = if engine_kind == "core" && exec_kind == "toy" {
            format!("+toy{graph_tag}")
        } else if use_qwen {
            format!("+qwen{graph_tag}")
        } else {
            graph_tag.to_string()
        };
        let rp_tag = if reasoning { "+reasoning" } else { "" };
        println!(
            "wein ({engine_kind}{mem_tag}{exec_tag}{rp_tag}) listening on http://{addr}  model={model} \
             ctx_limit={ctx_limit} drain_timeout={drain_ms}ms"
        );
        // SIGINT + SIGTERM 同一路径：停止接纳 → 等在飞 ≤ drain_ms → 退出码 0
        let sig = wait_for_signal().await;
        println!("收到 {sig}：停止接纳新请求，drain 上限 {drain_ms}ms");
        let left = server.shutdown(Duration::from_millis(drain_ms)).await;
        if left.is_empty() {
            println!("drain 完成（无在飞请求）：退出码 0");
        } else {
            println!("drain 超时：强制退出，未完成请求 {} 个：", left.len());
            for (engine_id, request_id) in &left {
                println!("  未完成 request_id={request_id}（引擎 #{engine_id}）");
            }
        }
        // 退出码 0：drain 超时也算"已尽力"——编排层看的是日志里的未完成清单
        std::process::exit(0);
    });
}

/// 等 SIGINT 或 SIGTERM（返回信号名）。
///
/// 旧实现只等 `ctrl_c()`：容器/编排层发 SIGTERM（docker stop、systemd）时
/// 进程被直接杀掉——在飞请求全断、退出码 143。现在两者同一路径：
/// 停止接纳 → drain → 退出码 0。
#[cfg(unix)]
async fn wait_for_signal() -> &'static str {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = match signal(SignalKind::terminate()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SIGTERM 处理注册失败（退回只接 SIGINT）：{e}");
            tokio::signal::ctrl_c().await.ok();
            return "SIGINT";
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => "SIGINT",
        _ = term.recv() => "SIGTERM",
    }
}

#[cfg(not(unix))]
async fn wait_for_signal() -> &'static str {
    tokio::signal::ctrl_c().await.ok();
    "SIGINT"
}
