//! wein-model 验收（q1-q4）——I1 竖切的加载侧（纯 CPU，无 GPU 门控，
//! 但需要模型目录在场：`WEIN_MODEL_DIR` 或 `~/.cache/wein/Qwen3.5-4B`，
//! 缺失时 SKIP——与 o/g/x 系列的门控同式）。
//!
//! - q1 · config 解析：Qwen3.5-4B 文本骨干字段与公开事实一致 + 形状自洽
//! - q2 · 权重目录：文本路径张量名/形状/dtype 齐（visual/mtp 不在必答题里）
//! - q3 · 权重健全性：抽查统计（有限性/量级/确定性）
//! - q4 · tokenizer：encode/decode 往返、特殊 token、词表边界
use wein_model::{model_dir, Dtype, HfTokenizer, Qwen35Config, WeightStore};

/// 所有用例共用的模型目录（无则 SKIP）
fn dir() -> Option<std::path::PathBuf> {
    match model_dir() {
        Some(d) => Some(d),
        None => {
            eprintln!("SKIP（模型目录不可用：设 WEIN_MODEL_DIR 或放 ~/.cache/wein/Qwen3.5-4B）");
            None
        }
    }
}

/// q1 · config 解析：公开事实（ModelScope Qwen/Qwen3.5-4B 的 text_config）
/// 与形状自洽校验一次钉死——加载器读错字段在这里现形
#[test]
fn q1_config_fields() {
    let Some(d) = dir() else { return };
    let cfg = Qwen35Config::load(&d).expect("config.json 解析");
    assert_eq!(cfg.hidden_size, 2560);
    assert_eq!(cfg.num_layers, 32);
    assert_eq!(cfg.num_heads, 16);
    assert_eq!(cfg.num_kv_heads, 4);
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(cfg.inter_size, 9216);
    assert_eq!(cfg.vocab_size, 248320);
    assert_eq!(cfg.rms_eps, 1e-6);
    assert_eq!(cfg.rope_theta, 1e7);
    assert_eq!(cfg.partial_rotary_factor, 0.25);
    assert!(cfg.tie_word_embeddings, "4B 词表与 lm_head 共享嵌入");
    assert_eq!(cfg.eos_token_id, 248044);
    // linear attention 侧
    assert_eq!(cfg.linear_num_v_heads, 32);
    assert_eq!(cfg.linear_num_k_heads, 16);
    assert_eq!(cfg.linear_k_dim, 128);
    assert_eq!(cfg.linear_v_dim, 128);
    assert_eq!(cfg.conv_kernel, 4);
    // 层型：every 4th is full_attention（full_attention_interval=4）
    let full: Vec<u32> =
        (0..cfg.num_layers).filter(|&i| cfg.layer_type(i) == wein_model::LayerType::Full).collect();
    assert_eq!(full, vec![3, 7, 11, 15, 19, 23, 27, 31], "8 个 full_attention 层，间隔 4");
    cfg.validate().expect("形状自洽");
    // 派生量
    assert_eq!(cfg.rot_dim(), 64, "partial rotary 0.25 × 256");
    assert_eq!(cfg.kv_group(), 4, "16 q / 4 kv");
    assert_eq!(cfg.linear_key_dim(), 2048, "16 k 头 × 128");
    assert_eq!(cfg.linear_value_dim(), 4096, "32 v 头 × 128");
    assert_eq!(cfg.conv_dim(), 8192, "2×2048 + 4096");
}

/// q2 · 权重目录：文本路径 426 张量名名齐、形状/dtype 与 config 自洽；
/// 按名取张量 + 缺名报错。索引到 shard 的路由必须对（index 一致性）
#[test]
fn q2_weights_catalog() {
    let Some(d) = dir() else { return };
    let cfg = Qwen35Config::load(&d).expect("config");
    let ws = WeightStore::open(&d).expect("权重目录打开");
    // 每层张量数：linear = 16（attn 10 + mlp 3 + norm 2 + input/post 已计）——按名断言
    for layer in 0..cfg.num_layers {
        let p = format!("model.language_model.layers.{layer}");
        let is_full = cfg.layer_type(layer) == wein_model::LayerType::Full;
        let attn: &[(&str, Vec<u32>)] = if is_full {
            &[
                ("self_attn.q_proj.weight", vec![cfg.num_heads * cfg.head_dim * 2, cfg.hidden_size]), // 2× = attn 输出门
                ("self_attn.k_proj.weight", vec![cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size]),
                ("self_attn.v_proj.weight", vec![cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size]),
                ("self_attn.o_proj.weight", vec![cfg.hidden_size, cfg.num_heads * cfg.head_dim]),
                ("self_attn.q_norm.weight", vec![cfg.head_dim]),
                ("self_attn.k_norm.weight", vec![cfg.head_dim]),
            ]
        } else {
            &[
                ("linear_attn.in_proj_qkv.weight", vec![cfg.conv_dim(), cfg.hidden_size]),
                ("linear_attn.in_proj_z.weight", vec![cfg.linear_value_dim(), cfg.hidden_size]),
                ("linear_attn.in_proj_a.weight", vec![cfg.linear_num_v_heads, cfg.hidden_size]),
                ("linear_attn.in_proj_b.weight", vec![cfg.linear_num_v_heads, cfg.hidden_size]),
                ("linear_attn.conv1d.weight", vec![cfg.conv_dim(), 1, cfg.conv_kernel]), // Conv1d [out, in/groups, k]
                ("linear_attn.A_log", vec![cfg.linear_num_v_heads]),       // F32
                ("linear_attn.dt_bias", vec![cfg.linear_num_v_heads]),    // BF16
                ("linear_attn.norm.weight", vec![cfg.linear_v_dim]),      // F32
                ("linear_attn.out_proj.weight", vec![cfg.hidden_size, cfg.linear_value_dim()]),
            ]
        };
        for (name, dims) in attn {
            let full = format!("{p}.{name}");
            let t = ws.tensor(&full).unwrap_or_else(|e| panic!("{full}: {e}"));
            assert_eq!(&t.shape, dims, "{full} 形状");
            // F32 只有 linear_attn.norm.weight 与 A_log；q_norm/k_norm 是 BF16
            let want = if name.ends_with("A_log") || *name == "linear_attn.norm.weight" {
                Dtype::F32
            } else {
                Dtype::BF16
            };
            assert_eq!(t.dtype, want, "{full} dtype");
        }
        for norm in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            let t = ws.tensor(&format!("{p}.{norm}")).expect(norm);
            assert_eq!(t.shape, &[cfg.hidden_size]);
        }
        for mlp in [
            ("mlp.gate_proj.weight", cfg.inter_size, cfg.hidden_size),
            ("mlp.up_proj.weight", cfg.inter_size, cfg.hidden_size),
            ("mlp.down_proj.weight", cfg.hidden_size, cfg.inter_size),
        ] {
            let t = ws.tensor(&format!("{p}.{}", mlp.0)).expect(mlp.0);
            assert_eq!(t.shape, &[mlp.1, mlp.2]);
        }
    }
    let embed = ws.tensor("model.language_model.embed_tokens.weight").expect("embed");
    assert_eq!(embed.shape, &[cfg.vocab_size, cfg.hidden_size]);
    assert_eq!(embed.dtype, Dtype::BF16);
    assert!(ws.tensor("model.language_model.norm.weight").is_ok(), "final norm");
    // 缺名报错（不是 panic，是 Err）
    assert!(ws.tensor("model.language_model.no_such.weight").is_err());
    // visual/mtp 在文件里但不在必答题——能查到也行，查不到不管（本测试只钉文本路径）
    eprintln!(
        "q2: 目录共 {} 张量，文本路径必答张量全齐",
        ws.names().len()
    );
}

/// q3 · 权重健全性：抽查统计——有限性、量级、确定性（两个独立 open 字节一致）。
/// 下载损坏/半截文件、名字张冠李戴在这里现形
#[test]
fn q3_weights_sanity() {
    let Some(d) = dir() else { return };
    let ws = WeightStore::open(&d).expect("open");
    let ws2 = WeightStore::open(&d).expect("second open");
    let bf16_stats = |name: &str| -> (f64, f64, f64) {
        let t = ws.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
        let v = t.bf16().unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut n = 0u64;
        let mut sum = 0f64;
        let mut abs_max = 0f64;
        for step in 0..v.len().max(1).min(100_000) {
            let idx = (step * 977) % v.len(); // 确定性抽样，覆盖远距离
            let x = wein_model::bf16_to_f32(v[idx]) as f64;
            assert!(x.is_finite(), "{name}[{idx}] 非有限值");
            n += 1;
            sum += x;
            abs_max = abs_max.max(x.abs());
        }
        (sum / n as f64, abs_max, n as f64)
    };
    let (mean, mx, _) = bf16_stats("model.language_model.embed_tokens.weight");
    assert!(mean.abs() < 0.1 && mx > 0.001 && mx < 1.0, "embed 统计异常：mean={mean} max={mx}");
    let (_, qmax, _) = bf16_stats("model.language_model.layers.3.self_attn.q_proj.weight");
    assert!(qmax > 0.001 && qmax < 1.0, "q_proj 量级异常：{qmax}");
    // A_log（F32）：init ~ U(0.01, 16) 的 log → [-4.6, 2.8]
    let a = ws
        .tensor("model.language_model.layers.0.linear_attn.A_log")
        .unwrap();
    let a = a.f32().unwrap();
    assert!(a.iter().all(|&x| x > -5.0 && x < 3.0 && x.is_finite()), "A_log 范围：{a:?}");
    // 零中心 RMSNorm 权重（init 0，训练后小）：input_layernorm 均值应近 0
    let w = ws
        .tensor("model.language_model.layers.0.input_layernorm.weight")
        .unwrap();
    let w = w.bf16().unwrap();
    let mean: f64 = w.iter().map(|&h| wein_model::bf16_to_f32(h) as f64).sum::<f64>() / w.len() as f64;
    // 零中心（init 0）训练后仍应是小量——0.5 是抓垃圾数据的下限不是精测
    assert!(mean.abs() < 0.5, "零中心 norm 权重量级异常：{mean}");
    // 确定性：两个独立 open 同张量字节一致
    let t1 = ws.tensor("model.language_model.layers.12.mlp.down_proj.weight").unwrap();
    let t2 = ws2.tensor("model.language_model.layers.12.mlp.down_proj.weight").unwrap();
    assert_eq!(t1.bytes, t2.bytes, "两次 open 的张量字节必须一致");
}

/// q4 · tokenizer：tokenizers crate 接线——encode 确定性、decode 往返、
/// 词表边界、eos。Qwen BPE（byte-level）
#[test]
fn q4_tokenizer_roundtrip() {
    let Some(d) = dir() else { return };
    let tok = HfTokenizer::open(&d).expect("tokenizer.json 加载");
    let texts = [
        "The capital of France is",
        "你好，世界！深度学习推理框架 wein。",
        "line1\nline2\twith  spaces  and !? punctuation...",
        "<|im_end|>",
    ];
    for t in texts {
        let ids = tok.encode(t);
        assert!(!ids.is_empty(), "{t:?} 编码非空");
        assert!(ids.iter().all(|&i| i < 248_320), "{t:?} 越词表：{ids:?}");
        assert_eq!(ids, tok.encode(t), "{t:?} 编码确定性");
        let back = tok.decode(&ids);
        // byte-level BPE 往返：特殊 token 除外应逐字节复原
        if t != "<|im_end|>" {
            assert_eq!(back, t, "{t:?} 往返：{back:?}");
        } else {
            assert_eq!(back, "<|im_end|>");
        }
    }
    // 双 eos：config 的 eos_token_id=248044 是 <|endoftext|>（训练侧），
    // chat 停止符是 <|im_end|>=248046——serve 层停用集用后者（这里钉死合同）
    assert_eq!(tok.eos_id(), 248_046, "chat eos = <|im_end|>");
    assert_eq!(tok.encode("<|im_end|>"), vec![248_046]);
    assert_eq!(tok.encode("<|endoftext|>"), vec![248_044]);
}
