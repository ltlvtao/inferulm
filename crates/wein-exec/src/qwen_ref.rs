//! Qwen3.5-4B 文本骨干的独立 CPU 镜像（q5 对拍对照端）。
//!
//! 与设备侧（qwen.rs）同一数值合同、独立实现路径：kref 原语（norm/rope/
//! sigmul/conv/deltanet——与 kernel 位对齐的 CPU 版）+ f64 gemm_wt 逐边界
//! 收 bf16 + 手写 eager 两遍 GQA attention（scores/权重全程 f32，同
//! paged_attn 的舍入位）。逐 token 递推：全注意力层缓存 K/V，线性层持
//! 递推状态与 conv 状态——与 exec 的批语义在段内逐步等价（conv/deltanet
//! 的段递推本身按 token 序展开）。
//!
//! 成本 ~秒级/token（f64 标量 GEMM 4.2G MAC）——q5 逐位对拍的一次性代价。
use std::path::Path;

use wein_kernel::reference as kref;
use wein_model::{LayerType, Qwen35Config, WeightStore};

const MAX_POS: usize = 256;
const L2_EPS: f32 = 1e-6;

struct FullHost {
    qg: Vec<u16>,  // [8192, 2560]
    k: Vec<u16>,   // [1024, 2560]
    v: Vec<u16>,   // [1024, 2560]
    o: Vec<u16>,   // [2560, 4096]
    qn: Vec<u16>,  // [256]
    kn: Vec<u16>,  // [256]
}

struct LinHost {
    qkv: Vec<u16>,   // [8192, 2560]
    z: Vec<u16>,     // [4096, 2560]
    a: Vec<u16>,     // [32, 2560]
    b: Vec<u16>,     // [32, 2560]
    conv: Vec<u16>,  // [8192, 4]
    a_log: Vec<f32>, // [32]
    dtb: Vec<u16>,   // [32]
    norm: Vec<f32>,  // [128]
    out: Vec<u16>,   // [2560, 4096]
}

enum AttnHost {
    Full(FullHost),
    Linear(LinHost),
}

struct LayerHost {
    ln1: Vec<u16>,
    attn: AttnHost,
    ln2: Vec<u16>,
    wg: Vec<u16>,
    wu: Vec<u16>,
    wd: Vec<u16>,
}

pub struct QwenMirror {
    cfg: Qwen35Config,
    embed: Vec<u16>,
    norm_f: Vec<u16>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    layers: Vec<LayerHost>,
    // 运行态（逐 token 递推）
    pos: u32,
    /// 每全注意力层的 K/V 缓存 [t, kv_heads*head_dim]（逐 token 追加）
    k_cache: Vec<Vec<u16>>,
    v_cache: Vec<Vec<u16>>,
    /// 每线性层递推状态 [v_heads, v_dim, k_dim] f32
    lin_state: Vec<Vec<u16>>, // 状态池 bf16（与设备侧同存储格式）
    /// 每线性层 conv 状态 [conv_dim, kernel-1]
    conv_state: Vec<Vec<u16>>,
}

/// f64 gemm → bf16（设备侧 cuBLASLt C 也是 bf16 落盘——边界舍入位对齐）
fn gemm_bf16(a: &[u16], w: &[u16], m: usize, n: usize, k: usize) -> Vec<u16> {
    kref::gemm_wt(a, w, m, n, k)
        .iter()
        .map(|&x| kref::f32_to_bf16(x as f32))
        .collect()
}

impl QwenMirror {
    pub fn open(dir: &Path) -> Result<Self, String> {
        let cfg = Qwen35Config::load(dir).map_err(|e| format!("config：{e}"))?;
        cfg.validate().map_err(|e| format!("config：{e}"))?;
        let store = WeightStore::open(dir).map_err(|e| format!("权重：{e}"))?;
        const P: &str = "model.language_model.";
        let bits = |name: &str| -> Result<Vec<u16>, String> {
            let t = store.tensor(name).map_err(|e| format!("{e}"))?;
            t.bf16().map(|s| s.to_vec()).map_err(|e| format!("{name}：{e}"))
        };
        let f32s = |name: &str| -> Result<Vec<f32>, String> {
            let t = store.tensor(name).map_err(|e| format!("{e}"))?;
            t.f32().map(|s| s.to_vec()).map_err(|e| format!("{name}：{e}"))
        };
        let mut layers = Vec::with_capacity(cfg.num_layers as usize);
        let mut k_cache = Vec::new();
        let mut v_cache = Vec::new();
        let mut lin_state = Vec::new();
        let mut conv_state = Vec::new();
        for i in 0..cfg.num_layers {
            let base = format!("{P}layers.{i}.");
            let attn = match cfg.layer_type(i) {
                LayerType::Full => {
                    k_cache.push(Vec::new());
                    v_cache.push(Vec::new());
                    AttnHost::Full(FullHost {
                        qg: bits(&format!("{base}self_attn.q_proj.weight"))?,
                        k: bits(&format!("{base}self_attn.k_proj.weight"))?,
                        v: bits(&format!("{base}self_attn.v_proj.weight"))?,
                        o: bits(&format!("{base}self_attn.o_proj.weight"))?,
                        qn: bits(&format!("{base}self_attn.q_norm.weight"))?,
                        kn: bits(&format!("{base}self_attn.k_norm.weight"))?,
                    })
                }
                LayerType::Linear => {
                    lin_state.push(vec![0u16; (cfg.linear_num_v_heads * cfg.linear_v_dim
                        * cfg.linear_k_dim) as usize]);
                    conv_state.push(vec![0u16; (cfg.conv_dim() * (cfg.conv_kernel - 1)) as usize]);
                    AttnHost::Linear(LinHost {
                        qkv: bits(&format!("{base}linear_attn.in_proj_qkv.weight"))?,
                        z: bits(&format!("{base}linear_attn.in_proj_z.weight"))?,
                        a: bits(&format!("{base}linear_attn.in_proj_a.weight"))?,
                        b: bits(&format!("{base}linear_attn.in_proj_b.weight"))?,
                        conv: bits(&format!("{base}linear_attn.conv1d.weight"))?,
                        a_log: f32s(&format!("{base}linear_attn.A_log"))?,
                        dtb: bits(&format!("{base}linear_attn.dt_bias"))?,
                        norm: f32s(&format!("{base}linear_attn.norm.weight"))?,
                        out: bits(&format!("{base}linear_attn.out_proj.weight"))?,
                    })
                }
            };
            layers.push(LayerHost {
                ln1: bits(&format!("{base}input_layernorm.weight"))?,
                attn,
                ln2: bits(&format!("{base}post_attention_layernorm.weight"))?,
                wg: bits(&format!("{base}mlp.gate_proj.weight"))?,
                wu: bits(&format!("{base}mlp.up_proj.weight"))?,
                wd: bits(&format!("{base}mlp.down_proj.weight"))?,
            });
        }
        // rope 表：与 executor 同构造（收 bf16 再回展——HF .to(x.dtype)）
        let rot_half = (cfg.rot_dim() / 2) as usize;
        let (mut cos, mut sin) = kref::rope_table(MAX_POS, rot_half, cfg.rope_theta);
        for v in cos.iter_mut().chain(sin.iter_mut()) {
            *v = kref::bf16_to_f32(kref::f32_to_bf16(*v));
        }
        Ok(QwenMirror {
            embed: bits(&format!("{P}embed_tokens.weight"))?,
            norm_f: bits(&format!("{P}norm.weight"))?,
            cos,
            sin,
            layers,
            cfg,
            pos: 0,
            k_cache,
            v_cache,
            lin_state,
            conv_state,
        })
    }

    /// 单 token 前向（状态推进），返回该位全词表 logits（bf16 → f32）
    pub fn forward_token(&mut self, token: u32) -> Vec<f32> {
        let cfg = &self.cfg;
        let (h, ii) = (cfg.hidden_size as usize, cfg.inter_size as usize);
        let eps = cfg.rms_eps;
        let pos = self.pos as usize;
        assert!(pos < MAX_POS, "镜像超 MAX_POS");
        let mut x = kref::embed(&[token], &self.embed, h);
        let (mut fi, mut li) = (0usize, 0usize);
        for layer in &self.layers {
            let n1 = kref::rms_norm_zc(&x, &layer.ln1, 1, h, eps);
            let o = match &layer.attn {
                AttnHost::Full(w) => {
                    let (qh_n, kvh, hd) =
                        (cfg.num_heads as usize, cfg.num_kv_heads as usize, cfg.head_dim as usize);
                    let qg = gemm_bf16(&n1, &w.qg, 1, 2 * qh_n * hd, h);
                    // q/gate 按头交错拆（头内前 256 q、后 256 gate）
                    let mut q = vec![0u16; qh_n * hd];
                    let mut gate = vec![0u16; qh_n * hd];
                    for hh in 0..qh_n {
                        let (qb, gb) = (hh * hd, hh * 2 * hd);
                        q[qb..qb + hd].copy_from_slice(&qg[gb..gb + hd]);
                        gate[qb..qb + hd].copy_from_slice(&qg[gb + hd..gb + 2 * hd]);
                    }
                    let q = kref::rms_norm_zc(&q, &w.qn, qh_n, hd, eps);
                    let k = gemm_bf16(&n1, &w.k, 1, kvh * hd, h);
                    let v = gemm_bf16(&n1, &w.v, 1, kvh * hd, h);
                    let k = kref::rms_norm_zc(&k, &w.kn, kvh, hd, eps);
                    let q = kref::rope_pt(&q, &self.cos, &self.sin, &[pos as u32], cfg.rot_dim() as usize, hd, qh_n);
                    let k = kref::rope_pt(&k, &self.cos, &self.sin, &[pos as u32], cfg.rot_dim() as usize, hd, kvh);
                    self.k_cache[fi].extend_from_slice(&k);
                    self.v_cache[fi].extend_from_slice(&v);
                    // eager 两遍 GQA（scores/权重 f32 不落 bf16——设备 paged_attn 同位）
                    let q_scale = 1.0f32 / (hd as f32).sqrt();
                    let mut attn = vec![0u16; qh_n * hd];
                    for qh in 0..qh_n {
                        let kh = qh / (qh_n / kvh); // GQA：kv 头 = qh // group
                        let mut scores = vec![0f32; pos + 1];
                        let mut m = f32::NEG_INFINITY;
                        for j in 0..=pos {
                            let kb = j * kvh * hd + kh * hd;
                            let mut dot = 0.0f32;
                            for d in 0..hd {
                                dot = kref::bf16_to_f32(q[qh * hd + d])
                                    * kref::bf16_to_f32(self.k_cache[fi][kb + d])
                                    + dot;
                            }
                            scores[j] = dot * q_scale;
                            if scores[j] > m {
                                m = scores[j];
                            }
                        }
                        let mut den = 0.0f32;
                        for s in scores.iter_mut() {
                            *s = kref::expf(*s - m);
                            den = *s + den;
                        }
                        for e in 0..hd {
                            let mut acc = 0.0f32;
                            for j in 0..=pos {
                                let vb = j * kvh * hd + kh * hd;
                                acc = scores[j] * kref::bf16_to_f32(self.v_cache[fi][vb + e]) + acc;
                            }
                            attn[qh * hd + e] = kref::f32_to_bf16(acc / den);
                        }
                    }
                    let agated = kref::sigmul(&attn, &gate);
                    fi += 1;
                    gemm_bf16(&agated, &w.o, 1, h, qh_n * hd)
                }
                AttnHost::Linear(w) => {
                    let (kh_n, vh_n, kd, vd) = (
                        cfg.linear_num_k_heads as usize,
                        cfg.linear_num_v_heads as usize,
                        cfg.linear_k_dim as usize,
                        cfg.linear_v_dim as usize,
                    );
                    let (keyd, vald) = (kh_n * kd, vh_n * vd);
                    let cd = 2 * keyd + vald;
                    let qkv = gemm_bf16(&n1, &w.qkv, 1, cd, h);
                    // conv 单步 + 状态滚动（state 行 0：镜像单请求；状态存原始 qkv）
                    let convd =
                        kref::conv1d_dw(&qkv, &w.conv, &self.conv_state[li], &[0], &[0], cd, 4);
                    self.conv_state[li] =
                        kref::conv_state_roll(&qkv, &self.conv_state[li], &[0], &[1], &[0], cd, 4, &[0]);
                    let q = convd[..keyd].to_vec();
                    let k = convd[keyd..2 * keyd].to_vec();
                    let v = convd[2 * keyd..].to_vec();
                    let z = gemm_bf16(&n1, &w.z, 1, vald, h);
                    let a = gemm_bf16(&n1, &w.a, 1, vh_n, h);
                    let b = gemm_bf16(&n1, &w.b, 1, vh_n, h);
                    let core = kref::deltanet(
                        &q, &k, &v, &a, &b, &w.a_log, &w.dtb, &mut self.lin_state[li], &[0], &[1],
                        &[0], kh_n, vh_n, kd, vd, L2_EPS, &[0],
                    );
                    let normed = kref::rms_norm_gated(&core, &z, &w.norm, vh_n, vd, eps);
                    li += 1;
                    gemm_bf16(&normed, &w.out, 1, h, vald)
                }
            };
            // 残差 + MLP
            let h2 = kref::add(&x, &o);
            let n2 = kref::rms_norm_zc(&h2, &layer.ln2, 1, h, eps);
            let g = gemm_bf16(&n2, &layer.wg, 1, ii, h);
            let u = gemm_bf16(&n2, &layer.wu, 1, ii, h);
            let act = kref::silu(&g);
            let m = kref::mul(&act, &u);
            let d = gemm_bf16(&m, &layer.wd, 1, h, ii);
            x = kref::add(&h2, &d);
        }
        let xn = kref::rms_norm_zc(&x, &self.norm_f, 1, h, eps);
        let logits = gemm_bf16(&xn, &self.embed, 1, cfg.vocab_size as usize, h);
        self.pos += 1;
        logits.iter().map(|&b| kref::bf16_to_f32(b)).collect()
    }
}
