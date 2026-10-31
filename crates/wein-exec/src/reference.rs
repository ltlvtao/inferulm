//! toy 模型的独立 CPU 镜像（x2 的对照端）。
//!
//! 数值纪律：与设备侧**同一份模型序**，每个算子边界的 bf16 存储逐位对齐
//! （设备侧 scratch 全 bf16），算子内部用 wein-kernel 的 golden 验证过的
//! reference 原语；GEMM 用 f64 累加（设备是 cuBLASLt TC f32 累加——容差的
//! 唯一来源，g2 准则在 exec 层的延伸）。attention 手写两遍法（与 kernel
//! 同序：f32 点积 → max → exp → 顺序累加 → 加权 V）——分页只是地址映射，
//! 镜像用线性 KV 缓存。
use wein_kernel::reference as kref;

use crate::toy::{ToyConfig, ToyWeights};

/// 前向整段 prompt，返回末位置 logits（bf16 存储 → f32）
pub fn toy_logits(prompt: &[u32], cfg: &ToyConfig) -> Vec<f32> {
    assert!(cfg.q_heads == cfg.kv_heads, "镜像与设备同约束（MHA）");
    let w = ToyWeights::synthesize(cfg);
    let (h, i, v) = (cfg.hidden as usize, cfg.inter as usize, cfg.vocab as usize);
    let hd = cfg.head_dim as usize;
    let heads = cfg.q_heads as usize;
    let (qe, ke) = (heads * hd, heads * hd);
    let n = prompt.len();
    let (cos, sin) = kref::rope_table(cfg.max_pos as usize, hd / 2, 10000.0);
    let scale = 1.0f32 / (hd as f32).sqrt();
    let mut k_cache = vec![0u16; n * ke];
    let mut v_cache = vec![0u16; n * ke];
    let mut last = Vec::new();
    for pos in 0..n {
        // embed → rms1 → {q,k,v}
        let x = kref::embed(&[prompt[pos]], &w.embed, h);
        let n1 = kref::rms_norm(&x, &w.rms1, 1, h, cfg.eps);
        let q = gemm_bf16(&n1, &w.wq, 1, qe, h);
        let k = gemm_bf16(&n1, &w.wk, 1, ke, h);
        let v_mat = gemm_bf16(&n1, &w.wv, 1, ke, h);
        // rope（每头一行；reference::rope 以 token 行为单位 → 头展开成行）
        let pos_rows = vec![pos as u32; heads];
        let (qr, kr) = kref::rope(&q, &k, &cos, &sin, &pos_rows, hd);
        k_cache[pos * ke..(pos + 1) * ke].copy_from_slice(&kr);
        v_cache[pos * ke..(pos + 1) * ke].copy_from_slice(&v_mat);
        // attention（两遍法，与 kernel 同序）
        let mut attn = vec![0u16; qe];
        for hq in 0..heads {
            let base = hq * hd; // MHA：kv 头 = q 头
            let mut sc = vec![0f32; pos + 1];
            for (j, s) in sc.iter_mut().enumerate() {
                let mut dot = 0f32;
                for d in 0..hd {
                    dot = kref::bf16_to_f32(qr[base + d])
                        .mul_add(kref::bf16_to_f32(k_cache[j * ke + base + d]), dot);
                }
                *s = dot * scale;
            }
            let m = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let p: Vec<f32> = sc.iter().map(|&s| kref::expf(s - m)).collect();
            let mut denom = 0f32;
            for &pj in &p {
                denom = pj + denom;
            }
            for e in 0..hd {
                let mut acc = 0f32;
                for (j, &pj) in p.iter().enumerate() {
                    acc = pj.mul_add(kref::bf16_to_f32(v_cache[j * ke + base + e]), acc);
                }
                attn[base + e] = kref::f32_to_bf16(acc / denom);
            }
        }
        // o 投影 → 残差 → rms2 → {gate,up} → silu×up → down → 残差 → lm_head
        let o = gemm_bf16(&attn, &w.wo, 1, h, qe);
        let h2 = kref::add(&x, &o);
        let n2 = kref::rms_norm(&h2, &w.rms2, 1, h, cfg.eps);
        let g = gemm_bf16(&n2, &w.wg, 1, i, h);
        let u = gemm_bf16(&n2, &w.wu, 1, i, h);
        let act = kref::silu(&g);
        let mm = kref::mul(&act, &u);
        let d = gemm_bf16(&mm, &w.wd, 1, h, i);
        let h3 = kref::add(&h2, &d);
        last = gemm_bf16(&h3, &w.lm, 1, v, h);
    }
    last.iter().map(|&b| kref::bf16_to_f32(b)).collect()
}

/// bf16 入出 GEMM：f64 累加（reference::gemm）→ RNE 收 bf16——
/// 与设备侧 scratch 的存储精度逐位对齐
fn gemm_bf16(a: &[u16], b: &[u16], m: usize, n: usize, k: usize) -> Vec<u16> {
    kref::gemm(a, b, m, n, k).iter().map(|&x| kref::f32_to_bf16(x as f32)).collect()
}
