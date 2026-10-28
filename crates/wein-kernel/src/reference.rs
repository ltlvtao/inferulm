//! CPU 参考实现——golden vector 的生成端与对拍基准（§4 数值对齐）。
//!
//! 与 `kernels/` 的设备实现是**独立代码路径**（不同 crate、不同代码），
//! 只共享数学定义；任何一侧漂移都会在 golden 对拍上现形——这正是
//! golden vector 框架的价值。cuda/asc 同名算子对拍同一套向量（§4）。
//!
//! 对齐纪律（两侧同守）：bf16→f32 精确（位左移）；乘/加/除/sqrt 全部
//! 正确舍入原语（IEEE 单算），禁 rsqrt 类近似与 fma 收缩；行内顺序累加。

/// bf16 → f32：精确（尾数位左移 16，无舍入）
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// f32 → bf16：round-to-nearest-even（与 kernels/ 同公式独立实现）
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits + 0x7fff + lsb) >> 16) as u16
}

/// splitmix64 → [-1, 1) 的确定性 f32（golden 输入生成，无 rand 依赖）。
/// [2,4) 均匀取尾数再平移——冒烟 acceptance.rs 同源，位上可复现
pub fn splitmix_f32(seed: u64) -> f32 {
    let z = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut x = z ^ (z >> 30);
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x = x.wrapping_mul(0x94d049bb133111eb);
    let u = (x >> 40) as u32; // 24 bit 尾数域
    f32::from_bits((u & 0xffffff) | 0x40000000) - 3.0 // [-1, 1)
}

/// RMSNorm（Llama/Qwen 形态）：out = x / sqrt(mean(x²) + eps) * w
/// 与 kernels::rms_norm 逐步同序：acc = v·v + acc；inv = 1/((acc/h+eps)^½)；
/// out = (x·inv)·w——顺序即精度，两侧一字不差
pub fn rms_norm(x: &[u16], w: &[u16], rows: usize, hidden: usize, eps: f32) -> Vec<u16> {
    let mut out = vec![0u16; rows * hidden];
    for r in 0..rows {
        let base = r * hidden;
        let mut acc = 0.0f32;
        for i in 0..hidden {
            let v = bf16_to_f32(x[base + i]);
            acc = v * v + acc; // 顺序累加，与 kernel 同向
        }
        let inv = 1.0f32 / ((acc / hidden as f32 + eps).sqrt());
        for i in 0..hidden {
            let o = bf16_to_f32(x[base + i]) * inv * bf16_to_f32(w[i]); // 左结合
            out[base + i] = f32_to_bf16(o);
        }
    }
    out
}

/// exp 的自控路径（silu/softmax 共用）：x·log2e → floor 取 k、f∈[0,1) →
/// 2^f 七次 Horner 多项式（截断误差 ≤1.4e-6 ≪ bf16 分辨率）→ 指数域加 k。
/// 禁 MUFU ex2 近似——两侧同字面量同 Horner 序才可达位对齐（A 计划，
/// 与 bf16 软实现同理：数值路径完全自控，不指望工具链 intrinsic）
pub fn expf(x: f32) -> f32 {
    const LOG2E: f32 = 1.4426950408889634;
    // a_n = ln2^n / n!
    const A7: f32 = 1.5252733804059838e-05;
    const A6: f32 = 0.00015403530393381606;
    const A5: f32 = 0.0013333558146428441;
    const A4: f32 = 0.009618129107628477;
    const A3: f32 = 0.055504108664821576;
    const A2: f32 = 0.2402265069591007;
    const A1: f32 = 0.6931471805599453;
    const A0: f32 = 1.0;
    let t = x * LOG2E;
    let kf = t.floor();
    let f = t - kf; // [0,1)，Sterbenz 精确
    let mut p = A7;
    p = p * f + A6;
    p = p * f + A5;
    p = p * f + A4;
    p = p * f + A3;
    p = p * f + A2;
    p = p * f + A1;
    p = p * f + A0;
    // 2^k 缩放：指数域直接加（尾数不动——精确）
    f32::from_bits(((p.to_bits() as i32).wrapping_add((kf as i32) << 23)) as u32)
}

/// rope cos/sin 表（host 生成，作为算子**输入**上传——图捕获友好）。
/// 频率在 f64 域算（表精度是一次性代价），存 f32
pub fn rope_table(max_pos: usize, rot_pairs: usize, base: f64) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0f32; max_pos * rot_pairs];
    let mut sin = vec![0f32; max_pos * rot_pairs];
    for p in 0..rot_pairs {
        let inv_freq = base.powf(-(p as f64) / rot_pairs as f64);
        for t in 0..max_pos {
            let (a, s) = ((t as f64) * inv_freq, (t as f64) * inv_freq);
            cos[t * rot_pairs + p] = a.cos() as f32;
            sin[t * rot_pairs + p] = s.sin() as f32;
        }
    }
    (cos, sin)
}

/// rope apply（half-split，Llama/Qwen 形态）：x1=前半、x2=后半，
/// out = (x1·c − x2·s, x2·c + x1·s)——与 kernel 同式同序
pub fn rope(
    q: &[u16],
    k: &[u16],
    cos: &[f32],
    sin: &[f32],
    pos: &[u32],
    rot_dim: usize,
) -> (Vec<u16>, Vec<u16>) {
    let half = rot_dim / 2;
    let mut oq = vec![0u16; q.len()];
    let mut ok = vec![0u16; k.len()];
    for t in 0..pos.len() {
        let ti = pos[t] as usize;
        for p in 0..half {
            let (c, s) = (cos[ti * half + p], sin[ti * half + p]);
            let (x1, x2) = (bf16_to_f32(q[t * rot_dim + p]), bf16_to_f32(q[t * rot_dim + half + p]));
            oq[t * rot_dim + p] = f32_to_bf16(x1 * c - x2 * s);
            oq[t * rot_dim + half + p] = f32_to_bf16(x2 * c + x1 * s);
            let (x1, x2) = (bf16_to_f32(k[t * rot_dim + p]), bf16_to_f32(k[t * rot_dim + half + p]));
            ok[t * rot_dim + p] = f32_to_bf16(x1 * c - x2 * s);
            ok[t * rot_dim + half + p] = f32_to_bf16(x2 * c + x1 * s);
        }
    }
    (oq, ok)
}

/// silu：out = x·sigmoid(x)，sigmoid = 1/(1+expf(−x))——与 kernel 同式
pub fn silu(x: &[u16]) -> Vec<u16> {
    x.iter()
        .map(|v| {
            let v = bf16_to_f32(*v);
            f32_to_bf16(v * (1.0f32 / (1.0f32 + expf(-v))))
        })
        .collect()
}

/// RNE 半偶取整到整值 f32（自控实现——不用 round()，默认就近偶要自己钉）
fn rne(x: f32) -> f32 {
    let fl = x.floor();
    let d = x - fl;
    if d > 0.5 {
        fl + 1.0
    } else if d < 0.5 {
        fl
    } else if (fl.to_bits() & 1) == 0 {
        fl
    } else {
        fl + 1.0
    }
}

/// W8A8 动态量化：逐 token amax → scale = amax/127 → q = clamp(RNE(x/scale))。
/// 除法走 div（正确舍入），W2 若换倒数乘 golden 须显式重生成
pub fn w8a8_quant(x: &[u16], tokens: usize, hidden: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; tokens * hidden];
    let mut scale = vec![0f32; tokens];
    for t in 0..tokens {
        let base = t * hidden;
        let mut amax = 0.0f32;
        for i in 0..hidden {
            let a = bf16_to_f32(x[base + i]).abs();
            amax = if a > amax { a } else { amax };
        }
        if amax == 0.0 {
            continue; // scale 0、q 0
        }
        scale[t] = amax / 127.0;
        for i in 0..hidden {
            let r = rne(bf16_to_f32(x[base + i]) / scale[t]);
            let r = if r > 127.0 { 127.0 } else if r < -127.0 { -127.0 } else { r };
            q[base + i] = r as i8;
        }
    }
    (q, scale)
}

/// KV scatter write 参考实现：返回整个池（未触及槽位 0）——对拍含"没写坏
/// 别人的槽"这一半语义
pub fn kv_scatter(
    src_k: &[u16],
    src_v: &[u16],
    block_table: &[u32],
    seq_ids: &[u32],
    positions: &[u32],
    slots: usize,
    bpr: usize,
    block_size: usize,
    head_elems: usize,
) -> (Vec<u16>, Vec<u16>) {
    let mut k = vec![0u16; slots * head_elems];
    let mut v = vec![0u16; slots * head_elems];
    for t in 0..seq_ids.len() {
        let blk = (positions[t] / block_size as u32) as usize;
        let off = (positions[t] % block_size as u32) as usize;
        let slot = block_table[seq_ids[t] as usize * bpr + blk] as usize * block_size + off;
        for e in 0..head_elems {
            k[slot * head_elems + e] = src_k[t * head_elems + e];
            v[slot * head_elems + e] = src_v[t * head_elems + e];
        }
    }
    (k, v)
}

/// varlen paged attention 参考（correctness-first 两遍法，与 kernel 同序）：
/// pass1 记分取 max；pass2 exp(score−m) 顺序累加分母与加权 V；
/// GQA：kv 头 = qh % kv_heads；因果：token 只看 0..=pos
pub fn paged_attn(
    q: &[u16],
    k_cache: &[u16],
    v_cache: &[u16],
    block_table: &[u32],
    seq_ids: &[u32],
    positions: &[u32],
    q_scale: f32,
    bpr: usize,
    block_size: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Vec<u16> {
    let he = kv_heads * head_dim;
    let mut out = vec![0u16; positions.len() * q_heads * head_dim];
    for t in 0..positions.len() {
        let (s, pos) = (seq_ids[t] as usize, positions[t] as usize);
        for qh in 0..q_heads {
            let kh = qh % kv_heads;
            // pass1：score[j]（j 递增同序）与 max
            let mut sc = vec![0f32; pos + 1];
            let mut m = f32::NEG_INFINITY;
            for j in 0..=pos {
                let blk = j / block_size;
                let off = j % block_size;
                let slot = block_table[s * bpr + blk] as usize * block_size + off;
                let kbase = slot * he + kh * head_dim;
                let qbase = (t * q_heads + qh) * head_dim;
                let mut dot = 0.0f32;
                for i in 0..head_dim {
                    let a = bf16_to_f32(q[qbase + i]);
                    let b = bf16_to_f32(k_cache[kbase + i]);
                    dot = a * b + dot;
                }
                sc[j] = dot * q_scale;
                m = if sc[j] > m { sc[j] } else { m };
            }
            // pass2：p_j = expf(sc_j − m)，分母与加权 V 顺序累加
            let mut p = vec![0f32; pos + 1];
            let mut denom = 0.0f32;
            for j in 0..=pos {
                p[j] = expf(sc[j] - m);
                denom = p[j] + denom;
            }
            for e in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..=pos {
                    let blk = j / block_size;
                    let off = j % block_size;
                    let slot = block_table[s * bpr + blk] as usize * block_size + off;
                    let vbase = slot * he + kh * head_dim;
                    acc = p[j] * bf16_to_f32(v_cache[vbase + e]) + acc;
                }
                out[(t * q_heads + qh) * head_dim + e] = f32_to_bf16(acc / denom);
            }
        }
    }
    out
}

/// GEMM 参考（f64 累加）：容差对齐的基准端（厂商库算子不做位对齐——
/// 累加序在库内，合同是 ±1 bf16 ulp，g2 准则）
pub fn gemm(a: &[u16], b: &[u16], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0f64; m * n];
    for r in 0..m {
        for col in 0..n {
            let mut acc = 0f64;
            for i in 0..k {
                acc += bf16_to_f32(a[r * k + i]) as f64 * bf16_to_f32(b[i * n + col]) as f64;
            }
            c[r * n + col] = acc;
        }
    }
    c
}

/// embed gather 参考：out[t·hidden+e] = w[id[t]·hidden+e]（位拷贝，与 kernel 同式）
pub fn embed(ids: &[u32], w: &[u16], hidden: usize) -> Vec<u16> {
    let mut out = vec![0u16; ids.len() * hidden];
    for (t, &id) in ids.iter().enumerate() {
        out[t * hidden..(t + 1) * hidden]
            .copy_from_slice(&w[id as usize * hidden..][..hidden]);
    }
    out
}

/// bf16 加参考（残差）：f32 域加法 RNE 收 bf16
pub fn add(a: &[u16], b: &[u16]) -> Vec<u16> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f32_to_bf16(bf16_to_f32(x) + bf16_to_f32(y)))
        .collect()
}

/// bf16 乘参考（门控）：f32 域乘法 RNE 收 bf16
pub fn mul(a: &[u16], b: &[u16]) -> Vec<u16> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f32_to_bf16(bf16_to_f32(x) * bf16_to_f32(y)))
        .collect()
}
