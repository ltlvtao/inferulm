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
    // 域钳（与 kernels::expf 镜像同式）：指数域直加只在 kf ∈ [-126,127]
    // 成立，域外回绕出垃圾且 CPU/GPU 的垃圾不一致（NaN payload 异源）——
    // kf < -126 落 0.0、kf > 127 落 +inf（vLLM/torch 同口径）
    if kf < -126.0 {
        return 0.0;
    }
    if kf > 127.0 {
        return f32::INFINITY;
    }
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
/// GQA：kv 头 = qh // group（group = q_heads/kv_heads，HF repeat_kv 语义）；因果：token 只看 0..=pos
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
            let kh = qh / (q_heads / kv_heads); // GQA：kv 头 = qh // group
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

// ───────────────────── I1 竖切镜像（Qwen3.5 文本骨干） ─────────────────────
// 与 kernels/ 新增 8 入口逐一同序同字面量。log/softplus 是 deltanet 门控链
// 的对数域原语——位对齐纪律延伸进来。

/// log2 自控路径（与 kernels::log2f 同式）：位分解 → √2 折半 → atanh 级数
/// 截到 r¹¹ → 2/ln2 缩放。输入域 [1, 5e8]（softplus 喂的 1+e^x）
pub fn log2f(z: f32) -> f32 {
    const SQRT2: f32 = 1.4142135;
    const LN2: f32 = 0.6931471805599453;
    let bits = z.to_bits();
    let e = ((bits >> 23) as i32) - 127;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    let mut e = e;
    if m > SQRT2 {
        m = m * 0.5;
        e += 1;
    }
    let r = (m - 1.0) / (m + 1.0);
    let r2 = r * r;
    let atanh = r
        * (1.0 + r2 * (1.0 / 3.0 + r2 * (1.0 / 5.0 + r2 * (1.0 / 7.0 + r2 * (1.0 / 9.0 + r2 * (1.0 / 11.0))))));
    e as f32 + atanh * (2.0 / LN2)
}

/// softplus：x>20 直通（torch 同阈），否则 ln(1+e^x) = log2(1+e^x)·ln2
pub fn softplus(x: f32) -> f32 {
    const LN2: f32 = 0.6931471805599453;
    if x > 20.0 {
        x
    } else {
        log2f(1.0 + expf(x)) * LN2
    }
}

/// 行平方和块归约镜像（与 kernels::block_sum_strided 同式同序）：
/// block 宽 bdim = min(hidden,256).next_power_of_two()（host 同式），
/// "线程" t 跨步占有 {t, t+bdim, ...} 部分和（i 升序，`acc = v·v + acc`）
/// → 线程 0 顺序合并 u 升序 `total = s[u] + total`。GPU 块归约的确定性
/// 序在标量域逐步复现——kernel↔参考位对齐保持（序 ≠ 单线程顺序累加，
/// 但两侧一致）
fn rms_block_sum(x: &[u16], base: usize, hidden: usize) -> f32 {
    let bdim = hidden.min(256).next_power_of_two();
    let mut s = vec![0f32; bdim];
    for t in 0..bdim {
        let mut acc = 0.0f32;
        let mut i = t;
        while i < hidden {
            let v = bf16_to_f32(x[base + i]);
            acc = v * v + acc;
            i += bdim;
        }
        s[t] = acc;
    }
    let mut total = 0.0f32;
    for u in 0..bdim {
        total = s[u] + total;
    }
    total
}

/// 零中心 RMSNorm（Qwen3_5RMSNorm 同语义）：out = x·rsqrt(mean(x²)+eps)·(1+w)，
/// f32 域单次收 bf16——与 kernel 逐步同序（块归约序同 rms_block_sum）
pub fn rms_norm_zc(x: &[u16], w: &[u16], rows: usize, hidden: usize, eps: f32) -> Vec<u16> {
    let mut out = vec![0u16; rows * hidden];
    for r in 0..rows {
        let base = r * hidden;
        let acc = rms_block_sum(x, base, hidden);
        let inv = 1.0f32 / ((acc / hidden as f32 + eps).sqrt());
        for i in 0..hidden {
            let o = bf16_to_f32(x[base + i]) * inv * (1.0 + bf16_to_f32(w[i]));
            out[base + i] = f32_to_bf16(o);
        }
    }
    out
}

/// partial RoPE（单张量 [tokens, heads, head_dim]，每 token 一 pos）：
/// 前 rot_dim 维 rotate_half 旋转、其余不动——与 kernel 同式同序。
/// cos/sin 表输入（表构造见 rope_table；注意 HF 把表收 bf16 再用——
/// executor 上载前应先收 bf16 再回展 f32，两侧共用同一张表即对齐）
pub fn rope_pt(
    x: &[u16],
    cos: &[f32],
    sin: &[f32],
    pos: &[u32],
    rot_dim: usize,
    head_dim: usize,
    heads: usize,
) -> Vec<u16> {
    let half = rot_dim / 2;
    let mut out = x.to_vec();
    for t in 0..pos.len() {
        let ti = pos[t] as usize;
        for h in 0..heads {
            let row = t * heads * head_dim + h * head_dim;
            for p in 0..half {
                let (c, s) = (cos[ti * half + p], sin[ti * half + p]);
                let (x1, x2) = (bf16_to_f32(x[row + p]), bf16_to_f32(x[row + half + p]));
                out[row + p] = f32_to_bf16(x1 * c - x2 * s);
                out[row + half + p] = f32_to_bf16(x2 * c + x1 * s);
            }
        }
    }
    out
}

/// 门控乘：out = x·sigmoid(g)——sigmoid 先收 bf16（HF 同两跳 round），与 kernel 同式
pub fn sigmul(x: &[u16], g: &[u16]) -> Vec<u16> {
    x.iter()
        .zip(g)
        .map(|(&v, &s)| {
            let (v, s) = (bf16_to_f32(v), bf16_to_f32(s));
            let sig = f32_to_bf16(1.0f32 / (1.0f32 + expf(-s)));
            f32_to_bf16(v * bf16_to_f32(sig))
        })
        .collect()
}

/// 深度wise 因果 conv1d + silu（两段 round：conv 收 bf16、激活再收）。
/// tap 序 k 升序与 kernel 同；不足 3 步取 state_in——与 kernel 同址同序
pub fn conv1d_dw(
    x: &[u16],             // [tokens, conv_dim]
    w: &[u16],             // [conv_dim, kernel]
    state_in: &[u16],      // [rows, conv_dim, kernel-1]
    tok_row: &[u32],       // [tokens]
    tok_pos: &[u32],       // [tokens]
    conv_dim: usize,
    kernel: usize,
) -> Vec<u16> {
    let st = kernel - 1;
    let mut out = vec![0u16; tok_row.len() * conv_dim];
    for g in 0..tok_row.len() {
        let (row, p) = (tok_row[g] as usize, tok_pos[g] as i32);
        for c in 0..conv_dim {
            let mut acc = 0.0f32;
            for k in 0..kernel {
                let idx = p - st as i32 + k as i32;
                let v = if idx < 0 {
                    bf16_to_f32(state_in[row * conv_dim * st + c * st + (idx + st as i32) as usize])
                } else {
                    bf16_to_f32(x[((g as i32 - p + idx) as usize) * conv_dim + c])
                };
                acc = v * bf16_to_f32(w[c * kernel + k]) + acc;
            }
            let conv = f32_to_bf16(acc);
            let v = bf16_to_f32(conv);
            out[g * conv_dim + c] = f32_to_bf16(v * (1.0f32 / (1.0f32 + expf(-v))));
        }
    }
    out
}

/// conv 状态滚动：new = (旧 state ++ 本段 x) 的末 kernel-1 个，span r 落在
/// **row_id[r]** 段（状态池按请求行分账）。返回新状态（kernel 原地——
/// 读槽位恒未被本线程写过，语义等价）
pub fn conv_state_roll(
    x: &[u16],
    state: &[u16],
    row_start: &[u32],
    row_count: &[u32],
    row_id: &[u32],
    conv_dim: usize,
    kernel: usize,
    specs: &[u32],
) -> Vec<u16> {
    let st = kernel - 1;
    assert_eq!(specs.len(), row_start.len(), "specs 须与段数同长（逐 span 标志）");
    let mut out = state.to_vec();
    for r in 0..row_start.len() {
        if specs[r] != 0 {
            // specs[r]==1 镜像 kernel：第 t 个 token 处理完把窗口写行
            // row_id[r]+t+1（行 j = 前 j 个 token 之后的 conv 状态）
            for c in 0..conv_dim {
                let rbase = row_id[r] as usize * conv_dim * st + c * st;
                let t0 = row_start[r] as i64;
                for ti in 0..row_count[r] as usize {
                    let wbase = (row_id[r] as usize + ti + 1) * conv_dim * st + c * st;
                    for j in 0..st {
                        let li = ti as i64 + 1 - st as i64 + j as i64;
                        out[wbase + j] = if li < 0 {
                            state[rbase + (li + st as i64) as usize]
                        } else {
                            x[(t0 + li) as usize * conv_dim + c]
                        };
                    }
                }
            }
        }
    }
    for r in 0..row_start.len() {
        if specs[r] != 0 {
            continue; // 逐位置段不走逐段滚动（与 kernel 同分流）
        }
        for c in 0..conv_dim {
            let base = row_id[r] as usize * conv_dim * st + c * st;
            for j in 0..st {
                let idx = row_count[r] as i32 - st as i32 + j as i32;
                out[base + j] = if idx < 0 {
                    state[base + (idx + st as i32) as usize]
                } else {
                    x[((row_start[r] as i32 + idx) as usize) * conv_dim + c]
                };
            }
        }
    }
    out
}

/// gated deltanet 递推参考（与 kernel 同步序）：每 (row, v_head) 持
/// S[v][0..k_dim] f32；token 循环内 decay→kvm→delta→S 更新→输出；
/// l2norm 后 q 再除 √k_dim（除法非倒数乘）。kvm/o 累加序：kd=vd=128
/// 走 E2 两级折叠（切片 4 项 + 跨切片 32 项——镜像 kernel 的 warp 归约
/// 树），其余 j 严格升序顺序累加
#[allow(clippy::too_many_arguments)]
pub fn deltanet(
    q: &[u16],        // [tokens, k_heads, k_dim]
    k: &[u16],        // [tokens, k_heads, k_dim]
    v: &[u16],        // [tokens, v_heads, v_dim]
    a: &[u16],        // [tokens, v_heads]
    b: &[u16],        // [tokens, v_heads]
    a_log: &[f32],    // [v_heads]
    dt_bias: &[u16],  // [v_heads]
    state: &mut [u16], // [rows, v_heads, v_dim, k_dim] bf16（in/out）
    row_start: &[u32],
    row_count: &[u32],
    row_id: &[u32],
    k_heads: usize,
    v_heads: usize,
    k_dim: usize,
    v_dim: usize,
    l2_eps: f32,
    specs: &[u32],
) -> Vec<u16> {
    let kd_sqrt = (k_dim as f32).sqrt();
    assert_eq!(specs.len(), row_start.len(), "specs 须与段数同长（逐 span 标志）");
    let total = row_start
        .iter()
        .zip(row_count)
        .map(|(&s, &c)| (s + c) as usize)
        .max()
        .unwrap_or(0); // 覆盖所有行的最大 token 下标
    let mut out = vec![0u16; total * v_heads * v_dim];
    for r in 0..row_start.len() {
        let st = row_start[r] as usize;
        let cnt = row_count[r] as usize;
        for h in 0..v_heads {
            let kh = h * k_heads / v_heads; // repeat_interleave 映射
            let big_a = expf(a_log[h]);
            let dtb = bf16_to_f32(dt_bias[h]);
            // E2 镜像（kd=vd=128）：kernel 的 kvm/o 是切片+两级折叠的
            // warp 归约树——同树镜像保位对齐；其余形状 j 严格升序
            let e2 = k_dim == 128 && v_dim == 128;
            // 行 stride = v_heads*v_dim*k_dim（与 kernel 同）。specs[r]==0：
            // 整段一份 S 原地递推；specs[r]==1：token i 读行 row_id[r]+i、
            // 写行 +i+1（逐位置落盘，与 kernel 逐 span 分支同树同序）。
            // 标志按段取（与 kernel 同）——同一次调用里可混普通段与链段。
            // S 装载即展宽、更新即收 bf16——逐 token 装载/回写与原一次性
            // 回写位等价
            let spec = specs[r];
            let rstride = v_heads * v_dim * k_dim;
            let base0 = (row_id[r] as usize * v_heads + h) * v_dim * k_dim;
            for i in 0..cnt {
                let sbase = if spec == 0 { base0 } else { base0 + i * rstride };
                let mut s: Vec<f32> = state[sbase..sbase + v_dim * k_dim]
                    .iter()
                    .map(|&b| bf16_to_f32(b))
                    .collect();
                let g = st + i;
                let decay = expf(0.0 - big_a * softplus(bf16_to_f32(a[g * v_heads + h]) + dtb));
                // β 先收 bf16（HF b.sigmoid() 在 bf16 张量上）——与 kernel 同两跳
                let beta = bf16_to_f32(f32_to_bf16(1.0f32 / (1.0f32 + expf(0.0 - bf16_to_f32(b[g * v_heads + h])))));
                let qb = (g * k_heads + kh) * k_dim;
                // l2norm（j 升序同 kernel）
                let (mut sq, mut sk) = (0.0f32, 0.0f32);
                for j in 0..k_dim {
                    let qv = bf16_to_f32(q[qb + j]);
                    let kv = bf16_to_f32(k[qb + j]);
                    sq = qv * qv + sq;
                    sk = kv * kv + sk;
                }
                let invq = 1.0f32 / ((sq + l2_eps).sqrt());
                let invk = 1.0f32 / ((sk + l2_eps).sqrt());
                for vv in 0..v_dim {
                    let srow = &mut s[vv * k_dim..][..k_dim];
                    for j in 0..k_dim {
                        srow[j] *= decay;
                    }
                    // E2 归约树镜像：切片内 j 升序 4 项（lane 部分和）→
                    // 跨切片 u 升序 32 项（块内折叠）——kernel 同树同序
                    let mut kvm = 0.0f32;
                    if e2 {
                        for sl in 0..32 {
                            let mut p = 0.0f32;
                            for i in 0..4 {
                                p = srow[sl * 4 + i] * (bf16_to_f32(k[qb + sl * 4 + i]) * invk) + p;
                            }
                            kvm = p + kvm;
                        }
                    } else {
                        for j in 0..k_dim {
                            kvm = srow[j] * (bf16_to_f32(k[qb + j]) * invk) + kvm;
                        }
                    }
                    let vin = bf16_to_f32(v[(g * v_heads + h) * v_dim + vv]);
                    let delta = (vin - kvm) * beta;
                    for j in 0..k_dim {
                        // 状态池 bf16：存储即收——o/下一 token 递推全吃收过的
                        // 同一值（kernel 同式：寄存器 f32 累加、落池收 bf16）
                        srow[j] = bf16_to_f32(f32_to_bf16(
                            (bf16_to_f32(k[qb + j]) * invk) * delta + srow[j],
                        ));
                    }
                    let mut o = 0.0f32;
                    if e2 {
                        for sl in 0..32 {
                            let mut p = 0.0f32;
                            for i in 0..4 {
                                p = srow[sl * 4 + i] * ((bf16_to_f32(q[qb + sl * 4 + i]) * invq) / kd_sqrt) + p;
                            }
                            o = p + o;
                        }
                    } else {
                        for j in 0..k_dim {
                            o = srow[j] * ((bf16_to_f32(q[qb + j]) * invq) / kd_sqrt) + o;
                        }
                    }
                    out[(g * v_heads + h) * v_dim + vv] = f32_to_bf16(o);
                }
                let wbase = if spec == 0 { sbase } else { sbase + rstride };
                for (d, f) in state[wbase..wbase + v_dim * k_dim].iter_mut().zip(s) {
                    *d = f32_to_bf16(f); // 已是 bf16 可表示值——恒等往返
                }
            }
        }
    }
    out
}

/// 带门 RMSNorm（Qwen3_5RMSNormGated 同语义）：xb=bf16(x·inv)（唯一中间
/// round），xw=w_f32·f32(xb) 不收，末级 xw·silu(z) 单次收 bf16——
/// 块归约序同 rms_block_sum（与 kernel 同式同序）
pub fn rms_norm_gated(x: &[u16], z: &[u16], w: &[f32], rows: usize, hidden: usize, eps: f32) -> Vec<u16> {
    let mut out = vec![0u16; rows * hidden];
    for r in 0..rows {
        let base = r * hidden;
        let acc = rms_block_sum(x, base, hidden);
        let inv = 1.0f32 / ((acc / hidden as f32 + eps).sqrt());
        for i in 0..hidden {
            let xb = f32_to_bf16(bf16_to_f32(x[base + i]) * inv);
            let xw = w[i] * bf16_to_f32(xb);
            let zv = bf16_to_f32(z[base + i]);
            out[base + i] = f32_to_bf16(xw * (zv * (1.0f32 / (1.0f32 + expf(-zv)))));
        }
    }
    out
}

/// 行 argmax 参考（bf16 → f32 比较，全序：值严格大或相等取小下标）。
/// kernel 是跨步扫描+块内归约——全序保证与镜像的单遍扫描结果恒同
pub fn argmax_rows(x: &[u16], rows: &[u32], n: usize) -> Vec<u32> {
    rows.iter()
        .map(|&r| {
            let base = r as usize * n;
            let mut bv = f32::NEG_INFINITY;
            let mut bi = 0xffffffffu32;
            for i in 0..n {
                let v = bf16_to_f32(x[base + i]);
                if v > bv || (v == bv && (i as u32) < bi) {
                    bv = v;
                    bi = i as u32;
                }
            }
            bi
        })
        .collect()
}

/// GEMM 参考的 weight-transposed 形态（gemm_wt 对拍基准）：C[m,n] =
/// A[m,k]·Wᵀ，W 按存储 [n,k] 直读——f64 累加（厂商库容差模式，g7）
pub fn gemm_wt(a: &[u16], w: &[u16], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0f64; m * n];
    for r in 0..m {
        for col in 0..n {
            let mut acc = 0f64;
            for i in 0..k {
                acc += bf16_to_f32(a[r * k + i]) as f64 * bf16_to_f32(w[col * k + i]) as f64;
            }
            c[r * n + col] = acc;
        }
    }
    c
}
