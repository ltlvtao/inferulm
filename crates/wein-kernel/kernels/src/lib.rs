//! 冒烟实验 kernel 集（02-architecture §13-2 验收线的设备侧）。
//!
//! **冒烟发现（风险登记）**：纯 Rust→PTX 路线在当前工具链上锁死 nightly +
//! 两个未稳定 feature（`abi_ptx` #38788、`stdarch_nvptx` #111199）+ build-std
//! （tier-3 无预编译 core）。M0 接受 nightly 钉死；若长期不稳定，走设计预留的
//! cubin 旁路（离线编 cubin + 运行时加载），架构不变（§13-2 决议 1）。
//!
//! **bf16 = u16 软实现**（位操作 + f32 计算）：不依赖 PTX bf16 指令，任意
//! `.target` 可跑；这本身就是"f16/bf16 intrinsic 覆盖够用"问题（§13-2 ③）的
//! A 计划答案——不指望工具链 intrinsic，数值路径完全自控。W2 真 attention
//! kernel 时再评估切 PTX bf16 intrinsic（届时有 golden vector 兜底）。
//!
//! 一维 grid/block 约定（与 host 侧 launch 封装一致）。
#![no_std]
#![feature(abi_ptx)]
#![feature(stdarch_nvptx)]
// sqrt/mul_add 等浮点方法在新 nightly 被移入 core_float_math（#137578）；
// 正确舍入原语直接走 intrinsics——语义等同 sqrt.rn.f32（冒烟风险登记项）
#![feature(core_intrinsics)]

use core::arch::nvptx::{_block_dim_x, _block_idx_x, _syncthreads, _thread_idx_x};

/// GPU 上没有展开——panic 即 trap（warp 死给驱动报，不挂整机）
#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    unsafe { core::arch::nvptx::trap() }
}

/// 全局线程 id（一维 grid）
#[inline(always)]
fn tid() -> u32 {
    unsafe { _block_idx_x() * _block_dim_x() + _thread_idx_x() }
}

/// bf16 → f32：精确（尾数位左移 16，无舍入）
#[inline(always)]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// f32 → bf16：round-to-nearest-even。NaN 未特判（冒烟输入不含 NaN，
/// W2 golden vector 框架再补全语义）。
#[inline(always)]
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits + 0x7fff + lsb) >> 16) as u16
}

/// |x|：位实现（清符号位）——语义与 std 的 abs 恒等（非 NaN 域），
/// 不依赖 f32 方法面（core_float_math #137578 迁移中，自控为上）
#[inline(always)]
fn fabs(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7fff_ffff)
}

/// 整除/取余（无除零检查——除数是形状参数，host 侧保证非零）。
/// rustc 对运行数除数默认插 panic 桩（PTX 里成 call），no_std kernel 禁 call
#[inline(always)]
fn udiv(x: usize, y: usize) -> usize {
    unsafe { core::intrinsics::unchecked_div(x, y) }
}
#[inline(always)]
fn urem(x: usize, y: usize) -> usize {
    unsafe { core::intrinsics::unchecked_rem(x, y) }
}

/// paged KV 槽位寻址：slot = block_table[s·bpr + j/bs]·bs + j%bs——与
/// wein-memory plan_kv 的 token 记账同构（[page][block_size][heads,head_dim]）
#[inline(always)]
fn slot_of(bt: *const u32, s: usize, j: usize, bpr: usize, bs: usize) -> usize {
    let blk = udiv(j, bs);
    let off = urem(j, bs);
    unsafe { *bt.add(s * bpr + blk) as usize * bs + off }
}

/// floor：intrinsic（LLVM→cvt.rmi.f32.f32，精确算子非近似）。
/// 注意此 nightly 上浮点 intrinsic 是泛型 const fn（floorf32 名已不存在）
#[inline(always)]
fn floor(x: f32) -> f32 {
    core::intrinsics::floor::<f32>(x)
}

/// exp 自控路径（与 reference::expf 同字面量同 Horner 序——位对齐的
/// 可达性依据；禁 MUFU ex2 近似）。t = x·log2e → k=⌊t⌋、f=t−k∈[0,1)
/// → 2^f 七次 Horner → 指数域加 k（尾数不动，精确）
#[inline(always)]
fn expf(x: f32) -> f32 {
    const LOG2E: f32 = 1.4426950408889634;
    // a_n = ln2^n / n!（截断误差 ≤1.4e-6 ≪ bf16 分辨率）
    const A7: f32 = 1.5252733804059838e-05;
    const A6: f32 = 0.00015403530393381606;
    const A5: f32 = 0.0013333558146428441;
    const A4: f32 = 0.009618129107628477;
    const A3: f32 = 0.055504108664821576;
    const A2: f32 = 0.2402265069591007;
    const A1: f32 = 0.6931471805599453;
    const A0: f32 = 1.0;
    let t = x * LOG2E;
    let kf = floor(t);
    let f = t - kf; // [0,1)，Sterbenz 精确
    let mut p = A7;
    p = p * f + A6;
    p = p * f + A5;
    p = p * f + A4;
    p = p * f + A3;
    p = p * f + A2;
    p = p * f + A1;
    p = p * f + A0;
    f32::from_bits(((p.to_bits() as i32).wrapping_add((kf as i32) << 23)) as u32)
}

/// RNE 半偶取整到整值 f32（自控实现，与 reference::rne 同式——量化取整
/// 语义要自己钉，round() 默认行为不可依赖）
#[inline(always)]
fn rne(x: f32) -> f32 {
    let fl = floor(x);
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

// ───────────────────── I1 竖切数值原语（deltanet 门控链） ─────────────────────
// 与 reference.rs 同字面量同 Horner 序（位对齐纪律延伸到 log 域）：
// softplus(x) = ln(1+e^x) 需要 log——atanh 级数 + 指数位分解的自控 log2。

/// log2（自控）：位分解 z = 2^e·m（m∈[1,2)，>√2 折半）→ m 归一到
/// [1/√2, √2) → atanh 级数（r=(m−1)/(m+1)≤0.1716，截到 r¹¹，误差 ~1e-11）。
/// 输入域：softplus 喂的 z=1+e^x ∈ [1, 5e8]——无零/次正规担忧
#[inline(always)]
fn log2f(z: f32) -> f32 {
    // √2 的最近 f32（0x3FB504F3）——与 reference 同字面量
    const SQRT2: f32 = 1.4142135;
    const LN2: f32 = 0.6931471805599453;
    let bits = z.to_bits();
    let e = ((bits >> 23) as i32) - 127;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000); // [1, 2)
    let mut e = e;
    if m > SQRT2 {
        m = m * 0.5;
        e += 1;
    }
    let r = (m - 1.0) / (m + 1.0);
    let r2 = r * r;
    // atanh(r) = r·(1 + r²·(1/3 + r²·(1/5 + …r²·(1/11))))
    let atanh = r
        * (1.0 + r2 * (1.0 / 3.0 + r2 * (1.0 / 5.0 + r2 * (1.0 / 7.0 + r2 * (1.0 / 9.0 + r2 * (1.0 / 11.0))))));
    // log2(m) = 2·atanh(r)/ln2
    e as f32 + atanh * (2.0 / LN2)
}

/// softplus：x>20 直通（e^x 溢出保护，torch 同阈），否则 ln(1+e^x)
#[inline(always)]
fn softplus(x: f32) -> f32 {
    const LN2: f32 = 0.6931471805599453;
    if x > 20.0 {
        x
    } else {
        log2f(1.0 + expf(x)) * LN2
    }
}

/// 向量加（f32 逐位可对齐——k5 的对象；也是 k8 launch 开销测量的载体）
#[no_mangle]
pub unsafe extern "ptx-kernel" fn vec_add_f32(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    n: u32,
) {
    let i = tid();
    if i < n {
        *c.add(i as usize) = *a.add(i as usize) + *b.add(i as usize);
    }
}

/// bf16 分段点积：线程 t 计算 x[t*seg .. t*seg+seg] 与 y 同段的 f32 累加点积，
/// RNE 舍回 bf16 写 out[t]。
///
/// 这是 attention QK·V 数值路径的最小完整形态（load bf16 → f32 fma 链 →
/// RNE 收尾），§13-2 ② "bf16 attention 逐位对齐"的冒烟化身——跨线程归约
/// （shared memory）是 W2 的活，不影响数值对齐结论。
///
/// 分离乘加而非 mul_add（冒烟决定：该 nightly 上 core 的 f32::mul_add 刚被
/// 移入未稳定 core_float_math #137578——不为一个 fma 多背一个 feature 门）。
/// mul.rn.f32 / add.rn.f32 都是 IEEE 正确舍入单算，逐位对齐可达性与 fma 相同
/// （两侧同操作序列即同位）。W2 真 kernel 时再定 fma 策略
/// （稳定化跟踪 / intrinsic / 内联 asm 三选一）。
#[no_mangle]
pub unsafe extern "ptx-kernel" fn bf16_dot(
    x: *const u16,
    y: *const u16,
    out: *mut u16,
    seg: u32,
    m: u32,
) {
    let t = tid();
    if t < m {
        let base = (t * seg) as usize;
        let mut acc = 0.0f32;
        let mut j = 0usize;
        while j < seg as usize {
            let a = bf16_to_f32(*x.add(base + j));
            let b = bf16_to_f32(*y.add(base + j));
            acc = a * b + acc; // rustc 默认不收缩乘加（fp-contract off）——k2 断言
            j += 1;
        }
        *out.add(t as usize) = f32_to_bf16(acc);
    }
}

/// RMSNorm（Llama/Qwen 形态）：out[r] = x[r] / sqrt(mean(x[r]²) + eps) * w。
/// M0 融合算子清单首个入表成员（§4）。
///
/// 对齐纪律（与 reference.rs 同守，golden 对拍的可达性依据）：
/// - 只用正确舍入原语：mul.rn / add.rn / div.rn / sqrt.rn——禁 rsqrt(MUFU)
///   近似与 fma 收缩（两侧同为 IEEE 单算才可能位位相等）
/// - 行内顺序累加（i 递增），与参考实现逐步同序
/// - 一行由单线程完整计算：launch 几何（row/blocked 档）不改变数值序
#[no_mangle]
pub unsafe extern "ptx-kernel" fn rms_norm(
    x: *const u16,
    w: *const u16,
    out: *mut u16,
    hidden: u32,
    eps: f32,
    rows: u32,
) {
    let r = tid();
    if r < rows {
        let h = hidden as usize;
        let base = (r as usize) * h;
        let mut acc = 0.0f32;
        let mut i = 0usize;
        while i < h {
            let v = bf16_to_f32(*x.add(base + i));
            acc = v * v + acc;
            i += 1;
        }
        let denom = core::intrinsics::sqrt::<f32>(acc / hidden as f32 + eps);
        let inv = 1.0f32 / denom;
        let mut i = 0usize;
        while i < h {
            let o = bf16_to_f32(*x.add(base + i)) * inv * bf16_to_f32(*w.add(i));
            *out.add(base + i) = f32_to_bf16(o);
            i += 1;
        }
    }
}

/// RoPE（half-split，Llama/Qwen 形态）。cos/sin 表是输入（host 生成上传，
/// 图捕获友好）——kernel 只做 apply。每线程一对 (t,p)，同一 pos 同时处理
/// q 与 k（表项只读一次）。out = (x1·c − x2·s, x2·c + x1·s)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn rope(
    q: *const u16,
    k: *const u16,
    cos: *const f32,
    sin: *const f32,
    pos: *const u32,
    out_q: *mut u16,
    out_k: *mut u16,
    rot_dim: u32,
    tokens: u32,
) {
    let half = (rot_dim / 2) as usize;
    let rd = rot_dim as usize;
    let i = tid() as usize;
    if i >= tokens as usize * half {
        return;
    }
    let (t, p) = (udiv(i, half), urem(i, half));
    let ti = *pos.add(t) as usize;
    let c = *cos.add(ti * half + p);
    let s = *sin.add(ti * half + p);
    let (x1, x2) = (bf16_to_f32(*q.add(t * rd + p)), bf16_to_f32(*q.add(t * rd + half + p)));
    *out_q.add(t * rd + p) = f32_to_bf16(x1 * c - x2 * s);
    *out_q.add(t * rd + half + p) = f32_to_bf16(x2 * c + x1 * s);
    let (x1, x2) = (bf16_to_f32(*k.add(t * rd + p)), bf16_to_f32(*k.add(t * rd + half + p)));
    *out_k.add(t * rd + p) = f32_to_bf16(x1 * c - x2 * s);
    *out_k.add(t * rd + half + p) = f32_to_bf16(x2 * c + x1 * s);
}

/// SiLU：out = x·sigmoid(x)，sigmoid = 1/(1+expf(−x))——逐元素，
/// 线程间无交互（几何不影响数值）
#[no_mangle]
pub unsafe extern "ptx-kernel" fn silu(x: *const u16, out: *mut u16, n: u32) {
    let i = tid() as usize;
    if i < n as usize {
        let v = bf16_to_f32(*x.add(i));
        *out.add(i) = f32_to_bf16(v * (1.0f32 / (1.0f32 + expf(-v))));
    }
}

/// W8A8 动态量化（逐 token）：amax → scale=amax/127 → RNE 半偶 → clamp ±127。
/// 一线程一 token（顺序扫 hidden，与参考同序）；scale=0 时 q 全 0
#[no_mangle]
pub unsafe extern "ptx-kernel" fn w8a8_quant(
    x: *const u16,
    q: *mut i8,
    scale: *mut f32,
    hidden: u32,
    tokens: u32,
) {
    let t = tid() as usize;
    if t >= tokens as usize {
        return;
    }
    let h = hidden as usize;
    let base = t * h;
    let mut amax = 0.0f32;
    let mut i = 0usize;
    while i < h {
        let a = fabs(bf16_to_f32(*x.add(base + i)));
        amax = if a > amax { a } else { amax };
        i += 1;
    }
    if amax == 0.0 {
        *scale.add(t) = 0.0;
        return;
    }
    let sc = amax / 127.0;
    *scale.add(t) = sc;
    let mut i = 0usize;
    while i < h {
        let r = rne(bf16_to_f32(*x.add(base + i)) / sc);
        let r = if r > 127.0 { 127.0 } else if r < -127.0 { -127.0 } else { r };
        *q.add(base + i) = r as i8;
        i += 1;
    }
}

/// embed gather：out[t·hidden+e] = w[id[t]·hidden+e]——纯读路由查表，
/// 位拷贝无数值（id 合法性是 host 责任：vocab 越界 = 未定义）
#[no_mangle]
pub unsafe extern "ptx-kernel" fn embed(
    ids: *const u32,
    w: *const u16,
    out: *mut u16,
    hidden: u32,
    tokens: u32,
) {
    let h = hidden as usize;
    let i = tid() as usize;
    if i >= tokens as usize * h {
        return;
    }
    let (t, e) = (udiv(i, h), urem(i, h));
    *out.add(t * h + e) = *w.add(*ids.add(t) as usize * h + e);
}

/// bf16 加（残差）：f32 域加法 RNE 收 bf16——f32+bf16→bf16 与参考同式
#[no_mangle]
pub unsafe extern "ptx-kernel" fn add(a: *const u16, b: *const u16, out: *mut u16, n: u32) {
    let i = tid() as usize;
    if i < n as usize {
        *out.add(i) = f32_to_bf16(bf16_to_f32(*a.add(i)) + bf16_to_f32(*b.add(i)));
    }
}

/// bf16 乘（门控：silu(g)·u）：f32 域乘法 RNE 收 bf16
#[no_mangle]
pub unsafe extern "ptx-kernel" fn mul(a: *const u16, b: *const u16, out: *mut u16, n: u32) {
    let i = tid() as usize;
    if i < n as usize {
        *out.add(i) = f32_to_bf16(bf16_to_f32(*a.add(i)) * bf16_to_f32(*b.add(i)));
    }
}

/// KV scatter write：新 token 的 K/V 按 (seq,pos) 页表寻址写入 paged 池。
/// 一线程一 (t,e) 槽内元素——纯拷贝，未触及槽位不碰（池属调度层所有）
#[no_mangle]
pub unsafe extern "ptx-kernel" fn kv_scatter(
    src_k: *const u16,
    src_v: *const u16,
    block_table: *const u32,
    seq_ids: *const u32,
    positions: *const u32,
    k_dst: *mut u16,
    v_dst: *mut u16,
    bpr: u32,
    block_size: u32,
    head_elems: u32,
    tokens: u32,
) {
    let he = head_elems as usize;
    let i = tid() as usize;
    if i >= tokens as usize * he {
        return;
    }
    let (t, e) = (udiv(i, he), urem(i, he));
    let slot = slot_of(block_table, *seq_ids.add(t) as usize, *positions.add(t) as usize,
        bpr as usize, block_size as usize);
    *k_dst.add(slot * he + e) = *src_k.add(t * he + e);
    *v_dst.add(slot * he + e) = *src_v.add(t * he + e);
}

/// varlen paged attention（correctness-first 两遍法）：一线程一 (t,qh)，
/// 因果只看 0..=pos；GQA：kv 头 = qh%kv_heads。pass1 记分取 max →
/// pass2 p_j=expf(s_j−m) 顺序累加分母 → 逐 e 加权 V（j 升序，与参考同序）。
/// M0 正确性档：单线程扫全 KV，序列长上限 256（W2 flash 分块移除此限）
#[no_mangle]
pub unsafe extern "ptx-kernel" fn paged_attn(
    q: *const u16,
    k_cache: *const u16,
    v_cache: *const u16,
    block_table: *const u32,
    seq_ids: *const u32,
    positions: *const u32,
    out: *mut u16,
    q_scale: f32,
    bpr: u32,
    block_size: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    tokens: u32,
) {
    const MAX_LEN: usize = 256;
    let hd = head_dim as usize;
    let qh_n = q_heads as usize;
    let i = tid() as usize;
    if i >= tokens as usize * qh_n {
        return;
    }
    let (t, qh) = (udiv(i, qh_n), urem(i, qh_n));
    let kh = urem(qh, kv_heads as usize);
    let s = *seq_ids.add(t) as usize;
    let pos = *positions.add(t) as usize;
    if pos + 1 > MAX_LEN {
        return; // 超限不写——启动期容量检查的责任在 exec 层（W2 移除）
    }
    let (bpr, bs) = (bpr as usize, block_size as usize);
    let he = kv_heads as usize * hd;
    let qbase = i * hd;

    // sc 经裸指针访问（数组下标检查在 PTX 里成 panic 桩 call——
    // pos+1 ≤ MAX_LEN 已由上方 return 保证）
    let mut sc = [0.0f32; MAX_LEN];
    let scp = sc.as_mut_ptr();
    let mut m = f32::NEG_INFINITY;
    let mut j = 0usize;
    while j <= pos {
        let kbase = slot_of(block_table, s, j, bpr, bs) * he + kh * hd;
        let mut dot = 0.0f32;
        let mut d = 0usize;
        while d < hd {
            let a = bf16_to_f32(*q.add(qbase + d));
            let b = bf16_to_f32(*k_cache.add(kbase + d));
            dot = a * b + dot;
            d += 1;
        }
        *scp.add(j) = dot * q_scale;
        let scj = *scp.add(j);
        m = if scj > m { scj } else { m };
        j += 1;
    }
    let mut denom = 0.0f32;
    let mut j = 0usize;
    while j <= pos {
        *scp.add(j) = expf(*scp.add(j) - m);
        denom = *scp.add(j) + denom;
        j += 1;
    }
    let mut e = 0usize;
    while e < hd {
        let mut acc = 0.0f32;
        let mut j = 0usize;
        while j <= pos {
            let vbase = slot_of(block_table, s, j, bpr, bs) * he + kh * hd;
            acc = *scp.add(j) * bf16_to_f32(*v_cache.add(vbase + e)) + acc;
            j += 1;
        }
        *out.add(qbase + e) = f32_to_bf16(acc / denom);
        e += 1;
    }
}

// ───────────────────── I1 竖切算子（Qwen3.5 文本骨干） ─────────────────────
// 零中心 norm / partial rope / 门控乘 / 深度因果卷积 / gated deltanet /
// 带门 RMSNorm / 设备端 argmax——数值对齐纪律与上文同守。

/// 零中心 RMSNorm（Qwen3.5 形态）：out = x·rsqrt(mean(x²)+eps)·(1+w)。
/// 与 transformers Qwen3_5RMSNorm 同语义：norm 在 f32、(1+w) 在 f32、
/// 单次收 bf16。行内顺序累加同 rms_norm
#[no_mangle]
pub unsafe extern "ptx-kernel" fn rms_norm_zc(
    x: *const u16,
    w: *const u16,
    out: *mut u16,
    hidden: u32,
    eps: f32,
    rows: u32,
) {
    let r = tid();
    if r < rows {
        let h = hidden as usize;
        let base = (r as usize) * h;
        let mut acc = 0.0f32;
        let mut i = 0usize;
        while i < h {
            let v = bf16_to_f32(*x.add(base + i));
            acc = v * v + acc;
            i += 1;
        }
        let inv = 1.0f32 / core::intrinsics::sqrt::<f32>(acc / hidden as f32 + eps);
        let mut i = 0usize;
        while i < h {
            let o = bf16_to_f32(*x.add(base + i)) * inv * (1.0 + bf16_to_f32(*w.add(i)));
            *out.add(base + i) = f32_to_bf16(o);
            i += 1;
        }
    }
}

/// partial RoPE（单张量、每 token 一 pos、rotate_half、前 rot_dim 维旋转、
/// 其余 pass-through 不触碰——out 可原地别名 x）。行 = tokens×heads 个
/// head_dim 宽的头行。x: [tokens, heads, head_dim]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn rope_pt(
    x: *const u16,
    cos: *const f32,
    sin: *const f32,
    pos: *const u32,
    out: *mut u16,
    rot_dim: u32,
    head_dim: u32,
    heads: u32,
    tokens: u32,
) {
    let half = (rot_dim / 2) as usize;
    let hd = head_dim as usize;
    let row_pairs = heads as usize * half; // 每 token 的 (头,对) 数
    let i = tid() as usize;
    if i >= tokens as usize * row_pairs {
        return;
    }
    let (t, rem) = (udiv(i, row_pairs), urem(i, row_pairs));
    let p = urem(rem, half);
    let row = t * (heads as usize * hd) + udiv(rem, half) * hd;
    let ti = *pos.add(t) as usize;
    let c = *cos.add(ti * half + p);
    let s = *sin.add(ti * half + p);
    let (x1, x2) = (bf16_to_f32(*x.add(row + p)), bf16_to_f32(*x.add(row + half + p)));
    *out.add(row + p) = f32_to_bf16(x1 * c - x2 * s);
    *out.add(row + half + p) = f32_to_bf16(x2 * c + x1 * s);
}

/// 门控乘（注意力输出门 / 任意 sigmoid 门）：out = x·sigmoid(g)。
/// sigmoid = 1/(1+expf(−g))——自控 exp 路径，与 silu 同源
#[no_mangle]
pub unsafe extern "ptx-kernel" fn sigmul(
    x: *const u16,
    g: *const u16,
    out: *mut u16,
    n: u32,
) {
    let i = tid() as usize;
    if i < n as usize {
        let v = bf16_to_f32(*x.add(i));
        let s = bf16_to_f32(*g.add(i));
        *out.add(i) = f32_to_bf16(v * (1.0f32 / (1.0f32 + expf(-s))));
    }
}

/// 深度wise 因果 conv1d（kernel=4，激活 silu）：varlen 批内每 token 依赖
/// 同请求前 3 个 token（跨步接 conv_state）。一线程一 (token, channel)：
/// out[t,c] = silu(Σ_k w[c,k]·tap_k)，tap 不足 3 步取 state_in。
/// conv 结果先收 bf16 再 silu 收 bf16（torch conv→act 两级 round）
#[no_mangle]
pub unsafe extern "ptx-kernel" fn conv1d_dw(
    x: *const u16,        // [tokens, conv_dim]
    w: *const u16,        // [conv_dim, kernel]
    state_in: *const u16, // [rows, conv_dim, kernel-1]
    out: *mut u16,        // [tokens, conv_dim]
    tok_row: *const u32,  // [tokens] → 状态行
    tok_pos: *const u32,  // [tokens] → 请求内局部位置
    conv_dim: u32,
    kernel: u32,
    tokens: u32,
) {
    let cd = conv_dim as usize;
    let ks = kernel as usize;
    let st = ks - 1;
    let i = tid() as usize;
    if i >= tokens as usize * cd {
        return;
    }
    let (g, c) = (udiv(i, cd), urem(i, cd));
    let (row, p) = (*tok_row.add(g) as usize, *tok_pos.add(g) as i32);
    let mut acc = 0.0f32;
    let mut k = 0usize;
    while k < ks {
        let idx = p - (st as i32) + (k as i32); // 本请求内局部索引
        let v = if idx < 0 {
            bf16_to_f32(*state_in.add(row * cd * st + c * st + (idx + st as i32) as usize))
        } else {
            // g - p + idx = 请求内 idx 的全局 token 下标（同请求段连续）
            bf16_to_f32(*x.add(((g as i32 - p + idx) as usize) * cd + c))
        };
        acc = v * bf16_to_f32(*w.add(c * ks + k)) + acc;
        k += 1;
    }
    let conv = f32_to_bf16(acc);
    let v = bf16_to_f32(conv);
    *out.add(g * cd + c) = f32_to_bf16(v * (1.0f32 / (1.0f32 + expf(-v))));
}

/// conv 状态滚动：state_out = 请求窗口（旧 state ++ 本段 x）的末 kernel-1 个。
/// j 升序时读槽位恒 > 写槽位（见 count<3 移位分析）——原地安全
#[no_mangle]
pub unsafe extern "ptx-kernel" fn conv_state_roll(
    x: *const u16,        // [tokens, conv_dim]
    state: *mut u16,      // [rows, conv_dim, kernel-1] 原地
    row_start: *const u32, // [rows] 本段起点（tokens 轴）
    row_count: *const u32, // [rows] 本段 token 数
    conv_dim: u32,
    kernel: u32,
    rows: u32,
) {
    let cd = conv_dim as usize;
    let st = (kernel - 1) as usize;
    let i = tid() as usize;
    if i >= rows as usize * cd {
        return;
    }
    let (r, c) = (udiv(i, cd), urem(i, cd));
    let base = r * cd * st + c * st;
    let mut j = 0usize;
    while j < st {
        let idx = *row_count.add(r) as i32 - (st as i32) + (j as i32);
        *state.add(base + j) = if idx < 0 {
            *state.add(base + (idx + st as i32) as usize)
        } else {
            *x.add(((*row_start.add(r) as i32 + idx) as usize) * cd + c)
        };
        j += 1;
    }
}

/// gated deltanet（recurrent correctness-first）：一块 = (请求行, v 头)，
/// 一线程一 v 维（线程私有 state 行 S[v][0..k_dim]，全局内存 L1 驻留）。
/// 每 token：q/k 逐头 l2norm（f32，eps=1e-6）→ q 再除 √k_dim →
/// S·=exp(g)；kv_mem=ΣS·k_n；delta=(v−kv_mem)·β；S+=k_n⊗delta；
/// out=ΣS·q_ns——与 torch_recurrent_gated_delta_rule 逐步同序，
/// 累加 j 升序；core out 收 bf16（norm/gate 在 rms_norm_gated 另做）。
/// 门控标量：β=sigmoid(b)，g=−exp(A_log)·softplus(a+dt_bias)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn deltanet(
    q: *const u16,        // [tokens, k_heads, k_dim]
    k: *const u16,        // [tokens, k_heads, k_dim]
    v: *const u16,        // [tokens, v_heads, v_dim]
    a: *const u16,        // [tokens, v_heads]
    b: *const u16,        // [tokens, v_heads]
    a_log: *const f32,    // [v_heads]
    dt_bias: *const u16,  // [v_heads]
    state: *mut f32,      // [rows, v_heads, v_dim, k_dim]（in/out）
    core_out: *mut u16,   // [tokens, v_heads*v_dim]
    row_start: *const u32, // [rows]
    row_count: *const u32, // [rows]
    row_id: *const u32,    // [rows] → state 行
    n_rows: u32,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    l2_eps: f32,
) {
    let bid = _block_idx_x() as usize;
    if bid >= n_rows as usize * v_heads as usize {
        return;
    }
    let (r, h) = (udiv(bid, v_heads as usize), urem(bid, v_heads as usize));
    let (kd, vd) = (k_dim as usize, v_dim as usize);
    let vh = v_heads as usize;
    // v 头 ↔ k 头：repeat_interleave（v_heads/k_heads 倍）→ h·kh/vh
    let kh = h * k_heads as usize / vh;
    let vv = _thread_idx_x() as usize; // 本线程拥有的 v 维
    if vv >= vd {
        return;
    }
    let big_a = expf(*a_log.add(h));
    let dtb = bf16_to_f32(*dt_bias.add(h));
    let srow = state.add(((*row_id.add(r) as usize) * vh + h) * vd * kd + vv * kd);
    let st = *row_start.add(r) as usize;
    let cnt = *row_count.add(r) as usize;
    let kd_sqrt = core::intrinsics::sqrt::<f32>(kd as f32); // 参考是除 √kd：q_n/kd**0.5
    let mut i = 0usize;
    while i < cnt {
        let g = st + i;
        // 每 token 标量门（各线程冗余算——同序同式位同）
        let decay = expf(0.0 - big_a * softplus(bf16_to_f32(*a.add(g * vh + h)) + dtb));
        let beta = 1.0f32 / (1.0f32 + expf(0.0 - bf16_to_f32(*b.add(g * vh + h))));
        // q/k l2norm：各线程顺序扫（j 升序同参考）
        let qb = (g * k_heads as usize + kh) * kd;
        let mut sq = 0.0f32;
        let mut sk = 0.0f32;
        let mut j = 0usize;
        while j < kd {
            let qv = bf16_to_f32(*q.add(qb + j));
            let kv = bf16_to_f32(*k.add(qb + j));
            sq = qv * qv + sq;
            sk = kv * kv + sk;
            j += 1;
        }
        let invq = 1.0f32 / core::intrinsics::sqrt::<f32>(sq + l2_eps);
        let invk = 1.0f32 / core::intrinsics::sqrt::<f32>(sk + l2_eps);
        // S *= decay
        let mut j = 0usize;
        while j < kd {
            *srow.add(j) = *srow.add(j) * decay;
            j += 1;
        }
        // kv_mem = Σ_j S[v][j]·k_n[j]（j 升序）
        let mut kvm = 0.0f32;
        let mut j = 0usize;
        while j < kd {
            kvm = *srow.add(j) * (bf16_to_f32(*k.add(qb + j)) * invk) + kvm;
            j += 1;
        }
        let vin = bf16_to_f32(*v.add((g * vh + h) * vd + vv));
        let delta = (vin - kvm) * beta;
        // S[v][j] += k_n[j]·delta
        let mut j = 0usize;
        while j < kd {
            *srow.add(j) = (bf16_to_f32(*k.add(qb + j)) * invk) * delta + *srow.add(j);
            j += 1;
        }
        // out[v] = Σ_j S[v][j]·(q_n[j]/√kd)——更新后读（同参考序）
        let mut o = 0.0f32;
        let mut j = 0usize;
        while j < kd {
            o = *srow.add(j) * ((bf16_to_f32(*q.add(qb + j)) * invq) / kd_sqrt) + o;
            j += 1;
        }
        *core_out.add((g * vh + h) * vd + vv) = f32_to_bf16(o);
        i += 1;
    }
}

/// 带门 RMSNorm（deltanet 输出，Qwen3_5RMSNormGated 语义）：
/// h=f32(x)；xn=h·rsqrt(mean(h²)+eps)；xb=bf16(xn)（唯一中间 round）；
/// xw=w_f32·f32(xb)（f32 不收）；out=bf16(xw·silu(f32(z)))——末级单 round
#[no_mangle]
pub unsafe extern "ptx-kernel" fn rms_norm_gated(
    x: *const u16,     // [rows, hidden]（core out bf16）
    z: *const u16,     // [rows, hidden]（z 门）
    w: *const f32,     // [hidden]（F32 权重）
    out: *mut u16,
    hidden: u32,
    eps: f32,
    rows: u32,
) {
    let r = tid();
    if r < rows {
        let h = hidden as usize;
        let base = (r as usize) * h;
        let mut acc = 0.0f32;
        let mut i = 0usize;
        while i < h {
            let v = bf16_to_f32(*x.add(base + i));
            acc = v * v + acc;
            i += 1;
        }
        let inv = 1.0f32 / core::intrinsics::sqrt::<f32>(acc / hidden as f32 + eps);
        let mut i = 0usize;
        while i < h {
            let xb = f32_to_bf16(bf16_to_f32(*x.add(base + i)) * inv);
            let xw = *w.add(i) * bf16_to_f32(xb);
            let zv = bf16_to_f32(*z.add(base + i));
            *out.add(base + i) = f32_to_bf16(xw * (zv * (1.0f32 / (1.0f32 + expf(-zv)))));
            i += 1;
        }
    }
}

/// 设备端行 argmax（bf16 logits → u32 索引）：一块一行（256 线程跨步扫，
/// 各线程局部最优 → 全局 scratch → syncthreads → 线程 0 全序挑选）。
/// 全序 (值严格大，或相等下标小) 与扫描序无关——归约确定
#[no_mangle]
pub unsafe extern "ptx-kernel" fn argmax_rows(
    x: *const u16,     // [row_total, n] bf16
    rows: *const u32,  // [s] 采样的全局行号
    scratch: *mut u32, // [s, 256]×2（val_bits, idx 对）
    out: *mut u32,     // [s]
    n: u32,
    s: u32,
) {
    let blk = _block_idx_x() as usize;
    if blk >= s as usize {
        return;
    }
    let lane = _thread_idx_x() as usize; // < 256（block 恒 256）
    let base = *rows.add(blk) as usize * n as usize;
    let mut bv = f32::NEG_INFINITY;
    let mut bi = 0xffffffffu32;
    let mut i = lane;
    while i < n as usize {
        let v = bf16_to_f32(*x.add(base + i));
        if v > bv || (v == bv && (i as u32) < bi) {
            bv = v;
            bi = i as u32;
        }
        i += 256;
    }
    let sb = scratch.add(blk * 512);
    *sb.add(lane * 2) = bv.to_bits();
    *sb.add(lane * 2 + 1) = bi;
    _syncthreads();
    if lane == 0 {
        let mut bv = f32::NEG_INFINITY;
        let mut bi = 0xffffffffu32;
        let mut l = 0usize;
        while l < 256 {
            let v = f32::from_bits(*sb.add(l * 2));
            let idx = *sb.add(l * 2 + 1);
            if v > bv || (v == bv && idx < bi) {
                bv = v;
                bi = idx;
            }
            l += 1;
        }
        *out.add(blk) = bi;
    }
}
