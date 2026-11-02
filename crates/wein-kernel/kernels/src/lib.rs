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
// `f32::mul_add` 在本 nightly 上的落脚是 `core::f32::math::mul_add`
// （core_float_math #137578；见下方 fma()）
#![feature(core_float_math)]
// 状态池 bf16 后 4×bf16 走 u64 整载/整存（ld/st.global.b64——paged_attn
// 同习语；portable_simd 的 f32x4 容器已随 f32 池退役）

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

/// 单次舍入融合乘加 `a·b + c`：落 **`fma.rn.f32`**（PTX 已验，k2 断言
/// 热路径的计数下界）。点积/归约链由此每元素吃 1 条指令而非 mul+add 两条，
/// 且只舍入一次（比两条原语**更精确**）。
///
/// 位级合同的形态变了，纪律没变：两侧必须同式同位。宿主侧
/// （`reference.rs`，stable）用 `f32::mul_add`——同一前端（`intrinsics::fma`
/// → LLVM `llvm.fma.f32`），宿主经 glibc `fmaf`（IEEE 754 融合，精确一次
/// 舍入），GPU 经 nvptx 后端直落 `fma.rn.f32`；两者同为**正确舍入的融合
/// 结果**，逐位相同 ⇒ kernel↔reference 仍可达位对齐。
///
/// 本 nightly 上 `f32::mul_add` 的自有方法面已迁到 `core::f32::math`
/// （`core_float_math` #137578）——`core::intrinsics::fma` 是同一物（冒烟
/// 双向验证：两写法产出的 PTX 逐字相同；`a·b + c` 字面量则产 16×mul +
/// 16×add，无收缩）。
#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    core::f32::math::mul_add(a, b, c)
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
    // 域钳（与 reference::expf 镜像同式）：指数域直加只在正规数指数域
    // （kf ∈ [-126,127]）成立——域外回绕出垃圾（负值/NaN），且 CPU 与 GPU
    // 的垃圾还不一致（NaN payload 异源）。kf < -126（x < -87.34）真值
    // ≤ 2^-126，落 0.0；kf > 127（x > 88.03）真值溢出，落 +inf。
    // vLLM/torch 同口径（真 underflow/overflow）。
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
/// 分离乘加而非 mul_add（W2 FMA 落地前：该 nightly 上 core 的 f32::mul_add
/// 曾在未稳定 core_float_math #137578 门后——现由 `fma()` 统一走这道门）。
/// **本 kernel 故意保留 mul.rn.f32 + add.rn.f32 两条**：它是"字面量不收缩"
/// 的活证据（k2 断言之一），热路径的点积全部走 `fma()`。
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
/// - 只用正确舍入原语：mul.rn / add.rn / div.rn / sqrt.rn / **fma.rn**——
///   禁 rsqrt(MUFU) 近似；FMA 是单次舍入的正确舍入算子，两侧同式同位即
///   可位位相等（`fma()` 的合同见文件头）
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
            acc = fma(v, v, acc);
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

/// KV scatter（**f32 池**）：源仍是 bf16（k_norm / v_proj 的输出），写入端做
/// 精确加宽 `bf16→f32`（尾数左移 16，无舍入）⇒ 池里每一位都是原 bf16 值的
/// 精确表示。故 f32 池**不改任何数值**：attention 读到的数、读到的顺序、
/// 累加序全同 bf16 档，只把"每 4 维 1 条 u64 载 + 4 条位展"换成"1 条 16B
/// 向量载"。写侧流量也翻倍（4B/维），但每 token 每层只有 2×4×256 维 = 4KB，
/// 相对读侧是噪声。
#[no_mangle]
pub unsafe extern "ptx-kernel" fn kv_scatter_f32(
    src_k: *const u16,
    src_v: *const u16,
    block_table: *const u32,
    seq_ids: *const u32,
    positions: *const u32,
    k_dst: *mut f32,
    v_dst: *mut f32,
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
    *k_dst.add(slot * he + e) = bf16_to_f32(*src_k.add(t * he + e));
    *v_dst.add(slot * he + e) = bf16_to_f32(*src_v.add(t * he + e));
}

/// varlen paged attention：一 block 一 (t,qh)（block=head_dim 线程）。
/// 旧形态"一线程一 (t,qh)"在 m=100 decode 时全 GPU 仅 3200 线程、
/// 每线程 256-f32 本地数组 + 每 e 重走页表——1.08ms/launch vs 25µs
/// 带宽地板。四相位块协作，**逐元素累加序与 reference.rs 全同**
/// （dot：j 属主单线程 d 升序；denom：线程 0 串行 j 升序
/// `p_j + denom`；加权 V：每 e 独立 j 升序）——位对齐零漂移：
/// · A 记分：线程 τ 跨步认领 j∈{τ,τ+bdim,…}，整条点积单线程
///   d 升序（u64 载 4×bf16——hd%4 守卫 + 标量回退），sc 落 scratch
/// · B 归一：各线程冗余扫 max（纯比较值同）→ j 属主原地 p=expf(sc−m)
/// · C 分母：线程 0 串行 j 升序（操作数序逐字同参考）
/// · D 加权 V：线程 τ=e 认领输出维，warp 内连续 e 合并访存，j 升序
/// scratch 是 kernel 参数（op 预分配 [块, slot_cap+2] f32——stdarch nvptx
/// 跨线程通道三断，argmax_rows/rms 同款 blockIdx 私有槽 + syncthreads）。
/// 因果只看 0..=pos；GQA：kv 头 = qh // group（HF repeat_kv 语义）。
/// **序列长无编译期上限**（W2）：`slot_cap` 是运行时槽宽（分值容量），
/// `pos+1 ≤ slot_cap` 走直路（上面的四相位，分值落 scratch），否则走
/// 流式路——两遍 j：遍 1 求块内 max（每线程寄存器极值 + 一次跨线程归约），
/// 遍 2 按 chunk 复算分值 p 并按 **j 升序**累加 denom（线程 0）与加权 V
/// （每线程一输出维）。两路**逐位同**：同一 dot 代码（`attn_dot`，d 升序）、
/// 同一 `expf(score − m)`、同一 j 升序累加/除法——容量只改 chunk 划分，
/// 不改任何一条算术序（chunk 均按 j 升序推进，denom 顺序拼接后 = 0..=pos）。
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
    slot_cap: u32,
    scratch: *mut f32,
) {
    let hd = head_dim as usize;
    let qh_n = q_heads as usize;
    let blk = _block_idx_x() as usize; // block = (t, qh)
    if blk >= tokens as usize * qh_n {
        return;
    }
    let (t, qh) = (udiv(blk, qh_n), urem(blk, qh_n));
    let kh = udiv(qh, udiv(qh_n, kv_heads as usize)); // GQA：kv 头 = qh // group
    let s = *seq_ids.add(t) as usize;
    let pos = *positions.add(t) as usize;
    let (bpr, bs) = (bpr as usize, block_size as usize);
    let he = kv_heads as usize * hd;
    let qbase = blk * hd; // (t·qh_n + qh)·hd
    let e = _thread_idx_x() as usize;
    let bdim = _block_dim_x() as usize;
    let cap = slot_cap as usize; // 分值/p 容量；分母槽 = sc[cap]
    let sc = scratch.add(blk * (cap + 2)); // 块私有槽：[0,cap)=分值/p，[cap]=分母
    let qp = q.add(qbase);
    let vec4 = hd % 4 == 0;
    if pos + 1 > cap {
        // ── 流式路（超槽宽）：保序两遍，逐位同直路 ──
        // 前提：block=head_dim（op 的启动几何）⇒ 每线程恰一输出维；
        // cap ≥ bdim（max 归约每线程一格，op 以 cap=256 ≥ 256 hd 保证）
        let mut j = e;
        let mut lm = f32::NEG_INFINITY;
        while j <= pos {
            let kbase = slot_of(block_table, s, j, bpr, bs) * he + kh * hd;
            let v = attn_dot(qp, k_cache.add(kbase), hd, vec4) * q_scale;
            lm = if v > lm { v } else { lm };
            j += bdim;
        }
        *sc.add(e) = lm;
        _syncthreads();
        // 块内 max：各线程冗余扫（纯比较——值与序无关位同），与直路 B 相同
        let mut m = f32::NEG_INFINITY;
        let mut u = 0usize;
        while u < bdim {
            let x = *sc.add(u);
            m = if x > m { x } else { m };
            u += 1;
        }
        _syncthreads();
        let mut denom = 0.0f32;
        let mut acc = 0.0f32;
        let mut c0 = 0usize;
        while c0 <= pos {
            let c1 = if c0 + cap - 1 < pos { c0 + cap - 1 } else { pos }; // 闭区间
            let mut j = c0 + e;
            while j <= c1 {
                let kbase = slot_of(block_table, s, j, bpr, bs) * he + kh * hd;
                *sc.add(j - c0) = expf(attn_dot(qp, k_cache.add(kbase), hd, vec4) * q_scale - m);
                j += bdim;
            }
            _syncthreads();
            // 分母：线程 0 串行 j 升序（chunk 升序拼接 = 全局 j 升序）
            if e == 0 {
                let mut jj = c0;
                while jj <= c1 {
                    denom = *sc.add(jj - c0) + denom;
                    jj += 1;
                }
            }
            // 加权 V：线程 e 认领输出维，j 升序（跨 chunk 寄存器续累加）
            if e < hd {
                let mut jj = c0;
                while jj <= c1 {
                    let vbase = slot_of(block_table, s, jj, bpr, bs) * he + kh * hd;
                    acc = fma(*sc.add(jj - c0), bf16_to_f32(*v_cache.add(vbase + e)), acc);
                    jj += 1;
                }
            }
            _syncthreads();
            c0 = c1 + 1;
        }
        if e == 0 {
            *sc.add(cap) = denom;
        }
        _syncthreads();
        let denom = *sc.add(cap);
        if e < hd {
            *out.add(qbase + e) = f32_to_bf16(acc / denom);
        }
        return;
    }
    let sc = sc;

    // A 记分：j 属主整条点积（d 升序）→ sc[j]
    let mut j = e;
    while j <= pos {
        let kbase = slot_of(block_table, s, j, bpr, bs) * he + kh * hd;
        *sc.add(j) = attn_dot(qp, k_cache.add(kbase), hd, vec4) * q_scale;
        j += bdim;
    }
    _syncthreads();
    // B 各线程冗余扫 max（纯比较——值与序无关位同）→ j 属主原地写 p
    let mut m = f32::NEG_INFINITY;
    let mut jj = 0usize;
    while jj <= pos {
        let scj = *sc.add(jj);
        m = if scj > m { scj } else { m };
        jj += 1;
    }
    let mut j = e;
    while j <= pos {
        *sc.add(j) = expf(*sc.add(j) - m);
        j += bdim;
    }
    _syncthreads();
    // C 分母：线程 0 串行 j 升序——`p_j + denom` 操作数序逐字同参考
    if e == 0 {
        let mut denom = 0.0f32;
        let mut jj = 0usize;
        while jj <= pos {
            denom = *sc.add(jj) + denom;
            jj += 1;
        }
        *sc.add(cap) = denom;
    }
    _syncthreads();
    let denom = *sc.add(cap);
    // D 加权 V：线程 e 认领输出维（+bdim 步进覆盖 hd>bdim），j 升序
    let mut ee = e;
    while ee < hd {
        let mut acc = 0.0f32;
        let mut j = 0usize;
        while j <= pos {
            let vbase = slot_of(block_table, s, j, bpr, bs) * he + kh * hd;
            acc = fma(*sc.add(j), bf16_to_f32(*v_cache.add(vbase + ee)), acc);
            j += 1;
        }
        *out.add(qbase + ee) = f32_to_bf16(acc / denom);
        ee += bdim;
    }
}

/// 单条点积（d 升序、u64 载 4×bf16；vec4 = hd%4==0）——直路/流式路共用
/// 同一段代码，保证"重算分值"与"存取分值"位同
#[inline(always)]
unsafe fn attn_dot(qp: *const u16, kp: *const u16, hd: usize, vec4: bool) -> f32 {
    let mut dot = 0.0f32;
    if vec4 {
        let mut d = 0usize;
        while d < hd {
            let qs = bf16x4(*(qp.add(d) as *const u64));
            let ks = bf16x4(*(kp.add(d) as *const u64));
            let mut l = 0usize;
            while l < 4 {
                dot = fma(bf16_to_f32(qs[l]), bf16_to_f32(ks[l]), dot);
                l += 1;
            }
            d += 4;
        }
    } else {
        let mut d = 0usize;
        while d < hd {
            let a = bf16_to_f32(*qp.add(d));
            let b = bf16_to_f32(*kp.add(d));
            dot = fma(a, b, dot);
            d += 1;
        }
    }
    dot
}

/// u64 拆 4×bf16（小端车道序——与标量 d 升序逐元素对应）
#[inline(always)]
unsafe fn bf16x4(v: u64) -> [u16; 4] {
    [(v & 0xffff) as u16, ((v >> 16) & 0xffff) as u16, ((v >> 32) & 0xffff) as u16, (v >> 48) as u16]
}

/// 16B 对齐的 4×f32 容器：f32 KV 池按 **ld.global.v4.f32 一条指令**取 4 维
/// （对齐由调用点保证：head_dim % 4 == 0、d0 = lane×ND、ND % 4 == 0，池基址
/// 256B 对齐）。池 dtype 的另一半（bf16）走下方的 u64 载 + 位展。
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct F4([f32; 4]);

/// KV 池元素字节宽（编译期由 dtype 定，WIDE=true ⇒ f32 池）
#[inline(always)]
const fn kv_esz<const WIDE: bool>() -> usize {
    if WIDE {
        4
    } else {
        2
    }
}

/// 载入 **ND 个 KV 维**（元素偏移 `off`）到 f32 寄存器：
/// · WIDE（f32 池）：16B 向量载，4 维 1 条指令，无解开
/// · bf16 池：u64 载 4×bf16 + `<<16` 位展（bf16→f32 精确）
/// 两种 dtype **数值逐位相同**（bf16 写入 f32 池是精确加宽），故 dtype 只
/// 改载入形状，不改任何算术序。
#[inline(always)]
unsafe fn kv_ld<const ND: usize, const WIDE: bool>(base: *const u8, off: usize) -> [f32; ND] {
    let mut r = [0.0f32; ND];
    if WIDE {
        let p = base as *const f32;
        let mut i = 0usize;
        while i < ND {
            let v = *(p.add(off + i) as *const F4);
            r[i] = v.0[0];
            r[i + 1] = v.0[1];
            r[i + 2] = v.0[2];
            r[i + 3] = v.0[3];
            i += 4;
        }
    } else {
        let p = base as *const u16;
        let mut i = 0usize;
        while i < ND {
            let v = bf16x4(*(p.add(off + i) as *const u64));
            r[i] = bf16_to_f32(v[0]);
            r[i + 1] = bf16_to_f32(v[1]);
            r[i + 2] = bf16_to_f32(v[2]);
            r[i + 3] = bf16_to_f32(v[3]);
            i += 4;
        }
    }
    r
}

/// 整条点积（d 升序单链；`kp` 已按元素偏移落好字节指针）——dtype 参数化：
/// WIDE 走 f32 载（4 维 1 指令），否则 u64 载 4×bf16。fma 序逐维同旧。
#[inline(always)]
unsafe fn attn_dot_g<const WIDE: bool>(qp: *const u16, kp: *const u8, hd: usize) -> f32 {
    let mut dot = 0.0f32;
    let mut d = 0usize;
    while d < hd {
        let qs = bf16x4(*(qp.add(d) as *const u64));
        let ks = kv_ld::<4, WIDE>(kp, d);
        let mut l = 0usize;
        while l < 4 {
            dot = fma(bf16_to_f32(qs[l]), ks[l], dot);
            l += 1;
        }
        d += 4;
    }
    dot
}

// ───────────────── 长序列分块 attention（按 KV 复用组织） ─────────────────
// 为什么换几何：旧形态 block=(t, q_head)——每个 q 行把整段 KV 各读一遍，
// GQA 下同一个 kv 头的 K/V 被 `group` 个 q 头各读一遍，prefill 时更被 t 个
// token 各读一遍 ⇒ KV 实际读量 = t·q_heads·L（L=序列长）。本几何
// block=(kv 头, token 片, j 分片)，一块内 R=ntok·group 个 q 行共享同一段
// KV ⇒ 读量 ≈ (t/ntok)·kv_heads·L = 旧形态 /(ntok·group)。本仓库无 smem/
// 无 shfl（nvptx 三断），跨块复用只能靠 L2（72MB）：一个 kv 头一层 8k 的
// KV 全长 8MB，常驻无压力。
//
// 位级合同：**jsplit=1 时与直路/流式路逐位相同**——每个输出元素的算术式
// 与累加序一字不改：
// · 分值 = `attn_dot`（同一函数、d 升序、u64 载 4×bf16）× q_scale
// · m = 全域 max（纯比较，与序无关 ⇒ 与旧路同值）
// · p = `expf(score − m)`
// · 分母 `den = p_j + den`（j 升序，跨 chunk 拼接 = 全局 j 升序）
// · 加权 V `acc = p_j·v_e + acc`（j 升序，每维单线程单链）
// · 出 = `f32_to_bf16(acc / den)`
// jsplit>1（decode 形：批太小、块不够用）时按 j 分段求偏 (m_s, l_s, acc_s)
// 再由 `attn_combine` 合并——分段求和改变结合律 ⇒ 该档是**容差档**
// （测试 o31 钉 rel2 + 峰值归一；o30 钉 jsplit=1 的位级）。
//
// 线程映射（单相，两遍共用）：nv = bdim/rows_nom 线程/行，线程 τ →
// r = τ/nv（行）、lane = τ%nv（维起点 = lane·4，以及 j 起点 = j0+lane）。
// 每线程 4 维累加器（u64 载 4×bf16 V）、j 跨步 nv。
/// 分块档槽位表填充：本 chunk 每个 j 的 paged KV 槽位只查一次，写进
/// `sb[u·cc + (j−c0)]`（u = token 的片内序号）。块内所有 q 行/线程共享。
#[inline(always)]
unsafe fn tile_fill_slots(
    sb: *mut f32,
    block_table: *const u32,
    seq_ids: *const u32,
    t0: usize,
    tn: usize,
    c0: usize,
    cc: usize,
    bpr: usize,
    bs: usize,
    tau: usize,
    bdim: usize,
) {
    let nent = tn * cc;
    let mut e = tau;
    while e < nent {
        let u = udiv(e, cc);
        let j = c0 + urem(e, cc);
        let s = *seq_ids.add(t0 + u) as usize;
        *sb.add(e) = slot_of(block_table, s, j, bpr, bs) as f32;
        e += bdim;
    }
}

/// 两遍档主体（dtype 参数化）：`WIDE=false` 跑 bf16 池（位级资产那一条，
/// `paged_attn_tile`），`WIDE=true` 跑 f32 池（`paged_attn_tile_f32`）。
/// 两者数值逐位相同（bf16→f32 加宽精确），只改 K/V 的载入指令形状。
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_body<const WIDE: bool>(
    q: *const u16,
    k_cache: *const u8,
    v_cache: *const u8,
    block_table: *const u32,
    seq_ids: *const u32,
    positions: *const u32,
    out: *mut u16,
    part_acc: *mut f32,
    part_ml: *mut f32,
    q_scale: f32,
    bpr: u32,
    block_size: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    tokens: u32,
    ntok: u32,
    jsplit: u32,
    chunk: u32,
    scratch: *mut f32,
) {
    let hd = head_dim as usize;
    let qh_n = q_heads as usize;
    let kv_n = kv_heads as usize;
    let ng = udiv(qh_n, kv_n); // GQA group（q 头/kv 头）
    let nt = ntok as usize;
    let js = jsplit as usize;
    let toks = tokens as usize;
    let bdim = _block_dim_x() as usize;
    let tau = _thread_idx_x() as usize;
    let ntile = udiv(toks + nt - 1, nt);
    let blk = _block_idx_x() as usize;
    if blk >= kv_n * ntile * js {
        return;
    }
    let kh = udiv(blk, ntile * js);
    let rest = urem(blk, ntile * js);
    let tile = udiv(rest, js);
    let sp = urem(rest, js);
    let t0 = tile * nt;
    let tn = if t0 + nt < toks { nt } else { toks - t0 };
    let rows_nom = nt * ng; // 命名行数（映射用，末片有空行）
    let nv = udiv(bdim, rows_nom);
    let r = udiv(tau, nv);
    let lane = urem(tau, nv);
    let rows = tn * ng; // 实际行数
    let vr = r < rows;
    let (bpr, bs) = (bpr as usize, block_size as usize);
    let he = kv_n * hd;
    // 本块的 j 轴：片内 token 位置的最大值定分段长（末片 token 少但仍共享）
    let mut pmax = 0usize;
    let mut u = 0usize;
    while u < tn {
        let p = *positions.add(t0 + u) as usize;
        pmax = if p > pmax { p } else { pmax };
        u += 1;
    }
    let lspan = udiv(pmax + js, js); // ceil((pmax+1)/js)
    let jmax = pmax + 1;
    let j0 = sp * lspan;
    let j1 = if (sp + 1) * lspan < jmax { (sp + 1) * lspan } else { jmax };
    let rglob = if vr { (t0 + udiv(r, ng)) * qh_n + kh * ng + urem(r, ng) } else { 0 };
    // 空分片：落 l=0（combine 跳过），保证 part 表被完整覆盖
    if j0 >= j1 {
        if vr && lane == 0 {
            *part_ml.add((rglob * js + sp) * 2) = f32::NEG_INFINITY;
            *part_ml.add((rglob * js + sp) * 2 + 1) = 0.0;
        }
        return;
    }
    // 块私有 scratch（与 host `ops::ATTN_TILE_SLOTS` 同式同值：max 归约
    // 1024 + p 暂存 chunk×rows + den 槽 rows + 槽位表 4×chunk）——块间并发，
    // 不偏移就是全体块共写同一片槽（o30 逮到的错形态）
    let scratch = scratch.add(blk * TILE_SLOT);
    let scm = scratch; // [bdim] 行内 max 归约
    let pb = scratch.add(bdim); // [chunk × rows_nom] p 暂存
    let cap = chunk as usize;
    // 槽位表 [nt × chunk]：一 chunk 内每个 j 的 block_table 查表只做一次，块内
    // 全体行共享。否则 B2 里每个线程的每个 j 都要一次「bt → V」两级依赖载入，
    // 那是分块档实测的唯一瓶颈（读量已降到 1/8，却被延迟吃回去）。
    let sb = pb.add(cap * rows_nom + rows_nom); // den 槽（rows_nom）之后
    let tt = udiv(r, ng); // 行所属 token 的片内序号
    let gq = urem(r, ng);
    let sr = if vr { *seq_ids.add(t0 + tt) as usize } else { 0 };
    let pr = if vr { *positions.add(t0 + tt) as usize } else { 0 };
    let qbase = if vr { ((t0 + tt) * qh_n + kh * ng + gq) * hd } else { 0 };
    let qp = q.add(qbase);
    let esz = kv_esz::<WIDE>();

    // ① 遍 1：分值极值（纯比较——值与序无关）。按 chunk 走（步长 nv 与
    // chunk 宽同余 ⇒ chunk 升序 = 全局 j 升序），每 chunk 复用一次槽位表。
    let mut m = f32::NEG_INFINITY;
    let mut c0 = j0;
    while c0 < j1 {
        let c1 = if c0 + cap - 1 < j1 - 1 { c0 + cap - 1 } else { j1 - 1 };
        let cc = c1 - c0 + 1;
        _syncthreads(); // 读侧屏障（表要重填）
        tile_fill_slots(sb, block_table, seq_ids, t0, tn, c0, cc, bpr, bs, tau, bdim);
        _syncthreads();
        if vr {
            let mut j = c0 + lane;
            while j <= c1 && j <= pr {
                let kbase = *sb.add(tt * cc + (j - c0)) as usize * he + kh * hd;
                let v = attn_dot_g::<WIDE>(qp, k_cache.add(kbase * esz), hd) * q_scale;
                m = if v > m { v } else { m };
                j += nv;
            }
        }
        c0 = c1 + 1;
    }
    if vr {
        *scm.add(tau) = m;
    }
    _syncthreads();
    let mut mm = f32::NEG_INFINITY;
    if vr {
        let mut v2 = 0usize;
        while v2 < nv {
            let x = *scm.add(r * nv + v2);
            mm = if x > mm { x } else { mm };
            v2 += 1;
        }
    }

    // ② 遍 2：chunk 升序复算 p → 分母/加权 V（j 升序累加，跨 chunk 续）
    let d0 = lane * 4; // 本线程 4 个输出维
    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut a3 = 0.0f32;
    let mut den = 0.0f32;
    let mut c0 = j0;
    while c0 < j1 {
        let c1 = if c0 + cap - 1 < j1 - 1 { c0 + cap - 1 } else { j1 - 1 };
        let jr = if pr < c1 { pr } else { c1 }; // 本行该 chunk 的有效末位
        let cc = c1 - c0 + 1;
        _syncthreads();
        tile_fill_slots(sb, block_table, seq_ids, t0, tn, c0, cc, bpr, bs, tau, bdim);
        _syncthreads();
        // B1：本 chunk 的 p → pb[(j−c0)·rows_nom + r]
        if vr {
            let mut j = c0 + lane;
            while j <= jr {
                let kbase = *sb.add(tt * cc + (j - c0)) as usize * he + kh * hd;
                let sc = attn_dot_g::<WIDE>(qp, k_cache.add(kbase * esz), hd) * q_scale;
                *pb.add((j - c0) * rows_nom + r) = expf(sc - mm);
                j += nv;
            }
        }
        _syncthreads();
        // B2：加权 V —— 线程认领 4 维（u64 载 4×bf16），j 升序；地址只查表
        if vr {
            let mut j = c0;
            while j <= jr {
                let p = *pb.add((j - c0) * rows_nom + r);
                let vbase = *sb.add(tt * cc + (j - c0)) as usize * he + kh * hd + d0;
                let vs = kv_ld::<4, WIDE>(v_cache, vbase);
                a0 = fma(p, vs[0], a0);
                a1 = fma(p, vs[1], a1);
                a2 = fma(p, vs[2], a2);
                a3 = fma(p, vs[3], a3);
                if lane == 0 {
                    den = p + den; // 分母一行一份（j 升序，跨 chunk 续）
                }
                j += 1;
            }
        }
        c0 = c1 + 1;
    }

    // ③ 落结果
    if vr && lane == 0 {
        *scm.add(r) = den;
    }
    _syncthreads();
    if vr {
        let d = *scm.add(r);
        if js == 1 {
            *out.add(qbase + d0) = f32_to_bf16(a0 / d);
            *out.add(qbase + d0 + 1) = f32_to_bf16(a1 / d);
            *out.add(qbase + d0 + 2) = f32_to_bf16(a2 / d);
            *out.add(qbase + d0 + 3) = f32_to_bf16(a3 / d);
        } else {
            let base = rglob * js + sp;
            *part_ml.add(base * 2) = mm;
            *part_ml.add(base * 2 + 1) = d;
            let pb0 = part_acc.add(base * PART_HD_STRIDE + d0);
            *pb0 = a0;
            *pb0.add(1) = a1;
            *pb0.add(2) = a2;
            *pb0.add(3) = a3;
        }
    }
}

// ── 导出：两遍档（位级资产 + 回退口）与单遍档的各几何/dtype 档 ──
// 每个 `ptx-kernel` 都是一份**单态实体**（`#[inline(always)]` 主体展开后无
// call）：寄存器用量按档独立，混在一个 kernel 里会被最费档拽高（ptxas 按
// 函数报告 regs），进而压掉 bdim 合法档位。
macro_rules! tile_export {
    ($name:ident, $wide:literal) => {
        #[no_mangle]
        pub unsafe extern "ptx-kernel" fn $name(
            q: *const u16,
            k_cache: *const u8,
            v_cache: *const u8,
            block_table: *const u32,
            seq_ids: *const u32,
            positions: *const u32,
            out: *mut u16,
            part_acc: *mut f32,
            part_ml: *mut f32,
            q_scale: f32,
            bpr: u32,
            block_size: u32,
            q_heads: u32,
            kv_heads: u32,
            head_dim: u32,
            tokens: u32,
            ntok: u32,
            jsplit: u32,
            chunk: u32,
            scratch: *mut f32,
        ) {
            tile_body::<$wide>(
                q, k_cache, v_cache, block_table, seq_ids, positions, out, part_acc, part_ml,
                q_scale, bpr, block_size, q_heads, kv_heads, head_dim, tokens, ntok, jsplit, chunk,
                scratch,
            )
        }
    };
}

/// 两遍档 bf16 池（o30 位级资产；几何 ntok=2/ND=4，一字不改）
tile_export!(paged_attn_tile, false);
/// 两遍档 f32 池（与上逐位同值：bf16→f32 加宽精确）
tile_export!(paged_attn_tile_f32, true);

macro_rules! online_export {
    ($name:ident, $nd:literal, $rpt:literal, $wide:literal) => {
        #[no_mangle]
        pub unsafe extern "ptx-kernel" fn $name(
            q: *const u16,
            k_cache: *const u8,
            v_cache: *const u8,
            block_table: *const u32,
            seq_ids: *const u32,
            positions: *const u32,
            out: *mut u16,
            part_acc: *mut f32,
            part_ml: *mut f32,
            q_scale: f32,
            bpr: u32,
            block_size: u32,
            q_heads: u32,
            kv_heads: u32,
            head_dim: u32,
            tokens: u32,
            ntok: u32,
            jsplit: u32,
            chunk: u32,
            scratch: *mut f32,
        ) {
            online_body::<$nd, $rpt, $wide>(
                q, k_cache, v_cache, block_table, seq_ids, positions, out, part_acc, part_ml,
                q_scale, bpr, block_size, q_heads, kv_heads, head_dim, tokens, ntok, jsplit, chunk,
                scratch,
            )
        }
    };
}

/// 基线档（ND=4 / 1 行每线程 / bf16）——o33 的对照列
online_export!(paged_attn_online, 4, 1, false);
/// A 档：8 维每线程（bdim 512 时 ntok 可到 4）
online_export!(paged_attn_online_n8, 8, 1, false);
/// A 档：16 维每线程（bdim 512 时 ntok 可到 8）
online_export!(paged_attn_online_n16, 16, 1, false);
/// R 档：2 行每线程（K/V 载入 ÷2）
online_export!(paged_attn_online_r2, 4, 2, false);
/// R 档：4 行每线程（= group，K/V 载入 ÷4）
online_export!(paged_attn_online_r4, 4, 4, false);
/// R 档：4 行 × 8 维每线程
online_export!(paged_attn_online_n8_r4, 8, 4, false);
/// BR 档：4 行 × 16 维每线程——一块覆盖 2×q 行（BR=16 时 bdim 仍 512）
online_export!(paged_attn_online_n16_r4, 16, 4, false);
/// BR 档：2 行 × 16 维每线程（BR=16、bdim 512、acc 32 regs 的镜像档）
online_export!(paged_attn_online_n16_r2, 16, 2, false);
/// B 档：f32 池（无位展）
online_export!(paged_attn_online_f32, 4, 1, true);
/// B×A：f32 池 + 8 维每线程
online_export!(paged_attn_online_f32_n8, 8, 1, true);
/// B×R：f32 池 + 4 行每线程
online_export!(paged_attn_online_f32_r4, 4, 4, true);
/// B×R：f32 池 + 2 行每线程（decode 形：r4 会把 bdim 压到一个 warp）
online_export!(paged_attn_online_f32_r2, 4, 2, true);

// ── 几何/dtype 三轴：ND 维每线程、RPT 行每线程、KV 池 dtype（WIDE=f32） ──
//
// **每 KV 字节服务的 q 行数 = RPT**，这是唯一有效的杠杆：块内 rows_nom 个 q
// 行分给 rows_nom/RPT 个线程组，一组线程对同一 j 只载入一次 K（整行）与 V
// （本线程 ND 维），再与 RPT 条 q 行分别乘加 ⇒ K/V 的 L2 载入次数 ÷RPT。
// 行组不跨 token（host 保证 group % RPT == 0），故槽表索引每线程一份、每条
// 分值仍由单线程按 d 升序算完（位级合同的形式不变）。
//
// 抬 ntok 只改"块数 × 每块行数"之比：KV 总载入量 = T·group·(1+2/RPT)/2 倍
// 独占量，**与 ntok 无关**（bench 实测：ntok 2→4 的流量不变）；q 行每 j 重读
// （整条点积要求单线程持全 hd 的 q，寄存器放不下），它命中 L1（块内 4KB），
// 故 RPT 做大后 q 变成指令侧的主项。这就是 dtype（WIDE）那条线的由来：
// f32 池把"每 4 维 1 条载入 + 4 条位展"压成"1 条 16B 向量载"，指令 ÷~3，
// 代价是字节 ×2。
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn online_body<const ND: usize, const RPT: usize, const WIDE: bool>(
    q: *const u16,
    k_cache: *const u8,
    v_cache: *const u8,
    block_table: *const u32,
    seq_ids: *const u32,
    positions: *const u32,
    out: *mut u16,
    part_acc: *mut f32,
    part_ml: *mut f32,
    q_scale: f32,
    bpr: u32,
    block_size: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    tokens: u32,
    ntok: u32,
    jsplit: u32,
    chunk: u32,
    scratch: *mut f32,
) {
    let hd = head_dim as usize;
    let qh_n = q_heads as usize;
    let kv_n = kv_heads as usize;
    let ng = udiv(qh_n, kv_n);
    let nt = ntok as usize;
    let js = jsplit as usize;
    let toks = tokens as usize;
    let bdim = _block_dim_x() as usize;
    let tau = _thread_idx_x() as usize;
    let ntile = udiv(toks + nt - 1, nt);
    let blk = _block_idx_x() as usize;
    if blk >= kv_n * ntile * js {
        return;
    }
    let kh = udiv(blk, ntile * js);
    let rest = urem(blk, ntile * js);
    let tile = udiv(rest, js);
    let sp = urem(rest, js);
    let t0 = tile * nt;
    let tn = if t0 + nt < toks { nt } else { toks - t0 };
    let rows_nom = nt * ng;
    let ngrp = udiv(rows_nom, RPT); // 线程组数（每组 RPT 行）
    let nv = udiv(bdim, ngrp); // 每组线程数 = j 步长 = hd/ND
    let rg = udiv(tau, nv);
    let lane = urem(tau, nv);
    let r0 = rg * RPT; // 本线程组的首行
    let rows = tn * ng;
    let vr = r0 < rows;
    let (bpr, bs) = (bpr as usize, block_size as usize);
    let he = kv_n * hd;
    // j 轴分段（与两遍档同式：块内最大位置定段长，末片 token 少但仍共享）
    let mut pmax = 0usize;
    let mut u = 0usize;
    while u < tn {
        let p = *positions.add(t0 + u) as usize;
        pmax = if p > pmax { p } else { pmax };
        u += 1;
    }
    let lspan = udiv(pmax + js, js);
    let jmax = pmax + 1;
    let j0 = sp * lspan;
    let j1 = if (sp + 1) * lspan < jmax { (sp + 1) * lspan } else { jmax };
    // 行组基：组内 RPT 行同 token（ng % RPT == 0）⇒ tt/gq0 只算一次
    let tt = udiv(r0, ng);
    let gq0 = urem(r0, ng);
    let pr = if vr { *positions.add(t0 + tt) as usize } else { 0 };
    if j0 >= j1 {
        if vr && lane == 0 {
            let mut k = 0usize;
            while k < RPT {
                let rglob = (t0 + tt) * qh_n + kh * ng + gq0 + k;
                *part_ml.add((rglob * js + sp) * 2) = f32::NEG_INFINITY;
                *part_ml.add((rglob * js + sp) * 2 + 1) = 0.0;
                k += 1;
            }
        }
        return;
    }
    // scratch 布局与两遍档同（复用同一份分配，host 侧零改动）：
    // scm[bdim] 行内 max 归约 / pb[chunk×rows_nom] 分值→p 暂存 / sb[nt×chunk] 槽位表
    let scratch = scratch.add(blk * TILE_SLOT);
    // 行内 max 归约表按 **(行, lane) 全对** 定尺：rows_nom × nv = bdim × RPT。
    // RPT=1 时恰好等于 bdim（老布局）；RPT>1 时 bdim 只有 rows_nom/RPT 组，
    // 拿 bdim 当长度会踩到 pb 区（曾实测：RPT>1 输出全错）。
    let scm = scratch;
    let scm_len = rows_nom * nv;
    let pb = scratch.add(scm_len);
    let cap = chunk as usize;
    let sb = pb.add(cap * rows_nom + rows_nom);
    let d0 = lane * ND; // 本线程的 ND 个输出维
    // 组内**首行**的 q/out 行基址：组内第 k 行 = qb0 + k·hd（行相邻）。
    // 存基址而不是 `[usize; RPT]`——数组版实测落 192B 的 `__local_depot`
    // （qb 32B + mrun/lrun 各 16B + …，见 ptxas -v 的 stack frame），
    // 每次取址走 ld.local；改成"基址 + 编译期常量偏移"后归零。
    let mut qb0 = 0usize;
    if vr {
        qb0 = ((t0 + tt) * qh_n + kh * ng + gq0) * hd;
    }

    // 在线状态（每行一份，跨 chunk 续）：m = 已见 j 全域 max、l = Σexp(·−m)、
    // acc = Σexp(·−m)·V。l 由**每线程各自**按同一 j 序累加（同值冗余）——
    // 省掉两遍档末尾那道 den 屏障（行内所有线程算出逐位相同的 l）。
    let mut mrun = [f32::NEG_INFINITY; RPT];
    let mut lrun = [0.0f32; RPT];
    let mut acc = [[0.0f32; ND]; RPT];
    let mut c0 = j0;
    while c0 < j1 {
        let c1 = if c0 + cap - 1 < j1 - 1 { c0 + cap - 1 } else { j1 - 1 };
        let cc = c1 - c0 + 1;
        let jr = if pr < c1 { pr } else { c1 }; // 本组该 chunk 的有效末位
        _syncthreads(); // 读侧屏障（表要重填）
        tile_fill_slots(sb, block_table, seq_ids, t0, tn, c0, cc, bpr, bs, tau, bdim);
        _syncthreads();
        // B1a：分值（j 属主整条点积，K 整行载入一次供 RPT 行共享）→ pb 原值
        let mut mx = [f32::NEG_INFINITY; RPT];
        if vr {
            let mut j = c0 + lane;
            while j <= jr {
                let kbase = *sb.add(tt * cc + (j - c0)) as usize * he + kh * hd;
                let mut s = [0.0f32; RPT];
                let mut d = 0usize;
                while d < hd {
                    let ks = kv_ld::<ND, WIDE>(k_cache, kbase + d);
                    let mut k = 0usize;
                    while k < RPT {
                        let qs = kv_ld::<ND, false>(q as *const u8, qb0 + k * hd + d);
                        let mut l = 0usize;
                        while l < ND {
                            s[k] = fma(qs[l], ks[l], s[k]);
                            l += 1;
                        }
                        k += 1;
                    }
                    d += ND;
                }
                let mut k = 0usize;
                while k < RPT {
                    let sc = s[k] * q_scale;
                    *pb.add((j - c0) * rows_nom + r0 + k) = sc;
                    mx[k] = if sc > mx[k] { sc } else { mx[k] };
                    k += 1;
                }
                j += nv;
            }
            let mut k = 0usize;
            while k < RPT {
                *scm.add((r0 + k) * nv + lane) = mx[k];
                k += 1;
            }
        }
        _syncthreads();
        // 行内 max（各线程冗余扫 nv 格，纯比较——值与序无关）
        let mut mc = [f32::NEG_INFINITY; RPT];
        if vr {
            let mut k = 0usize;
            while k < RPT {
                let mut v2 = 0usize;
                while v2 < nv {
                    let x = *scm.add((r0 + k) * nv + v2);
                    mc[k] = if x > mc[k] { x } else { mc[k] };
                    v2 += 1;
                }
                k += 1;
            }
        }
        // 在线合并系数：m 可能仍为 −inf（本行本 chunk 无有效 j，或整个分片空）
        // ——域钳掉，避免 exp(−inf − (−inf)) = NaN 从空 chunk 漏进 l/acc。
        let mut mn = [0.0f32; RPT];
        let mut scale = [0.0f32; RPT];
        let mut echunk = [0.0f32; RPT];
        {
            let mut k = 0usize;
            while k < RPT {
                let x = if mc[k] > mrun[k] { mc[k] } else { mrun[k] };
                mn[k] = x;
                scale[k] = if mrun[k] == f32::NEG_INFINITY { 0.0 } else { expf(mrun[k] - x) };
                echunk[k] = if mc[k] == f32::NEG_INFINITY { 0.0 } else { expf(mc[k] - x) };
                k += 1;
            }
        }
        // B1b：p = exp(分值 − 本 chunk max) ∈ (0,1]（j 属主原地写）
        if vr {
            let mut j = c0 + lane;
            while j <= jr {
                let mut k = 0usize;
                while k < RPT {
                    let o = (j - c0) * rows_nom + r0 + k;
                    *pb.add(o) = expf(*pb.add(o) - mc[k]);
                    k += 1;
                }
                j += nv;
            }
        }
        _syncthreads();
        // B2：加权 V（线程认领 ND 维、j 升序全扫；V 载入一次供 RPT 行共享）
        // → 本 chunk 偏量 s[ND]/dl，再与 running 状态合并（rescale 收在 chunk 边界）
        if vr {
            let mut s = [[0.0f32; ND]; RPT];
            let mut dl = [0.0f32; RPT];
            let mut j = c0;
            while j <= jr {
                let vbase = *sb.add(tt * cc + (j - c0)) as usize * he + kh * hd + d0;
                let vs = kv_ld::<ND, WIDE>(v_cache, vbase);
                let mut k = 0usize;
                while k < RPT {
                    let p = *pb.add((j - c0) * rows_nom + r0 + k);
                    let mut l = 0usize;
                    while l < ND {
                        s[k][l] = fma(p, vs[l], s[k][l]);
                        l += 1;
                    }
                    dl[k] = p + dl[k];
                    k += 1;
                }
                j += 1;
            }
            let mut k = 0usize;
            while k < RPT {
                let mut l = 0usize;
                while l < ND {
                    acc[k][l] = fma(echunk[k], s[k][l], acc[k][l] * scale[k]);
                    l += 1;
                }
                lrun[k] = fma(echunk[k], dl[k], lrun[k] * scale[k]);
                k += 1;
            }
        }
        let mut k = 0usize;
        while k < RPT {
            mrun[k] = mn[k];
            k += 1;
        }
        c0 = c1 + 1;
    }

    if vr {
        let mut k = 0usize;
        while k < RPT {
            if js == 1 {
                let mut l = 0usize;
                while l < ND {
                    *out.add(qb0 + k * hd + d0 + l) = f32_to_bf16(acc[k][l] / lrun[k]);
                    l += 1;
                }
            } else {
                // 偏量语义与两遍档同（m = 本分片 max），attn_combine 直接可吃
                let rglob = (t0 + tt) * qh_n + kh * ng + gq0 + k;
                let base = rglob * js + sp;
                if lane == 0 {
                    *part_ml.add(base * 2) = mrun[k];
                    *part_ml.add(base * 2 + 1) = lrun[k];
                }
                let pb0 = part_acc.add(base * PART_HD_STRIDE + d0);
                let mut l = 0usize;
                while l < ND {
                    *pb0.add(l) = acc[k][l];
                    l += 1;
                }
            }
            k += 1;
        }
    }
}

/// 分块档块私有 scratch 槽数：与 host 侧 `ops::ATTN_TILE_SLOTS` 同值。
/// `ROWS_CAP` = 一块内 q 行上限（host 侧 `ATTN_TILE_ROWS_CAP`，BR 维度的
/// 上界 = ROWS_CAP/group）。四项的定尺依据：
/// · 行内 max 归约表（scm）= rows_nom × nv = bdim × rpt ≤ 512 × 4 = 2048
/// · p 暂存 = chunk × rows_nom
/// · den 表 = rows_nom
/// · 槽位表 = ntok × chunk ≤ 32 × 512（host 侧 `ATTN_TILE_SLOT_TAB` 同尺，
///   越界由 host 的 `launch_tile` assert 兜住）
const ROWS_CAP: usize = 128;
/// 槽位表定尺 = ntok(BR) 上限 32 × chunk 上限 512（host 侧 `ATTN_TILE_SLOT_TAB`
/// 同值）。**两处常数必须逐值相同**：块私有槽 stride 不一致 = 块间互踩（静默错值）
const SLOT_TAB: usize = 32 * 512;
const TILE_SLOT: usize = 2048 + 256 * ROWS_CAP + ROWS_CAP + SLOT_TAB;

/// 偏结果表的行距（head_dim 方向容量）：与 host 侧 `ops::ATTN_TILE_HD_CAP`
/// 同值——表按 `[rows, split, PART_HD_STRIDE]` 预分配，head_dim 更小时尾部不用。
const PART_HD_STRIDE: usize = 256;

/// j 分片合并：偏 (m_s, l_s, acc_s) → 输出。s 升序累加（确定性）。
/// 只有一分片非空时与直路**逐位相同**：l = l_s·expf(0) = l_s、
/// acc = acc_s·expf(0) = acc_s（expf(0) 精确为 1，乘 1 精确）；
/// 加 0.0 精确不改值 ⇒ `acc/l` 与原式同。空分片（l=0）直接跳过。
///
/// **两种上游同吃**（语义同一套代数，本函数不动）：两遍档 `paged_attn_tile`
/// 的偏量是"全局 max 基下的单链求和"，单遍档 `paged_attn_online` 的偏量是
/// 在线 softmax 收敛后的 (running max, running l, running acc)——后者与前者
/// 的差别只在累加路径（chunk 内 rescale），三者恰好就是本函数要的
/// "m_s、Σexp(·−m_s)、Σexp(·−m_s)·V"。故 rescale 到全局 max 的那一步
/// `e = expf(m_s − m)` 对两种上游都成立，一字不改。
#[no_mangle]
pub unsafe extern "ptx-kernel" fn attn_combine(
    part_acc: *const f32,
    part_ml: *const f32,
    out: *mut u16,
    nrows: u32,
    jsplit: u32,
    head_dim: u32,
) {
    let hd = head_dim as usize;
    let js = jsplit as usize;
    let i = tid() as usize;
    if i >= nrows as usize * hd {
        return;
    }
    let r = udiv(i, hd);
    let d = urem(i, hd);
    let base = part_ml.add(r * js * 2);
    let mut m = f32::NEG_INFINITY;
    let mut s = 0usize;
    while s < js {
        let x = *base.add(s * 2);
        m = if x > m { x } else { m };
        s += 1;
    }
    let mut l = 0.0f32;
    let mut acc = 0.0f32;
    let mut s = 0usize;
    while s < js {
        let ls = *base.add(s * 2 + 1);
        if ls > 0.0 {
            let e = expf(*base.add(s * 2) - m);
            l = fma(ls, e, l);
            acc = fma(*part_acc.add((r * js + s) * PART_HD_STRIDE + d), e, acc);
        }
        s += 1;
    }
    *out.add(i) = f32_to_bf16(acc / l);
}

// ───────────────────── I1 竖切算子（Qwen3.5 文本骨干） ─────────────────────
// 零中心 norm / partial rope / 门控乘 / 深度因果卷积 / gated deltanet /
// 带门 RMSNorm / 设备端 argmax——数值对齐纪律与上文同守。

/// 行平方和的跨线程归约：scratch 是 kernel 参数（op 预分配，每 block 按
/// blockIdx 私有槽 row*bdim）——stdarch nvptx 三条跨线程通道全断：
/// 无 shfl intrinsic、asm! 未实现（无寄存器类）、`#[link_section=".shared"]`
/// 被 rustc 后端忽略（static 落 .global 模块级，并发 block 互踩）。
/// 线程 t 跨步占有元素 {t, t+bdim, ...}（连续线程连续地址——合并访存），
/// 部分和 i 升序；`_syncthreads`（CUDA 语义 = 块内含 global 的可见性
/// 屏障）后线程 0 顺序合并 u 升序 `total = s[u] + total`——reference.rs
/// 按同式同序标量镜像（位对齐合同保持）
#[inline(always)]
unsafe fn block_sum_strided(
    x: *const u16,
    scratch: *mut f32,
    row: usize,
    base: usize,
    h: usize,
    t: usize,
    bdim: usize,
) -> f32 {
    unsafe {
        let slot = scratch.add(row * bdim);
        let mut acc = 0.0f32;
        let mut i = t;
        while i < h {
            let v = bf16_to_f32(*x.add(base + i));
            acc = fma(v, v, acc);
            i += bdim;
        }
        *slot.add(t) = acc;
        _syncthreads();
        if t == 0 {
            let mut total = 0.0f32;
            let mut u = 0usize;
            while u < bdim {
                total = *slot.add(u) + total;
                u += 1;
            }
            *slot = total;
        }
        _syncthreads();
        *slot
    }
}

/// 零中心 RMSNorm（Qwen3.5 形态）：out = x·rsqrt(mean(x²)+eps)·(1+w)。
/// 与 transformers Qwen3_5RMSNorm 同语义：norm 在 f32、(1+w) 在 f32、
/// 单次收 bf16。block-per-row（跨步合并 + scratch 块归约——row-per-thread
/// 串行扫整行在 rows=m 时全 GPU 只 m 个线程在跑，喂不饱 DRAM 延迟）
#[no_mangle]
pub unsafe extern "ptx-kernel" fn rms_norm_zc(
    x: *const u16,
    w: *const u16,
    out: *mut u16,
    hidden: u32,
    eps: f32,
    rows: u32,
    scratch: *mut f32,
) {
    let row = _block_idx_x() as usize;
    if row >= rows as usize {
        return;
    }
    let t = _thread_idx_x() as usize;
    let bdim = _block_dim_x() as usize;
    let h = hidden as usize;
    let base = row * h;
    let total = unsafe { block_sum_strided(x, scratch, row, base, h, t, bdim) };
    let inv = 1.0f32 / core::intrinsics::sqrt::<f32>(total / hidden as f32 + eps);
    let mut i = t;
    while i < h {
        let o = bf16_to_f32(*x.add(base + i)) * inv * (1.0 + bf16_to_f32(*w.add(i)));
        *out.add(base + i) = f32_to_bf16(o);
        i += bdim;
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
/// sigmoid = 1/(1+expf(−g)) 自控 exp 路径；sigmoid 先收 bf16 再乘
/// （HF torch.sigmoid 在 bf16 张量上先落存储、attn 输出后乘——同两跳 round）
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
        let sig = f32_to_bf16(1.0f32 / (1.0f32 + expf(-s)));
        *out.add(i) = f32_to_bf16(v * bf16_to_f32(sig));
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
    state: *mut u16,      // [rows, conv_dim, kernel-1] 原地，行号经 row_id 间接
    row_start: *const u32, // [rows] 本段起点（tokens 轴）
    row_count: *const u32, // [rows] 本段 token 数
    row_id: *const u32,   // [rows] 本段请求的状态行号
    conv_dim: u32,
    kernel: u32,
    rows: u32,
    specs: *const u32, // [rows] 逐 span 投机标志（0 = 逐段滚动；1 = 逐位置落盘）
) {
    let cd = conv_dim as usize;
    let st = (kernel - 1) as usize;
    let i = tid() as usize;
    if i >= rows as usize * cd {
        return;
    }
    let (r, c) = (udiv(i, cd), urem(i, cd));
    // specs[r]==1（MTP 验证链的逐位置落盘）：第 t 个 token（段内局部下标 t）
    // 处理完把窗口写行 row_id[r]+t+1——行 j 即"前 j 个 token 之后"的 conv
    // 状态，与 deltanet 的 spec 分支同构。读写行不相交（读 row_id[r]、
    // 写 row_id[r]+1..=cnt），无原地序依赖；specs[r]==0 路径文本不动。
    // **逐 span 数组**（不是整批标量）：真实一步里普通 decode 行、prefill
    // 行与投机链行同批，标志必须按段取——整批单标志只能表达"全批一条链"。
    if *specs.add(r) != 0 {
        let rbase = *row_id.add(r) as usize * cd * st + c * st;
        let t0 = *row_start.add(r) as usize;
        let cnt = *row_count.add(r) as usize;
        let mut ti = 0usize;
        while ti < cnt {
            let wbase = (*row_id.add(r) as usize + ti + 1) * cd * st + c * st;
            let mut j = 0usize;
            while j < st {
                // 窗口末 st 个 token 以局部下标 li 计（<0 取旧 state 槽位）
                let li = ti as i32 + 1 - st as i32 + j as i32;
                *state.add(wbase + j) = if li < 0 {
                    *state.add(rbase + (li + st as i32) as usize)
                } else {
                    *x.add((t0 + li as usize) * cd + c)
                };
                j += 1;
            }
            ti += 1;
        }
        return;
    }
    let base = *row_id.add(r) as usize * cd * st + c * st;
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

/// kd=128 行驻寄存器的单 (v 行, token) 更新：kvm 趟→delta→更新趟（读 S0
/// 写 S2）→o 累加→core_out 落点（通用路径 kd=128、vd≠128 用；kd=vd=128
/// 走 E2 分支）。行指针/落点下标全走参数。
#[inline(always)]
unsafe fn dn_row128(
    srow: *mut u16,
    k: *const u16,
    q: *const u16,
    v: *const u16,
    core_out: *mut u16,
    vin_ix: usize, // v 读/core_out 写共用落点：(g·vh+h)·vd + v 行号
    decay: f32,
    beta: f32,
    invq: f32,
    invk: f32,
    kd_sqrt: f32,
    qb: usize,
) {
    let mut kvm = 0.0f32;
    macro_rules! kvm4 {
        ($e:expr, $c:literal) => {{
            let (lo, hi) = ($e as u32, ($e >> 32) as u32);
            kvm = fma(f32::from_bits(lo << 16) * decay, bf16_to_f32(*k.add(qb + $c * 4)) * invk, kvm);
            kvm = fma(f32::from_bits(lo & 0xffff_0000) * decay, bf16_to_f32(*k.add(qb + $c * 4 + 1)) * invk, kvm);
            kvm = fma(f32::from_bits(hi << 16) * decay, bf16_to_f32(*k.add(qb + $c * 4 + 2)) * invk, kvm);
            kvm = fma(f32::from_bits(hi & 0xffff_0000) * decay, bf16_to_f32(*k.add(qb + $c * 4 + 3)) * invk, kvm);
        }};
    }
    let w128 = (
        *(srow.add(0) as *const u64), *(srow.add(4) as *const u64),
        *(srow.add(8) as *const u64), *(srow.add(12) as *const u64),
        *(srow.add(16) as *const u64), *(srow.add(20) as *const u64),
        *(srow.add(24) as *const u64), *(srow.add(28) as *const u64),
        *(srow.add(32) as *const u64), *(srow.add(36) as *const u64),
        *(srow.add(40) as *const u64), *(srow.add(44) as *const u64),
        *(srow.add(48) as *const u64), *(srow.add(52) as *const u64),
        *(srow.add(56) as *const u64), *(srow.add(60) as *const u64),
        *(srow.add(64) as *const u64), *(srow.add(68) as *const u64),
        *(srow.add(72) as *const u64), *(srow.add(76) as *const u64),
        *(srow.add(80) as *const u64), *(srow.add(84) as *const u64),
        *(srow.add(88) as *const u64), *(srow.add(92) as *const u64),
        *(srow.add(96) as *const u64), *(srow.add(100) as *const u64),
        *(srow.add(104) as *const u64), *(srow.add(108) as *const u64),
        *(srow.add(112) as *const u64), *(srow.add(116) as *const u64),
        *(srow.add(120) as *const u64), *(srow.add(124) as *const u64),
    );
    kvm4!(w128.0, 0); kvm4!(w128.1, 1); kvm4!(w128.2, 2); kvm4!(w128.3, 3);
    kvm4!(w128.4, 4); kvm4!(w128.5, 5); kvm4!(w128.6, 6); kvm4!(w128.7, 7);
    kvm4!(w128.8, 8); kvm4!(w128.9, 9); kvm4!(w128.10, 10); kvm4!(w128.11, 11);
    kvm4!(w128.12, 12); kvm4!(w128.13, 13); kvm4!(w128.14, 14); kvm4!(w128.15, 15);
    kvm4!(w128.16, 16); kvm4!(w128.17, 17); kvm4!(w128.18, 18); kvm4!(w128.19, 19);
    kvm4!(w128.20, 20); kvm4!(w128.21, 21); kvm4!(w128.22, 22); kvm4!(w128.23, 23);
    kvm4!(w128.24, 24); kvm4!(w128.25, 25); kvm4!(w128.26, 26); kvm4!(w128.27, 27);
    kvm4!(w128.28, 28); kvm4!(w128.29, 29); kvm4!(w128.30, 30); kvm4!(w128.31, 31);
    let vin = bf16_to_f32(*v.add(vin_ix));
    let delta = (vin - kvm) * beta;
    let mut o = 0.0f32;
    macro_rules! upd4 {
        ($e:expr, $c:literal) => {{
            let (lo, hi) = ($e as u32, ($e >> 32) as u32);
            let r0 = f32_to_bf16_hi(fma(bf16_to_f32(*k.add(qb + $c * 4)) * invk, delta, f32::from_bits(lo << 16) * decay));
            let r1 = f32_to_bf16_hi(fma(bf16_to_f32(*k.add(qb + $c * 4 + 1)) * invk, delta, f32::from_bits(lo & 0xffff_0000) * decay));
            let r2 = f32_to_bf16_hi(fma(bf16_to_f32(*k.add(qb + $c * 4 + 2)) * invk, delta, f32::from_bits(hi << 16) * decay));
            let r3 = f32_to_bf16_hi(fma(bf16_to_f32(*k.add(qb + $c * 4 + 3)) * invk, delta, f32::from_bits(hi & 0xffff_0000) * decay));
            // lane0 ← r0>>16；lane1 已在位；lane2 ← r2<<16；lane3 ← r3<<32
            *(srow.add($c * 4) as *mut u64) =
                (r0 >> 16) as u64 | r1 as u64 | (r2 as u64) << 16 | (r3 as u64) << 32;
            o = fma(f32::from_bits(r0), (bf16_to_f32(*q.add(qb + $c * 4)) * invq) / kd_sqrt, o);
            o = fma(f32::from_bits(r1), (bf16_to_f32(*q.add(qb + $c * 4 + 1)) * invq) / kd_sqrt, o);
            o = fma(f32::from_bits(r2), (bf16_to_f32(*q.add(qb + $c * 4 + 2)) * invq) / kd_sqrt, o);
            o = fma(f32::from_bits(r3), (bf16_to_f32(*q.add(qb + $c * 4 + 3)) * invq) / kd_sqrt, o);
        }};
    }
    upd4!(w128.0, 0); upd4!(w128.1, 1); upd4!(w128.2, 2); upd4!(w128.3, 3);
    upd4!(w128.4, 4); upd4!(w128.5, 5); upd4!(w128.6, 6); upd4!(w128.7, 7);
    upd4!(w128.8, 8); upd4!(w128.9, 9); upd4!(w128.10, 10); upd4!(w128.11, 11);
    upd4!(w128.12, 12); upd4!(w128.13, 13); upd4!(w128.14, 14); upd4!(w128.15, 15);
    upd4!(w128.16, 16); upd4!(w128.17, 17); upd4!(w128.18, 18); upd4!(w128.19, 19);
    upd4!(w128.20, 20); upd4!(w128.21, 21); upd4!(w128.22, 22); upd4!(w128.23, 23);
    upd4!(w128.24, 24); upd4!(w128.25, 25); upd4!(w128.26, 26); upd4!(w128.27, 27);
    upd4!(w128.28, 28); upd4!(w128.29, 29); upd4!(w128.30, 30); upd4!(w128.31, 31);
    *core_out.add(vin_ix) = f32_to_bf16(o);
}

/// gated deltanet（recurrent correctness-first）：kd=vd=128 走 E2 重映射
/// （一块 = (行, v 头) × 4 warp，warp 顺序吃 v 行、lane 持 j 切片 u64
/// 全合并——见 E2 分支内注释）；其余一块 = (请求行, v 头)，一线程一 v 维
/// （线程私有 state 行 S[v][0..k_dim]，全局内存 L1 驻留）。
/// 每 token：q/k 的 l2norm 倒数（f32，eps=1e-6）由前置 kernel
/// `dn_qk_l2norm` 逐 (token, k 头) 算一次（本 kernel 只读）→ q 再除 √k_dim →
/// 存取走 u64 整载/整存（4×bf16，ld.global.b64——同 paged_attn 习语），
/// 抽取/打包全在 u32 双车道域——deltanet 状态池 bf16 的存侧；
/// kd%4==0 时行内偏移按 8B 对齐（cuMemAlloc ≥256B 底座）；kd=128 时
/// 整行 S0 驻 32×u64 寄存器（见 token 循环内注释）
///
/// f32→bf16 舍入但**值留在 u32 高半**（位同 f32_to_bf16——低 16 位恒零）：
/// 展宽回 f32 是 from_bits 零成本，打包省一次移位。deltanet bf16 池的
/// 抽取/打包全在 u32 域（u64 域逐车道 shift/and 是 ALU 大头）
#[inline(always)]
fn f32_to_bf16_hi(x: f32) -> u32 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    (bits + 0x7fff + lsb) & 0xffff_0000
}

/// q/k 的逐 (token, k 头) l2norm 倒数——deltanet 的前置小 kernel（每层一次）。
///
/// 去重的对象：同一 (状态行, v 头) 有 `vsplit` 个块、每块 `nwarps` 个 warp，
/// 早先**每个 warp 各把 q/k 的 k_dim 维扫一遍**求同一份 invq/invk（生产几何
/// vsplit=8 × 2 warp ⇒ 同一份数被算 16 遍）；而 k[j]/q[j] 对整条 v 维共享。
/// 这一刀把它变成每 (token, k 头) 算一次，主 kernel 只读 2 个 f32。
///
/// **归约树照抄主 kernel 原文**（不是"新的归约"）：平方和不是精确和（加法
/// 非结合），换树必改末位——这里只是把同一条 j 升序单链从"每 warp 算一遍"
/// 挪到"每 (token, 头) 算一遍"。kd%4==0 走 32 次 u64 整载 + 4 项展开（与
/// 主 kernel E2 l2norm 循环逐字同式，逐位同）；否则走标量 j++ 链（与通用
/// 路径同式）。ε 在加法后、sqrt 前（同式同位）。
/// inv[(token·k_heads+kh)·2] = invq、+1 = invk（f32 原样落，无舍入）。
#[no_mangle]
pub unsafe extern "ptx-kernel" fn dn_qk_l2norm(
    q: *const u16, // [tokens, k_heads, k_dim]
    k: *const u16, // [tokens, k_heads, k_dim]
    inv: *mut f32, // [tokens*k_heads, 2]
    tokens: u32,
    k_heads: u32,
    k_dim: u32,
    l2_eps: f32,
) {
    let kd = k_dim as usize;
    let i = tid() as usize;
    if i >= tokens as usize * k_heads as usize {
        return;
    }
    let qb = i * kd;
    let mut sq = 0.0f32;
    let mut sk = 0.0f32;
    if kd % 4 == 0 {
        let mut j = 0usize;
        while j < kd {
            let qw4 = *(q.add(qb + j) as *const u64);
            let kw4 = *(k.add(qb + j) as *const u64);
            let (qlo4, qhi4) = (qw4 as u32, (qw4 >> 32) as u32);
            let (klo4, khi4) = (kw4 as u32, (kw4 >> 32) as u32);
            let qa = f32::from_bits(qlo4 << 16);
            let qb2 = f32::from_bits(qlo4 & 0xffff_0000);
            let qc = f32::from_bits(qhi4 << 16);
            let qd = f32::from_bits(qhi4 & 0xffff_0000);
            let ka = f32::from_bits(klo4 << 16);
            let kb2 = f32::from_bits(klo4 & 0xffff_0000);
            let kc = f32::from_bits(khi4 << 16);
            let kd2 = f32::from_bits(khi4 & 0xffff_0000);
            sq = qa * qa + sq;
            sq = qb2 * qb2 + sq;
            sq = qc * qc + sq;
            sq = qd * qd + sq;
            sk = ka * ka + sk;
            sk = kb2 * kb2 + sk;
            sk = kc * kc + sk;
            sk = kd2 * kd2 + sk;
            j += 4;
        }
    } else {
        let mut j = 0usize;
        while j < kd {
            let qv = bf16_to_f32(*q.add(qb + j));
            let kv = bf16_to_f32(*k.add(qb + j));
            sq = qv * qv + sq;
            sk = kv * kv + sk;
            j += 1;
        }
    }
    *inv.add(i * 2) = 1.0f32 / core::intrinsics::sqrt::<f32>(sq + l2_eps);
    *inv.add(i * 2 + 1) = 1.0f32 / core::intrinsics::sqrt::<f32>(sk + l2_eps);
}

/// S·=exp(g)；kv_mem=ΣS·k_n；delta=(v−kv_mem)·β；S+=k_n⊗delta；
/// out=ΣS·q_ns——与 torch_recurrent_gated_delta_rule 逐步同序，
/// 累加 j 升序；core out 收 bf16（norm/gate 在 rms_norm_gated 另做）。
/// 门控标量：β=sigmoid(b)，g=−exp(A_log)·softplus(a+dt_bias)
///
/// 状态池 bf16：寄存器 f32 累加、**落池收 bf16**——o 与下一 token 递推
/// 全吃收过的同一存储值（reference 同式镜像，kernel↔reference 位对齐
/// 保持；重开的是对外合同：vs vLLM 逐字节全同 → 量化容差，o25 钉预算）。
///
/// 访存形态：每 token 对状态行 3 趟（kvm 读、更新读、写；u64=4×bf16）——
/// decay 写/读往返由更新趟同式重算 S1=S0·decay（同表达式位不变），
/// kvm/o 累加与逐元素运算序逐操作数保持
#[no_mangle]
pub unsafe extern "ptx-kernel" fn deltanet(
    q: *const u16,        // [tokens, k_heads, k_dim]
    k: *const u16,        // [tokens, k_heads, k_dim]
    v: *const u16,        // [tokens, v_heads, v_dim]
    a: *const u16,        // [tokens, v_heads]
    b: *const u16,        // [tokens, v_heads]
    a_log: *const f32,    // [v_heads]
    dt_bias: *const u16,  // [v_heads]
    state: *mut u16,      // [rows, v_heads, v_dim, k_dim] bf16（in/out）
    core_out: *mut u16,   // [tokens, v_heads*v_dim]
    row_start: *const u32, // [rows]
    row_count: *const u32, // [rows]
    row_id: *const u32,    // [rows] → state 行
    n_rows: u32,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    inv: *const f32, // [tokens*k_heads, 2]——q/k l2norm 倒数（前置 kernel `dn_qk_l2norm` 写，本 kernel 只读）
    scratch: *mut f32, // [块, 1024]——E2 kvm/o 两级折叠槽（op 预分配，块按 blockIdx 私有）
    specs: *const u32, // [rows] 逐 span：0 = 现有语义（逐字不变）；1 = 逐位置 state 行（MTP 验证链）
    vsplit: u32, // E2 chunked：v 行切分数（1 = 不切；grid = rows×v_heads×vsplit）
) {
    let (kd, vd) = (k_dim as usize, v_dim as usize);
    let vh = v_heads as usize;
    // E2 调度重映射（kd=vd=128）：一块 = (行, v 头) × 128 线程 = 4 warp，
    // 每 warp 顺序吃 32 个连续 v 行（行段 32w..32w+32）、每批 4 行；lane s
    // 持 j 切片 4s..4s+3 的 u64 整载/整存——warp 指令 256B 全合并。散射
    // u64 跨步（一线程一 v 行时 32 sector/指令）才是状态流的 LSU/带宽真
    // 瓶颈（探针：全合并 ~880GB/s vs 散射 280-500）；每线程流长度只是
    // 第二变量（2 行/线程仅 +4%，4 行反劣——行驻寄存器串行化在途载入）。
    // kvm/o 的跨 lane 归约走 global scratch + _syncthreads（nvptx 后端无
    // shfl/无 smem——block_sum_strided 同习语；kvm/o 分区、(block, warp)
    // 私有槽），归约树 = 切片内 j 升序 4 项 + 跨 lane u 升序 32 项两级
    // 折叠——reference.rs kd=vd=128 分支同树镜像。kvm/o 分区使跨批覆写
    // 无竞态（下批 kvm 写与本批 o 折叠读不同区，两道屏障序足够）；批
    // 4 行：u64 在途 ×4 藏 DRAM 时延、屏障 16/token；k/q 切片每 token
    // 载一次复用整段行。行/槽下标全 literal 宏展开（[u64;4] 动态索引
    // 会落 local memory）
    //
    // B 档 chunked（保守档，`vsplit` > 1）：块从 (行, v 头) 扩成
    // (行, v 头, v 行块)，每块只负责 vd/vsplit 个连续 v 行（vbase 起）。
    // 切块**只改"哪个 block 负责哪几行"**：每行的 kvm/o 折叠树、元素级
    // 表达式、批内槽位算术序全部逐字不变 ⇒ 位级输出与不切块同（v 行彼此
    // 独立、归约在 warp 内的 32 lane 上，块边界不切任何一条依赖链）。
    // 状态读/写按行私有（state[v][kd]），块间不共享 ⇒ 无 scratch 广播、
    // 无重复状态读。q/k 切片仍是每块每 token 各载一次（每块 2×kd bf16，
    // vsplit 倍于不切块时的量，全在 L2），但**l2norm 已不再按块重复**：
    // 它在前置 kernel `dn_qk_l2norm` 里逐 (token, k 头) 算一次（同一条
    // j 升序平方和链），本 kernel 只读 invq/invk 两个 f32。
    if kd == 128 && vd == 128 {
        let bid = _block_idx_x() as usize;
        let nsp = if vsplit == 0 { 1 } else { vsplit as usize };
        if bid >= n_rows as usize * vh * nsp {
            return;
        }
        // (r, h, chunk) ← bid：chunk 是最内维（同 (行, v 头) 的 v 行块相邻）
        let chunk = urem(bid, nsp);
        let rest = udiv(bid, nsp);
        let (r, h) = (udiv(rest, vh), urem(rest, vh));
        let rows_pb = udiv(vd, nsp); // 本块负责的 v 行数
        let vbase = chunk * rows_pb; // 本块首 v 行
        // ── specs[r]==1：逐位置 state 行（MTP 验证链的零拷贝回滚）──
        // 逐 span 取标志（不是整批标量）：同一步里链行与其他请求的普通行
        // 同批——只有链段走逐位置落盘，其余段仍走下面的 spec==0 路径。
        // 链占连续 state 行 row_id[r] ..= row_id[r]+cnt：token i 读行
        // row_id[r]+i、写行 row_id[r]+i+1（行 stride = vh*vd*kd）。cnt 个 token
        // 跑完后行 +j 即"处理了前 j 个 token 之后"的递推状态——接受 j 个
        // token 时下一步直接从该行起算（零拷贝回滚：不撤销、不重算、不拍快照）。
        // 元素级表达式/归约树与下方 spec==0 路径逐字同式（这里是拷贝而非改
        // 公共代码）——spec==0 的文本与算术序原样不动，位对齐合同与寄存器
        // 占用都不受影响。读基址 sbase 与写基址 swrite 分离、逐 token 推进。
        if *specs.add(r) != 0 {
            let kh = h * k_heads as usize / vh;
            let t = _thread_idx_x() as usize;
            let (w, s) = (t >> 5, t & 31);
            let nwarps = _block_dim_x() as usize >> 5;
            let rows_pw = udiv(rows_pb, nwarps); // 本 warp 行数（切块后 = vd/(vsplit·nwarps)）
            let big_a = expf(*a_log.add(h));
            let dtb = bf16_to_f32(*dt_bias.add(h));
            let st = *row_start.add(r) as usize;
            let cnt = *row_count.add(r) as usize;
            let kd_sqrt = core::intrinsics::sqrt::<f32>(kd as f32);
            let stride = vh * vd * kd; // 一行 state 的 bf16 单元数
            let mut sbase = state.add((*row_id.add(r) as usize * vh + h) * vd * kd);
            // 块私有槽基址与 spec==0 路径**同一块步长**（`bid·nwarps·1024`）：
            // 两条路径的块在同一 grid 里按 bid 交错（普通行在前、链段在后），
            // 块步长不一致时后者的槽区会落进前者的槽区——跨块偶发覆写
            // （qs11 混合批 sp=2/4 的位分叉即此）。本路径只用 4 行 × 32 lane
            // = 128 槽/warp，块内相对槽址（w·128、o 区 +nwarps·128）不变
            // ⇒ 归约树与落点逐字不变。
            let wslot = scratch.add(bid * nwarps * 1024 + w * 128);
            let oslot = wslot.add(nwarps * 128);
            let mut i = 0usize;
            while i < cnt {
                let g = st + i;
                let decay = expf(0.0 - big_a * softplus(bf16_to_f32(*a.add(g * vh + h)) + dtb));
                let beta = bf16_to_f32(f32_to_bf16(1.0f32 / (1.0f32 + expf(0.0 - bf16_to_f32(*b.add(g * vh + h))))));
                let qb = (g * k_heads as usize + kh) * kd;
                // invq/invk 由前置 kernel 逐 (token, k 头) 算一次（本块只读
                // 2 个 f32）——同一条 j 升序平方和链，位同
                let invq = *inv.add((g * k_heads as usize + kh) * 2);
                let invk = *inv.add((g * k_heads as usize + kh) * 2 + 1);
                let kw = *(k.add(qb + s * 4) as *const u64);
                let qw = *(q.add(qb + s * 4) as *const u64);
                let (klo, khi) = (kw as u32, (kw >> 32) as u32);
                let (qlo, qhi) = (qw as u32, (qw >> 32) as u32);
                let (k0, k1) = (f32::from_bits(klo << 16), f32::from_bits(klo & 0xffff_0000));
                let (k2, k3) = (f32::from_bits(khi << 16), f32::from_bits(khi & 0xffff_0000));
                let (q0, q1) = (f32::from_bits(qlo << 16), f32::from_bits(qlo & 0xffff_0000));
                let (q2, q3) = (f32::from_bits(qhi << 16), f32::from_bits(qhi & 0xffff_0000));
                let obase = (g * vh + h) * vd;
                let swrite = sbase.add(stride); // 本 token 的落盘行 = 读行 + 1
                let mut vrow = vbase + w * rows_pw; // warp 行段游标（批 4 行步进）
                // 同 dn_e2!，写指针显式传入（swrite = 读基址 + 1 行）
                macro_rules! dn_e2w {
                    ($sw:expr, $ww:expr, $b:literal) => {{
                        let mut kvm = 0.0f32;
                        let mut u = 0usize;
                        while u < 32 {
                            kvm = *wslot.add($b * 32 + u) + kvm;
                            u += 1;
                        }
                        let vin = bf16_to_f32(*v.add(obase + vrow + $b));
                        let delta = (vin - kvm) * beta;
                        let (lo, hi) = ($ww as u32, ($ww >> 32) as u32);
                        let r0 = f32_to_bf16_hi(fma(k0 * invk, delta, f32::from_bits(lo << 16) * decay));
                        let r1 = f32_to_bf16_hi(fma(k1 * invk, delta, f32::from_bits(lo & 0xffff_0000) * decay));
                        let r2 = f32_to_bf16_hi(fma(k2 * invk, delta, f32::from_bits(hi << 16) * decay));
                        let r3 = f32_to_bf16_hi(fma(k3 * invk, delta, f32::from_bits(hi & 0xffff_0000) * decay));
                        *($sw.add((vrow + $b) * kd + s * 4) as *mut u64) =
                            (r0 >> 16) as u64 | r1 as u64 | (r2 as u64) << 16 | (r3 as u64) << 32;
                        let mut o = 0.0f32;
                        o = fma(f32::from_bits(r0), (q0 * invq) / kd_sqrt, o);
                        o = fma(f32::from_bits(r1), (q1 * invq) / kd_sqrt, o);
                        o = fma(f32::from_bits(r2), (q2 * invq) / kd_sqrt, o);
                        o = fma(f32::from_bits(r3), (q3 * invq) / kd_sqrt, o);
                        *oslot.add($b * 32 + s) = o;
                    }};
                }
                macro_rules! kvm4w {
                    ($ww:expr, $b:literal) => {{
                        let (lo, hi) = ($ww as u32, ($ww >> 32) as u32);
                        let mut kvm = 0.0f32;
                        kvm = fma(f32::from_bits(lo << 16) * decay, k0 * invk, kvm);
                        kvm = fma(f32::from_bits(lo & 0xffff_0000) * decay, k1 * invk, kvm);
                        kvm = fma(f32::from_bits(hi << 16) * decay, k2 * invk, kvm);
                        kvm = fma(f32::from_bits(hi & 0xffff_0000) * decay, k3 * invk, kvm);
                        *wslot.add($b * 32 + s) = kvm;
                    }};
                }
                while vrow < vbase + (w + 1) * rows_pw {
                    let w0 = *(sbase.add((vrow + 0) * kd + s * 4) as *const u64);
                    let w1 = *(sbase.add((vrow + 1) * kd + s * 4) as *const u64);
                    let w2 = *(sbase.add((vrow + 2) * kd + s * 4) as *const u64);
                    let w3 = *(sbase.add((vrow + 3) * kd + s * 4) as *const u64);
                    kvm4w!(w0, 0); kvm4w!(w1, 1); kvm4w!(w2, 2); kvm4w!(w3, 3);
                    _syncthreads();
                    dn_e2w!(swrite, w0, 0); dn_e2w!(swrite, w1, 1);
                    dn_e2w!(swrite, w2, 2); dn_e2w!(swrite, w3, 3);
                    _syncthreads();
                    let mut b = 0usize;
                    while b < 4 {
                        let mut o = 0.0f32;
                        let mut u = 0usize;
                        while u < 32 {
                            o = *oslot.add(b * 32 + u) + o;
                            u += 1;
                        }
                        if s == b {
                            *core_out.add(obase + vrow + b) = f32_to_bf16(o);
                        }
                        b += 1;
                    }
                    vrow += 4;
                }
                sbase = sbase.add(stride);
                i += 1;
            }
            return;
        }
        let kh = h * k_heads as usize / vh;
        let t = _thread_idx_x() as usize;
        let (w, s) = (t >> 5, t & 31); // warp = 行段号；lane = j 切片号
        // 并行度旋钮：block 宽 ÷ 32 = warp 数，本块的 rows_pb 行按 warp 等分
        // （128 线程 = 4 warp、256 线程 = 8 warp；vsplit=1 时 vd 行按 warp 等分
        // —— 128 线程 = 旧 4 warp × 32 行、256 线程 = 8 × 16；vsplit>1 时每块
        // 只吃 vd/vsplit 行）。v 行彼此独立 ⇒ 行内 32 lane 折叠树、批内算术序
        // 完全不变
        let nwarps = _block_dim_x() as usize >> 5;
        let rows_pw = udiv(rows_pb, nwarps);
        let big_a = expf(*a_log.add(h));
        let dtb = bf16_to_f32(*dt_bias.add(h));
        let st = *row_start.add(r) as usize;
        let cnt = *row_count.add(r) as usize;
        let kd_sqrt = core::intrinsics::sqrt::<f32>(kd as f32);
        let sbase = state.add((*row_id.add(r) as usize * vh + h) * vd * kd);
        // 批 16 行：kvm 槽 16×32/warp，o 槽同量，块内 kvm/o 区各 nwarps×512。
        // 屏障每批 2 道（kvm→更新、更新→o 折叠）⇒ 每 warp 32 行时 4 道/token
        // （旧 16 道），每 warp 16 行时 2 道/token
        let wslot = scratch.add(bid * nwarps * 1024 + w * 512); // kvm 槽（+b·32+lane）
        let oslot = wslot.add(nwarps * 512); // o 槽区（与 kvm 分区——跨批覆写无竞态）
        let mut i = 0usize;
        while i < cnt {
            let g = st + i;
            let decay = expf(0.0 - big_a * softplus(bf16_to_f32(*a.add(g * vh + h)) + dtb));
            let beta = bf16_to_f32(f32_to_bf16(1.0f32 / (1.0f32 + expf(0.0 - bf16_to_f32(*b.add(g * vh + h))))));
            let qb = (g * k_heads as usize + kh) * kd;
            // invq/invk 由前置 kernel `dn_qk_l2norm` 逐 (token, k 头) 算一次。
            // **和树没动**：那套 j 升序 128 项平方和单链（32 次 u64 整载 + 4
            // 项展开、acc 在前）逐字搬进了前置 kernel，本 kernel 只读 2 个
            // f32——早先本块的每个 warp 都各扫一遍同一份 q/k（vsplit×nwarps
            // 倍重复），那是切块后剩下的重复账。
            let invq = *inv.add((g * k_heads as usize + kh) * 2);
            let invk = *inv.add((g * k_heads as usize + kh) * 2 + 1);
            // k/q 的 j 切片每 token 载一次（u64 全合并），整段 32 行复用
            let kw = *(k.add(qb + s * 4) as *const u64);
            let qw = *(q.add(qb + s * 4) as *const u64);
            let (klo, khi) = (kw as u32, (kw >> 32) as u32);
            let (qlo, qhi) = (qw as u32, (qw >> 32) as u32);
            let (k0, k1) = (f32::from_bits(klo << 16), f32::from_bits(klo & 0xffff_0000));
            let (k2, k3) = (f32::from_bits(khi << 16), f32::from_bits(khi & 0xffff_0000));
            let (q0, q1) = (f32::from_bits(qlo << 16), f32::from_bits(qlo & 0xffff_0000));
            let (q2, q3) = (f32::from_bits(qhi << 16), f32::from_bits(qhi & 0xffff_0000));
            let obase = (g * vh + h) * vd;
            // 32 项 u 升序串行折叠，下标全 literal：32 次 LDG 一起发（同一切片
            // 区内 128B 连续——一次 sector 组），只剩 32 深入 f32 加法链。
            // 逐项序与滚动 `while u < 32` 完全一致（u 升序、acc = v + acc），
            // **位同**；拿掉的只是循环回边（滚动版里每次 LDG 都串在回边上、
            // 吃满一次 L1 时延——本刀 prefill delta 92.36→64.39ms 的主项）
            //
            // **FMA 在这里吃不到**：槽里的值是各 lane 已经乘好的部分和
            // （`kvm4e!`/`dn_e2!` 里 fma 的第三操作数），本折叠是纯加法第二
            // 级归约——没有乘子可融。产出前的那一次乘法（decay/`q·invq`）
            // 已在第一级 fma 里融掉了。
            macro_rules! fold32 {
                ($base:expr) => {{
                    let p = $base;
                    let mut acc = 0.0f32;
                    acc = *p.add(0) + acc;
                    acc = *p.add(1) + acc;
                    acc = *p.add(2) + acc;
                    acc = *p.add(3) + acc;
                    acc = *p.add(4) + acc;
                    acc = *p.add(5) + acc;
                    acc = *p.add(6) + acc;
                    acc = *p.add(7) + acc;
                    acc = *p.add(8) + acc;
                    acc = *p.add(9) + acc;
                    acc = *p.add(10) + acc;
                    acc = *p.add(11) + acc;
                    acc = *p.add(12) + acc;
                    acc = *p.add(13) + acc;
                    acc = *p.add(14) + acc;
                    acc = *p.add(15) + acc;
                    acc = *p.add(16) + acc;
                    acc = *p.add(17) + acc;
                    acc = *p.add(18) + acc;
                    acc = *p.add(19) + acc;
                    acc = *p.add(20) + acc;
                    acc = *p.add(21) + acc;
                    acc = *p.add(22) + acc;
                    acc = *p.add(23) + acc;
                    acc = *p.add(24) + acc;
                    acc = *p.add(25) + acc;
                    acc = *p.add(26) + acc;
                    acc = *p.add(27) + acc;
                    acc = *p.add(28) + acc;
                    acc = *p.add(29) + acc;
                    acc = *p.add(30) + acc;
                    acc = *p.add(31) + acc;
                    acc
                }};
            }
            // 批内单行：kvm 折叠→delta→更新趟（旧行值 S0 → 新值，原位回写
            // 该 lane 的 j 切片 u64）→o 部分和。元素级表达式/算术序与
            // dn_row128 逐字同（位对齐合同不变），变的只是归约树（两级折叠）
            // 与块↔工作的调度映射；返回值 = 新 S0 的 u64 字（供调用处回写）
            macro_rules! dn_e2 {
                ($ww:expr, $vr:expr, $b:literal) => {{
                    let kvm = fold32!(wslot.add($b * 32));
                    let vin = bf16_to_f32(*v.add($vr));
                    let delta = (vin - kvm) * beta;
                    let (lo, hi) = ($ww as u32, ($ww >> 32) as u32);
                    let r0 = f32_to_bf16_hi(fma(k0 * invk, delta, f32::from_bits(lo << 16) * decay));
                    let r1 = f32_to_bf16_hi(fma(k1 * invk, delta, f32::from_bits(lo & 0xffff_0000) * decay));
                    let r2 = f32_to_bf16_hi(fma(k2 * invk, delta, f32::from_bits(hi << 16) * decay));
                    let r3 = f32_to_bf16_hi(fma(k3 * invk, delta, f32::from_bits(hi & 0xffff_0000) * decay));
                    let mut o = 0.0f32;
                    o = fma(f32::from_bits(r0), (q0 * invq) / kd_sqrt, o);
                    o = fma(f32::from_bits(r1), (q1 * invq) / kd_sqrt, o);
                    o = fma(f32::from_bits(r2), (q2 * invq) / kd_sqrt, o);
                    o = fma(f32::from_bits(r3), (q3 * invq) / kd_sqrt, o);
                    *oslot.add($b * 32 + s) = o;
                    // 新状态回寄存器：lane0 ← r0>>16；lane1 已在位；
                    // lane2 ← r2<<16；lane3 ← r3<<32（与原存储字逐位同）
                    (r0 >> 16) as u64 | r1 as u64 | (r2 as u64) << 16 | (r3 as u64) << 32
                }};
            }
            // 批内单行的 kvm 部分和（切片内 j 升序）→ 槽
            macro_rules! kvm4e {
                ($ww:expr, $b:literal) => {{
                    let (lo, hi) = ($ww as u32, ($ww >> 32) as u32);
                    let mut kvm = 0.0f32;
                    kvm = fma(f32::from_bits(lo << 16) * decay, k0 * invk, kvm);
                    kvm = fma(f32::from_bits(lo & 0xffff_0000) * decay, k1 * invk, kvm);
                    kvm = fma(f32::from_bits(hi << 16) * decay, k2 * invk, kvm);
                    kvm = fma(f32::from_bits(hi & 0xffff_0000) * decay, k3 * invk, kvm);
                    *wslot.add($b * 32 + s) = kvm;
                }};
            }
            // 批 n 行（rows_pw / n 批 ⇒ 屏障 2 道/批）：kvm 趟 →（屏障）→
            // 更新趟 + o 部分和（屏障）→ o 折叠 + core_out。scratch 槽位是批内
            // 行号 0..n-1（warp 私有区，槽址 = b·32 + lane）。**每行自己的
            // kvm/o 折叠算术序一个字不动**（切片内 j 升序 4 项 + 跨 lane u 升序
            // 32 项两级折叠，16/8/4 行批逐字同式）——变的只是分组：同批共一道
            // 屏障、S0 一次取齐 n 行（u64 在途 ×n vs 旧 4）。
            // o 折叠仍是 4 行一组、lane 0..3 分领（32 lanes 分领时 o 槽 128B
            // 跨步打点组过大，实测更慢且无必要——行内 u 升序折叠序不变）。
            //
            // 批内行号 / o 组首行号由宏参数列表给**字面量**：`[u64;16]` 的常量
            // 下标 ⇒ SROA 提到寄存器（动态下标会落 local memory，见上）。
            macro_rules! dn_batch {
                ($vr:expr, [$($r:literal)*], [$($g:literal)*]) => {{
                    let vrow = $vr;
                    let mut sw = [0u64; 16];
                    // kvm 趟：n 行的 S0 一次取齐（u64 在途 ×n 藏 DRAM 时延；行
                    // stride 256B），每行的 kvm 部分和按切片内 j 升序 4 项折叠
                    // 落槽（逐字同 4 行批）
                    $( sw[$r] = *(sbase.add((vrow + $r) * kd + s * 4) as *const u64); )*
                    $( kvm4e!(sw[$r], $r); )*
                    _syncthreads();
                    // 更新趟：直接用持有的 sw[..]（S0 一行只读一次——状态流 2 趟）
                    $( { let nw = dn_e2!(sw[$r], obase + vrow + $r, $r);
                         sw[$r] = nw;
                         *(sbase.add((vrow + $r) * kd + s * 4) as *mut u64) = nw; } )*
                    _syncthreads();
                    // o 两级折叠第二级 + core_out 落点：4 行一组由 lane 0..3
                    // 分领（访问形态与旧 4 行批逐字同）
                    if s < 4 {
                        $( { let o = fold32!(oslot.add(($g + s) * 32));
                             *core_out.add(obase + vrow + $g + s) = f32_to_bf16(o); } )*
                    }
                }};
            }
            // 批 = min(16, rows_pw)（rows_pw 32/16 → 批 16；8 → 批 8；4 → 批 4）：
            // 批大小只改屏障密度与在途 u64 数，不改任何一行的算术
            let mut vrow = vbase + w * rows_pw;
            if rows_pw >= 16 {
                while vrow < vbase + (w + 1) * rows_pw {
                    dn_batch!(vrow, [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15], [0 4 8 12]);
                    vrow += 16;
                }
            } else if rows_pw >= 8 {
                while vrow < vbase + (w + 1) * rows_pw {
                    dn_batch!(vrow, [0 1 2 3 4 5 6 7], [0 4]);
                    vrow += 8;
                }
            } else {
                while vrow < vbase + (w + 1) * rows_pw {
                    dn_batch!(vrow, [0 1 2 3], [0]);
                    vrow += 4;
                }
            }
            i += 1;
        }
        return;
    }
    // specs[r]==1（逐位置 state 行）只在 kd=vd=128 的 E2 fast path 上实现——
    // exec 侧保证投机验证链只在该形状下走 spec 分支（链形状 = 真模型快路径）；
    // 通用路径不镜像：误用即 trap（warp 死给驱动报，不展开 panic 宿主调用——
    // k2 的"PTX 无 call"结构门禁也保持）。specs[r]==0 的通用路径文本不动。
    if *specs.add(udiv(_block_idx_x() as usize, vh)) != 0 {
        unsafe { core::arch::nvptx::trap() };
    }
    let bid = _block_idx_x() as usize;
    if bid >= n_rows as usize * v_heads as usize {
        return;
    }
    let (r, h) = (udiv(bid, v_heads as usize), urem(bid, v_heads as usize));
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
        // 每 token 标量门（各线程冗余算——同序同式位同）。
        // β 先收 bf16（HF b.sigmoid() 在 bf16 张量上、recurrent 内再展 f32）
        let decay = expf(0.0 - big_a * softplus(bf16_to_f32(*a.add(g * vh + h)) + dtb));
        let beta = bf16_to_f32(f32_to_bf16(1.0f32 / (1.0f32 + expf(0.0 - bf16_to_f32(*b.add(g * vh + h))))));
        // q/k l2norm 的倒数：前置 kernel 逐 (token, k 头) 算一次（j 升序
        // 单链，与这里原来那圈逐字同式），本线程只读 2 个 f32
        let qb = (g * k_heads as usize + kh) * kd;
        let invq = *inv.add((g * k_heads as usize + kh) * 2);
        let invk = *inv.add((g * k_heads as usize + kh) * 2 + 1);
        // kvm 趟 + 更新趟（读 S0 写 S2）。kd=128 走 dn_row128（宏展开·行驻
        // 寄存器：32×u64 组元组，无循环、恒定索引，SROA 必拆 SSA 驻寄存器；
        // 更新趟零重读——状态访存 3 趟 → 2 趟）。通用路径：装载即展宽（存储值
        // bf16），u64 一次载 4×bf16、u32 双车道展宽；S1=S0·decay 两趟同式各算
        // （位不变）；S2 落池收 bf16（值留 u32 高半）、o 就地 j 升序累加——
        // 吃收过的存储值，与 reference 同式同位
        if kd == 128 {
            dn_row128(srow, k, q, v, core_out, (g * vh + h) * vd + vv, decay, beta, invq, invk, kd_sqrt, qb);
        } else {
            let mut kvm = 0.0f32;
            if kd % 4 == 0 {
                let mut j = 0usize;
                while j < kd {
                    let w = *(srow.add(j) as *const u64);
                    let (lo, hi) = (w as u32, (w >> 32) as u32);
                    kvm = fma(f32::from_bits(lo << 16) * decay, bf16_to_f32(*k.add(qb + j)) * invk, kvm);
                    kvm = fma(f32::from_bits(lo & 0xffff_0000) * decay, bf16_to_f32(*k.add(qb + j + 1)) * invk, kvm);
                    kvm = fma(f32::from_bits(hi << 16) * decay, bf16_to_f32(*k.add(qb + j + 2)) * invk, kvm);
                    kvm = fma(f32::from_bits(hi & 0xffff_0000) * decay, bf16_to_f32(*k.add(qb + j + 3)) * invk, kvm);
                    j += 4;
                }
            } else {
                let mut j = 0usize;
                while j < kd {
                    kvm = fma(bf16_to_f32(*srow.add(j)) * decay, bf16_to_f32(*k.add(qb + j)) * invk, kvm);
                    j += 1;
                }
            }
            let vin = bf16_to_f32(*v.add((g * vh + h) * vd + vv));
            let delta = (vin - kvm) * beta;
            let mut o = 0.0f32;
            if kd % 4 == 0 {
                let mut j = 0usize;
                while j < kd {
                    let w = *(srow.add(j) as *const u64);
                    let (lo, hi) = (w as u32, (w >> 32) as u32);
                    let r0 = f32_to_bf16_hi(fma(bf16_to_f32(*k.add(qb + j)) * invk, delta, f32::from_bits(lo << 16) * decay));
                    let r1 = f32_to_bf16_hi(fma(bf16_to_f32(*k.add(qb + j + 1)) * invk, delta, f32::from_bits(lo & 0xffff_0000) * decay));
                    let r2 = f32_to_bf16_hi(fma(bf16_to_f32(*k.add(qb + j + 2)) * invk, delta, f32::from_bits(hi << 16) * decay));
                    let r3 = f32_to_bf16_hi(fma(bf16_to_f32(*k.add(qb + j + 3)) * invk, delta, f32::from_bits(hi & 0xffff_0000) * decay));
                    // lane0 ← r0>>16；lane1 已在位；lane2 ← r2<<16；lane3 ← r3<<32
                    *(srow.add(j) as *mut u64) =
                        (r0 >> 16) as u64 | r1 as u64 | (r2 as u64) << 16 | (r3 as u64) << 32;
                    o = fma(f32::from_bits(r0), (bf16_to_f32(*q.add(qb + j)) * invq) / kd_sqrt, o);
                    o = fma(f32::from_bits(r1), (bf16_to_f32(*q.add(qb + j + 1)) * invq) / kd_sqrt, o);
                    o = fma(f32::from_bits(r2), (bf16_to_f32(*q.add(qb + j + 2)) * invq) / kd_sqrt, o);
                    o = fma(f32::from_bits(r3), (bf16_to_f32(*q.add(qb + j + 3)) * invq) / kd_sqrt, o);
                    j += 4;
                }
            } else {
                let mut j = 0usize;
                while j < kd {
                    let s2 = f32_to_bf16(fma(bf16_to_f32(*k.add(qb + j)) * invk, delta, bf16_to_f32(*srow.add(j)) * decay));
                    *srow.add(j) = s2;
                    o = fma(bf16_to_f32(s2), (bf16_to_f32(*q.add(qb + j)) * invq) / kd_sqrt, o);
                    j += 1;
                }
            }
            *core_out.add((g * vh + h) * vd + vv) = f32_to_bf16(o);
        }
        i += 1;
    }
}

/// 带门 RMSNorm（deltanet 输出，Qwen3_5RMSNormGated 语义）：
/// h=f32(x)；xn=h·rsqrt(mean(h²)+eps)；xb=bf16(xn)（唯一中间 round）；
/// xw=w_f32·f32(xb)（f32 不收）；out=bf16(xw·silu(f32(z)))——末级单 round。
/// block-per-row（同 rms_norm_zc：跨步合并 + scratch 块归约）
#[no_mangle]
pub unsafe extern "ptx-kernel" fn rms_norm_gated(
    x: *const u16,     // [rows, hidden]（core out bf16）
    z: *const u16,     // [rows, hidden]（z 门）
    w: *const f32,     // [hidden]（F32 权重）
    out: *mut u16,
    hidden: u32,
    eps: f32,
    rows: u32,
    scratch: *mut f32,
) {
    let row = _block_idx_x() as usize;
    if row >= rows as usize {
        return;
    }
    let t = _thread_idx_x() as usize;
    let bdim = _block_dim_x() as usize;
    let h = hidden as usize;
    let base = row * h;
    let total = unsafe { block_sum_strided(x, scratch, row, base, h, t, bdim) };
    let inv = 1.0f32 / core::intrinsics::sqrt::<f32>(total / hidden as f32 + eps);
    let mut i = t;
    while i < h {
        let xb = f32_to_bf16(bf16_to_f32(*x.add(base + i)) * inv);
        let xw = *w.add(i) * bf16_to_f32(xb);
        let zv = bf16_to_f32(*z.add(base + i));
        *out.add(base + i) = f32_to_bf16(xw * (zv * (1.0f32 / (1.0f32 + expf(-zv)))));
        i += bdim;
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

// ───────────────────── I2 设备端采样器（top-k / top-p） ─────────────────────
// argmax_rows 只给 greedy；采样（temperature/top_p）此前在 serve 侧校验后
// 丢弃。这里把它放回设备：一步一次 launch 覆盖整批采样行，参数与 RNG 状态
// 全在设备缓冲里（图内只读写缓冲，捕获期零 host 分配/同步）。
//
// 语义顺序（与 reference::sample_rows 逐字镜像）：
//   ① temp ≤ 0 → 等价 argmax（argmax_rows 同算法同归约序，位同合同）；
//   ② v = logits · (1/temp)（一次倒数 + 乘；参考镜像同式同序）；
//   ③ 线程内 top-8 梯子（无分支 compare-exchange，u64 全序键）+ block 内
//      **排名选择**（每线程给自己 8 个候选算 rank = #{池中比它大的}）→ 全局
//      top-k 降序表（k ≤ 64）；
//   ④ 在 top-k 集合上 softmax（f32、键降序累加）→ 累加概率 ≥ top_p 即止
//      （至少保留 1 个）；
//   ⑤ 按归一概率采样（PCG-XSH-RR 64/32，状态存设备缓冲、图内前进）。
//
// 复杂度账（n=248320、s 行、block=SMP_T=256 lane）：
//   - ③ 一级：每线程 n/256 ≈ 970 个元素 = 242 次 u64 整载（一次 4 个 bf16，
//     2 路展开 ⇒ 121 次迭代）+ 每元素 ~58 条 ALU（键构造 10 + 8 级
//     compare-exchange 48，其中 u64 比较/选择各 2 条 32 位指令）；
//     ⇒ 单块 ≈ 0.13~0.14ms（**单块 = 单 SM，扫描是延迟/发射受限的**，与
//     greedy 的同规模扫描同量级——k9b 实测 greedy 0.13ms）；
//   - ③ 二级：池 = 256 lane × 8 键 = 16KB，每线程单遍广播读 2048 个 u64 +
//     命中过阈值的候选再比 8 次 ⇒ 实测 ≤ 10µs/块（k9d 阶段探针：一级
//     0.248ms、+二级 0.248ms、+softmax 采样 0.249ms，后两段在噪声内）；
//   - ④⑤ 顺序段：≤ 64 项串行，忽略不计；
//   - 实测（k9，n=248320，含 H2D 小表上传）：t=1/16/100 → 0.260/0.275/0.262
//     ms/发，同 kernel 的 greedy 基线 0.13~0.14ms ⇒ **采样增量 0.117~0.131ms
//     < 0.2ms 预算**；块数与 t 无关（每行一块并行，128 SM 下 t≤100 全并行）。
//   - 线程数扫描（说明瓶颈归属）：T=256 → 0.24ms，T=512 → 0.45ms，
//     T=1024 → 1.33ms。扫描步数减半但总 ALU 不变，反而因 u64 键的 2 寄存器
//     压力 + 单块 warp 数上升拖慢 ⇒ **保持 256 lane**（也保住聚合访存的
//     256B/warp 合并）。想再快只能把一行拆给多块（两级 kernel），
//     当前步时 (<0.3ms) 不值得。
//
// 为什么不支持 top_k ≤ 0（全词表）：top-p 需要对全词表降序累加，等价于
// 248K 项全排序（≥4 倍词表流量的排序网络 + 无 .shared 只能走 global
// scratch），单行就要 MB 级 scratch、单步毫秒级——不值当，且会静默改变
// 采样分布。主机侧直接拒（见 exec 的 set_sampling/step），设备侧这里落
// 0xFFFFFFFF 错误标记做兜底，绝不静默退化成"取前 64"。

/// 采样器 scratch 每行槽位（u32 字数；与 wein-kernel/src/ops 的
/// `SAMPLE_SLOT` 常量一一对应——改动需两侧同改）
/// 布局：[0, SMP_POOL_W) 池 = lane 的 8 个 u64 键（lane*16 字）；
/// SMP_SEL 选中表（按排名落位）64 个 u64 = 128 字；SMP_PROB 概率 64 字
pub const SMP_T: usize = 256;
const SMP_LADDER: usize = 8;
const SMP_POOL_W: usize = SMP_T * SMP_LADDER * 2; // 池（u64 键）字数
const SMP_SEL: usize = SMP_POOL_W + 64; // 选中表 64 个 u64 = 128 字
const SMP_PROB: usize = SMP_SEL + 64 * 2; // 概率 64
pub const SMP_SLOT: usize = SMP_PROB + 64;

/// 采样器 RNG 派生：splitmix64 终混 | 1（状态 0 保留为"未初始化"哨兵，
/// 与 reference::rng_seed 逐字同式）
#[inline(always)]
fn rng_seed(seed: u64) -> u64 {
    let mut x = seed.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    (x ^ (x >> 31)) | 1
}

/// PCG-XSH-RR 64/32 一步：推进 u64 LCG 状态、回 32 位输出。
/// 纯整数运算 ⇒ CPU 镜像逐位同（reference::rng_next）
#[inline(always)]
fn rng_next(st: &mut u64) -> u32 {
    let old = *st;
    *st = old.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let x = (((old >> 18) ^ old) >> 27) as u32;
    let rot = (old >> 59) as u32;
    (x >> rot) | (x << ((32 - rot) & 31))
}

/// [0,1) 均匀 f32：取输出低 24 位（f32 尾数域精确表示，无舍入）
#[inline(always)]
fn rng_f32(st: &mut u64) -> f32 {
    (rng_next(st) >> 8) as f32 * (1.0 / 16777216.0)
}

/// 采样键：`(单调化 f32 位 << 32) | (0xFFFFFFFF - 下标)`。
/// 单调化对有限浮点是保序双射（正值翻最高位、负值按位取反），±0 由
/// `+ 0.0` 规范化，NaN 归到 -inf（与 argmax_rows 的浮点比较同语义：永不
/// 入选）。于是"值降序、同值下标升序"的全序塌成一次 u64 比较。
#[inline(always)]
fn smp_key(h: u16, idx: u32, inv_t: f32) -> u64 {
    let mut v = bf16_to_f32(h) * inv_t + 0.0;
    if v != v {
        v = f32::NEG_INFINITY;
    }
    let b = v.to_bits();
    let flip = (((b as i32) >> 31) as u32) | 0x8000_0000;
    (((b ^ flip) as u64) << 32) | (0xffff_ffff - idx) as u64
}

/// 键 → f32 值（smp_key 的逆映射）
#[inline(always)]
fn smp_val(k: u64) -> f32 {
    let m = (k >> 32) as u32;
    // 逆映射看**映射后**的最高位：1 ⇒ 原值非负（曾翻 0x8000_0000）；
    // 0 ⇒ 原值为负（曾按位取反）。两者不可混用（写反会把负 logits 的概率
    // 曲线整体颠倒——k8 的卡方立刻炸到 1e6）。
    let b = if (m >> 31) != 0 { m ^ 0x8000_0000 } else { !m };
    f32::from_bits(b)
}

/// 键 → 原 logits 下标
#[inline(always)]
fn smp_idx(k: u64) -> u32 {
    0xffff_ffff - (k as u32)
}

/// 设备端采样（top-k / top-p）。block=256、grid=s（一行一块）。
/// - x: [row_total, n] bf16 logits；rows: [s] 采样行号
/// - slots: [s] RNG 槽号（= 请求行号，状态跨步常驻）；rng: [槽数] u64
/// - temp/top_k/top_p/seed: [s] 逐行参数（主机每步覆写设备缓冲）
/// - scratch: [s, SMP_SLOT] u32；out: [s]（0xFFFFFFFF = 参数非法兜底）
#[no_mangle]
pub unsafe extern "ptx-kernel" fn sample_rows(
    x: *const u16,
    rows: *const u32,
    slots: *const u32,
    temp: *const f32,
    top_k: *const u32,
    top_p: *const f32,
    seed: *const u64,
    rng: *mut u64,
    scratch: *mut u32,
    out: *mut u32,
    n: u32,
    s: u32,
) {
    let blk = _block_idx_x() as usize;
    if blk >= s as usize {
        return;
    }
    let lane = _thread_idx_x() as usize; // < SMP_T
    let nn = n as usize;
    let base = *rows.add(blk) as usize * nn;
    let t = *temp.add(blk);
    let sb = scratch.add(blk * SMP_SLOT);

    // ── ① greedy（temp ≤ 0，含 NaN）：与 argmax_rows 同算法同归约序 ──
    if !(t > 0.0) {
        let mut bv = f32::NEG_INFINITY;
        let mut bi = 0xffff_ffffu32;
        let mut i = lane;
        while i < nn {
            let v = bf16_to_f32(*x.add(base + i));
            if v > bv || (v == bv && (i as u32) < bi) {
                bv = v;
                bi = i as u32;
            }
            i += SMP_T;
        }
        *sb.add(lane * 2) = bv.to_bits();
        *sb.add(lane * 2 + 1) = bi;
        _syncthreads();
        if lane == 0 {
            let mut bv = f32::NEG_INFINITY;
            let mut bi = 0xffff_ffffu32;
            let mut l = 0usize;
            while l < SMP_T {
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
        return;
    }

    // ── 参数解码（主机已校验；这里只做设备侧兜底）──
    let kk = {
        let k = *top_k.add(blk) as usize;
        if k == 0 {
            // 全词表 top-p/k 不支持（248K 全排序不可接受，见文件头注释）
            if lane == 0 {
                *out.add(blk) = 0xffff_ffff;
            }
            return;
        }
        if k > 64 {
            64
        } else {
            k
        }
    };
    let pp = {
        let p = *top_p.add(blk);
        if p > 0.0 && p < 1.0 {
            p
        } else {
            1.0
        }
    };

    // ── ③ 一级归约：线程内 top-8 梯子（**无分支 + u64 全序键**）──
    // 键 = (单调化 f32 位 << 32) | (0xFFFFFFFF - 下标)：与参考镜像的
    // "值降序、同值下标升序"是同一个全序（单调化对有限值是保序双射，±0 由
    // `+ 0.0` 规范化），但比较从"值比较 + 相等 + 下标比较"塌成一条 ISETP。
    // 插入用 8 级 compare-exchange（每元素固定 24 条 ALU，**零分支/零 BSSY**）：
    // 分支版（`if 命中 { 7 级冒泡 }`）在 warp 级几乎每个元素都会"至少一个
    // lane 命中"（p ≈ 1-(1-8/k)^32，k 是已处理量），发散开销远高于固定的
    // 24 条——实测 0.30ms → 见 k9d/k9。
    let mut k0 = 0u64;
    let mut k1 = 0u64;
    let mut k2 = 0u64;
    let mut k3 = 0u64;
    let mut k4 = 0u64;
    let mut k5 = 0u64;
    let mut k6 = 0u64;
    let mut k7 = 0u64;
    macro_rules! cx {
        ($v:ident, $x:ident) => {{
            let g = $x > $v;
            let t = $v;
            $v = if g { $x } else { t };
            $x = if g { t } else { $x };
        }};
    }
    macro_rules! ins_key {
        ($key:expr) => {{
            let mut x = $key;
            cx!(k0, x);
            cx!(k1, x);
            cx!(k2, x);
            cx!(k3, x);
            cx!(k4, x);
            cx!(k5, x);
            cx!(k6, x);
            cx!(k7, x);
        }};
    }
    // ② 温度缩放用**一次倒数 + 乘**（不是逐元素除）：div.rn.f32 是一条
    // ~15 指令的 Newton 序列；1/t 只算一次（temp=1 时倒数精确 = 1.0 ⇒ 免费）。
    let inv_t = 1.0f32 / t;
    macro_rules! ins4 {
        ($w:expr, $i0:expr) => {{
            let w = $w;
            let b = $i0 as u32;
            ins_key!(smp_key((w & 0xffff) as u16, b, inv_t));
            ins_key!(smp_key(((w >> 16) & 0xffff) as u16, b + 1, inv_t));
            ins_key!(smp_key(((w >> 32) & 0xffff) as u16, b + 2, inv_t));
            ins_key!(smp_key((w >> 48) as u16, b + 3, inv_t));
        }};
    }
    // 装载：一个 u64 装 4 个连续 bf16（lane 拥有 [4·lane, 4·lane+4) 起、步长
    // 1024 的组），2 路展开 ⇒ 每线程 n/(256·8) ≈ 121 次装载。所有权划分与
    // 参考镜像的"单遍全局 top-k"不同，但仍是全词表的一个划分、每 lane 仍是
    // 约 n/256 个元素 ⇒ 一级归约的正确性论证（每 lane 至多贡献 8 个全局
    // top-k）不变；全序比较与处理次序无关。未对齐（base % 4 ≠ 0）走同一
    // 划分的标量尾循环，结果逐位一致。
    let step = SMP_T * 4;
    let mut i = lane * 4;
    if base % 4 == 0 {
        while i + step + 3 < nn {
            let w0 = *(x.add(base + i) as *const u64);
            let w1 = *(x.add(base + i + step) as *const u64);
            ins4!(w0, i);
            ins4!(w1, i + step);
            i += 2 * step;
        }
        while i + 3 < nn {
            let w = *(x.add(base + i) as *const u64);
            ins4!(w, i);
            i += step;
        }
    }
    // 尾部（含未对齐兜底）：与上面同一个组划分，逐元素补
    while i < nn {
        ins_key!(smp_key(*x.add(base + i), i as u32, inv_t));
        if i + 1 < nn {
            ins_key!(smp_key(*x.add(base + i + 1), (i + 1) as u32, inv_t));
        }
        if i + 2 < nn {
            ins_key!(smp_key(*x.add(base + i + 2), (i + 2) as u32, inv_t));
        }
        if i + 3 < nn {
            ins_key!(smp_key(*x.add(base + i + 3), (i + 3) as u32, inv_t));
        }
        i += step;
    }
    // 梯子写回池（每 lane 8 个 u64 键，一次性写）：二级归约要跨 lane 读
    let pool = sb.add(lane * SMP_LADDER * 2) as *mut u64;
    *pool.add(0) = k0;
    *pool.add(1) = k1;
    *pool.add(2) = k2;
    *pool.add(3) = k3;
    *pool.add(4) = k4;
    *pool.add(5) = k5;
    *pool.add(6) = k6;
    *pool.add(7) = k7;
    _syncthreads();

    // ── ③ 二级归约：排名选择（池 = 256 lane × 8 个 u64 键 = 16KB）。每线程把
    //    自己 8 个候选放寄存器，**单遍扫全池**给它们攒排名：rank(c) = #{y : y > c}
    //    （全序严格 ⇒ 排名两两不同）。rank < 64 的候选直接落到 SEL[rank]——
    //    天然有序，正是 top-p 要的降序。池长 2048 次广播读（所有线程同时读
    //    同一槽，L1 广播）+ 8×2048 次比较 ≈ 10~20µs/块。对照：lane0 串行
    //    锦标赛 64 轮 × 256 槽 = 0.87ms（k9b 实测），差 ~50×——并发归约不能
    //    靠"逐轮抽取"。
    let cv = [k0, k1, k2, k3, k4, k5, k6, k7];
    let mut rk = [0u32; SMP_LADDER];
    let selp = sb.add(SMP_SEL) as *mut u64;
    macro_rules! rk_one {
        ($y:expr) => {{
            let y = $y;
            if y > k7 {
                let mut k = 0usize;
                while k < SMP_LADDER {
                    if y > cv[k] {
                        rk[k] += 1;
                    }
                    k += 1;
                }
            }
        }};
    }
    let pooln = SMP_T * SMP_LADDER; // 池内 u64 字数
    let poolq = sb as *const u64; // 行槽基址（SMP_SLOT·4B 恒 8 对齐）
    let mut g = 0usize;
    while g + 8 <= pooln {
        let q0 = *poolq.add(g);
        let q1 = *poolq.add(g + 1);
        let q2 = *poolq.add(g + 2);
        let q3 = *poolq.add(g + 3);
        let q4 = *poolq.add(g + 4);
        let q5 = *poolq.add(g + 5);
        let q6 = *poolq.add(g + 6);
        let q7 = *poolq.add(g + 7);
        rk_one!(q0);
        rk_one!(q1);
        rk_one!(q2);
        rk_one!(q3);
        rk_one!(q4);
        rk_one!(q5);
        rk_one!(q6);
        rk_one!(q7);
        g += 8;
    }
    while g < pooln {
        rk_one!(*poolq.add(g));
        g += 1;
    }
    let mut k = 0usize;
    while k < SMP_LADDER {
        if rk[k] < 64 {
            *selp.add(rk[k] as usize) = cv[k];
        }
        k += 1;
    }
    _syncthreads();
    _syncthreads();
    if lane != 0 {
        return;
    }
    // 有效项 = SEL 前 kk 槽里的真实候选（0 = 未填充哨兵，只会在候选不足时出现）
    let mut got = 0usize;
    while got < kk {
        if *selp.add(got) == 0 {
            break;
        }
        got += 1;
    }
    if got == 0 {
        *out.add(blk) = 0xffff_ffff; // 理论不可达（n ≥ 1）
        return;
    }
    // ── ④ softmax + top-p 截断（f32、键降序；至少保留 1 个）──
    let m = smp_val(*selp);
    let mut denom = 0.0f32;
    let mut pj = 0usize;
    while pj < got {
        let e = expf(smp_val(*selp.add(pj)) - m);
        *sb.add(SMP_PROB + pj) = e.to_bits();
        denom = e + denom;
        pj += 1;
    }
    let mut c = 0.0f32;
    let mut keep = 1usize;
    let mut pj = 0usize;
    while pj < got {
        c = f32::from_bits(*sb.add(SMP_PROB + pj)) / denom + c;
        keep = pj + 1;
        if c >= pp {
            break;
        }
        pj += 1;
    }
    // ── ⑤ 采样：PCG 状态常驻设备缓冲，图内前进（0 = 未初始化 → seed 派生）──
    let slot = *slots.add(blk) as usize;
    let mut st = *rng.add(slot);
    if st == 0 {
        st = rng_seed(*seed.add(blk));
    }
    let u = rng_f32(&mut st);
    *rng.add(slot) = st;
    let mut acc = 0.0f32;
    let mut pick = keep - 1;
    let mut pj = 0usize;
    while pj < keep {
        acc = f32::from_bits(*sb.add(SMP_PROB + pj)) / denom + acc;
        if u < acc {
            pick = pj;
            break;
        }
        pj += 1;
    }
    *out.add(blk) = smp_idx(*selp.add(pick));
}

