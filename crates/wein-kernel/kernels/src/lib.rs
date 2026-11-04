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
// llvm.nvvm intrinsic 旁路（shfl 等，stdarch nvptx 未覆盖）——实验性链接
#![feature(link_llvm_intrinsics)]
// 状态池 bf16 后 4×bf16 走 u64 整载/整存（ld/st.global.b64——paged_attn
// 同习语；portable_simd 的 f32x4 容器已随 f32 池退役）

use core::arch::nvptx::{_block_dim_x, _block_idx_x, _syncthreads, _thread_idx_x};

/// GPU 上没有展开——panic 即 trap（warp 死给驱动报，不挂整机）
#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    unsafe { core::arch::nvptx::trap() }
}

// —— 能力探针：warp shuffle（stdarch nvptx 无 shfl，走 llvm.nvvm intrinsic
//    link_name 旁路）。若可用，跨线程归约可用 5 步 warp 链替代 scratch 两级
//    折叠（attention 点积 / lm_head splitK / deltanet 折叠的共同解锁）。
unsafe extern "C" {
    #[link_name = "llvm.nvvm.shfl.sync.down.i32"]
    fn shfl_down(mask: u32, a: i32, offset: i32, clamp: i32) -> i32;
    #[link_name = "llvm.nvvm.shfl.sync.idx.f32"]
    fn shfl_idx_f32(mask: u32, a: f32, src: i32, clamp: i32) -> f32;
    #[link_name = "llvm.nvvm.shfl.sync.bfly.i32"]
    fn shfl_bfly(mask: u32, a: i32, offset: i32, clamp: i32) -> i32;
    // warp 内屏障（smem p 缓冲 pass1 写 → pass2 读的可见性；shfl.sync 只有
    // 收敛语义，内存序要 bar.warp.sync）
    #[link_name = "llvm.nvvm.bar.warp.sync"]
    fn bar_warp_sync(mask: u32);
}

/// f32 的 bfly shfl（位往返；v3 点积的段内归约用）
#[inline(always)]
fn shfl_bfly_f32(x: f32, off: i32) -> f32 {
    f32::from_bits(unsafe { shfl_bfly(0xffff_ffff, x.to_bits() as i32, off, 31) } as u32)
}

/// 快速 exp（v3 容差档专用）：2^(x·log2e) 直落 `ex2.approx.f32`
/// （llvm.exp2f32 在 nvptx 的唯一 HW 落点；相对误差 ~2^-22 ≪ rel2 门 3e-3）。
/// 位级档（online/tile/decode）仍走精确 `expf`，一字不动。
/// x ≤ 0 域：x=−inf → +0（MUFU 定义），上溢不可能。
#[inline(always)]
fn fexp(x: f32) -> f32 {
    unsafe { core::intrinsics::exp2::<f32>(x * 1.4426950408889634) }
}

/// `.shared` 段静态数组：若 PTX 出模块级 `.shared` 符号，则 smem 可达
/// （K/V tile 广播 tiling 的地基）。
#[link_section = ".shared"]
static mut SMEM_PROBE: [u32; 64] = [0; 64];

/// smem 探针：每线程写 tid 到 SMEM[tid]，同步后读 SMEM[tid^1] 回写。
#[no_mangle]
pub unsafe extern "ptx-kernel" fn smem_probe(out: *mut u32) {
    let t = unsafe { _thread_idx_x() } as usize;
    unsafe {
        // usize 往返 → 泛型地址空间指针（ld/st 不带 .global 限定，
        // 配合 PTX 侧 cvta.shared 改写使用）
        let base = core::ptr::addr_of_mut!(SMEM_PROBE) as usize;
        let p = core::ptr::with_exposed_provenance_mut::<u32>(base);
        *p.add(t) = t as u32;
        _syncthreads();
        *out.add(t) = *p.add(t ^ 1);
    }
}

/// 每 lane v=lane+1，5 步 shfl.down 归约：全部 lane 应得 Σ=528。
/// 第二个写口验证 shfl.idx.f32（lane 0 广播 f32 到全 warp）。
#[no_mangle]
pub unsafe extern "ptx-kernel" fn shfl_probe(out_i: *mut i32, out_f: *mut f32) {
    let t = unsafe { _thread_idx_x() };
    let lane = t as i32;
    let mut v = lane + 1;
    let mut off = 16;
    while off > 0 {
        v += unsafe { shfl_down(0xffff_ffff, v, off, 31) };
        off /= 2;
    }
    unsafe { *out_i.add(t as usize) = v };
    // bfly 归约：全 lane 同和（down 只有 lane 0 是全和）
    let mut w = lane + 1;
    let mut xo = 16;
    while xo > 0 {
        w += unsafe { shfl_bfly(0xffff_ffff, w, xo, 31) };
        xo /= 2;
    }
    let x = if lane == 0 { 3.5f32 } else { -1.0f32 };
    let y = unsafe { shfl_idx_f32(0xffff_ffff, x, 0, 31) };
    unsafe { *out_f.add(t as usize) = y + w as f32 };
}

// —— 能力探针：wmma（tensor core HMMA）—— attention 下一量级的钥匙。
// m16n16k16：bf16 输入 f32 累加。LLVM 签名事实（llvm-link 裁决）：
// 全家族标量/struct 形式——load 返回 {i32×4}（Rust 元组恰好映射 literal
// struct）、mma 是 16 标量参数 (i32×4 A, i32×4 B, f32×8 C) -> {f32×8}、
// store 的非 stride 版 ldm 是 ImmArg（Rust 无法传立即数）——**必须用
// .stride 变体**（ldm 走普通参数）。
// 元组进/出 extern 签名触发 improper_ctypes——本就是 llvm intrinsic 旁路，放行
#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.nvvm.wmma.m16n16k16.load.a.row.stride.bf16"]
    fn wmma_load_a(p: *const u32, ldm: i32) -> (i32, i32, i32, i32);
    #[link_name = "llvm.nvvm.wmma.m16n16k16.load.b.col.stride.bf16"]
    fn wmma_load_b(p: *const u32, ldm: i32) -> (i32, i32, i32, i32);
    #[link_name = "llvm.nvvm.wmma.m16n16k16.mma.row.col.bf16"]
    fn wmma_mma(
        a0: i32,
        a1: i32,
        a2: i32,
        a3: i32,
        b0: i32,
        b1: i32,
        b2: i32,
        b3: i32,
        c0: f32,
        c1: f32,
        c2: f32,
        c3: f32,
        c4: f32,
        c5: f32,
        c6: f32,
        c7: f32,
    ) -> (f32, f32, f32, f32, f32, f32, f32, f32);
    #[link_name = "llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32"]
    fn wmma_store_c(
        p: *mut f32,
        c0: f32,
        c1: f32,
        c2: f32,
        c3: f32,
        c4: f32,
        c5: f32,
        c6: f32,
        c7: f32,
        ldm: i32,
    );
}

/// wmma 探针：1 warp 协作一次 m16n16k16。C[16][16]f32(row) =
/// A[16][16]bf16(row-major, ldm=16) @ B（以 col-major 视图读，即 host 侧
/// 传 B_math 的转置摆放：b[n*16+k] = B_math[k][n]）。
#[no_mangle]
pub unsafe extern "ptx-kernel" fn wmma_probe(a: *const u16, b: *const u16, c: *mut f32) {
    unsafe {
        let (a0, a1, a2, a3) = wmma_load_a(a as *const u32, 16);
        let (b0, b1, b2, b3) = wmma_load_b(b as *const u32, 16);
        let d = wmma_mma(
            a0, a1, a2, a3, b0, b1, b2, b3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        );
        wmma_store_c(c, d.0, d.1, d.2, d.3, d.4, d.5, d.6, d.7, 16);
    }
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

/// silu + mul 融合（SwiGLU 激活，silu(g)·u 两发并一发）：act 的 bf16
/// 中间舍入保留（分离路径落 act 缓冲再读回），逐表达式逐序复刻
#[no_mangle]
pub unsafe extern "ptx-kernel" fn silu_mul(g: *const u16, u: *const u16, out: *mut u16, n: u32) {
    let i = tid() as usize;
    if i < n as usize {
        let v = bf16_to_f32(*g.add(i));
        let act = f32_to_bf16(v * (1.0f32 / (1.0f32 + expf(-v))));
        *out.add(i) = f32_to_bf16(bf16_to_f32(act) * bf16_to_f32(*u.add(i)));
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

// ───────── 块档（Br 参数化）：O 累加器与 P 落块私有 global scratch ─────────
// 为什么换存储位置：KV 前缀的**块内重读次数 = rpt**（每线程组的 q 行数）——
// 这是唯一的流量杠杆，而 rpt 被"每线程 rpt×ND 个 f32 累加器"顶死在寄存器
// 预算内（512 线程/SM ⇒ ≤128 regs/线程 ⇒ rpt ≤ 16 @ND=4）。把 O（rows×hd
// 个 f32）搬到块私有 scratch 后，一个线程只需持有**一个 pass 的 G 行**累加器
// （G×ND = 16 f32），rpt 就能到 32/64（Br=64/128）：代价是每 j 片一次 O 往返
// （rows×hd×8B/片）与同一 j 片内各 pass 的 K/V 重读（落在 L1/L2，j 片越小越稳）。
//
// 分工（一 block = (kv 头, Br 个 q token 的片, j 分片)）：
//   · B1 分值：线程 (组 rg, lane) 认领 j = c0+lane+k·nv，对 pass 的 G 行一次
//     算 G 条 dot（K 载入一次供 G 行；q 每行一次）→ 原值落 pb
//   · B1b 块内 max：同步后按**行**扫一遍 pb（线程=r，跨 lane 合并 —— 读侧
//     连续寻址），再算在线合并系数 (scale, echunk) 落 scratch
//   · B1c p = exp(分值 − 本片 max)：j 属主原地改写 pb
//   · B2 加权 V：每 pass 认领 ND 维、j 升序全扫（V 载入一次供 G 行）；
//     pass 内 O 在寄存器，片末 `O = O·scale + Σp·V` 落 scratch
// 数值：与单遍档**同一套代数**（每 j 片一次 rescale + 块内 max 归一），只是
// 把"每行单链"切成"每行每片单链"——容差档，判据同 o33。
#[inline(always)]
unsafe fn blk_dot<const ND: usize, const G: usize, const WIDE: bool>(
    q: *const u16,
    qb: usize,
    kp: *const u8,
    kb: usize,
    hd: usize,
) -> [f32; G] {
    let mut s = [0.0f32; G];
    let mut d = 0usize;
    while d < hd {
        let ks = kv_ld::<ND, WIDE>(kp, kb + d);
        let mut k = 0usize;
        while k < G {
            let qs = kv_ld::<ND, false>(q as *const u8, qb + k * hd + d);
            let mut l = 0usize;
            while l < ND {
                s[k] = fma(qs[l], ks[l], s[k]);
                l += 1;
            }
            k += 1;
        }
        d += ND;
    }
    s
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn blk_body<const ND: usize, const G: usize, const WIDE: bool>(
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
    stride: u32,
    scratch: *mut f32,
) {
    let hd = head_dim as usize;
    let qh_n = q_heads as usize;
    let kv_n = kv_heads as usize;
    let group = udiv(qh_n, kv_n);
    let nt = ntok as usize;
    let js = jsplit as usize;
    let cap = chunk as usize;
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
    let rows_nom = nt * group;
    let rows = tn * group;
    let nv = udiv(hd, ND);
    let ngrp = udiv(bdim, nv);
    let rg = udiv(tau, nv);
    let lane = urem(tau, nv);
    let rpt = udiv(rows_nom, ngrp); // 本线程组的 q 行数 = npass × group
    let npass = udiv(rpt, group); // 每 pass 一个 token（G 个 q 头）
    let r0g = rg * rpt;
    let tt0 = udiv(r0g, group); // 本组首 pass 的片内 token 序号
    let d0 = lane * ND;
    let (bpr, bs) = (bpr as usize, block_size as usize);
    let he = kv_n * hd;

    // j 轴分段（与两遍档/单遍档同式）
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
    if j0 >= j1 {
        // 空分片：落 l=0（combine 跳过），保证 part 表被完整覆盖
        if js > 1 {
            let mut p = 0usize;
            while p < npass {
                let tt = tt0 + p;
                if tt < tn && lane == 0 {
                    let mut k = 0usize;
                    while k < G {
                        let rglob = (t0 + tt) * qh_n + kh * group + k;
                        *part_ml.add((rglob * js + sp) * 2) = f32::NEG_INFINITY;
                        *part_ml.add((rglob * js + sp) * 2 + 1) = 0.0;
                        k += 1;
                    }
                }
                p += 1;
            }
        }
        return;
    }

    // 块私有 scratch 布局（host 侧 `ops::blk_slot` 同式；stride 由 host 传入）：
    //   acc[rows_nom × hd]（O）+ pb[rows_nom × cap]（P/分值）+ mch/mru/lru/scl/ech
    //   [rows_nom] + sb[nt × cap]（槽位表）
    let sc = scratch.add(blk * stride as usize);
    let acc = sc;
    let pb = acc.add(rows_nom * hd);
    let mch = pb.add(rows_nom * cap);
    let mru = mch.add(rows_nom);
    let lru = mru.add(rows_nom);
    let scl = lru.add(rows_nom);
    let ech = scl.add(rows_nom);
    let sb = ech.add(rows_nom);

    // ① 每行 running 状态清底 + O 清零（块内协作；rows_nom 可 > bdim）
    let mut r = tau;
    while r < rows_nom {
        *mru.add(r) = f32::NEG_INFINITY;
        *lru.add(r) = 0.0;
        r += bdim;
    }
    let nacc = rows_nom * hd;
    let mut e = tau;
    while e < nacc {
        *acc.add(e) = 0.0;
        e += bdim;
    }
    _syncthreads();

    // 每线程的常驻累加器（仅 npass == 1 时跨片复用；否则只活一个 pass）
    let mut areg = [[0.0f32; ND]; G];
    let mut c0 = j0;
    while c0 < j1 {
        let c1 = if c0 + cap - 1 < j1 - 1 { c0 + cap - 1 } else { j1 - 1 };
        let cc = c1 - c0 + 1;
        _syncthreads(); // 读侧屏障（槽位表要重填）
        tile_fill_slots(sb, block_table, seq_ids, t0, tn, c0, cc, bpr, bs, tau, bdim);
        _syncthreads();

        // ── B1：分值（j 属主整条点积；K 一次载入供 G 行）→ pb 原值
        let mut p = 0usize;
        while p < npass {
            let tt = tt0 + p;
            if tt < tn {
                let qb = ((t0 + tt) * qh_n + kh * group) * hd;
                let pr = *positions.add(t0 + tt) as usize;
                let jr = if pr < c1 { pr } else { c1 };
                let mut j = c0 + lane;
                while j <= jr {
                    let kbase = *sb.add(tt * cc + (j - c0)) as usize * he + kh * hd;
                    let s = blk_dot::<ND, G, WIDE>(q, qb, k_cache, kbase, hd);
                    let mut k = 0usize;
                    while k < G {
                        *pb.add((j - c0) * rows_nom + r0g + p * group + k) = s[k] * q_scale;
                        k += 1;
                    }
                    j += nv;
                }
            }
            p += 1;
        }
        _syncthreads();

        // ── B1b：本片 max（按行扫 pb —— 跨 lane 合并；跨行连续寻址）+ 在线合并
        // 扫的边界必须是**该行自己的** j 上界（`min(pr,c1)-c0`）：超出部分 pb
        // 可能留着上一片的旧值（本片没写），拿它算 max 会把 running 状态
        // 按错的比例缩放（自测：不做这步 ⇒ 贪心第 3 个 token 起分叉）
        let mut r = tau;
        while r < rows_nom {
            let ttr = udiv(r, group);
            let mut m = f32::NEG_INFINITY;
            if ttr < tn {
                let pr = *positions.add(t0 + ttr) as usize;
                if pr >= c0 {
                    let jrr = (if pr < c1 { pr } else { c1 }) - c0;
                    let mut jj = 0usize;
                    while jj <= jrr {
                        let x = *pb.add(jj * rows_nom + r);
                        m = if x > m { x } else { m };
                        jj += 1;
                    }
                }
            }
            *mch.add(r) = m;
            r += bdim;
        }
        _syncthreads();
        let mut r = tau;
        while r < rows_nom {
            let mr = *mru.add(r);
            let mc = *mch.add(r);
            let mn = if mc > mr { mc } else { mr };
            // 空片（mr/mc 仍 −inf）域钳掉，避免 exp(−inf −(−inf)) = NaN 漏进 l/O
            *scl.add(r) = if mr == f32::NEG_INFINITY { 0.0 } else { expf(mr - mn) };
            *ech.add(r) = if mc == f32::NEG_INFINITY { 0.0 } else { expf(mc - mn) };
            *mru.add(r) = mn;
            r += bdim;
        }
        _syncthreads();

        // ── B1c：p = exp(分值 − 本片 max)（j 属主原地写）
        let mut p = 0usize;
        while p < npass {
            let tt = tt0 + p;
            if tt < tn {
                let pr = *positions.add(t0 + tt) as usize;
                let jr = if pr < c1 { pr } else { c1 };
                let mut j = c0 + lane;
                while j <= jr {
                    let mut k = 0usize;
                    while k < G {
                        let o = (j - c0) * rows_nom + r0g + p * group + k;
                        *pb.add(o) = expf(*pb.add(o) - *mch.add(r0g + p * group + k));
                        k += 1;
                    }
                    j += nv;
                }
            }
            p += 1;
        }
        _syncthreads();

        // ── B2：加权 V（每 pass 认领 ND 维、j 升序全扫；V 载入一次供 G 行）
        // 片内偏量 s 从 0 累起，片末一次在线合并 `a = a·scale + echunk·s`——
        // 与单遍档**逐位同式**（jchunk 相同 ⇒ 与单遍档逐位同值），只改存储位置
        let mut p = 0usize;
        while p < npass {
            let tt = tt0 + p;
            if tt < tn {
                let pr = *positions.add(t0 + tt) as usize;
                let jr = if pr < c1 { pr } else { c1 };
                let r0 = r0g + p * group;
                let mut a = [[0.0f32; ND]; G];
                let mut s = [[0.0f32; ND]; G];
                let mut ef = [0.0f32; G];
                let mut sf = [0.0f32; G];
                let mut dl = [0.0f32; G];
                // running O：npass==1 常驻寄存器，否则读块私有 scratch
                let mut k = 0usize;
                while k < G {
                    ef[k] = *ech.add(r0 + k);
                    sf[k] = *scl.add(r0 + k);
                    let mut l = 0usize;
                    while l < ND {
                        a[k][l] = if npass > 1 {
                            *acc.add((r0 + k) * hd + d0 + l)
                        } else {
                            areg[k][l]
                        };
                        l += 1;
                    }
                    k += 1;
                }
                let mut j = c0;
                while j <= jr {
                    let vbase = *sb.add(tt * cc + (j - c0)) as usize * he + kh * hd + d0;
                    let vs = kv_ld::<ND, WIDE>(v_cache, vbase);
                    let mut k = 0usize;
                    while k < G {
                        let q = *pb.add((j - c0) * rows_nom + r0 + k);
                        let mut l = 0usize;
                        while l < ND {
                            s[k][l] = fma(q, vs[l], s[k][l]);
                            l += 1;
                        }
                        dl[k] = q + dl[k];
                        k += 1;
                    }
                    j += 1;
                }
                let mut k = 0usize;
                while k < G {
                    let mut l = 0usize;
                    while l < ND {
                        a[k][l] = fma(ef[k], s[k][l], a[k][l] * sf[k]);
                        if npass > 1 {
                            *acc.add((r0 + k) * hd + d0 + l) = a[k][l];
                        } else {
                            areg[k][l] = a[k][l];
                        }
                        l += 1;
                    }
                    if lane == 0 {
                        *lru.add(r0 + k) = fma(ef[k], dl[k], *lru.add(r0 + k) * sf[k]);
                    }
                    k += 1;
                }
            }
            p += 1;
        }
        _syncthreads();
        c0 = c1 + 1;
    }

    // ③ 落结果（每 pass 一行组；jsplit=1 直落 out，否则落偏量表由 combine 合并）
    let mut p = 0usize;
    while p < npass {
        let tt = tt0 + p;
        if tt < tn {
            let r0 = r0g + p * group;
            let mut o = [[0.0f32; ND]; G];
            let mut lv = [0.0f32; G];
            let mut mv = [0.0f32; G];
            let mut k = 0usize;
            while k < G {
                lv[k] = *lru.add(r0 + k);
                mv[k] = *mru.add(r0 + k);
                let mut l = 0usize;
                while l < ND {
                    o[k][l] = if npass > 1 {
                        *acc.add((r0 + k) * hd + d0 + l)
                    } else {
                        areg[k][l]
                    };
                    l += 1;
                }
                k += 1;
            }
            let qb = ((t0 + tt) * qh_n + kh * group) * hd;
            let mut k = 0usize;
            while k < G {
                if js == 1 {
                    let mut l = 0usize;
                    while l < ND {
                        *out.add(qb + k * hd + d0 + l) = f32_to_bf16(o[k][l] / lv[k]);
                        l += 1;
                    }
                } else {
                    let rglob = (t0 + tt) * qh_n + kh * group + k;
                    let base = rglob * js + sp;
                    if lane == 0 {
                        *part_ml.add(base * 2) = mv[k];
                        *part_ml.add(base * 2 + 1) = lv[k];
                    }
                    let pb0 = part_acc.add(base * PART_HD_STRIDE + d0);
                    let mut l = 0usize;
                    while l < ND {
                        *pb0.add(l) = o[k][l];
                        l += 1;
                    }
                }
                k += 1;
            }
        }
        p += 1;
    }
    let _ = rows;
}

macro_rules! blk_export {
    ($name:ident, $nd:literal, $g:literal, $wide:literal) => {
        /// 块档（Br 参数化、O/P 落块私有 scratch）：prefill 长序列专用。
        /// `ntok` = Br、`chunk` = j 片宽、`stride` = 每块 scratch 槽数。
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
            stride: u32,
            scratch: *mut f32,
        ) {
            blk_body::<$nd, $g, $wide>(
                q, k_cache, v_cache, block_table, seq_ids, positions, out, part_acc, part_ml,
                q_scale, bpr, block_size, q_heads, kv_heads, head_dim, tokens, ntok, jsplit, chunk,
                stride, scratch,
            )
        }
    };
}

/// 块档 bf16 池（group=4 定态：Qwen3.5-4B 的 16q/4kv）——prefill 长序列
blk_export!(paged_attn_blk, 4, 4, false);
/// 块档 f32 池（与 bf16 池数值逐位同值：只改载入形状）
blk_export!(paged_attn_blk_f32, 4, 4, true);

// ───────────── qblk 档（K/V 行驻寄存器供全 q 行）：prefill 长序列 ─────────────
// 几何病根：现役 prefill（online 档 ntok=4/rpt=4）每块 16 行却流式读整个
// 前缀 ⇒ KV 流量 ∝ (T/ntok)·T（贴几何地板）；blk 档 Br 上去了但每 pass
// 重载+重解 K/V（指令吞吐墙）。本档做**循环互换**：块 = (kv 头, Br token
// 片, j 分片)，j 行外层**载一次 K/V 行片（ND 维）进寄存器**，内层全 RPT
// 行消费——K/V 每字节每块只读一次（流量 ∝ (T/Br)·T），点积按 lane 切 ND
// 维、跨 lane 的 nv 路 partial 落块私有 fs 由读者线程串行折叠（工作集
// rows_nom×JTR×nv f32，L1 驻）。在线 softmax 与 online/blk 同族：每 chunk
// 一次 rescale；p = expf(sc − mn) 直接以新 max mn 为基（echunk 并进 p——
// 数学同值、舍入路径不同 ⇒ 对单遍档是**容差档**，判据同 o33）。
// 混合段块（tile 跨请求边界）自动回退**逐 pass 槽位**的慢路（正确性优先，
// 频率 = 段边界块；单请求 prefill 全走快路）。
//
// 分工（线程 (组 rg, lane) 持 RPT 行 × ND 维）：
//   · D：j-tile（JTR 个 j）一轮——快路 K 行经 sb[u=0] 槽位载入寄存器、
//     每 (行, j) 一条 ND 维 partial 落 fs → 屏障 → 折叠线程串行加 nv 路
//     ×q_scale 落 pb 原始分 → 屏障 → 下一轮
//   · M/P（行属主，无屏障）：扫本 chunk 自己行的有效 j 段取 max → mn/scale
//     → pb 原地 expf(sc − mn)；acc/lrun 预乘 scale
//   · V：j 行外层 V 行片载寄存器一次，内层 RPT 行 `acc += p·v` 单 fma
//   · 片末 mrun = mn（rescale 收在 chunk 边界）
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn qblk_body<const ND: usize, const RPT: usize, const JTR: usize, const WIDE: bool>(
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
    stride: u32,
    scratch: *mut f32,
) {
    let hd = head_dim as usize;
    let qh_n = q_heads as usize;
    let kv_n = kv_heads as usize;
    let group = udiv(qh_n, kv_n);
    let nt = ntok as usize;
    let js = jsplit as usize;
    let cap = chunk as usize;
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
    let rows_nom = nt * group;
    let rows = tn * group;
    let npass = udiv(RPT, group);
    let nv = udiv(hd, ND);
    let ngrp = udiv(bdim, nv);
    let rg = udiv(tau, nv);
    let lane = urem(tau, nv);
    let r0 = rg * RPT;
    let tt0 = rg * npass;
    let vr = r0 < rows;
    let d0 = lane * ND;
    let (bpr, bs) = (bpr as usize, block_size as usize);
    let he = kv_n * hd;

    // j 轴分段（与 online/blk 同式）
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
    if j0 >= j1 {
        if js > 1 {
            let mut p = 0usize;
            while p < npass {
                let tt = tt0 + p;
                if tt < tn && lane == 0 {
                    let mut k = 0usize;
                    while k < group {
                        let rglob = (t0 + tt) * qh_n + kh * group + k;
                        *part_ml.add((rglob * js + sp) * 2) = f32::NEG_INFINITY;
                        *part_ml.add((rglob * js + sp) * 2 + 1) = 0.0;
                        k += 1;
                    }
                }
                p += 1;
            }
        }
        return;
    }
    // uniform-seq 判定（块级一次）：整片同请求 ⇒ K/V 行片可全行共享（快路）
    let s0 = *seq_ids.add(t0) as usize;
    let mut uniform = true;
    let mut u = 1usize;
    while u < tn {
        if *seq_ids.add(t0 + u) as usize != s0 {
            uniform = false;
        }
        u += 1;
    }

    // 块私有 scratch 布局（host 侧 `ops::qblk_stride` 同式；stride 由 host 传入）：
    //   pb[rows_nom × cap]（原始分，行主：r·cap + (j−c0)）+ pe[rows_nom × cap]
    //   （p = expf(sc − mn)，与 pb **分区**——同组 nv 条 lane 冗余跑 M/P，
    //   原地改写 pb 会踩仍在 max 扫的兄弟 lane：pe 分区 + 仅 lane 0 写，
    //   P/V 间一道屏障收口）+ fs[rows_nom × JTR × nv]（(r·JTR+jj)·nv + lane
    //   ——nv 连续供折叠串行读）+ sb[nt × cap]（槽位表）
    let sc = scratch.add(blk * stride as usize);
    let pb = sc;
    let pe = pb.add(rows_nom * cap);
    let fs = pe.add(rows_nom * cap);
    let sb = fs.add(rows_nom * JTR * nv);

    // 在线状态（每行一份，跨 chunk 续；l 由每线程按同一 j 序冗余累加）
    let mut mrun = [f32::NEG_INFINITY; RPT];
    let mut lrun = [0.0f32; RPT];
    let mut acc = [[0.0f32; ND]; RPT];
    // 行组基（组内第 k 行 = token tt0 + k/group、q 头 kh·group + k%group）
    let mut qb0 = 0usize;
    if vr {
        qb0 = ((t0 + tt0) * qh_n + kh * group) * hd;
    }

    let mut c0 = j0;
    while c0 < j1 {
        let c1 = if c0 + cap - 1 < j1 - 1 { c0 + cap - 1 } else { j1 - 1 };
        let cc = c1 - c0 + 1;
        _syncthreads(); // 读侧屏障（槽位表要重填）
        tile_fill_slots(sb, block_table, seq_ids, t0, tn, c0, cc, bpr, bs, tau, bdim);
        _syncthreads();

        // 行常量预计算（每 chunk 一次）：prk=行位置上界、vtk=行有效、
        // qok=行 q 元素基（(p·qh_n+g)·hd）——D/V/M 内层全部零除法零位载
        let mut prk = [0usize; RPT];
        let mut vtk = [false; RPT];
        let mut qok = [0usize; RPT];
        {
            let mut k = 0usize;
            while k < RPT {
                let p = udiv(k, group);
                let tt = tt0 + p;
                if tt < tn {
                    prk[k] = *positions.add(t0 + tt) as usize;
                    vtk[k] = true;
                }
                qok[k] = qb0 + (p * qh_n + urem(k, group)) * hd;
                k += 1;
            }
        }

        // ── D：分值（j-tile 轮转：K 行片载一次供全行 → fs → 折叠 → pb）
        let mut jb = c0;
        while jb <= c1 {
            let je = if jb + JTR - 1 < c1 { jb + JTR - 1 } else { c1 };
            let jtcc = je - jb + 1;
            if vr {
                if uniform {
                    // 快路：槽位查一次（u=0 行），K 行片全行共享；行枚举
                    // const-RPT（寄存器数组可解索）+ prk/vtk/qok 寄存器索引
                    let mut jj = 0usize;
                    while jj < jtcc {
                        let kb = *sb.add(jb - c0 + jj) as usize * he + kh * hd + d0;
                        let kt = kv_ld::<ND, WIDE>(k_cache, kb);
                        let jjabs = jb + jj;
                        let mut k = 0usize;
                        while k < RPT {
                            if vtk[k] && jjabs <= prk[k] {
                                let qs = kv_ld::<ND, false>(q as *const u8, qok[k] + d0);
                                let mut s = 0.0f32;
                                let mut l = 0usize;
                                while l < ND {
                                    s = fma(qs[l], kt[l], s);
                                    l += 1;
                                }
                                *fs.add(((r0 + k) * JTR + jj) * nv + lane) = s;
                            }
                            k += 1;
                        }
                        jj += 1;
                    }
                } else {
                    // 慢路（混合段边界块）：槽位逐 token——正确性路径
                    let mut p = 0usize;
                    while p < npass {
                        let tt = tt0 + p;
                        if tt < tn {
                            let pr = *positions.add(t0 + tt) as usize;
                            let mut jj = 0usize;
                            while jj < jtcc {
                                if jb + jj <= pr {
                                    let kb = *sb.add(tt * cc + (jb - c0 + jj)) as usize * he
                                        + kh * hd
                                        + d0;
                                    let kt = kv_ld::<ND, WIDE>(k_cache, kb);
                                    let mut g = 0usize;
                                    while g < group {
                                        let k = p * group + g;
                                        let qs = kv_ld::<ND, false>(
                                            q as *const u8,
                                            qb0 + (p * qh_n + g) * hd + d0,
                                        );
                                        let mut s = 0.0f32;
                                        let mut l = 0usize;
                                        while l < ND {
                                            s = fma(qs[l], kt[l], s);
                                            l += 1;
                                        }
                                        *fs.add(((r0 + k) * JTR + jj) * nv + lane) = s;
                                        g += 1;
                                    }
                                }
                                jj += 1;
                            }
                        }
                        p += 1;
                    }
                }
            }
            _syncthreads();
            // 折叠：线程摊 (行, jj) 对——nv 路 partial 串行加（lane 升序，
            // 确定性）×q_scale 落 pb 原始分
            // 对枚举递增：jtcc | bdim（bdim%32==0）⇒ e+=bdim 即 r+=bdim/jtcc、
            // jj 不变；group | dr 时 ttr 也递增（非整除走除法回退路）
            let dr = udiv(bdim, jtcc);
            let jb0 = jb - c0;
            let r0p = udiv(tau, jtcc);
            let jj = urem(tau, jtcc);
            let jjabs = jb + jj;
            if r0p < rows && urem(dr, group) == 0 {
                let mut r = r0p;
                while r < rows {
                    let ttr = t0 + udiv(r, group);
                    let fsb = (r * JTR + jj) * nv;
                    let pbb = r * cap + jb0 + jj;
                    if jjabs <= *positions.add(ttr) as usize {
                        let fb = fs.add(fsb);
                        let mut s = 0.0f32;
                        let mut l = 0usize;
                        while l < nv {
                            s = *fb.add(l) + s;
                            l += 1;
                        }
                        *pb.add(pbb) = s * q_scale;
                    }
                    r += dr;
                }
            } else {
                let mut e = tau;
                while e < rows * jtcc {
                    let r = udiv(e, jtcc);
                    let j = urem(e, jtcc);
                    if jb + j <= *positions.add(t0 + udiv(r, group)) as usize {
                        let fb = fs.add((r * JTR + j) * nv);
                        let mut s = 0.0f32;
                        let mut l = 0usize;
                        while l < nv {
                            s = *fb.add(l) + s;
                            l += 1;
                        }
                        *pb.add(r * cap + (jb - c0 + j)) = s * q_scale;
                    }
                    e += bdim;
                }
            }
            _syncthreads();
            jb = je + 1;
        }

        // ── M/P（行属主，无屏障；pb 本行段只被本线程读写）：本 chunk 行 max →
        // 在线系数 → pb 原地 p = expf(sc − mn)（echunk 并入 p）→ acc/l 预缩
        let mut mn = [0.0f32; RPT];
        let mut scl = [0.0f32; RPT];
        if vr {
            let mut k = 0usize;
            while k < RPT {
                if vtk[k] {
                    let pr = prk[k];
                    if pr >= c0 {
                        let jrr = (if pr < c1 { pr } else { c1 }) - c0;
                        let rb = pb.add((r0 + k) * cap);
                        let mut m = f32::NEG_INFINITY;
                        let mut jj = 0usize;
                        while jj <= jrr {
                            let x = *rb.add(jj);
                            m = if x > m { x } else { m };
                            jj += 1;
                        }
                        let x = if m > mrun[k] { m } else { mrun[k] };
                        mn[k] = x;
                        scl[k] = if mrun[k] == f32::NEG_INFINITY { 0.0 } else { expf(mrun[k] - x) };
                        // P：p = expf(分值 − mn) 落 **pe 分区**（仅 lane 0 写——
                        // 同组 lane 冗余原地改写 pb 会互踩 max 扫读）
                        if lane == 0 {
                            let reb = pe.add((r0 + k) * cap);
                            let mut jj = 0usize;
                            while jj <= jrr {
                                *reb.add(jj) = expf(*rb.add(jj) - x);
                                jj += 1;
                            }
                        }
                        lrun[k] = lrun[k] * scl[k];
                        let mut l = 0usize;
                        while l < ND {
                            acc[k][l] = acc[k][l] * scl[k];
                            l += 1;
                        }
                    } else {
                        // 空片（本行本 chunk 无有效 j）：状态原样过（scale=1）
                        mn[k] = mrun[k];
                        scl[k] = 1.0;
                    }
                } else {
                    mn[k] = mrun[k];
                    scl[k] = 1.0;
                }
                k += 1;
            }
        }

        // P（lane 0 写 pe）→ V（全 lane 读 pe）：一道屏障收口
        _syncthreads();
        // ── V：加权 V（V 行片载一次供全行；p 在 pe，单 fma 累加）
        if vr {
            if uniform {
                let mut j = c0;
                while j <= c1 {
                    let vb = *sb.add(j - c0) as usize * he + kh * hd + d0;
                    let vt = kv_ld::<ND, WIDE>(v_cache, vb);
                    let mut k = 0usize;
                    while k < RPT {
                        if vtk[k] && j <= prk[k] {
                            let p = *pe.add((r0 + k) * cap + (j - c0));
                            let mut l = 0usize;
                            while l < ND {
                                acc[k][l] = fma(p, vt[l], acc[k][l]);
                                l += 1;
                            }
                            lrun[k] = p + lrun[k];
                        }
                        k += 1;
                    }
                    j += 1;
                }
            } else {
                // k=p·group+g 的 group 是运行期值，SCEV 证不出 k<RPT ⇒
                // 数组索引留 panic 桩（PTX 成 call，禁项）——外层裸指针免检
                let accp = acc.as_mut_ptr();
                let lrp = lrun.as_mut_ptr();
                let mut p = 0usize;
                while p < npass {
                    let tt = tt0 + p;
                    if tt < tn {
                        let pr = *positions.add(t0 + tt) as usize;
                        let mut j = c0;
                        while j <= pr && j <= c1 {
                            let vb = *sb.add(tt * cc + (j - c0)) as usize * he + kh * hd + d0;
                            let vt = kv_ld::<ND, WIDE>(v_cache, vb);
                            let mut g = 0usize;
                            while g < group {
                                let k = p * group + g;
                                let pv = *pe.add((r0 + k) * cap + (j - c0));
                                let arow = &mut *accp.add(k);
                                let mut l = 0usize;
                                while l < ND {
                                    arow[l] = fma(pv, vt[l], arow[l]);
                                    l += 1;
                                }
                                *lrp.add(k) = pv + *lrp.add(k);
                                g += 1;
                            }
                            j += 1;
                        }
                    }
                    p += 1;
                }
            }
        }
        let mut k = 0usize;
        while k < RPT {
            mrun[k] = mn[k];
            k += 1;
        }
        _syncthreads(); // pb/fs 下一 chunk 重写前的写侧屏障
        c0 = c1 + 1;
    }

    // 落结果（js=1 直落 out；否则偏量表由 attn_combine 合并——语义同 online 档）
    if vr {
        let mut k = 0usize;
        while k < RPT {
            let tt = tt0 + udiv(k, group);
            if tt < tn {
                if js == 1 {
                    let qo = qb0 + (udiv(k, group) * qh_n + urem(k, group)) * hd;
                    let mut l = 0usize;
                    while l < ND {
                        *out.add(qo + d0 + l) = f32_to_bf16(acc[k][l] / lrun[k]);
                        l += 1;
                    }
                } else {
                    let rglob = (t0 + tt) * qh_n + kh * group + urem(k, group);
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
            }
            k += 1;
        }
    }
}

macro_rules! qblk_export {
    ($name:ident, $nd:literal, $rpt:literal, $jtr:literal, $wide:literal) => {
        /// qblk 档（K/V 行片驻寄存器供全 q 行）：prefill 长序列专用。
        /// `ntok` = Br（= bdim/nv × RPT/group，host 侧 `qblk_geom` 裁决）、
        /// `chunk` = j 片宽、`stride` = 每块 scratch 槽数（ABI 与块档同形）。
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
            stride: u32,
            scratch: *mut f32,
        ) {
            qblk_body::<$nd, $rpt, $jtr, $wide>(
                q, k_cache, v_cache, block_table, seq_ids, positions, out, part_acc, part_ml,
                q_scale, bpr, block_size, q_heads, kv_heads, head_dim, tokens, ntok, jsplit,
                chunk, stride, scratch,
            )
        }
    };
}

/// qblk 档变体（bf16 池；ND 维/lane、RPT 行/线程组、JTR = 折叠批宽）。
/// 寄存器主项 = acc[RPT][ND]：r4 系 64 档（bdim 512 可 2 块/SM）、r8 系
/// 128 档（bdim 384 上限 170 内的冲量档——Br=48 的载体）
qblk_export!(paged_attn_qblk_n16r4j2, 16, 4, 2, false);
qblk_export!(paged_attn_qblk_n16r4j4, 16, 4, 4, false);
qblk_export!(paged_attn_qblk_n16r8j2, 16, 8, 2, false);
qblk_export!(paged_attn_qblk_n16r8j4, 16, 8, 4, false);
qblk_export!(paged_attn_qblk_n8r8j4, 8, 8, 4, false);
qblk_export!(paged_attn_qblk_n8r16j2, 8, 16, 2, false);

// ───────── v3：smem K/V tile + shfl warp 归约（prefill 长序列容差档） ─────────
// 解锁能力后的第三次几何迭代。对照：
// · online（单遍）：块=(kv头, token 片, j 分片)，RPT 行/线程组共享 K 载，
//   归约走 global scratch 两级折叠——每块每 KV 字节服务 RPT 行，流量贴 DRAM。
// · qblk：循环互换后流量砍到 6.2GB/层，但 acc[RPT][ND] 超寄存器预算落
//   local + scratch 折叠 ⇒ 反慢 4.2×。
// · v3：**K/V tile 驻 smem**（KT 个 token × hd 维 f32，一道协作装载，
//   全块 q 行消费——L2 重载 ÷passes、DRAM 只过一次）+ **点积归约走 shfl**
//   （段内 log2(NV) 步 bfly，无 scratch 无块屏障）+ **acc/q/m/l 全驻寄存器**
//   （pass=行批外层、tile 内层，跨 tile 续在线状态）。
//
// 线程映射（ND=16 维/lane、RL=2 行/lane 组，常量）：
//   warp = 32 lane = GPW(=32/NV) 个行组 × NV 个维 lane；lane sl 持行内维
//   [16·sl, 16·sl+16)。组每 j 步推 RL=2 行（K 的 4×F4 smem 读一次供 2 行）。
//   pass p：warp w 认领行 [(p·nw+w)·RPW + g·RL, +RL)。块 = (kv头, Br=ntok
//   token, j 分片 sp)；行 r → (token = r/ng, q 头 = r%ng)。
//
// smem 布局（swizzle 免 bank 冲突）：token jj 的 256 维 = 4·NV 个 F4；维 d
// 的 F4 组 g4=d/4 落槽 (g4%4)·NV + g4/4 ⇒ 消费侧 lane sl 第 k 个 F4 读槽
// k·NV+sl（16 lane 连续 256B 无冲突），恰好取回维 16·sl+4k+e。staging 侧
// 按 16B(8 维) chunk 从 global 载 bf16、解包成 2×F4 写两个槽。
// pbuf：每 warp [RPW][KT] f32 分值窗——pass1 写（组内 sl==0）、bar.warp.sync、
// pass2 全 lane 广播读（兼作归约结果的组内广播）。
//
// 循环序：segment（块内多 seq 回退：逐段重跑，段外行 mask 成 −inf）→
// pass（行批，状态驻寄存器跨 tile）→ tile（KT token，staging 后、消费后
// 各一道 __syncthreads）→ pass1（QK 点积→bfly 归约→mask→pbuf）→
// bar.warp.sync → pass2（tile max→mn/scale→p=fexp(s−mn)→acc+=p·V、l+=p）
// → tile 末 rescale 收进 acc/l。
//
// 数值：与 online 同一套在线 softmax 代数，三处舍入差 ⇒ 容差档（o37 钉
// rel2 ≤ 3e-3）：① p 直接对 mn 归一（online 是 p·echunk 两步）② exp 走
// `fexp`（ex2.approx，rel ~2^-22）③ 点积归约从"d 升序单链"变 4 链树+bfly。
// 掩码：无效 (行, j) 的 s=−inf（pass2 select 成 p=0，防 fma(0,NaN)）；
// 全掩码 tile（mn=−inf）整 tile 跳过。jsplit 偏量语义照抄 online
// （m=分片 running max、acc 不归一、空分片写 (−inf,0)），attn_combine 不改。
#[link_section = ".shared"]
static mut V3_K16: [F4; 16 * 64] = [F4([0.0; 4]); 16 * 64];
#[link_section = ".shared"]
static mut V3_V16: [F4; 16 * 64] = [F4([0.0; 4]); 16 * 64];
/// pbuf：warp(≤12) × RPW(≤8) × KT(16)
#[link_section = ".shared"]
static mut V3_P16: [f32; 12 * 8 * 16] = [0.0; 12 * 8 * 16];
#[link_section = ".shared"]
static mut V3_K8: [F4; 8 * 64] = [F4([0.0; 4]); 8 * 64];
#[link_section = ".shared"]
static mut V3_V8: [F4; 8 * 64] = [F4([0.0; 4]); 8 * 64];
#[link_section = ".shared"]
static mut V3_P8: [f32; 12 * 8 * 8] = [0.0; 12 * 8 * 8];

/// KT 档选 smem 组（const 折叠，死档不进 PTX）
#[inline(always)]
fn v3_smem<const KT: usize>() -> (*mut F4, *mut F4, *mut f32) {
    let (k, v, p) = if KT == 16 {
        unsafe {
            (
                core::ptr::addr_of_mut!(V3_K16) as usize,
                core::ptr::addr_of_mut!(V3_V16) as usize,
                core::ptr::addr_of_mut!(V3_P16) as usize,
            )
        }
    } else {
        unsafe {
            (
                core::ptr::addr_of_mut!(V3_K8) as usize,
                core::ptr::addr_of_mut!(V3_V8) as usize,
                core::ptr::addr_of_mut!(V3_P8) as usize,
            )
        }
    };
    (
        core::ptr::with_exposed_provenance_mut::<F4>(k),
        core::ptr::with_exposed_provenance_mut::<F4>(v),
        core::ptr::with_exposed_provenance_mut::<f32>(p),
    )
}

/// staging：global bf16 → smem f32（swizzle 落槽）。`ebase` = 元素偏移
/// （token 槽 × he + kh·hd + 8c），`jj` = tile 内 token 序。
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn v3_stage8<const NV: usize>(
    dst: *mut F4,
    src: *const u8,
    ebase: usize,
    jj: usize,
    c: usize,
) {
    let p = src as *const u16;
    let a = bf16x4(*(p.add(ebase) as *const u64));
    let b = bf16x4(*(p.add(ebase + 4) as *const u64));
    let g0 = 2 * c;
    let g1 = 2 * c + 1;
    let tbase = jj * (NV * 4);
    *dst.add(tbase + (g0 % 4) * NV + g0 / 4) =
        F4([bf16_to_f32(a[0]), bf16_to_f32(a[1]), bf16_to_f32(a[2]), bf16_to_f32(a[3])]);
    *dst.add(tbase + (g1 % 4) * NV + g1 / 4) =
        F4([bf16_to_f32(b[0]), bf16_to_f32(b[1]), bf16_to_f32(b[2]), bf16_to_f32(b[3])]);
}

/// q 行 16 维切片载寄存器（const 索引小循环，照 online 档位驻留模式）
#[inline(always)]
unsafe fn v3_qload(q: *const u16, dst: &mut [f32; 16], qb: usize) {
    let mut k = 0usize;
    while k < 4 {
        let qs = bf16x4(*(q.add(qb + 4 * k) as *const u64));
        dst[4 * k] = bf16_to_f32(qs[0]);
        dst[4 * k + 1] = bf16_to_f32(qs[1]);
        dst[4 * k + 2] = bf16_to_f32(qs[2]);
        dst[4 * k + 3] = bf16_to_f32(qs[3]);
        k += 1;
    }
}

/// 16 维切片点积 + 段内 bfly 归约（NV lane 同和）。
/// 必须在 warp 收敛点调用：shfl.sync 全掩码，调用侧不得有分支遮蔽。
#[inline(always)]
fn v3_dot16<const NV: usize>(qr: &[f32; 16], k0: &F4, k1: &F4, k2: &F4, k3: &F4) -> f32 {
    // 4 链树（链长 4，ILP 友好；容差档不要求 d 升序单链）
    let mut a = fma(qr[0], k0.0[0], 0.0);
    let mut b = fma(qr[1], k0.0[1], 0.0);
    let mut c = fma(qr[2], k0.0[2], 0.0);
    let mut d = fma(qr[3], k0.0[3], 0.0);
    a = fma(qr[4], k1.0[0], a);
    b = fma(qr[5], k1.0[1], b);
    c = fma(qr[6], k1.0[2], c);
    d = fma(qr[7], k1.0[3], d);
    a = fma(qr[8], k2.0[0], a);
    b = fma(qr[9], k2.0[1], b);
    c = fma(qr[10], k2.0[2], c);
    d = fma(qr[11], k2.0[3], d);
    a = fma(qr[12], k3.0[0], a);
    b = fma(qr[13], k3.0[1], b);
    c = fma(qr[14], k3.0[2], c);
    d = fma(qr[15], k3.0[3], d);
    let mut s = (a + b) + (c + d);
    // bfly 归约必须是 +=（远端值累加）；赋值版 4 步后 lane 拿到的是
    // lane^(NV-1) 的原偏和（曾实测 s 塌成单 lane 分量，p̂ 全均匀）
    let mut off = NV / 2;
    while off > 0 {
        s += shfl_bfly_f32(s, off as i32);
        off /= 2;
    }
    s
}

/// acc[16] 标量 rescale（const 索引）
#[inline(always)]
fn v3_scale16(acc: &mut [f32; 16], s: f32) {
    let mut l = 0usize;
    while l < 16 {
        acc[l] *= s;
        l += 1;
    }
}

/// acc += p·v（16 维，const 索引）
#[inline(always)]
fn v3_acc16(acc: &mut [f32; 16], p: f32, v0: &F4, v1: &F4, v2: &F4, v3: &F4) {
    acc[0] = fma(p, v0.0[0], acc[0]);
    acc[1] = fma(p, v0.0[1], acc[1]);
    acc[2] = fma(p, v0.0[2], acc[2]);
    acc[3] = fma(p, v0.0[3], acc[3]);
    acc[4] = fma(p, v1.0[0], acc[4]);
    acc[5] = fma(p, v1.0[1], acc[5]);
    acc[6] = fma(p, v1.0[2], acc[6]);
    acc[7] = fma(p, v1.0[3], acc[7]);
    acc[8] = fma(p, v2.0[0], acc[8]);
    acc[9] = fma(p, v2.0[1], acc[9]);
    acc[10] = fma(p, v2.0[2], acc[10]);
    acc[11] = fma(p, v2.0[3], acc[11]);
    acc[12] = fma(p, v3.0[0], acc[12]);
    acc[13] = fma(p, v3.0[1], acc[13]);
    acc[14] = fma(p, v3.0[2], acc[14]);
    acc[15] = fma(p, v3.0[3], acc[15]);
}

/// ex2 直落（pass2 的 p：q_scale·log2e 已折进 qs2/mnf，免 selp 免分支）
#[inline(always)]
fn fexp2(x: f32) -> f32 {
    unsafe { core::intrinsics::exp2::<f32>(x) }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn v3_body<const NV: usize, const KT: usize>(
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
    _head_dim: u32,
    tokens: u32,
    ntok: u32,
    jsplit: u32,
    _chunk: u32,
    _scratch: *mut f32,
) {
    const RL: usize = 2;
    let hd = NV * 16;
    let qh_n = q_heads as usize;
    let kv_n = kv_heads as usize;
    let ng = udiv(qh_n, kv_n);
    let nt = ntok as usize; // = Br
    let js = jsplit as usize;
    let toks = tokens as usize;
    let bdim = _block_dim_x() as usize;
    let tau = _thread_idx_x() as usize;
    let gpw = 32 / NV; // 行组/warp
    let rpw = gpw * RL; // 行/warp/pass
    let ntile = udiv(toks + nt - 1, nt);
    let blk = _block_idx_x() as usize;
    if blk >= kv_n * ntile * js {
        return;
    }
    let kh = udiv(blk, ntile * js);
    let rest = urem(blk, ntile * js);
    let tile = ntile - 1 - udiv(rest, js); // 反序：长 span 块先发射藏因果尾
    let sp = urem(rest, js);
    let t0 = tile * nt;
    let tn = if t0 + nt < toks { nt } else { toks - t0 };
    let rows = tn * ng;
    let (bpr, bs) = (bpr as usize, block_size as usize);
    let he = kv_n * hd;
    let nw = udiv(bdim, 32);
    let warp = udiv(tau, 32);
    let lane32 = urem(tau, 32);
    let g = udiv(lane32, NV); // 行组号
    let sl = urem(lane32, NV); // 维 lane（持 [16sl, 16sl+16)）
    let npass = udiv(rows + nw * rpw - 1, nw * rpw);
    let (ksm, vsm, psm) = v3_smem::<KT>();
    // pbuf 存 raw 点积（不乘 q_scale）；pass2 折叠：p = 2^(s·qs2 − mn·qs2)，
    // 掩码位 s=−inf → fma(−inf,qs2,mnf)=−inf → ex2=+0，免 selp/免分支。
    // 空 tile（mn=−inf）走 mnf=−inf 同式归零，acc/l 不被污染。
    let qs2 = q_scale * 1.4426950408889634;

    // 多 seq 回退：块内逐段（连续同 seq 的 token 段）重跑同一套 tile 机制，
    // 段外行 mask 成 −inf（p=0 不污染状态）；uniform 快路 = 单段零浪费
    let mut u0 = 0usize;
    while u0 < tn {
        let s_seq = *seq_ids.add(t0 + u0) as usize;
        let mut u1 = u0 + 1;
        while u1 < tn && *seq_ids.add(t0 + u1) as usize == s_seq {
            u1 += 1;
        }
        let mut pmax = 0usize;
        let mut u = u0;
        while u < u1 {
            let p = *positions.add(t0 + u) as usize;
            pmax = if p > pmax { p } else { pmax };
            u += 1;
        }
        let lspan = udiv(pmax + js, js);
        let jmax = pmax + 1;
        let j0 = sp * lspan;
        let j1 = if j0 + lspan < jmax { j0 + lspan } else { jmax };
        let mut pass = 0usize;
        while pass < npass {
            // 本 warp 本 pass 的行批：组 g 持 RL=2 行（rb、rb+1）。
            // 行态全程标量/const 索引数组（运行时索引会落 local，online 档同款教训）。
            let rb = (pass * nw + warp) * rpw + g * RL;
            let tt0 = udiv(rb, ng);
            let live0 = rb < rows && tt0 >= u0 && tt0 < u1;
            let r1 = rb + 1;
            let tt1 = udiv(r1, ng);
            let live1 = r1 < rows && tt1 >= u0 && tt1 < u1;
            let mut pr0 = 0usize;
            let mut pr1 = 0usize;
            let mut rg0 = 0usize;
            let mut rg1 = 0usize;
            let mut qreg0 = [0.0f32; 16];
            let mut qreg1 = [0.0f32; 16];
            if live0 {
                pr0 = *positions.add(t0 + tt0) as usize;
                rg0 = (t0 + tt0) * qh_n + kh * ng + urem(rb, ng);
                v3_qload(q, &mut qreg0, rg0 * hd + 16 * sl);
            }
            if live1 {
                pr1 = *positions.add(t0 + tt1) as usize;
                rg1 = (t0 + tt1) * qh_n + kh * ng + urem(r1, ng);
                v3_qload(q, &mut qreg1, rg1 * hd + 16 * sl);
            }
            let mut acc0 = [0.0f32; 16];
            let mut acc1 = [0.0f32; 16];
            let mut mrun0 = f32::NEG_INFINITY;
            let mut mrun1 = f32::NEG_INFINITY;
            let mut lrun0 = 0.0f32;
            let mut lrun1 = 0.0f32;
            let pb0 = (warp * rpw + g * RL) * KT;
            let pb1 = pb0 + KT;
            let mut c0 = j0;
            while c0 < j1 {
                let cc = if c0 + KT < j1 { KT } else { j1 - c0 };
                // ── staging：K/V tile 协作装载（bf16→f32 解包 + swizzle）
                let nch = NV * 2; // 每 token 的 16B chunk 数 = hd/8
                let nent = cc * nch;
                let mut e = tau;
                while e < nent {
                    let jj = udiv(e, nch);
                    let c = urem(e, nch);
                    let slot = slot_of(block_table, s_seq, c0 + jj, bpr, bs);
                    let ebase = slot * he + kh * hd + c * 8;
                    v3_stage8::<NV>(ksm, k_cache, ebase, jj, c);
                    v3_stage8::<NV>(vsm, v_cache, ebase, jj, c);
                    e += bdim;
                }
                _syncthreads();
                // ── pass1：QK 点积（K 读一次供 RL 行）→ 段内 bfly 归约 → pbuf(raw)
                // （jj 2× 展开实测反伤：regs 顶穿 128 → spill 增大 136→232B，弃）
                let mut jj = 0usize;
                while jj < cc {
                    let kb = jj * (NV * 4) + sl;
                    let kf0 = *ksm.add(kb);
                    let kf1 = *ksm.add(kb + NV);
                    let kf2 = *ksm.add(kb + 2 * NV);
                    let kf3 = *ksm.add(kb + 3 * NV);
                    let s0 = v3_dot16::<NV>(&qreg0, &kf0, &kf1, &kf2, &kf3);
                    let s1 = v3_dot16::<NV>(&qreg1, &kf0, &kf1, &kf2, &kf3);
                    let j = c0 + jj;
                    let sc0 = if live0 && j <= pr0 { s0 } else { f32::NEG_INFINITY };
                    let sc1 = if live1 && j <= pr1 { s1 } else { f32::NEG_INFINITY };
                    if sl == 0 {
                        *psm.add(pb0 + jj) = sc0;
                        *psm.add(pb1 + jj) = sc1;
                    }
                    jj += 1;
                }
                bar_warp_sync(0xffff_ffff); // pbuf 写 → 读的 warp 内可见性
                // ── pass2：tile max(raw 域) → 在线合并系数 → p·V（V 载一次供双行）
                let mut tm0 = f32::NEG_INFINITY;
                let mut tm1 = f32::NEG_INFINITY;
                let mut jj = 0usize;
                while jj < cc {
                    let x0 = *psm.add(pb0 + jj);
                    let x1 = *psm.add(pb1 + jj);
                    tm0 = if x0 > tm0 { x0 } else { tm0 };
                    tm1 = if x1 > tm1 { x1 } else { tm1 };
                    jj += 1;
                }
                let mn0 = if tm0 > mrun0 { tm0 } else { mrun0 };
                let mn1 = if tm1 > mrun1 { tm1 } else { mrun1 };
                // mrun=−inf ⟹ scale=0（acc 原本全 0）；mn≥mrun 恒成立（指数 ≤0）
                let sc0 = if mrun0 == f32::NEG_INFINITY {
                    0.0
                } else {
                    fexp((mrun0 - mn0) * q_scale)
                };
                let sc1 = if mrun1 == f32::NEG_INFINITY {
                    0.0
                } else {
                    fexp((mrun1 - mn1) * q_scale)
                };
                v3_scale16(&mut acc0, sc0);
                v3_scale16(&mut acc1, sc1);
                lrun0 *= sc0;
                lrun1 *= sc1;
                let mnf0 = if mn0 == f32::NEG_INFINITY {
                    f32::NEG_INFINITY
                } else {
                    -mn0 * qs2
                };
                let mnf1 = if mn1 == f32::NEG_INFINITY {
                    f32::NEG_INFINITY
                } else {
                    -mn1 * qs2
                };
                let mut ls0 = 0.0f32;
                let mut ls1 = 0.0f32;
                let mut jj = 0usize;
                while jj < cc {
                    let vb = jj * (NV * 4) + sl;
                    let vf0 = *vsm.add(vb);
                    let vf1 = *vsm.add(vb + NV);
                    let vf2 = *vsm.add(vb + 2 * NV);
                    let vf3 = *vsm.add(vb + 3 * NV);
                    let p0 = fexp2(fma(*psm.add(pb0 + jj), qs2, mnf0));
                    let p1 = fexp2(fma(*psm.add(pb1 + jj), qs2, mnf1));
                    ls0 += p0;
                    ls1 += p1;
                    v3_acc16(&mut acc0, p0, &vf0, &vf1, &vf2, &vf3);
                    v3_acc16(&mut acc1, p1, &vf0, &vf1, &vf2, &vf3);
                    jj += 1;
                }
                lrun0 += ls0;
                lrun1 += ls1;
                mrun0 = mn0;
                mrun1 = mn1;
                _syncthreads();
                c0 += KT;
            }
            // ── 段末落盘（js=1 写 out；js>1 写偏结果供 attn_combine，
            // 空分片/全掩码行天然落 (−inf,0)：mrun=−inf、lrun=0、acc=0）
            if js == 1 {
                if live0 {
                    let ob = rg0 * hd + 16 * sl;
                    let mut l = 0usize;
                    while l < 16 {
                        *out.add(ob + l) = f32_to_bf16(acc0[l] / lrun0);
                        l += 1;
                    }
                }
                if live1 {
                    let ob = rg1 * hd + 16 * sl;
                    let mut l = 0usize;
                    while l < 16 {
                        *out.add(ob + l) = f32_to_bf16(acc1[l] / lrun1);
                        l += 1;
                    }
                }
            } else {
                if live0 {
                    let base = rg0 * js + sp;
                    if sl == 0 {
                        *part_ml.add(base * 2) = mrun0 * q_scale;
                        *part_ml.add(base * 2 + 1) = lrun0;
                    }
                    let pa = part_acc.add(base * PART_HD_STRIDE + 16 * sl);
                    let mut l = 0usize;
                    while l < 16 {
                        *pa.add(l) = acc0[l];
                        l += 1;
                    }
                }
                if live1 {
                    let base = rg1 * js + sp;
                    if sl == 0 {
                        *part_ml.add(base * 2) = mrun1 * q_scale;
                        *part_ml.add(base * 2 + 1) = lrun1;
                    }
                    let pa = part_acc.add(base * PART_HD_STRIDE + 16 * sl);
                    let mut l = 0usize;
                    while l < 16 {
                        *pa.add(l) = acc1[l];
                        l += 1;
                    }
                }
            }
            pass += 1;
        }
        u0 = u1;
    }
}

macro_rules! v3_export {
    ($name:ident, $nv:literal, $kt:literal) => {
        /// v3 档（smem K/V tile + shfl 归约）：prefill 长序列容差档。
        /// `ntok` = Br、`chunk`/`scratch` 忽略（ABI 与单遍档同形）。
        /// hd = NV×16 由变体钉死（host 侧按 head_dim 选入口）。
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
            v3_body::<$nv, $kt>(
                q, k_cache, v_cache, block_table, seq_ids, positions, out, part_acc, part_ml,
                q_scale, bpr, block_size, q_heads, kv_heads, head_dim, tokens, ntok, jsplit,
                chunk, scratch,
            )
        }
    };
}

/// v3 档变体：NV=维 lane 数（hd/16）、KT=K/V tile token 数。
/// KT=32 档未导：K+V 双 tile = 64KB 超静态 .shared 48KB/块硬顶
/// （动态 smem+属性可解，但 sm_89 静态档先吃 16 的收益）
v3_export!(paged_attn_v3_n16k16, 16, 16);
v3_export!(paged_attn_v3_n16k8, 16, 8);
v3_export!(paged_attn_v3_n8k16, 8, 16);
v3_export!(paged_attn_v3_n8k8, 8, 8);

// ═══ v3c：NV=32（全 warp 单一行组，gpw=1）× ND=8 维/lane × RL=4 行/lane ═══
// 相对 n16 的两笔账：① gpw=1 ⟹ K/V 的 smem 读不被行组复制（n16 每 warp
// 每 jj 两组合载 2KB 供 4 行；v3c 单组 1KB 供 4 行——(row,jj) smem 读减半）
// ② 点积 8 fma + 5 级 bfly、PV 8 fma：(row,jj) 指令 ~17 vs n16 ~30。
// q/acc 状态 4 行 × 8 维 = 64 寄存器全程标量化（运行时索引必落 local，
// 纪律同 n16/online）。smem 槽 swizzle slot(g)=(g%2)·32+g/2（g = 4 维组号
// ∈[0,64) 双射；lane sl 读 g=2sl,2sl+1 ⟹ 槽 sl 与 32+sl，LDS.128 相位内
// bank 4sl%32 互异无冲突）。在线 softmax/jsplit 偏量/掩码 −inf/ex2 折叠
// 语义与 v3/online 逐字一致，attn_combine 不改。hd=256 专用（NV·ND=256）。

/// v3c staging：global bf16 → smem f32（槽 (g%2)·32+g/2 落位）
#[inline(always)]
unsafe fn v3c_stage8(dst: *mut F4, src: *const u8, ebase: usize, jj: usize, c: usize) {
    let p = src as *const u16;
    let a = bf16x4(*(p.add(ebase) as *const u64));
    let b = bf16x4(*(p.add(ebase + 4) as *const u64));
    let tbase = jj * 64;
    *dst.add(tbase + c) =
        F4([bf16_to_f32(a[0]), bf16_to_f32(a[1]), bf16_to_f32(a[2]), bf16_to_f32(a[3])]);
    *dst.add(tbase + 32 + c) =
        F4([bf16_to_f32(b[0]), bf16_to_f32(b[1]), bf16_to_f32(b[2]), bf16_to_f32(b[3])]);
}

/// q 行 8 维切片载寄存器（const 索引小循环）
#[inline(always)]
unsafe fn v3c_qload(q: *const u16, dst: &mut [f32; 8], qb: usize) {
    let mut k = 0usize;
    while k < 2 {
        let qs = bf16x4(*(q.add(qb + 4 * k) as *const u64));
        dst[4 * k] = bf16_to_f32(qs[0]);
        dst[4 * k + 1] = bf16_to_f32(qs[1]);
        dst[4 * k + 2] = bf16_to_f32(qs[2]);
        dst[4 * k + 3] = bf16_to_f32(qs[3]);
        k += 1;
    }
}

/// 8 维点积 + 全 warp bfly 归约（5 级；全 lane 同和）。
/// 必须在 warp 收敛点调用：shfl.sync 全掩码，调用侧不得有分支遮蔽。
/// bfly 归约必须是 +=（远端值累加）——赋值版只拿到远端偏和。
#[inline(always)]
fn v3c_dot8(qr: &[f32; 8], k0: &F4, k1: &F4) -> f32 {
    let mut a = fma(qr[0], k0.0[0], 0.0);
    let mut b = fma(qr[1], k0.0[1], 0.0);
    a = fma(qr[2], k0.0[2], a);
    b = fma(qr[3], k0.0[3], b);
    a = fma(qr[4], k1.0[0], a);
    b = fma(qr[5], k1.0[1], b);
    a = fma(qr[6], k1.0[2], a);
    b = fma(qr[7], k1.0[3], b);
    let mut s = a + b;
    s += shfl_bfly_f32(s, 16);
    s += shfl_bfly_f32(s, 8);
    s += shfl_bfly_f32(s, 4);
    s += shfl_bfly_f32(s, 2);
    s += shfl_bfly_f32(s, 1);
    s
}

/// acc[8] 标量 rescale（const 索引）
#[inline(always)]
fn v3c_scale8(acc: &mut [f32; 8], s: f32) {
    let mut l = 0usize;
    while l < 8 {
        acc[l] *= s;
        l += 1;
    }
}

/// acc += p·v（8 维，const 索引）
#[inline(always)]
fn v3c_acc8(acc: &mut [f32; 8], p: f32, v0: &F4, v1: &F4) {
    acc[0] = fma(p, v0.0[0], acc[0]);
    acc[1] = fma(p, v0.0[1], acc[1]);
    acc[2] = fma(p, v0.0[2], acc[2]);
    acc[3] = fma(p, v0.0[3], acc[3]);
    acc[4] = fma(p, v1.0[0], acc[4]);
    acc[5] = fma(p, v1.0[1], acc[5]);
    acc[6] = fma(p, v1.0[2], acc[6]);
    acc[7] = fma(p, v1.0[3], acc[7]);
}

/// v3c 主循环：结构逐行对位 v3_body（见上），差异只在行几何
/// （RL=4、ND=8、gpw=1、staging nch=hd/8=32、槽 swizzle）。
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn v3c_body<const KT: usize>(
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
    _head_dim: u32,
    tokens: u32,
    ntok: u32,
    jsplit: u32,
    _chunk: u32,
    _scratch: *mut f32,
) {
    const RL: usize = 4;
    let hd = 256usize;
    let qh_n = q_heads as usize;
    let kv_n = kv_heads as usize;
    let ng = udiv(qh_n, kv_n);
    let nt = ntok as usize; // = Br
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
    let tile = ntile - 1 - udiv(rest, js); // 反序：长 span 块先发射藏因果尾
    let sp = urem(rest, js);
    let t0 = tile * nt;
    let tn = if t0 + nt < toks { nt } else { toks - t0 };
    let rows = tn * ng;
    let (bpr, bs) = (bpr as usize, block_size as usize);
    let he = kv_n * hd;
    let nw = udiv(bdim, 32);
    let warp = udiv(tau, 32);
    let sl = urem(tau, 32); // 维 lane（持 [8sl, 8sl+8)）
    let npass = udiv(rows + nw * RL - 1, nw * RL);
    let (ksm, vsm, psm) = v3_smem::<KT>();
    // pbuf 存 raw 点积（不乘 q_scale）；pass2 折叠：p = 2^(s·qs2 − mn·qs2)，
    // 掩码位 s=−inf → fma(−inf,qs2,mnf)=−inf → ex2=+0，免 selp/免分支。
    // 空 tile（mn=−inf）走 mnf=−inf 同式归零，acc/l 不被污染。
    let qs2 = q_scale * 1.4426950408889634;

    // 多 seq 回退：块内逐段（连续同 seq 的 token 段）重跑同一套 tile 机制，
    // 段外行 mask 成 −inf（p=0 不污染状态）；uniform 快路 = 单段零浪费
    let mut u0 = 0usize;
    while u0 < tn {
        let s_seq = *seq_ids.add(t0 + u0) as usize;
        let mut u1 = u0 + 1;
        while u1 < tn && *seq_ids.add(t0 + u1) as usize == s_seq {
            u1 += 1;
        }
        let mut pmax = 0usize;
        let mut u = u0;
        while u < u1 {
            let p = *positions.add(t0 + u) as usize;
            pmax = if p > pmax { p } else { pmax };
            u += 1;
        }
        let lspan = udiv(pmax + js, js);
        let jmax = pmax + 1;
        let j0 = sp * lspan;
        let j1 = if j0 + lspan < jmax { j0 + lspan } else { jmax };
        let mut pass = 0usize;
        while pass < npass {
            // 本 warp 本 pass 的行批：RL=4 行全标量化（运行时索引会落 local）。
            let rb = (pass * nw + warp) * RL;
            let tt0 = udiv(rb, ng);
            let live0 = rb < rows && tt0 >= u0 && tt0 < u1;
            let r1 = rb + 1;
            let tt1 = udiv(r1, ng);
            let live1 = r1 < rows && tt1 >= u0 && tt1 < u1;
            let r2 = rb + 2;
            let tt2 = udiv(r2, ng);
            let live2 = r2 < rows && tt2 >= u0 && tt2 < u1;
            let r3 = rb + 3;
            let tt3 = udiv(r3, ng);
            let live3 = r3 < rows && tt3 >= u0 && tt3 < u1;
            let mut pr0 = 0usize;
            let mut pr1 = 0usize;
            let mut pr2 = 0usize;
            let mut pr3 = 0usize;
            let mut rg0 = 0usize;
            let mut rg1 = 0usize;
            let mut rg2 = 0usize;
            let mut rg3 = 0usize;
            let mut qreg0 = [0.0f32; 8];
            let mut qreg1 = [0.0f32; 8];
            let mut qreg2 = [0.0f32; 8];
            let mut qreg3 = [0.0f32; 8];
            if live0 {
                pr0 = *positions.add(t0 + tt0) as usize;
                rg0 = (t0 + tt0) * qh_n + kh * ng + urem(rb, ng);
                v3c_qload(q, &mut qreg0, rg0 * hd + 8 * sl);
            }
            if live1 {
                pr1 = *positions.add(t0 + tt1) as usize;
                rg1 = (t0 + tt1) * qh_n + kh * ng + urem(r1, ng);
                v3c_qload(q, &mut qreg1, rg1 * hd + 8 * sl);
            }
            if live2 {
                pr2 = *positions.add(t0 + tt2) as usize;
                rg2 = (t0 + tt2) * qh_n + kh * ng + urem(r2, ng);
                v3c_qload(q, &mut qreg2, rg2 * hd + 8 * sl);
            }
            if live3 {
                pr3 = *positions.add(t0 + tt3) as usize;
                rg3 = (t0 + tt3) * qh_n + kh * ng + urem(r3, ng);
                v3c_qload(q, &mut qreg3, rg3 * hd + 8 * sl);
            }
            let mut acc0 = [0.0f32; 8];
            let mut acc1 = [0.0f32; 8];
            let mut acc2 = [0.0f32; 8];
            let mut acc3 = [0.0f32; 8];
            let mut mrun0 = f32::NEG_INFINITY;
            let mut mrun1 = f32::NEG_INFINITY;
            let mut mrun2 = f32::NEG_INFINITY;
            let mut mrun3 = f32::NEG_INFINITY;
            let mut lrun0 = 0.0f32;
            let mut lrun1 = 0.0f32;
            let mut lrun2 = 0.0f32;
            let mut lrun3 = 0.0f32;
            let pb0 = (warp * RL) * KT;
            let pb1 = pb0 + KT;
            let pb2 = pb1 + KT;
            let pb3 = pb2 + KT;
            let mut c0 = j0;
            while c0 < j1 {
                let cc = if c0 + KT < j1 { KT } else { j1 - c0 };
                // ── staging：K/V tile 协作装载（bf16→f32 解包 + swizzle）
                let nch = 32usize; // 每 token 的 16B chunk 数 = hd/8
                let nent = cc * nch;
                let mut e = tau;
                while e < nent {
                    let jj = e / 32;
                    let c = e % 32;
                    let slot = slot_of(block_table, s_seq, c0 + jj, bpr, bs);
                    let ebase = slot * he + kh * hd + c * 8;
                    v3c_stage8(ksm, k_cache, ebase, jj, c);
                    v3c_stage8(vsm, v_cache, ebase, jj, c);
                    e += bdim;
                }
                _syncthreads();
                // ── pass1：QK 点积（K 读一次供 RL=4 行）→ 全 warp bfly → pbuf(raw)
                let mut jj = 0usize;
                while jj < cc {
                    let kb = jj * 64 + sl;
                    let kf0 = *ksm.add(kb);
                    let kf1 = *ksm.add(kb + 32);
                    let s0 = v3c_dot8(&qreg0, &kf0, &kf1);
                    let s1 = v3c_dot8(&qreg1, &kf0, &kf1);
                    let s2 = v3c_dot8(&qreg2, &kf0, &kf1);
                    let s3 = v3c_dot8(&qreg3, &kf0, &kf1);
                    let j = c0 + jj;
                    let sc0 = if live0 && j <= pr0 { s0 } else { f32::NEG_INFINITY };
                    let sc1 = if live1 && j <= pr1 { s1 } else { f32::NEG_INFINITY };
                    let sc2 = if live2 && j <= pr2 { s2 } else { f32::NEG_INFINITY };
                    let sc3 = if live3 && j <= pr3 { s3 } else { f32::NEG_INFINITY };
                    if sl == 0 {
                        *psm.add(pb0 + jj) = sc0;
                        *psm.add(pb1 + jj) = sc1;
                        *psm.add(pb2 + jj) = sc2;
                        *psm.add(pb3 + jj) = sc3;
                    }
                    jj += 1;
                }
                bar_warp_sync(0xffff_ffff); // pbuf 写 → 读的 warp 内可见性
                // ── pass2：tile max(raw 域) → 在线合并系数 → p·V（V 载一次供四行）
                let mut tm0 = f32::NEG_INFINITY;
                let mut tm1 = f32::NEG_INFINITY;
                let mut tm2 = f32::NEG_INFINITY;
                let mut tm3 = f32::NEG_INFINITY;
                let mut jj = 0usize;
                while jj < cc {
                    let x0 = *psm.add(pb0 + jj);
                    let x1 = *psm.add(pb1 + jj);
                    let x2 = *psm.add(pb2 + jj);
                    let x3 = *psm.add(pb3 + jj);
                    tm0 = if x0 > tm0 { x0 } else { tm0 };
                    tm1 = if x1 > tm1 { x1 } else { tm1 };
                    tm2 = if x2 > tm2 { x2 } else { tm2 };
                    tm3 = if x3 > tm3 { x3 } else { tm3 };
                    jj += 1;
                }
                let mn0 = if tm0 > mrun0 { tm0 } else { mrun0 };
                let mn1 = if tm1 > mrun1 { tm1 } else { mrun1 };
                let mn2 = if tm2 > mrun2 { tm2 } else { mrun2 };
                let mn3 = if tm3 > mrun3 { tm3 } else { mrun3 };
                // mrun=−inf ⟹ scale=0（acc 原本全 0）；mn≥mrun 恒成立（指数 ≤0）
                let sc0 = if mrun0 == f32::NEG_INFINITY { 0.0 } else { fexp((mrun0 - mn0) * q_scale) };
                let sc1 = if mrun1 == f32::NEG_INFINITY { 0.0 } else { fexp((mrun1 - mn1) * q_scale) };
                let sc2 = if mrun2 == f32::NEG_INFINITY { 0.0 } else { fexp((mrun2 - mn2) * q_scale) };
                let sc3 = if mrun3 == f32::NEG_INFINITY { 0.0 } else { fexp((mrun3 - mn3) * q_scale) };
                v3c_scale8(&mut acc0, sc0);
                v3c_scale8(&mut acc1, sc1);
                v3c_scale8(&mut acc2, sc2);
                v3c_scale8(&mut acc3, sc3);
                lrun0 *= sc0;
                lrun1 *= sc1;
                lrun2 *= sc2;
                lrun3 *= sc3;
                let mnf0 = if mn0 == f32::NEG_INFINITY { f32::NEG_INFINITY } else { -mn0 * qs2 };
                let mnf1 = if mn1 == f32::NEG_INFINITY { f32::NEG_INFINITY } else { -mn1 * qs2 };
                let mnf2 = if mn2 == f32::NEG_INFINITY { f32::NEG_INFINITY } else { -mn2 * qs2 };
                let mnf3 = if mn3 == f32::NEG_INFINITY { f32::NEG_INFINITY } else { -mn3 * qs2 };
                let mut ls0 = 0.0f32;
                let mut ls1 = 0.0f32;
                let mut ls2 = 0.0f32;
                let mut ls3 = 0.0f32;
                let mut jj = 0usize;
                while jj < cc {
                    let vb = jj * 64 + sl;
                    let vf0 = *vsm.add(vb);
                    let vf1 = *vsm.add(vb + 32);
                    let p0 = fexp2(fma(*psm.add(pb0 + jj), qs2, mnf0));
                    let p1 = fexp2(fma(*psm.add(pb1 + jj), qs2, mnf1));
                    let p2 = fexp2(fma(*psm.add(pb2 + jj), qs2, mnf2));
                    let p3 = fexp2(fma(*psm.add(pb3 + jj), qs2, mnf3));
                    ls0 += p0;
                    ls1 += p1;
                    ls2 += p2;
                    ls3 += p3;
                    v3c_acc8(&mut acc0, p0, &vf0, &vf1);
                    v3c_acc8(&mut acc1, p1, &vf0, &vf1);
                    v3c_acc8(&mut acc2, p2, &vf0, &vf1);
                    v3c_acc8(&mut acc3, p3, &vf0, &vf1);
                    jj += 1;
                }
                lrun0 += ls0;
                lrun1 += ls1;
                lrun2 += ls2;
                lrun3 += ls3;
                mrun0 = mn0;
                mrun1 = mn1;
                mrun2 = mn2;
                mrun3 = mn3;
                _syncthreads();
                c0 += KT;
            }
            // ── 段末落盘（js=1 写 out；js>1 写偏结果供 attn_combine，
            // 空分片/全掩码行天然落 (−inf,0)：mrun=−inf、lrun=0、acc=0）
            if js == 1 {
                if live0 {
                    let ob = rg0 * hd + 8 * sl;
                    let mut l = 0usize;
                    while l < 8 {
                        *out.add(ob + l) = f32_to_bf16(acc0[l] / lrun0);
                        l += 1;
                    }
                }
                if live1 {
                    let ob = rg1 * hd + 8 * sl;
                    let mut l = 0usize;
                    while l < 8 {
                        *out.add(ob + l) = f32_to_bf16(acc1[l] / lrun1);
                        l += 1;
                    }
                }
                if live2 {
                    let ob = rg2 * hd + 8 * sl;
                    let mut l = 0usize;
                    while l < 8 {
                        *out.add(ob + l) = f32_to_bf16(acc2[l] / lrun2);
                        l += 1;
                    }
                }
                if live3 {
                    let ob = rg3 * hd + 8 * sl;
                    let mut l = 0usize;
                    while l < 8 {
                        *out.add(ob + l) = f32_to_bf16(acc3[l] / lrun3);
                        l += 1;
                    }
                }
            } else {
                if live0 {
                    let base = rg0 * js + sp;
                    if sl == 0 {
                        *part_ml.add(base * 2) = mrun0 * q_scale;
                        *part_ml.add(base * 2 + 1) = lrun0;
                    }
                    let pa = part_acc.add(base * PART_HD_STRIDE + 8 * sl);
                    let mut l = 0usize;
                    while l < 8 {
                        *pa.add(l) = acc0[l];
                        l += 1;
                    }
                }
                if live1 {
                    let base = rg1 * js + sp;
                    if sl == 0 {
                        *part_ml.add(base * 2) = mrun1 * q_scale;
                        *part_ml.add(base * 2 + 1) = lrun1;
                    }
                    let pa = part_acc.add(base * PART_HD_STRIDE + 8 * sl);
                    let mut l = 0usize;
                    while l < 8 {
                        *pa.add(l) = acc1[l];
                        l += 1;
                    }
                }
                if live2 {
                    let base = rg2 * js + sp;
                    if sl == 0 {
                        *part_ml.add(base * 2) = mrun2 * q_scale;
                        *part_ml.add(base * 2 + 1) = lrun2;
                    }
                    let pa = part_acc.add(base * PART_HD_STRIDE + 8 * sl);
                    let mut l = 0usize;
                    while l < 8 {
                        *pa.add(l) = acc2[l];
                        l += 1;
                    }
                }
                if live3 {
                    let base = rg3 * js + sp;
                    if sl == 0 {
                        *part_ml.add(base * 2) = mrun3 * q_scale;
                        *part_ml.add(base * 2 + 1) = lrun3;
                    }
                    let pa = part_acc.add(base * PART_HD_STRIDE + 8 * sl);
                    let mut l = 0usize;
                    while l < 8 {
                        *pa.add(l) = acc3[l];
                        l += 1;
                    }
                }
            }
            pass += 1;
        }
        u0 = u1;
    }
}

macro_rules! v3c_export {
    ($name:ident, $kt:literal) => {
        /// v3c 档（NV=32 单行组 + RL=4）：prefill 长序列容差档，hd=256 专用。
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
            v3c_body::<$kt>(
                q, k_cache, v_cache, block_table, seq_ids, positions, out, part_acc, part_ml,
                q_scale, bpr, block_size, q_heads, kv_heads, head_dim, tokens, ntok, jsplit,
                chunk, scratch,
            )
        }
    };
}

v3c_export!(paged_attn_v3_n32k16, 16);
v3c_export!(paged_attn_v3_n32k8, 8);

// ═══ v3b：ND=32 维/lane × NV=8 lane/行组 × RL=1（hd=256 专用档）═══
// 与 v3 的几何差异：warp 的 32 lane = 4 行组 × 8 维 lane，每条 shfl 链 3 级
// （v3 是 4 级），一条 warp 指令同步推进 4 行（v3 是 2 组×RL2 行间串行）。
// pbuf 存 raw 点积、掩码 −inf、ex2 折叠 q_scale·log2e、jsplit 偏量语义
// 与 v3/online 完全一致。行态标量化纪律同 v3（运行时索引数组会落 local）。

/// v3b staging：token 槽 = NV×8 个 F4（ND=32）。槽位 = 转置 swizzle
/// slot(g) = NV·(g%NV) + g/NV（g = F4 chunk 号 ∈ [0,64)）：
/// 对 NV=8 是 [0,64) 双射（旧式 (g%4)·NV+g/4 在 NV=8 时 g/4≥NV 碰撞），
/// 且 lane sl 读 chunk g=NV·sl+i ⟹ slot%NV=sl，v4 载跨 lane 无 bank 冲突。
#[inline(always)]
unsafe fn v3b_stage8<const NV: usize>(
    dst: *mut F4,
    src: *const u8,
    ebase: usize,
    jj: usize,
    c: usize,
) {
    let p = src as *const u16;
    let a = bf16x4(*(p.add(ebase) as *const u64));
    let b = bf16x4(*(p.add(ebase + 4) as *const u64));
    let g0 = 2 * c;
    let g1 = 2 * c + 1;
    let tbase = jj * (NV * 8);
    *dst.add(tbase + (g0 % NV) * NV + g0 / NV) =
        F4([bf16_to_f32(a[0]), bf16_to_f32(a[1]), bf16_to_f32(a[2]), bf16_to_f32(a[3])]);
    *dst.add(tbase + (g1 % NV) * NV + g1 / NV) =
        F4([bf16_to_f32(b[0]), bf16_to_f32(b[1]), bf16_to_f32(b[2]), bf16_to_f32(b[3])]);
}

/// q 行 32 维切片载寄存器
#[inline(always)]
unsafe fn v3b_qload(q: *const u16, dst: &mut [f32; 32], qb: usize) {
    let mut k = 0usize;
    while k < 8 {
        let qs = bf16x4(*(q.add(qb + 4 * k) as *const u64));
        dst[4 * k] = bf16_to_f32(qs[0]);
        dst[4 * k + 1] = bf16_to_f32(qs[1]);
        dst[4 * k + 2] = bf16_to_f32(qs[2]);
        dst[4 * k + 3] = bf16_to_f32(qs[3]);
        k += 1;
    }
}

/// 32 维切片点积 + 段内 bfly 归约（3 级，NV lane 同和）。收敛点调用。
/// sb = tile 内 F4 槽基（jj·NV·8 + sl）；转置槽位映射下：
/// sb+m·NV = 槽 8m+sl 持维 [32sl+4m, +4)，
/// sb+m·NV+NV·(NV/2) = 槽 8(m+4)+sl 持维 [32sl+16+4m, +4)。
#[inline(always)]
fn v3b_dot32<const NV: usize>(qr: &[f32; 32], ksm: *mut F4, sb: usize) -> f32 {
    let mut ch = [0.0f32; 4];
    let mut m = 0usize;
    while m < 4 {
        let k0 = unsafe { *ksm.add(sb + m * NV) };
        let k1 = unsafe { *ksm.add(sb + m * NV + NV * (NV / 2)) };
        ch[0] = fma(qr[4 * m], k0.0[0], ch[0]);
        ch[1] = fma(qr[4 * m + 1], k0.0[1], ch[1]);
        ch[2] = fma(qr[4 * m + 2], k0.0[2], ch[2]);
        ch[3] = fma(qr[4 * m + 3], k0.0[3], ch[3]);
        ch[0] = fma(qr[16 + 4 * m], k1.0[0], ch[0]);
        ch[1] = fma(qr[16 + 4 * m + 1], k1.0[1], ch[1]);
        ch[2] = fma(qr[16 + 4 * m + 2], k1.0[2], ch[2]);
        ch[3] = fma(qr[16 + 4 * m + 3], k1.0[3], ch[3]);
        m += 1;
    }
    let mut s = (ch[0] + ch[1]) + (ch[2] + ch[3]);
    let mut off = NV / 2;
    while off > 0 {
        s += shfl_bfly_f32(s, off as i32);
        off /= 2;
    }
    s
}

/// acc[32] += p·v 已内联在 v3b_body 的 pass2（槽位映射同 v3b_dot32）。

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn v3b_body<const NV: usize, const KT: usize>(
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
    _head_dim: u32,
    tokens: u32,
    ntok: u32,
    jsplit: u32,
    _chunk: u32,
    _scratch: *mut f32,
) {
    let hd = NV * 32;
    let qh_n = q_heads as usize;
    let kv_n = kv_heads as usize;
    let ng = udiv(qh_n, kv_n);
    let nt = ntok as usize; // = Br
    let js = jsplit as usize;
    let toks = tokens as usize;
    let bdim = _block_dim_x() as usize;
    let tau = _thread_idx_x() as usize;
    let gpw = 32 / NV; // 行组/warp（RL=1 ⟹ 行/warp/pass）
    let ntile = udiv(toks + nt - 1, nt);
    let blk = _block_idx_x() as usize;
    if blk >= kv_n * ntile * js {
        return;
    }
    let kh = udiv(blk, ntile * js);
    let rest = urem(blk, ntile * js);
    let tile = ntile - 1 - udiv(rest, js); // 反序：长 span 块先发射藏因果尾
    let sp = urem(rest, js);
    let t0 = tile * nt;
    let tn = if t0 + nt < toks { nt } else { toks - t0 };
    let rows = tn * ng;
    let (bpr, bs) = (bpr as usize, block_size as usize);
    let he = kv_n * hd;
    let nw = udiv(bdim, 32);
    let warp = udiv(tau, 32);
    let lane32 = urem(tau, 32);
    let g = udiv(lane32, NV); // 行组号
    let sl = urem(lane32, NV); // 维 lane（持 [32sl, 32sl+32)）
    let npass = udiv(rows + nw * gpw - 1, nw * gpw);
    let (ksm, vsm, psm) = v3_smem::<KT>();
    let qs2 = q_scale * 1.4426950408889634;

    let mut u0 = 0usize;
    while u0 < tn {
        let s_seq = *seq_ids.add(t0 + u0) as usize;
        let mut u1 = u0 + 1;
        while u1 < tn && *seq_ids.add(t0 + u1) as usize == s_seq {
            u1 += 1;
        }
        let mut pmax = 0usize;
        let mut u = u0;
        while u < u1 {
            let p = *positions.add(t0 + u) as usize;
            pmax = if p > pmax { p } else { pmax };
            u += 1;
        }
        let lspan = udiv(pmax + js, js);
        let jmax = pmax + 1;
        let j0 = sp * lspan;
        let j1 = if j0 + lspan < jmax { j0 + lspan } else { jmax };
        let mut pass = 0usize;
        while pass < npass {
            let row = (pass * nw + warp) * gpw + g;
            let tt = udiv(row, ng);
            let live = row < rows && tt >= u0 && tt < u1;
            let mut pr = 0usize;
            let mut rg = 0usize;
            let mut qreg = [0.0f32; 32];
            if live {
                pr = *positions.add(t0 + tt) as usize;
                rg = (t0 + tt) * qh_n + kh * ng + urem(row, ng);
                v3b_qload(q, &mut qreg, rg * hd + 32 * sl);
            }
            let mut acc = [0.0f32; 32];
            let mut mrun = f32::NEG_INFINITY;
            let mut lrun = 0.0f32;
            let pb = (warp * gpw + g) * KT;
            let mut c0 = j0;
            while c0 < j1 {
                let cc = if c0 + KT < j1 { KT } else { j1 - c0 };
                // staging（nch = hd/8 = NV×4 chunk/token）
                let nch = NV * 4;
                let nent = cc * nch;
                let mut e = tau;
                while e < nent {
                    let jj = udiv(e, nch);
                    let c = urem(e, nch);
                    let slot = slot_of(block_table, s_seq, c0 + jj, bpr, bs);
                    let ebase = slot * he + kh * hd + c * 8;
                    v3b_stage8::<NV>(ksm, k_cache, ebase, jj, c);
                    v3b_stage8::<NV>(vsm, v_cache, ebase, jj, c);
                    e += bdim;
                }
                _syncthreads();
                // pass1：QK + 3 级 bfly + 融合 tile max（全 lane 冗余同值）→ pbuf(raw)
                let mut tm = f32::NEG_INFINITY;
                let mut jj = 0usize;
                while jj < cc {
                    let s = v3b_dot32::<NV>(&qreg, ksm, jj * (NV * 8) + sl);
                    let j = c0 + jj;
                    let sc = if live && j <= pr { s } else { f32::NEG_INFINITY };
                    tm = if sc > tm { sc } else { tm };
                    if sl == 0 {
                        *psm.add(pb + jj) = sc;
                    }
                    jj += 1;
                }
                bar_warp_sync(0xffff_ffff); // pbuf 写 → 读的 warp 内可见性
                // pass2：在线合并系数 → p·V（ex2 折叠；空 tile 走 mnf=−inf 归零）
                let mn = if tm > mrun { tm } else { mrun };
                let sc = if mrun == f32::NEG_INFINITY {
                    0.0
                } else {
                    fexp((mrun - mn) * q_scale)
                };
                let mut l = 0usize;
                while l < 32 {
                    acc[l] *= sc;
                    l += 1;
                }
                lrun *= sc;
                let mnf = if mn == f32::NEG_INFINITY {
                    f32::NEG_INFINITY
                } else {
                    -mn * qs2
                };
                let mut ls = 0.0f32;
                let mut jj = 0usize;
                while jj < cc {
                    let p = fexp2(fma(*psm.add(pb + jj), qs2, mnf));
                    ls += p;
                    let vb = jj * (NV * 8) + sl;
                    let mut m = 0usize;
                    while m < 4 {
                        let v0 = *vsm.add(vb + m * NV);
                        let v1 = *vsm.add(vb + m * NV + NV * (NV / 2));
                        acc[4 * m] = fma(p, v0.0[0], acc[4 * m]);
                        acc[4 * m + 1] = fma(p, v0.0[1], acc[4 * m + 1]);
                        acc[4 * m + 2] = fma(p, v0.0[2], acc[4 * m + 2]);
                        acc[4 * m + 3] = fma(p, v0.0[3], acc[4 * m + 3]);
                        acc[16 + 4 * m] = fma(p, v1.0[0], acc[16 + 4 * m]);
                        acc[16 + 4 * m + 1] = fma(p, v1.0[1], acc[16 + 4 * m + 1]);
                        acc[16 + 4 * m + 2] = fma(p, v1.0[2], acc[16 + 4 * m + 2]);
                        acc[16 + 4 * m + 3] = fma(p, v1.0[3], acc[16 + 4 * m + 3]);
                        m += 1;
                    }
                    jj += 1;
                }
                lrun += ls;
                mrun = mn;
                _syncthreads();
                c0 += KT;
            }
            // 段末落盘（js=1 写 out；js>1 偏结果，空分片天然 (−inf,0)）
            if live {
                if js == 1 {
                    let ob = rg * hd + 32 * sl;
                    let mut l = 0usize;
                    while l < 32 {
                        *out.add(ob + l) = f32_to_bf16(acc[l] / lrun);
                        l += 1;
                    }
                } else {
                    let base = rg * js + sp;
                    if sl == 0 {
                        *part_ml.add(base * 2) = mrun * q_scale;
                        *part_ml.add(base * 2 + 1) = lrun;
                    }
                    let pa = part_acc.add(base * PART_HD_STRIDE + 32 * sl);
                    let mut l = 0usize;
                    while l < 32 {
                        *pa.add(l) = acc[l];
                        l += 1;
                    }
                }
            }
            pass += 1;
        }
        u0 = u1;
    }
}

macro_rules! v3b_export {
    ($name:ident, $nv:literal, $kt:literal) => {
        /// v3b 档（ND=32/NV=8/RL=1）：prefill 长序列容差档，hd=256 专用。
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
            v3b_body::<$nv, $kt>(
                q, k_cache, v_cache, block_table, seq_ids, positions, out, part_acc, part_ml,
                q_scale, bpr, block_size, q_heads, kv_heads, head_dim, tokens, ntok, jsplit,
                chunk, scratch,
            )
        }
    };
}

v3b_export!(paged_attn_v3_d32k16, 8, 16);
v3b_export!(paged_attn_v3_d32k8, 8, 8);

// ═══ v4：wmma（tensor core HMMA）FA2 形态——S=Q@K^T 与 P@V 全走 m16n16k16 ═══
// 几何：block = 1 warp = (q 头, 16 q 行片)；grid = q_heads × ⌈tokens/16⌉。
// K 从 global 直读（wmma load.b.col：页内 16 token 连续，B(k=hd,c=tok) 的
// col-major 视图恰好就是 K row-major——零转置零 staging）；V 转置进 smem
//（P@V 的 B 需要 col-major = V^T[hd][tok]）；P 经 smem 走 bf16 量化。
// 在线 softmax 与 online_body/v3 同语义（tile max 归一 + 行级 rescale）。
// fragment 布局（f32 累加器 m16n16k16，PTX ISA）：lane l 持 2 行 × 4 列——
// 行 g=l/4 与 g+8，列 2·(l%4)+{0,1} 与 +8；d0/d1=(g,2t/2t+1)、d2/d3=(g+8,·)、
// d4/d5=(g,2t+8/2t+9)、d6/d7=(g+8,·)。行 max/sum = lane 内 4 值 + 2 级
// bfly（lane 组 {4g..4g+3}）。O = 16 组（hd 256/16）× 8 f32 全驻寄存器。
// 合同（host 档选择保证）：16 行同序列、行序 = token 序、纯 prefill 批、
// hd=256、页=16 token、jsplit 忽略恒 1（直写 out，无偏量表）。
// 数值：mma 内部累加序 ≠ reference ⇒ 容差档（o38 钉 rel2 ≤ 3e-3），
/// v4 档（wmma tensor core，1 warp/块 = (q 头, 16 行片)）：纯 prefill 长序列用。
/// KV 池布局 = 生产 [slot=page·bs+tok][kvh][hd]（与 `slot_of` 同构）。
/// **host 合同：tokens 须 16 对齐且 ≤ 真实行数**（尾部行由 host 切给 v3）；
/// kernel 内部对 pos 读/写回另有钳位保险带。GQA：kvh = qh/(qh_n/kv_n)。
// exp 走 fexp（ex2.approx，v3 容差档同款）。
#[link_section = ".shared"]
static mut V4_VT: [u32; 256 * 8] = [0; 2048]; // V^T tile [hd][16tok] bf16 对
#[link_section = ".shared"]
static mut V4_P: [u32; 16 * 8] = [0; 128]; // P tile [16q][16kv] bf16 对

macro_rules! v4_unroll16 {
    ($m:ident) => {
        $m!(0); $m!(1); $m!(2); $m!(3); $m!(4); $m!(5); $m!(6); $m!(7);
        $m!(8); $m!(9); $m!(10); $m!(11); $m!(12); $m!(13); $m!(14); $m!(15);
    };
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn paged_attn_v4(
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
    let _ = (part_acc, part_ml, ntok, jsplit, chunk, scratch, head_dim, block_size);
    let lane = unsafe { _thread_idx_x() } as usize;
    let g = lane >> 2; // fragment 行 g（另一行 g+8）
    let t2 = (lane & 3) << 1; // fragment 列基 2t（另一列组 +8）
    let bid = unsafe { _block_idx_x() } as usize;
    // grid = q_heads(=16) × ⌈tokens/16⌉，头/片分解用位运算——整数除法会
    // emit panic 路径（k2 禁宿主调用）；几何常量与 wmma ldm ImmArg 同源
    let qh = bid & 15;
    let row0 = bid & !15;
    let kvh = qh >> 2; // GQA 组 = q_heads/kv_heads = 4 内建
    let last = tokens as usize - 1;
    let seq = unsafe { *seq_ids.add(row0) } as usize;
    // 尾片（tokens 非 16 倍数）：越界行索引钳到末行——死行只算不写
    let pos_g = unsafe { *positions.add((row0 + g).min(last)) } as usize;
    let pos_g8 = unsafe { *positions.add((row0 + g + 8).min(last)) } as usize;
    let max_pos = unsafe { *positions.add((row0 + 15).min(last)) } as usize; // 行序 = token 序
    let n_tile = (max_pos >> 4) + 1;

    let bt = unsafe { block_table.add(seq * bpr as usize) };
    let qbase = ((row0 * q_heads as usize + qh) * 256) >> 1; // u32 元素单位
    let qp = q as *const u32;
    let outp = out as *mut u16;
    let orow_g = ((row0 + g) * q_heads as usize + qh) * 256;
    let orow_g8 = ((row0 + g + 8) * q_heads as usize + qh) * 256;

    // V^T smem / P smem 的泛型指针（build.rs 已改写 .shared）
    let vt = core::ptr::with_exposed_provenance_mut::<u32>(
        core::ptr::addr_of_mut!(V4_VT) as usize,
    );
    let pp = core::ptr::with_exposed_provenance_mut::<u32>(
        core::ptr::addr_of_mut!(V4_P) as usize,
    );

    // O 16 组 × 8 f32 全驻寄存器（literal 下标宏展开防 local——SROA 拆）
    type O8 = (f32, f32, f32, f32, f32, f32, f32, f32);
    let mut o: [O8; 16] = [(0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32); 16];
    let mut m_g = f32::NEG_INFINITY;
    let mut m_g8 = f32::NEG_INFINITY;
    let mut l_g = 0f32;
    let mut l_g8 = 0f32;
    let mut tile = 0usize;
    while tile < n_tile {
        // 上一 tile 的 o_mma 读 vt 与本 tile staging 覆写之间必须隔断
        //（warp 内 smem 写→读跨 lane 可见性）
        unsafe { bar_warp_sync(0xffff_ffff) };
        let page = unsafe { *bt.add(tile) } as usize;
        // 生产布局 [slot=page·16+tok][kvh][hd]（slot_of 同构）：token 行距 he=kv_heads·256
        let he = kv_heads as usize * 256;
        let kbase = page * 16 * he + kvh * 256;
        let kpage = (k_cache as *const u32).add(kbase >> 1);
        let vpage = (v_cache as *const u16).add(kbase);

        // ── V^T 转置 staging：lane 认领 hd（8 轮 × 32），global 合并读、
        //    smem 散写（V[tok][hd] → VT[hd][tok 对]）──
        let mut r = 0usize;
        while r < 8 {
            let hd = lane + (r << 5);
            let mut j = 0usize;
            while j < 8 {
                let lo = unsafe { *vpage.add((j << 1) * he + hd) } as u32;
                let hi = unsafe { *vpage.add(((j << 1) + 1) * he + hd) } as u32;
                unsafe { *vt.add((hd << 3) + j) = lo | (hi << 16) };
                j += 1;
            }
            r += 1;
        }
        // ── S = Q@K^T：16 步 k（hd 256/16），Q/K 全 global 直读 ──
        let mut s = (0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32);
        let mut k = 0usize;
        while k < 16 {
            let qa = unsafe { wmma_load_a(qp.add(qbase + (k << 3)), 4096) };
            let kb = unsafe { wmma_load_b(kpage.add(k << 3), 1024) };
            s = unsafe {
                wmma_mma(
                    qa.0, qa.1, qa.2, qa.3, kb.0, kb.1, kb.2, kb.3, s.0, s.1, s.2, s.3, s.4, s.5,
                    s.6, s.7,
                )
            };
            k += 1;
        }

        // ── 掩码（因果）+ 行 max 归约（lane 内 4 值 → 2 级 bfly）──
        let j0 = tile << 4;
        let inf = f32::NEG_INFINITY;
        let s0 = if j0 + t2 <= pos_g { s.0 * q_scale } else { inf };
        let s1 = if j0 + t2 + 1 <= pos_g { s.1 * q_scale } else { inf };
        let s4 = if j0 + t2 + 8 <= pos_g { s.4 * q_scale } else { inf };
        let s5 = if j0 + t2 + 9 <= pos_g { s.5 * q_scale } else { inf };
        let s2 = if j0 + t2 <= pos_g8 { s.2 * q_scale } else { inf };
        let s3 = if j0 + t2 + 1 <= pos_g8 { s.3 * q_scale } else { inf };
        let s6 = if j0 + t2 + 8 <= pos_g8 { s.6 * q_scale } else { inf };
        let s7 = if j0 + t2 + 9 <= pos_g8 { s.7 * q_scale } else { inf };
        let mut mg = s0.max(s1).max(s4).max(s5);
        let mut mg8 = s2.max(s3).max(s6).max(s7);
        // bfly 只交换不合并：必须 max/加法回叠（v3 点积同款纪律）
        mg = mg.max(shfl_bfly_f32(mg, 1));
        mg = mg.max(shfl_bfly_f32(mg, 2));
        mg8 = mg8.max(shfl_bfly_f32(mg8, 1));
        mg8 = mg8.max(shfl_bfly_f32(mg8, 2));
        let mn_g = m_g.max(mg);
        let mn_g8 = m_g8.max(mg8);
        let ag = fexp(m_g - mn_g); // 首 tile：exp(−inf)=0 ✓（行至少 1 可见列）
        let ag8 = fexp(m_g8 - mn_g8);

        // ── p = exp(s − mn)，行 sum 归约，O rescale ──
        let p0 = fexp(s0 - mn_g);
        let p1 = fexp(s1 - mn_g);
        let p4 = fexp(s4 - mn_g);
        let p5 = fexp(s5 - mn_g);
        let p2 = fexp(s2 - mn_g8);
        let p3 = fexp(s3 - mn_g8);
        let p6 = fexp(s6 - mn_g8);
        let p7 = fexp(s7 - mn_g8);
        let mut sg = p0 + p1 + p4 + p5;
        let mut sg8 = p2 + p3 + p6 + p7;
        sg = sg + shfl_bfly_f32(sg, 1);
        sg = sg + shfl_bfly_f32(sg, 2);
        sg8 = sg8 + shfl_bfly_f32(sg8, 1);
        sg8 = sg8 + shfl_bfly_f32(sg8, 2);
        l_g = l_g * ag + sg;
        l_g8 = l_g8 * ag8 + sg8;
        macro_rules! o_rescale {
            ($i:literal) => {
                o[$i].0 *= ag;
                o[$i].1 *= ag;
                o[$i].4 *= ag;
                o[$i].5 *= ag;
                o[$i].2 *= ag8;
                o[$i].3 *= ag8;
                o[$i].6 *= ag8;
                o[$i].7 *= ag8;
            };
        }
        v4_unroll16!(o_rescale);

        // ── P 落 smem（bf16 量化，fragment 位置直写）→ warp 屏障 → A 读回 ──
        let pk = |lo: f32, hi: f32| -> u32 {
            (f32_to_bf16(lo) as u32) | ((f32_to_bf16(hi) as u32) << 16)
        };
        unsafe {
            *pp.add((g << 3) + (t2 >> 1)) = pk(p0, p1);
            *pp.add(((g + 8) << 3) + (t2 >> 1)) = pk(p2, p3);
            *pp.add((g << 3) + ((t2 + 8) >> 1)) = pk(p4, p5);
            *pp.add(((g + 8) << 3) + ((t2 + 8) >> 1)) = pk(p6, p7);
        }
        unsafe { bar_warp_sync(0xffff_ffff) };
        let (a0, a1, a2, a3) = unsafe { wmma_load_a(pp as *const u32, 16) };
        // ── O += P@V^T：16 步 n（hd 256/16）──
        // VT=[hd][16tok]，B 片段 d 维沿 ldm=16 走 hd ⇒ 组 i 的 hd 基 = 16i ⇒ u32 偏移 16i*8=i<<7
        macro_rules! o_mma {
            ($i:literal) => {
                let vb = unsafe { wmma_load_b(vt.add($i << 7), 16) };
                o[$i] = unsafe {
                    wmma_mma(
                        a0, a1, a2, a3, vb.0, vb.1, vb.2, vb.3, o[$i].0, o[$i].1, o[$i].2,
                        o[$i].3, o[$i].4, o[$i].5, o[$i].6, o[$i].7,
                    )
                };
            };
        }
        v4_unroll16!(o_mma);

        m_g = mn_g;
        m_g8 = mn_g8;
        tile += 1;
    }

    // ── 尾写：O/l → bf16 散写 out（行归属：0/1/4/5→g，2/3/6/7→g+8）──
    let inv_g = 1.0 / l_g;
    let inv_g8 = 1.0 / l_g8;
    // 尾片：越界行（row ≥ tokens）不写
    let (wg, wg8) = (row0 + g < last + 1, row0 + g + 8 < last + 1);
    macro_rules! o_write {
        ($i:literal) => {
            unsafe {
                if wg {
                    *outp.add(orow_g + ($i << 4) + t2) = f32_to_bf16(o[$i].0 * inv_g);
                    *outp.add(orow_g + ($i << 4) + t2 + 1) = f32_to_bf16(o[$i].1 * inv_g);
                    *outp.add(orow_g + ($i << 4) + t2 + 8) = f32_to_bf16(o[$i].4 * inv_g);
                    *outp.add(orow_g + ($i << 4) + t2 + 9) = f32_to_bf16(o[$i].5 * inv_g);
                }
                if wg8 {
                    *outp.add(orow_g8 + ($i << 4) + t2) = f32_to_bf16(o[$i].2 * inv_g8);
                    *outp.add(orow_g8 + ($i << 4) + t2 + 1) = f32_to_bf16(o[$i].3 * inv_g8);
                    *outp.add(orow_g8 + ($i << 4) + t2 + 8) = f32_to_bf16(o[$i].6 * inv_g8);
                    *outp.add(orow_g8 + ($i << 4) + t2 + 9) = f32_to_bf16(o[$i].7 * inv_g8);
                }
            }
        };
    }
    v4_unroll16!(o_write);
}

// ═══ cp.async 占位函数（build.rs `patch_cp_async` 整体替换函数体）═══
// NVVM 有 `llvm.nvvm.cp.async.*` intrinsic，但指针参数要求 addrspace(3)/(1)
// ——Rust 类型系统表达不了（实测报 intrinsic signature mismatch）；nvptx
// 后端又无 asm!（无寄存器类）。路子与 `patch_smem_symbols` 同族：占位体
// 用 volatile 直拷（保证 rustc 不 IPO 删调用），build.rs 把函数体文本替换
// 为真指令。dst 传 smem 泛型指针（补丁内先 cvta.to.shared）。
// **合同：补丁失败即 build panic——占位体（volatile 写）绝不允许上线**。
#[inline(never)]
#[no_mangle]
pub unsafe fn cp_async_cg16(dst: *mut u8, src: *const u8) {
    unsafe {
        core::ptr::write_volatile(
            dst as *mut u64,
            core::ptr::read_volatile(src as *const u64),
        )
    }
}

#[inline(never)]
#[no_mangle]
pub fn cp_async_commit(_dummy: *mut u32) {
    unsafe { core::ptr::write_volatile(_dummy, 0) }
}

#[inline(never)]
#[no_mangle]
pub fn cp_async_wait_all(_dummy: *mut u32) {
    unsafe { core::ptr::write_volatile(_dummy, 0) }
}

// ═══ v5：4 warp × 64 行 CTA 共享 K/V smem（KV 流量 ÷4——每字节供 64 行 ═══
// ═══ 消费而非 16 行）。几何：block = 128 线程（4 warp）= (q 头, 64 行组），
// grid = q_heads × ⌈tokens/64⌉。K 平铺进 smem（wmma load.b 的 col-major
// 视图 = row-major K，零转置仅换源）；V^T 转置 staging 同 v4 由 128 线程
// 分摊；P 每 warp 私有 128 u32 槽。在线 softmax 按行分 warp，无跨 warp
// 交互。K/V 双缓冲：tile t 计算与 tile t+1 staging 重叠，唯一 CTA 屏障在
// 迭代头（buf (t+1)%2 上轮计算已读完 + 本轮 staging 已到齐），P 写→读
// 仍是 warp 屏障。n_tile 取 64 行组最大 pos（行序 = token 序），早停粒度
// 64 行——与 vLLM FA2 Br=64 同粒度。
// **位级合同：v5 ≡ v4 逐位**（同 fragment 算术，只换 K/V 来源与行分属，
// o39 钉死）；host 合同：tokens 64 对齐、单 span、16q/4kv/hd256、bf16 池。
#[link_section = ".shared"]
static mut V5_KS: [u32; 2 * 16 * 128] = [0; 4096]; // K tile ×2 缓冲 [16tok][256hd] bf16 对（平铺）
#[link_section = ".shared"]
static mut V5_VT: [u32; 2 * 256 * 8] = [0; 4096]; // V^T tile ×2 缓冲 [hd][16tok] bf16 对
#[link_section = ".shared"]
static mut V5_P: [u32; 4 * 128] = [0; 512]; // P tile ×4 warp 私有 [16q][16kv]

#[no_mangle]
pub unsafe extern "ptx-kernel" fn paged_attn_v5(
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
    let _ = (part_acc, part_ml, ntok, jsplit, chunk, scratch, head_dim, block_size);
    let tid = unsafe { _thread_idx_x() } as usize;
    let lane = tid & 31;
    let w = tid >> 5; // warp 号 = 本 warp 的 16 行子组
    let g = lane >> 2; // fragment 行 g（另一行 g+8）
    let t2 = (lane & 3) << 1; // fragment 列基 2t（另一列组 +8）
    let bid = unsafe { _block_idx_x() } as usize;
    // grid = q_heads(=16) × ⌈tokens/64⌉：qh 占低 4 位，行组号 = bid>>4
    let qh = bid & 15;
    let row0 = (bid >> 4) << 6;
    let kvh = qh >> 2; // GQA 组 = q_heads/kv_heads = 4 内建
    let last = tokens as usize - 1;
    let seq = unsafe { *seq_ids.add(row0) } as usize;
    let rbase = row0 + (w << 4); // 本 warp 的 16 行基
    let pos_g = unsafe { *positions.add((rbase + g).min(last)) } as usize;
    let pos_g8 = unsafe { *positions.add((rbase + g + 8).min(last)) } as usize;
    // CTA 公共 tile 数：64 行组最大 pos（行序 = token 序 ⇒ 末行最大）。
    // pos 较小的 warp 多吃的 tile 全列被掩成 −inf ⇒ p=0、l/o 不动（v4 对
    // 组内短行同款语义）
    let max_pos = unsafe { *positions.add((row0 + 63).min(last)) } as usize;
    let n_tile = (max_pos >> 4) + 1;

    let bt = unsafe { block_table.add(seq * bpr as usize) };
    let qbase = ((rbase * q_heads as usize + qh) * 256) >> 1; // u32 元素单位
    let qp = q as *const u32;
    let outp = out as *mut u16;
    let orow_g = ((rbase + g) * q_heads as usize + qh) * 256;
    let orow_g8 = ((rbase + g + 8) * q_heads as usize + qh) * 256;

    let ks = core::ptr::with_exposed_provenance_mut::<u32>(
        core::ptr::addr_of_mut!(V5_KS) as usize,
    );
    let vt = core::ptr::with_exposed_provenance_mut::<u32>(
        core::ptr::addr_of_mut!(V5_VT) as usize,
    );
    let ppw = core::ptr::with_exposed_provenance_mut::<u32>(
        core::ptr::addr_of_mut!(V5_P) as usize,
    )
    .add(w << 7); // warp 私有 128 u32

    type O8 = (f32, f32, f32, f32, f32, f32, f32, f32);
    let mut o: [O8; 16] = [(0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32); 16];
    let mut m_g = f32::NEG_INFINITY;
    let mut m_g8 = f32::NEG_INFINITY;
    let mut l_g = 0f32;
    let mut l_g8 = 0f32;

    // ── staging（global→smem，128 线程分摊）：K 走 cp.async.cg 16B ×4 块/
    //    线程（占位函数经 build.rs 补丁装载——绕过寄存器直达 smem，无
    //    ld→st 依赖链）。寻址与旧 u64 版逐槽等价：块号 c → token=c>>5、
    //    源跨步 he<<1 字节/token（global [16tok][kv_heads][256d] 交错，
    //    dst 是 [16tok][256d] 平铺）；dst/src 均 16B 对齐（KS 声明 .align 16
    //    由补丁提升，global 侧 kbase = 32768·page + 512·kvh）。V^T 转置仍
    //    ld+st（含 lo|hi<<16 计算——cp.async 做不了转置）──
    let he = kv_heads as usize * 256;
    let stage = |tile: usize, buf: usize, k_cache: *const u8, v_cache: *const u8, bt: *const u32| {
        let page = unsafe { *bt.add(tile) } as usize;
        let kbase = (page * 16 * he + kvh * 256) << 1; // 字节偏移
        let ksrc = k_cache.add(kbase);
        let kdst = (ks as *mut u8).add(buf << 13); // 每缓冲 8KB
        let mut r = 0usize;
        while r < 4 {
            let c = tid + (r << 7); // 16B 块号 0..511
            let soff = ((c >> 5) * he << 1) + ((c & 31) << 4);
            unsafe { cp_async_cg16(kdst.add(c << 4), ksrc.add(soff)) };
            r += 1;
        }
        unsafe { cp_async_commit(ppw as *mut u32) };
        let vpage = (v_cache as *const u16).add(kbase >> 1);
        let vt2 = vt.add(buf << 11);
        let mut r = 0usize;
        while r < 2 {
            let hd = tid + (r << 7);
            let mut j = 0usize;
            while j < 8 {
                let lo = unsafe { *vpage.add((j << 1) * he + hd) } as u32;
                let hi = unsafe { *vpage.add(((j << 1) + 1) * he + hd) } as u32;
                unsafe { *vt2.add((hd << 3) + j) = lo | (hi << 16) };
                j += 1;
            }
            r += 1;
        }
    };

    // ── 双缓冲流水：tile t 计算读 buf t%2，tile t+1 的 staging 写 buf
    //    (t+1)%2——缓冲不相交，staging 与计算重叠；迭代头 = cp.async
    //    wait_all（本线程在途拷贝到齐）+ CTA 屏障（全部线程可见 + 上轮
    //    计算已读完）。位级合同不变：staging 值与来源逐字同 ──
    stage(0, 0, k_cache, v_cache, bt);
    let mut tile = 0usize;
    while tile < n_tile {
        unsafe { cp_async_wait_all(ppw as *mut u32) };
        unsafe { _syncthreads() };
        let buf = tile & 1;
        let nxt = tile + 1;
        if nxt < n_tile {
            stage(nxt, nxt & 1, k_cache, v_cache, bt); // 在途；与下方计算重叠
        }
        let ksb = ks.add(buf << 11);
        let vtb = vt.add(buf << 11);

        // ── S = Q@K^T：16 步 k（hd 256/16），Q global 直读、K 自 smem ──
        let mut s = (0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0f32);
        let mut k = 0usize;
        while k < 16 {
            let qa = unsafe { wmma_load_a(qp.add(qbase + (k << 3)), 4096) };
            let kb = unsafe { wmma_load_b(ksb.add(k << 3), 256) };
            s = unsafe {
                wmma_mma(
                    qa.0, qa.1, qa.2, qa.3, kb.0, kb.1, kb.2, kb.3, s.0, s.1, s.2, s.3, s.4, s.5,
                    s.6, s.7,
                )
            };
            k += 1;
        }

        // ── 掩码（因果）+ 行 max 归约（lane 内 4 值 → 2 级 bfly）──
        let j0 = tile << 4;
        let inf = f32::NEG_INFINITY;
        let s0 = if j0 + t2 <= pos_g { s.0 * q_scale } else { inf };
        let s1 = if j0 + t2 + 1 <= pos_g { s.1 * q_scale } else { inf };
        let s4 = if j0 + t2 + 8 <= pos_g { s.4 * q_scale } else { inf };
        let s5 = if j0 + t2 + 9 <= pos_g { s.5 * q_scale } else { inf };
        let s2 = if j0 + t2 <= pos_g8 { s.2 * q_scale } else { inf };
        let s3 = if j0 + t2 + 1 <= pos_g8 { s.3 * q_scale } else { inf };
        let s6 = if j0 + t2 + 8 <= pos_g8 { s.6 * q_scale } else { inf };
        let s7 = if j0 + t2 + 9 <= pos_g8 { s.7 * q_scale } else { inf };
        let mut mg = s0.max(s1).max(s4).max(s5);
        let mut mg8 = s2.max(s3).max(s6).max(s7);
        mg = mg.max(shfl_bfly_f32(mg, 1));
        mg = mg.max(shfl_bfly_f32(mg, 2));
        mg8 = mg8.max(shfl_bfly_f32(mg8, 1));
        mg8 = mg8.max(shfl_bfly_f32(mg8, 2));
        let mn_g = m_g.max(mg);
        let mn_g8 = m_g8.max(mg8);
        let ag = fexp(m_g - mn_g); // 首 tile：exp(−inf)=0 ✓（行至少 1 可见列）
        let ag8 = fexp(m_g8 - mn_g8);

        // ── p = exp(s − mn)，行 sum 归约，O rescale ──
        let p0 = fexp(s0 - mn_g);
        let p1 = fexp(s1 - mn_g);
        let p4 = fexp(s4 - mn_g);
        let p5 = fexp(s5 - mn_g);
        let p2 = fexp(s2 - mn_g8);
        let p3 = fexp(s3 - mn_g8);
        let p6 = fexp(s6 - mn_g8);
        let p7 = fexp(s7 - mn_g8);
        let mut sg = p0 + p1 + p4 + p5;
        let mut sg8 = p2 + p3 + p6 + p7;
        sg = sg + shfl_bfly_f32(sg, 1);
        sg = sg + shfl_bfly_f32(sg, 2);
        sg8 = sg8 + shfl_bfly_f32(sg8, 1);
        sg8 = sg8 + shfl_bfly_f32(sg8, 2);
        l_g = l_g * ag + sg;
        l_g8 = l_g8 * ag8 + sg8;
        macro_rules! o_rescale {
            ($i:literal) => {
                o[$i].0 *= ag;
                o[$i].1 *= ag;
                o[$i].4 *= ag;
                o[$i].5 *= ag;
                o[$i].2 *= ag8;
                o[$i].3 *= ag8;
                o[$i].6 *= ag8;
                o[$i].7 *= ag8;
            };
        }
        v4_unroll16!(o_rescale);

        // ── P 落 smem（bf16 量化，fragment 位置直写，warp 私有槽）→ warp
        //    屏障 → A 读回 ──
        let pk = |lo: f32, hi: f32| -> u32 {
            (f32_to_bf16(lo) as u32) | ((f32_to_bf16(hi) as u32) << 16)
        };
        unsafe {
            *ppw.add((g << 3) + (t2 >> 1)) = pk(p0, p1);
            *ppw.add(((g + 8) << 3) + (t2 >> 1)) = pk(p2, p3);
            *ppw.add((g << 3) + ((t2 + 8) >> 1)) = pk(p4, p5);
            *ppw.add(((g + 8) << 3) + ((t2 + 8) >> 1)) = pk(p6, p7);
        }
        unsafe { bar_warp_sync(0xffff_ffff) };
        let (a0, a1, a2, a3) = unsafe { wmma_load_a(ppw as *const u32, 16) };
        // ── O += P@V^T：16 步 n（hd 256/16），B 片段自 smem 当前缓冲 ──
        macro_rules! o_mma {
            ($i:literal) => {
                let vb = unsafe { wmma_load_b(vtb.add($i << 7), 16) };
                o[$i] = unsafe {
                    wmma_mma(
                        a0, a1, a2, a3, vb.0, vb.1, vb.2, vb.3, o[$i].0, o[$i].1, o[$i].2,
                        o[$i].3, o[$i].4, o[$i].5, o[$i].6, o[$i].7,
                    )
                };
            };
        }
        v4_unroll16!(o_mma);

        m_g = mn_g;
        m_g8 = mn_g8;
        tile += 1;
    }

    // ── 尾写：O/l → bf16 散写 out（行归属：0/1/4/5→g，2/3/6/7→g+8）──
    let inv_g = 1.0 / l_g;
    let inv_g8 = 1.0 / l_g8;
    let (wg, wg8) = (rbase + g < last + 1, rbase + g + 8 < last + 1);
    macro_rules! o_write {
        ($i:literal) => {
            unsafe {
                if wg {
                    *outp.add(orow_g + ($i << 4) + t2) = f32_to_bf16(o[$i].0 * inv_g);
                    *outp.add(orow_g + ($i << 4) + t2 + 1) = f32_to_bf16(o[$i].1 * inv_g);
                    *outp.add(orow_g + ($i << 4) + t2 + 8) = f32_to_bf16(o[$i].4 * inv_g);
                    *outp.add(orow_g + ($i << 4) + t2 + 9) = f32_to_bf16(o[$i].5 * inv_g);
                }
                if wg8 {
                    *outp.add(orow_g8 + ($i << 4) + t2) = f32_to_bf16(o[$i].2 * inv_g8);
                    *outp.add(orow_g8 + ($i << 4) + t2 + 1) = f32_to_bf16(o[$i].3 * inv_g8);
                    *outp.add(orow_g8 + ($i << 4) + t2 + 8) = f32_to_bf16(o[$i].6 * inv_g8);
                    *outp.add(orow_g8 + ($i << 4) + t2 + 9) = f32_to_bf16(o[$i].7 * inv_g8);
                }
            }
        };
    }
    v4_unroll16!(o_write);
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

/// add + rms_norm_zc 融合（残差后 ln2 / final norm，两发并一发）。
/// 位级合同：sum = f32→bf16 舍入(f32(a)+f32(b)) 落 sum_out（与 add kernel
/// 逐位同）；归约与输出从 sum_out 读回 round 后的 bf16（两 kernel 分离
/// 路径 norm 从 bf16 缓冲读回同语义）——block_sum_strided 树、inv 与
/// 输出表达式逐字复刻 rms_norm_zc，块几何同（block-per-row）
#[no_mangle]
pub unsafe extern "ptx-kernel" fn add_norm_zc(
    a: *const u16,
    b: *const u16,
    w: *const u16,
    sum_out: *mut u16,
    norm_out: *mut u16,
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
    unsafe {
        let mut i = t;
        while i < h {
            *sum_out.add(base + i) =
                f32_to_bf16(bf16_to_f32(*a.add(base + i)) + bf16_to_f32(*b.add(base + i)));
            i += bdim;
        }
        // 写/读同线程同地址（同跨步占有）——无需屏障即读到自写值
        let total = block_sum_strided(sum_out, scratch, row, base, h, t, bdim);
        let inv = 1.0f32 / core::intrinsics::sqrt::<f32>(total / hidden as f32 + eps);
        let mut i = t;
        while i < h {
            let o = bf16_to_f32(*sum_out.add(base + i)) * inv * (1.0 + bf16_to_f32(*w.add(i)));
            *norm_out.add(base + i) = f32_to_bf16(o);
            i += bdim;
        }
    }
}

/// linear 层 conv 输出去交错（3×embed gather 并一发）：src [t, 2·kd+vd]
/// 每 token 交错 q|k|v，一发拆到 q [t,kd]、k [t,kd]、v [t,vd]——纯位
/// 搬运无算术（与按行 gather 逐位同）
#[no_mangle]
pub unsafe extern "ptx-kernel" fn dn_split_qkv(
    src: *const u16,
    q: *mut u16,
    k: *mut u16,
    v: *mut u16,
    k_dim: u32,
    v_dim: u32,
    tokens: u32,
) {
    let (kd, vd) = (k_dim as usize, v_dim as usize);
    let cd = 2 * kd + vd;
    let i = tid() as usize;
    if i >= tokens as usize * cd {
        return;
    }
    let (t, e) = (udiv(i, cd), urem(i, cd));
    let s = *src.add(i);
    if e < kd {
        *q.add(t * kd + e) = s;
    } else if e < 2 * kd {
        *k.add(t * kd + (e - kd)) = s;
    } else {
        *v.add(t * vd + (e - 2 * kd)) = s;
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

// ───────────── delta BT-chunked（WY/UT 分块递推；prefill 大 span 容差档） ─────────────
//
// 代数（与 FLA chunk.py/wy_fast.py 同构；标量衰减版——本模型 a 是 [tokens, v_heads]
// 逐 (token, 头) 标量，非逐 k 维向量，UT 变换因此更简）：
//
//   逐 token（kernel 现状，reference 同式）：S₁=S₀·λ；kvm=S₁k̂；δ=(v−kvm)·β；
//   S₂=S₁+k̂⊗δ；o=S₂·q̂/√kd。β=sigmoid(b)（收 bf16 同式）；
//   λ=exp(−X)，X=exp(a_log)·softplus(a+dt_bias)（≥0，逐 (token,头) 标量）。
//
//   块内（BT token，C_t = Σ_{s≤t}X_s 块内累计——e^{−C_t} 即逐 token 衰减积）：
//   ① A[t,s] = β_t·(k̂_t·k̂_s)·e^{C_s−C_t}（s<t 严格下三角）
//   ② ū = (I+A)⁻¹(β⊙v)          [BT,V]（状态无关）
//      w  = (I+A)⁻¹(β·e^{−C}⊙k̂) [BT,K]（状态无关）
//   ③ 块间串行：x = w·S₀ᵀ；δ = ū − x；S ← e^{−C_last}·S₀ + Σ_t e^{C_t−C_last}·k̂_t⊗δ_t
//   ④ 输出（全并行）：o_t = e^{−C_t}·(S₀·q̂_t) + Σ_{s≤t}(q̂_t·k̂_s)·e^{C_s−C_t}·δ_s，再 /√kd
//
//   对拍（/tmp 原型 vs 朴素递推，f32）：rel2 out 6.2e-7 / 末态 6.2e-8——代数精确；
//   与 reference 的差全在舍入路径（reference 逐 token 收 bf16 状态、本路 f32 链
//   + 块边界 bf16 快照），容差档 o35 钉预算。
//
// 三 kernel（host 侧 ops::DeltanetChunkedCuda 串三次 launch；前置 dn_qk_l2norm 复用）：
//   dn_chunk_prep：逐 (全局块, v 头) 并行——门控 scan（C/β + eneg/epos 端点缩放）、
//     rhs 直落 u/w（4 列 1×u64 载 2×u64 存）、A 下三角物化（行主线程：8 车道
//     共享 k_t 行、4-bf16 u64 载）、右看前代（tp 成对——A 相邻对 1×u64 载 +
//     8 车道 ILP；fma 应用序对每元素 p 升序 = 左看同位）；
//   dn_chunk_h：逐 (行, v 头, v 行片) 块间串行，2 屏障/块——δ=ū−w·S₀ᵀ
//     （车道共享 S 行、w 2-f32 u64 载，u 原位）、更新+快照融合（run = 固定 j ×
//     rl 连续 v 行：k̂·ds 合并标量每 tp 一次、δ 行 u64 载）；
//   dn_chunk_o：逐 (全局块, v 头) 全并行——Mᵀ 转置物化（列主线程共享 k_s 行；
//     转置使 ③ 的 M 读取变 8-t 连续 u64 载）、② base 与 ③ 累加融合成
//     (8t×4v) 瓦片（q 行/S₀ 行双边共享、δ 4-v u64 载、出 4-bf16 u64 存）。
//
// 热路形律（v2，"散乱行指针每 fma 一载"是 v1 慢 50× 的根因）：
// ① 序列传载一律 u64（4×bf16 / 2×f32）——每 fma 载指令 ≤0.5；
// ② 点积操作数在车道间共享（行主/列主/瓦片双边）；
// ③ expf 全析出循环：e^{C_s−C_t} = eneg_t·epos_s·E（E=e^{C_last}；eneg/epos
//   各 ≤1 恒安全，E 在 C_last ≤ 60 档内 f32 安全，超档逐对 expf 兜底）；
// ④ 独立链 8-16 路藏 L2 延迟。v2 不保证逐位（本路容差档，o35 钉）。
//
// 约束同仓库既有：无 smem/shfl，块间通信走块私有 global scratch + _syncthreads；
// 禁 asm!；f32 累加、fma 允许。

/// prep 的块私有 scratch 布局（f32，5·BT 槽）：
///   [0, bt) scan 缓冲 A｜[bt, 2bt) scan 缓冲 B｜[2bt, 3bt) β
///   [3bt, 4bt) eneg[t]=e^{−C_t}｜[4bt, 5bt) epos[t]=e^{C_t−C_last}
/// （o kernel 经同槽位只读 eneg/epos——衰减析出的端点缩放）
/// A 矩阵走独立的 `ascratch`（[块, BT·BT] f32——host 以 snap 缓冲别名传入：
/// prep 先于 h 运行且不读 snap，生命周期零交叠）

// 诊断档：0=全跑（生产）；>0 为临时分相排除法（见 git 历史/探针注释）

#[no_mangle]
pub unsafe extern "ptx-kernel" fn dn_chunk_prep(
    k: *const u16,      // [tokens, k_heads, k_dim]
    v: *const u16,      // [tokens, v_heads, v_dim]
    a: *const u16,      // [tokens, v_heads]
    b: *const u16,      // [tokens, v_heads]
    a_log: *const f32,  // [v_heads]
    dt_bias: *const u16, // [v_heads]
    inv: *const f32,    // [tokens*k_heads, 2]（dn_qk_l2norm 前置产物，只读）
    gcum: *mut f32,     // [tokens, v_heads]（块内累计 C，出）
    u: *mut f32,        // [tokens, v_heads, v_dim]（(I+A)⁻¹(βv)，出）
    w: *mut f32,        // [tokens, v_heads, k_dim]（(I+A)⁻¹(βe^{−C}k̂)，出）
    row_start: *const u32, // [rows]
    row_count: *const u32, // [rows]
    snap_off: *const u32,  // [rows+1]（逐行块数前缀和）
    n_rows: u32,
    chunks: u32, // 全局块总数 = snap_off[n_rows]
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    bt: u32, // 块 token 数（32/64/128）
    ascratch: *mut f32, // [块, BT·BT]——A 下三角（行主 [t][s]）
    scratch: *mut f32,  // [块, 5·BT]——scan/β/eneg/epos
) {
    let (kd, vd) = (k_dim as usize, v_dim as usize);
    let (kh_n, vh) = (k_heads as usize, v_heads as usize);
    let bts = bt as usize;
    // A 的 Gram 核 dot(k_t,k_s) 只依赖 k 头——v 头专属因子（β/invk/eneg/epos）
    // 全部在端点尾乘（原 dn_ast! 即此形）⇒ 一块 = (块, k 头)，跨 vph = vh/kh_n
    // 个 v 头共用一份 Gram，逐位等价（同一 dot 结果同一尾乘序）
    let vph = udiv(vh, kh_n);
    let bid = _block_idx_x() as usize;
    if bid >= chunks as usize * kh_n {
        return;
    }
    let gc = udiv(bid, kh_n);
    let kk = urem(bid, kh_n);
    // 行定位：snap_off 前缀上线性扫（行数 ≤ 百级，全线程冗余扫）
    let mut r = 0usize;
    while r + 1 < n_rows as usize && gc >= *snap_off.add(r + 1) as usize {
        r += 1;
    }
    let c = gc - *snap_off.add(r) as usize;
    let st = *row_start.add(r) as usize;
    let cnt = *row_count.add(r) as usize;
    let t0 = st + c * bts;
    let n = if cnt - c * bts < bts { cnt - c * bts } else { bts };
    let kh = kk;
    let krow = |g: usize, j: usize| (g * kh_n + kh) * kd + j; // q/k 的 (token, k 头) 行内偏移
    let t = _thread_idx_x() as usize;
    // ── 逐 v 头：门控 scan + gcum/neg/pos + rhs 直落 u/w ──
    let mut hv = 0usize;
    while hv < vph {
        let h = kk * vph + hv;
        let vb = gc * vh + h; // 旧 (块,v 头) 槽位语义（scratch/ascratch、o/h 读法不变）
        let big_a = expf(*a_log.add(h));
        let dtb = bf16_to_f32(*dt_bias.add(h));
        let sc = scratch.add(vb * (5 * bts));
        let (buf0, buf1, bet, neg, pos) = (
            sc,
            sc.add(bts),
            sc.add(2 * bts),
            sc.add(3 * bts),
            sc.add(4 * bts),
        );
        // 门控：X = −ln λ（λ=exp(−X) 免对数）与 β（sigmoid 收 bf16——同 reference 两跳）
        if t < n {
            let g = t0 + t;
            *buf0.add(t) = big_a * softplus(bf16_to_f32(*a.add(g * vh + h)) + dtb);
            *bet.add(t) = bf16_to_f32(f32_to_bf16(
                1.0f32 / (1.0f32 + expf(0.0 - bf16_to_f32(*b.add(g * vh + h)))),
            ));
        }
        _syncthreads();
        // 包含式 scan（Hillis-Steele 双缓冲；n 外槽位垃圾不被读）
        let (mut src, mut dst) = (buf0, buf1);
        let mut off = 1usize;
        while off < bts {
            _syncthreads();
            if t < n {
                let acc = if t >= off { *src.add(t) + *src.add(t - off) } else { *src.add(t) };
                *dst.add(t) = acc;
            }
            let tmp = src;
            src = dst;
            dst = tmp;
            off <<= 1;
        }
        _syncthreads();
        // C 落 gcum（o/h/A 尾乘只读）+ 端点缩放（neg/epos 各 ≤1 恒安全；E=e^{C_last}
        // 由用点自算 1 次 expf，C_last ≤ 60 档内 f32 安全）
        let clast = *src.add(n - 1);
        if t < n {
            *gcum.add((t0 + t) * vh + h) = *src.add(t);
            *neg.add(t) = expf(0.0 - *src.add(t));
            *pos.add(t) = expf(*src.add(t) - clast);
        }
        _syncthreads();
        // rhs 直落 u/w：任务 = (行 tp, 4 列 run)（64 run/行——vd=kd=128 钉死）。
        // 4 列 1×u64 载、2×u64 存；scale（β / β·eneg·invk）每 run 一次
        {
            let mut e = t;
            while e < n * 64 {
                let tp = udiv(e, 64);
                let run = urem(e, 64);
                let cc = run * 4;
                let (w4, s_, p) = if run < 32 {
                    (
                        *(v.add((t0 + tp) * vh * vd + h * vd + cc) as *const u64),
                        *bet.add(tp),
                        u.add((t0 + tp) * vh * vd + h * vd + cc) as *mut u64,
                    )
                } else {
                    (
                        *(k.add(krow(t0 + tp, cc - 128)) as *const u64),
                        *bet.add(tp)
                            * *neg.add(tp)
                            * *inv.add((t0 + tp) * kh_n * 2 + kh * 2 + 1),
                        w.add((t0 + tp) * vh * kd + h * kd + (cc - 128)) as *mut u64,
                    )
                };
                let (lo, hi) = (w4 as u32, (w4 >> 32) as u32);
                let f0 = f32::from_bits(lo << 16) * s_;
                let f1 = f32::from_bits(lo & 0xffff_0000) * s_;
                let f2 = f32::from_bits(hi << 16) * s_;
                let f3 = f32::from_bits(hi & 0xffff_0000) * s_;
                *p = (f0.to_bits() as u64) | ((f1.to_bits() as u64) << 32);
                *(p.add(1)) = (f2.to_bits() as u64) | ((f3.to_bits() as u64) << 32);
                e += 256;
            }
        }
        hv += 1;
    }
    // ── A 下三角物化：任务 = (t 八元 × s 八元) 8×8 寄存器瓦片（绿版结构；
    //    4×4 改写实测累加器散点污染——散点 e37/清零，未定位即弃）——每 j4
    //    步 k 行 8+8×u64（L1 驻）→ 256 fma/16 载。Gram（k 头份）算一次，
    //    端点缩放（β/invk/衰减）尾乘逐 v 头落位（fastf 档 A[t][s] =
    //    (β_t·invk_t·eneg_t)·(invk_s·epos_s·E)·(k_t·k_s)；兜底档 e^{C_s−C_t}
    //    逐对 expf，C 改读 gcum 全局——与 scan 缓冲等值）──
    if true {
        let ntg = udiv(n + 7, 8);
        let ntasks = ntg * ntg;
        let mut e = t;
        while e < ntasks {
            let tr0 = udiv(e, ntg) * 8;
            let sr0 = urem(e, ntg) * 8;
            if sr0 <= tr0 + 7 {
                let (mut a0, mut a1, mut a2, mut a3) = ([0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8]);
                let (mut a4, mut a5, mut a6, mut a7) = ([0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8]);
                let mut kt = [0u64; 8];
                let mut ks = [0u64; 8];
                macro_rules! dn_arow {
                    ($acc:ident, $r:expr) => {{
                        let (tlo, thi) = (kt[$r] as u32, (kt[$r] >> 32) as u32);
                        let ta = f32::from_bits(tlo << 16);
                        let tb = f32::from_bits(tlo & 0xffff_0000);
                        let tc = f32::from_bits(thi << 16);
                        let td = f32::from_bits(thi & 0xffff_0000);
                        for c in 0..8 {
                            let (slo, shi) = (ks[c] as u32, (ks[c] >> 32) as u32);
                            $acc[c] = fma(ta, f32::from_bits(slo << 16), $acc[c]);
                            $acc[c] = fma(tb, f32::from_bits(slo & 0xffff_0000), $acc[c]);
                            $acc[c] = fma(tc, f32::from_bits(shi << 16), $acc[c]);
                            $acc[c] = fma(td, f32::from_bits(shi & 0xffff_0000), $acc[c]);
                        }
                    }};
                }
                let mut j4 = 0usize;
                while j4 < 32 {
                    let jb = j4 * 4;
                    for r in 0..8 {
                        if tr0 + r < n {
                            kt[r] = *(k.add(krow(t0 + tr0 + r, jb)) as *const u64);
                        }
                    }
                    for c in 0..8 {
                        if sr0 + c < n {
                            ks[c] = *(k.add(krow(t0 + sr0 + c, jb)) as *const u64);
                        }
                    }
                    dn_arow!(a0, 0);
                    dn_arow!(a1, 1);
                    dn_arow!(a2, 2);
                    dn_arow!(a3, 3);
                    dn_arow!(a4, 4);
                    dn_arow!(a5, 5);
                    dn_arow!(a6, 6);
                    dn_arow!(a7, 7);
                    j4 += 1;
                }
                // 逐 v 头尾乘落位（端点缩放尾乘；s<t 掩码；tr<n 尾块守卫）
                let mut hv = 0usize;
                while hv < vph {
                    let h = kk * vph + hv;
                    let vb = gc * vh + h;
                    let sc = scratch.add(vb * (5 * bts));
                    let (bet, neg, pos) = (sc.add(2 * bts), sc.add(3 * bts), sc.add(4 * bts));
                    let abase = ascratch.add(vb * bts * bts);
                    let clast = *gcum.add((t0 + n - 1) * vh + h);
                    let fastf = clast <= 60.0f32;
                    let bige = if fastf { expf(clast) } else { 1.0f32 };
                    macro_rules! dn_ast {
                        ($ar:ident, $r:expr) => {{
                            let tr = tr0 + $r;
                            if tr < n {
                                let g0 = *bet.add(tr)
                                    * *inv.add((t0 + tr) * kh_n * 2 + kh * 2 + 1)
                                    * if fastf { *neg.add(tr) } else { 1.0f32 };
                                for half in 0..4 {
                                    let sa = sr0 + half * 2;
                                    let sb = sr0 + half * 2 + 1;
                                    let va = if sa < tr {
                                        let hsa = *inv.add((t0 + sa) * kh_n * 2 + kh * 2 + 1)
                                            * if fastf {
                                                *pos.add(sa) * bige
                                            } else {
                                                expf(
                                                    *gcum.add((t0 + sa) * vh + h)
                                                        - *gcum.add((t0 + tr) * vh + h),
                                                )
                                            };
                                        g0 * hsa * $ar[half * 2]
                                    } else {
                                        0.0f32
                                    };
                                    let vb2 = if sb < tr {
                                        let hsb = *inv.add((t0 + sb) * kh_n * 2 + kh * 2 + 1)
                                            * if fastf {
                                                *pos.add(sb) * bige
                                            } else {
                                                expf(
                                                    *gcum.add((t0 + sb) * vh + h)
                                                        - *gcum.add((t0 + tr) * vh + h),
                                                )
                                            };
                                        g0 * hsb * $ar[half * 2 + 1]
                                    } else {
                                        0.0f32
                                    };
                                    *(abase.add(tr * bts + sa) as *mut u64) =
                                        (va.to_bits() as u64) | ((vb2.to_bits() as u64) << 32);
                                }
                            }
                        }};
                    }
                    dn_ast!(a0, 0);
                    dn_ast!(a1, 1);
                    dn_ast!(a2, 2);
                    dn_ast!(a3, 3);
                    dn_ast!(a4, 4);
                    dn_ast!(a5, 5);
                    dn_ast!(a6, 6);
                    dn_ast!(a7, 7);
                    hv += 1;
                }
            }
            e += 256;
        }
    }
    _syncthreads(); // A 物化完成（跨线程）后才进 solve
    // ── 左看前代：256 线程列并行（t<128 → u 列 t；t≥128 → w 列 t−128；
    //    vd=kd=128 钉死）。每线程独占一列串行走完 n 行：X[t][c] = R[t][c] −
    //    Σ_{p<t} A[t][p]·X[p][c]（p 升序 fma——与旧 8 宽瓦片版逐元素同序）。
    //    列独占 ⇒ 零跨线程依赖零屏障；warp 内列连续 → 每行 1 条 128B 合并
    //    载；逐 v 头串行各解一份（A/u/w/ascratch 均按 (块,v 头) 槽位）
    if true {
        let mut hv = 0usize;
        while hv < vph {
            let h = kk * vph + hv;
            let abase = ascratch.add((gc * vh + h) * bts * bts);
            let isu = t < 128;
            let c = if isu { t } else { t - 128 };
            let (base, pitch) = if isu {
                (u.add(t0 * vh * vd + h * vd + c), vh * vd)
            } else {
                (w.add(t0 * vh * kd + h * kd + c), vh * kd)
            };
            let mut ti = 0usize;
            while ti < n {
                let mut x = *base.add(ti * pitch);
                let arow = abase.add(ti * bts);
                // p 链 4× 展开：X 行 4 载 + A 行 2×u64（warp 同址广播）先行批量
                // 发出再跑 4 fma，藏 L1 延迟（4 路部分和树实测更慢——批量载齐
                // 才放 fma，反而失去 load/fma 交错；且序变无精度收益）
                let mut p = 0usize;
                let p4 = ti & !3usize;
                while p < p4 {
                    let (a01, a23) = (
                        *(arow.add(p) as *const u64),
                        *(arow.add(p + 2) as *const u64),
                    );
                    let x0 = *base.add(p * pitch);
                    let x1 = *base.add((p + 1) * pitch);
                    let x2 = *base.add((p + 2) * pitch);
                    let x3 = *base.add((p + 3) * pitch);
                    x = fma(0.0 - f32::from_bits(a01 as u32), x0, x);
                    x = fma(0.0 - f32::from_bits((a01 >> 32) as u32), x1, x);
                    x = fma(0.0 - f32::from_bits(a23 as u32), x2, x);
                    x = fma(0.0 - f32::from_bits((a23 >> 32) as u32), x3, x);
                    p += 4;
                }
                while p < ti {
                    let a = *arow.add(p);
                    x = fma(0.0 - a, *base.add(p * pitch), x);
                    p += 1;
                }
                *base.add(ti * pitch) = x;
                ti += 1;
            }
            hv += 1;
        }
    }
}

/// dn_chunk_h v3 的逐块上下文。
struct DnH {
    k: *const u16,
    inv: *const f32,
    gcum: *const f32,
    u: *mut f32,       // in: ū；out: δ″ = invk·δ（o 读 δ″ 配 M[t][s]=invq·(q·k)·e^{C_s−C_t}）
    w: *const f32,
    s_mat: *mut f32,   // [vrows, kd] f32 状态片（① 读；② 每 chunk 末写回）
    pool0: *mut u16,   // 状态池本片基址（初态 in / 末态 out）
    snap0: *mut u16,   // 本片 chunk0 快照基址
    kh_n: usize,
    vh: usize,
    kd: usize,
    vd: usize,
    bts: usize,
    vrows: usize,
    vbase: usize,
    kh: usize,
    h: usize,
    st: usize,
    cnt: usize,
    nch: usize,
    t: usize,
}

/// u64 → 4×f32（bf16 高位展开——纯移位/掩码，无查表）
#[inline(always)]
fn dn_dec4(q: u64) -> [f32; 4] {
    let (lo, hi) = (q as u32, (q >> 32) as u32);
    [
        f32::from_bits(lo << 16),
        f32::from_bits(lo & 0xffff_0000),
        f32::from_bits(hi << 16),
        f32::from_bits(hi & 0xffff_0000),
    ]
}

/// 4×f32 → u64（bf16 打包）
#[inline(always)]
fn dn_pack4(v: [f32; 4]) -> u64 {
    (f32_to_bf16(v[0]) as u64)
        | ((f32_to_bf16(v[1]) as u64) << 16)
        | ((f32_to_bf16(v[2]) as u64) << 32)
        | ((f32_to_bf16(v[3]) as u64) << 48)
}

/// chunk 主循环 v3（VW = 每 warp 的 v 行数 = vrows/8）：
///   ① δ′ = m·(ū − w·S₀ᵀ)：任务 = (t 四元 × v 四元 × j 半程)——jh=0 把
///      ū−p₀ 原位写回 u；jh=1 的 p₁ 驻寄存器过屏障再 RMW + m 尾乘。
///      每 j 步 16 条 v4 载 / 128 fma（w/S 均 f32，零解码）。
///   ② S = e^{−C_last}·S + Σ_t k̂_t⊗δ′_t：warp-per-row——warp w 吃 v 行段
///      [w·VW, +VW)，lane l 持 j 片 [l·4, +4)。k̂ 1×u64（warp 内合并 256B）+
///      δ′ VW/4×F4（全 lane 同址广播）→ VW·4 fma / (1+VW/4) 载 + 8 解码；
///      状态 VW×4 驻寄存器跨 chunk，块末写 s_mat（① 用）+ 快照。
#[inline(always)]
unsafe fn dn_h_run3<const VW: usize>(d: &DnH) {
    let (kd, vd, vh, kh_n) = (d.kd, d.vd, d.vh, d.kh_n);
    let (h, kh, vbase) = (d.h, d.kh, d.vbase);
    let (t0row, cnt, bts) = (d.st, d.cnt, d.bts);
    let t = d.t;
    let warp = t >> 5;
    let lane = t & 31;
    let vb0 = warp * VW; // 本 warp 首行（块内局部）
    let j4 = lane * 4; //   本 lane j 片基
    let vrows = d.vrows;
    let nvq = vrows >> 2; // v 四元组数（vrows ≥ 16 ⇒ ≥ 4）
    let bdim = _block_dim_x() as usize;
    // ── S₀ → 寄存器 + s_mat + chunk0 快照（一趟；vrows<VW·8 时尾部 warp 空转）──
    let mut acc = [[0f32; 4]; VW];
    if vb0 < vrows {
        for r in 0..VW {
            let e = (vb0 + r) * kd + j4;
            let v = dn_dec4(*(d.pool0.add(e) as *const u64));
            acc[r] = v;
            *(d.s_mat.add(e) as *mut F4) = F4(v);
            *(d.snap0.add(e) as *mut u64) = dn_pack4(v);
        }
    }
    _syncthreads(); // s_mat 全片就绪（① 任务跨 warp 读）
    let mut ci = 0usize;
    while ci < d.nch {
        let t0 = t0row + ci * bts;
        let n = if cnt - ci * bts < bts { cnt - ci * bts } else { bts };
        let clast = *d.gcum.add((t0 + n - 1) * vh + h);
        // g_t = e^{C_t−C_last}：② 状态累加用的块内权重。每 chunk 每线程 ≤1 次
        // expf 落 scratch 尾槽（"旧 ds"尾巴复用），② warp 同址广播读——替代
        // ② 每 tp 每 lane 一次 expf（实测 +1.1ms/层）。欠流为 0 在此是正确语义
        // （旧 token 对块末态贡献≈0）。
        let mut gt = t;
        while gt < bts {
            if gt < n {
                *d.s_mat.add(vrows * kd + gt) = expf(*d.gcum.add((t0 + gt) * vh + h) - clast);
            }
            gt += bdim;
        }
        // ── ① δ″ = invk·(ū − w·S₀ᵀ)：任务 = (t **对** × v 四元)——2×4 瓦片
        //    全 j。2×4 而非 4×4：bt=64/vsplit=4 下任务数 = 32·8 = 256 = 线程数
        //    （4×4 只有 128 ⇒ 半线程空转）；vsplit=8 恰 128。每线程 1024 fma /
        //    192 F4 载，零跨 lane 归约零屏障。invk 尾乘（无 g 因子：② 在
        //    寄存器内补乘 e^{C_t−C_last}）──
        let ntq = bts >> 1;
        if true {
            let wk = d.vh * d.kd;
            let mut e = t;
            while e < ntq * nvq {
                let tq = udiv(e, nvq);
                let vq = e - tq * nvq;
                let tr0 = tq * 2;
                // 行钳位（末块尾：越界行读回块首行，tr<n 守卫丢结果）
                let x0 = if tr0 < n { t0 + tr0 } else { t0 };
                let x1 = if tr0 + 1 < n { t0 + tr0 + 1 } else { t0 };
                let pw0 = d.w.add(x0 * wk + d.h * d.kd);
                let pw1 = d.w.add(x1 * wk + d.h * d.kd);
                let sb = d.s_mat.add(vq * 4 * kd);
                let mut aa = [[0f32; 4]; 2]; // aa[rr][c] = dot(w[x_rr], S₀[vq*4+c])
                let mut jb = 0usize;
                while jb < kd {
                    let s00 = (*(sb.add(jb) as *const F4)).0;
                    let s01 = (*(sb.add(jb + 4) as *const F4)).0;
                    let s10 = (*(sb.add(jb + kd) as *const F4)).0;
                    let s11 = (*(sb.add(jb + kd + 4) as *const F4)).0;
                    let s20 = (*(sb.add(jb + 2 * kd) as *const F4)).0;
                    let s21 = (*(sb.add(jb + 2 * kd + 4) as *const F4)).0;
                    let s30 = (*(sb.add(jb + 3 * kd) as *const F4)).0;
                    let s31 = (*(sb.add(jb + 3 * kd + 4) as *const F4)).0;
                    macro_rules! dn_p1c {
                        ($rr:literal, $cc:literal, $w0:ident, $w1:ident, $sa:ident, $sz:ident) => {{
                            let a = fma($w0[0], $sa[0], aa[$rr][$cc]);
                            let a = fma($w0[1], $sa[1], a);
                            let a = fma($w0[2], $sa[2], a);
                            let a = fma($w0[3], $sa[3], a);
                            let a = fma($w1[0], $sz[0], a);
                            let a = fma($w1[1], $sz[1], a);
                            let a = fma($w1[2], $sz[2], a);
                            let a = fma($w1[3], $sz[3], a);
                            aa[$rr][$cc] = a;
                        }};
                    }
                    macro_rules! dn_p1w {
                        ($pw:expr, $rr:literal) => {{
                            let wv0 = (*($pw.add(jb) as *const F4)).0;
                            let wv1 = (*($pw.add(jb + 4) as *const F4)).0;
                            dn_p1c!($rr, 0, wv0, wv1, s00, s01);
                            dn_p1c!($rr, 1, wv0, wv1, s10, s11);
                            dn_p1c!($rr, 2, wv0, wv1, s20, s21);
                            dn_p1c!($rr, 3, wv0, wv1, s30, s31);
                        }};
                    }
                    dn_p1w!(pw0, 0);
                    dn_p1w!(pw1, 1);
                    jb += 8;
                }
                for rr in 0..2usize {
                    let tr = tr0 + rr;
                    if tr < n {
                        let m = *d.inv.add((t0 + tr) * kh_n * 2 + kh * 2 + 1);
                        let p = d.u.add((t0 + tr) * vh * vd + h * vd + vbase + vq * 4);
                        let old = (*(p as *const F4)).0;
                        *(p as *mut F4) = F4([
                            m * (old[0] - aa[rr][0]),
                            m * (old[1] - aa[rr][1]),
                            m * (old[2] - aa[rr][2]),
                            m * (old[3] - aa[rr][3]),
                        ]);
                    }
                }
                e += bdim;
            }
        }
        _syncthreads();
        // ── ② 衰减 + 秩 n 更新 + 快照：warp-per-row ──
        let dlast = expf(0.0 - clast);
        if vb0 < vrows {
            for r in 0..VW {
                for j in 0..4usize {
                    acc[r][j] *= dlast;
                }
            }
            let mut tp = 0usize;
            // δ″ F4 载（VW/4 条；vb0 恒 4 倍 ⇒ 16B 对齐）+ 2 深软件流水 + 2×
            // token 展开：A/B 双槽轮转，当拍 fma 与 tp+2 预取并发——1 深时
            // 6016 步链每步裸吃整条载入延迟（~400cy），是 ② 的主导项。
            // rq 字面展开（d_a/d_b/acc 全寄存器）；rq ≥ VW/4 有 vrows 守卫
            let krow = d.k.add(t0 * kh_n * kd + kh * kd + j4);
            let urow = d.u.add(t0 * vh * vd + h * vd + vbase + vb0);
            let gs = d.s_mat.add(vrows * kd);
            let nq = VW / 4;
            macro_rules! dn_h2ld {
                ($arr:ident, $rq:literal, $ub:expr) => {{
                    if $rq < nq {
                        $arr[$rq] = *($ub.add($rq * 4) as *const F4);
                    }
                }};
            }
            macro_rules! dn_h2pf {
                ($ka:ident, $ga:ident, $da:ident, $tt:expr) => {{
                    $ka = *(krow.add($tt * kh_n * kd) as *const u64);
                    $ga = *gs.add($tt);
                    let un = urow.add($tt * vh * vd);
                    dn_h2ld!($da, 0, un);
                    dn_h2ld!($da, 1, un);
                    dn_h2ld!($da, 2, un);
                    dn_h2ld!($da, 3, un);
                }};
            }
            macro_rules! dn_h2q {
                ($rq:literal, $kq:ident, $g:ident, $dv:ident) => {{
                    if $rq < nq {
                        for rr in 0..4usize {
                            let dvs = $dv[rr] * $g;
                            for j in 0..4usize {
                                acc[$rq * 4 + rr][j] = fma($kq[j], dvs, acc[$rq * 4 + rr][j]);
                            }
                        }
                    }
                }};
            }
            let (mut k_a, mut g_a) = (*(krow as *const u64), *gs);
            let (mut k_b, mut g_b) = (0u64, 0f32);
            let (mut k_c, mut g_c) = (0u64, 0f32);
            let (mut k_d, mut g_d) = (0u64, 0f32);
            let mut d_a = [F4([0f32; 4]), F4([0f32; 4]), F4([0f32; 4]), F4([0f32; 4])];
            let mut d_b = [F4([0f32; 4]), F4([0f32; 4]), F4([0f32; 4]), F4([0f32; 4])];
            let mut d_c = [F4([0f32; 4]), F4([0f32; 4]), F4([0f32; 4]), F4([0f32; 4])];
            let mut d_d = [F4([0f32; 4]), F4([0f32; 4]), F4([0f32; 4]), F4([0f32; 4])];
            dn_h2ld!(d_a, 0, urow);
            dn_h2ld!(d_a, 1, urow);
            dn_h2ld!(d_a, 2, urow);
            dn_h2ld!(d_a, 3, urow);
            if n > 1 {
                dn_h2pf!(k_b, g_b, d_b, 1);
            }
            if n > 2 {
                dn_h2pf!(k_c, g_c, d_c, 2);
            }
            if n > 3 {
                dn_h2pf!(k_d, g_d, d_d, 3);
            }
            macro_rules! dn_h2tick {
                ($kq:ident, $g:ident, $ka:ident, $ga:ident, $da:ident) => {{
                    let kq = dn_dec4($kq);
                    let g = $g;
                    let d0 = $da[0].0;
                    let d1 = $da[1].0;
                    let d2 = $da[2].0;
                    let d3 = $da[3].0;
                    if tp + 4 < n {
                        dn_h2pf!($ka, $ga, $da, tp + 4);
                    }
                    dn_h2q!(0, kq, g, d0);
                    dn_h2q!(1, kq, g, d1);
                    dn_h2q!(2, kq, g, d2);
                    dn_h2q!(3, kq, g, d3);
                    tp += 1;
                }};
            }
            while tp < n {
                // A/B/C/D 四槽轮转：4 深预取，每拍快照后再补 tp+4
                dn_h2tick!(k_a, g_a, k_a, g_a, d_a);
                if tp >= n {
                    break;
                }
                dn_h2tick!(k_b, g_b, k_b, g_b, d_b);
                if tp >= n {
                    break;
                }
                dn_h2tick!(k_c, g_c, k_c, g_c, d_c);
                if tp >= n {
                    break;
                }
                dn_h2tick!(k_d, g_d, k_d, g_d, d_d);
            }
            for r in 0..VW {
                let e0 = (vb0 + r) * kd + j4;
                *(d.s_mat.add(e0) as *mut F4) = F4(acc[r]);
                if ci + 1 < d.nch {
                    *(d.snap0.add((ci + 1) * vh * vd * kd + e0) as *mut u64) = dn_pack4(acc[r]);
                }
            }
        }
        _syncthreads();
        ci += 1;
    }
    // 末态回池（bf16）
    if vb0 < vrows {
        for r in 0..VW {
            let e = (vb0 + r) * kd + j4;
            *(d.pool0.add(e) as *mut u64) = dn_pack4(acc[r]);
        }
    }
}

/// 块间串行状态推进（fwd_h）v3：一块 = (行, v 头, v 行片)，bd=256（8 warp）。
/// 状态 VW×4/线程驻寄存器；① 半程点积 + ② warp-per-row 更新，每 chunk
/// 3 屏障。δ″ = invk·δ 原位落 u（o 以逐对指数 M[t][s]=invq·(q·k)·e^{C_s−C_t} 配对读）。
/// 末态回状态池（bf16）；快照到 nchunks−1 块首 + chunk 0 初态。
#[no_mangle]
pub unsafe extern "ptx-kernel" fn dn_chunk_h(
    k: *const u16,     // [tokens, k_heads, k_dim]
    inv: *const f32,   // [tokens*k_heads, 2]
    gcum: *const f32,  // [tokens, v_heads]
    u: *mut f32,       // [tokens, v_heads, v_dim]（in: ū；out: δ′）
    w: *const f32,     // [tokens, v_heads, k_dim]
    snap: *mut u16,    // [chunks, v_heads, v_dim, k_dim] bf16（块首状态）
    state: *mut u16,   // [rows, v_heads, v_dim, k_dim] bf16（初态 in / 末态 out）
    row_start: *const u32,
    row_count: *const u32,
    row_id: *const u32,
    snap_off: *const u32,
    n_rows: u32,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    bt: u32,
    vsplit: u32, // v 行切分（1/2/4/8）
    scratch: *mut f32, // [块, vrows·kd]（s_mat；旧 ds 尾巴弃用）
) {
    let (kd, vd) = (k_dim as usize, v_dim as usize);
    let (kh_n, vh) = (k_heads as usize, v_heads as usize);
    let bts = bt as usize;
    let nsp = if vsplit == 0 { 1 } else { vsplit as usize };
    let bid = _block_idx_x() as usize;
    if bid >= n_rows as usize * vh * nsp {
        return;
    }
    let sp = urem(bid, nsp);
    let rest = udiv(bid, nsp);
    let (r, h) = (udiv(rest, vh), urem(rest, vh));
    let vrows = udiv(vd, nsp);
    let vbase = sp * vrows;
    let kh = udiv(h * kh_n, vh);
    let st = *row_start.add(r) as usize;
    let cnt = *row_count.add(r) as usize;
    let nch = udiv(cnt + bts - 1, bts);
    let t = _thread_idx_x() as usize;
    let sc = scratch.add(bid * (vrows * kd + bts + 8192)); // +① 两相部分和区（2任务×16输出×32lane×nwarp=8192 f32）
    let pool0 = state.add(((*row_id.add(r) as usize) * vh + h) * vd * kd + vbase * kd);
    let snap0 = snap.add((((*snap_off.add(r) as usize) * vh + h) * vd * kd) + vbase * kd);
    let d = DnH {
        k,
        inv,
        gcum,
        u,
        w,
        s_mat: sc,
        pool0,
        snap0,
        kh_n,
        vh,
        kd,
        vd,
        bts,
        vrows,
        vbase,
        kh,
        h,
        st,
        cnt,
        nch,
        t,
    };
    // VW 派发按 warp 数覆盖行数（bdim=64：2 warp × VW8=16 行恰满 vs=8 片；
    // 128/256 时与旧阈值等价）。VW ≥4 保 ② 的 F4 v 列 16B 对齐
    let warps = (_block_dim_x() as usize) >> 5;
    let need = udiv(vrows + warps - 1, warps);
    if need <= 4 || vrows < 16 {
        dn_h_run3::<4>(&d);
    } else if need <= 8 {
        dn_h_run3::<8>(&d);
    } else if need <= 16 {
        dn_h_run3::<16>(&d);
    } else {
        dn_h_run3::<32>(&d);
    }
}

/// 输出组装（chunk_fwd_o 同构）：一块 = (全局块, v 头)，全并行——
/// ① Mᵀ 转置物化（mscratch = w 缓冲别名；列主线程：8 车道 t 共享 k_s 行；
///   转置使 ③ 的 M 读取变 8-t 连续 u64 载，免散乱列读）；
/// ② base 与 ③ 累加融合：(8t × 4v) 瓦片 × j 半程——q 行/S₀ 行双边共享，
///   δ 4-v u64 载，出 4-bf16 u64 存。行缩放（e^{−C_t}·invq_t）析出为
///   base 尾乘（neg 槽位读 prep 的 scratch）。
#[no_mangle]
pub unsafe extern "ptx-kernel" fn dn_chunk_o(
    q: *const u16,    // [tokens, k_heads, k_dim]
    k: *const u16,    // [tokens, k_heads, k_dim]
    inv: *const f32,  // [tokens*k_heads, 2]
    gcum: *const f32, // [tokens, v_heads]
    dn: *const f32,   // [tokens, v_heads, v_dim]（δ——u 缓冲，fwd_h 原位后）
    snap: *const u16, // [chunks, v_heads, v_dim, k_dim] bf16
    core_out: *mut u16, // [tokens, v_heads*v_dim]
    row_start: *const u32,
    row_count: *const u32,
    snap_off: *const u32,
    n_rows: u32,
    chunks: u32,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    bt: u32,
    mscratch: *mut f32, // [块, BT·BT]——Mᵀ（列主 [s][t]，别名 w 缓冲）
    scratch: *const f32, // prep 的块私有 scratch（只读 neg/pos 槽）
) {
    let (kd, vd) = (k_dim as usize, v_dim as usize);
    let (kh_n, vh) = (k_heads as usize, v_heads as usize);
    let bts = bt as usize;
    let bid = _block_idx_x() as usize;
    if bid >= chunks as usize * kh_n {
        return;
    }
    let gc = udiv(bid, kh_n);
    let kk = urem(bid, kh_n); // 本块 k 头：① 的 q·k Gram 按 k 头算一份共用
    let vph = udiv(vh, kh_n); // 每份 Gram 服务的 v 头数（host 断言整除）
    let mut r = 0usize;
    while r + 1 < n_rows as usize && gc >= *snap_off.add(r + 1) as usize {
        r += 1;
    }
    let c = gc - *snap_off.add(r) as usize;
    let st = *row_start.add(r) as usize;
    let cnt = *row_count.add(r) as usize;
    let t0 = st + c * bts;
    let n = if cnt - c * bts < bts { cnt - c * bts } else { bts };
    let kh = kk; // h = kk·vph + hv ⇒ ⌊h·kh_n/vh⌋ = kk（整除时严格）
    let t = _thread_idx_x() as usize;
    let kd_sqrt = core::intrinsics::sqrt::<f32>(kd as f32);
    let qrow = |g: usize, j: usize| (g * kh_n + kh) * kd + j;
    // ── ① Mᵀ：任务 = (t 八元 × s 八元) 8×8 寄存器瓦片（同 A 形）——每 j4 步
    //    k_s 行 8×u64 + q 行 8×u64 → 256 fma/16 载；存转置 Mᵀ[s][t]（行连续），
    //    s ≤ t 含对角（对角 e^0 由析出式自然给出），M[t][s] =
    //    (invq_t·eneg_t)·(invk_s·epos_s·E)·(q_t·k_s)；兜底档逐对 expf ──
    if true {
        let ntg = udiv(n + 7, 8);
        let ntasks = ntg * ntg;
        let mut e = t;
        while e < ntasks {
            let tr0 = udiv(e, ntg) * 8;
            let sr0 = urem(e, ntg) * 8;
            if tr0 + 7 >= sr0 {
                let (mut a0, mut a1, mut a2, mut a3) = ([0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8]);
                let (mut a4, mut a5, mut a6, mut a7) = ([0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8]);
                let mut qt = [0u64; 8];
                let mut ks = [0u64; 8];
                // t 行宏：qt[r] 解 4 bf16 × ks[c]（c 常量展开）4 fma
                macro_rules! dn_mrow {
                    ($acc:ident, $r:expr) => {{
                        let (qlo, qhi) = (qt[$r] as u32, (qt[$r] >> 32) as u32);
                        let qa = f32::from_bits(qlo << 16);
                        let qb = f32::from_bits(qlo & 0xffff_0000);
                        let qc = f32::from_bits(qhi << 16);
                        let qd = f32::from_bits(qhi & 0xffff_0000);
                        for c in 0..8 {
                            let (slo, shi) = (ks[c] as u32, (ks[c] >> 32) as u32);
                            $acc[c] = fma(qa, f32::from_bits(slo << 16), $acc[c]);
                            $acc[c] = fma(qb, f32::from_bits(slo & 0xffff_0000), $acc[c]);
                            $acc[c] = fma(qc, f32::from_bits(shi << 16), $acc[c]);
                            $acc[c] = fma(qd, f32::from_bits(shi & 0xffff_0000), $acc[c]);
                        }
                    }};
                }
                let mut j4 = 0usize;
                while j4 < 32 {
                    let jb = j4 * 4;
                    for r in 0..8 {
                        if tr0 + r < n {
                            qt[r] = *(q.add(qrow(t0 + tr0 + r, jb)) as *const u64);
                        }
                    }
                    for c in 0..8 {
                        if sr0 + c < n {
                            ks[c] = *(k.add(qrow(t0 + sr0 + c, jb)) as *const u64);
                        }
                    }
                    dn_mrow!(a0, 0);
                    dn_mrow!(a1, 1);
                    dn_mrow!(a2, 2);
                    dn_mrow!(a3, 3);
                    dn_mrow!(a4, 4);
                    dn_mrow!(a5, 5);
                    dn_mrow!(a6, 6);
                    dn_mrow!(a7, 7);
                    j4 += 1;
                }
                // 列落位：Mᵀ[sr][tr0..+8]（4×u64，行连续）；端点缩放 +
                // s ≤ t 掩码（对角含）。c 展开为 8 支常量分支——运行期 c 会
                // 把 a0..a7 打回 local（循环体过大不展开 → 动态索引）
                // M[t][s] = invq_t·(q_t·k_s)·e^{C_s−C_t}（h 落 δ″=invk_s·δ_s
                // 配对）：指数 ≤ 0 恒安全——**不可**拆 e^{C_last−C_t}·e^{C_s−C_last}
                // （大 X 头两侧同时欠流 → 0·0，o35 head1-3 全零根因）；fastf 档
                // (C_last≤60) neg_t·pos_s·E 与 prep 的 A 同式，兜底逐对 expf
                // （宏体自由标识符按定义点解析 ⇒ 定义须在 hv 循环内）
                let mut hv = 0usize;
                while hv < vph {
                    let h = kk * vph + hv;
                    let vb = gc * vh + h;
                    let sc = scratch.add(vb * (5 * bts));
                    let neg = sc.add(3 * bts);
                    let pos = sc.add(4 * bts);
                    let mbase = mscratch.add(vb * bts * bts);
                    let clast = *gcum.add((t0 + n - 1) * vh + h);
                    let fastf = clast <= 60.0f32;
                    let bige = if fastf { expf(clast) } else { 1.0f32 };
                    macro_rules! dn_mstc {
                        ($c:expr) => {{
                            let sr = sr0 + $c;
                            if sr < n {
                                macro_rules! mv2 {
                                    ($ar:ident, $tr:expr) => {
                                        if $tr < n && $tr >= sr {
                                            if fastf {
                                                *inv.add((t0 + $tr) * kh_n * 2 + kh * 2)
                                                    * *neg.add($tr)
                                                    * *pos.add(sr)
                                                    * bige
                                                    * $ar[$c]
                                            } else {
                                                *inv.add((t0 + $tr) * kh_n * 2 + kh * 2)
                                                    * $ar[$c]
                                                    * expf(
                                                        *gcum.add((t0 + sr) * vh + h)
                                                            - *gcum.add((t0 + $tr) * vh + h),
                                                    )
                                            }
                                        } else {
                                            0.0f32
                                        }
                                    };
                                }
                                let p0 = mbase.add(sr * bts + tr0) as *mut u64;
                                *p0 = (mv2!(a0, tr0).to_bits() as u64)
                                    | ((mv2!(a1, tr0 + 1).to_bits() as u64) << 32);
                                *(p0.add(1)) = (mv2!(a2, tr0 + 2).to_bits() as u64)
                                    | ((mv2!(a3, tr0 + 3).to_bits() as u64) << 32);
                                *(p0.add(2)) = (mv2!(a4, tr0 + 4).to_bits() as u64)
                                    | ((mv2!(a5, tr0 + 5).to_bits() as u64) << 32);
                                *(p0.add(3)) = (mv2!(a6, tr0 + 6).to_bits() as u64)
                                    | ((mv2!(a7, tr0 + 7).to_bits() as u64) << 32);
                            }
                        }};
                    }
                    dn_mstc!(0);
                    dn_mstc!(1);
                    dn_mstc!(2);
                    dn_mstc!(3);
                    dn_mstc!(4);
                    dn_mstc!(5);
                    dn_mstc!(6);
                    dn_mstc!(7);
                    hv += 1;
                }
            }
            e += 256;
        }
    }
    _syncthreads();
    // ── ②+③ 任务 = (t 八元 tg × v 八元 vg) 寄存器瓦片（跨步 256 领）：
    //    ② base = Σ_j S₀[v][j]·q̂_t[j]（S₀ bf16 8 行 + q bf16 8 行，
    //    16×u64/256 fma）；③ o += Σ_{s≤t} M[t][s]·δ[s][v]（Mᵀ 8-t 2×u64 +
    //    δ 8-v 4×u64 → 4 载/64 fma）；行缩放 rs = e^{−C_t}·invq_t 尾乘，
    //    /√kd 收 bf16 ──
    if true {
        // t 八元 × v 四元：bt=64 下任务 = 8·32 = 256 = 线程数（8×8 只有 128，
        // 半线程空转——③ 的 s 链载入延迟无 warp 可藏）。v 轴切（而非 t 轴）：
        // δ/M 每 s 步载入字节数不变，总流量不涨
        let ntg = udiv(n + 7, 8);
        let nvt = vd / 4;
        let hvt = ntg * nvt;
        let ntasks = hvt * vph; // hv 并入任务轴：块数减半、任务翻倍，总量不变
        let mut e = t;
        while e < ntasks {
            let hv = udiv(e, hvt);
            let e2 = e - hv * hvt;
            let h = kk * vph + hv;
            let vb = gc * vh + h;
            let mbase = mscratch.add(vb * bts * bts);
            let s0 = snap.add(vb * vd * kd); // 本块首状态
            let neg = scratch.add(vb * (5 * bts) + 3 * bts);
            let tr0 = udiv(e2, nvt) * 8;
            let v0 = urem(e2, nvt) * 4;
            let (mut a0, mut a1, mut a2, mut a3) = ([0f32; 4], [0f32; 4], [0f32; 4], [0f32; 4]);
            let (mut a4, mut a5, mut a6, mut a7) = ([0f32; 4], [0f32; 4], [0f32; 4], [0f32; 4]);
            let mut qt = [0u64; 8];
            let mut st = [0u64; 4];
            // ② base 行宏：qt[r] 解 4 bf16 × S₀ 4 行（c 常量展开）4 fma
            macro_rules! dn_orow {
                ($acc:ident, $r:expr) => {{
                    let (qlo, qhi) = (qt[$r] as u32, (qt[$r] >> 32) as u32);
                    let qa = f32::from_bits(qlo << 16);
                    let qb = f32::from_bits(qlo & 0xffff_0000);
                    let qc = f32::from_bits(qhi << 16);
                    let qd = f32::from_bits(qhi & 0xffff_0000);
                    for c in 0..4 {
                        let (slo, shi) = (st[c] as u32, (st[c] >> 32) as u32);
                        $acc[c] = fma(qa, f32::from_bits(slo << 16), $acc[c]);
                        $acc[c] = fma(qb, f32::from_bits(slo & 0xffff_0000), $acc[c]);
                        $acc[c] = fma(qc, f32::from_bits(shi << 16), $acc[c]);
                        $acc[c] = fma(qd, f32::from_bits(shi & 0xffff_0000), $acc[c]);
                    }
                }};
            }
            // base 行缩放（rs = e^{−C_t}·invq_t）
            macro_rules! dn_oscale {
                ($acc:ident, $r:expr) => {{
                    if tr0 + $r < n {
                        let rs = *neg.add(tr0 + $r)
                            * *inv.add((t0 + tr0 + $r) * kh_n * 2 + kh * 2);
                        for c in 0..4 {
                            $acc[c] *= rs;
                        }
                    }
                }};
            }
            // 落盘行宏：/√kd 收 bf16，4 连续 v = 1×u64 存
            macro_rules! dn_ost {
                ($acc:ident, $r:expr) => {{
                    if tr0 + $r < n {
                        let po =
                            core_out.add((t0 + tr0 + $r) * vh * vd + h * vd + v0) as *mut u64;
                        *po = (f32_to_bf16($acc[0] / kd_sqrt) as u64)
                            | ((f32_to_bf16($acc[1] / kd_sqrt) as u64) << 16)
                            | ((f32_to_bf16($acc[2] / kd_sqrt) as u64) << 32)
                            | ((f32_to_bf16($acc[3] / kd_sqrt) as u64) << 48);
                    }
                }};
            }
            let mut j4 = 0usize;
            while j4 < 32 {
                let jb = j4 * 4;
                for r in 0..8 {
                    if tr0 + r < n {
                        qt[r] = *(q.add(qrow(t0 + tr0 + r, jb)) as *const u64);
                    }
                }
                for c in 0..4 {
                    st[c] = *(s0.add((v0 + c) * kd + jb) as *const u64);
                }
                dn_orow!(a0, 0);
                dn_orow!(a1, 1);
                dn_orow!(a2, 2);
                dn_orow!(a3, 3);
                dn_orow!(a4, 4);
                dn_orow!(a5, 5);
                dn_orow!(a6, 6);
                dn_orow!(a7, 7);
                j4 += 1;
            }
            dn_oscale!(a0, 0);
            dn_oscale!(a1, 1);
            dn_oscale!(a2, 2);
            dn_oscale!(a3, 3);
            dn_oscale!(a4, 4);
            dn_oscale!(a5, 5);
            dn_oscale!(a6, 6);
            dn_oscale!(a7, 7);
            // ③ s 升序链（s ≤ t）：Mᵀ[s][tr0..+8] 4×u64 + δ[v0..+4] 2×u64
            let smax = if tr0 + 8 < n { tr0 + 8 } else { n };
            let mut s = 0usize;
            while s < smax {
                let m2a = *(mbase.add(s * bts + tr0) as *const u64);
                let m2b = *(mbase.add(s * bts + tr0 + 2) as *const u64);
                let m2c = *(mbase.add(s * bts + tr0 + 4) as *const u64);
                let m2d = *(mbase.add(s * bts + tr0 + 6) as *const u64);
                let d2a = *(dn.add((t0 + s) * vh * vd + h * vd + v0) as *const u64);
                let d2b = *(dn.add((t0 + s) * vh * vd + h * vd + v0 + 2) as *const u64);
                let dv = [
                    f32::from_bits(d2a as u32),
                    f32::from_bits((d2a >> 32) as u32),
                    f32::from_bits(d2b as u32),
                    f32::from_bits((d2b >> 32) as u32),
                ];
                let mv = [
                    f32::from_bits(m2a as u32),
                    f32::from_bits((m2a >> 32) as u32),
                    f32::from_bits(m2b as u32),
                    f32::from_bits((m2b >> 32) as u32),
                    f32::from_bits(m2c as u32),
                    f32::from_bits((m2c >> 32) as u32),
                    f32::from_bits(m2d as u32),
                    f32::from_bits((m2d >> 32) as u32),
                ];
                macro_rules! dn_ofma {
                    ($acc:ident, $r:literal) => {{
                        for c in 0..4 {
                            $acc[c] = fma(mv[$r], dv[c], $acc[c]);
                        }
                    }};
                }
                dn_ofma!(a0, 0);
                dn_ofma!(a1, 1);
                dn_ofma!(a2, 2);
                dn_ofma!(a3, 3);
                dn_ofma!(a4, 4);
                dn_ofma!(a5, 5);
                dn_ofma!(a6, 6);
                dn_ofma!(a7, 7);
                s += 1;
            }
            // 落盘（dn_ost 行宏，tr<n 守卫）
            dn_ost!(a0, 0);
            dn_ost!(a1, 1);
            dn_ost!(a2, 2);
            dn_ost!(a3, 3);
            dn_ost!(a4, 4);
            dn_ost!(a5, 5);
            dn_ost!(a6, 6);
            dn_ost!(a7, 7);
            e += 256;
        }
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

