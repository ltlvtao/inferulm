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

use core::arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x};

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
