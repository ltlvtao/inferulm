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
