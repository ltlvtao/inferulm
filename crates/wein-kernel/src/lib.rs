//! wein-kernel：算子层（02-architecture §4）。
//!
//! - `kernels/` 子 crate：no_std Rust 编译到 nvptx64 → PTX 文本（build.rs）
//! - `cuda::driver`：dlopen(libcuda.so.1) 的最小 Driver API 面（无 toolkit 依赖）
//! - `cuda::Gpu/Module/Func/Stream/DevBuf`：高层封装
//! - `ops::`：**trait Op 家族 + 启动时注册表**——按 HardwareProfile 解析一次，
//!   具体类型单态化，热路径零动态分发（§4）；首个成员 rms_norm
//! - `reference::`：CPU 参考实现——golden vector 生成端（cuda/asc 共享，§4）
//!
//! 厂商库 FFI（cuBLASLt）与 M0 其余算子按 §4 逐个入表。
pub mod cuda;
pub mod ops;
pub mod reference;

/// PTX 产物路径（build.rs 注入）。加载用文本读它；JIT 缓存由 driver 管。
pub fn ptx_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("WEIN_PTX"))
}

/// 组装 cuLaunchKernel 的参数指针数组：每个参数给出"存着参数值的 host 内存"
/// 的地址。实现方：标量（按值传参的 kernel 形参）与 DevBuf（设备指针形参）。
/// 用法：`let args = kernel_args![da, db, dc, n];`（n 为 u32 局部变量）
pub trait AsKernelArg {
    fn as_arg(&self) -> *mut std::ffi::c_void;
}

macro_rules! impl_as_kernel_arg {
    ($($t:ty),* $(,)?) => {
        $(impl AsKernelArg for $t {
            fn as_arg(&self) -> *mut std::ffi::c_void {
                self as *const _ as *mut std::ffi::c_void
            }
        })*
    };
}
impl_as_kernel_arg!(u8, u16, u32, u64, usize, i32, i64, f32, f64, bool);

#[macro_export]
macro_rules! kernel_args {
    ($($arg:expr),* $(,)?) => {
        vec![$($crate::AsKernelArg::as_arg(&$arg),)*]
    };
}
