//! wein-kernel：算子层（02-architecture §4）。
//!
//! 当前为 **cuda crate 冒烟实验**（§13-2）的落点：
//! - `kernels/` 子 crate：no_std Rust 编译到 nvptx64 → PTX 文本（build.rs）
//! - `cuda::driver`：dlopen(libcuda.so.1) 的最小 Driver API 面（无 toolkit 依赖）
//! - `cuda::Gpu/Module/Func/Stream/DevBuf`：高层封装，exec 层（后续）经此提交 kernel
//!
//! trait Op 注册表与厂商库 FFI（cuBLASLt）按 §4 在 W2 展开——本层冒烟先行，
//! 回答全项目最大技术风险：纯 Rust→PTX 路线能不能走。
pub mod cuda;

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
