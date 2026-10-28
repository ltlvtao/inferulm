//! CUDA Driver API 最小面：dlopen(libcuda.so.1) + dlsym，零 toolkit 依赖。
//!
//! 只绑定冒烟所需符号；带版本后缀的符号**必须 _v2 优先**——裸名是 CUDA 4
//! 时代 32 位 dptr 的 v1 入口（显存函数上返回 201 invalid context，冒烟
//! 在 WSL 驱动 610.57 上实抓），裸名仅当 _v2 缺失时兜底。全部函数返回 CUresult（i32，0 = 成功），错误经 cuGetErrorString
//! 人话化。exec 层接入后按需扩面（event/graph），不做大而全的绑定层。
use std::ffi::{c_char, c_int, c_void, CStr};

use libloading::Library;

pub type CuResult = c_int;
pub const CUDA_SUCCESS: CuResult = 0;

/// cuDeviceGetAttribute 的属性号（只列用到的）
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

#[derive(Debug)]
pub struct CuError {
    pub code: CuResult,
    pub symbol: &'static str,
    pub text: String,
}

impl std::fmt::Display for CuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} 失败：{} ({})", self.symbol, self.text, self.code)
    }
}
impl std::error::Error for CuError {}

pub struct Driver {
    _lib: Library, // dlopen 句柄须比符号活得久
    cu_get_error_string: unsafe extern "C" fn(c_int, *mut *const c_char) -> CuResult,
    cu_init: unsafe extern "C" fn(c_int) -> CuResult,
    cu_device_get_count: unsafe extern "C" fn(*mut c_int) -> CuResult,
    cu_device_get_name: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> CuResult,
    cu_device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> CuResult,
    cu_device_primary_ctx_retain: unsafe extern "C" fn(*mut *mut c_void, c_int) -> CuResult,
    cu_ctx_set_current: unsafe extern "C" fn(*mut c_void) -> CuResult,
    cu_module_load_data_ex: unsafe extern "C" fn(
        *mut *mut c_void,          // CUmodule*
        *const c_void,             // image（PTX 文本 / cubin）
        c_int,                     // numJitOptions
        *mut c_int,                // CUjit_option*
        *mut *mut c_void,          // void**
    ) -> CuResult,
    cu_module_get_function:
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CuResult,
    cu_mem_alloc: unsafe extern "C" fn(*mut u64, usize) -> CuResult,
    cu_mem_free: unsafe extern "C" fn(u64) -> CuResult,
    cu_memcpy_htod: unsafe extern "C" fn(u64, *const c_void, usize) -> CuResult,
    cu_memcpy_dtoh: unsafe extern "C" fn(*mut c_void, u64, usize) -> CuResult,
    cu_memset_d8: unsafe extern "C" fn(u64, u8, usize, *mut c_void) -> CuResult,
    cu_stream_create: unsafe extern "C" fn(*mut *mut c_void, u32) -> CuResult,
    cu_stream_synchronize: unsafe extern "C" fn(*mut c_void) -> CuResult,
    cu_launch_kernel: unsafe extern "C" fn(
        *mut c_void, // CUfunction
        u32, u32, u32, // grid x/y/z
        u32, u32, u32, // block x/y/z
        u32,          // sharedMemBytes
        *mut c_void,  // hStream
        *mut *mut c_void, // kernelParams
        *mut c_void,  // extra
    ) -> CuResult,
}

// 逐符号经 try_get 解析（版本化符号回退链：dlsym 第一个命中者胜），
// 拷出函数指针存字段——Library 由 _lib 字段保活，指针全程有效。
impl Driver {
    /// dlopen 驱动库并解析全部符号。WSL 下 libcuda.so.1 在 /usr/lib/wsl/lib，
    /// 已在 ld 路径内；裸名失败再试显式 WSL 路径。
    pub fn open() -> Result<Self, Box<dyn std::error::Error>> {
        let lib = unsafe {
            Library::new("libcuda.so.1")
                .or_else(|_| Library::new("/usr/lib/wsl/lib/libcuda.so.1"))?
        };
        // 符号全部解析完再移入 lib（借用在构造前结束）
        unsafe {
            let d = Driver {
                cu_get_error_string: try_get(&lib, &["cuGetErrorString"])?,
                cu_init: try_get(&lib, &["cuInit"])?,
                cu_device_get_count: try_get(&lib, &["cuDeviceGetCount"])?,
                cu_device_get_name: try_get(&lib, &["cuDeviceGetName"])?,
                cu_device_get_attribute: try_get(&lib, &["cuDeviceGetAttribute"])?,
                cu_device_primary_ctx_retain: try_get(&lib, &["cuDevicePrimaryCtxRetain"])?,
                cu_ctx_set_current: try_get(&lib, &["cuCtxSetCurrent"])?,
                cu_module_load_data_ex: try_get(&lib, &["cuModuleLoadDataEx"])?,
                cu_module_get_function: try_get(&lib, &["cuModuleGetFunction"])?,
                cu_mem_alloc: try_get(&lib, &["cuMemAlloc_v2", "cuMemAlloc"])?,
                cu_mem_free: try_get(&lib, &["cuMemFree_v2", "cuMemFree"])?,
                cu_memcpy_htod: try_get(&lib, &["cuMemcpyHtoD_v2", "cuMemcpyHtoD"])?,
                cu_memcpy_dtoh: try_get(&lib, &["cuMemcpyDtoH_v2", "cuMemcpyDtoH"])?,
                cu_memset_d8: try_get(&lib, &["cuMemsetD8Async"])?,
                cu_stream_create: try_get(&lib, &["cuStreamCreate"])?,
                cu_stream_synchronize: try_get(&lib, &["cuStreamSynchronize"])?,
                cu_launch_kernel: try_get(&lib, &["cuLaunchKernel"])?,
                _lib: lib,
            };
            Ok(d)
        }
    }

    fn check(&self, code: CuResult, symbol: &'static str) -> Result<(), CuError> {
        if code == CUDA_SUCCESS {
            return Ok(());
        }
        let mut msg: *const c_char = std::ptr::null();
        let _ = unsafe { (self.cu_get_error_string)(code, &mut msg) };
        let text = if msg.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(msg) }.to_string_lossy().into_owned()
        };
        Err(CuError { code, symbol, text })
    }

    // —— 原语（pub(crate)：高层封装在 mod.rs）——

    pub(crate) fn init(&self) -> Result<(), CuError> {
        self.check(unsafe { (self.cu_init)(0) }, "cuInit")
    }
    pub(crate) fn device_count(&self) -> Result<i32, CuError> {
        let mut n = 0i32;
        self.check(unsafe { (self.cu_device_get_count)(&mut n) }, "cuDeviceGetCount")?;
        Ok(n)
    }
    pub(crate) fn device_name(&self, dev: i32) -> Result<String, CuError> {
        let mut buf = [0u8; 256];
        self.check(
            unsafe { (self.cu_device_get_name)(buf.as_mut_ptr().cast(), buf.len() as i32 - 1, dev) },
            "cuDeviceGetName",
        )?;
        Ok(String::from_utf8_lossy(&buf).trim_end_matches('\0').to_string())
    }
    pub(crate) fn device_attribute(&self, attr: c_int, dev: i32) -> Result<i32, CuError> {
        let mut v = 0i32;
        self.check(unsafe { (self.cu_device_get_attribute)(&mut v, attr, dev) }, "cuDeviceGetAttribute")?;
        Ok(v)
    }
    pub(crate) fn primary_ctx_retain(&self, dev: i32) -> Result<*mut c_void, CuError> {
        let mut ctx = std::ptr::null_mut();
        self.check(
            unsafe { (self.cu_device_primary_ctx_retain)(&mut ctx, dev) },
            "cuDevicePrimaryCtxRetain",
        )?;
        Ok(ctx)
    }
    pub(crate) fn ctx_set_current(&self, ctx: *mut c_void) -> Result<(), CuError> {
        self.check(unsafe { (self.cu_ctx_set_current)(ctx) }, "cuCtxSetCurrent")
    }
    pub(crate) fn module_load_data(&self, image: &[u8]) -> Result<*mut c_void, CuError> {
        let mut module = std::ptr::null_mut();
        self.check(
            unsafe {
                (self.cu_module_load_data_ex)(
                    &mut module,
                    image.as_ptr().cast(),
                    0, // 不带 JIT 选项——缓存行为交给驱动默认（k7 验证）
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            },
            "cuModuleLoadDataEx",
        )?;
        Ok(module)
    }
    pub(crate) fn module_get_function(
        &self,
        module: *mut c_void,
        name: &str,
    ) -> Result<*mut c_void, CuError> {
        let c_name = std::ffi::CString::new(name).expect("kernel 名不含 NUL");
        let mut f = std::ptr::null_mut();
        self.check(
            unsafe { (self.cu_module_get_function)(&mut f, module, c_name.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        Ok(f)
    }
    pub(crate) fn mem_alloc(&self, bytes: usize) -> Result<u64, CuError> {
        let mut ptr = 0u64;
        self.check(unsafe { (self.cu_mem_alloc)(&mut ptr, bytes) }, "cuMemAlloc")?;
        Ok(ptr)
    }
    pub(crate) fn mem_free(&self, ptr: u64) -> Result<(), CuError> {
        self.check(unsafe { (self.cu_mem_free)(ptr) }, "cuMemFree")
    }
    pub(crate) fn memcpy_htod(&self, dst: u64, src: &[u8]) -> Result<(), CuError> {
        self.check(
            unsafe { (self.cu_memcpy_htod)(dst, src.as_ptr().cast(), src.len()) },
            "cuMemcpyHtoD",
        )
    }
    pub(crate) fn memcpy_dtoh(&self, dst: &mut [u8], src: u64) -> Result<(), CuError> {
        self.check(
            unsafe { (self.cu_memcpy_dtoh)(dst.as_mut_ptr().cast(), src, dst.len()) },
            "cuMemcpyDtoH",
        )
    }
    /// 异步字节清零（流上排队——状态池行回收的零化用）
    pub(crate) fn memset_d8(&self, dst: u64, byte: u8, n: usize, stream: *mut c_void) -> Result<(), CuError> {
        self.check(unsafe { (self.cu_memset_d8)(dst, byte, n, stream) }, "cuMemsetD8Async")
    }
    pub(crate) fn stream_create(&self) -> Result<*mut c_void, CuError> {
        let mut s = std::ptr::null_mut();
        self.check(unsafe { (self.cu_stream_create)(&mut s, 0) }, "cuStreamCreate")?;
        Ok(s)
    }
    pub(crate) fn stream_synchronize(&self, s: *mut c_void) -> Result<(), CuError> {
        self.check(unsafe { (self.cu_stream_synchronize)(s) }, "cuStreamSynchronize")
    }
    pub(crate) fn launch(
        &self,
        f: *mut c_void,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        params: &[*mut c_void],
        stream: *mut c_void,
    ) -> Result<(), CuError> {
        self.check(
            unsafe {
                (self.cu_launch_kernel)(
                    f,
                    grid.0, grid.1, grid.2,
                    block.0, block.1, block.2,
                    0,
                    stream,
                    params.as_ptr() as *mut *mut c_void,
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }
}

/// 按名字列表依次尝试 dlsym（版本化符号回退链），全 miss 才报错。
/// 拷出函数指针（Copy）返回——Library 由 Driver._lib 保活，指针全程有效
unsafe fn try_get<T: Copy>(
    lib: &Library,
    names: &[&'static str],
) -> Result<T, CuError> {
    for name in names {
        if let Ok(sym) = lib.get::<T>(name.as_bytes()) {
            return Ok(*sym);
        }
    }
    Err(CuError {
        code: -1,
        symbol: names[0],
        text: format!("符号缺失（已试 {names:?}）"),
    })
}
