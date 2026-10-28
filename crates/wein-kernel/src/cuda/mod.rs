//! cuda 高层封装：driver.rs 的最小面之上给出 exec 层将来直接消费的形状。
//!
//! 冒烟实验（§13-2）的对象；W2 扩 trait Op 注册表时，launch 的形状从这里长。
//! 主上下文进程生命周期保活（primary ctx 不释放——引擎即进程，冒烟不做
//! 情景化上下文管理）。
pub mod cublaslt;
pub mod driver;

use std::ffi::c_void;

use driver::Driver;

pub use cublaslt::BlasLt;

/// 已初始化的 GPU（设备 0 的主上下文已 current）
pub struct Gpu {
    drv: Driver,
    ctx: *mut std::ffi::c_void,
    arch: String,
    name: String,
    cc: (i32, i32),
    count: i32,
}

impl Gpu {
    /// dlopen 驱动 → cuInit → 取设备 0 主上下文并设为 current。
    /// 失败返回人话错误（无卡环境据此 SKIP，见测试 gpu() 门控）。
    pub fn probe() -> Result<Self, String> {
        let drv = Driver::open().map_err(|e| e.to_string())?;
        drv.init().map_err(|e| e.to_string())?;
        let count = drv.device_count().map_err(|e| e.to_string())?;
        if count < 1 {
            return Err("无 CUDA 设备".into());
        }
        let name = drv.device_name(0).map_err(|e| e.to_string())?;
        let maj = drv
            .device_attribute(driver::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0)
            .map_err(|e| e.to_string())?;
        let min = drv
            .device_attribute(driver::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0)
            .map_err(|e| e.to_string())?;
        let ctx = drv.primary_ctx_retain(0).map_err(|e| e.to_string())?;
        drv.ctx_set_current(ctx).map_err(|e| e.to_string())?;
        Ok(Self { drv, ctx, arch: format!("sm_{maj}{min}"), name, cc: (maj, min), count })
    }

    /// 把 primary context 设为**当前线程** current（Driver API 的 context 是
    /// 线程局部的——probe 只在调用线程 current）。跨线程消费（engine 任务
    /// 可在 worker 间迁移）在进入设备路径前统一调用；幂等且亚微秒。
    /// 单线程消费者无需关心（probe 时已 current）
    pub fn bind_thread(&self) {
        self.drv.ctx_set_current(self.ctx).expect("cuCtxSetCurrent");
    }

    pub fn device_count(&self) -> i32 {
        self.count
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn compute_capability(&self) -> (i32, i32) {
        self.cc
    }
    /// PTX JIT 目标字串（如 sm_89）——三层能力模型的身份层出口
    pub fn arch(&self) -> &str {
        &self.arch
    }

    pub fn stream(&self) -> Stream<'_> {
        Stream { drv: &self.drv, handle: self.drv.stream_create().expect("cuStreamCreate") }
    }

    /// 加载 PTX 文本（driver JIT；缓存落盘由 k7 验证）。
    /// 镜像必须 NUL 结尾（驱动合同）——Rust &str 不带，这里补上
    pub fn load_module(&self, ptx_text: &str) -> Module<'_> {
        let mut image = Vec::with_capacity(ptx_text.len() + 1);
        image.extend_from_slice(ptx_text.as_bytes());
        image.push(0);
        let handle = self.drv.module_load_data(&image).expect("cuModuleLoadDataEx");
        Module { drv: &self.drv, handle }
    }

    pub fn alloc_bytes(&self, bytes: usize) -> DevBuf<'_> {
        DevBuf { drv: &self.drv, ptr: self.drv.mem_alloc(bytes).expect("cuMemAlloc"), bytes }
    }
    pub fn alloc_from_slice<T: Copy>(&self, slice: &[T]) -> DevBuf<'_> {
        let buf = self.alloc_bytes(slice.len() * std::mem::size_of::<T>());
        self.drv.memcpy_htod(buf.ptr, unsafe { as_bytes(slice) }).expect("cuMemcpyHtoD");
        buf
    }

    /// 一维约定提交（冒烟够用；W2 按算子形状扩多维/动态 shmem）
    pub fn launch(
        &self,
        f: &Func,
        grid: u32,
        block: u32,
        args: &[*mut c_void],
        stream: &Stream,
    ) {
        self.drv
            .launch(f.handle, (grid, 1, 1), (block, 1, 1), args, stream.handle)
            .expect("cuLaunchKernel");
    }
}

/// 已加载模块（CUmodule 句柄）——kernel 按名取用
pub struct Module<'a> {
    drv: &'a Driver,
    handle: *mut c_void,
}

impl Module<'_> {
    pub fn func(&self, name: &str) -> Func {
        Func { handle: self.drv.module_get_function(self.handle, name).expect("cuModuleGetFunction") }
    }
}

/// kernel 入口（CUfunction 句柄）
pub struct Func {
    handle: *mut c_void,
}

pub struct Stream<'a> {
    drv: &'a Driver,
    handle: *mut c_void,
}

impl Stream<'_> {
    pub fn sync(&self) {
        self.drv.stream_synchronize(self.handle).expect("cuStreamSynchronize");
    }
}

/// 设备缓冲。Drop 释放（同步语义够冒烟；DeferredRelease 语义在 exec 层引入）
pub struct DevBuf<'a> {
    drv: &'a Driver,
    ptr: u64,
    bytes: usize,
}

impl DevBuf<'_> {
    /// launch 参数位：指向**存着设备地址的 host 内存**（AsKernelArg 消费）
    fn arg_slot(&self) -> *const core::ffi::c_void {
        &self.ptr as *const u64 as *const core::ffi::c_void
    }
    /// 设备地址（CUdeviceptr）——非零校验/日志用，不进 launch 参数
    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }
    pub fn len_bytes(&self) -> usize {
        self.bytes
    }
    /// 异步字节清零（流上排队——与后续 launch 保序）
    pub fn zero_on(&self, stream: &Stream) {
        self.drv.memset_d8(self.ptr, 0, self.bytes, stream.handle).expect("cuMemsetD8Async");
    }
    /// 区间字节清零（状态池单行回收零化：行内偏移 + 字节数）
    pub fn zero_range_on(&self, start: usize, len: usize, stream: &Stream) {
        assert!(start + len <= self.bytes, "零化区间越界");
        self.drv
            .memset_d8(self.ptr + start as u64, 0, len, stream.handle)
            .expect("cuMemsetD8Async");
    }
    pub fn read_to_vec<T: Copy + Default>(&self) -> Vec<T> {
        let n = self.bytes / std::mem::size_of::<T>();
        let mut v = vec![T::default(); n];
        let byte_len = n * std::mem::size_of::<T>();
        let dst = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast(), byte_len) };
        self.drv.memcpy_dtoh(dst, self.ptr).expect("cuMemcpyDtoH");
        v
    }
}

impl Drop for DevBuf<'_> {
    fn drop(&mut self) {
        self.drv.mem_free(self.ptr).expect("cuMemFree");
    }
}

impl crate::AsKernelArg for DevBuf<'_> {
    fn as_arg(&self) -> *mut core::ffi::c_void {
        self.arg_slot() as *mut core::ffi::c_void
    }
}

/// &[T] → &[u8]（Copy 语义，无对齐改编）
unsafe fn as_bytes<T: Copy>(slice: &[T]) -> &[u8] {
    std::slice::from_raw_parts(slice.as_ptr().cast(), slice.len() * std::mem::size_of::<T>())
}
