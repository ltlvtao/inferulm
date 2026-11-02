//! cuda 高层封装：driver.rs 的最小面之上给出 exec 层将来直接消费的形状。
//!
//! 冒烟实验（§13-2）的对象；W2 扩 trait Op 注册表时，launch 的形状从这里长。
//! 主上下文进程生命周期保活（primary ctx 不释放——引擎即进程，冒烟不做
//! 情景化上下文管理）。
pub mod cublaslt;
pub mod driver;

use std::cell::Cell;
use std::ffi::c_void;

use driver::Driver;

pub use cublaslt::{BlasLt, GemmTuneRecord};

// ── 捕获区间守卫（§4 图捕获安全规则：ZoS forbid_cuda_sync 的 Rust 化）──
// 捕获中的隐式同步（sync / 回读 / 同步 H2D / 分配）会污染或炸掉图——
// debug 构建直接 panic（意外同步零容忍）；release 静默（守卫是开发期纪律）
thread_local! {
    static CAPTURING: Cell<bool> = const { Cell::new(false) };
}

/// 当前线程是否在流捕获区间内（begin..end 之间）
pub fn capturing() -> bool {
    CAPTURING.with(|c| c.get())
}

fn capture_guard() {
    debug_assert!(!capturing(), "捕获区间内禁止隐式同步/分配——图会被污染（§4 安全规则）");
}

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

    /// 计时事件（CUevent）：流上成对 record + elapsed_ms——步时分解
    /// 插桩的根基。同一事件可重复 record（插桩的事件池复用此合同）
    pub fn event(&self) -> Event<'_> {
        Event { drv: &self.drv, handle: self.drv.event_create().expect("cuEventCreate") }
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
        capture_guard();
        DevBuf {
            drv: &self.drv,
            ptr: self.drv.mem_alloc(bytes).expect("cuMemAlloc"),
            bytes,
            owned: true,
        }
    }
    pub fn alloc_from_slice<T: Copy>(&self, slice: &[T]) -> DevBuf<'_> {
        capture_guard();
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

/// 计时事件（CUevent）——GPU 时间轴上的标记点。record_on 排进流，
/// elapsed_ms 量两点间 GPU 时距（须已完成）。插桩用事件池反复 record。
pub struct Event<'a> {
    drv: &'a Driver,
    handle: *mut c_void,
}

impl Event<'_> {
    /// 流上记录（保序：排在前序 kernel 完成之后）
    pub fn record_on(&self, stream: &Stream) {
        self.drv.event_record(self.handle, stream.handle).expect("cuEventRecord");
    }
    /// 自 `since` 到本事件的 GPU 时距（ms；须两者均已完成）
    pub fn elapsed_ms(&self, since: &Event) -> f32 {
        let mut ms = 0f32;
        self.drv
            .event_elapsed_time(&mut ms, since.handle, self.handle)
            .expect("cuEventElapsedTime");
        ms
    }
}

impl Drop for Event<'_> {
    fn drop(&mut self) {
        self.drv.event_destroy(self.handle).expect("cuEventDestroy");
    }
}

impl Stream<'_> {
    pub fn sync(&self) {
        capture_guard();
        self.drv.stream_synchronize(self.handle).expect("cuStreamSynchronize");
    }

    /// 流间 join：本流后续工作排队等待事件完成（跨流依赖边）。
    /// fork/join 重叠的根基；图捕获区间内合法——依赖边烤进图拓扑
    pub fn wait_event(&self, e: &Event) {
        self.drv.stream_wait_event(self.handle, e.handle, 0).expect("cuStreamWaitEvent");
    }

    /// 批量 D2D（流上排队，段间独立无序）：一次驱动调用搬多段——快照
    /// 逐层小拷贝的调用开销摊销（48 段 × 每步多次跨块：WSL 每调用 ~4µs
    /// 会把 0.25ms 的流量账放大成 ms 级）
    pub fn memcpy_batch(&self, ops: &[driver::CuBatchOp]) {
        assert!(!ops.is_empty(), "空批不发行（cuMemcpyBatchAsync 须 numOps > 0）");
        self.drv.memcpy_batch_async(ops, self.handle).expect("cuMemcpyBatchAsync");
    }

    /// 进入流捕获：此后的 launch 序列录制进图（THREAD_LOCAL 模式）。
    /// 区间内隐式同步/分配在 debug 构建直接 panic（§4 安全规则）
    pub fn begin_capture(&self) {
        self.drv.stream_begin_capture(self.handle).expect("cuStreamBeginCapture");
        CAPTURING.with(|c| c.set(true));
    }

    /// 结束捕获，返回图。设备指针烤进节点、数据留在缓冲——回放读执行时内存
    pub fn end_capture(&self) -> Graph<'_> {
        CAPTURING.with(|c| c.set(false));
        let handle = self.drv.stream_end_capture(self.handle).expect("cuStreamEndCapture");
        Graph { drv: self.drv, handle }
    }
}

/// 已捕获的图（CUgraph）。`instantiate` 得可回放 exec；拓扑演化须重捕。
/// 生命周期只锚 Driver（exec 可与图本体脱离——图 Drop 不缚 exec）
pub struct Graph<'a> {
    drv: &'a Driver,
    handle: *mut c_void,
}

impl<'a> Graph<'a> {
    /// 实例化为可回放体（一次性上传到设备——launch 热路径零解析）
    pub fn instantiate(&self) -> GraphExec<'a> {
        let exec = self.drv.graph_instantiate(self.handle).expect("cuGraphInstantiateWithFlags");
        GraphExec { drv: self.drv, handle: exec }
    }
}

impl Drop for Graph<'_> {
    fn drop(&mut self) {
        self.drv.graph_destroy(self.handle).expect("cuGraphDestroy");
    }
}

/// 实例化图（CUgraphExec）——回放体。`update` 做参数重绑（拓扑须同）
pub struct GraphExec<'a> {
    drv: &'a Driver,
    handle: *mut c_void,
}

impl GraphExec<'_> {
    /// 回放（流上排队，一次 launch 提交全图 kernel 序）
    pub fn launch(&self, stream: &Stream) {
        self.drv.graph_launch(self.handle, stream.handle).expect("cuGraphLaunch");
    }

    /// 参数重绑：同拓扑的新图烤进本 exec（免重实例化；分桶/缓冲迁移的更新路）
    pub fn update(&self, graph: &Graph) {
        self.drv.graph_exec_update(self.handle, graph.handle).expect("cuGraphExecUpdate");
    }
}

impl Drop for GraphExec<'_> {
    fn drop(&mut self) {
        self.drv.graph_exec_destroy(self.handle).expect("cuGraphExecDestroy");
    }
}

/// 设备缓冲。Drop 释放（同步语义够冒烟；DeferredRelease 语义在 exec 层引入）
pub struct DevBuf<'a> {
    drv: &'a Driver,
    ptr: u64,
    bytes: usize,
    /// false = `narrow` 出来的偏移视图（源缓冲的内部指针）——drop 不回收
    owned: bool,
}

impl DevBuf<'_> {
    /// 字节容量
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// launch 参数位：指向**存着设备地址的 host 内存**（AsKernelArg 消费）
    fn arg_slot(&self) -> *const core::ffi::c_void {
        &self.ptr as *const u64 as *const core::ffi::c_void
    }
    /// 设备地址（CUdeviceptr）——非零校验/日志用，不进 launch 参数
    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }
    /// 偏移视图（行分片用）：同一驱动句柄、ptr 前移 start 字节、容量收窄到
    /// len。零拷贝零分配——attn 大块 prefill 的"单次 launch 行容量"不足时
    /// 逐片发 kernel，各片拿本缓冲的窄视图（lifetime 跟随源缓冲）
    pub fn narrow(&self, start: usize, len: usize) -> DevBuf<'_> {
        assert!(start + len <= self.bytes, "narrow 区间越界：{start}+{len} > {}", self.bytes);
        DevBuf { drv: self.drv, ptr: self.ptr + start as u64, bytes: len, owned: false }
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
    /// 区间 D2D 拷贝（流上排队——快照存/恢复：本缓冲区间 ← 源缓冲区间，位拷贝）
    pub fn copy_range_on(
        &self,
        dst_start: usize,
        src: &DevBuf,
        src_start: usize,
        len: usize,
        stream: &Stream,
    ) {
        assert!(dst_start + len <= self.bytes, "D2D 目标区间越界");
        assert!(src_start + len <= src.bytes, "D2D 源区间越界");
        self.drv
            .memcpy_dtod_async(self.ptr + dst_start as u64, src.ptr + src_start as u64, len, stream.handle)
            .expect("cuMemcpyDtoDAsync");
    }
    pub fn read_to_vec<T: Copy + Default>(&self) -> Vec<T> {
        capture_guard();
        let n = self.bytes / std::mem::size_of::<T>();
        let mut v = vec![T::default(); n];
        let byte_len = n * std::mem::size_of::<T>();
        let dst = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast(), byte_len) };
        self.drv.memcpy_dtoh(dst, self.ptr).expect("cuMemcpyDtoH");
        v
    }

    /// 异步整段覆写上传（流上排队——常驻输入缓冲的每步写入口）。
    /// 切片长度须与缓冲字节容量对齐（整段覆写语义，防残留）
    pub fn upload_on<T: Copy>(&self, slice: &[T], stream: &Stream) {
        let n = slice.len() * std::mem::size_of::<T>();
        assert_eq!(n, self.bytes, "upload_on 整段覆写：{n} 字节 ≠ 缓冲 {} 字节", self.bytes);
        self.drv
            .memcpy_htod_async(self.ptr, slice.as_ptr().cast(), n, stream.handle)
            .expect("cuMemcpyHtoDAsync");
    }

    /// 异步前缀上传（流上排队）：只搬 `slice.len()` 项到缓冲开头，尾部残留
    /// 不动——常驻 T_MAX 容量缓冲按"本步实际用的行数"喂（eager 混合批的
    /// 十几次小表上传既不想每步 alloc，也不想整段 512 行搬）。
    /// 合同：缓冲容量 ≥ slice，且 kernel 只读 slice 覆盖的前缀
    pub fn upload_prefix_on<T: Copy>(&self, slice: &[T], stream: &Stream) {
        let n = slice.len() * std::mem::size_of::<T>();
        assert!(n <= self.bytes, "upload_prefix_on 越界：{n} 字节 > 缓冲 {} 字节", self.bytes);
        if n == 0 {
            return;
        }
        self.drv
            .memcpy_htod_async(self.ptr, slice.as_ptr().cast(), n, stream.handle)
            .expect("cuMemcpyHtoDAsync");
    }
}

impl Drop for DevBuf<'_> {
    fn drop(&mut self) {
        if self.owned {
            self.drv.mem_free(self.ptr).expect("cuMemFree");
        }
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
