//! 设备会话：exec 层对 (Gpu, OpTable, Stream) 的唯一持有形态。
//!
//! self-referential 钉扎：gpu 装在 Box 里（堆地址稳定、字段永不替换），
//! 它的借用（OpTable / Stream / 各 DevBuf）经 transmute 提为 'static 挂回
//! 同一结构——字段声明序保证借用者先于 gpu 析构。W2 计划用 arena
//! （02-architecture §5）取代手写钉扎。
use wein_kernel::cuda::{Gpu, Stream};
use wein_kernel::ops::{OpTable, ScratchCaps};
use wein_platform::{Device, HardwareProfile};

/// 生命周期钉扎（见模块注释）；调用点的目标类型（'static）驱动推断
macro_rules! pin {
    ($v:expr) => {{
        let v = $v;
        unsafe { std::mem::transmute(v) }
    }};
}
pub(crate) use pin;

// CUDA driver API 本身线程安全；本结构独占持有全部句柄，exec 由单线程驱动
// （§5 拓扑）——上移 Send 是安全的
unsafe impl Send for DeviceSession {}

pub struct DeviceSession {
    // 声明序 = 析构序：借用者（stream/table）先走，gpu 最后
    stream: Stream<'static>,
    table: OpTable<'static>,
    gpu: Box<Gpu>,
}

impl DeviceSession {
    /// 探测→画像→resolve→流。任何一环失败都是启动期错误（Degrade 链
    /// 原样透出，不带病运行）。gemm 全 heuristic 钉（瞬态形状路径）
    pub fn open() -> Result<Self, String> {
        Self::open_tuned(&[])
    }

    /// 同 `open`，另指定 gemm autotune 的 m 集（稳态 decode 桶集）
    pub fn open_tuned(tune_m: &[u32]) -> Result<Self, String> {
        Self::open_tuned_with(tune_m, &ScratchCaps::default())
    }

    /// `open_tuned` + scratch 容量参数（大块 prefill 的 t_max 抬档：rms/
    /// deltanet l2 scratch 随批行数放大，exec 按模型几何 + ctx::t_max 推导）
    pub fn open_tuned_with(tune_m: &[u32], caps: &ScratchCaps) -> Result<Self, String> {
        let gpu = Box::new(Gpu::probe().map_err(|e| format!("GPU 探测失败：{e}"))?);
        let profile = HardwareProfile::with_device(Device::Cuda).with_arch(gpu.arch());
        let table = OpTable::resolve_tuned_with(&gpu, &profile, tune_m, caps).map_err(|ds| {
            let reasons: Vec<String> = ds.iter().map(|d| d.to_string()).collect();
            format!("算子表解析失败（{} 项降级）：{}", reasons.len(), reasons.join("; "))
        })?;
        let stream = gpu.stream();
        Ok(DeviceSession { stream: pin!(stream), table: pin!(table), gpu })
    }

    pub fn table(&self) -> &OpTable<'static> {
        &self.table
    }

    pub fn stream(&self) -> &Stream<'static> {
        &self.stream
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }
}
