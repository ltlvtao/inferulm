//! 算子注册表（02-architecture §4）。
//!
//! **启动时解析一次 → 具体类型，热路径零动态分发**：
//! - 槽位是具体 enum 字段（无 Box<dyn>、无字符串查表、无每步分配）
//! - trait（Op 家族）定义算子 ABI；OpTable 存具体类型——分支在 resolve
//!   后冻结，编译器可内联
//! - 解析失败 = Degrade 三元组列表（声明式降级，wein-platform §3）——
//!   启动期裁决，绝不带病运行也绝不隐式硬改
use crate::cuda::{DevBuf, Func, Gpu, Stream};
use wein_platform::{Device, Degrade, HardwareProfile};

/// rms_norm 的家族 trait（§4 "trait Op 家族"首个成员）。
/// trait 钉 ABI；热路径走 OpTable 的具体类型字段（不 dyn）
pub trait RmsNormOp {
    /// x/out: [rows, hidden] bf16 行主序；w: [hidden] bf16；eps: f32
    fn run(
        &self,
        x: &DevBuf,
        w: &DevBuf,
        out: &DevBuf,
        rows: u32,
        hidden: u32,
        eps: f32,
        stream: &Stream,
    );
    /// 解析结果标识（测试/日志；含实现档位）
    fn name(&self) -> &'static str;
}

/// rms_norm 的实现档位（同一 kernel、不同 launch 几何——W2 真 kernel
/// 落地时在此扩档并按 profile 选择，调用方代码不变）
pub enum RmsNormImpl<'a> {
    /// 每行一 block×1 线程——老架构保守档（< sm_80）
    Row { gpu: &'a Gpu, f: Func },
    /// block=256 承载行——sm_80+ 档（本机 sm_89 走这档）
    Blocked { gpu: &'a Gpu, f: Func, block: u32 },
}

/// 算子表：启动时 resolve 一次，全程只读
pub struct OpTable<'a> {
    pub rms_norm: RmsNormImpl<'a>,
    // M0 清单其余算子（rope/silu/varlen attention/量化/KV scatter）逐个入表
}

impl OpTable<'_> {
    /// 按 HardwareProfile 解析。设备族不匹配 → Degrade 列表（启动期裁决）。
    /// PTX 模块不显式卸载——算子表即进程生命周期，CUfunction 句柄常驻
    pub fn resolve<'a>(
        gpu: &'a Gpu,
        profile: &HardwareProfile,
    ) -> Result<OpTable<'a>, Vec<Degrade>> {
        if profile.device != Device::Cuda {
            return Err(vec![Degrade {
                key: "kernel.backend",
                reason: format!("画像设备为 {:?}，CUDA 算子表无法解析", profile.device),
                impact: "无可用 rms_norm 实现，拒绝带病启动——须显式选择正确后端".into(),
            }]);
        }
        let arch = profile.arch.clone().unwrap_or_else(|| gpu.arch().to_string());
        let (maj, _min) = parse_arch(&arch).ok_or_else(|| {
            vec![Degrade {
                key: "kernel.arch",
                reason: format!("arch 字串不可解析：{arch}"),
                impact: "无法裁决 launch 档位（row/blocked）".into(),
            }]
        })?;
        let ptx = std::fs::read_to_string(crate::ptx_path()).map_err(|e| {
            vec![Degrade {
                key: "kernel.ptx",
                reason: format!("PTX 产物读取失败：{e}"),
                impact: "rms_norm 无法加载（先构建 kernels 子 crate）".into(),
            }]
        })?;
        let f = gpu.load_module(&ptx).func("rms_norm");
        // 档位裁决：sm_80+ 走 blocked（block=256 承载行）；老架构保守 row 档
        let rms_norm = if maj >= 8 {
            RmsNormImpl::Blocked { gpu, f, block: 256 }
        } else {
            RmsNormImpl::Row { gpu, f }
        };
        Ok(OpTable { rms_norm })
    }
}

/// "sm_89" → (8, 9)；"sm_120" → (12, 0)——PTX 命名法 minor 恒一位，
/// 末位拆 minor 其余为 major；非法输入 → None
pub fn parse_arch(s: &str) -> Option<(u32, u32)> {
    let digits = s.strip_prefix("sm_")?;
    let mut chars = digits.chars();
    let min = chars.next_back()?.to_digit(10)?;
    let maj_str: String = chars.collect();
    if maj_str.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some((maj_str.parse().ok()?, min))
}

impl RmsNormOp for RmsNormImpl<'_> {
    fn run(
        &self,
        x: &DevBuf,
        w: &DevBuf,
        out: &DevBuf,
        rows: u32,
        hidden: u32,
        eps: f32,
        stream: &Stream,
    ) {
        // launch 几何分档，kernel 本体同一份（数值序不变——o5 对拍的依据）
        let (gpu, f, grid, block) = match self {
            RmsNormImpl::Row { gpu, f } => (*gpu, f, rows, 1),
            RmsNormImpl::Blocked { gpu, f, block } => {
                (*gpu, f, rows.div_ceil(*block), *block)
            }
        };
        // x/w/out 形参是 &DevBuf，解引用成 place 再取址（宏按 &$arg 取位）
        let args = crate::kernel_args![*x, *w, *out, hidden, eps, rows];
        gpu.launch(f, grid, block, &args, stream);
    }

    fn name(&self) -> &'static str {
        match self {
            RmsNormImpl::Row { .. } => "rms_norm/cuda-row",
            RmsNormImpl::Blocked { .. } => "rms_norm/cuda-blocked",
        }
    }
}
