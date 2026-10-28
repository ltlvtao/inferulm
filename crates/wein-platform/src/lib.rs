//! wein-platform：平台层（02-architecture §3）——业务代码唯一查询口。
//!
//! 三层能力模型（vllm-ascend 原样落地）：
//! 1. 身份层：`Device` + 零依赖探测（PCI vendor 扫描，无任何 vendor SDK）
//! 2. build 层：compile-time feature（cuda/asc）——M0 暂未启用
//! 3. `HardwareProfile`（不可变）：`supports(Capability)` + block_size 等参数
//!
//! 声明式降级：不支持项 = `(改哪个键, 原因, 影响)` 三元组列表由**调用方**消费，
//! 本层绝不隐式硬改配置（vllm-ascend `_fix_incompatible_config` 教训）。
//! 注意 arch（sm_89 之类）来自驱动而非 PCI——kernel 层持有 Gpu 时补齐
//! （`HardwareProfile::with_arch`）。

/// 设备身份（探测结果；enum 而非型号字符串——业务禁止按型号分支，§0-4）
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    /// NVIDIA GPU（vendor 0x10de）
    Cuda,
    /// 华为昇腾 NPU（vendor 0x19e5；M1 接入）
    Ascend,
    /// 未探测到已知加速设备
    None,
}

impl Device {
    /// PCI vendor 零依赖探测（/sys/bus/pci）。不 panic——探测失败即 None。
    /// 收集后裁决而非首见即返回：遍历序不定，flags 保证确定性
    pub fn probe() -> Device {
        let (mut cuda, mut ascend) = (false, false);
        if let Ok(rd) = std::fs::read_dir("/sys/bus/pci/devices") {
            for e in rd.flatten() {
                let Ok(v) = std::fs::read_to_string(e.path().join("vendor")) else { continue };
                match v.trim() {
                    "0x10de" => cuda = true,
                    "0x19e5" => ascend = true,
                    _ => {}
                }
            }
        }
        if cuda {
            Device::Cuda
        } else if ascend {
            Device::Ascend
        } else {
            Device::None
        }
    }
}

/// 能力项（清单随 M0→M1 生长；业务只准查 supports，禁止型号分支）
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Capability {
    /// paged KV 块式显存管理
    PagedKv,
    /// radix 前缀缓存
    PrefixCache,
    /// CUDA Graph / 昇腾图捕获（M0 只到 NONE 档也算 true——指硬件可捕获）
    GraphCapture,
    /// W8A8 权重激活量化
    W8A8,
}

/// 不可变硬件画像：启动时构造一次，全程只读
#[derive(Clone, Debug)]
pub struct HardwareProfile {
    pub device: Device,
    /// 架构字串（"sm_89"）——驱动侧信息，kernel 层补齐；None = 未知
    pub arch: Option<String>,
    /// KV 块大小（平台参数，02 §5：CUDA 16/32 待 W2 反推——先 16；昇腾 128）
    pub block_size: u32,
}

impl HardwareProfile {
    pub fn detect() -> Self {
        Self::with_device(Device::probe())
    }

    /// 合成画像（测试/旁路用；不探测）
    pub fn with_device(device: Device) -> Self {
        let block_size = match device {
            Device::Cuda => 16,    // §13-3：16/32 待 W2 attention tile 反推，先 16
            Device::Ascend => 128, // §5 昇腾块大小
            Device::None => 16,    // 无设备时占位——supports 全 false，不会被消费
        };
        Self { device, arch: None, block_size }
    }

    /// 补驱动侧 arch（kernel 层持 Gpu 后回填）
    pub fn with_arch(self, arch: &str) -> Self {
        Self { arch: Some(arch.into()), ..self }
    }

    /// 能力裁决只按设备族（enum），绝不按型号字符串（§0-4）
    pub fn supports(&self, c: Capability) -> bool {
        match (self.device, c) {
            (Device::Cuda, _) => true,
            (Device::Ascend, Capability::GraphCapture) => false, // M1 接入后再开
            (Device::Ascend, _) => true,
            (Device::None, _) => false,
        }
    }
}

/// 声明式降级三元组：不支持项的完整陈述，启动期交调用方裁决
#[derive(Clone, Debug)]
pub struct Degrade {
    /// 该改哪个配置键
    pub key: &'static str,
    pub reason: String,
    pub impact: String,
}

impl std::fmt::Display for Degrade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "降级 [{}]：{}（影响：{}）", self.key, self.reason, self.impact)
    }
}
