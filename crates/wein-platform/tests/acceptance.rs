//! 平台层验收（02-architecture §3 三层能力模型）。
//!
//! pf1 · 身份层零依赖探测：PCI vendor 扫描（无 vendor SDK），结果稳定；
//!       本机（有 0x10de 设备）必须识别为 Cuda
//! pf2 · HardwareProfile 形状：supports() 按设备族回答、block_size 平台注入
//!       （CUDA 16 / 昇腾 128 / None 全 false）
//! pf3 · 声明式降级三元组：完整陈述（键/原因/影响），Display 可读——
//!       这是"不隐式硬改"的类型化载体（o2 在 kernel 侧消费它）
use wein_platform::{Capability, Degrade, Device, HardwareProfile};

#[test]
fn pf1_pci_probe_zero_dep_deterministic() {
    let d1 = Device::probe();
    let d2 = Device::probe();
    assert_eq!(d1, d2, "探测必须确定（两次一致）");

    // 本机有 NVIDIA PCI 设备（vendor 0x10de）则必须识别为 Cuda——
    // 零依赖探测的最小真值校验（读 sysfs，不引入任何 SDK）
    let has_nvidia = std::fs::read_dir("/sys/bus/pci/devices")
        .map(|rd| {
            rd.flatten().any(|e| {
                std::fs::read_to_string(e.path().join("vendor"))
                    .map(|v| v.trim() == "0x10de")
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);
    if has_nvidia {
        assert_eq!(d1, Device::Cuda, "有 0x10de 设备却未识别为 Cuda");
    }
    // 探测到的设备必属已知集合（不猜型号）
    assert!(matches!(d1, Device::Cuda | Device::Ascend | Device::None));
}

#[test]
fn pf2_hardware_profile_capability_shape() {
    // CUDA：M0 全能力 + block_size 16（§13-3：16/32 待 W2 attention tile 反推）
    let cuda = HardwareProfile::with_device(Device::Cuda);
    assert!(cuda.supports(Capability::PagedKv));
    assert!(cuda.supports(Capability::PrefixCache));
    assert!(cuda.supports(Capability::GraphCapture));
    assert!(cuda.supports(Capability::W8A8));
    assert_eq!(cuda.block_size, 16);
    assert_eq!(cuda.arch, None, "arch 是驱动侧信息，探测期未知");

    // 昇腾：M1 才接入——图捕获此阶段为 false；块大小 128（§5）
    let asc = HardwareProfile::with_device(Device::Ascend);
    assert!(asc.supports(Capability::PagedKv));
    assert!(!asc.supports(Capability::GraphCapture));
    assert_eq!(asc.block_size, 128);

    // 无设备：能力全 false（宁可全拒，不猜）
    let none = HardwareProfile::with_device(Device::None);
    assert!(!none.supports(Capability::PagedKv));
    assert!(!none.supports(Capability::W8A8));

    // detect 走真实探测且与 probe 一致
    let detected = HardwareProfile::detect();
    assert_eq!(detected.device, Device::probe());
}

#[test]
fn pf3_degrade_is_explicit_triple() {
    let d = Degrade {
        key: "kernel.backend",
        reason: "无 CUDA 设备".into(),
        impact: "算子表无法解析".into(),
    };
    let s = d.to_string();
    assert!(s.contains("kernel.backend"), "Display 须含键：{s}");
    assert!(s.contains("无 CUDA 设备"), "Display 须含原因：{s}");
    assert!(s.contains("算子表无法解析"), "Display 须含影响：{s}");
}
