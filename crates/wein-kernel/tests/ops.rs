//! 算子注册表验收（02-architecture §4）。
//!
//! §4 的核心主张：**启动时按 HardwareProfile 解析一次 → 具体类型单态化，
//! 热路径零动态分发**（无 Box<dyn>、无字符串查表）。首个入表算子 rms_norm
//! （M0 融合算子清单成员），golden vector 框架同步立起（§4 数值对齐：
//! cuda/asc 同名算子共享同一套 golden——asc 将来直接跑同一文件）。
//!
//! - o1 · 解析一次成表：本机（sm_89）rms_norm 槽位解析为 blocked 档，
//!        槽位是具体类型（编译期形状），arch 解析 sm_89 → (8,9)
//! - o2 · 无卡启动期 Err：返回 Degrade 三元组列表（声明式降级），非 panic
//! - o3 · 解析确定性：两次 resolve 同一 profile 选择一致
//! - o4 · rms_norm golden 位对齐：GPU 与 CPU 参考（独立代码路径）位位相等
//! - o5 · 双实现同 golden 位一致：合成 sm_70 强制 row 档——profile 真的在
//!        选实现，且两档在 golden 向量上位位相等（同名算子同 golden）
//! - o6 · golden 文件框架：缺失时生成落盘；从文件回读跑 GPU 仍位对齐
//!        （asc 接入时复用同一文件，§4）
//!
//! 数值对齐纪律（k6 同源）：bf16→f32 精确；乘/加/除/sqrt 全部正确舍入原语
//! （禁 rsqrt/MUFU 近似与 fma 收缩）；行内顺序累加两侧逐步同序。
use wein_kernel::cuda::Gpu;
use wein_kernel::ops::{parse_arch, OpTable, RmsNormOp};
use wein_kernel::reference;
use wein_platform::{Device, HardwareProfile};

fn gpu() -> Option<Gpu> {
    match Gpu::probe() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("SKIP（无可用 CUDA 设备：{e}）");
            None
        }
    }
}

fn profile_with_arch(arch: &str) -> HardwareProfile {
    HardwareProfile::with_device(Device::Cuda).with_arch(arch)
}

/// o1 · 解析一次成表（本机 sm_89 → blocked 档；槽位具体类型）
#[test]
fn o1_resolve_once_into_table() {
    let Some(gpu) = gpu() else { return };
    let profile = profile_with_arch(gpu.arch());
    let table = OpTable::resolve(&gpu, &profile).expect("sm_89 画像须解析成功");
    assert_eq!(table.rms_norm.name(), "rms_norm/cuda-blocked", "sm_80+ 应选 blocked 档");

    // arch 字串解析（注册表选择维度）
    assert_eq!(parse_arch("sm_89"), Some((8, 9)));
    assert_eq!(parse_arch("sm_70"), Some((7, 0)));
    assert_eq!(parse_arch("bogus"), None);
}

/// o2 · 无卡启动期 Err：声明式降级三元组，不是 panic
#[test]
fn o2_unsupported_device_degrades_explicitly() {
    let Some(gpu) = gpu() else { return };
    let r = OpTable::resolve(&gpu, &HardwareProfile::with_device(Device::None));
    let Err(degrades) = r else {
        panic!("无设备画像必须 Err（不许硬选）");
    };
    assert!(!degrades.is_empty());
    for d in &degrades {
        assert!(!d.key.is_empty(), "降级三元组缺键：{d:?}");
        assert!(!d.reason.is_empty(), "降级三元组缺原因：{d:?}");
        assert!(!d.impact.is_empty(), "降级三元组缺影响：{d:?}");
    }
}

/// o3 · 解析确定性：同一画像两次解析，选择一致
#[test]
fn o3_resolve_deterministic() {
    let Some(gpu) = gpu() else { return };
    let profile = profile_with_arch(gpu.arch());
    let t1 = OpTable::resolve(&gpu, &profile).expect("resolve 1");
    let t2 = OpTable::resolve(&gpu, &profile).expect("resolve 2");
    assert_eq!(t1.rms_norm.name(), t2.rms_norm.name());
}

/// o4 · rms_norm golden 位对齐（blocked 档，独立 CPU 参考实现对拍）
#[test]
fn o4_rms_norm_golden_bitwise() {
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let table = OpTable::resolve(&gpu, &profile_with_arch(gpu.arch())).expect("resolve");
    for case in golden_cases() {
        let expect = reference::rms_norm(&case.x, &case.w, case.rows, case.hidden, case.eps);
        let dx = gpu.alloc_from_slice(&case.x);
        let dw = gpu.alloc_from_slice(&case.w);
        let dout = gpu.alloc_bytes(case.out_len_bytes());
        table.rms_norm.run(&dx, &dw, &dout, case.rows as u32, case.hidden as u32, case.eps, &stream);
        stream.sync();
        let got: Vec<u16> = dout.read_to_vec();
        assert_eq!(got.len(), expect.len(), "{}", case.label);
        for i in 0..expect.len() {
            assert_eq!(got[i], expect[i], "{}[{i}] 位不对齐（GPU {:#06x} vs 参考 {:#06x}）",
                case.label, got[i], expect[i]);
        }
    }
}

/// o5 · 双实现同 golden 位一致：sm_70 强制 row 档——profile 真的在选实现，
/// 且两档在 golden 向量上位位相等（§4 同名算子共享 golden 的 cuda 侧兑现）
#[test]
fn o5_impls_agree_bitwise_on_golden() {
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let blocked = OpTable::resolve(&gpu, &profile_with_arch(gpu.arch())).expect("blocked");
    let row = OpTable::resolve(&gpu, &profile_with_arch("sm_70")).expect("row");
    assert_ne!(blocked.rms_norm.name(), row.rms_norm.name(), "两画像必须选不同档");

    for case in golden_cases() {
        let dx = gpu.alloc_from_slice(&case.x);
        let dw = gpu.alloc_from_slice(&case.w);
        let ob = gpu.alloc_bytes(case.out_len_bytes());
        blocked.rms_norm.run(&dx, &dw, &ob, case.rows as u32, case.hidden as u32, case.eps, &stream);
        let or = gpu.alloc_bytes(case.out_len_bytes());
        row.rms_norm.run(&dx, &dw, &or, case.rows as u32, case.hidden as u32, case.eps, &stream);
        stream.sync();
        let (b, r): (Vec<u16>, Vec<u16>) = (ob.read_to_vec(), or.read_to_vec());
        for i in 0..b.len() {
            assert_eq!(b[i], r[i], "{}[{i}] blocked/row 两档位不一致", case.label);
        }
    }
}

/// o6 · golden 文件框架：缺失生成落盘、回读重跑仍对齐（asc 复用同一文件）
#[test]
fn o6_golden_file_roundtrip() {
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let table = OpTable::resolve(&gpu, &profile_with_arch(gpu.arch())).expect("resolve");

    let path = golden_path();
    if !path.exists() {
        let file = golden_file(&golden_cases());
        std::fs::write(&path, serde_json::to_string_pretty(&file).unwrap()).unwrap();
        eprintln!("golden 首次生成：{}", path.display());
    }
    // 回读（schema 演化敏感：文件格式即 cuda/asc 契约）
    let text = std::fs::read_to_string(&path).unwrap();
    let file: GoldenFile = serde_json::from_str(&text).expect("golden 文件 schema");
    assert_eq!(file.op, "rms_norm");
    assert!(!file.cases.is_empty());
    assert_eq!(file.cases.len(), golden_cases().len(), "用例数漂移须显式重生成");

    for case in &file.cases {
        let expect = reference::rms_norm(&case.x, &case.w, case.rows, case.hidden, case.eps);
        assert_eq!(&expect, &case.out, "golden 文件与参考实现漂移（重生成须有意为之）");
        let dx = gpu.alloc_from_slice(&case.x);
        let dw = gpu.alloc_from_slice(&case.w);
        let dout = gpu.alloc_bytes(case.out.len() * 2);
        table.rms_norm.run(&dx, &dw, &dout, case.rows as u32, case.hidden as u32, case.eps, &stream);
        stream.sync();
        let got: Vec<u16> = dout.read_to_vec();
        assert_eq!(&got, &case.out, "文件回读后 GPU 对拍失位");
    }
}

// ───────────────────────── 用例与工具 ─────────────────────────

struct Case {
    label: &'static str,
    rows: usize,
    hidden: usize,
    eps: f32,
    x: Vec<u16>,
    w: Vec<u16>,
}

impl Case {
    fn out_len_bytes(&self) -> usize {
        self.rows * self.hidden * 2
    }
}

/// 两档用例：规整 8×64；33 行（非 block 边界）×256
fn golden_cases() -> Vec<Case> {
    vec![mk("8x64", 8, 64), mk("33x256", 33, 256)]
}

fn mk(label: &'static str, rows: usize, hidden: usize) -> Case {
    let n = rows * hidden;
    let x: Vec<u16> = (0..n as u64).map(|i| reference::f32_to_bf16(reference::splitmix_f32(i))).collect();
    let w: Vec<u16> = (0..hidden as u64)
        .map(|i| reference::f32_to_bf16(reference::splitmix_f32(i ^ 0x517c_c1b3) * 0.5 + 0.75))
        .collect();
    Case { label, rows, hidden, eps: 1e-6, x, w }
}

fn golden_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/rms_norm.json")
}

// ───── golden 文件契约（serde，dev 侧；asc 集成测试经 mod 复用同一 schema）─────

#[derive(serde::Serialize, serde::Deserialize)]
pub struct GoldenCase {
    pub rows: usize,
    pub hidden: usize,
    pub eps: f32,
    pub x: Vec<u16>,
    pub w: Vec<u16>,
    pub out: Vec<u16>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct GoldenFile {
    pub op: String,
    pub note: String,
    pub cases: Vec<GoldenCase>,
}

fn golden_file(cases: &[Case]) -> GoldenFile {
    GoldenFile {
        op: "rms_norm".into(),
        note: "cuda/asc 共享 golden（02-architecture §4）；out 由 wein-kernel::reference 生成".into(),
        cases: cases
            .iter()
            .map(|c| GoldenCase {
                rows: c.rows,
                hidden: c.hidden,
                eps: c.eps,
                x: c.x.clone(),
                w: c.w.clone(),
                out: reference::rms_norm(&c.x, &c.w, c.rows, c.hidden, c.eps),
            })
            .collect(),
    }
}
