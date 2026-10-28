//! cuBLASLt 最小 FFI 验收——GEMM 入表（§4 标准算子走厂商库）。
//!
//! 与自控 kernel 的位对齐验收（o 系列）不同：GEMM 累加序由厂商库决定
//! （tensor core 分块/split-k），位对齐**不可达也不该求**。对齐标准是
//! 容差：f64 参考 ± 1 bf16 ulp——f32 累加误差（~K·2⁻²⁴）≪ bf16 分辨率
//! （2⁻⁸），只有贴舍入边界的元素会翻转最后一位。golden 框架自此有
//! 两种模式：**自控算子位对齐 / 厂商库算子容差**（§4 数值对齐的完整
//! 陈述）。
//!
//! - g1 · dlopen libcublasLt + handle + 版本自洽（≥ 11.0）
//! - g2 · bf16 GEMM 精度（128³）：全元素 |got − ref64| ≤ max(2⁻⁷·|ref|, 2⁻¹¹)
//! - g3 · 变 M 形状（decode M=1 / prefill M=33；K=N=512）同准则
//! - g4 · 同参重跑位确定（同 algo 同输入 → 输出位位相同）
//! - g5 · gemm 槽位入表：name 可审 + 无设备画像仍 Err（o2 语义延续）
//! - g6 · 吞吐 canary（2048³ bf16）：TFLOPS 下限门禁——防退化到
//!        非 tensor core 路径（canary 不是基准：单跑下限，见噪声纪律）
use wein_kernel::cuda::{BlasLt, Gpu};
use wein_kernel::ops::{GemmOp, OpTable};
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

/// 容差判据：|f32(got) − ref_f64| ≤ max(2⁻⁷·|ref|, 2⁻¹¹)。
/// 2⁻⁷·|ref| ≈ 1 bf16 ulp（尾数 8 位）；绝对底 2⁻¹¹ 兜近零
fn assert_close(label: &str, got: &[u16], expect: &[f64]) {
    assert_eq!(got.len(), expect.len(), "{label} 长度");
    let mut worst = 0.0f64;
    for i in 0..expect.len() {
        let (g, r) = (reference::bf16_to_f32(got[i]) as f64, expect[i]);
        let tol = (2f64.powi(-7) * r.abs()).max(2f64.powi(-11));
        let err = (g - r).abs();
        assert!(err <= tol, "{label}[{i}] 偏差 {err} 超容差 {tol}（got {g} vs ref {r}）");
        worst = worst.max(err / tol);
    }
    eprintln!("{label}: 最差元素用到容差的 {:.0}%", worst * 100.0);
}

fn mk_mat(seed: u64, r: usize, c: usize) -> Vec<u16> {
    (0..(r * c) as u64)
        .map(|i| reference::f32_to_bf16(reference::splitmix_f32(i ^ seed)))
        .collect()
}

/// g1 · 库探测：dlopen → handle → 版本（记录在案，运行环境可审计）
#[test]
fn g1_cublaslt_open_handle_version() {
    let Some(gpu) = gpu() else { return };
    let lt = BlasLt::open(&gpu).expect("libcublasLt 须可打开（vendor 路径见 ops/mod.rs 注释）");
    let v = lt.version();
    assert!(v >= 110000, "cublasLt 版本过低：{v}");
    eprintln!("cublasLt 版本 {v}");
}

/// g2 · bf16 GEMM 精度（128³，方阵基线）
#[test]
fn g2_gemm_bf16_accuracy() {
    let Some(gpu) = gpu() else { return };
    let (stream, lt) = (gpu.stream(), BlasLt::open(&gpu).expect("open"));
    let (m, n, k) = (128usize, 128usize, 128usize);
    let (a, b) = (mk_mat(0, m, k), mk_mat(0x517c_c1b3, k, n));
    let expect = reference::gemm(&a, &b, m, n, k);
    let (da, db, dc) = (
        gpu.alloc_from_slice(&a),
        gpu.alloc_from_slice(&b),
        gpu.alloc_bytes(m * n * 2),
    );
    lt.gemm(&da, &db, &dc, m as u32, n as u32, k as u32, &stream);
    stream.sync();
    let got: Vec<u16> = dc.read_to_vec();
    assert_close("gemm 128³", &got, &expect);
}

/// g3 · 变 M 形状：decode（M=1）与 prefill（M=33）——M 维不齐不影响准则
#[test]
fn g3_gemm_varlen_shapes() {
    let Some(gpu) = gpu() else { return };
    let (stream, lt) = (gpu.stream(), BlasLt::open(&gpu).expect("open"));
    for m in [1usize, 33] {
        let (n, k) = (512usize, 512usize);
        let (a, b) = (mk_mat(0, m, k), mk_mat(0x9e37_79b9, k, n));
        let expect = reference::gemm(&a, &b, m, n, k);
        let (da, db, dc) = (
            gpu.alloc_from_slice(&a),
            gpu.alloc_from_slice(&b),
            gpu.alloc_bytes(m * n * 2),
        );
        lt.gemm(&da, &db, &dc, m as u32, n as u32, k as u32, &stream);
        stream.sync();
        assert_close(&format!("gemm {m}x{n}x{k}"), &dc.read_to_vec(), &expect);
    }
}

/// g4 · 同参重跑位确定：同 algo + 同输入 → 输出位位相同（不可复现的
/// 算法选择/原子路径会在这里现形）
#[test]
fn g4_gemm_deterministic_rerun() {
    let Some(gpu) = gpu() else { return };
    let (stream, lt) = (gpu.stream(), BlasLt::open(&gpu).expect("open"));
    let (m, n, k) = (33usize, 256usize, 512usize);
    let (a, b) = (mk_mat(0, m, k), mk_mat(0xbf58_476d, k, n));
    let (da, db, dc) = (
        gpu.alloc_from_slice(&a),
        gpu.alloc_from_slice(&b),
        gpu.alloc_bytes(m * n * 2),
    );
    lt.gemm(&da, &db, &dc, m as u32, n as u32, k as u32, &stream);
    stream.sync();
    let run1: Vec<u16> = dc.read_to_vec();
    lt.gemm(&da, &db, &dc, m as u32, n as u32, k as u32, &stream);
    stream.sync();
    let run2: Vec<u16> = dc.read_to_vec();
    assert_eq!(run1, run2, "同参重跑必须位位相同");
}

/// g5 · gemm 槽位入表：OpTable 路径可跑 + 名字可审；无设备画像 Err
///（o2 语义在全表维度上延续——backend 裁决在 gemm 解析之前）
#[test]
fn g5_gemm_slot_in_table() {
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let profile = HardwareProfile::with_device(Device::Cuda).with_arch("sm_89");
    let t = OpTable::resolve(&gpu, &profile).expect("resolve");
    assert_eq!(t.gemm.name(), "gemm/cuda-lt");

    let (m, n, k) = (8usize, 128usize, 256usize);
    let (a, b) = (mk_mat(0, m, k), mk_mat(0xdead_beef, k, n));
    let expect = reference::gemm(&a, &b, m, n, k);
    let (da, db, dc) = (
        gpu.alloc_from_slice(&a),
        gpu.alloc_from_slice(&b),
        gpu.alloc_bytes(m * n * 2),
    );
    t.gemm.run(&da, &db, &dc, m as u32, n as u32, k as u32, &stream);
    stream.sync();
    assert_close("gemm 经 OpTable", &dc.read_to_vec(), &expect);

    // 无设备画像：整表 Err（backend 先于 gemm 裁决）
    let r = OpTable::resolve(&gpu, &HardwareProfile::with_device(Device::None));
    let Err(degrades) = r else { panic!("无设备画像必须 Err") };
    assert!(degrades.iter().any(|d| d.key == "kernel.backend"));
}

/// g6 · 吞吐 canary：2048³ bf16，TFLOPS 下限 10（RTX 4090 TC 峰值 ~165；
/// 下限防"退化到非 TC 路径/错走 workspace 溢出回退"这类结构性回归，
/// 不是性能基准——基准进 battery 噪声纪律管）
#[test]
fn g6_gemm_throughput_canary() {
    let Some(gpu) = gpu() else { return };
    let (stream, lt) = (gpu.stream(), BlasLt::open(&gpu).expect("open"));
    let (m, n, k) = (2048usize, 2048usize, 2048usize);
    let (a, b) = (mk_mat(0, m, k), mk_mat(0x517c_c1b3, k, n));
    let (da, db, dc) = (
        gpu.alloc_from_slice(&a),
        gpu.alloc_from_slice(&b),
        gpu.alloc_bytes(m * n * 2),
    );
    // 保活到计时结束（下划线前缀变量持有所有权，不能用 `let _ =`——立即析构）
    let (_da, _db, _dc) = (&da, &db, &dc);
    for _ in 0..3 {
        lt.gemm(&da, &db, &dc, m as u32, n as u32, k as u32, &stream);
    }
    stream.sync();
    let reps = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        lt.gemm(&da, &db, &dc, m as u32, n as u32, k as u32, &stream);
    }
    stream.sync();
    let secs = t0.elapsed().as_secs_f64();
    let tflops = 2.0 * (m * n * k) as f64 * reps as f64 / secs / 1e12;
    eprintln!("gemm 2048³ bf16: {tflops:.1} TFLOPS（{reps} 次均值 {secs:.3}s）");
    assert!(tflops >= 10.0, "吞吐 {tflops:.1} TFLOPS 低于下限 10——疑似未走 tensor core 路径");
}
