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
//! - g8 · 每形状 algo 钉（autotune，受控路径）：调用方声明稳态 m 集
//!        （`open_tuned`），集内形状 heuristic 前 4 候选逐个实测（1 预热 +
//!        事件对 3 计时）、滞回择优后缓存 + 进程级钉表共享；集外瞬态形状
//!        （服务 ramp/drain 的 m 扫描）只钉 heuristic 免调优债——无门控
//!        实测债：满负荷 tps 2827→1678、ttft ×4。滞回/钉表保跨会话选型
//!        一致（qg2 零漂移合同的 gemm 侧前提）
use wein_kernel::cuda::{BlasLt, Gpu};
// g8 走 open_tuned（调优 m 集 = [100]）——autotune 是受控路径：调用方
// 声明稳态 m 集，瞬态形状（服务 ramp/drain 扫描）只钉 heuristic 免调优债
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

/// g8 · 每形状 algo 钉（autotune）：首 call 用**真实缓冲**逐候选实测择优
/// （事件对在调用者流上——前序工作排队在外，段计量纯），胜者连同
/// desc/layouts 缓存，其后纯 launch。验收四钉：
/// ① 每形状恰调优一次（第二套缓冲命中缓存）；
/// ② 择优不劣于 heuristic 首选（同测同缓冲的配对比较，非绝对门）；
/// ③ 胜者结果仍在容差合同内（g2 准则——候选换人不换合同）；
/// ④ 钉住的选型跨地址位确定（同 algo 同输入 → 位位相同，g4 的跨缓冲集版）
#[test]
fn g8_gemm_algo_pinned_autotune() {
    let Some(gpu) = gpu() else { return };
    let (stream, lt) = (gpu.stream(), BlasLt::open_tuned(&gpu, &[100]).expect("open"));
    // m=100 满负荷 decode 的线性层投影稳态形状（n×k 与 exec 权重表同）
    let shapes: [(u32, u32, u32); 6] = [
        (100, 8192, 2560), // 线性层 qkv
        (100, 4096, 2560), // z
        (100, 32, 2560),   // a/b
        (100, 2560, 4096), // out
        (100, 9728, 2560), // MLP wg/wu
        (100, 2560, 9728), // MLP wd
    ];
    for &(m, n, k) in &shapes {
        let (a, w) = (mk_mat(0x21, m as usize, k as usize), mk_mat(0x43, n as usize, k as usize));
        let expect = reference::gemm_wt(&a, &w, m as usize, n as usize, k as usize);
        // 两套并存缓冲（地址必不同）各跑一次：首跑调优、次跑命中缓存
        let mk_set = || {
            (
                gpu.alloc_from_slice(&a),
                gpu.alloc_from_slice(&w),
                gpu.alloc_bytes(m as usize * n as usize * 2),
            )
        };
        let (s0, s1) = (mk_set(), mk_set());
        let run = |set: &(_, _, _)| -> Vec<u16> {
            lt.gemm_wt(&set.0, &set.1, &set.2, m, n, k, &stream);
            stream.sync();
            set.2.read_to_vec()
        };
        let (out0, out1) = (run(&s0), run(&s1));
        assert_close(&format!("gemm_wt {m}x{n}x{k}（tuned）"), &out0, &expect);
        assert_eq!(out0, out1, "钉住的 algo 跨地址须位确定");
    }
    // 调优审计面：每形状恰一条记录 + 择优不劣于首选（配对比较）
    let report = lt.tune_report();
    for &(m, n, k) in &shapes {
        let recs: Vec<_> = report
            .iter()
            .filter(|r| r.m == m && r.n == n && r.k == k && r.wt)
            .collect();
        assert_eq!(recs.len(), 1, "形状 {m}x{n}x{k} 须恰调优一次（实得 {} 条）", recs.len());
        let r = recs[0];
        assert!(r.candidates >= 1, "至少一个可用候选");
        assert!(r.best_ms.is_finite(), "胜者须有实测（capturing 回退才允许 0）");
        assert!(
            r.best_ms <= r.first_ms * 1.02 + 1e-6,
            "择优 {:.4}ms 劣于首选 {:.4}ms（{m}x{n}x{k}）",
            r.best_ms,
            r.first_ms
        );
        let gain = (r.first_ms - r.best_ms) / r.first_ms * 100.0;
        eprintln!(
            "g8 tune {m}x{n}x{k}: 候选 {} 钉 #{chosen}（实测最优 #{argmin}）{best:.3}ms vs 首选 {first:.3}ms（{gain:+.1}%）",
            r.candidates, chosen = r.chosen, argmin = r.argmin, best = r.best_ms, first = r.first_ms
        );
    }
}
