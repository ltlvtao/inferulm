//! 图捕获验收（02-architecture §7：图档位 NONE→FULL_DECODE 的机制层）。
//!
//! 图走 Driver API：per-bucket capture；**回放读执行时内存**（参数在录制期
//! 烤进图节点的是设备指针——指向常驻缓冲，输入数据本身每步可换）；
//! cuGraphExecUpdate 即 updatable graph（**参数重绑而非重捕获**）。
//!
//! - gr1 · 捕获-实例化-回放 ≡ 直发位一致（同 kernel 序、同缓冲、同输入）
//! - gr2 · 重放读活输入：两次回放之间改写输入缓冲（upload_on）——
//!         图不是录制期快照，是执行期读内存的 kernel 序
//! - gr3 · exec update 参数重绑：同拓扑异参数重捕 → update 旧 exec →
//!         语义换新（免重实例化；update 合同：拓扑须同、参数可换）
//! - gr4 · cuBLASLt 入图：workspace 常驻 + algo 形状缓存命中 → 捕获安全；
//!         回放结果对 f64 参考容差对拍（g 系准则——图不改变数值合同）
//! - gr5 · 捕获中隐式同步守卫：sync/回读/分配在捕获区间内 debug 直接
//!         panic（ZoS forbid_cuda_sync 的 Rust 化——意外同步是图污染源）
use std::sync::{Mutex, MutexGuard, OnceLock};

use wein_kernel::cuda::Gpu;
use wein_kernel::ops::{OpTable, RmsNormOp, SiluOp};
use wein_kernel::reference;
use wein_platform::{Device, HardwareProfile};

/// 捕获串行锁：THREAD_LOCAL 捕获模式下，并行线程的隐式同步（legacy
/// stream）会击穿捕获态（906 → 捕获作废 901）——图测试全串行（引擎侧
/// 捕获本就在单线程任务里，测试同构）
fn guard() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

fn gpu() -> Option<Gpu> {
    match Gpu::probe() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("SKIP（无可用 CUDA 设备：{e}）");
            None
        }
    }
}

fn table(gpu: &Gpu) -> OpTable<'_> {
    let profile = HardwareProfile::with_device(Device::Cuda).with_arch(gpu.arch());
    OpTable::resolve(gpu, &profile).expect("算子表解析")
}

/// gr1 · 捕获→实例化→回放，输出与同序列直发**位一致**（图是录制，不是重写）
#[test]
fn gr1_replay_bitwise_matches_direct() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let (rows, hidden) = (4u32, 1024u32);
    let x: Vec<u16> = (0..rows as usize * hidden as usize)
        .map(|i| reference::f32_to_bf16(reference::splitmix_f32(i as u64 ^ 0xa1)))
        .collect();
    let w: Vec<u16> = (0..hidden as usize)
        .map(|i| reference::f32_to_bf16(reference::splitmix_f32(i as u64 ^ 0xb2)))
        .collect();

    // 直发侧（独立缓冲）
    let (dx, dw, dout1, dout2) = (
        gpu.alloc_from_slice(&x),
        gpu.alloc_from_slice(&w),
        gpu.alloc_bytes(rows as usize * hidden as usize * 2),
        gpu.alloc_bytes(rows as usize * hidden as usize * 2),
    );
    t.rms_norm.run(&dx, &dw, &dout1, rows, hidden, 1e-6, &stream);
    t.silu.run(&dout1, &dout2, rows * hidden, &stream);
    stream.sync();
    let direct2 = dout2.read_to_vec::<u16>();

    // 图侧（另一组缓冲，同输入同形状——图录 kernel 序）
    let (gx, gw, gout1, gout2) = (
        gpu.alloc_from_slice(&x),
        gpu.alloc_from_slice(&w),
        gpu.alloc_bytes(rows as usize * hidden as usize * 2),
        gpu.alloc_bytes(rows as usize * hidden as usize * 2),
    );
    stream.begin_capture();
    t.rms_norm.run(&gx, &gw, &gout1, rows, hidden, 1e-6, &stream);
    t.silu.run(&gout1, &gout2, rows * hidden, &stream);
    let graph = stream.end_capture();
    let exec = graph.instantiate();
    exec.launch(&stream);
    stream.sync();
    assert_eq!(gout2.read_to_vec::<u16>(), direct2, "图回放与直发须位一致");
}

/// gr2 · 重放读活输入：输入缓冲在两次回放之间换内容，输出跟着换——
/// 设备指针烤进图、数据留在缓冲里（桶共享输入存储的机制前提）
#[test]
fn gr2_replay_reads_live_inputs() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let (rows, hidden) = (2u32, 512u32);
    let mk = |seed: u64| -> Vec<u16> {
        (0..rows as usize * hidden as usize)
            .map(|i| reference::f32_to_bf16(reference::splitmix_f32(i as u64 ^ seed)))
            .collect()
    };
    let w: Vec<u16> = (0..hidden as usize)
        .map(|i| reference::f32_to_bf16(reference::splitmix_f32(i as u64 ^ 0x77)))
        .collect();

    let (dx, dw, dout) = (
        gpu.alloc_from_slice(&mk(0x11)),
        gpu.alloc_from_slice(&w),
        gpu.alloc_bytes(rows as usize * hidden as usize * 2),
    );
    stream.begin_capture();
    t.rms_norm.run(&dx, &dw, &dout, rows, hidden, 1e-6, &stream);
    let exec = stream.end_capture().instantiate();

    // 两轮：改输入 → 回放 → 与独立直发对拍
    for seed in [0x11u64, 0x22] {
        let x = mk(seed);
        dx.upload_on(&x, &stream);
        exec.launch(&stream);
        stream.sync();
        let got = dout.read_to_vec::<u16>();

        let (ex, ew, eo) = (
            gpu.alloc_from_slice(&x),
            gpu.alloc_from_slice(&w),
            gpu.alloc_bytes(rows as usize * hidden as usize * 2),
        );
        t.rms_norm.run(&ex, &ew, &eo, rows, hidden, 1e-6, &stream);
        stream.sync();
        assert_eq!(got, eo.read_to_vec::<u16>(), "seed {seed:#x}：重放须读活输入");
    }
}

/// gr3 · cuGraphExecUpdate 参数重绑：同拓扑异参数（输出换缓冲）重捕 →
/// update 旧 exec → 语义换新且旧缓冲不被再写——免重实例化的重绑路径
/// （update 合同：拓扑须同，烤进节点的参数可换）
#[test]
fn gr3_exec_update_rebinds_params() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let (rows, hidden) = (2u32, 768u32);
    let mk = |seed: u64, n: usize| -> Vec<u16> {
        (0..n).map(|i| reference::f32_to_bf16(reference::splitmix_f32(i as u64 ^ seed))).collect()
    };
    let n = rows as usize * hidden as usize;
    let (dx, dw) = (
        gpu.alloc_from_slice(&mk(0x31, n)),
        gpu.alloc_from_slice(&mk(0x32, hidden as usize)),
    );
    // v1 输出对 / v2 输出对（update 重绑的目标）
    let (o1a, o1b) = (gpu.alloc_bytes(n * 2), gpu.alloc_bytes(n * 2));
    let (o2a, o2b) = (gpu.alloc_bytes(n * 2), gpu.alloc_bytes(n * 2));

    stream.begin_capture();
    t.rms_norm.run(&dx, &dw, &o1a, rows, hidden, 1e-6, &stream);
    t.silu.run(&o1a, &o1b, rows * hidden, &stream);
    let exec = stream.end_capture().instantiate();
    exec.launch(&stream);
    stream.sync();
    let v1_out = o1b.read_to_vec::<u16>();

    // v2：同拓扑，输出重绑到另一对缓冲
    stream.begin_capture();
    t.rms_norm.run(&dx, &dw, &o2a, rows, hidden, 1e-6, &stream);
    t.silu.run(&o2a, &o2b, rows * hidden, &stream);
    let g2 = stream.end_capture();
    exec.update(&g2);
    o1b.zero_on(&stream); // v1 旧输出清零——update 后不得再被写
    exec.launch(&stream);
    stream.sync();

    // 直发对照（v2 语义）
    let (ea, ew, eo1, eo2) = (
        gpu.alloc_from_slice(&dx.read_to_vec::<u16>()),
        gpu.alloc_from_slice(&dw.read_to_vec::<u16>()),
        gpu.alloc_bytes(n * 2),
        gpu.alloc_bytes(n * 2),
    );
    t.rms_norm.run(&ea, &ew, &eo1, rows, hidden, 1e-6, &stream);
    t.silu.run(&eo1, &eo2, rows * hidden, &stream);
    stream.sync();
    assert_eq!(o2b.read_to_vec::<u16>(), eo2.read_to_vec::<u16>(), "update 后须是 v2 参数语义");
    assert!(
        o1b.read_to_vec::<u16>().iter().all(|&b| b == 0),
        "v1 旧输出缓冲不得再被 update 后的图写入"
    );
    // v1 输出在重绑前确实非零（上面的全零断言不是平凡真）
    assert!(v1_out.iter().any(|&b| b != 0), "v1 输出须非零（重绑前确实写过）");
}

/// gr4 · cuBLASLt 入图：形状先预热（algo 落缓存）→ 捕获 → 换输入回放 →
/// f64 参考容差对拍（厂商库算子的图路径数值合同不变，g 系准则）
#[test]
fn gr4_cublaslt_capturable() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let (m, n, k) = (64u32, 128u32, 256u32);
    let mk = |seed: u64, n: usize| -> Vec<u16> {
        (0..n).map(|i| reference::f32_to_bf16(reference::splitmix_f32(i as u64 ^ seed))).collect()
    };
    let a0 = mk(0x41, m as usize * k as usize);
    let w = mk(0x42, n as usize * k as usize);
    let (da, dw, dc) = (
        gpu.alloc_from_slice(&a0),
        gpu.alloc_from_slice(&w),
        gpu.alloc_bytes(m as usize * n as usize * 2),
    );
    // 预热：同形状直发一次（desc/layout/heuristic 解析进缓存）
    use wein_kernel::ops::GemmOp;
    t.gemm.run_wt(&da, &dw, &dc, m, n, k, &stream);
    stream.sync();

    stream.begin_capture();
    t.gemm.run_wt(&da, &dw, &dc, m, n, k, &stream);
    let exec = stream.end_capture().instantiate();

    // 换输入回放：容差对拍（图路径 ≠ 数值捷径）
    let a1 = mk(0x43, m as usize * k as usize);
    da.upload_on(&a1, &stream);
    exec.launch(&stream);
    stream.sync();
    let got = dc.read_to_vec::<u16>();
    let expect = reference::gemm_wt(&a1, &w, m as usize, n as usize, k as usize);
    let mut worst = 0.0f64;
    for i in 0..expect.len() {
        let (g, r) = (reference::bf16_to_f32(got[i]) as f64, expect[i]);
        let tol = (2f64.powi(-7) * r.abs()).max(2f64.powi(-11));
        let err = (g - r).abs();
        assert!(err <= tol, "gemm[{i}] 偏差 {err} 超容差 {tol}（got {g} vs ref {r}）");
        worst = worst.max(err / tol);
    }
    eprintln!("gr4 图内 cuBLASLt 容差占用 {:.0}%", worst * 100.0);
}

/// gr5 · 捕获区间内隐式同步守卫（debug 构建）：sync / 回读 / 分配在
/// begin..end 之间直接 panic——意外同步是图污染的头号来源（§4 安全规则）
#[test]
#[cfg(debug_assertions)]
fn gr5_capture_forbids_implicit_sync() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let (rows, hidden) = (2u32, 256u32);
    let (dx, dw, dout) = (
        gpu.alloc_from_slice(&vec![0u16; rows as usize * hidden as usize]),
        gpu.alloc_from_slice(&vec![0u16; hidden as usize]),
        gpu.alloc_bytes(rows as usize * hidden as usize * 2),
    );
    stream.begin_capture();
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {})); // 静音预期 panic
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        stream.sync(); // 捕获中同步：必须炸
    }));
    let r2 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = dout.read_to_vec::<u16>(); // 捕获中回读：必须炸
    }));
    let r3 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _b = gpu.alloc_bytes(64); // 捕获中分配：必须炸
    }));
    std::panic::set_hook(prev);
    assert!(r.is_err(), "捕获中 sync 须 debug panic");
    assert!(r2.is_err(), "捕获中回读须 debug panic");
    assert!(r3.is_err(), "捕获中分配须 debug panic");
    // end_capture 复位守卫后，同步恢复合法（区间外不受限）
    let graph = stream.end_capture();
    drop(graph);
    stream.sync();
    let _ = t.rms_norm.name(); // table 保活
    let _ = (&dx, &dw);
}
