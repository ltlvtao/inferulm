//! 单流（m=1）gemm 的**真实字节账**：按模型每步的实际矩阵集合铺满显存，
//! 逐组计时 → 每组 GB/s 与每发固定项。**同进程交错 A/B**（cuBLASLt vs
//! gemv_bf16），跨轮漂移由配对抵消（本机 D2D 标定曾见 695-2911 GB/s）。
//!
//! 动机：c=1 图档 ITL 11.49 vs vLLM 10.31（配对 −9%）。nsys 已定
//! gemm 占 10.55/11.79ms，但「同 8.4GB 权重为何慢」需要**冷态逐形状**
//! 读数——`WEIN_TUNE_DUMP` 的 best_ms 是 L2 热态（n=8192k=2560 报 17µs
//! = 2.4TB/s，不可能），不能用来算带宽。
//!
//! 做法：按 `WEIN_GEMM_BYTES` 普查出的每步发数分布（64/32/32/1/32/24/16/48）
//! 分配**互不相同**的权重缓冲（合计 8.400 GB），一轮扫完全部再到下一轮
//! ⇒ 每个缓冲都是冷的，与真步的访问序等价。
//!
//!   cargo run --release -p wein-kernel --example gemm_m1_sweep [轮数]
use wein_kernel::cuda::{BlasLt, Gpu};
use wein_kernel::ops::{GemvOp, OpTable};
use wein_platform::{Device, HardwareProfile};

/// (名, n, k, 每步发数) —— 来自 m=1 真实普查（20 步均值）
const MIX: &[(&str, u32, u32, usize)] = &[
    ("gate/up  9216x2560", 9216, 2560, 64),
    ("down     2560x9216", 2560, 9216, 32),
    ("qkv      8192x2560", 8192, 2560, 32),
    ("lm_head 248320x2560", 248320, 2560, 1),
    ("o(a)     2560x4096", 2560, 4096, 32),
    ("o(b)     4096x2560", 4096, 2560, 24),
    ("k/v      1024x2560", 1024, 2560, 16),
    ("a/b        32x2560", 32, 2560, 48),
];

fn main() {
    let sweeps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(5);
    let gpu = Gpu::probe().expect("GPU");
    let lt = BlasLt::open_tuned(&gpu, &[]).expect("libcublasLt");
    let profile = HardwareProfile::with_device(Device::Cuda).with_arch("sm_89");
    let t = OpTable::resolve(&gpu, &profile).expect("sm_89 画像");
    let stream = gpu.stream();

    let mut ws: Vec<Vec<wein_kernel::cuda::DevBuf>> = Vec::new();
    let mut total_bytes = 0f64;
    for &(_, n, k, cnt) in MIX {
        let mut v = Vec::with_capacity(cnt);
        for _ in 0..cnt {
            let b = gpu.alloc_bytes(n as usize * k as usize * 2);
            b.zero_on(&stream);
            v.push(b);
        }
        total_bytes += (n as f64 * k as f64 * 2.0) * cnt as f64;
        ws.push(v);
    }
    let kmax = MIX.iter().map(|x| x.2).max().unwrap();
    let nmax = MIX.iter().map(|x| x.1).max().unwrap();
    let a = gpu.alloc_bytes(kmax as usize * 2);
    let c = gpu.alloc_bytes(nmax as usize * 2);
    a.zero_on(&stream);
    c.zero_on(&stream);
    stream.sync();
    println!("权重 {:.3} GB｜{} 轮交错 A/B（同进程，逐轮配对）", total_bytes / 1e9, sweeps);

    let (e0, e1) = (gpu.event(), gpu.event());
    // 两组读数：lt[gi] / gemv[gi]，每轮各测一次（交错 ⇒ 同轮环境一致）
    let mut lt_ms: Vec<Vec<f64>> = vec![Vec::new(); MIX.len()];
    let mut gv_ms: Vec<Vec<f64>> = vec![Vec::new(); MIX.len()];
    for s in 0..(sweeps + 2) {
        for (gi, &(_, n, k, _)) in MIX.iter().enumerate() {
            e0.record_on(&stream);
            for w in &ws[gi] {
                lt.gemm_wt(&a, w, &c, 1, n, k, &stream);
            }
            e1.record_on(&stream);
            stream.sync();
            if s >= 2 {
                lt_ms[gi].push(e1.elapsed_ms(&e0) as f64);
            }
            e0.record_on(&stream);
            for w in &ws[gi] {
                t.gemv.run(w, &a, &c, n, k, &stream);
            }
            e1.record_on(&stream);
            stream.sync();
            if s >= 2 {
                gv_ms[gi].push(e1.elapsed_ms(&e0) as f64);
            }
        }
    }
    println!(
        "{:<22} {:>7} {:>8} {:>9} {:>9} {:>9} {:>7}",
        "shape", "发/步", "GB/步", "lt ms", "gemv ms", "gemv GB/s", "配对差"
    );
    let (mut sl, mut sg, mut sgb) = (0.0, 0.0, 0.0);
    for (gi, &(name, n, k, cnt)) in MIX.iter().enumerate() {
        lt_ms[gi].sort_by(|x, y| x.partial_cmp(y).unwrap());
        gv_ms[gi].sort_by(|x, y| x.partial_cmp(y).unwrap());
        let (l, g) = (lt_ms[gi][lt_ms[gi].len() / 2], gv_ms[gi][gv_ms[gi].len() / 2]);
        let gb = (n as f64 * k as f64 * 2.0) * cnt as f64 / 1e9;
        sl += l;
        sg += g;
        sgb += gb;
        println!(
            "{name:<22} {cnt:>7} {gb:>8.3} {l:>9.3} {g:>9.3} {:>9.0} {:>6.1}%",
            gb / (g / 1e3),
            (g - l) / l * 100.0
        );
    }
    println!(
        "组和 lt {sl:.2} ms / gemv {sg:.2} ms（{:.1}%）｜GB/s {:.0} → {:.0}｜理论@899 {:.2} ms",
        (sg - sl) / sl * 100.0,
        sgb / (sl / 1e3),
        sgb / (sg / 1e3),
        sgb / 899.0 * 1e3
    );
}
