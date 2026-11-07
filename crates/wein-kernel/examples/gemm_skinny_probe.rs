//! gemm_skinny 单形状微基准：同一题面上 cuBLASLt（`run_wt` 且 WEIN_SKINNY=0）
//! vs 自写 kernel（直接发 `gemm_skinny`，与 `run_wt` 的 skinny 分支同参）。
//! 报 per-launch ms 与**权重字节** GB/s（权重流是满载的真账）。
//!   cargo run --release -p wein-kernel --example gemm_skinny_probe -- [iters]
use wein_kernel::cuda::Gpu;
use wein_kernel::ops::{GemmOp, OpTable};
use wein_kernel::reference;
use wein_platform::{Device, HardwareProfile};

fn bf16_vec(seed_hi: u64, n: usize, scale: f32) -> Vec<u16> {
    (0..n as u64)
        .map(|i| {
            let z = i.wrapping_mul(6364136223846793005).wrapping_add(seed_hi);
            let f = ((z >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * scale;
            reference::f32_to_bf16(f)
        })
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let iters: usize = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(20);
    let mode: String = a.get(2).cloned().unwrap_or_else(|| "both".into());
    let gpu = Gpu::probe().expect("CUDA");
    let ptx = std::fs::read_to_string(wein_kernel::ptx_path()).expect("PTX");
    let m = gpu.load_module(&ptx);
    let s = gpu.stream();
    let sk = m.func("gemm_skinny");
    let profile = HardwareProfile::with_device(Device::Cuda).with_arch("sm_89");
    // sk 单跑时不建算子表（排除"双模块/巨量缓冲"对照污染）
    let t = if mode == "sk" { None } else { Some(OpTable::resolve(&gpu, &profile).expect("sm_89 画像")) };

    // 带宽标定（读口径：src→dst 拷贝的读+写折半前记全量）
    let (cap, bwi) = (256usize << 20, 10usize);
    let (bsrc, bdst) = (gpu.alloc_bytes(cap), gpu.alloc_bytes(cap));
    for _ in 0..3 {
        bdst.copy_range_on(0, &bsrc, 0, cap, &s);
    }
    s.sync();
    let (eb0, eb1) = (gpu.event(), gpu.event());
    eb0.record_on(&s);
    for _ in 0..bwi {
        bdst.copy_range_on(0, &bsrc, 0, cap, &s);
    }
    eb1.record_on(&s);
    s.sync();
    let bw = (cap * 2 * bwi) as f64 / (eb1.elapsed_ms(&eb0) as f64 * 1e-3) / 1e9;
    eprintln!("D2D 带宽标定：{bw:.0} GB/s（读+写合计）");

    // 真步面形状（m=100 满载）：qkv/a/b/o-z/out/max-up/down/lm_head
    // 形状从命令行取（m,n,k 三元组，逗号分隔）——k 档二分用
    let mut shapes: Vec<(usize, usize, usize)> = if a.get(3).is_some_and(|x| x.contains(',')) {
        a[3..]
            .iter()
            .map(|x| {
                let v: Vec<usize> = x.split(',').map(|t| t.parse().unwrap()).collect();
                (v[0], v[1], v[2])
            })
            .collect()
    } else {
        [
        (100, 32, 2560),
        (100, 2560, 4096),
        (100, 4096, 2560),
        (100, 9216, 2560),
        (100, 8192, 2560),
        (100, 248320, 2560),
        (100, 2560, 9216),
        (100, 2560, 2560),
        ]
        .to_vec()
    };
    let (e0, e1) = (gpu.event(), gpu.event());
    eprintln!("\n  m      n       k       lt µs   sk µs   加速   lt GB/s  sk GB/s");
    for &(mm, n, k) in &shapes {
        if n % 128 != 0 || k % 32 != 0 {
            eprintln!("  {mm} {n:>7} {k:>7}   （归档外形状，跳过：sk 不接管）");
            continue;
        }
        let da = gpu.alloc_from_slice(&bf16_vec(0x9a11, mm * k, 1.0));
        let dw = gpu.alloc_from_slice(&bf16_vec(0x9b22, n * k, 1.0));
        let dc = gpu.alloc_bytes(mm * n * 2);
        let wb = (n * k * 2) as f64; // 权重字节（GB/s 口径）
        // 注意 kernel_args 合同：实参必须是活过 launch 的 place（`x as u32`
        // 这类临时值会先于 launch 失效 → 驱动读到野地址）
        let (mu, nu, ku) = (mm as u32, n as u32, k as u32);
        let lt = |st: &wein_kernel::cuda::Stream| {
            t.as_ref().expect("lt 档需算子表").gemm.run_wt(&da, &dw, &dc, mu, nu, ku, st);
        };
        let grid = nu / 128;
        let bdim = 256;
        let skl = |st: &wein_kernel::cuda::Stream| {
            let args = wein_kernel::kernel_args![da, dw, dc, mu, nu, ku];
            gpu.launch(&sk, grid, bdim, &args, st);
        };
        // 两路各 3 发热身，再交替计时（避免先后顺序的温度/频率偏置）
        if mode != "lt" {
            for _ in 0..3 {
                skl(&s);
            }
            s.sync();
            eprintln!("  sk 热身通过（n={n} k={k}）");
        }
        if mode != "sk" {
            for _ in 0..3 {
                lt(&s);
            }
            s.sync();
            eprintln!("  lt 热身通过（n={n} k={k}）");
        }
        let mut best = [f64::INFINITY; 2];
        for _ in 0..iters {
            if mode != "sk" {
                e0.record_on(&s);
                lt(&s);
                e1.record_on(&s);
                s.sync();
                let v = e1.elapsed_ms(&e0) as f64;
                best[0] = best[0].min(v);
            }
            if mode != "lt" {
                e0.record_on(&s);
                skl(&s);
                e1.record_on(&s);
                s.sync();
                let v = e1.elapsed_ms(&e0) as f64;
                best[1] = best[1].min(v);
            }
        }
        let (t0, t1) = (best[0], best[1]);
        eprintln!(
            "  {mm} {n:>7} {k:>7}  {:>8.3} {:>7.3}  {:.2}×  {:>7.0}  {:>7.0}",
            t0 * 1e3,
            t1 * 1e3,
            t0 / t1,
            wb / (t0 * 1e-3) / 1e9,
            wb / (t1 * 1e-3) / 1e9
        );
    }
}
