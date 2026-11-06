//! 小 N 投影的**每发真实占用**探针（组织形态刀 B 第 1/2 步）。
//!
//! 纯背靠背连发（无 host 间隔、无 sync）：一条 stream 上把同一形状发 N 次，
//! 事件夹住总时距 ÷ N = 「一发占 GPU 多少 µs」（含 kernel 间空隙）。这正是
//! decode 步里 a/b（32×2560）与 k/v（1024×2560）每发的占用量级；也是
//! 「合并成 n=64 一发」能省多少的直接判据——n=64 一发约等于 n=32 一发时，
//! 合并只省「少发一次」的钱。
//!
//! 跑法：`cargo run --release -p wein-kernel --example gemm_smalln_probe [m]`
//! （默认 m=100，即满载 decode 的行数；R=10 轮取中位）
use wein_kernel::cuda::BlasLt;

fn chain_ms(lt: &BlasLt<'_>, gpu: &wein_kernel::cuda::Gpu, m: u32, n: u32, k: u32, reps: u32) -> (f64, f64) {
    let stream = gpu.stream();
    let (a, w, c) = (
        gpu.alloc_bytes(m as usize * k as usize * 2),
        gpu.alloc_bytes(n as usize * k as usize * 2),
        gpu.alloc_bytes(m as usize * n as usize * 2),
    );
    a.zero_on(&stream);
    w.zero_on(&stream);
    c.zero_on(&stream);
    for _ in 0..20 {
        lt.gemm_wt(&a, &w, &c, m, n, k, &stream);
    }
    stream.sync();
    let (e0, e1) = (gpu.event(), gpu.event());
    let mut ts = Vec::with_capacity(9);
    let mut hs = Vec::with_capacity(9);
    for _ in 0..9 {
        e0.record_on(&stream);
        let h0 = std::time::Instant::now();
        for _ in 0..reps {
            lt.gemm_wt(&a, &w, &c, m, n, k, &stream);
        }
        let host_us = h0.elapsed().as_secs_f64() * 1e6 / reps as f64;
        e1.record_on(&stream);
        stream.sync();
        ts.push(e1.elapsed_ms(&e0) as f64 / reps as f64 * 1e3); // µs/发（GPU 时间轴）
        hs.push(host_us); // µs/发（host 提交；与 GPU 重叠）
    }
    ts.sort_by(|x, y| x.partial_cmp(y).unwrap());
    hs.sort_by(|x, y| x.partial_cmp(y).unwrap());
    (ts[ts.len() / 2], hs[hs.len() / 2])
}

fn main() {
    let m: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let gpu = wein_kernel::cuda::Gpu::probe().expect("GPU");
    let lt = BlasLt::open_tuned(&gpu, &[]).expect("libcublasLt");
    let k = 2560u32;
    // (名, n)：a/b 现状、合并档、k/v、以及 k+v 合成 2048 / q+k+v 合成 5120 的假想档
    let shapes: &[(&str, u32)] = &[
        ("a/b   现状 n=32", 32),
        ("a+b   合并 n=64", 64),
        ("k/v   现状 n=1024", 1024),
        ("q     现状 n=4096", 4096),
        ("k+v   合并 n=2048", 2048),
        ("q+k+v 合并 n=5120", 5120),
    ];
    println!("m={m} k={k}｜背靠背连发（每形状 200 发/轮 × 9 轮取中位，冷档 heuristic 选型）");
    println!(
        "{:<22} {:>10} {:>10} {:>12} {:>10}",
        "shape", "µs/发", "host 提交", "纯带宽 µs", "固定项 µs"
    );
    for &(name, n) in shapes {
        let (us, host) = chain_ms(&lt, &gpu, m, n, k, 200);
        // 纯带宽：读 x(m×k) + 读 w(n×k) + 写 c(m×n)，@900GB/s
        let bytes = (m as f64 * k as f64 + n as f64 * k as f64 + m as f64 * n as f64) * 2.0;
        let bw = bytes / 900e9 * 1e6;
        println!("{name:<22} {us:>10.2} {host:>10.2} {bw:>12.2} {:>10.2}", us - bw);
    }
}

