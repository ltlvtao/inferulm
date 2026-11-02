//! GEMM 大形状探针（prefill lane 大块的算力墙审计）。
//!
//! 大块 prefill 后 gemm 的 m 从 256 抬到 2048-8192——本探针按模型真投影
//! 形状扫 m∈{256..8192}，给 ms/TFLOPS/GB/s 三列，验收"大 m 走到算力墙
//! （≥140TFLOPS，vLLM 同卡 162）还是仍困在带宽墙（chunk256 档 94）"。
//! 用 heuristic 档（`open_tuned` 空 m 集）——exec 的 prefill m 不在稳态
//! 调优集，与生产路径同选型。
//! 跑法：`cargo run --release -p wein-kernel --example gemm_m_probe`
use wein_kernel::cuda::BlasLt;

fn main() {
    let gpu = wein_kernel::cuda::Gpu::probe().expect("GPU");
    let stream = gpu.stream();
    let lt = BlasLt::open_tuned(&gpu, &[]).expect("libcublasLt");
    // 模型投影形状（hidden 2560 / inter 9216 / full qkvo + linear qkvz）：
    // (名, n, k)。C=A·Wᵀ 的 A=[m,k]、W=[n,k]（nn.Linear 权重布局）
    let shapes: &[(&str, u32, u32)] = &[
        ("qkv/qg 8192×2560", 8192, 2560),
        ("k/v    1024×2560", 1024, 2560),
        ("o      2560×4096", 2560, 4096),
        ("gate/up 9216×2560", 9216, 2560),
        ("down   2560×9216", 2560, 9216),
    ];
    let ms: Vec<u32> = vec![256, 512, 1024, 2048, 4096, 8192];
    println!("shape                m     ms   TFLOPS    GB/s   (heuristic 档，10 次中位)");
    for &(name, n, k) in shapes {
        for &m in &ms {
            let (a, w, c) = (
                gpu.alloc_bytes(m as usize * k as usize * 2),
                gpu.alloc_bytes(n as usize * k as usize * 2),
                gpu.alloc_bytes(m as usize * n as usize * 2),
            );
            a.zero_on(&stream);
            w.zero_on(&stream);
            let ev0 = gpu.event();
            let ev1 = gpu.event();
            for _ in 0..3 {
                lt.gemm_wt(&a, &w, &c, m, n, k, &stream);
            }
            let mut ts = Vec::with_capacity(10);
            for _ in 0..10 {
                ev0.record_on(&stream);
                lt.gemm_wt(&a, &w, &c, m, n, k, &stream);
                ev1.record_on(&stream);
                stream.sync();
                ts.push(ev1.elapsed_ms(&ev0));
            }
            ts.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let t = ts[ts.len() / 2] as f64;
            let fl = 2.0 * m as f64 * n as f64 * k as f64;
            let by = (m as usize * k as usize + n as usize * k as usize + m as usize * n as usize) as f64 * 2.0;
            println!(
                "{name:18} {m:5} {t:6.2} {:8.1} {:7.1}",
                fl / t / 1e9,
                by / t / 1e6
            );
        }
    }
}
