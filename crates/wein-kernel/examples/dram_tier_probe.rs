//! DRAM 分层探针：WSL 下 pinned/pageable H2D/D2H 带宽与换入延迟。
//! 回答"驱逐块下沉 DRAM、命中搬回"是否划算：
//!   换入 215MB（一张快照 + 190 KV 页）目标 < 20ms（重算同段 ~600ms）。
//!   cargo run --release -p wein-kernel --example dram_tier_probe
use wein_kernel::cuda::Gpu;

fn main() {
    let gpu = Gpu::probe().expect("CUDA");
    let s = gpu.stream();
    for mb in [25usize, 215, 1024] {
        let bytes = mb * 1024 * 1024;
        let dev = gpu.alloc_bytes(bytes);
        let pin = gpu.alloc_host_bytes(bytes);
        // 设备侧内容初始化（模拟待换出的热数据）
        pin.as_slice().iter_mut().for_each(|b| *b = 0xA5);
        pin.upload_to_on(&dev, 0, bytes, &s);
        s.sync();

        let iters = if mb >= 1024 { 5 } else { 10 };

        // pinned H2D（换入方向：DRAM 命中 → 设备）
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            pin.upload_to_on(&dev, 0, bytes, &s);
            s.sync();
        }
        let h2d_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

        // pinned D2H（换出方向：设备 → DRAM）
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            pin.download_on(&dev, 0, bytes, &s);
            s.sync();
        }
        let d2h_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;

        let gbps = |ms: f64| bytes as f64 / 1e9 / (ms / 1e3);
        println!(
            "{mb:5} MB pinned: H2D {h2d_ms:7.2} ms ({:5.1} GB/s) | D2H {d2h_ms:7.2} ms ({:5.1} GB/s) | 换入 {h2d_ms:5.2} ms",
            gbps(h2d_ms), gbps(d2h_ms),
        );
        std::hint::black_box(pin.as_slice()[0]);
    }
    println!("判定：215MB 换入 < 20ms 即成立（重算同段 ~600ms，>30× 赢面）");
}
