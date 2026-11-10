//! 多行 decode attention 形状探针（agent c=8 稳态归因）。
//!
//!   cargo run --release -p wein-kernel --example attn_rows_probe
//!
//! R 行（各自独立序列/页表，行 r 的上下文 = ctx + 小抖动制造页离散）×
//! decode 形（tokens=1、pos=ctx-1），测单遍 online 档（decode cfg nd4/r2）
//! 的单发时长与有效带宽。对照 js64 基准（R=1, ctx=8k ≈ 590GB/s）。
use wein_kernel::cuda::Gpu;
use wein_kernel::ops::{AttnCfg, KvDtype, OpTable, PagedAttnCuda};
use wein_platform::{Device, HardwareProfile};

fn bf16_vec(seed_hi: u64, n: usize, scale: f32) -> Vec<u16> {
    (0..n as u64)
        .map(|i| {
            let z = i.wrapping_mul(6364136223846793005).wrapping_add(seed_hi);
            let f = ((z >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * scale;
            wein_kernel::reference::f32_to_bf16(f)
        })
        .collect()
}

/// R 行 × 每行 ~ctx token；页表每行独立分配。interleave=true 时行间物理交错
/// （行 r 的页基 = r 偏移交错，模拟真实 BlockPool 并发分配），false 时连续铺。
fn case(rows: usize, ctx: usize, q_heads: usize, kv_heads: usize, hd: usize, bs: usize, interleave: bool) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u32>, Vec<u32>, Vec<u32>, usize, usize) {
    let he = kv_heads * hd;
    let pages_per_row = ctx.div_ceil(bs);
    let npages = pages_per_row * rows;
    let slots = npages * bs;
    // 行 r 的页占 [r, r+pages, 2r+pages…] 交错——KV 池里行间物理交错（真实）
    let mut bt = vec![0u32; rows * pages_per_row];
    let mut used = vec![false; npages];
    for r in 0..rows {
        for p in 0..pages_per_row {
            let page = if interleave { (r + p * rows) % npages } else { r * pages_per_row + p };
            assert!(!used[page]);
            used[page] = true;
            bt[r * pages_per_row + p] = page as u32;
        }
    }
    let q = bf16_vec(0x7a11, rows * q_heads * hd, 2.0);
    let k = bf16_vec(0x7b33, slots * he, 1.0);
    let v = bf16_vec(0x7c55, slots * he, 1.0);
    let si = (0..rows as u32).collect::<Vec<_>>();
    let pos: Vec<u32> = (0..rows).map(|r| (ctx - 1 - (r % 3)) as u32).collect();
    (q, k, v, bt, si, pos, pages_per_row, slots)
}

fn main() {
    let gpu = Gpu::probe().expect("须有 CUDA 设备");
    let profile = HardwareProfile::with_device(Device::Cuda).with_arch("sm_89");
    let t = OpTable::resolve(&gpu, &profile).expect("sm_89 画像须解析成功");
    let stream = gpu.stream();
    let (bs, q_heads, kv_heads, hd) = (16usize, 16usize, 4usize, 256usize);
    let q_scale = 1.0 / (hd as f32).sqrt();
    let dec_cfg = AttnCfg { rpt: 2, ..AttnCfg::baseline() };
    let _ = KvDtype::Bf16;

    println!("D2D 标定：");
    let n = 96usize * 1024 * 1024;
    let a = gpu.alloc_bytes(n);
    let b = gpu.alloc_bytes(n);
    for _ in 0..3 {
        b.copy_range_on(0, &a, 0, n, &stream);
    }
    stream.sync();
    let (e0, e1) = (gpu.event(), gpu.event());
    e0.record_on(&stream);
    for _ in 0..10 {
        b.copy_range_on(0, &a, 0, n, &stream);
    }
    e1.record_on(&stream);
    stream.sync();
    let ms: f64 = e1.elapsed_ms(&e0) as f64 / 10.0;
    println!("  {:.0} GB/s", 2.0 * n as f64 / 1e9 / (ms / 1e3));

    // 对照矩阵：行数 × 页布局 × ctx——分离"页交错损失"与"行数/批形损失"
    println!("\n对照矩阵（现役 prefill cfg n4r4/ntok4 vs 单组档 n8r4/ntok1）：");
    println!("| rows | ctx | KV MB | 现役 ms | 现役 GB/s | 单组 ms | 单组 GB/s |");
    println!("|---|---|---|---|---|---|---|");
    let cur = AttnCfg { rpt: 4, ..AttnCfg::baseline() };          // t>4 生产档
    let cur_dec = AttnCfg { rpt: 2, ..AttnCfg::baseline() };      // t≤4 生产档
    let single = AttnCfg { nd: 8, rpt: 4, ntok_max: 1, ..AttnCfg::baseline() }; // 单组 1×
    for rows in [1usize, 2, 4, 8, 16, 32, 64] {
        for ctx in [1280usize, 2560, 5120, 10240] {
            let (ms_a, kv_mb) = bench(&t, &gpu, &stream, if rows <= 4 { cur_dec } else { cur },
                rows, ctx, q_heads, kv_heads, hd, bs, q_scale, true);
            let (ms_b, _) = bench(&t, &gpu, &stream, single,
                rows, ctx, q_heads, kv_heads, hd, bs, q_scale, true);
            let f = |ms: f64| kv_mb as f64 * 1.048576 / ms;
            println!("| {rows} | {ctx} | {kv_mb} | {ms_a:.4} | {:.0} | {ms_b:.4} | {:.0} |", f(ms_a), f(ms_b));
        }
    }

    // 档位扫描：R=8@5120（agent c=8 稳态形）固定，扫 (nd, rpt, ntok_max, bdim_max)
    println!("\n档位扫描 R=8 ctx=5120（KV 160MB，D2D 为顶）：");
    println!("| nd | rpt | ntok_max | bdim_max | nt | ntile | js | 块数 | bdim | ms | GB/s |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    // 杀放大档：ntok=1+rpt=group ⇒ 每 block 单行组，K/V 每行只载一次
    for (nd, rpt, ntok_max) in [
        (4usize, 2usize, 8usize),   // 现役生产档（放大 8×）
        (4usize, 4usize, 1usize),   // 单组 1×（bdim=64）
        (8usize, 4usize, 1usize),   // 单组 1×（bdim=32）
        (4usize, 4usize, 2usize),   // 放大 2×（bdim=128）
        (4usize, 4usize, 4usize),   // 放大 4×（bdim=256）
    ] {
        for bdim_max in [512usize, 1024] {
            let cfg = AttnCfg {
                nd, rpt, ntok_max, bdim_max,
                ..AttnCfg::baseline()
            };
            let g = match std::panic::catch_unwind(|| {
                PagedAttnCuda::online_geom(cfg, q_heads as u32, kv_heads as u32, hd as u32, 8)
            }) {
                Ok(g) => g,
                Err(_) => continue,
            };
            let (ms, kv_mb) = bench(&t, &gpu, &stream, cfg, 8, 5120, q_heads, kv_heads, hd, bs, q_scale, true);
            let blocks = g.jsplit * kv_heads as usize * g.ntile;
            let gbs = kv_mb as f64 * 1.048576 / ms;
            println!("| {nd} | {rpt} | {ntok_max} | {bdim_max} | {} | {} | {} | {blocks} | {} | {ms:.4} | {gbs:.0} |",
                g.ntok, g.ntile, g.jsplit, g.bdim);
        }
    }
}

fn bench(
    t: &OpTable,
    gpu: &wein_kernel::cuda::Gpu,
    stream: &wein_kernel::cuda::Stream,
    dec_cfg: AttnCfg,
    rows: usize,
    ctx: usize,
    q_heads: usize,
    kv_heads: usize,
    hd: usize,
    bs: usize,
    q_scale: f32,
    interleave: bool,
) -> (f64, usize) {
    let (q, k, v, bt, si, pos, bpr, _slots) = case(rows, ctx, q_heads, kv_heads, hd, bs, interleave);
    let dq = gpu.alloc_from_slice(&q);
    let dk = gpu.alloc_from_slice(&k);
    let dv = gpu.alloc_from_slice(&v);
    let dbt = gpu.alloc_from_slice(&bt);
    let dsi = gpu.alloc_from_slice(&si);
    let dpo = gpu.alloc_from_slice(&pos);
    let dout = gpu.alloc_bytes(rows * q_heads * hd * 2);
    let run = || {
        t.paged_attn.run_tile_online_cfg(
            dec_cfg, &dq, &dk, &dv, &dbt, &dsi, &dpo, &dout, q_scale,
            bpr as u32, bs as u32, q_heads as u32, kv_heads as u32,
            hd as u32, rows as u32, stream,
        )
    };
    for _ in 0..5 {
        run();
    }
    stream.sync();
    let (e0, e1) = (gpu.event(), gpu.event());
    e0.record_on(stream);
    for _ in 0..20 {
        run();
    }
    e1.record_on(stream);
    stream.sync();
    let ms = e1.elapsed_ms(&e0) as f64 / 20.0;
    let kv_mb = rows * ctx * kv_heads * hd * 2 * 2 / 1024 / 1024;
    let _ = interleave;
    (ms, kv_mb)
}
