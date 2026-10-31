//! deltanet invq/invk 去重后的 vsplit A/B 探针（单层、单 launch 事件计时）。
//!
//!   cargo run --release -p wein-kernel --example dn_split_probe
//!
//! 形状取自真机（Qwen3.5-4B，24 个线性层）：kh=16、vh=32、kd=vd=128。
//! 两组几何：
//! - **prefill**：1 行 × 250 token（单 span 长 prefill 的真实批形——同一
//!   状态行在一个 block 里串行吃 t 个 token，块 = (行, v 头, v 行块)）
//! - **decode**：100 行 × 1 token（满负荷 m=100；门控下 vsplit=1）
//!
//! 报三样：主 kernel（deltanet）、前置 l2norm kernel（dn_qk_l2norm）单独
//! 计时、以及两者之和 ÷ token = 每 token 每层 µs（与"地板"2µs 对比：状态流
//! = vh·vd·kd·2B 读 + 同量写 ≈ 2MB/token/层）。
use wein_kernel::cuda::{Gpu, Stream};
use wein_kernel::ops::{DeltanetOp, OpTable};
use wein_platform::{Device, HardwareProfile};

fn bf16_vec(seed_hi: u64, n: usize, scale: f32) -> Vec<u16> {
    (0..n as u64)
        .map(|i| {
            let z = i.wrapping_mul(6364136223846793005).wrapping_add(seed_hi);
            let f = ((z >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * scale;
            f.to_bits() as u16
        })
        .collect()
}

fn main() {
    let gpu = Gpu::probe().expect("须有 CUDA 设备（本探针是设备实验）");
    let profile = HardwareProfile::with_device(Device::Cuda).with_arch("sm_89");
    let t = OpTable::resolve(&gpu, &profile).expect("sm_89 画像须解析成功");
    let ptx = std::fs::read_to_string(wein_kernel::ptx_path()).expect("PTX 产物");
    let l2f = gpu.load_module(&ptx).func("dn_qk_l2norm");
    let s = gpu.stream();

    let (kh, vh, kd, vd) = (16usize, 32usize, 128usize, 128usize);
    let iters = 20usize;

    // 带宽标定：同一台机器上"地板"随实测带宽漂移（外部占用会压低有效
    // 带宽）——所以先量一次 D2D 拷贝吞吐，再据此算本次运行的 µs 地板
    let (cap, bw_iters) = (256usize << 20, 10usize);
    let (bsrc, bdst) = (gpu.alloc_bytes(cap), gpu.alloc_bytes(cap));
    for _ in 0..3 {
        bdst.copy_range_on(0, &bsrc, 0, cap, &s);
    }
    s.sync();
    let (eb0, eb1) = (gpu.event(), gpu.event());
    eb0.record_on(&s);
    for _ in 0..bw_iters {
        bdst.copy_range_on(0, &bsrc, 0, cap, &s);
    }
    eb1.record_on(&s);
    s.sync();
    let bw = (cap * 2 * bw_iters) as f64 / (eb1.elapsed_ms(&eb0) as f64 * 1e-3) / 1e9;
    eprintln!("D2D 带宽标定：{bw:.0} GB/s（{cap} B 往返 {bw_iters} 次，读+写计）");

    // 两组几何：(标签, n_rows, tokens/行)
    for &(tag, n_rows, tpr) in &[
        ("prefill 1x250", 1usize, 250usize),
        ("prefill 2x125", 2, 125),
        ("prefill 4x64", 4, 64),
        ("prefill 8x32", 8, 32),
        ("decode 100x1", 100, 1),
    ] {
        let tokens = n_rows * tpr;
        let mk = |seed: u64, n: usize| gpu.alloc_from_slice(&bf16_vec(seed, n, 1.0));
        let dq = mk(0x6a12, tokens * kh * kd);
        let dk = mk(0x6b34, tokens * kh * kd);
        let dv = mk(0x6c56, tokens * vh * vd);
        let da = mk(0x6d78, tokens * vh);
        let db = mk(0x6e9a, tokens * vh);
        let da_log =
            gpu.alloc_from_slice(&(0..vh).map(|h| 0.3 + 0.2 * h as f32).collect::<Vec<_>>());
        let ddtb = mk(0x6fbc, vh);
        let state_len = n_rows * vh * vd * kd;
        let dstate = gpu.alloc_from_slice(&vec![0u16; state_len]);
        let dout = gpu.alloc_bytes(tokens * vh * vd * 2);
        let drs = gpu.alloc_from_slice(
            &(0..n_rows).map(|r| (r * tpr) as u32).collect::<Vec<_>>(),
        );
        let drc = gpu.alloc_from_slice(&vec![tpr as u32; n_rows]);
        let dri = gpu.alloc_from_slice(&(0..n_rows as u32).collect::<Vec<_>>());
        let dspec0 = gpu.alloc_from_slice(&vec![0u32; n_rows]);

        // 前置 l2norm kernel 的字节账：2×(tokens·kh·kd) bf16 读 + tokens·kh·2 f32 写
        let l2_inv = gpu.alloc_bytes(tokens * kh * 2 * 4);
        // kernel_args! 的合同：实参必须是活过 launch 的 place（临时表达式会
        // 落成被复用的栈槽）——标量一律先绑局部
        let (n_l2, tk, tkh, tkd, l2eps) =
            ((tokens * kh) as u32, tokens as u32, kh as u32, kd as u32, 1e-6f32);
        let l2 = |s: &Stream| {
            let args = wein_kernel::kernel_args![dq, dk, l2_inv, tk, tkh, tkd, l2eps];
            gpu.launch(&l2f, n_l2.div_ceil(128), 128, &args, s);
        };

        let state_bytes = (state_len * 2 * 2) as f64; // 读 + 写（bf16 池，一趟 kvm 一趟更新）
        eprintln!("\n== {tag}（{n_rows} 行 × {tpr} token = {tokens} token，kh={kh} vh={vh} kd={kd}）==");
        // 地板：每 token 每层状态流 = vh·vd·kd·2B 读 + 同量写（本次实测量带宽）
        let floor_us = (vh * vd * kd * 2 * 2) as f64 / (bw * 1e9) * 1e6;
        eprintln!("  状态流 {:.2} MB/launch（读+写），地板（实测 {bw:.0} GB/s）{floor_us:.2} µs/token/层",
            state_bytes / 1e6);

        // 前置 kernel 单跑
        for _ in 0..3 {
            l2(&s);
        }
        s.sync();
        let (e0, e1) = (gpu.event(), gpu.event());
        e0.record_on(&s);
        for _ in 0..iters {
            l2(&s);
        }
        e1.record_on(&s);
        s.sync();
        let l2_ms = e1.elapsed_ms(&e0) as f64 / iters as f64;
        let l2_us = l2_ms * 1000.0;
        eprintln!("  dn_qk_l2norm      {l2_us:8.2} µs/launch（{:.4} µs/token/层）",
            l2_us / tpr as f64);

        for &split in &[1u32, 2, 4, 8, 16] {
            let run = |st: &Stream| {
                t.deltanet.run(&dq, &dk, &dv, &da, &db, &da_log, &ddtb, &dstate, &dout, &drs,
                    &drc, &dri, n_rows as u32, kh as u32, vh as u32, kd as u32, vd as u32,
                    tokens as u32, 1e-6, &dspec0, split, st);
            };
            for _ in 0..3 {
                run(&s);
            }
            s.sync();
            e0.record_on(&s);
            for _ in 0..iters {
                run(&s);
            }
            e1.record_on(&s);
            s.sync();
            let ms = e1.elapsed_ms(&e0) as f64 / iters as f64;
            let tot_us = ms * 1000.0;
            let main_us = tot_us - l2_us;
            eprintln!(
                "  vsplit={split:<2} grid={:>4}  合计 {tot_us:8.2} µs/launch = {:5.2} µs/token/层（主 kernel 扣前置 {main_us:7.2} µs）",
                n_rows * vh * split as usize,
                tot_us / tpr as f64
            );
        }
    }
    eprintln!("\n（decode 组按门控 vsplit=1 定档；prefill 组按 A/B 选档——`DELTA_SPLIT` 是上限钳制）");
}
