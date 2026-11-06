//! 小算子集群刀 A：发射地图 + 逐 kernel 单发成本探针（本轮优化用）。
//!
//!   cargo run --release -p wein-kernel --example org_map -- [iters]
//!
//! 一次性给出（c=100 纯 decode 单步的真实形状）：
//!   - 每个小 kernel 的 **单发 µs**（同一形状、稳态重复取事件均值）
//!   - 每发的**固定开销** = 实测 µs − 该发流量 / 900GB/s
//!   - 各 kernel 的静态 smem 属性（`.shared` 静态数组是否落在 block 私有段）
//! 每步发数不在这里测（由 `wein-exec/examples/qwen_launch_map` + nsys 相减得）。
use wein_kernel::cuda::{Gpu, Stream};
use wein_kernel::ops::{
    AddNormZcOp, AddOp, ArgmaxRowsOp, Conv1dDwOp, ConvStateRollOp, OpTable, RmsNormGatedOp,
    RmsNormZcOp, SiluMulOp,
};
use wein_kernel::reference;
use wein_platform::{Device, HardwareProfile};

fn bf16_vec(seed: u64, n: usize, scale: f32) -> Vec<u16> {
    (0..n as u64)
        .map(|i| {
            let z = i.wrapping_mul(6364136223846793005).wrapping_add(seed);
            let f = ((z >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * scale;
            reference::f32_to_bf16(f)
        })
        .collect()
}

fn bench(gpu: &Gpu, s: &Stream, iters: usize, name: &str, bytes: f64, f: &dyn Fn(&Stream)) -> f64 {
    for _ in 0..5 {
        f(s);
    }
    s.sync();
    let (e0, e1) = (gpu.event(), gpu.event());
    e0.record_on(s);
    for _ in 0..iters {
        f(s);
    }
    e1.record_on(s);
    s.sync();
    let us = e1.elapsed_ms(&e0) as f64 / iters as f64 * 1000.0;
    let bw = bytes / (us * 1e-6) / 1e9;
    let fixed = us - bytes / 900e9 * 1e6;
    println!(
        "  {name:34} {us:8.2} µs/发  {bytes:8.2} MB  纯带宽(900GB/s) {:5.2} µs  ⇒ 固定 {fixed:6.2} µs  实测 {bw:6.1} GB/s",
        bytes / 900e9 * 1e6
    );
    us
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|t| t.parse().ok()).unwrap_or(200);
    let gpu = Gpu::probe().expect("CUDA");
    let s = gpu.stream();
    let prof = HardwareProfile::with_device(Device::Cuda).with_arch("sm_89");
    let t = OpTable::resolve(&gpu, &prof).expect("OpTable");
    println!("iters={iters}");

    // 静态 smem 归属核查：`.shared` 静态数组是否计入每个 kernel 的块私有段
    for k in ["rms_norm_zc", "add_norm_zc", "rms_norm_gated", "argmax_rows_p1", "add", "deltanet"] {
        let f = gpu.load_module(&std::fs::read_to_string(wein_kernel::ptx_path()).unwrap()).func(k);
        println!(
            "  smem {k:16} {:6} B   regs {:3}",
            gpu.func_attribute(&f, 1),
            gpu.func_attribute(&f, 4)
        );
    }

    let (rows, h) = (100usize, 2560usize);
    let x = gpu.alloc_from_slice(&bf16_vec(1, rows * h, 4.0));
    let y = gpu.alloc_from_slice(&bf16_vec(2, rows * h, 4.0));
    let w = gpu.alloc_from_slice(&bf16_vec(3, h, 1.0));
    let o1 = gpu.alloc_bytes(rows * h * 2);
    let o2 = gpu.alloc_bytes(rows * h * 2);
    bench(&gpu, &s, iters, "rms_norm_zc 100x2560", (rows * h * 2 * 2) as f64, &|st| {
        t.rms_norm_zc.run(&x, &w, &o1, rows as u32, h as u32, 1e-6, st)
    });
    bench(&gpu, &s, iters, "add_norm_zc 100x2560", (rows * h * 2 * 5 / 2) as f64, &|st| {
        t.add_norm_zc.run(&x, &y, &w, &o1, &o2, rows as u32, h as u32, 1e-6, st)
    });
    bench(&gpu, &s, iters, "add 100x2560", (rows * h * 2 * 3 / 2) as f64, &|st| {
        t.add.run(&x, &y, &o1, (rows * h) as u32, st)
    });

    // q/k norm 形状：rows = t·q_heads = 3200，hidden = head_dim 128（块 128）
    let (qr, qh) = (3200usize, 128usize);
    let qx = gpu.alloc_from_slice(&bf16_vec(4, qr * qh, 4.0));
    let qw = gpu.alloc_from_slice(&bf16_vec(5, qh, 1.0));
    let qo = gpu.alloc_bytes(qr * qh * 2);
    bench(&gpu, &s, iters, "rms_norm_zc 3200x128 (q/k)", (qr * qh * 2 * 2) as f64, &|st| {
        t.rms_norm_zc.run(&qx, &qw, &qo, qr as u32, qh as u32, 1e-6, st)
    });

    // rms_norm_gated：rows = t·v_heads = 3200，hidden = v_dim 128；w 是 f32
    let gw = gpu.alloc_from_slice(&(0..qh).map(|i| 0.1 * i as f32).collect::<Vec<_>>());
    bench(&gpu, &s, iters, "rms_norm_gated 3200x128", (qr * qh * 2 * 3) as f64, &|st| {
        t.rms_norm_gated.run(&qx, &qx, &gw, &qo, qr as u32, qh as u32, 1e-6, st)
    });

    // silu_mul：n = t·inter = 921600
    let ii = 9216usize;
    let sg = gpu.alloc_from_slice(&bf16_vec(6, rows * ii, 4.0));
    let su = gpu.alloc_from_slice(&bf16_vec(7, rows * ii, 4.0));
    let sm = gpu.alloc_bytes(rows * ii * 2);
    bench(&gpu, &s, iters, "silu_mul 100x9216", (rows * ii * 2 * 3) as f64, &|st| {
        t.silu_mul.run(&sg, &su, &sm, (rows * ii) as u32, st)
    });

    // argmax：100 行 × 248320 bf16（50MB）
    let n = 248320usize;
    let lg = gpu.alloc_from_slice(&bf16_vec(8, rows * n, 4.0));
    let ridx: Vec<u32> = (0..rows as u32).collect();
    let dr = gpu.alloc_from_slice(&ridx);
    let sc = gpu.alloc_bytes(rows * wein_kernel::ops::ARGMAX_SLOT as usize * 4);
    let ao = gpu.alloc_bytes(rows * 4);
    bench(&gpu, &s, iters, "argmax_rows 100x248320", (rows * n * 2) as f64, &|st| {
        t.argmax_rows.run(&lg, &dr, &sc, &ao, n as u32, rows as u32, st)
    });

    // conv：conv_dim=8192、kernel=4、100 行
    let cd = 8192usize;
    let (tk, tks) = (4usize, 3usize);
    let cx = gpu.alloc_from_slice(&bf16_vec(9, rows * cd, 4.0));
    let cw = gpu.alloc_from_slice(&bf16_vec(10, cd * tk, 1.0));
    let cst = gpu.alloc_from_slice(&bf16_vec(11, rows * cd * tks, 4.0));
    let co = gpu.alloc_bytes(rows * cd * 2);
    let tr = gpu.alloc_from_slice(&ridx);
    let tp = gpu.alloc_from_slice(&vec![0u32; rows]);
    let rs = gpu.alloc_from_slice(&(0..rows as u32).collect::<Vec<_>>());
    let rc = gpu.alloc_from_slice(&vec![1u32; rows]);
    let sp = gpu.alloc_from_slice(&vec![0u32; rows]);
    bench(&gpu, &s, iters, "conv1d_dw 100x8192 k4", (rows * cd * 2 * 5) as f64, &|st| {
        t.conv1d_dw.run(&cx, &cw, &cst, &co, &tr, &tp, cd as u32, tk as u32, rows as u32, st)
    });
    bench(&gpu, &s, iters, "conv_state_roll 100x8192 k4", (rows * cd * 2 * 4) as f64, &|st| {
        t.conv_state_roll.run(&cx, &cst, &rs, &rc, &tr, cd as u32, tk as u32, rows as u32, &sp, st)
    });
}
