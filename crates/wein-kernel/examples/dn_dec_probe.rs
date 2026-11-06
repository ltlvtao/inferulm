//! deltanet decode 单发性能探针（同进程交错 A/B）。
//!
//!   cargo run --release -p wein-kernel --example dn_dec_probe [rounds]
//!
//! 批形 = o22 金丝雀：m=100 行 × 32 v 头 × kd=vd=128，每行 1 token。
//! 状态池流量 = 100·32·128·128·2B × 2（读+写）= 209.7 MB。
//!
//! 对照组：
//!   · op   —— `DeltanetOp::run` 原样（含前置 l2norm 一发），= o22 口径
//!   · kern —— 直接 launch `deltanet`（不含 l2），按 (vsplit, bdim) 扫几何
//! 每组先热身，再与其它组**交错**同轮计时（抵消时钟漂移），报中位 µs 与 GB/s。
//! 另有 D2D 拷贝带宽标定作为"纯状态流"上限（同口径 209.7MB 折算的占 D2D%）。
use wein_kernel::cuda::{Gpu, Stream};
use wein_kernel::ops::{DeltanetOp, OpTable};
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

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Op,
    Kern,
}

struct V {
    label: String,
    kind: Kind,
    sp: u32,
    bd: u32,
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let rounds: usize = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(15);
    let gpu = Gpu::probe().expect("CUDA");
    let profile = HardwareProfile::with_device(Device::Cuda).with_arch("sm_89");
    let t = OpTable::resolve(&gpu, &profile).expect("resolve");
    let ptx = std::fs::read_to_string(wein_kernel::ptx_path()).expect("PTX");
    let m = gpu.load_module(&ptx);
    let (f_dn, l2f) = (m.func("deltanet"), m.func("dn_qk_l2norm"));
    let s = gpu.stream();

    let (n_rows, kh, vh, kd, vd) = (100usize, 16usize, 32usize, 128usize, 128usize);
    let tokens = n_rows;
    let mk = |seed: u64, n: usize| gpu.alloc_from_slice(&bf16_vec(seed, n, 1.0));
    let dq = mk(0x1a12, tokens * kh * kd);
    let dk = mk(0x1b34, tokens * kh * kd);
    let dv = mk(0x1c56, tokens * vh * vd);
    let da = mk(0x1d78, tokens * vh);
    let db = mk(0x1e9a, tokens * vh);
    let da_log = gpu.alloc_from_slice(&(0..vh).map(|h| 0.3 + 0.2 * h as f32).collect::<Vec<_>>());
    let ddtb = mk(0x1fbc, vh);
    let dstate = gpu.alloc_from_slice(&vec![0u16; n_rows * vh * vd * kd]);
    let dout = gpu.alloc_bytes(tokens * vh * vd * 2);
    let drs: Vec<u32> = (0..n_rows as u32).collect();
    let drcv = vec![1u32; n_rows];
    let dri = drs.clone();
    let (drs, drc, dri) = (
        gpu.alloc_from_slice(&drs),
        gpu.alloc_from_slice(&drcv),
        gpu.alloc_from_slice(&dri),
    );
    let dspec0 = gpu.alloc_from_slice(&vec![0u32; n_rows]);
    // 与生产同尺的 scratch/inv（DeltanetCuda 字段私有，探针自备一份；
    // scratch = DELTA_BLOCK_CAP(4096) × (256/32)·1024 f32，spec==1 路径才用）
    let scratch = gpu.alloc_bytes(4096 * 8 * 1024 * 4);
    let inv = gpu.alloc_bytes(tokens * kh * 2 * 4);
    // inv 由前置 kernel 填一次（否则全 0——不影响时延，只为形状真实）
    {
        let (tk, tkh, tkd, eps) = (tokens as u32, kh as u32, kd as u32, 1e-6f32);
        let n2 = tk * tkh;
        let a1 = wein_kernel::kernel_args![dq, dk, inv, tk, tkh, tkd, eps];
        gpu.launch(&l2f, n2.div_ceil(128), 128, &a1, &s);
        s.sync();
    }

    // 状态流字节（读 + 写）
    let bytes = (n_rows * vh * vd * kd * 2 * 2) as f64;
    let gbs = |us: f64| bytes / (us * 1e-6) / 1e9;

    // ── D2D 带宽标定 ──
    let (cap, bw_iters) = (64usize << 20, 20usize);
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
    eprintln!("D2D 拷贝带宽标定：{bw:.0} GB/s（读+写计）");

    // ── 对照组清单 ──
    let mut vars: Vec<V> = vec![V {
        label: "op   (o22 口径: l2+kern)".to_string(),
        kind: Kind::Op,
        sp: 1,
        bd: 128,
    }];
    // (vsplit, bdim)
    let geoms: [(u32, u32); 6] = [(1, 128), (1, 256), (2, 64), (4, 64), (4, 128), (8, 64)];
    for &(sp, bd) in &geoms {
        if bd / 32 > vd as u32 / sp {
            continue;
        }
        let rows_pw = (vd as u32 / sp) / (bd / 32);
        if rows_pw < 4 || rows_pw % 4 != 0 {
            continue;
        }
        vars.push(V {
            label: format!("kern vsplit={sp} bdim={bd:<3} rows_pw={rows_pw}"),
            kind: Kind::Kern,
            sp,
            bd,
        });
    }

    let nrs = n_rows as u32;
    let (kh_u, vh_u, kd_u, vd_u) = (kh as u32, vh as u32, kd as u32, vd as u32);
    let tks = tokens as u32;
    let fire = |v: &V, st: &Stream| match v.kind {
        Kind::Op => {
            t.deltanet.run(&dq, &dk, &dv, &da, &db, &da_log, &ddtb, &dstate, &dout, &drs, &drc,
                &dri, nrs, kh_u, vh_u, kd_u, vd_u, tks, 1e-6, &dspec0, v.sp, st);
        }
        Kind::Kern => {
            let args = wein_kernel::kernel_args![dq, dk, dv, da, db, da_log, ddtb, dstate, dout,
                drs, drc, dri, nrs, kh_u, vh_u, kd_u, vd_u, inv, scratch, dspec0, v.sp];
            gpu.launch(&f_dn, nrs * vh_u * v.sp, v.bd, &args, st);
        }
    };

    for v in &vars {
        for _ in 0..3 {
            fire(v, &s);
        }
    }
    s.sync();

    // 交错同轮计时：每轮每组连发 per 拍
    let per = 5usize;
    let (e0, e1) = (gpu.event(), gpu.event());
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); vars.len()];
    for _ in 0..rounds {
        for (i, v) in vars.iter().enumerate() {
            e0.record_on(&s);
            for _ in 0..per {
                fire(v, &s);
            }
            e1.record_on(&s);
            s.sync();
            samples[i].push(e1.elapsed_ms(&e0) as f64 * 1000.0 / per as f64);
        }
    }

    eprintln!(
        "\n== deltanet decode 单发（m=100×32×128²，状态流 {:.1} MB，{rounds} 轮交错）==",
        bytes / 1e6
    );
    eprintln!("  {:<40} {:>9} {:>8} {:>7}", "变体", "中位 µs", "GB/s", "占D2D");
    for (i, v) in vars.iter().enumerate() {
        let us = median(samples[i].clone());
        eprintln!("  {:<40} {us:9.2} {:8.0} {:6.0}%", v.label, gbs(us), gbs(us) / bw * 100.0);
    }
}
