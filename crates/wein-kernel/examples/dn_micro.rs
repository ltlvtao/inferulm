//! 临时诊断（本轮优化用，结束时删）：dn 三 kernel 分 kernel 计时 + 相位分解。
//!   cargo run --release -p wein-kernel --example dn_micro -- [bt] [vsplit] [iters]
use wein_kernel::cuda::{Gpu, Stream};
use wein_kernel::reference;

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
    let bt: usize = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(64);
    let sp: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(4);
    let iters: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(20);
    let gpu = Gpu::probe().expect("CUDA");
    let ptx = std::fs::read_to_string(wein_kernel::ptx_path()).expect("PTX");
    let m = gpu.load_module(&ptx);
    let s = gpu.stream();
    let (f_prep, f_h, f_o, l2f) = (
        m.func("dn_chunk_prep"),
        m.func("dn_chunk_h"),
        m.func("dn_chunk_o"),
        m.func("dn_qk_l2norm"),
    );
    let (kh, vh, kd, vd) = (16usize, 32usize, 128usize, 128usize);
    let tokens = 6009usize;
    let chunks = tokens.div_ceil(bt);
    let mk = |seed: u64, n: usize| gpu.alloc_from_slice(&bf16_vec(seed, n, 1.0));
    let dq = mk(0x1a12, tokens * kh * kd);
    let dk = mk(0x1b34, tokens * kh * kd);
    let dv = mk(0x1c56, tokens * vh * vd);
    let da = mk(0x1d78, tokens * vh);
    let db = mk(0x1e9a, tokens * vh);
    let da_log = gpu.alloc_from_slice(&(0..vh).map(|h| 0.3 + 0.2 * h as f32).collect::<Vec<_>>());
    let ddtb = mk(0x1fbc, vh);
    let dstate = gpu.alloc_from_slice(&vec![0u16; vh * vd * kd]);
    let dout = gpu.alloc_bytes(tokens * vh * vd * 2);
    let drs = gpu.alloc_from_slice(&vec![0u32]);
    let drc = gpu.alloc_from_slice(&vec![tokens as u32]);
    let dri = gpu.alloc_from_slice(&vec![0u32]);
    let inv = gpu.alloc_bytes(tokens * kh * 2 * 4);
    let gcum = gpu.alloc_bytes(tokens * vh * 4);
    let du = gpu.alloc_bytes(tokens * vh * vd * 4);
    let dw = gpu.alloc_bytes(tokens * vh * kd * 4);
    let snap = gpu.alloc_bytes(chunks * vh * vd * kd * 2);
    let sc1 = gpu.alloc_bytes(chunks * vh * 5 * bt * 4);
    let sc2 = gpu.alloc_bytes(vh * sp * (kd * vd + sp * bt + sp * 8192) * 4);
    let dsn = gpu.alloc_from_slice(&vec![0u32, chunks as u32]);
    let (tk, tkh, tvh, tkd, tvd) = (tokens as u32, kh as u32, vh as u32, kd as u32, vd as u32);
    let (nr, nch, btu, eps, spu) = (1u32, chunks as u32, bt as u32, 1e-6f32, sp as u32);
    eprintln!("bt={bt} chunks={chunks} vsplit={sp}");

    let l2 = |st: &Stream| {
        let n = tk * tkh;
        let a1 = wein_kernel::kernel_args![dq, dk, inv, tk, tkh, tkd, eps];
        gpu.launch(&l2f, n.div_ceil(128), 128, &a1, st);
    };
    let prep = |st: &Stream| {
        let a2 = wein_kernel::kernel_args![dk, dv, da, db, da_log, ddtb, inv, gcum, du, dw,
            drs, drc, dsn, nr, nch, tkh, tvh, tkd, tvd, btu, snap, sc1];
        gpu.launch(&f_prep, nch * tkh, 256, &a2, st);
    };
    let h = |st: &Stream| {
        let a3 = wein_kernel::kernel_args![dk, inv, gcum, du, dw, snap, dstate,
            drs, drc, dri, dsn, nr, tkh, tvh, tkd, tvd, btu, spu, sc2];
        let hbd = if spu == 8 { 128u32 } else { 256u32 };
        gpu.launch(&f_h, nr * tvh * spu, hbd, &a3, st);
    };
    let o = |st: &Stream| {
        let a4 = wein_kernel::kernel_args![dq, dk, inv, gcum, du, snap, dout,
            drs, drc, dsn, nr, nch, tkh, tvh, tkd, tvd, btu, dw, sc1];
        gpu.launch(&f_o, nch * tkh, 256, &a4, st);
    };
    let all: [&dyn Fn(&Stream); 4] = [&l2, &prep, &h, &o];
    for f in all {
        for _ in 0..3 {
            f(&s);
        }
    }
    s.sync();
    let bench = |name: &str, f: &dyn Fn(&Stream)| {
        let (e0, e1) = (gpu.event(), gpu.event());
        e0.record_on(&s);
        for _ in 0..iters {
            f(&s);
        }
        e1.record_on(&s);
        s.sync();
        let us = e1.elapsed_ms(&e0) as f64 / iters as f64 * 1000.0;
        eprintln!("  {name:6} {us:9.1} µs/层   ×24 = {:7.2} ms", us * 24.0 / 1000.0);
        us
    };
    let tl = bench("l2", &l2);
    let tp = bench("prep", &prep);
    let th = bench("h", &h);
    let to = bench("o", &o);
    eprintln!(
        "  合计 {:.1} µs/层（×24 = {:.2} ms）",
        tl + tp + th + to,
        (tl + tp + th + to) * 24.0 / 1000.0
    );
}
