//! deltanet chunked（BT 分块递推）单发性能探针：6009-token span（6009 冷
//! prefill 的真实批形），24 层等价负载。
//!
//!   cargo run --release -p wein-kernel --example dn_chunk_probe
//!
//! 报三样：① op 级整段（l2+prep+h+o × 24 层）ms——delta 段口径（老路
//! 332ms 基线、vLLM 同模型 22.7ms）；② BT 三档（32/64/128）× vsplit
//! （1/2/4/8）表；③ BT=64 的分 kernel 分解（prep/h/o 各自事件计时——
//! fwd_h 块间串行是延迟链，链长 94）。
//! 状态流量账：h 读 S₀ + 写快照 + o 读快照 = chunks×vh×3×32KB ≈ 288MB
//! （老路每 token 读写一遍 = 288GB，千分之一）。
use wein_kernel::cuda::{Gpu, Stream};
use wein_kernel::ops::{DeltanetChunkedOp, OpTable, ScratchCaps};
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
    let gpu = Gpu::probe().expect("须有 CUDA 设备（本探针是设备实验）");
    let profile = HardwareProfile::with_device(Device::Cuda).with_arch("sm_89");
    // 容量按 6009 token 全档定尺：BT=32 需 188 块 → delta_chunk_tokens=12288
    // （chunks_cap=200）；exec 侧 t_max=8192 时 BT=32 全量超容走回退门
    let caps = ScratchCaps {
        delta_chunk_tokens: 12288,
        ..ScratchCaps::default()
    };
    let t = OpTable::resolve_tuned_with(&gpu, &profile, &[], &caps).expect("resolve");
    let ptx = std::fs::read_to_string(wein_kernel::ptx_path()).expect("PTX 产物");
    let m = gpu.load_module(&ptx);
    let (f_prep, f_h, f_o, l2f) = (
        m.func("dn_chunk_prep"),
        m.func("dn_chunk_h"),
        m.func("dn_chunk_o"),
        m.func("dn_qk_l2norm"),
    );
    let s = gpu.stream();

    let (kh, vh, kd, vd) = (16usize, 32usize, 128usize, 128usize);
    let (n_rows, tpr) = (1usize, 6009usize);
    let tokens = n_rows * tpr;
    let layers = 24usize;
    let iters = 20usize;

    // 带宽标定（同 dn_split_probe：地板随实测带宽漂移）
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
    eprintln!("D2D 带宽标定：{bw:.0} GB/s");

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
    let drs = gpu.alloc_from_slice(&vec![0u32; n_rows]);
    let drc = gpu.alloc_from_slice(&vec![tpr as u32; n_rows]);
    let dri = gpu.alloc_from_slice(&vec![0u32; n_rows]);
    let (tc, nch_max) = (t.deltanet_chunked.caps(), tokens.div_ceil(32));
    eprintln!(
        "\n== 6009 token 单 span（kh={kh} vh={vh} kd=vd=128，24 层等价；caps tokens={} chunks={} ≥ {nch_max}）==",
        tc.0, tc.1
    );

    // ── ①+② op 级整段：BT × vsplit 表 ──
    let (e0, e1) = (gpu.event(), gpu.event());
    eprintln!("  BT  vsplit   单层 µs   delta 段（×24 层）ms");
    for &bt in &[32u32, 64, 128] {
        let chunks = (tpr as u32).div_ceil(bt);
        let snap_off: Vec<u32> = vec![0, chunks];
        let dsn = gpu.alloc_from_slice(&snap_off);
        for &vsp in &[1u32, 2, 4, 8] {
            let run = |st: &Stream| {
                t.deltanet_chunked.run(&dq, &dk, &dv, &da, &db, &da_log, &ddtb, &dstate, &dout,
                    &drs, &drc, &dri, &dsn, n_rows as u32, chunks, kh as u32, vh as u32,
                    kd as u32, vd as u32, tokens as u32, 1e-6, bt, vsp, st);
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
            let layer_us = e1.elapsed_ms(&e0) as f64 / iters as f64 * 1000.0;
            eprintln!("  {bt:<3} {vsp:<6} {layer_us:9.2}   {:7.2}", layer_us * layers as f64 / 1000.0);
        }
    }

    // ── ③ BT=64 分 kernel 分解（手动 launch 独立缓冲）──
    let bt = 64usize;
    let chunks = tpr.div_ceil(bt);
    let inv = gpu.alloc_bytes(tokens * kh * 2 * 4);
    let gcum = gpu.alloc_bytes(tokens * vh * 4);
    let du = gpu.alloc_bytes(tokens * vh * vd * 4);
    // w：前半 f32 solve 区 + hi/lo bf16 镜像（8 字节/元素）
    let dw = gpu.alloc_bytes(tokens * vh * kd * 8);
    let wimg = (tokens * vh * kd) as u32;
    let (w_hi_off, w_lo_off) = (wimg * 2, wimg * 3);
    let snap = gpu.alloc_bytes(chunks * vh * vd * kd * 2);
    let sc1 = gpu.alloc_bytes(chunks * vh * 5 * bt * 4);
    let sc2 = gpu.alloc_bytes(1 * vh * 8 * (vd / 8 * kd + 64 + 8192) * 4);
    let snap_off: Vec<u32> = vec![0, chunks as u32];
    let dsn = gpu.alloc_from_slice(&snap_off);
    let (tk, tkh, tvh, tkd, tvd) = (tokens as u32, kh as u32, vh as u32, kd as u32, vd as u32);
    let (nr, nch, btu, eps) = (n_rows as u32, chunks as u32, bt as u32, 1e-6f32);
    let launch = |st: &Stream, sp: u32| {
        let n_l2 = tk * tkh;
        let a1 = wein_kernel::kernel_args![dq, dk, inv, tk, tkh, tkd, eps];
        gpu.launch(&l2f, n_l2.div_ceil(128), 128, &a1, st);
        let a2 = wein_kernel::kernel_args![dk, dv, da, db, da_log, ddtb, inv, gcum, du, dw,
            drs, drc, dsn, nr, nch, tkh, tvh, tkd, tvd, btu, snap, sc1, w_hi_off, w_lo_off];
        gpu.launch(&f_prep, nch * tkh, 256, &a2, st);
        let a3 = wein_kernel::kernel_args![dk, inv, gcum, du, dw, snap, dstate,
            drs, drc, dri, dsn, nr, tkh, tvh, tkd, tvd, btu, sp, sc2, w_hi_off, w_lo_off];
        let hbd = if sp == 8 { 128u32 } else { 256u32 };
        gpu.launch(&f_h, nr * tvh * sp, hbd, &a3, st);
        let a4 = wein_kernel::kernel_args![dq, dk, inv, gcum, du, snap, dout,
            drs, drc, dsn, nr, nch, tkh, tvh, tkd, tvd, btu, dw, sc1];
        gpu.launch(&f_o, nch * tkh, 256, &a4, st);
    };
    eprintln!("\n  BT=64 分 kernel 分解（vsplit=8；µs/层）");
    for &sp in &[1u32, 2, 4, 8] {
        for _ in 0..3 {
            launch(&s, sp);
        }
        s.sync();
        // 四段各自计时：段间事件
        let evs: Vec<_> = (0..=4).map(|_| gpu.event()).collect();
        evs[0].record_on(&s);
        let n_l2 = tk * tkh;
        let a1 = wein_kernel::kernel_args![dq, dk, inv, tk, tkh, tkd, eps];
        gpu.launch(&l2f, n_l2.div_ceil(128), 128, &a1, &s);
        evs[1].record_on(&s);
        let a2 = wein_kernel::kernel_args![dk, dv, da, db, da_log, ddtb, inv, gcum, du, dw,
            drs, drc, dsn, nr, nch, tkh, tvh, tkd, tvd, btu, snap, sc1, w_hi_off, w_lo_off];
        gpu.launch(&f_prep, nch * tkh, 256, &a2, &s);
        evs[2].record_on(&s);
        let a3 = wein_kernel::kernel_args![dk, inv, gcum, du, dw, snap, dstate,
            drs, drc, dri, dsn, nr, tkh, tvh, tkd, tvd, btu, sp, sc2, w_hi_off, w_lo_off];
        let hbd = if sp == 8 { 128u32 } else { 256u32 };
        gpu.launch(&f_h, nr * tvh * sp, hbd, &a3, &s);
        evs[3].record_on(&s);
        let a4 = wein_kernel::kernel_args![dq, dk, inv, gcum, du, snap, dout,
            drs, drc, dsn, nr, nch, tkh, tvh, tkd, tvd, btu, dw, sc1];
        gpu.launch(&f_o, nch * tkh, 256, &a4, &s);
        evs[4].record_on(&s);
        s.sync();
        let seg = |i: usize| evs[i].elapsed_ms(&evs[i - 1]) as f64 * 1000.0;
        let tot: f64 = (1..=4).map(seg).sum();
        eprintln!(
            "  vsplit={sp}: l2 {:7.2}  prep {:7.2}  h {:7.2}  o {:7.2}  合计 {tot:8.2} µs/层（×24 = {:.2} ms）",
            seg(1), seg(2), seg(3), seg(4),
            tot * layers as f64 / 1000.0
        );
    }
    // 状态流量账（千倍削减的对外数字）
    let traffic_mb = (chunks * vh * 3 * vd * kd * 2) as f64 / 1e6;
    let floor_ms = traffic_mb * 2.0 / 1e3 / bw * layers as f64; // 快照写+读两趟近似
    eprintln!("\n  状态流量：chunked ≈ {traffic_mb:.0} MB/层（快照写+读+池）→ {floor_ms:.1} ms/24 层 @ {bw:.0} GB/s（老路 288 GB/层 → 332 ms 实测）",);
}
