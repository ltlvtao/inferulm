//! 多流重叠决定性实验（§7 前置账）：delta（状态池带宽地板，629MB r+w/
//! launch）与层 GEMM 簇（权重带宽 + TC，234MB/层）——今天 exec 同流串行，
//! 重叠是否有肉取决于**混合流的聚合带宽**能否超过各自单跑的 ~800GB/s。
//!
//! 形状全部取自真实 m=100 满负荷 decode：
//! - delta：o22 形状（100 行 × 16 k 头 × 32 v 头 × 128²，3 趟状态访存）
//! - gemm 簇：线性层全量（qkv 8192×2560 + z 4096×2560 + a/b 32×2560 +
//!   out 2560×4096 + MLP wg/wu/wd 9728×2560）——与 8.4GB/步权重账对齐
//!
//! 四模式（各 3 热身 + 20 拍事件计时）：
//!   delta_only / gemm_only —— 各自单独的带宽地板（校准用）
//!   serial    —— delta+簇 同流交替（今天 exec 的形态）
//!   dual      —— delta 走 s0、簇走 s1，fork/join 并发（重叠天花板）
//! 判决：dual 聚合带宽显著高于 serial（≥ +10%）→ exec 双流重构有肉；
//! 否则证伪——DRAM 单墙，重叠换不来带宽（记录归档，另择杠杆）。
use wein_kernel::cuda::Gpu;
use wein_kernel::ops::{DeltanetOp, GemmOp, OpTable};
use wein_platform::{Device, HardwareProfile};

fn bf16_vec(seed_hi: u64, n: usize, scale: f32) -> Vec<u16> {
    (0..n as u64).map(|i| {
        let z = i.wrapping_mul(6364136223846793005).wrapping_add(seed_hi);
        let f = ((z >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * scale;
        f.to_bits() as u16 // 低位截断即 bf16 舍入——计时与值无关
    }).collect()
}

fn main() {
    let gpu = Gpu::probe().expect("须有 CUDA 设备（overlap probe 是设备实验）");
    let profile = HardwareProfile::with_device(Device::Cuda).with_arch("sm_89");
    let t = OpTable::resolve(&gpu, &profile).expect("sm_89 画像须解析成功");
    let s0 = gpu.stream();
    let s1 = gpu.stream();

    // ── delta：o22 满负荷形状 ──
    let (n_rows, kh, vh, kd, vd) = (100usize, 16usize, 32usize, 128usize, 128usize);
    let tokens = n_rows;
    let mk = |seed: u64, n: usize| gpu.alloc_from_slice(&bf16_vec(seed, n, 1.0));
    let dq = mk(0x6a12, tokens * kh * kd);
    let dk = mk(0x6b34, tokens * kh * kd);
    let dv = mk(0x6c56, tokens * vh * vd);
    let da = mk(0x6d78, tokens * vh);
    let db = mk(0x6e9a, tokens * vh);
    let da_log = gpu.alloc_from_slice(&(0..vh).map(|h| 0.3 + 0.2 * h as f32).collect::<Vec<_>>());
    let ddtb = mk(0x6fbc, vh);
    let state_len = n_rows * vh * vd * kd;
    let dstate = gpu.alloc_from_slice(&vec![0f32; state_len]);
    let dout = gpu.alloc_bytes(tokens * vh * vd * 2);
    let drs = gpu.alloc_from_slice(&(0..n_rows as u32).collect::<Vec<_>>());
    let drc = gpu.alloc_from_slice(&vec![1u32; n_rows]);
    let dri = gpu.alloc_from_slice(&(0..n_rows as u32).collect::<Vec<_>>());

    let delta_bytes = (state_len * 4 * 3) as f64; // 3 趟状态 r+w（o22 同账）

    let delta = |s: &wein_kernel::cuda::Stream| {
        t.deltanet.run(&dq, &dk, &dv, &da, &db, &da_log, &ddtb, &dstate, &dout, &drs, &drc, &dri,
            n_rows as u32, kh as u32, vh as u32, kd as u32, vd as u32, 1e-6, s);
    };

    // ── gemm 簇：线性层全量（m=100, h=2560, ii=9728）──
    let (m, h, ii) = (100usize, 2560usize, 9728usize);
    let w = |n: usize, k: usize| gpu.alloc_from_slice(&bf16_vec(0x7000 + n as u64, n * k, 0.05));
    let wqkv = w(2 * kh * kd + vh * vd, h); // 8192×2560
    let wz = w(vh * vd, h);                 // 4096×2560
    let wa = w(32, h);
    let wb = w(32, h);
    let wout = w(h, vh * vd);               // [n=2560, k=4096]
    let wg = w(ii, h);
    let wu = w(ii, h);
    let wd = w(h, ii);                      // [n=2560, k=9728]
    let n1 = gpu.alloc_from_slice(&bf16_vec(0x9100, m * h, 1.0));      // [m,h]
    let normed = gpu.alloc_from_slice(&bf16_vec(0x9200, m * vh * vd, 1.0)); // [m,4096]
    let n2 = gpu.alloc_from_slice(&bf16_vec(0x9300, m * h, 1.0));      // [m,h]
    let mmid = gpu.alloc_from_slice(&bf16_vec(0x9400, m * ii, 1.0));   // [m,ii]
    let qkvo = gpu.alloc_bytes(m * (2 * kh * kd + vh * vd) * 2);
    let zo = gpu.alloc_bytes(m * vh * vd * 2);
    let ao = gpu.alloc_bytes(m * 32 * 2);
    let bo = gpu.alloc_bytes(m * 32 * 2);
    let oo = gpu.alloc_bytes(m * h * 2);
    let go = gpu.alloc_bytes(m * ii * 2);
    let uo = gpu.alloc_bytes(m * ii * 2);
    let ddo = gpu.alloc_bytes(m * h * 2);
    let m32 = m as u32;

    // 权重 + 激活 r+w 的字节账（带宽对账用）
    let gemm_bytes = (wqkv.bytes() + wz.bytes() + wa.bytes() + wb.bytes() + wout.bytes()
        + wg.bytes() + wu.bytes() + wd.bytes()
        + n1.bytes() * 4 + normed.bytes() * 2 + n2.bytes() * 2 + mmid.bytes() * 2
        + qkvo.bytes() + zo.bytes() + ao.bytes() + bo.bytes() + oo.bytes()
        + go.bytes() + uo.bytes() + ddo.bytes()) as f64;

    let cluster = |s: &wein_kernel::cuda::Stream| {
        t.gemm.run_wt(&n1, &wqkv, &qkvo, m32, (2 * kh * kd + vh * vd) as u32, h as u32, s);
        t.gemm.run_wt(&n1, &wz, &zo, m32, (vh * vd) as u32, h as u32, s);
        t.gemm.run_wt(&n1, &wa, &ao, m32, 32, h as u32, s);
        t.gemm.run_wt(&n1, &wb, &bo, m32, 32, h as u32, s);
        t.gemm.run_wt(&normed, &wout, &oo, m32, h as u32, (vh * vd) as u32, s);
        t.gemm.run_wt(&n2, &wg, &go, m32, ii as u32, h as u32, s);
        t.gemm.run_wt(&n2, &wu, &uo, m32, ii as u32, h as u32, s);
        t.gemm.run_wt(&mmid, &wd, &ddo, m32, h as u32, ii as u32, s);
    };

    let iters = 20;
    // 热身（两流各跑——JIT/缓存/heuristic 就位）
    for _ in 0..3 {
        delta(&s0);
        cluster(&s1);
    }
    s0.sync();
    s1.sync();

    let (a0, b0) = (gpu.event(), gpu.event()); // 单流计时对
    let (f0s, f0e, f1s, f1e) = (gpu.event(), gpu.event(), gpu.event(), gpu.event()); // 双流计时

    let report = |label: &str, ms: f64, bytes: f64| {
        let per_iter = ms / iters as f64;
        eprintln!("{label:12} {ms:9.3} ms  {per_iter:7.3} ms/iter  {:7.0} GB/s聚合",
            bytes / (ms / 1e3) / 1e9);
    };

    // ① delta 单跑
    a0.record_on(&s0);
    for _ in 0..iters { delta(&s0); }
    b0.record_on(&s0);
    s0.sync();
    let t_delta = b0.elapsed_ms(&a0) as f64;
    report("delta_only", t_delta, delta_bytes * iters as f64);

    // ② gemm 簇单跑
    a0.record_on(&s0);
    for _ in 0..iters { cluster(&s0); }
    b0.record_on(&s0);
    s0.sync();
    let t_gemm = b0.elapsed_ms(&a0) as f64;
    report("gemm_only", t_gemm, gemm_bytes * iters as f64);

    // ③ 串行（今天 exec 的形态：同流交替）
    a0.record_on(&s0);
    for _ in 0..iters { delta(&s0); cluster(&s0); }
    b0.record_on(&s0);
    s0.sync();
    let t_serial = b0.elapsed_ms(&a0) as f64;
    report("serial", t_serial, (delta_bytes + gemm_bytes) * iters as f64);

    // ④ 双流并发（fork：delta@s0 ∥ 簇@s1；join：双流 end 事件取 max）
    f0s.record_on(&s0);
    f1s.record_on(&s1);
    for _ in 0..iters { delta(&s0); }
    for _ in 0..iters { cluster(&s1); }
    f0e.record_on(&s0);
    f1e.record_on(&s1);
    s0.sync();
    s1.sync();
    let t_dual = f0e.elapsed_ms(&f0s).max(f1e.elapsed_ms(&f1s)) as f64;
    report("dual", t_dual, (delta_bytes + gemm_bytes) * iters as f64);

    let gain = (t_serial - t_dual) / t_serial * 100.0;
    eprintln!("\n串行 {t_serial:.3} ms → 双流 {t_dual:.3} ms（省 {gain:.1}%；单跑和 {:.3}，理想重排 {:.3}）",
        t_delta + t_gemm, t_delta.max(t_gemm));
    eprintln!(concat!("判决：dual 聚合带宽比 serial 高 ≥10% → exec 双流重构有肉；",
        "≈0% → 证伪（DRAM 单墙，重叠换不来带宽）"));
}
