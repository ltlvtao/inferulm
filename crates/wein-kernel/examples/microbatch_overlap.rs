//! 判决性实验：**算力受限 kernel（gemm）∥ 带宽受限 kernel（deltanet）在两个
//! 独立 token 块上并发**能拿到多少重叠收益。纯测量——不碰 gemm/deltanet/
//! attention 的生产语义，只调 ops 层的公开入口。
//!
//!   cargo run --release -p wein-kernel --example microbatch_overlap
//!
//! 块 A（gemm，m=256）：一"层簇" = 9216×2560 / 8192×4096 / 2560×4096 /
//! 4096×2560 各一发（权重 156.24MB），摞 54 簇 ⇒ 8.44GB 权重（流量 9.47GB）。
//! 每簇权重是独立缓冲——真实层各持各的权重；复用同一块会让 L2（72MB）吃掉
//! 流量、把有效带宽做虚。
//!
//! 块 B（deltanet，256 token）：几何 kh=16 vh=32 kd=vd=128（Qwen3.5-4B 线性层），
//! 记账 = 每 token 每层状态 2·vh·vd·kd·2B = 2MB ⇒ 24 层 × 256 token × 2MB
//! = 12.29GB（对齐题面 12.3GB）。**行/token 的切法会显著改变实测**（行数定 L2
//! 命中、块数定 SM 占用、token/行定块长），故先跑一组几何扫描（4 档 + 1/8 档），
//! 取与题面 13.88ms 最接近的那档作主档；扫描表本身就是"delta 卡在哪"的证据。
//!
//! 四跑法（各 2 热身 + 5 拍取中位，全部 **CUDA event** 计时，非 host 墙钟）
//! - S   ：串行（gemm 全跑完 → delta 全跑完，同流）
//! - O2  ：双流并发，gemm 先发（s0=gemm ∥ s1=delta）
//! - O2r ：双流并发，delta 先发（排除发射顺序偏差）
//! - OJ  ：单流交替（每 9 发 gemm 插一发 delta —— 模拟融合内核的交错节奏）
//! 另一组把 delta 的 token 按 1/8（32 token，1.54GB）跑同四式。
use std::time::Instant;

use wein_kernel::cuda::{DevBuf, Event, Gpu, Stream};
use wein_kernel::ops::{DeltanetOp, GemmOp, OpTable};
use wein_platform::{Device, HardwareProfile};

const M: u32 = 256;
const LAYERS: usize = 24;
const G_TARGET: f64 = 8.4e9;

const KH: usize = 16;
const VH: usize = 32;
const KD: usize = 128;
const VD: usize = 128;
const POOL_ROWS: usize = 128; // exec 的 MAX_ROWS：状态池池深（1MB/行/层）
const TOKENS: usize = 256; // 本步 token 总数（chunk256）

const SHAPES: [(u32, u32); 4] = [(9216, 2560), (8192, 4096), (2560, 4096), (4096, 2560)];

/// 一档 delta 几何：rows 行 × tpr token/行 × calls 发/层（rows·tpr·calls = 该档 token 数）
struct DCfg<'a> {
    label: &'static str,
    rows: usize,
    tpr: usize,
    calls: usize,
    start: DevBuf<'a>,
    count: DevBuf<'a>,
}

fn bf16_vec(seed_hi: u64, n: usize, scale: f32) -> Vec<u16> {
    (0..n as u64)
        .map(|i| {
            let z = i.wrapping_mul(6364136223846793005).wrapping_add(seed_hi);
            let f = ((z >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * scale;
            f.to_bits() as u16
        })
        .collect()
}

struct Ev<'a> {
    g0: Event<'a>,
    g1: Event<'a>,
    d0: Event<'a>,
    d1: Event<'a>,
}

impl Ev<'_> {
    /// 一次计时：两流各记 start → f() 发行 → 各记 end → 双流 sync。
    /// 返回 (跨流真实总跨度, s0 侧跨度, s1 侧跨度)——跨流 elapsed 合法（同一时间轴）
    fn once(&self, s0: &Stream, s1: &Stream, f: &mut dyn FnMut()) -> (f64, f64, f64) {
        self.g0.record_on(s0);
        self.d0.record_on(s1);
        f();
        self.g1.record_on(s0);
        self.d1.record_on(s1);
        s0.sync();
        s1.sync();
        let t0 = self.g1.elapsed_ms(&self.g0) as f64;
        let t1 = self.d1.elapsed_ms(&self.d0) as f64;
        let span = if t1 <= 0.0 {
            t0
        } else if t0 <= 0.0 {
            t1
        } else {
            let off = self.d0.elapsed_ms(&self.g0) as f64; // s1 start 晚于 s0 start 的偏移
            if off >= 0.0 {
                (off + t1).max(t0)
            } else {
                (-off + t0).max(t1)
            }
        };
        (span, t0, t1)
    }
}

/// reps 拍取中位（每分量各自排序取中——各拍发序相同，分量可配对）
fn repeated(ev: &Ev, s0: &Stream, s1: &Stream, reps: usize, f: &mut dyn FnMut()) -> (f64, f64, f64) {
    let v: Vec<(f64, f64, f64)> = (0..reps).map(|_| ev.once(s0, s1, f)).collect();
    let med = |idx: fn(&(f64, f64, f64)) -> f64| {
        let mut c: Vec<f64> = v.iter().map(idx).collect();
        c.sort_by(|a, b| a.partial_cmp(b).expect("无 NaN"));
        c[c.len() / 2]
    };
    (med(|x| x.0), med(|x| x.1), med(|x| x.2))
}

fn main() {
    let gpu = Gpu::probe().expect("须有 CUDA 设备（本探针是设备实验）");
    let profile = HardwareProfile::with_device(Device::Cuda).with_arch("sm_89");
    let t = OpTable::resolve(&gpu, &profile).expect("sm_89 画像须解析成功");
    let s0 = gpu.stream();
    let s1 = gpu.stream();

    // ── 块 A：gemm 权重（每簇独立缓冲，总量对齐 8.4GB）──
    let cluster_bytes: usize = SHAPES.iter().map(|&(n, k)| n as usize * k as usize * 2).sum();
    let iters = (G_TARGET / cluster_bytes as f64).ceil() as usize;
    let mut w: Vec<Vec<DevBuf>> = Vec::with_capacity(iters);
    for _ in 0..iters {
        w.push(SHAPES.iter().map(|&(n, k)| gpu.alloc_bytes(n as usize * k as usize * 2)).collect());
    }
    let ag = gpu.alloc_from_slice(&bf16_vec(0x9a00, M as usize * 2560, 1.0)); // [M,2560]
    let ao = gpu.alloc_from_slice(&bf16_vec(0x9b00, M as usize * 4096, 1.0)); // [M,4096]
    let og: Vec<DevBuf> =
        SHAPES.iter().map(|&(n, _)| gpu.alloc_bytes(M as usize * n as usize * 2)).collect();
    for row in &w {
        for b in row {
            b.zero_on(&s0); // 数值与计时无关（零 bf16 无 NaN/denormal）
        }
    }
    s0.sync();

    // ── 块 B：deltanet（独立状态池/输入，与块 A 无别名）──
    let mk = |seed: u64, n: usize| gpu.alloc_from_slice(&bf16_vec(seed, n, 1.0));
    let dq = mk(0x6a12, TOKENS * KH * KD);
    let dk = mk(0x6b34, TOKENS * KH * KD);
    let dv = mk(0x6c56, TOKENS * VH * VD);
    let da = mk(0x6d78, TOKENS * VH);
    let db = mk(0x6e9a, TOKENS * VH);
    let da_log = gpu.alloc_from_slice(&(0..VH).map(|h| 0.3 + 0.2 * h as f32).collect::<Vec<_>>());
    let ddtb = mk(0x6fbc, VH);
    let dout = gpu.alloc_bytes(TOKENS * VH * VD * 2);
    let row_bytes = VH * VD * KD * 2; // 单行状态字节（bf16 池，1MB）
    let mut state: Vec<DevBuf> = Vec::with_capacity(LAYERS);
    for _ in 0..LAYERS {
        let b = gpu.alloc_bytes(POOL_ROWS * row_bytes);
        b.zero_on(&s1);
        state.push(b);
    }
    let d_row_id = gpu.alloc_from_slice(&(0..POOL_ROWS as u32).collect::<Vec<_>>());
    let d_specs = gpu.alloc_from_slice(&vec![0u32; POOL_ROWS]);

    // ── 对照：纯带宽 kernel（D2D 拷贝，SM 足迹极小）——用来把"delta 抢带宽"
    // 与"delta 抢 SM 资源"分开 ──
    const CPY_CHUNK: usize = 512 << 20;
    const CPY_N: usize = 13; // 13 × (512MB 读 + 512MB 写) = 13.0GB ≈ delta 记账量
    let csrc = gpu.alloc_bytes(CPY_CHUNK);
    let cdst = gpu.alloc_bytes(CPY_CHUNK);

    // 几何扫描四档（rows·tpr·calls == 256 token）
    let mut cfgs: Vec<DCfg> = Vec::new();
    for &(label, rows, tpr, calls) in &[
        ("1×256", 1usize, 256usize, 1usize),
        ("8×32", 8, 32, 1),
        ("128×2", 128, 2, 1),
        ("128×1×2发", 128, 1, 2),
    ] {
        assert_eq!(rows * tpr * calls, TOKENS);
        let start =
            gpu.alloc_from_slice(&(0..rows).map(|r| (r * tpr) as u32).collect::<Vec<_>>());
        let count = gpu.alloc_from_slice(&vec![tpr as u32; rows]);
        cfgs.push(DCfg { label, rows, tpr, calls, start, count });
    }
    // 1/8 档：1 行 × 32 token（同一支真实 prefill 几何，状态流 1.54GB）
    let short = DCfg {
        label: "1×32(1/8)",
        rows: 1,
        tpr: 32,
        calls: 1,
        start: gpu.alloc_from_slice(&[0u32]),
        count: gpu.alloc_from_slice(&[32u32]),
    };
    s1.sync();

    // ── 字节账 ──
    let g_bytes = iters as f64
        * (cluster_bytes as f64
            + SHAPES
                .iter()
                .map(|&(n, k)| (M as usize * n as usize * 2 + M as usize * k as usize * 2) as f64)
                .sum::<f64>());
    let g_flops = iters as f64
        * SHAPES.iter().map(|&(n, k)| 2.0 * M as f64 * n as f64 * k as f64).sum::<f64>();
    // 每 token 状态 r+w = 2·vh·vd·kd·2B；另加 q/k/v/a/b 读 + out 写
    let per_token = 2.0 * (VH * VD * KD * 2) as f64;
    let per_token_io = (2 * KH * KD * 2 + VH * VD * 2 * 2 + 2 * VH * 2) as f64;
    let d_bytes = |tok: usize| (LAYERS * tok) as f64 * (per_token + per_token_io);

    // ── 发行原语 ──
    let gemm_at = |s: &Stream, it: usize, si: usize| {
        let (n, k) = SHAPES[si];
        let a = if k == 2560 { &ag } else { &ao };
        t.gemm.run_wt(a, &w[it][si], &og[si], M, n, k, s);
    };
    let delta_at = |s: &Stream, l: usize, rows: usize, tokens: usize, st: &DevBuf, rc: &DevBuf| {
        t.deltanet.run(
            &dq, &dk, &dv, &da, &db, &da_log, &ddtb, &state[l], &dout, st, rc, &d_row_id,
            rows as u32, KH as u32, VH as u32, KD as u32, VD as u32, tokens as u32, 1e-6, &d_specs,
            if tokens > rows { 16 } else { 1 }, s,
        );
    };
    let gemm_all = |s: &Stream| {
        for it in 0..iters {
            for si in 0..4 {
                gemm_at(s, it, si);
            }
        }
    };
    let delta_run = |s: &Stream, c: &DCfg| {
        for l in 0..LAYERS {
            for _ in 0..c.calls {
                delta_at(s, l, c.rows, c.rows * c.tpr, &c.start, &c.count);
            }
        }
    };

    // ── 热身 ──
    for _ in 0..2 {
        gemm_all(&s0);
        for c in &cfgs {
            delta_run(&s1, c);
        }
        delta_run(&s1, &short);
    }
    s0.sync();
    s1.sync();
    gemm_all(&s0);
    s0.sync();

    let ev = Ev { g0: gpu.event(), g1: gpu.event(), d0: gpu.event(), d1: gpu.event() };
    let reps = 5usize;

    // ── delta 几何扫描（每档 24 层 × 256 token，记账 12.29GB 全同）──
    let mut sweep: Vec<(&'static str, f64)> = Vec::new(); // (label, ms)
    for c in &cfgs {
        let r = repeated(&ev, &s0, &s1, reps, &mut || delta_run(&s1, c));
        sweep.push((c.label, r.2));
    }
    let rs = repeated(&ev, &s0, &s1, reps, &mut || delta_run(&s1, &short));
    sweep.push((short.label, rs.2));

    // 主档 = 与前表 13.88ms 最接近的那档
    let target = 13.88;
    let pick = sweep
        .iter()
        .enumerate()
        .take(cfgs.len())
        .min_by(|a, b| (a.1 .1 - target).abs().partial_cmp(&(b.1 .1 - target).abs()).expect("无 NaN"))
        .map(|(i, _)| i)
        .expect("扫描非空");
    let main = &cfgs[pick];

    // ── 单跑基线 ──
    let g_only = repeated(&ev, &s0, &s1, reps, &mut || gemm_all(&s0));
    let d_only = repeated(&ev, &s0, &s1, reps, &mut || delta_run(&s1, main));
    let d_only_s = repeated(&ev, &s0, &s1, reps, &mut || delta_run(&s1, &short));

    // ── 四跑法（全 delta 字节）──
    let ser = repeated(&ev, &s0, &s1, reps, &mut || {
        gemm_all(&s0);
        delta_run(&s0, main);
    });
    let o2 = repeated(&ev, &s0, &s1, reps, &mut || {
        gemm_all(&s0);
        delta_run(&s1, main);
    });
    let o2r = repeated(&ev, &s0, &s1, reps, &mut || {
        delta_run(&s1, main);
        gemm_all(&s0);
    });
    let oj = repeated(&ev, &s0, &s1, reps, &mut || {
        oj_loop(&gemm_at, &delta_at, main, &s0, iters)
    });

    // ── delta 1/8（32 token，1.54GB）同四式 ──
    let ser_s = repeated(&ev, &s0, &s1, reps, &mut || {
        gemm_all(&s0);
        delta_run(&s0, &short);
    });
    let o2_s = repeated(&ev, &s0, &s1, reps, &mut || {
        gemm_all(&s0);
        delta_run(&s1, &short);
    });
    let o2r_s = repeated(&ev, &s0, &s1, reps, &mut || {
        delta_run(&s1, &short);
        gemm_all(&s0);
    });
    let oj_s = repeated(&ev, &s0, &s1, reps, &mut || {
        oj_loop(&gemm_at, &delta_at, &short, &s0, iters)
    });

    // ── 对照：gemm ∥ 纯带宽 kernel（同 13.0GB 流量）──
    let cpy_all = |s: &Stream| {
        for _ in 0..CPY_N {
            cdst.copy_range_on(0, &csrc, 0, CPY_CHUNK, s);
        }
    };
    for _ in 0..2 {
        cpy_all(&s1);
    }
    s1.sync();
    let cpy_dst = CPY_CHUNK as f64 * 2.0 * CPY_N as f64;
    let c_only = repeated(&ev, &s0, &s1, reps, &mut || cpy_all(&s1));
    let c_ser = repeated(&ev, &s0, &s1, reps, &mut || {
        gemm_all(&s0);
        cpy_all(&s0);
    });
    let c_o2 = repeated(&ev, &s0, &s1, reps, &mut || {
        gemm_all(&s0);
        cpy_all(&s1);
    });

    // ── 发行开销 + 实际起跑延迟（host 发行是 O2/O2r 偏差的根源）──
    let t_enq_g = {
        let t0 = Instant::now();
        gemm_all(&s0);
        let d = t0.elapsed().as_secs_f64() * 1e3;
        s0.sync();
        d
    };
    let t_enq_d = {
        let t0 = Instant::now();
        delta_run(&s1, main);
        let d = t0.elapsed().as_secs_f64() * 1e3;
        s1.sync();
        d
    };
    ev.g0.record_on(&s0);
    gemm_all(&s0); // O2 序：216 发 gemm 先入队（host 占满 t_enq_g）
    ev.d0.record_on(&s1); // 此刻 s1 才可能起跑
    delta_run(&s1, main);
    s0.sync();
    s1.sync();
    let start_delay = ev.d0.elapsed_ms(&ev.g0) as f64;

    // ── 报表 ──
    let gb = |b: f64, ms: f64| b / (ms * 1e-3) / 1e9;
    let tf = |f: f64, ms: f64| f / (ms * 1e-3) / 1e12;
    eprintln!(
        "gemm 簇 {iters} × {:.1}MB（权重 {:.2}GB / 流量 {:.2}GB）｜形状 {:?}｜m={M}",
        cluster_bytes as f64 / 1e6,
        iters as f64 * cluster_bytes as f64 / 1e9,
        g_bytes / 1e9,
        SHAPES
    );
    eprintln!(
        "delta 记账：24 层 × 256 token × 2MB/token = {:.2}GB（1/8 档 {:.2}GB）；池 {POOL_ROWS} 行 × 1MB/层",
        d_bytes(TOKENS) / 1e9,
        d_bytes(32) / 1e9
    );
    eprintln!();
    eprintln!("【delta 几何扫描】同记账字节，只改行/token 切法（目标 = 题面 13.88ms）");
    eprintln!("{:<12}{:>9}{:>10}", "档", "ms", "记账GB/s");
    for &(label, ms) in &sweep {
        let tok = if label.contains("1/8") { 32 } else { TOKENS };
        eprintln!("{label:<12}{ms:>9.2}{:>10.0}", gb(d_bytes(tok), ms));
    }
    let (dg, db) = delta_geom(main.rows, main.rows * main.tpr);
    eprintln!(
        "  ← 主档取最接近 13.88ms 者：{}（{:.2}ms，发行几何 {dg} 块 × {db} 线程 = {} 线程）",
        main.label,
        sweep[pick].1,
        dg as usize * db as usize
    );
    eprintln!();
    eprintln!(
        "{:<11}{:>9}{:>9}{:>9}{:>10}{:>8}{:>7}",
        "变体", "总ms", "gemm段", "delta段", "聚合GB/s", "gemmTF", "倍数"
    );
    let row = |label: &str, r: (f64, f64, f64), bytes: f64, flops: f64, base: f64| {
        let (tot, a, b) = r;
        let num = |x: f64| {
            if x > 0.0 {
                format!("{x:8.2}")
            } else {
                format!("{:>8}", "-")
            }
        };
        eprintln!(
            "{label:<11}{:>9.2}{}{}{:>10.0}{:>8.1}{:>7.3}",
            tot,
            num(a),
            num(b),
            gb(bytes, tot),
            if flops > 0.0 { tf(flops, tot) } else { 0.0 },
            if base > 0.0 { base / tot } else { 0.0 },
        );
    };
    let sum = g_only.0 + d_only.0;
    let sum_s = g_only.0 + d_only_s.0;
    row("G_only", g_only, g_bytes, g_flops, 0.0);
    row("D_only", d_only, d_bytes(TOKENS), 0.0, 0.0);
    row("D_only1/8", d_only_s, d_bytes(32), 0.0, 0.0);
    row("S", ser, g_bytes + d_bytes(TOKENS), g_flops, sum);
    row("O2", o2, g_bytes + d_bytes(TOKENS), g_flops, sum);
    row("O2r", o2r, g_bytes + d_bytes(TOKENS), g_flops, sum);
    row("OJ", oj, g_bytes + d_bytes(TOKENS), g_flops, sum);
    row("S-1/8", ser_s, g_bytes + d_bytes(32), g_flops, sum_s);
    row("O2-1/8", o2_s, g_bytes + d_bytes(32), g_flops, sum_s);
    row("O2r-1/8", o2r_s, g_bytes + d_bytes(32), g_flops, sum_s);
    row("OJ-1/8", oj_s, g_bytes + d_bytes(32), g_flops, sum_s);
    let csum = g_only.0 + c_only.0;
    row("Cpy_only", c_only, cpy_dst, 0.0, 0.0);
    row("S+copy", c_ser, g_bytes + cpy_dst, g_flops, csum);
    row("O2+copy", c_o2, g_bytes + cpy_dst, g_flops, csum);
    eprintln!();
    eprintln!("gemm段/delta段 = 各模式下该流自身的事件跨度；S/OJ 单流不可分解 ⇒ \"-\"");
    eprintln!("倍数 = (G_only + D_only) / 本行总耗时（>1 才有重叠收益）");
    eprintln!();
    eprintln!("【发行开销（host）】216 发 gemm {t_enq_g:.2}ms｜24 发 delta {t_enq_d:.2}ms");
    eprintln!(
        "  O2 序下 delta 实际起跑延迟（g0 → s1 有空可跑）= {start_delay:.2}ms（纯 host 发行排队）"
    );
    eprintln!();

    // ── 判词数据 ──
    let tsum = g_only.0 + d_only.0;
    let floor = g_only.0.max(d_only.0);
    eprintln!("【判词数据】");
    eprintln!(
        "  单跑：gemm {:.2}ms（{:.0}GB/s、{:.1}TFLOPS）｜delta {:.2}ms（{:.0}GB/s）｜和 {:.2}ms",
        g_only.0,
        gb(g_bytes, g_only.0),
        tf(g_flops, g_only.0),
        d_only.0,
        gb(d_bytes(TOKENS), d_only.0),
        tsum
    );
    eprintln!(
        "  理论地板 max(算力段, 带宽段) = {:.2}ms ⇒ 上限 {:.3}×（题面 1.5× 的实测版）",
        floor,
        tsum / floor
    );
    eprintln!(
        "  S {:.2} → O2 {:.2}（{:.3}×，省 {:.1}%）｜O2r {:.2}（{:.3}×）｜OJ {:.2}（{:.3}×）",
        ser.0,
        o2.0,
        ser.0 / o2.0,
        (1.0 - o2.0 / ser.0) * 100.0,
        o2r.0,
        ser.0 / o2r.0,
        oj.0,
        ser.0 / oj.0
    );
    eprintln!(
        "  争用溢价：gemm 段 {:.2}ms（×{:.3}）｜delta 段 {:.2}ms（×{:.3}）",
        o2.1,
        o2.1 / g_only.0,
        o2.2,
        o2.2 / d_only.0
    );
    eprintln!(
        "  O2 距地板 {:.1}%（{:.2}ms vs {:.2}ms）",
        (o2.0 / floor - 1.0) * 100.0,
        o2.0,
        floor
    );
    eprintln!(
        "  对照 gemm∥D2D 拷贝（13.0GB 纯带宽、SM 足迹极小）：拷贝单跑 {:.2}ms（{:.0}GB/s）｜S {:.2} → O2 {:.2}（{:.3}×）｜gemm 段 ×{:.3}",
        c_only.0,
        gb(cpy_dst, c_only.0),
        c_ser.0,
        c_o2.0,
        c_ser.0 / c_o2.0,
        c_o2.1 / g_only.0
    );
    eprintln!(
        "  delta 1/8（{:.2}ms，{:.0}GB/s）：S {:.2} → O2 {:.3}×｜O2r {:.3}×｜OJ {:.3}×｜串行和 {:.2}ms、地板 {:.2}ms ⇒ 上限 {:.3}×",
        d_only_s.0,
        gb(d_bytes(32), d_only_s.0),
        ser_s.0,
        ser_s.0 / o2_s.0,
        ser_s.0 / o2r_s.0,
        ser_s.0 / oj_s.0,
        sum_s,
        g_only.0.max(d_only_s.0),
        sum_s / g_only.0.max(d_only_s.0)
    );
}

/// OJ：单流交替——每 per 发 gemm 插一发 delta（216 发 gemm / 24 层 = 9）
fn oj_loop(
    gemm_at: &dyn Fn(&Stream, usize, usize),
    delta_at: &dyn Fn(&Stream, usize, usize, usize, &DevBuf, &DevBuf),
    c: &DCfg,
    s: &Stream,
    iters: usize,
) {
    let per = (iters * 4) / LAYERS;
    let (mut k, mut l) = (0usize, 0usize);
    for it in 0..iters {
        for si in 0..4 {
            gemm_at(s, it, si);
            if (k + 1) % per == 0 && l < LAYERS {
                for _ in 0..c.calls {
                    delta_at(s, l, c.rows, c.rows * c.tpr, &c.start, &c.count);
                }
                l += 1;
            }
            k += 1;
        }
    }
    assert_eq!(l, LAYERS, "OJ 交错未覆盖全部层（{l}/{LAYERS}）");
}

/// 复刻 `DeltanetOp::run` 内的 vsplit/块宽门（只为报表——不改生产语义）
fn delta_geom(rows: usize, tokens: usize) -> (u32, u32) {
    const BLOCK_CAP: usize = 4096;
    const SPLIT_TARGET: usize = 512;
    const SPLIT_MAX: usize = 512;
    const SPLIT: usize = 16;
    const SPLIT_DIM: usize = 64;
    const BLOCK_DIM: usize = 128;
    const WIDE_DIM: usize = 256;
    const WIDE_GRID_MAX: usize = 64;
    let v_heads = VH;
    let v_dim = VD;
    let base = (rows * v_heads).max(1);
    let cap = BLOCK_CAP / base;
    let mut s = (SPLIT_TARGET / base).min(if tokens > rows { SPLIT } else { 1 }).clamp(1, SPLIT);
    let ok = |s: usize| {
        if v_dim % s != 0 {
            return false;
        }
        let rows_pw = (v_dim / s) / (SPLIT_DIM / 32);
        rows_pw >= 4 && rows_pw <= 32 && rows_pw.is_power_of_two()
    };
    if base >= SPLIT_MAX {
        s = 1;
    }
    while s > 1 && (s > cap || !ok(s)) {
        s -= 1;
    }
    let grid = rows * v_heads * s;
    let bdim = if s > 1 {
        SPLIT_DIM as u32
    } else if grid <= WIDE_GRID_MAX {
        WIDE_DIM as u32
    } else {
        BLOCK_DIM as u32
    };
    (grid as u32, bdim)
}
