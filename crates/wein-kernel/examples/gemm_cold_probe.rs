//! gemm 选型偏差探针：**热态口径 vs 冷态口径**逐候选实测（m=100 真实形状集）。
//!
//!   cargo run --release -p wein-kernel --example gemm_cold_probe -- [m] [rounds]
//!
//! 要回答的问题（调优器选型是否偏）：
//!   ① heuristic 到底给几个候选、各自工作区/wave 数（现调优器只测前 4）；
//!   ② 每个候选在两个口径下的单发耗时——
//!      · **warm**：连续发、不冲洗（= 现调优器口径，L2 全热）；
//!      · **cold**：每发前用 128MB 缓冲做一次**读扫**（真实流水里每层权重
//!        都是冷的，8GB/步流过 72MB L2）；
//!   ③ warm-argmin 与 cold-argmin 是否同一候选（= "改调优器值不值"）。
//!
//! 读扫 = 调优器同一份 `BlasT::l2_flush`（内联 PTX：`ld.global.nc.v4.u32` +
//! 不可达条件写——**纯读**：写会把脏行留 L2，反被后续 gemm 的写回挤占）。
//! 协议：每轮每候选 = 1 次读扫 → 第 1 发计 cold → 第 2 发弃（回热）→
//! 第 3 发计 warm；轮间交错（抵消时钟漂移），取中位。
//! **整轮一次性入队、轮末才 sync**：事件对只在流内定位 GPU 时距——中途
//! 同步会让小形状的 warm 口径量到 host 提交间隙（实测 a/b 形状 warm 23.6µs
//! 假象 vs 真实 8µs），量到的是发射速度而非候选本身。
//!
//! 另附读扫自身带宽（= 本机纯读地板），供 cold GB/s 折算对照。
use wein_kernel::cuda::{BlasLt, Event, Gpu, Stream};

/// L2 稀释缓冲：72MB L2 上 128MB 顺序读扫，读完后 L2 只剩本缓冲尾部
const FLUSH_BYTES: usize = wein_kernel::cuda::cublaslt::L2_FLUSH_BYTES;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn timed(
    gpu: &Gpu,
    stream: &Stream,
    ev0: &Event,
    ev1: &Event,
    f: impl FnOnce(),
) -> f64 {
    let _ = gpu;
    ev0.record_on(stream);
    f();
    ev1.record_on(stream);
    stream.sync();
    ev1.elapsed_ms(ev0) as f64
}

/// m=100 decode 的真实投影形状（n, k，wt 形态）：hidden 2560 / inter 9216 /
/// 线性层 qkv 8192 / o 4096 / a-b 32 / lm_head 248320
const SHAPES: &[(&str, u32, u32)] = &[
    ("lm_head  248320×2560", 248320, 2560),
    ("gate/up   9216×2560", 9216, 2560),
    ("down      2560×9216", 2560, 9216),
    ("qkv       8192×2560", 8192, 2560),
    ("o/z       4096×2560", 4096, 2560),
    ("out       2560×4096", 2560, 4096),
    ("k/v       1024×2560", 1024, 2560),
    ("a/b         32×2560", 32, 2560),
];


/// **流水态 A/B**（决定性口径）：把 8 个形状的权重缓冲一次性铺好，然后
/// 循环「逐形状发一发」——一轮搬的权重量 ≈1.4GB ≫ 72MB L2，于是每个形状
/// 的权重在轮内都是**冷的**，正是真实 decode 步的形态（每步 8GB 权重各读
/// 一遍）。同一轮里交错 A（全序号 0）与 B（冷态 argmin）两组选型，时钟漂移
/// 对两组同等，差值即"冷态选型的稳态收益"。
///
/// 与 `qwen_step_profile` 的差别：那边调优开销落在计量窗内（首步调优被记进
/// gemm 段），这边调优与测量完全分离。
fn stream_mode(iters: usize) {
    const M: u32 = 100;
    let gpu = Gpu::probe().expect("CUDA");
    let stream = gpu.stream();
    let lt = BlasLt::open(&gpu).expect("libcublasLt");

    // 每形状一套真实尺寸缓冲（wt 形态：B 存 [n,k]）
    let mut bufs = Vec::new();
    for &(name, n, k) in SHAPES {
        let (a, w, c) = (
            gpu.alloc_bytes(M as usize * k as usize * 2),
            gpu.alloc_bytes(n as usize * k as usize * 2),
            gpu.alloc_bytes(M as usize * n as usize * 2),
        );
        w.zero_on(&stream);
        c.zero_on(&stream);
        a.zero_on(&stream);
        let set = lt.probe_candidates(&a, &w, &c, M, n, k, true, 8);
        assert!(!set.is_empty(), "{name}: 无候选");
        for vi in 0..set.len() {
            assert_eq!(lt.probe_launch(&set, vi, &a, &w, &c, &stream), 0, "{name} 候选 {vi} 失败");
        }
        bufs.push((name, n, k, a, w, c, set));
    }
    stream.sync();

    // 冷态逐候选实测（与主模式同一协议，rounds 少些）→ 每形状 cold-argmin
    let mut pick_cold = vec![0usize; bufs.len()];
    let mut cold_tab = Vec::new();
    for (si, (name, _n, _k, a, w, c, set)) in bufs.iter().enumerate() {
        let nc = set.len();
        let evs: Vec<Event> = (0..2 * nc).map(|_| gpu.event()).collect();
        let mut cold: Vec<Vec<f64>> = vec![Vec::new(); nc];
        for _ in 0..5 {
            for vi in 0..nc {
                lt.l2_flush(&stream);
                evs[2 * vi].record_on(&stream);
                lt.probe_launch(set, vi, a, w, c, &stream);
                evs[2 * vi + 1].record_on(&stream);
            }
            stream.sync();
            for vi in 0..nc {
                cold[vi].push(evs[2 * vi + 1].elapsed_ms(&evs[2 * vi]) as f64);
            }
        }
        let med: Vec<f64> = cold.iter_mut().map(|v| median(v)).collect();
        let ix = (0..nc).min_by(|&x, &y| med[x].total_cmp(&med[y])).unwrap();
        pick_cold[si] = ix;
        cold_tab.push((name.to_string(), ix, med[ix] * 1e3, med[0] * 1e3));
    }
    println!("== 流水态 A/B（{iters} 轮，每轮 8 形状各 1 发，轮内 A/B 交错）==");
    println!("A = 全序号 0（改前调优器在 8 形状里 6 个钉 0）；B = 冷态 argmin（改后口径）");
    println!("── 冷态单发基线（探针口径）──");
    for (name, ix, t_b, t_a) in &cold_tab {
        println!("  {name}: A#0 {:>7.1}µs → B#{ix} {:>7.1}µs（{:+.1}%）", t_a, t_b, (t_b / t_a - 1.0) * 100.0);
    }

    // 逐形状事件（两组各一对），每轮结束 sync 读数
    let evs: Vec<Event> = (0..4 * bufs.len()).map(|_| gpu.event()).collect();
    let mut ta: Vec<Vec<f64>> = vec![Vec::new(); bufs.len()];
    let mut tb: Vec<Vec<f64>> = vec![Vec::new(); bufs.len()];
    for _ in 0..iters {
        for mode in 0..2 {
            for (si, (_name, _n, _k, a, w, c, set)) in bufs.iter().enumerate() {
                let vi = if mode == 0 { 0 } else { pick_cold[si] };
                let base = 4 * si + 2 * mode;
                evs[base].record_on(&stream);
                lt.probe_launch(set, vi, a, w, c, &stream);
                evs[base + 1].record_on(&stream);
            }
        }
        stream.sync();
        for si in 0..bufs.len() {
            ta[si].push(evs[4 * si + 1].elapsed_ms(&evs[4 * si]) as f64);
            tb[si].push(evs[4 * si + 3].elapsed_ms(&evs[4 * si + 2]) as f64);
        }
    }
    println!("── 流水态（每轮 8 发，权重 1.4GB 流过 72MB L2 ⇒ 每发都是冷的）──");
    println!("  形状                    A(全#0) µs   B(冷选) µs     差%    A GB/s   B GB/s");
    let (mut sa, mut sb, mut wbytes) = (0.0f64, 0.0f64, 0.0f64);
    for (si, (name, n, k, _a, _w, _c, _set)) in bufs.iter().enumerate() {
        let a = median(&mut ta[si]);
        let b = median(&mut tb[si]);
        let wb = (*n as f64) * (*k as f64) * 2.0; // 权重流量（真实 decode 口径）
        wbytes += wb;
        sa += a;
        sb += b;
        println!(
            "  {name:<22} {:9.1} {:9.1} {:8.1} {:8.0} {:8.0}",
            a * 1e3,
            b * 1e3,
            (b / a - 1.0) * 100.0,
            wb / (a * 1e-3) / 1e9,
            wb / (b * 1e-3) / 1e9,
        );
    }
    println!(
        "  合计                     {:9.1} {:9.1} {:8.1} {:8.0} {:8.0}（权重 {:.2} GB/轮）",
        sa * 1e3,
        sb * 1e3,
        (sb / sa - 1.0) * 100.0,
        wbytes / (sa * 1e-3) / 1e9,
        wbytes / (sb * 1e-3) / 1e9,
        wbytes / 1e9,
    );
    // 34 层折算：真实模型每层 4 发（qkv=3 / o=4 / up=1 / down=2），lm_head 1 发
    let a_med: Vec<f64> = (0..bufs.len()).map(|i| median(&mut ta[i])).collect();
    let b_med: Vec<f64> = (0..bufs.len()).map(|i| median(&mut tb[i])).collect();
    let per_layer_a: f64 = [1usize, 2, 3, 4].iter().map(|&i| a_med[i]).sum();
    let per_layer_b: f64 = [1usize, 2, 3, 4].iter().map(|&i| b_med[i]).sum();
    let step_a = per_layer_a * 34.0 + a_med[0];
    let step_b = per_layer_b * 34.0 + b_med[0];
    let layer_bytes: f64 = [1usize, 2, 3, 4]
        .iter()
        .map(|&i| bufs[i].1 as f64 * bufs[i].2 as f64 * 2.0)
        .sum();
    let step_bytes = layer_bytes * 34.0 + bufs[0].1 as f64 * bufs[0].2 as f64 * 2.0;
    println!(
        "  34 层折算（(qkv+o+up+down)×34 + lm_head，权重 {:.2} GB/步）：A {:.2} ms/步 → B {:.2} ms/步（{:+.1}%）⇒ {:.0} → {:.0} GB/s",
        step_bytes / 1e9,
        step_a,
        step_b,
        (step_b / step_a - 1.0) * 100.0,
        step_bytes / (step_a * 1e-3) / 1e9,
        step_bytes / (step_b * 1e-3) / 1e9,
    );
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if argv.get(1).map(|s| s.as_str()) == Some("stream") {
        let iters: usize = argv.get(2).and_then(|s| s.parse().ok()).unwrap_or(40);
        return stream_mode(iters);
    }
    let m: u32 = argv.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let rounds: usize = argv.get(2).and_then(|s| s.parse().ok()).unwrap_or(7);

    let gpu = Gpu::probe().expect("CUDA");
    let stream = gpu.stream();
    let lt = BlasLt::open(&gpu).expect("libcublasLt");
    // 读扫件 = 调优器同一份（`BlasLt::l2_flush`，缓冲/读核首调惰性建）
    let sweep = |stream: &Stream| lt.l2_flush(stream);

    println!("== gemm 冷/热双口径选型探针（m={m}，{rounds} 轮中位，wt 形态）==");
    // 读扫自身带宽（纯读地板）：顺带标定，供 cold GB/s 折算对照
    {
        let ev0 = gpu.event();
        let ev1 = gpu.event();
        let mut ts = Vec::new();
        for _ in 0..5 {
            ts.push(timed(&gpu, &stream, &ev0, &ev1, || sweep(&stream)));
        }
        let t = median(&mut ts);
        println!(
            "读扫 128MB：{:.1} µs（{:.0} GB/s 纯读地板）\n",
            t * 1e3,
            FLUSH_BYTES as f64 / (t * 1e-3) / 1e9
        );
    }

    let mut summary: Vec<(String, usize, usize, usize, bool)> = Vec::new();
    let mut sum_warm = 0.0f64;
    let mut sum_cold = 0.0f64;
    for &(name, n, k) in SHAPES {
        let (a, w, c) = (
            gpu.alloc_bytes(m as usize * k as usize * 2),
            gpu.alloc_bytes(n as usize * k as usize * 2),
            gpu.alloc_bytes(m as usize * n as usize * 2),
        );
        a.zero_on(&stream);
        w.zero_on(&stream);
        c.zero_on(&stream);
        stream.sync();

        let set = lt.probe_candidates(&a, &w, &c, m, n, k, true, 8);
        let cands = set.candidates();
        // 请求 16 个时库还给更多吗（候选截断的另一半答案）
        let wide = lt.probe_candidates(&a, &w, &c, m, n, k, true, 16).len();
        let nc = cands.len();
        if nc == 0 {
            println!("{name}: 无有效候选（跳过）");
            continue;
        }

        // 预热：每候选 2 发（与生产同：真实缓冲上试跑）
        for vi in 0..nc {
            for _ in 0..2 {
                let st = lt.probe_launch(&set, vi, &a, &w, &c, &stream);
                assert_eq!(st, 0, "{name} 候选 {vi} launch 失败 status {st}");
            }
        }
        stream.sync();

        let mut warm: Vec<Vec<f64>> = vec![Vec::new(); nc];
        let mut cold: Vec<Vec<f64>> = vec![Vec::new(); nc];
        // 每候选四条事件（cold 对 + warm 对）：整轮入队、轮末一次 sync 后读数
        let evs: Vec<Event> = (0..4 * nc).map(|_| gpu.event()).collect();
        for _ in 0..rounds {
            for vi in 0..nc {
                // 冷口径：先读扫稀释 L2，第 1 发计 cold
                sweep(&stream);
                evs[4 * vi].record_on(&stream);
                lt.probe_launch(&set, vi, &a, &w, &c, &stream);
                evs[4 * vi + 1].record_on(&stream);
                // 回热：1 发弃（把本候选的权重重新灌回 L2）
                lt.probe_launch(&set, vi, &a, &w, &c, &stream);
                // 热口径：连续发（不冲洗）——现调优器口径
                evs[4 * vi + 2].record_on(&stream);
                lt.probe_launch(&set, vi, &a, &w, &c, &stream);
                evs[4 * vi + 3].record_on(&stream);
            }
            stream.sync();
            for vi in 0..nc {
                cold[vi].push(evs[4 * vi + 1].elapsed_ms(&evs[4 * vi]) as f64);
                warm[vi].push(evs[4 * vi + 3].elapsed_ms(&evs[4 * vi + 2]) as f64);
            }
        }
        let bytes = (m as usize * k as usize + n as usize * k as usize + m as usize * n as usize) as f64 * 2.0;
        let warm_med: Vec<f64> = warm.iter_mut().map(|v| median(v)).collect();
        let cold_med: Vec<f64> = cold.iter_mut().map(|v| median(v)).collect();
        let warm_ix = (0..nc).min_by(|&x, &y| warm_med[x].total_cmp(&warm_med[y])).unwrap();
        let cold_ix = (0..nc).min_by(|&x, &y| cold_med[x].total_cmp(&cold_med[y])).unwrap();

        println!("── {name}（m={m} n={n} k={k}；有效候选 {nc}，请求 16 给 {wide}；流量 {:.1} MB）──", bytes / 1e6);
        println!("  idx     ws   waves     warm µs    cold µs   cold GB/s   标");
        for vi in 0..nc {
            let mut tag = String::new();
            if vi == warm_ix {
                tag.push_str("W");
            }
            if vi == cold_ix {
                tag.push_str("C");
            }
            println!(
                "  {vi:3} {:7} {:7.2} {:10.1} {:10.1} {:10.1}   {tag}",
                cands[vi].ws,
                cands[vi].waves,
                warm_med[vi] * 1e3,
                cold_med[vi] * 1e3,
                bytes / (cold_med[vi] * 1e-3) / 1e9,
            );
        }
        let same = warm_ix == cold_ix;
        println!(
            "  → warm-argmin #{warm_ix} {:.1}µs / cold-argmin #{cold_ix} {:.1}µs ⇒ {}{}",
            warm_med[warm_ix] * 1e3,
            cold_med[cold_ix] * 1e3,
            if same { "一致" } else { "**不一致**" },
            if cold_ix >= 4 && cold_ix != warm_ix { "（冷赢家落在前 4 之外！）" } else { "" },
        );
        // 现调优器规则仿真（warm 前 4 候选 + 滞回 5%/1µs + ws0 偏好 3%）：
        // 它钉到谁、该人在**冷态**下多花多少——"改调优器值不值"的 µs 账
        let n4 = nc.min(4);
        let warm_ix4 = (0..n4).min_by(|&x, &y| warm_med[x].total_cmp(&warm_med[y])).unwrap();
        let first = warm_med[0];
        let mut cur = if warm_med[warm_ix4] <= first * 0.95 && first - warm_med[warm_ix4] >= 0.001 {
            warm_ix4
        } else {
            0
        };
        if cands[cur].ws > 0 {
            if let Some(w0) = (0..n4)
                .filter(|&i| cands[i].ws == 0)
                .min_by(|&x, &y| warm_med[x].total_cmp(&warm_med[y]))
            {
                if warm_med[w0] <= warm_med[warm_ix4] * 1.03 {
                    cur = w0;
                }
            }
        }
        println!(
            "  → 现调优器规则（warm 前 4+滞回+ws0）钉 #{cur}：冷态 {:>6.1}µs vs 冷最优 {:>6.1}µs（{:+.1}%）",
            cold_med[cur] * 1e3,
            cold_med[cold_ix] * 1e3,
            (cold_med[cur] / cold_med[cold_ix] - 1.0) * 100.0,
        );
        println!(
            "  → 冷态赢家折算 {:.1} GB/s；冷/热 = {:.2}×\n",
            bytes / (cold_med[cold_ix] * 1e-3) / 1e9,
            cold_med[cold_ix] / warm_med[warm_ix],
        );
        sum_warm += cold_med[cur];
        sum_cold += cold_med[cold_ix];
        summary.push((name.to_string(), warm_ix, cold_ix, nc, same));
    }

    println!("== 汇总（8 形状 argmin 之和，单步 100 行 decode 的 gemm 段折算）==");
    println!("  现调优器规则（warm+滞回+ws0）的冷态代价：{:.0} µs", sum_warm * 1e3);
    println!("  冷态口径（cold-argmin）：{:.0} µs（{:.1}% 之差）", sum_cold * 1e3, (sum_cold / sum_warm - 1.0) * 100.0);
    let bad: Vec<&(String, usize, usize, usize, bool)> = summary.iter().filter(|s| !s.4).collect();
    println!("  warm/cold argmin 一致：{} / {} 个形状", summary.len() - bad.len(), summary.len());
    for s in &bad {
        println!("  · {} warm #{} vs cold #{}（候选数 {}）", s.0, s.1, s.2, s.3);
    }
}
