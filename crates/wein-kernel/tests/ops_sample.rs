//! I2 设备端采样器验收（top-k / top-p）——`sample_rows` 与 reference 镜像对拍
//! + 统计合同（卡方/边界/复现）+ 单发延迟表。
//!
//! 数值纪律同 I1：逐位对拍走 reference::sample_rows（独立代码路径，同序同式）；
//! 统计类用例按分布断言（卡方），边界类用例按语义断言（temp=0 位同 argmax、
//! top_k/top_p 截断集合、temp→0 退化、等值 logits 均匀）。
//!
//! - k1 · temp=0 → 与 argmax_rows 逐行位同（多条形状，含 n>256 多 lane）
//! - k2 · 同 seed 同 logits → 逐位相同序列（10 次调用，中间不重置状态）
//! - k3 · 不同 seed → 序列不同（统计：200 次抽样至少差 ≥5 项）
//! - k4 · top_k=2 → 只出前二
//! - k5 · top_p=0.6 → 只出前二（0.5/0.3/0.15/0.05 分布）
//! - k6 · temp=1e-6 → 退化为 argmax
//! - k7 · 全等 logits → 均匀（16 词表卡方）
//! - k8 · 卡方：已知分布 200k 抽样，α=1e-4 临界值内
//! - k9 · 单发延迟 canary（t=1/16/100 @ n=248320）
//! - k10 · kernel ↔ reference 逐位对拍（随机 logits/参数/种子）
#![allow(dead_code)]
use std::sync::{Mutex, MutexGuard, OnceLock};

use wein_kernel::cuda::{DevBuf, Gpu, Stream};
use wein_kernel::ops::{OpTable, SampleRowsOp};
use wein_kernel::reference::{self, SampleParam};
use wein_platform::{Device, HardwareProfile};

fn gpu() -> Option<Gpu> {
    match Gpu::probe() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("SKIP（无可用 CUDA 设备：{e}）");
            None
        }
    }
}

/// GPU 串行锁：cargo test 的并行线程会互相闯入事件计时窗（o22/o23 同款）
fn guard() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

fn profile() -> HardwareProfile {
    HardwareProfile::with_device(Device::Cuda).with_arch("sm_89")
}

fn table(gpu: &Gpu) -> OpTable<'_> {
    OpTable::resolve(gpu, &profile()).expect("sm_89 画像须解析成功")
}

/// 采样器测试夹具：设备缓冲常驻（RNG 状态跨调用前进——与 exec 的图档同语义）
struct Smp<'a> {
    gpu: &'a Gpu,
    x: DevBuf<'a>,
    rows: DevBuf<'a>,
    slots: DevBuf<'a>,
    temp: DevBuf<'a>,
    topk: DevBuf<'a>,
    topp: DevBuf<'a>,
    seed: DevBuf<'a>,
    rng: DevBuf<'a>,
    out: DevBuf<'a>,
    cap: usize,
}

impl<'a> Smp<'a> {
    fn new(gpu: &'a Gpu, _cap: usize) -> Self {
        // 占位缓冲（cuMemAlloc(0) 非法——layout 之前先给个 8B 占位）
        Self {
            gpu,
            x: gpu.alloc_bytes(8),
            rows: gpu.alloc_bytes(8),
            slots: gpu.alloc_bytes(8),
            temp: gpu.alloc_bytes(8),
            topk: gpu.alloc_bytes(8),
            topp: gpu.alloc_bytes(8),
            seed: gpu.alloc_bytes(8),
            rng: gpu.alloc_bytes(8),
            out: gpu.alloc_bytes(8),
            cap: 0,
        }
    }

    /// 按 (行数, 词表, RNG 槽数) 重配缓冲：logits 表 + 逐行参数 + 状态
    fn layout(&mut self, s: usize, n: usize, slots: usize) {
        let g = self.gpu;
        self.x = g.alloc_bytes(s * n * 2);
        self.rows = g.alloc_bytes(s * 4);
        self.slots = g.alloc_bytes(s * 4);
        self.temp = g.alloc_bytes(s * 4);
        self.topk = g.alloc_bytes(s * 4);
        self.topp = g.alloc_bytes(s * 4);
        self.seed = g.alloc_bytes(s * 8);
        self.rng = g.alloc_bytes(slots * 8);
        self.out = g.alloc_bytes(s * 4);
        self.cap = s;
        let stream = g.stream();
        self.rng.zero_on(&stream); // 0 = 未初始化 → kernel 由 seed 派生
        stream.sync();
    }

    /// 复位 RNG 状态（"同一 seed 再来一次"的入口——图档里等价于 set_sampling）
    fn reset_rng(&self, stream: &Stream) {
        self.rng.zero_on(stream);
    }

    /// 只发参数 + launch（logits 已在设备上——延迟 canary 用这条路径，
    /// 免得把 50MB H2D 算进 kernel 账）
    fn dispatch(&self, t: &OpTable, stream: &Stream, n: usize, p: &[SampleParam]) {
        let s = p.len();
        let temps: Vec<f32> = p.iter().map(|q| q.temp).collect();
        let ks: Vec<u32> = p.iter().map(|q| q.top_k).collect();
        let ps: Vec<f32> = p.iter().map(|q| q.top_p).collect();
        let seeds: Vec<u64> = p.iter().map(|q| q.seed).collect();
        let slots: Vec<u32> = (0..s as u32).collect();
        self.slots.upload_prefix_on(&slots, stream);
        self.temp.upload_prefix_on(&temps, stream);
        self.topk.upload_prefix_on(&ks, stream);
        self.topp.upload_prefix_on(&ps, stream);
        self.seed.upload_prefix_on(&seeds, stream);
        t.sample_rows.run(
            &self.x, &self.rows, &self.slots, &self.temp, &self.topk, &self.topp, &self.seed,
            &self.rng, &self.out, n as u32, s as u32, stream,
        );
    }

    fn run(&self, t: &OpTable, stream: &Stream, logits: &[u16], n: usize, p: &[SampleParam]) -> Vec<u32> {
        let s = p.len();
        assert!(s <= self.cap, "行数 {s} 超夹具容量 {}", self.cap);
        let temps: Vec<f32> = p.iter().map(|q| q.temp).collect();
        let ks: Vec<u32> = p.iter().map(|q| q.top_k).collect();
        let ps: Vec<f32> = p.iter().map(|q| q.top_p).collect();
        let seeds: Vec<u64> = p.iter().map(|q| q.seed).collect();
        let rows: Vec<u32> = (0..s as u32).collect();
        let slots: Vec<u32> = (0..s as u32).collect();
        self.x.upload_prefix_on(&logits[..s * n], stream);
        self.rows.upload_prefix_on(&rows, stream);
        self.slots.upload_prefix_on(&slots, stream);
        self.temp.upload_prefix_on(&temps, stream);
        self.topk.upload_prefix_on(&ks, stream);
        self.topp.upload_prefix_on(&ps, stream);
        self.seed.upload_prefix_on(&seeds, stream);
        t.sample_rows.run(
            &self.x, &self.rows, &self.slots, &self.temp, &self.topk, &self.topp, &self.seed,
            &self.rng, &self.out, n as u32, s as u32, stream,
        );
        stream.sync();
        let out: Vec<u32> = self.out.read_to_vec();
        out[..s].to_vec()
    }
}

fn bf(v: f32) -> u16 {
    reference::f32_to_bf16(v)
}

/// 4 候选的已知分布（temp=1 时 softmax 还原 p=0.5/0.3/0.15/0.05）
fn dist4() -> Vec<u16> {
    vec![bf(0.5f32.ln()), bf(0.3f32.ln()), bf(0.15f32.ln()), bf(0.05f32.ln())]
}

fn param(temp: f32, top_k: u32, top_p: f32, seed: u64) -> SampleParam {
    SampleParam { temp, top_k, top_p, seed }
}

/// k1 · temp=0 → 与 argmax_rows 逐行位同（含等值/极端值，多形状）
#[test]
fn k1_greedy_equals_argmax_bitwise() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let mut smp = Smp::new(&gpu, 0);
    for (s, n) in [(1usize, 7usize), (5, 300), (3, 1024)] {
        // 随机 + 人为等值（全序取小下标合同）
        let mut x: Vec<u16> = (0..s * n)
            .map(|i| reference::f32_to_bf16(reference::splitmix_f32(i as u64 ^ 0x51) * 4.0))
            .collect();
        for r in 0..s {
            for j in 0..4 {
                x[r * n + j] = bf(9.0); // 等值极大：argmax 必取 r*n+0
            }
        }
        smp.layout(s, n, s);
        let p = vec![param(0.0, 0, 1.0, 1); s];
        let got = smp.run(&t, &stream, &x, n, &p);
        let want = reference::argmax_rows(&x, &(0..s as u32).collect::<Vec<_>>(), n);
        assert_eq!(got, want, "temp=0 须与 argmax_rows 逐行位同（s={s} n={n}）");
    }
}

/// k2 · 同 seed 同 logits → 逐位相同序列（10 次调用；每次复位状态 = 同一
/// 请求重跑）
#[test]
fn k2_same_seed_reproducible() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let (s, n) = (4usize, 4096usize);
    let x: Vec<u16> = (0..s * n)
        .map(|i| reference::f32_to_bf16(reference::splitmix_f32(i as u64 ^ 0xa1) * 3.0))
        .collect();
    let mut smp = Smp::new(&gpu, 0);
    smp.layout(s, n, s);
    let p: Vec<SampleParam> = (0..s).map(|j| param(0.8, 16, 0.9, 0x5eed_0000 + j as u64)).collect();
    let mut first: Option<Vec<u32>> = None;
    for rep in 0..10 {
        smp.reset_rng(&stream);
        stream.sync();
        let got = smp.run(&t, &stream, &x, n, &p);
        match &first {
            None => first = Some(got),
            Some(f) => assert_eq!(f, &got, "第 {rep} 次调用序列与首次不同（同 seed 须逐位同）"),
        }
    }
}

/// k3 · 不同 seed → 序列不同（统计；200 次抽样至少差 5 项）
#[test]
fn k3_different_seed_diverges() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let (s, n) = (1usize, 64usize);
    let x: Vec<u16> = (0..n)
        .map(|i| reference::f32_to_bf16(reference::splitmix_f32(i as u64 ^ 0xb2) * 2.0))
        .collect();
    let mut smp = Smp::new(&gpu, 0);
    smp.layout(s, n, 2);
    // 两个 seed 各起一条独立流（先复位状态 = 同一请求重跑，只换 seed）
    let mut a = Vec::new();
    smp.reset_rng(&stream);
    stream.sync();
    for _ in 0..200 {
        let p = vec![param(1.0, 64, 1.0, 11)];
        a.push(smp.run(&t, &stream, &x, n, &p)[0]);
    }
    let mut b = Vec::new();
    smp.reset_rng(&stream);
    stream.sync();
    for _ in 0..200 {
        let p = vec![param(1.0, 64, 1.0, 12)];
        b.push(smp.run(&t, &stream, &x, n, &p)[0]);
    }
    let diff = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    eprintln!("[k3] 不同 seed 200 次抽样差异 {diff} 项");
    assert!(diff >= 5, "不同 seed 的序列几乎相同（只差 {diff} 项）——种子未生效");
}

/// k4 · top_k=2 → 只出前二
#[test]
fn k4_top_k_2_only_top_two() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let (s, n) = (8usize, 4usize);
    let mut x = Vec::new();
    for _ in 0..s {
        x.extend_from_slice(&dist4());
    }
    let mut smp = Smp::new(&gpu, 0);
    smp.layout(s, n, s);
    let p = vec![param(1.0, 2, 1.0, 77); s];
    for _ in 0..30 {
        let got = smp.run(&t, &stream, &x, n, &p);
        for (j, &id) in got.iter().enumerate() {
            assert!(id <= 1, "top_k=2 采出了第 {id} 号候选（行 {j}）");
        }
    }
}

/// k5 · top_p=0.6 → 只出前二（0.5+0.3=0.8 ≥ 0.6，第三个 0.15 不进来）
#[test]
fn k5_top_p_06_only_top_two() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let (s, n) = (8usize, 4usize);
    let mut x = Vec::new();
    for _ in 0..s {
        x.extend_from_slice(&dist4());
    }
    let mut smp = Smp::new(&gpu, 0);
    smp.layout(s, n, s);
    let p = vec![param(1.0, 64, 0.6, 99); s];
    for _ in 0..30 {
        let got = smp.run(&t, &stream, &x, n, &p);
        for (j, &id) in got.iter().enumerate() {
            assert!(id <= 1, "top_p=0.6 采出了第 {id} 号候选（行 {j}）");
        }
    }
}

/// k6 · 极端：temp=1e-6 → 退化为 argmax（除后差距被放大到 inf 域之前，
/// 最大项独占 >0.999999 概率）
#[test]
fn k6_tiny_temp_degenerates_to_argmax() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let (s, n) = (16usize, 64usize);
    let x: Vec<u16> = (0..s * n)
        .map(|i| reference::f32_to_bf16(reference::splitmix_f32(i as u64 ^ 0xc3) * 3.0))
        .collect();
    let mut smp = Smp::new(&gpu, 0);
    smp.layout(s, n, s);
    let p = vec![param(1e-6, 64, 1.0, 3); s];
    let got = smp.run(&t, &stream, &x, n, &p);
    let want = reference::argmax_rows(&x, &(0..s as u32).collect::<Vec<_>>(), n);
    assert_eq!(got, want, "temp=1e-6 须退化为 argmax");
}

/// k7 · 全等 logits → 均匀（16 词表，200k 抽样的卡方应远低于临界值）
#[test]
fn k7_equal_logits_uniform() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let (s, n) = (512usize, 16usize);
    let x: Vec<u16> = vec![bf(0.25); s * n];
    let mut smp = Smp::new(&gpu, 0);
    smp.layout(s, n, s);
    let p: Vec<SampleParam> = (0..s).map(|j| param(1.0, 64, 1.0, 0x1000 + j as u64)).collect();
    let mut cnt = vec![0u64; n];
    let reps = 392; // 200,704 次
    for _ in 0..reps {
        let got = smp.run(&t, &stream, &x, n, &p);
        for &id in &got {
            cnt[id as usize] += 1;
        }
    }
    let tot: u64 = cnt.iter().sum();
    let expect = tot as f64 / n as f64;
    let chi: f64 = cnt.iter().map(|&c| {
        let d = c as f64 - expect;
        d * d / expect
    }).sum();
    // χ²(15, 0.9999) ≈ 45.3
    eprintln!("[k7] 均匀 logits: 抽样 {tot}，卡方 {chi:.3}（dof=15，α=1e-4 临界 45.3）");
    assert!(chi < 45.3, "均匀分布卡方超标：{chi:.3}");
}

/// k8 · 卡方：已知分布 0.5/0.3/0.15/0.05，200k 抽样，α=1e-4（dof=3，
/// 临界值 χ²_{0.9999}(3) = 21.108）
#[test]
fn k8_chi_square_known_distribution() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let (s, n) = (512usize, 4usize);
    let mut x = Vec::new();
    for _ in 0..s {
        x.extend_from_slice(&dist4());
    }
    let mut smp = Smp::new(&gpu, 0);
    smp.layout(s, n, s);
    let p: Vec<SampleParam> = (0..s).map(|j| param(1.0, 64, 1.0, 0xabc000 + j as u64)).collect();
    let mut cnt = vec![0u64; n];
    for _ in 0..391 {
        let got = smp.run(&t, &stream, &x, n, &p);
        for &id in &got {
            cnt[id as usize] += 1;
        }
    }
    let tot: u64 = cnt.iter().sum();
    let want = [0.5f64, 0.3, 0.15, 0.05];
    let chi: f64 = (0..n)
        .map(|i| {
            let e = tot as f64 * want[i];
            let d = cnt[i] as f64 - e;
            d * d / e
        })
        .sum();
    eprintln!(
        "[k8] 分布抽样 {tot}：计数 {cnt:?}（期望 {:?}），卡方 {chi:.3}（临界 21.108）",
        want.iter().map(|p| (tot as f64 * p) as u64).collect::<Vec<_>>()
    );
    assert!(chi < 21.108, "已知分布卡方超标：{chi:.3}（α=1e-4 临界 21.108）");
}

/// k10 · kernel ↔ reference 逐位对拍（随机 logits/参数/种子；n>256 才有多
/// lane 梯子——但不构造"单 lane 独占 >8 个 top-k"的病态输入）
#[test]
fn k10_kernel_matches_reference_bitwise() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let mut smp = Smp::new(&gpu, 0);
    for (s, n, tk, tp, temp) in [
        (6usize, 33usize, 4u32, 0.9f32, 1.0f32),
        (5, 257, 64, 1.0, 0.7),
        (4, 1000, 8, 0.5, 1.3),
        (3, 513, 1, 1.0, 1.0),
    ] {
        let x: Vec<u16> = (0..s * n)
            .map(|i| reference::f32_to_bf16(reference::splitmix_f32(i as u64 ^ 0xd4) * 5.0))
            .collect();
        smp.layout(s, n, s);
        let p: Vec<SampleParam> =
            (0..s).map(|j| param(temp, tk, tp, 0x7000 + j as u64 * 977)).collect();
        let got = smp.run(&t, &stream, &x, n, &p);
        // 参考镜像：同一份 RNG 初始状态（0 = 由 seed 派生）
        let mut rng = vec![0u64; s];
        let slots: Vec<u32> = (0..s as u32).collect();
        let want = reference::sample_rows(&x, &(0..s as u32).collect::<Vec<_>>(), n, &p, &slots, &mut rng);
        assert_eq!(got, want, "kernel/reference 逐位不一致（s={s} n={n} top_k={tk} top_p={tp} temp={temp}）");
    }
}

/// 诊断：把 sample_rows 的耗时拆成 greedy 分支（纯 argmax 扫描）与完整采样
/// 两条路径（临时探针，配合 k9 的延迟账）
#[test]
fn k9b_latency_breakdown() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    eprintln!("[k9b] device = {}", gpu.name());
    let stream = gpu.stream();
    let t = table(&gpu);
    let n = 248320usize;
    let mut smp = Smp::new(&gpu, 0);
    let big: Vec<u16> = (0..100 * n)
        .map(|i| reference::f32_to_bf16(reference::splitmix_f32(i as u64 ^ 0xe5) * 6.0))
        .collect();
    let e0 = gpu.event();
    let e1 = gpu.event();
    for (label, temp, tk) in [("greedy(temp=0)", 0.0f32, 0u32), ("sample(top_k=64)", 1.0, 64), ("sample(top_k=1)", 1.0, 1)] {
        for s in [1usize, 16, 100] {
            smp.layout(s, n, s);
            let p: Vec<SampleParam> = (0..s).map(|j| param(temp, tk, 0.95, 0x9000 + j as u64)).collect();
            smp.x.upload_prefix_on(&big[..s * n], &stream);
            let rows: Vec<u32> = (0..s as u32).collect();
            smp.rows.upload_prefix_on(&rows, &stream);
            smp.dispatch(&t, &stream, n, &p);
            stream.sync();
            let iters = 20;
            e0.record_on(&stream);
            for _ in 0..iters {
                // 只发 launch（参数缓冲内容不变——估纯 kernel 时间）
                t.sample_rows.run(&smp.x, &smp.rows, &smp.slots, &smp.temp, &smp.topk, &smp.topp,
                    &smp.seed, &smp.rng, &smp.out, n as u32, s as u32, &stream);
            }
            e1.record_on(&stream);
            stream.sync();
            let ms = e1.elapsed_ms(&e0) / iters as f32;
            eprintln!("[k9b] {label:18} t={s:3}：{ms:.3} ms");
        }
    }
}

/// k9 · 单发延迟 canary（n=248320 词表；t=1/16/100）+ **相对 greedy 的额外耗时**。
///
/// 单块一行 ⇒ 扫描是延迟受限的，绝对耗时的基线是同一 kernel 的 greedy 分支
/// （= exec 里本来就有的 argmax 尾，k9b 实测同量级），所以验收看的是
/// **采样相对 greedy 的增量**（要求 < 0.2ms），绝对值另设回归闸门。
#[test]
fn k9_latency_canary() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let n = 248320usize;
    let mut smp = Smp::new(&gpu, 0);
    let big: Vec<u16> = (0..100 * n)
        .map(|i| reference::f32_to_bf16(reference::splitmix_f32(i as u64 ^ 0xe5) * 6.0))
        .collect();
    let e0 = gpu.event();
    let e1 = gpu.event();
    let iters = 20;
    let mut ms_of = |s: usize, temp: f32, tk: u32| -> f32 {
        smp.layout(s, n, s);
        let p: Vec<SampleParam> = (0..s).map(|j| param(temp, tk, 0.95, 0x9000 + j as u64)).collect();
        smp.x.upload_prefix_on(&big[..s * n], &stream);
        let rows: Vec<u32> = (0..s as u32).collect();
        smp.rows.upload_prefix_on(&rows, &stream);
        smp.dispatch(&t, &stream, n, &p); // 预热（含 PTX 载入 / 时钟拉起）
        stream.sync();
        for _ in 0..200 {
            t.sample_rows.run(
                &smp.x, &smp.rows, &smp.slots, &smp.temp, &smp.topk, &smp.topp, &smp.seed,
                &smp.rng, &smp.out, n as u32, s as u32, &stream,
            );
        }
        e0.record_on(&stream);
        for _ in 0..iters {
            smp.dispatch(&t, &stream, n, &p);
        }
        e1.record_on(&stream);
        stream.sync();
        e1.elapsed_ms(&e0) / iters as f32
    };
    let mut extra_max = 0f32;
    for s in [1usize, 16, 100] {
        let ms = ms_of(s, 1.0, 64);
        let base = ms_of(s, 0.0, 0);
        extra_max = extra_max.max(ms - base);
        eprintln!(
            "[k9] sample_rows n=248320 t={s:3}：采样 {ms:.3} ms/发 · greedy 基线 {base:.3} ms/发 · 增量 {:.3} ms",
            ms - base
        );
        if s == 100 {
            std::fs::write(
                std::env::temp_dir().join("wein_sample_latency.txt"),
                format!("t=1/16/100 @ n=248320；t=100 采样 {ms:.3} ms、增量 {:.3} ms\n", ms - base),
            )
            .ok();
        }
    }
    assert!(
        extra_max < 0.2,
        "采样相对 greedy 的增量 {extra_max:.3} ms 超 0.2ms 预算"
    );
}
