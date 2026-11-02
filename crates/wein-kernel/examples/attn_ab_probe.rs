//! 单遍 attention 的 A/B 探针（几何 × 池 dtype）。
//!
//!   cargo run --release -p wein-kernel --example attn_ab_probe
//!
//! 逐档（`AttnCfg`）跑 decode 1k/3k/8k 与 prefill t=256 @768/@1536/@3072：
//! 报 ms、KV 读量（K+V 的**实际载入**，含 RPT 之前的 q 行重复）、有效 GB/s、
//! 相对 baseline 倍数、以及极差（与 baseline 单遍档的位差数——RPT/ND 只改
//! "哪个线程载 K"，每条分值仍由单线程按 d 升序算完，故**逐位相同**）。
//!
//! KV 读量口径：一块（kv 头, token 片, j 分片）里线程组数 = rows_nom/RPT，
//! 每组对每个 j 载入一次 K 整行 + 一次 V 的 ND 维 ⇒
//!   read = 2 · nblk · (avg_l/js) · (rows_nom/RPT) · hd · esz
//! **独占量**（每 KV 字节只读一次）= toks · avg_l · kv_n · hd · esz · 2，
//! 放大倍数 = read/独占 = avg_l · group / RPT——这就是"每 KV 字节服务多少 q 行"。
use wein_kernel::cuda::Gpu;
use wein_kernel::ops::OpTable;
use wein_kernel::ops::{AttnCfg, KvDtype};
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

struct Case {
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
    bt: Vec<u32>,
    si: Vec<u32>,
    pos: Vec<u32>,
    bpr: usize,
}

fn case(tokens: usize, pos0: usize, q_heads: usize, kv_heads: usize, hd: usize, bs: usize) -> Case {
    let he = kv_heads * hd;
    let pmax = pos0 + tokens - 1;
    let bpr = (pmax + 1).div_ceil(bs);
    let slots = bpr * bs;
    Case {
        q: bf16_vec(0x7a11, tokens * q_heads * hd, 2.0),
        k: bf16_vec(0x7b33, slots * he, 1.0),
        v: bf16_vec(0x7c55, slots * he, 1.0),
        bt: (0..bpr as u32).collect(),
        si: vec![0u32; tokens],
        pos: (0..tokens).map(|i| (pos0 + i) as u32).collect(),
        bpr,
    }
}

fn cfgs() -> Vec<(&'static str, AttnCfg)> {
    let mut out = Vec::new();
    let mut add = |name: &'static str, nd: usize, rpt: usize, kv: KvDtype| {
        out.push((name, AttnCfg { nd, rpt, ntok_max: 4, bdim_max: 1024, kv }));
    };
    add("baseline", 4, 1, KvDtype::Bf16);
    add("A1 n8", 8, 1, KvDtype::Bf16);
    add("A1 n16", 16, 1, KvDtype::Bf16);
    add("A r2", 4, 2, KvDtype::Bf16);
    add("A r4", 4, 4, KvDtype::Bf16);
    add("A n8r4", 8, 4, KvDtype::Bf16);
    add("B f32", 4, 1, KvDtype::F32);
    add("B f32n8", 8, 1, KvDtype::F32);
    add("AB f32r2", 4, 2, KvDtype::F32);
    add("AB f32r4", 4, 4, KvDtype::F32);
    // nt=8 档（rows_nom = 8×group = 32 = ROWS_CAP）：块数再减半、屏障更少
    let mk = |name: &'static str, nd: usize, rpt: usize| {
        (name, AttnCfg { nd, rpt, ntok_max: 8, bdim_max: 512, kv: KvDtype::Bf16 })
    };
    out.push(mk("A1 n16 nt8", 16, 1));
    out.push(mk("A n8r4 nt8", 8, 4));
    out
}

fn main() {
    let gpu = Gpu::probe().expect("须有 CUDA 设备");
    let profile = HardwareProfile::with_device(Device::Cuda).with_arch("sm_89");
    let t = OpTable::resolve(&gpu, &profile).expect("sm_89 画像须解析成功");
    let stream = gpu.stream();
    let (bs, q_heads, kv_heads, hd) = (16usize, 16usize, 4usize, 256usize);
    let q_scale = 1.0 / (hd as f32).sqrt();
    let group = q_heads / kv_heads;

    println!("形状：q_heads={q_heads} kv_heads={kv_heads} hd={hd} block={bs}");
    println!("\n| 档 | nd | rpt | regs | bdim | ntok | ntile | jsplt |");
    println!("|---|---|---|---|---|---|---|---|");
    for (name, cfg) in cfgs() {
        let g = wein_kernel::ops::PagedAttnCuda::online_geom(cfg, q_heads as u32, kv_heads as u32, hd as u32, 256);
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            name,
            cfg.nd,
            cfg.rpt,
            wein_kernel::ops::attn_regs_of(cfg),
            g.bdim,
            g.ntok,
            g.ntile,
            g.jsplit
        );
    }

    // (标签, tokens, 首位置)
    let mut points: Vec<(String, usize, usize)> = Vec::new();
    for pos in [1024usize, 3072, 8192] {
        points.push((format!("dec{}", pos / 1024), 1, pos - 1));
    }
    for (tag, end) in [("pf@768", 768usize), ("pf@1536", 1536), ("pf@3072", 3072), ("pf@6010", 6010)] {
        points.push((format!("{tag}"), 256, end - 256));
    }

    println!(
        "\n| 工作点 | 档 | ms | ms/基线 | KV读 MB | 独占 MB | 放大 | GB/s | 位差 |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    let mut tally: Vec<(String, Vec<f64>)> = Vec::new();
    for (pi, (tag, tokens, pos0)) in points.iter().enumerate() {
        let tokens = *tokens;
        let pos0 = *pos0;
        let c = case(tokens, pos0, q_heads, kv_heads, hd, bs);
        let dq = gpu.alloc_from_slice(&c.q);
        let dbt = gpu.alloc_from_slice(&c.bt);
        let dsi = gpu.alloc_from_slice(&c.si);
        let dpo = gpu.alloc_from_slice(&c.pos);
        let dout = gpu.alloc_bytes(tokens * q_heads * hd * 2);
        // bf16 池 + f32 池（f32 由同一 bf16 位型精确加宽 ⇒ 两种池同值）
        let dk = gpu.alloc_from_slice(&c.k);
        let dv = gpu.alloc_from_slice(&c.v);
        let kf: Vec<f32> =
            c.k.iter().map(|&b| wein_kernel::reference::bf16_to_f32(b)).collect();
        let vf: Vec<f32> =
            c.v.iter().map(|&b| wein_kernel::reference::bf16_to_f32(b)).collect();
        let dkf = gpu.alloc_from_slice(&kf);
        let dvf = gpu.alloc_from_slice(&vf);

        let avg_l = (0..tokens).map(|i| (pos0 + i + 1) as f64).sum::<f64>() / tokens as f64;
        let mut base_ms = 1.0f64;
        let mut refout: Vec<u16> = Vec::new();
        for (idx, (name, cfg)) in cfgs().into_iter().enumerate() {
            let (kk, vv): (&_, &_) = if cfg.kv == KvDtype::F32 {
                (&dkf, &dvf)
            } else {
                (&dk, &dv)
            };
            let esz = if cfg.kv == KvDtype::F32 { 4.0 } else { 2.0 };
            let g = wein_kernel::ops::PagedAttnCuda::online_geom(
                cfg, q_heads as u32, kv_heads as u32, hd as u32, tokens as u32,
            );
            let run = || {
                t.paged_attn.run_tile_online_cfg(
                    cfg, &dq, kk, vv, &dbt, &dsi, &dpo, &dout, q_scale, c.bpr as u32,
                    bs as u32, q_heads as u32, kv_heads as u32, hd as u32, tokens as u32, &stream,
                )
            };
            for _ in 0..3 {
                run();
            }
            stream.sync();
            let (e0, e1) = (gpu.event(), gpu.event());
            e0.record_on(&stream);
            let n = if tokens > 1 { 20 } else { 200 };
            for _ in 0..n {
                run();
            }
            e1.record_on(&stream);
            stream.sync();
            let ms = (e1.elapsed_ms(&e0) / n as f32) as f64;
            // 读出（第一个档就是参考）
            run();
            stream.sync();
            let o: Vec<u16> = dout.read_to_vec();
            if idx == 0 {
                refout = o.clone();
                base_ms = ms;
            }
            let ndiff = o.iter().zip(refout.iter()).filter(|(a, b)| a != b).count();
            let nblk = kv_heads as f64 * g.ntile as f64 * g.jsplit as f64;
            let jspan = avg_l / g.jsplit as f64;
            let read = 2.0 * nblk * jspan * (g.rows_nom / g.rpt) as f64 * hd as f64 * esz;
            // 独占量 = 每 (token, K/V) 的 hd×kv_n 只读一次（不含 pos 维）
            let uniq = tokens as f64 * kv_heads as f64 * hd as f64 * esz * 2.0;
            let gbs = read / (ms / 1e3) / 1e9;
            println!(
                "| {} | {} | {:.4} | {:.2}× | {:.1} | {:.2} | {:.0}× | {:.0} | {} |",
                tag,
                name,
                ms,
                ms / base_ms,
                read / 1e6,
                uniq / 1e6,
                read / uniq,
                gbs,
                ndiff
            );
            let pos_t = match tally.iter().position(|(n, _)| n == name) {
                Some(i) => i,
                None => {
                    tally.push((name.to_string(), vec![0.0; points.len()]));
                    tally.len() - 1
                }
            };
            tally[pos_t].1[pi] = ms / base_ms;
        }
        // 两遍档 + f32 池：必须与两遍 bf16 位级相同（o30 的 f32 版）
        t.paged_attn.run_tile(
            &dq, &dk, &dv, &dbt, &dsi, &dpo, &dout, q_scale, c.bpr as u32, bs as u32,
            q_heads as u32, kv_heads as u32, hd as u32, tokens as u32, &stream,
        );
        stream.sync();
        let a: Vec<u16> = dout.read_to_vec();
        t.paged_attn.run_tile_f32(
            &dq, &dkf, &dvf, &dbt, &dsi, &dpo, &dout, q_scale, c.bpr as u32, bs as u32,
            q_heads as u32, kv_heads as u32, hd as u32, tokens as u32, &stream,
        );
        stream.sync();
        let b: Vec<u16> = dout.read_to_vec();
        let d2 = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
        eprintln!("[两遍档 bf16 vs f32 位差 {d2} @ {tag}]");
        assert_eq!(d2, 0, "两遍档 f32 池须与 bf16 池位级相同（{tag}）");
        drop((dq, dk, dv, dkf, dvf, dbt, dsi, dpo, dout));
    }
    println!("\n| 档 | dec1k | dec3k | dec8k | pf@768 | pf@1536 | pf@3072 | pf@6010 |");
    println!("|---|---|---|---|---|---|---|---|");
    for (name, v) in tally {
        println!(
            "| {} | {:.2}× | {:.2}× | {:.2}× | {:.2}× | {:.2}× | {:.2}× | {:.2}× |",
            name, v[0], v[1], v[2], v[3], v[4], v[5], v[6]
        );
    }
    let _ = group;
}
