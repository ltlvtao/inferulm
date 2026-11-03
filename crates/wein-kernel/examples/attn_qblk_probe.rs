//! qblk 档（K/V 行片驻寄存器供全 q 行）探针。
//!
//!   cargo run --release -p wein-kernel --example attn_qblk_probe [-- 6009]
//!
//! ① 正确性 spot：t=512（pos=0 起）对拍单遍档（`run_tile_online`）——
//!    ndiff/rel2/峰值（容差档，判据同 o33：rel2 ≤3e-3）。
//! ② 单发计时 @T=3009/6009/8009：变体 × bdim × jchunk 全档表，
//!    报 ms、regs/local（cuFuncGetAttribute）、Br/blocks、8 层折算，
//!    以及 KV 流量模型与 FLOP 地板（对账用）。
//! GPU 一次一进程：本探针独占设备（纪律）。
use wein_kernel::cuda::Gpu;
use wein_kernel::ops::OpTable;
use wein_kernel::ops::{AttnCfg, KvDtype, PagedAttnCuda, QblkCfg, QblkVar};

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

/// 单请求 prefill：tokens 个、位置 0..tokens-1（从头预填）
fn case(tokens: usize, q_heads: usize, kv_heads: usize, hd: usize, bs: usize) -> Case {
    let he = kv_heads * hd;
    let bpr = tokens.div_ceil(bs);
    let slots = bpr * bs;
    Case {
        q: bf16_vec(0x7a11, tokens * q_heads * hd, 2.0),
        k: bf16_vec(0x7b33, slots * he, 1.0),
        v: bf16_vec(0x7c55, slots * he, 1.0),
        bt: (0..bpr as u32).collect(),
        si: vec![0u32; tokens],
        pos: (0..tokens as u32).collect(),
        bpr,
    }
}

/// (bdim, jchunk) 扫描档
const GEOMS: &[(usize, usize)] = &[
    (128, 16),
    (128, 32),
    (256, 16),
    (256, 32),
    (256, 64),
    (384, 16),
    (384, 32),
    (384, 64),
    (512, 16),
    (512, 32),
    (512, 64),
];

/// 该变体在该 bdim 下能否发射（regs×bdim ≤ 65536/SM；探针预筛防 panic）
fn launchable(t: &OpTable, v: QblkVar, bdim: usize) -> bool {
    let (regs, _) = t.paged_attn.qblk_meta(v);
    regs as usize * bdim <= 65536 && bdim % 32 == 0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let only_t: Option<usize> = args.get(1).and_then(|s| s.parse().ok());
    let only_v: Option<String> = args.get(2).cloned();
    let gpu = Gpu::probe().expect("须有 CUDA 设备");
    let profile = wein_platform::HardwareProfile::with_device(wein_platform::Device::Cuda)
        .with_arch("sm_89");
    let t = OpTable::resolve(&gpu, &profile).expect("sm_89 画像须解析成功");
    let stream = gpu.stream();
    let (bs, q_heads, kv_heads, hd) = (16usize, 16usize, 4usize, 256usize);
    let group = q_heads / kv_heads;
    let q_scale = 1.0 / (hd as f32).sqrt();

    // ── 变体属性表（regs/local 一次打出）
    println!("## 变体属性（cuFuncGetAttribute）");
    println!("| 变体 | nd | rpt | jtr | regs | local B |");
    println!("|---|---|---|---|---|---|");
    for v in QblkVar::ALL {
        let (regs, local) = t.paged_attn.qblk_meta(v);
        let bcap = 65536 / regs as usize / 32 * 32;
        println!(
            "| {} | {} | {} | {} | {} | {} | bdim≤{bcap} |",
            v.name().trim_start_matches("paged_attn_qblk_"),
            v.nd(),
            v.rpt(),
            v.jtr(),
            regs,
            local
        );
    }
    if true {
        use wein_kernel::cuda::driver::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES;
        let _ = CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES;
        let (oregs, olocal) = t.paged_attn.online_meta();
        println!("| (对照 online 基线档) | | | | {oregs} | {olocal} | |");
    }

    // ── ① 正确性 spot：t=512 对拍单遍档
    println!("\n## 正确性 spot（t=512，对拍 run_tile_online；门 rel2≤3e-3/峰值≤1.5e-2）");
    println!("| 变体 | bdim | cap | Br | rows | 位差 | rel2 | 峰值 |");
    println!("|---|---|---|---|---|---|---|---|");
    let tokens = only_t.unwrap_or(512).min(512);
    let c = case(tokens, q_heads, kv_heads, hd, bs);
    let dq = gpu.alloc_from_slice(&c.q);
    let dk = gpu.alloc_from_slice(&c.k);
    let dv = gpu.alloc_from_slice(&c.v);
    let dbt = gpu.alloc_from_slice(&c.bt);
    let dsi = gpu.alloc_from_slice(&c.si);
    let dpo = gpu.alloc_from_slice(&c.pos);
    let dout = gpu.alloc_bytes(tokens * q_heads * hd * 2);
    let online_cfg = AttnCfg { nd: 8, rpt: 4, ntok_max: 4, bdim_max: 512, kv: KvDtype::Bf16 };
    t.paged_attn.run_tile_online_cfg(
        online_cfg, &dq, &dk, &dv, &dbt, &dsi, &dpo, &dout, q_scale, c.bpr as u32, bs as u32,
        q_heads as u32, kv_heads as u32, hd as u32, tokens as u32, &stream,
    );
    stream.sync();
    let refout: Vec<u16> = dout.read_to_vec();
    let poison = vec![0x7f7fu16; tokens * q_heads * hd];
    let mut ok_cnt = 0;
    let mut bad = 0usize;
    for v in QblkVar::ALL {
        for &(bdim, cap) in GEOMS {
            if let Some(n) = &only_v {
                if !v.name().contains(n.as_str()) || bdim != 128 {
                    continue;
                }
            }
            if !launchable(&t, v, bdim) || cap < v.jtr() {
                continue;
            }
            let cfg = QblkCfg { variant: v, bdim, jchunk: cap, jsplit: 0 };
            let g = PagedAttnCuda::qblk_geom(
                cfg,
                q_heads as u32,
                kv_heads as u32,
                hd as u32,
                tokens as u32,
            );
            if g.jsplit != 1 {
                continue; // spot 只看 js=1（jsplit 自动补并行会换合并路，o36 覆盖）
            }
            dout.upload_on(&poison, &stream);
            t.paged_attn.run_tile_qblk(
                cfg, &dq, &dk, &dv, &dbt, &dsi, &dpo, &dout, q_scale, c.bpr as u32, bs as u32,
                q_heads as u32, kv_heads as u32, hd as u32, tokens as u32, &stream,
            );
            stream.sync();
            let o: Vec<u16> = dout.read_to_vec();
            if o == poison {
                println!("| {} | {bdim} | {cap} | {} | {} | 未写输出 ❌ | | |", v.name().trim_start_matches("paged_attn_qblk_"), g.br, g.rows_nom);
                bad += 1;
                continue;
            }
            let (mut sse, mut sy, mut md, mut my) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            let mut nd = 0usize;
            for (&a, &b) in o.iter().zip(refout.iter()) {
                let (u, w) = (
                    wein_kernel::reference::bf16_to_f32(a) as f64,
                    wein_kernel::reference::bf16_to_f32(b) as f64,
                );
                sse += (u - w) * (u - w);
                sy += w * w;
                md = md.max((u - w).abs());
                my = my.max(w.abs());
                nd += (a != b) as usize;
            }
            let (r2, pk) = ((sse / sy).sqrt(), md / my);
            let flag = if r2 <= 3.0e-3 && pk <= 1.5e-2 { "✅" } else { "❌" };
            if flag == "✅" { ok_cnt += 1 } else { bad += 1 }
            println!(
                "| {} | {bdim} | {cap} | {} | {} | {nd} | {r2:.3e} | {pk:.3e} {flag} |",
                v.name().trim_start_matches("paged_attn_qblk_"),
                g.br,
                g.rows_nom
            );
        }
    }
    println!("正确性：{ok_cnt} 档过门，{bad} 档越门");
    dout.upload_on(&poison, &stream);
    let cfg = QblkCfg { variant: QblkVar::N16R8J2, bdim: 128, jchunk: 16, jsplit: 0 };
    t.paged_attn.run_tile_qblk(cfg, &dq, &dk, &dv, &dbt, &dsi, &dpo, &dout, q_scale, c.bpr as u32, bs as u32, q_heads as u32, kv_heads as u32, hd as u32, tokens as u32, &stream);
    stream.sync();
    let o: Vec<u16> = dout.read_to_vec();
    let f = |h:&str,x:&[u16]| println!("{h}: {}", x.iter().take(8).map(|&b| format!("{:.4}", wein_kernel::reference::bf16_to_f32(b))).collect::<Vec<_>>().join(" "));
    f("ref ", &refout); f("qblk", &o);
    let g2 = PagedAttnCuda::qblk_geom(cfg, q_heads as u32, kv_heads as u32, hd as u32, tokens as u32);
    println!("geom: br={} rows={} bdim={} ntile={} stride={} nblk={}", g2.br, g2.rows_nom, g2.bdim, g2.ntile, g2.stride, g2.nblk(kv_heads as u32));
    // 双跑 diff（同参数同输入）：非零 = 竞态
    let o2: Vec<u16> = { t.paged_attn.run_tile_qblk(cfg, &dq, &dk, &dv, &dbt, &dsi, &dpo, &dout, q_scale, c.bpr as u32, bs as u32, q_heads as u32, kv_heads as u32, hd as u32, tokens as u32, &stream); stream.sync(); dout.read_to_vec() };
    let d12 = o.iter().zip(o2.iter()).filter(|(x,y)| x != y).count();
    println!("双跑 diff: {d12}/{}", o.len());
    // 逐 token 位差图（前 64 token）：定位哪些 tile/行炸
    let rn = q_heads * hd;
    for tt in (0..tokens).step_by((tokens/16).max(1)) {
        let (a, b) = (&o[tt*rn..(tt+1)*rn], &refout[tt*rn..(tt+1)*rn]);
        let d = a.iter().zip(b.iter()).filter(|(x,y)| x != y).count();
        let mx = a.iter().zip(b.iter()).map(|(&x,&y)| (wein_kernel::reference::bf16_to_f32(x)-wein_kernel::reference::bf16_to_f32(y)).abs()).fold(0.0f32, f32::max);
        println!("tok {tt}: diff {d}/{rn} maxΔ {mx:.3e}");
    }
    // 逐 head 位差（tt=0/1）：定位炸的行 k（head=h，行内 4 头一组）
    for tt in 0..2usize.min(tokens) {
        let mut hs = String::new();
        for h in 0..q_heads {
            let d = o[tt*rn+h*hd..tt*rn+(h+1)*hd].iter().zip(refout[tt*rn+h*hd..tt*rn+(h+1)*hd].iter()).filter(|(x,y)| x != y).count();
            hs.push_str(&format!("{d} "));
        }
        println!("tok {tt} per-head diff: {hs}");
    }
    drop((dq, dk, dv, dbt, dsi, dpo, dout));

    // ── ② 单发计时 @T
    for &tokens in &[3009usize, 6009, 8009] {
        if let Some(want) = only_t {
            if want != tokens {
                continue;
            }
        }
        let c = case(tokens, q_heads, kv_heads, hd, bs);
        let dq = gpu.alloc_from_slice(&c.q);
        let dk = gpu.alloc_from_slice(&c.k);
        let dv = gpu.alloc_from_slice(&c.v);
        let (dbt, dsi, dpo) = (
            gpu.alloc_from_slice(&c.bt),
            gpu.alloc_from_slice(&c.si),
            gpu.alloc_from_slice(&c.pos),
        );
        let dout = gpu.alloc_bytes(tokens * q_heads * hd * 2);
        // 校准：同框架下计时 online 基线（n8r4 单遍档）
        {
            let run_o = || {
                t.paged_attn.run_tile_online_cfg(
                    online_cfg, &dq, &dk, &dv, &dbt, &dsi, &dpo, &dout, q_scale, c.bpr as u32,
                    bs as u32, q_heads as u32, kv_heads as u32, hd as u32, tokens as u32, &stream,
                )
            };
            for _ in 0..3 {
                run_o();
            }
            stream.sync();
            let (e0, e1) = (gpu.event(), gpu.event());
            e0.record_on(&stream);
            for _ in 0..10 {
                run_o();
            }
            e1.record_on(&stream);
            stream.sync();
            let oms = (e1.elapsed_ms(&e0) / 10.0) as f64;
            println!("online 基线 n8r4 单层：{oms:.3} ms（8 层 {:.1}）", oms * 8.0);
        }
        // 流量/算力对账（每层）：
        // KV DRAM 模型 = 块数 × T×hd×4B（K+V bf16，L2 命中前）；FLOP = 2×(T²/2)×qh×hd×2
        let fl = tokens as f64 * tokens as f64 / 2.0 * q_heads as f64 * hd as f64 * 2.0;
        println!(
            "\n## T={tokens}：单层 FLOP {fl:.3} G（fma 计 2）、纯 fma 地板 {:.1} ms@41.3T、8 层 {:.0} ms",
            fl / 41.3e12 * 1e3,
            fl / 41.3e12 * 1e3 * 8.0
        );
        println!(
            "| 变体 | bdim | cap | regs | local | Br | rows | 块数 | stride | ms | 8层 ms | KV MB | GB/s | 块/SM |"
        );
        println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
        let mut rows_out: Vec<String> = Vec::new();
        for v in QblkVar::ALL {
            for &(bdim, cap) in GEOMS {
                if !launchable(&t, v, bdim) || cap < v.jtr() {
                    continue;
                }
                let cfg = QblkCfg { variant: v, bdim, jchunk: cap, jsplit: 0 };
                let g = PagedAttnCuda::qblk_geom(
                    cfg,
                    q_heads as u32,
                    kv_heads as u32,
                    hd as u32,
                    tokens as u32,
                );
                if g.jsplit != 1 {
                    continue;
                }
                let run = || {
                    t.paged_attn.run_tile_qblk(
                        cfg, &dq, &dk, &dv, &dbt, &dsi, &dpo, &dout, q_scale, c.bpr as u32,
                        bs as u32, q_heads as u32, kv_heads as u32, hd as u32, tokens as u32,
                        &stream,
                    )
                };
                for _ in 0..3 {
                    run();
                }
                stream.sync();
                let (e0, e1) = (gpu.event(), gpu.event());
                e0.record_on(&stream);
                let n = 10;
                for _ in 0..n {
                    run();
                }
                e1.record_on(&stream);
                stream.sync();
                let ms = (e1.elapsed_ms(&e0) / n as f32) as f64;
                let (regs, local) = t.paged_attn.qblk_meta(v);
                let nblk = g.nblk(kv_heads as u32);
                let kv_mb = nblk as f64 * tokens as f64 * hd as f64 * 4.0 / 1e6;
                let gbs = kv_mb / 1e3 / (ms / 1e3);
                rows_out.push(format!(
                    "| {} | {bdim} | {cap} | {regs} | {local} | {} | {} | {nblk} | {} | {ms:.3} | {:.1} | {kv_mb:.0} | {gbs:.0} | {:.1} |",
                    v.name().trim_start_matches("paged_attn_qblk_"),
                    g.br,
                    g.rows_nom,
                    g.stride,
                    ms * 8.0,
                    nblk as f64 / 128.0
                ));
            }
        }
        for r in rows_out {
            println!("{r}");
        }
        drop((dq, dk, dv, dbt, dsi, dpo, dout));
    }
    let _ = group;
}
