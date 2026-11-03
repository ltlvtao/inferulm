//! v3 档（smem K/V tile + shfl warp 归约）探针。
//!
//!   cargo run --release -p wein-kernel --example attn_v3_probe [-- 6009]
//!
//! ① 变体属性表：regs / local spill / smem（cuFuncGetAttribute）。
//! ② 正确性 spot：t=512（pos=0 起）对拍单遍档（`run_tile_online`）——
//!    rel2 ≤3e-3、峰值 ≤1.5e-2（容差档，判据同 o33/o37）；双跑 diff 查竞态。
//! ③ 单发计时 @T=3009/6009/8009：Br × KT × bdim 全档表 + online 基线校准
//!    + KV 流量/FLOP 地板对账。硬门槛：6009 全 T 8 层 attn ≤120ms。
//! GPU 一次一进程：本探针独占设备（纪律）。
use wein_kernel::cuda::Gpu;
use wein_kernel::ops::OpTable;
use wein_kernel::ops::{AttnCfg, KvDtype, PagedAttnCuda, V3Cfg, V3Var};

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

/// (Br, KT, bdim) 扫描档
const GEOMS: &[(usize, usize, usize)] = &[
    (8, 16, 256),
    (8, 16, 384),
    (10, 16, 384),
    (12, 16, 384),
    (16, 16, 256),
    (12, 16, 256),
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let only_t: Option<usize> = args.get(1).and_then(|s| s.parse().ok());
    let gpu = Gpu::probe().expect("须有 CUDA 设备");
    let profile = wein_platform::HardwareProfile::with_device(wein_platform::Device::Cuda)
        .with_arch("sm_89");
    let t = OpTable::resolve(&gpu, &profile).expect("sm_89 画像须解析成功");
    let stream = gpu.stream();
    let (bs, q_heads, kv_heads, hd) = (16usize, 16usize, 4usize, 256usize);
    let q_scale = 1.0 / (hd as f32).sqrt();

    // ── ① 变体属性表（regs/local/smem）
    println!("## 变体属性（cuFuncGetAttribute）");
    println!("| 变体 | NV | KT | regs | local B | smem B | smem 档 |");
    println!("|---|---|---|---|---|---|---|");
    for v in V3Var::ALL {
        let (regs, local, smem) = t.paged_attn.v3_meta(v);
        let occ2 = if smem * 2 <= 100 * 1024 { "2块/SM✓" } else { "1块/SM" };
        println!(
            "| {} | {} | {} | {} | {} | {} | {occ2} |",
            v.name().trim_start_matches("paged_attn_v3_"),
            v.nv(),
            v.kt(),
            regs,
            local,
            smem
        );
    }
    let (oregs, olocal) = t.paged_attn.online_meta();
    println!("| (对照 online 基线档) | | | {oregs} | {olocal} | 0 | |");

    // ── ② 正确性 spot：t=512 对拍单遍档
    println!("\n## 正确性 spot（t=512，对拍 run_tile_online；门 rel2≤3e-3/峰值≤1.5e-2）");
    println!("| 变体 | Br | KT | bdim | 位差 | rel2 | 峰值 |");
    println!("|---|---|---|---|---|---|---|");
    let tokens = 512usize;
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
    for v in V3Var::ALL {
        if v.hd() != 256 {
            continue; // hd=256 形跳过 hd128 档（N8 由 o37 的 hd128 用例覆盖）
        }
        for &(br, kt, bdim) in GEOMS {
            if v.kt() != kt {
                continue;
            }
            let cfg = V3Cfg { br, kt, bdim, jsplit: 1 };
            dout.upload_on(&poison, &stream);
            t.paged_attn.run_tile_v3(
                cfg, &dq, &dk, &dv, &dbt, &dsi, &dpo, &dout, q_scale, c.bpr as u32, bs as u32,
                q_heads as u32, kv_heads as u32, hd as u32, tokens as u32, &stream,
            );
            stream.sync();
            let o: Vec<u16> = dout.read_to_vec();
            if o == poison {
                println!("| {} | {br} | {kt} | {bdim} | 未写输出 ❌ | | |", v.name());
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
                "| {} | {br} | {kt} | {bdim} | {nd} | {r2:.3e} | {pk:.3e} {flag} |",
                v.name().trim_start_matches("paged_attn_v3_")
            );
        }
    }
    println!("正确性：{ok_cnt} 档过门，{bad} 档越门");
    // 双跑 diff（同参数同输入）：非零 = 竞态
    {
        let cfg = V3Cfg { br: 64, kt: 16, bdim: 256, jsplit: 1 };
        let run = || {
            t.paged_attn.run_tile_v3(
                cfg, &dq, &dk, &dv, &dbt, &dsi, &dpo, &dout, q_scale, c.bpr as u32, bs as u32,
                q_heads as u32, kv_heads as u32, hd as u32, tokens as u32, &stream,
            );
            stream.sync();
            dout.read_to_vec()
        };
        let o1: Vec<u16> = run();
        let o2: Vec<u16> = run();
        let d12 = o1.iter().zip(o2.iter()).filter(|(x, y)| x != y).count();
        println!("双跑 diff: {d12}/{}", o1.len());
    }
    drop((dq, dk, dv, dbt, dsi, dpo, dout));

    // ── ③ 单发计时 @T
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
        let dbt = gpu.alloc_from_slice(&c.bt);
        let dsi = gpu.alloc_from_slice(&c.si);
        let dpo = gpu.alloc_from_slice(&c.pos);
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
        // FLOP = 2×(T²/2)×qh×hd×2；DRAM 独占量 = T×kvh×hd×2B×2（K+V 各过一次）
        let fl = tokens as f64 * tokens as f64 / 2.0 * q_heads as f64 * hd as f64 * 2.0;
        let dram = tokens as f64 * kv_heads as f64 * hd as f64 * 2.0 * 2.0 / 1e6;
        println!(
            "\n## T={tokens}：单层 FLOP {fl:.3} G（fma 计 2）、纯 fma 地板 {:.1} ms@41.3T（8 层 {:.0} ms）、KV DRAM 独占 {dram:.0} MB",
            fl / 41.3e12 * 1e3,
            fl / 41.3e12 * 1e3 * 8.0,
        );
        println!("| Br | KT | bdim | regs | local | 块数 | js | passes | ms | 8层 ms | L2载 GB | GB/s |");
        println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
        for &(br, kt, bdim) in GEOMS {
            let js0: usize = std::env::var("WEIN_V3_JS").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
            let cfg = V3Cfg { br, kt, bdim, jsplit: js0 };
            let g = PagedAttnCuda::v3_geom(cfg, q_heads as u32, kv_heads as u32, hd as u32, tokens as u32);
            let run = || {
                t.paged_attn.run_tile_v3(
                    cfg, &dq, &dk, &dv, &dbt, &dsi, &dpo, &dout, q_scale, c.bpr as u32,
                    bs as u32, q_heads as u32, kv_heads as u32, hd as u32, tokens as u32, &stream,
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
            let (regs, local, _) = t.paged_attn.v3_meta(g.variant);
            let nblk = g.nblk(kv_heads as u32);
            let rpw = g.variant.rpw();
            let npass = (br * g.group).div_ceil(bdim / 32 * rpw);
            // L2 载入模型：npass × span × hd × 2(KV) × 2B × 块数（每 pass 全 span 重载）
            let l2_gb = nblk as f64 * npass as f64 * tokens as f64 * hd as f64 * 2.0 * 2.0 / 1e9;
            let gbs = l2_gb / (ms / 1e3);
            println!(
                "| {br} | {kt} | {bdim} | {regs} | {local} | {nblk} | {} | {npass} | {ms:.3} | {:.1} | {l2_gb:.1} | {gbs:.0} |",
                g.jsplit,
                ms * 8.0
            );
        }
        drop((dq, dk, dv, dbt, dsi, dpo, dout));
    }
}
