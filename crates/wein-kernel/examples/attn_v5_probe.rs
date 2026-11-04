//! v5 档（4 warp × 64 行 CTA 共享 K/V smem）探针。
//!
//!   cargo run --release -p wein-kernel --example attn_v5_probe [-- 6009]
//!
//! ① 正确性 spot：t=512 / t=1088 v5 ≡ v4 逐位（位级合同，判据同 o39）+
//!    双跑 diff 查竞态。
//! ② 单发计时 @T=3009/6009/8009：v5 vs v4 + v5_meta（regs/local/smem）。
//! GPU 一次一进程：本探针独占设备（纪律）。
use wein_kernel::cuda::{DevBuf, Gpu};
use wein_kernel::ops::OpTable;

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

struct Devs<'a> {
    q: DevBuf<'a>,
    k: DevBuf<'a>,
    v: DevBuf<'a>,
    bt: DevBuf<'a>,
    si: DevBuf<'a>,
    pos: DevBuf<'a>,
    out: DevBuf<'a>,
    dummy: DevBuf<'a>,
}

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

    let (r4, l4, s4) = t.paged_attn.v4_meta();
    let (r5, l5, s5) = t.paged_attn.v5_meta();
    println!("## 编译属性：v4 {r4} regs/{l4}B local/{s4}B smem｜v5 {r5} regs/{l5}B local/{s5}B smem");

    // ── ① 正确性 spot：v5 ≡ v4 逐位（t=512 八组、t=1088 十七组=奇数组数）
    for tokens in [512usize, 1088] {
        if let Some(want) = only_t {
            if tokens != want {
                continue;
            }
        }
        let c = case(tokens, q_heads, kv_heads, hd, bs);
        let d = Devs {
            q: gpu.alloc_from_slice(&c.q),
            k: gpu.alloc_from_slice(&c.k),
            v: gpu.alloc_from_slice(&c.v),
            bt: gpu.alloc_from_slice(&c.bt),
            si: gpu.alloc_from_slice(&c.si),
            pos: gpu.alloc_from_slice(&c.pos),
            out: gpu.alloc_bytes(tokens * q_heads * hd * 2),
            dummy: gpu.alloc_bytes(4),
        };
        let poison = vec![0x7f7fu16; tokens * q_heads * hd];
        let run = |v5: bool| {
            d.out.upload_on(&poison, &stream);
            if v5 {
                t.paged_attn.run_tile_v5(
                    &d.q, &d.k, &d.v, &d.bt, &d.si, &d.pos, &d.out, q_scale, c.bpr as u32,
                    bs as u32, q_heads as u32, kv_heads as u32, hd as u32, tokens as u32,
                    &stream,
                );
            } else {
                t.paged_attn.run_tile_v4(
                    &d.q, &d.k, &d.v, &d.bt, &d.si, &d.pos, &d.out, q_scale, c.bpr as u32,
                    bs as u32, q_heads as u32, kv_heads as u32, hd as u32, tokens as u32,
                    &stream,
                );
            }
            stream.sync();
            d.out.read_to_vec::<u16>()
        };
        let v4 = run(false);
        let v5 = run(true);
        let v5b = run(true);
        let diff = v4.iter().zip(v5.iter()).filter(|(a, b)| a != b).count();
        let race = v5.iter().zip(v5b.iter()).filter(|(a, b)| a != b).count();
        let flag = if diff == 0 && race == 0 { "✅" } else { "❌" };
        println!("## t={tokens}：v5≡v4 位差 {diff}/{}、双跑 diff {race} {flag}", v4.len());
        drop(d);
    }

    // ── ② 计时 @3009/6009/8009：v5 vs v4（各取 64/16 对齐主段）
    println!("\n## 单发计时（10 发均摊，ms/单层）");
    println!("| T | v4 | v5 | v4/v5 | v5 8 层 | KV 流量(GB/层) v4→v5 |");
    println!("|---|---|---|---|---|---|");
    for tt in [3009usize, 6009usize, 8009usize, 32000] {
        if let Some(want) = only_t {
            if tt != want {
                continue;
            }
        }
        let c = case(tt, q_heads, kv_heads, hd, bs);
        let d = Devs {
            q: gpu.alloc_from_slice(&c.q),
            k: gpu.alloc_from_slice(&c.k),
            v: gpu.alloc_from_slice(&c.v),
            bt: gpu.alloc_from_slice(&c.bt),
            si: gpu.alloc_from_slice(&c.si),
            pos: gpu.alloc_from_slice(&c.pos),
            out: gpu.alloc_bytes(tt * q_heads * hd * 2),
            dummy: gpu.alloc_bytes(4),
        };
        let timeit = |run: &dyn Fn()| -> f64 {
            for _ in 0..3 {
                run();
            }
            stream.sync();
            let (e0, e1) = (gpu.event(), gpu.event());
            e0.record_on(&stream);
            for _ in 0..10 {
                run();
            }
            e1.record_on(&stream);
            stream.sync();
            (e1.elapsed_ms(&e0) / 10.0) as f64
        };
        let t4 = (tt & !15) as u32;
        let t64 = (tt & !63) as u32;
        let v4_ms = timeit(&|| {
            t.paged_attn.run_tile_v4(
                &d.q, &d.k, &d.v, &d.bt, &d.si, &d.pos, &d.out, q_scale, c.bpr as u32, bs as u32,
                q_heads as u32, kv_heads as u32, hd as u32, t4, &stream,
            )
        });
        let v5_ms = timeit(&|| {
            t.paged_attn.run_tile_v5(
                &d.q, &d.k, &d.v, &d.bt, &d.si, &d.pos, &d.out, q_scale, c.bpr as u32, bs as u32,
                q_heads as u32, kv_heads as u32, hd as u32, t64, &stream,
            )
        });
        // 每 kvh 流被消费次数：v4 每 16 行组×每 q 头一次（÷16 粒度重读 4 头），
        // v5 每 64 行组一次——近似 KV→SM 流量 = 组数×4q头×平均前缀×1KB
        let avg = |t: usize| t as f64 / 2.0;
        let g4 = (t4 as f64 / 16.0) * 4.0 * avg(t4 as usize) / 1.0e9;
        let g5 = (t64 as f64 / 64.0) * 4.0 * avg(t64 as usize) / 1.0e9;
        println!(
            "| {tt} | {v4_ms:.3} | {v5_ms:.3} | {:.2}x | {:.1} | {g4:.2}→{g5:.2} |",
            v4_ms / v5_ms,
            v5_ms * 8.0
        );
        drop(d);
    }
}
