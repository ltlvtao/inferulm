//! v3 调试：t=64 单块单 seq，逐 token/head 位差图 + 打印首行值
use wein_kernel::cuda::Gpu;
use wein_kernel::ops::OpTable;
use wein_kernel::ops::{AttnCfg, KvDtype, V3Cfg};

fn bf16_vec(seed_hi: u64, n: usize, scale: f32) -> Vec<u16> {
    (0..n as u64)
        .map(|i| {
            let z = i.wrapping_mul(6364136223846793005).wrapping_add(seed_hi);
            let f = ((z >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * scale;
            wein_kernel::reference::f32_to_bf16(f)
        })
        .collect()
}

fn main() {
    let gpu = Gpu::probe().unwrap();
    let profile = wein_platform::HardwareProfile::with_device(wein_platform::Device::Cuda)
        .with_arch("sm_89");
    let t = OpTable::resolve(&gpu, &profile).unwrap();
    let stream = gpu.stream();
    let (bs, qh, kvh, hd) = (16usize, 16usize, 4usize, 256usize);
    let q_scale = 1.0 / (hd as f32).sqrt();
    let tokens = 64usize;
    let he = kvh * hd;
    let bpr = tokens.div_ceil(bs);
    let slots = bpr * bs;
    // 判别实验：WEIN_DBGQ=ones → q 全 1（s_j=q_scale·Σ_d k_j[d]，强 jj 依赖）；
    // 默认 one-hot dim0。若 ones 下 p̂ 仍均匀 ⟹ K smem 对 jj 塌；若变化 ⟹ q 读 0。
    let mode = std::env::var("WEIN_DBGQ").unwrap_or_default();
    let mut qf = vec![0.0f32; tokens * qh * hd];
    for tt in 0..tokens {
        for h in 0..qh {
            if mode == "ones" {
                for d in 0..hd {
                    qf[(tt * qh + h) * hd + d] = 1.0;
                }
            } else {
                qf[(tt * qh + h) * hd] = 1.0;
            }
        }
    }
    // WEIN_DBGQ=ramp：k[j][kh][d]=j·0.01+d·0.0001，q=ones ⟹ s_j−s_0=0.16·j，
    // 由 ln(p̂_j/p̂_0)/0.16 直读内核实际读到的 token 置换 σ(j)
    if mode == "ramp" {
        for tt in 0..tokens {
            for h in 0..qh {
                for d in 0..hd {
                    qf[(tt * qh + h) * hd + d] = 1.0;
                }
            }
        }
    }
    let q: Vec<u16> = qf.iter().map(|&x| wein_kernel::reference::f32_to_bf16(x)).collect();
    let k = if mode == "ramp" {
        let mut kf = vec![0.0f32; slots * he];
        for j in 0..slots {
            for kh2 in 0..kvh {
                for d in 0..hd {
                    kf[j * he + kh2 * hd + d] = j as f32 * 0.01 + d as f32 * 0.0001;
                }
            }
        }
        kf.iter().map(|&x| wein_kernel::reference::f32_to_bf16(x)).collect::<Vec<u16>>()
    } else {
        bf16_vec(0x7b33, slots * he, 1.0)
    };
    // v one-hot：v[j][kh][d] = (d == j)——out[d] 直接读出归一化 p_j
    let mut vf = vec![0.0f32; slots * he];
    for j in 0..tokens {
        for kh2 in 0..kvh {
            vf[j * he + kh2 * hd + j] = 1.0;
        }
    }
    let v: Vec<u16> = vf.iter().map(|&x| wein_kernel::reference::f32_to_bf16(x)).collect();
    let bt: Vec<u32> = (0..bpr as u32).collect();
    let si = vec![0u32; tokens];
    let po: Vec<u32> = (0..tokens as u32).collect();
    let dq = gpu.alloc_from_slice(&q);
    let dk = gpu.alloc_from_slice(&k);
    let dv = gpu.alloc_from_slice(&v);
    let dbt = gpu.alloc_from_slice(&bt);
    let dsi = gpu.alloc_from_slice(&si);
    let dpo = gpu.alloc_from_slice(&po);
    let dout = gpu.alloc_bytes(tokens * qh * hd * 2);
    let online_cfg = AttnCfg { nd: 8, rpt: 4, ntok_max: 4, bdim_max: 512, kv: KvDtype::Bf16 };
    t.paged_attn.run_tile_online_cfg(
        online_cfg, &dq, &dk, &dv, &dbt, &dsi, &dpo, &dout, q_scale, bpr as u32, bs as u32,
        qh as u32, kvh as u32, hd as u32, tokens as u32, &stream,
    );
    stream.sync();
    let refout: Vec<u16> = dout.read_to_vec();
    let cfg = V3Cfg { br: 64, kt: 16, bdim: 32, jsplit: 1 };
    t.paged_attn.run_tile_v3(
        cfg, &dq, &dk, &dv, &dbt, &dsi, &dpo, &dout, q_scale, bpr as u32, bs as u32,
        qh as u32, kvh as u32, hd as u32, tokens as u32, &stream,
    );
    stream.sync();
    let o: Vec<u16> = dout.read_to_vec();
    let f = |h: &str, x: &[u16]| {
        println!("{h}: {}", x.iter().take(8).map(|&b| format!("{:.4}", wein_kernel::reference::bf16_to_f32(b))).collect::<Vec<_>>().join(" "))
    };
    f("ref ", &refout);
    f("v3  ", &o);
    // 第二行（head 0, token 1）
    f("ref t1", &refout[qh * hd..]);
    f("v3  t1", &o[qh * hd..]);
    // per-token diff 图（前 16 token）
    let rn = qh * hd;
    for tt in 0..16 {
        let (a, b) = (&o[tt * rn..(tt + 1) * rn], &refout[tt * rn..(tt + 1) * rn]);
        let d = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
        println!("tok {tt}: diff {d}/{rn}");
    }
    // per-head diff（tt=0）
    let mut hs = String::new();
    for h in 0..qh {
        let d = o[h * hd..(h + 1) * hd].iter().zip(refout[h * hd..(h + 1) * hd].iter()).filter(|(x, y)| x != y).count();
        hs.push_str(&format!("{d} "));
    }
    println!("tok0 per-head diff: {hs}");
    // v one-hot 直读 p̂：out[d]=p̂_{j=d}；对比期望 + 隐含对数权重差 vs 分值差
    let bf = |x: u16| wein_kernel::reference::bf16_to_f32(x) as f64;
    for &tt in &[2usize, 8, 20, 40] {
        let (h, kh) = (0usize, 0usize);
        let nj = tt + 1;
        let s: Vec<f64> = (0..nj)
            .map(|j| {
                if mode == "ones" || mode == "ramp" {
                    (0..hd).map(|d| bf(k[j * he + kh * hd + d])).sum::<f64>() * q_scale as f64
                } else {
                    bf(k[(j * he) + kh * hd]) * q_scale as f64
                }
            })
            .collect();
        let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let den: f64 = s.iter().map(|&x| (x - m).exp()).sum();
        let p0 = bf(o[(tt * qh + h) * hd]);
        println!("== t{tt} h{h}: p̂_j | exp | ln(p̂_j/p̂_0) | s_j−s_0 ==");
        for j in 0..nj {
            let pj = bf(o[(tt * qh + h) * hd + j]);
            let e = (s[j] - m).exp() / den;
            let ld = if pj > 0.0 && p0 > 0.0 { (pj / p0).ln() } else { f64::NAN };
            println!("  j{j}: {pj:.6} | {e:.6} | {ld:+.4} | {:+.4}", s[j] - s[0]);
        }
    }
}
