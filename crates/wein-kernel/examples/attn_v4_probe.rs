//! v4 档（wmma tensor core）探针。
//!
//!   cargo run --release -p wein-kernel --example attn_v4_probe [-- 6009]
//!
//! ① 正确性 spot：t=512 对拍单遍档（run_tile_online）——rel2 ≤3e-3、
//!    峰值 ≤1.5e-2（容差档，判据同 o33/o37）；双跑 diff 查竞态。
//!    （历史 stage0/1/2/4/5/6/7/8 诊断块已随 kernel 调试代码一并退役。）
//! ② 单发计时 @T=3009/6009/8009：v4 vs v3 基线 + tensor core 地板对账。
//! GPU 一次一进程：本探针独占设备（纪律）。
use wein_kernel::cuda::{DevBuf, Gpu};
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

    let ptx = std::fs::read_to_string(wein_kernel::ptx_path()).expect("PTX 产物");
    let m = gpu.load_module(&ptx);
    let f4 = m.func("paged_attn_v4");

    let launch_v4 = |gpu: &Gpu, d: &Devs, tokens: usize, bpr: usize| {
        let tokens = tokens & !15; // host 合同：v4 只吃 16 对齐行，尾部行归 v3
        let row_tiles = tokens.div_ceil(16);
        // 合同：临时值必须先绑局部（as u32 字面量在语句尾即失效）
        let (bpr_u, bs_u, qh_u, kvh_u, hd_u, tk_u, n0, js1, c0) = (
            bpr as u32,
            bs as u32,
            q_heads as u32,
            kv_heads as u32,
            hd as u32,
            tokens as u32,
            0u32,
            1u32,
            0u32,
        );
        let args = wein_kernel::kernel_args![
            d.q,
            d.k,
            d.v,
            d.bt,
            d.si,
            d.pos,
            d.out,
            d.dummy,
            d.dummy,
            q_scale,
            bpr_u,
            bs_u,
            qh_u,
            kvh_u,
            hd_u,
            tk_u,
            n0,
            js1,
            c0,
            d.dummy
        ];
        gpu.launch(&f4, (q_heads * row_tiles) as u32, 32, &args, &stream);
    };

    // ── ① 参数自检：stage0 dump 指针，host 比对
    let tokens = 512usize;
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
    // ── 正确性 spot：t=512 对拍单遍档
    let online_cfg = AttnCfg { nd: 8, rpt: 4, ntok_max: 4, bdim_max: 512, kv: KvDtype::Bf16 };
    t.paged_attn.run_tile_online_cfg(
        online_cfg, &d.q, &d.k, &d.v, &d.bt, &d.si, &d.pos, &d.out, q_scale, c.bpr as u32,
        bs as u32, q_heads as u32, kv_heads as u32, hd as u32, tokens as u32, &stream,
    );
    stream.sync();
    let refout: Vec<u16> = d.out.read_to_vec();

    let poison = vec![0x7f7fu16; tokens * q_heads * hd];
    d.out.upload_on(&poison, &stream);
    launch_v4(&gpu, &d, tokens, c.bpr);
    stream.sync();
    let o: Vec<u16> = d.out.read_to_vec();
    if o == poison {
        println!("v4 未写输出 ❌");
        return;
    }
    let (mut sse, mut sy, mut md, mut my) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in o.iter().zip(refout.iter()) {
        let (u, w) = (
            wein_kernel::reference::bf16_to_f32(a) as f64,
            wein_kernel::reference::bf16_to_f32(b) as f64,
        );
        sse += (u - w) * (u - w);
        sy += w * w;
        md = md.max((u - w).abs());
        my = my.max(w.abs());
    }
    let (r2, pk) = ((sse / sy).sqrt(), md / my);
    let flag = if r2 <= 3.0e-3 && pk <= 1.5e-2 { "✅" } else { "❌" };
    println!("## 正确性 t=512：rel2={r2:.3e} 峰值={pk:.3e} {flag}");
    // 诊断：inf/nan 计数 + 行 0/头 0 前 8 值对比 + 首个坏元素坐标
    {
        let (mut ninf, mut nnan) = (0usize, 0usize);
        for &a in o.iter() {
            let f = wein_kernel::reference::bf16_to_f32(a);
            if f.is_infinite() { ninf += 1; }
            if f.is_nan() { nnan += 1; }
        }
        println!("  inf={ninf} nan={nnan} / {}", o.len());
        let f2 = |v: u16| wein_kernel::reference::bf16_to_f32(v);
        print!("  row0/h0 v4  : ");
        for i in 0..8 { print!("{:.4} ", f2(o[i])); }
        print!("\n  row0/h0 ref : ");
        for i in 0..8 { print!("{:.4} ", f2(refout[i])); }
        let mut bad = None;
        for (idx, (&a, &b)) in o.iter().zip(refout.iter()).enumerate() {
            let (u, w) = (f2(a) as f64, f2(b) as f64);
            if (u - w).abs() > 1e-2 * w.abs().max(1.0) { bad = Some(idx); break; }
        }
        if let Some(i) = bad {
            let (row, head, col) = (i / (q_heads * hd), (i / hd) % q_heads, i % hd);
            println!("\n  首坏 idx={i} row={row} head={head} col={col} v4={} ref={}", f2(o[i]), f2(refout[i]));
        }
        // 错误分布：按头 × 行段（16 token 段）计数
        let mut per_head = vec![0usize; q_heads];
        let mut per_rt = vec![0usize; tokens / 16];
        for (idx, (&a, &b)) in o.iter().zip(refout.iter()).enumerate() {
            let (u, w) = (f2(a) as f64, f2(b) as f64);
            if (u - w).abs() > 1e-2 * w.abs().max(1.0) {
                per_head[(idx / hd) % q_heads] += 1;
                per_rt[idx / (q_heads * hd * 16)] += 1;
            }
        }
        print!("  per-head: ");
        for h in 0..q_heads { print!("{} ", per_head[h]); }
        print!("\n  per-rowtile: ");
        for r in 0..tokens / 16 { print!("{} ", per_rt[r]); }
        println!();
        // rt0×head0 逐行坏列数 + 行1前8值对照
        for row in 0..16usize {
            let base = row * q_heads * hd;
            let badc = (0..hd).filter(|&c| {
                let (u, w) = (f2(o[base + c]) as f64, f2(refout[base + c]) as f64);
                (u - w).abs() > 1e-2 * w.abs().max(1.0)
            }).count();
            print!("{badc} ");
        }
        println!("  <- rt0/h0 per-row bad cols");
        let b1 = 1 * q_heads * hd;
        print!("  row1/h0 v4  : ");
        for c in 0..8 { print!("{:.4} ", f2(o[b1 + c])); }
        print!("\n  row1/h0 ref : ");
        for c in 0..8 { print!("{:.4} ", f2(refout[b1 + c])); }
        println!();
        // head4/row0：期望=V_kvh1[0][:] 精确透传（P=e_0,l=1）
        let b4 = 4 * hd;
        print!("  row0/h4 v4  : ");
        for cc in 0..16 { print!("{:.4} ", f2(o[b4 + cc])); }
        print!("\n  row0/h4 ref : ");
        for cc in 0..16 { print!("{:.4} ", f2(refout[b4 + cc])); }
        print!("\n  V_kvh1[0][:]: ");
        for cc in 0..16 { print!("{:.4} ", f2(c.v[hd + cc])); }
        print!("\n  V_kvh0[0][:]: ");
        for cc in 0..16 { print!("{:.4} ", f2(c.v[cc])); }
        println!();
    }
    let o2 = {
        launch_v4(&gpu, &d, tokens, c.bpr);
        stream.sync();
        d.out.read_to_vec::<u16>()
    };
    println!(
        "双跑 diff：{}（非 0 = 竞态）",
        o.iter().zip(o2.iter()).filter(|(a, b)| a != b).count()
    );
    drop(d);

    // ── ② 计时 @3009/6009/8009：v4 vs v3（br64 档）
    println!("\n## 单发计时（10 发均摊，ms/单层）");
    println!("| T | v4 | v3(br64) | v4/v3 | tensor 地板(ms) | 8 层 v4 |");
    println!("|---|---|---|---|---|---|");
    for tt in [3009usize, 6009usize, 8009usize] {
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
        let v4_ms = timeit(&|| launch_v4(&gpu, &d, tt, c.bpr));
        let cfg = V3Cfg { br: 64, kt: 16, bdim: 256, jsplit: 1 };
        let v3_ms = timeit(&|| {
            t.paged_attn.run_tile_v3(
                cfg, &d.q, &d.k, &d.v, &d.bt, &d.si, &d.pos, &d.out, q_scale, c.bpr as u32,
                bs as u32, q_heads as u32, kv_heads as u32, hd as u32, tt as u32, &stream,
            )
        });
        // attention FLOP = 2×(T²/2)×qh×hd×2（S+O 各一遍乘加）
        let fl = tt as f64 * tt as f64 / 2.0 * q_heads as f64 * hd as f64 * 2.0;
        let tfloor = fl / 165.0e12 * 1e3;
        println!(
            "| {tt} | {v4_ms:.3} | {v3_ms:.3} | {:.2}x | {tfloor:.2} | {:.1} |",
            v3_ms / v4_ms,
            v4_ms * 8.0
        );
    }
}

