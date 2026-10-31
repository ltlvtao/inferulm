//! 长上下文 attention 分块档实测（旧几何 vs 分块几何）。
//!
//!   cargo run --release -p wein-kernel --example attn_tile_probe
//!
//! 三种工作点：
//! - decode 形：tokens=1 × q 头全量，pos = 1k/3k/8k（长上下文单发吞吐）
//! - prefill 形：tokens=256 的 chunk，位置尾 = 768/1536/3072（chunk 几何）
//! 每个工作点跑旧 kernel（`paged_attn`，长 pos 走流式路）与分块档
//! （`paged_attn_tile`，prefill 落 jsplit=1），各 3 热身 + 10 拍事件计时，
//! 并给 KV 实际读量（按各自几何算）与距 DRAM 地板倍数。
use wein_kernel::cuda::Gpu;
use wein_kernel::ops::{OpTable, PagedAttnOp};
use wein_platform::{Device, HardwareProfile};

fn bf16_vec(seed_hi: u64, n: usize, scale: f32) -> Vec<u16> {
    (0..n as u64)
        .map(|i| {
            let z = i.wrapping_mul(6364136223846793005).wrapping_add(seed_hi);
            let f = ((z >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * scale;
            f.to_bits() as u16
        })
        .collect()
}

/// 单序列、页顺排的 paged 池（页表间接寻址真实：每 token 一次查表）
struct Case {
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
    bt: Vec<u32>,
    si: Vec<u32>,
    pos: Vec<u32>,
    bpr: usize,
}

fn case(tokens: usize, pos0: usize, q_heads: usize, kv_heads: usize, hd: usize, bs: usize)
    -> Case
{
    let he = kv_heads * hd;
    let pmax = pos0 + tokens - 1;
    let bpr = (pmax + 1).div_ceil(bs);
    let npages = bpr;
    let slots = npages * bs;
    Case {
        q: bf16_vec(0x7a11, tokens * q_heads * hd, 2.0),
        k: bf16_vec(0x7b33, slots * he, 1.0),
        v: bf16_vec(0x7c55, slots * he, 1.0),
        bt: (0..npages as u32).collect(),
        si: vec![0u32; tokens],
        pos: (0..tokens).map(|i| (pos0 + i) as u32).collect(),
        bpr,
    }
}

fn main() {
    let gpu = Gpu::probe().expect("须有 CUDA 设备");
    let profile = HardwareProfile::with_device(Device::Cuda).with_arch("sm_89");
    let t = OpTable::resolve(&gpu, &profile).expect("sm_89 画像须解析成功");
    let stream = gpu.stream();
    let (bs, mut q_heads, mut kv_heads, mut hd) = (16usize, 16usize, 4usize, 256usize);
    // 形状可覆盖（WEIN_PROBE_SHAPE="32,16,128"）：与 o29 的旧形状对照
    if let Ok(v) = std::env::var("WEIN_PROBE_SHAPE") {
        let a: Vec<usize> = v.split(',').filter_map(|t| t.trim().parse().ok()).collect();
        assert_eq!(a.len(), 3, "WEIN_PROBE_SHAPE 须为 q,kv,hd");
        (q_heads, kv_heads, hd) = (a[0], a[1], a[2]);
    }
    let he = kv_heads * hd;
    let q_scale = 1.0 / (hd as f32).sqrt();

    println!(
        "形状：q_heads={q_heads} kv_heads={kv_heads} hd={hd} block={bs}；\
         每 token 每层唯一 KV = {:.1} KB",
        (kv_heads * hd * 2 * 2) as f64 / 1024.0
    );

    // (标签, tokens, 首位置)
    let mut points: Vec<(String, usize, usize)> = Vec::new();
    for pos in [1024usize, 3072, 8192] {
        points.push((format!("decode pos={}", pos - 1), 1, pos - 1));
    }
    for (tag, end) in [("prefill chunk@768", 768usize), ("@1536", 1536), ("@3072", 3072)] {
        points.push((format!("{tag} (t=256)"), 256, end - 256));
    }

    println!("\n| 工作点 | 旧 ms | 新 ms | 加速 | 旧 KV 读 | 新 KV 读 | 新/地板 |");
    println!("|---|---|---|---|---|---|---|");
    for (tag, tokens, pos0) in points {
        let c = case(tokens, pos0, q_heads, kv_heads, hd, bs);
        let (ntok, nv, rows, bdim, ntile, jsplit) =
            wein_kernel::ops::PagedAttnCuda::tile_geom(
                q_heads as u32, kv_heads as u32, hd as u32, tokens as u32,
            );
        eprintln!("[case {tag}] ntok={ntok} nv={nv} rows={rows} bdim={bdim} ntile={ntile} jsplit={jsplit}");
        let slots = c.bt.len() * bs;
        let dq = gpu.alloc_from_slice(&c.q);
        let dk = gpu.alloc_from_slice(&c.k);
        let dv = gpu.alloc_from_slice(&c.v);
        let dbt = gpu.alloc_from_slice(&c.bt);
        let dsi = gpu.alloc_from_slice(&c.si);
        let dpo = gpu.alloc_from_slice(&c.pos);
        let dout = gpu.alloc_bytes(tokens * q_heads * hd * 2);
        let _ = slots;
        let run_old = || {
            t.paged_attn.run(
                &dq, &dk, &dv, &dbt, &dsi, &dpo, &dout, q_scale, c.bpr as u32, bs as u32,
                q_heads as u32, kv_heads as u32, hd as u32, tokens as u32, &stream,
            )
        };
        let run_new = || {
            t.paged_attn.run_tile(
                &dq, &dk, &dv, &dbt, &dsi, &dpo, &dout, q_scale, c.bpr as u32, bs as u32,
                q_heads as u32, kv_heads as u32, hd as u32, tokens as u32, &stream,
            )
        };
        let time = |f: &dyn Fn()| -> f64 {
            for _ in 0..3 {
                f();
            }
            stream.sync();
            let (e0, e1) = (gpu.event(), gpu.event());
            e0.record_on(&stream);
            for _ in 0..10 {
                f();
            }
            e1.record_on(&stream);
            stream.sync();
            (e1.elapsed_ms(&e0) / 10.0) as f64
        };
        // 旧路径 = 参考；新路跑前把 out 污化，确保"同"不是残留
        let old_ms = time(&run_old);
        let new_ms = time(&run_new);
        run_old();
        stream.sync();
        let o: Vec<u16> = dout.read_to_vec();
        let poison = vec![0xdeadu16; o.len()];
        dout.upload_on(&poison, &stream);
        run_new();
        stream.sync();
        let n: Vec<u16> = dout.read_to_vec();
        let same = o == n;
        assert_ne!(n, poison, "分块档没写 out");

        // KV 读量：旧 = tokens·q_heads·(pos+1)·hd·4B（每 q 行各读一遍 K/V）；
        // 新 = nblk·Lsplit·hd·4B（一块内 ntok·group 行共享一段）

        let avg_l = (0..tokens).map(|i| pos0 + i + 1).sum::<usize>() as f64 / tokens as f64;
        let old_bytes = tokens as f64 * q_heads as f64 * avg_l * hd as f64 * 4.0;
        let nblk = kv_heads * ntile * jsplit;
        let new_bytes = nblk as f64 * (avg_l / jsplit as f64) * hd as f64 * 4.0;
        let floor_bytes = avg_l * kv_heads as f64 * hd as f64 * 4.0;
        println!(
            "| {} | {:.3} | {:.3} | {:.2}× | {:.1} MB | {:.1} MB | {:.1}× |",
            tag,
            old_ms,
            new_ms,
            old_ms / new_ms,
            old_bytes / 1e6,
            new_bytes / 1e6,
            new_bytes / floor_bytes
        );
        let ms_floor = floor_bytes / 1.0e12 * 1.0e3; // @1TB/s
        // prefill 档每条 token 片都重读整段 j 区间（无 smem，只能靠 L2）：
        // "新/地板" 就是 ntile，即 L2 承担的超读倍数；有效速率才是真约束
        println!(
            "   └ 位级同 {}｜理想单读 {:.3} ms｜新/理想单读 {:.1}×（=ntile，L2 承担）             ｜旧/理想单读 {:.1}×｜新有效速率 {:.0} GB/s｜几何 nblk={} (ntile={} jsplit={})",
            same,
            ms_floor,
            new_bytes / floor_bytes,
            old_bytes / floor_bytes,
            new_bytes / 1e9 / (new_ms / 1e3),
            nblk,
            ntile,
            jsplit
        );
        assert!(same, "分块档（jsplit=1）应与旧路位级相同");
        drop((dq, dk, dv, dbt, dsi, dpo, dout));
        let _ = he;
    }
}
