//! o44 · **单组档**（ntok=1 ⇒ 每 block 单行组）容差档。
//!
//! 生产 `cfg_rows`（exec 层）对行数 ∈[4, PART_ROWS/qh] 的 online 段批换
//! `attn_cfg_sgl`（nd=8/rpt=4/ntok=1）——杀行组读放大（现役几何 8×）。
//! 数值上只可能差两点：jsplit 与 (nd,rpt)——同一行的逐元素算术序不变
//! （fma 链 d 升序、行内 max 扫 v 升序、l 按 j 升序），jsplit 变 ⇒ 偏量
//! 合并的结合律变 ⇒ 与两遍档（位级锚）只到容差。判据照 o31/o33：
//! rel2 + 峰值归一 + 逐行最坏，另钉「两 cfg 同 jsplit 时逐位同」的强档
//! （js=1 大 t 形：ntok 影响块映射不影响任何归约序）。
use wein_kernel::cuda::Gpu;
use wein_kernel::ops::{AttnCfg, OpTable, PagedAttnCuda};
use wein_kernel::reference;
use wein_platform::{Device, HardwareProfile};

/// GPU 串行锁（ops_m0/ops_i1 同款）：并行测试线程的 kernel 会互相闯入计时窗。
fn guard() -> std::sync::MutexGuard<'static, ()> {
    static L: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    L.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

fn gpu() -> Option<Gpu> {
    match Gpu::probe() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("SKIP（无可用 CUDA 设备：{e}）");
            None
        }
    }
}

fn table(gpu: &Gpu) -> OpTable<'_> {
    let profile = HardwareProfile::with_device(Device::Cuda).with_arch("sm_89");
    OpTable::resolve(gpu, &profile).expect("sm_89 画像须解析成功")
}

fn vec_bf16(seed: u64, n: usize, scale: f32) -> Vec<u16> {
    (0..n as u64)
        .map(|i| {
            let z = i.wrapping_mul(6364136223846793005).wrapping_add(seed);
            let f = ((z >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * scale;
            reference::f32_to_bf16(f)
        })
        .collect()
}

fn norm(x: &[u16], y: &[u16]) -> (f64, f64, usize) {
    let (mut sse, mut sy, mut max_d, mut max_y) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in x.iter().zip(y.iter()) {
        let (u, v) = (reference::bf16_to_f32(a) as f64, reference::bf16_to_f32(b) as f64);
        sse += (u - v) * (u - v);
        sy += v * v;
        max_d = max_d.max((u - v).abs());
        max_y = max_y.max(v.abs());
    }
    ((sse / sy).sqrt(), max_d / max_y, x.iter().zip(y.iter()).filter(|(a, b)| a != b).count())
}

/// 同一缓冲跑两 cfg（tile_case 同款脚手架，cfg 参数化）。
fn online_with(
    gpu: &Gpu, t: &OpTable<'_>, cfg: AttnCfg,
    qh: usize, kvh: usize, hd: usize, tokens: usize, pos: usize,
) -> (Vec<u16>, usize) {
    let stream = gpu.stream();
    let block_size = 16usize;
    let bpr = pos.div_ceil(block_size);
    let nseq = 2usize;
    let block_table: Vec<u32> = (0..(nseq * bpr) as u32).collect();
    let seq_ids: Vec<u32> = (0..tokens).map(|r| (r % nseq) as u32).collect();
    let positions = vec![pos as u32; tokens];
    let he = kvh * hd;
    let slots = nseq * bpr * block_size;
    let (k_cache, v_cache) = (vec_bf16(0x51ab, slots * he, 1.0), vec_bf16(0x62cd, slots * he, 1.0));
    let qn = tokens * qh * hd;
    let q = vec_bf16(0x73ef, qn, 2.0);
    let q_scale = 1.0 / (hd as f32).sqrt();
    let (dq, dkc, dvc) = (
        gpu.alloc_from_slice(&q), gpu.alloc_from_slice(&k_cache), gpu.alloc_from_slice(&v_cache),
    );
    let (dbt, dsi, dpo) = (
        gpu.alloc_from_slice(&block_table), gpu.alloc_from_slice(&seq_ids),
        gpu.alloc_from_slice(&positions),
    );
    let dout = gpu.alloc_bytes(qn * 2);
    let poison = vec![0x7f7fu16; qn];
    dout.upload_on(&poison, &stream);
    t.paged_attn.run_tile_online_cfg(
        cfg, &dq, &dkc, &dvc, &dbt, &dsi, &dpo, &dout, q_scale, bpr as u32, block_size as u32,
        qh as u32, kvh as u32, hd as u32, tokens as u32, &stream,
    );
    stream.sync();
    let out = dout.read_to_vec();
    assert_ne!(out, poison, "o44 单组档未写输出（{qh}qh/{kvh}kvh/hd{hd} t={tokens}）");
    let g = PagedAttnCuda::online_geom(cfg, qh as u32, kvh as u32, hd as u32, tokens as u32);
    (out, g.jsplit)
}

#[test]
fn o44_online_single_group_tolerance() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let t = table(&gpu);
    let sgl = AttnCfg { nd: 8, rpt: 4, ntok_max: 1, ..AttnCfg::baseline() };
    let cur = AttnCfg { rpt: 4, ..AttnCfg::baseline() }; // t>4 生产档（放大 8×）
    // (qh, kvh, hd, tokens, pos)：覆盖 4/8/16/48 行 × 中长 ctx（生产换档区间）
    let cases = [
        (16usize, 4usize, 256usize, 4usize, 3071usize),
        (16, 4, 256, 8, 5000),
        (16, 4, 256, 16, 8191),
        (16, 4, 256, 48, 6000),
    ];
    for (qh, kvh, hd, tokens, pos) in cases {
        let tag = format!("{qh}qh/{kvh}kvh/hd{hd} t={tokens} pos={pos}");
        let (tile, _) = online_with(&gpu, &t, cur, qh, kvh, hd, tokens, pos);
        let (single, js) = online_with(&gpu, &t, sgl, qh, kvh, hd, tokens, pos);
        let (r2, pk, d) = norm(&single, &tile);
        // 逐行最坏峰值（全局量会被均值稀释）
        let row_n = qh * hd;
        let mut worst = 0.0f64;
        for r in 0..tokens {
            let (a, b) = (&single[r * row_n..(r + 1) * row_n], &tile[r * row_n..(r + 1) * row_n]);
            let (_, p, _) = norm(a, b);
            worst = worst.max(p);
        }
        eprintln!(
            "o44 {tag}：jsplit={js}｜单组 vs 现役 rel2 {r2:.3e} 峰值 {pk:.3e}（{d}/{} 位差）｜逐行最坏 {worst:.3e}",
            single.len()
        );
        // 门照 o31/o33（结合律差 + bf16 末位量子；实测见 eprintln）
        assert!(r2 <= 3.0e-3, "o44 rel2 越门：{r2:.3e}（{tag}）");
        assert!(pk <= 1.5e-2, "o44 峰值越门：{pk:.3e}（{tag}）");
        assert!(worst <= 1.5e-2, "o44 逐行最坏越门：{worst:.3e}（{tag}）");
    }

    // 强档：两 cfg 同落 jsplit=1（t×qh > PART_ROWS ⇒ 无分片 ⇒ 偏量合并不参与）
    // 时逐位同——单组只改线程映射 (ntok, bdim)，不动任何归约序。
    let (big, big_js) = online_with(&gpu, &t, cur, 16, 4, 256, 100, 3000);
    let (big_s, big_s_js) = online_with(&gpu, &t, sgl, 16, 4, 256, 100, 3000);
    assert_eq!(big_js, 1, "o44 强档须落 jsplit=1（得 {big_js}）");
    assert_eq!(big_s_js, 1, "o44 强档单组须落 jsplit=1（得 {big_s_js}）");
    assert_eq!(big, big_s, "o44 同 jsplit=1 两 cfg 须逐位同（100×16=1600 > PART_ROWS）");
}
