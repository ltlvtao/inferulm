//! 块档（Br 参数化、O/P 落块私有 scratch）的**几何与流量账**（纯函数，无 GPU）。
//!
//! 每行一档：块数 / 线程 / 行 / npass / 每块 scratch / 实付 scratch，
//! 以及"一次 prefill 调用"的 KV 读量与 O 往返量（解析式，与 kernel 同式）。
//!
//! 流量口径（一次调用，tokens=T，前缀长 P）：
//! · KV 读 = 块数 × ngrp × `j 片长` × hd × 2(K,V) × 2B —— j 片长 = P/jsplit
//!   （j 分片不改总量：每片只扫自己那段 j）
//! · P 往返 = 4 遍（B1 写、B1b 读、B1c 读+写、B2 读）× rows × P × 4B
//!   = 5 × rows × P × 4B
//! · O 往返 = (P/jchunk) 片 × rows × hd × 8B（读+写）
//!
//! 用法：`attn_blk_geom [tokens] [prefix] [q_heads] [kv_heads] [head_dim]`
fn main() {
    use wein_kernel::ops::{BlkCfg, KvDtype, PagedAttnCuda};
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let tokens = a.first().copied().unwrap_or(256) as u32;
    let prefix = a.get(1).copied().unwrap_or(6009) as u64;
    let qh = a.get(2).copied().unwrap_or(16) as u32;
    let kvh = a.get(3).copied().unwrap_or(4) as u32;
    let hd = a.get(4).copied().unwrap_or(256) as u32;
    println!(
        "tokens={tokens} prefix={prefix} qh={qh} kvh={kvh} hd={hd} \
         （KV 读/往返按一次调用、前缀全长计；scratch 池上限 {}MB）",
        wein_kernel::ops::BLK_SCRATCH_FLOATS * 4 / (1 << 20)
    );
    // 现状对照：单遍档 Br=4 / rpt=4 / nd=4（寄存器累加器）
    let g0 = PagedAttnCuda::online_geom(
        wein_kernel::ops::AttnCfg { rpt: 4, ..wein_kernel::ops::AttnCfg::baseline() },
        qh,
        kvh,
        hd,
        tokens,
    );
    // KV = 块数 × ngrp × (前缀/js) × hd × 2(K,V) × 2B
    let kv_mb = |nblk: u64, ngrp: u64, js: u64| -> f64 {
        (nblk * ngrp * (prefix / js) * hd as u64 * 4) as f64 / 1e6
    };
    let kv0 = kv_mb(
        (kvh as u64) * g0.ntile as u64 * g0.jsplit as u64,
        // 单遍档每线程组 = rows/rpt 行，ngrp = rows_nom/rpt = bdim/nv
        (g0.rows_nom / g0.rpt) as u64,
        g0.jsplit as u64,
    );
    println!(
        "\n## 现状（单遍档，寄存器累加器）rpt=4 nd=4：ntok(Br)={} rows={} bdim={} ngrp={} \
         nblk={} jsplit={} — KV 读 {:.1} MB/调用",
        g0.ntok,
        g0.rows_nom,
        g0.bdim,
        g0.rows_nom / 4,
        (kvh as usize) * g0.ntile * g0.jsplit,
        g0.jsplit,
        kv0
    );
    println!(
        "\n| Br | jchunk | rows | bdim | ngrp | npass | 块数 | js | 每块 MB | 实付 MB | KV 读 MB | P 往返 MB | O 往返 MB |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for &br in &[4usize, 16, 32, 64, 128] {
        for &cap in &[256usize, 512, 1024, 64] {
            let cfg = BlkCfg { br, jchunk: cap, bdim: 256, jsplit: 0, kv: KvDtype::Bf16 };
            let g = PagedAttnCuda::blk_geom(cfg, qh, kvh, hd, tokens);
            let nblk = g.nblk(kvh) as u64;
            let kv = kv_mb(nblk, g.ngrp as u64, g.jsplit as u64);
            let p_mb = 5.0 * (g.rows_nom as f64) * prefix as f64 * 4.0 / 1e6;
            let o_mb =
                (prefix as f64 / cap as f64) * (g.rows_nom * hd as usize) as f64 * 8.0 / 1e6;
            println!(
                "| {br} | {cap} | {} | {} | {} | {} | {} | {} | {:.3} | {:.1} | {:.1} | {:.1} | {:.1} |",
                g.rows_nom,
                g.bdim,
                g.ngrp,
                g.npass,
                nblk,
                g.jsplit,
                g.stride as f64 * 4.0 / 1e6,
                g.scratch_bytes(kvh) as f64 / 1e6,
                kv,
                p_mb,
                o_mb
            );
        }
    }
}
