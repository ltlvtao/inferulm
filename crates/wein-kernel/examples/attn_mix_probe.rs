//! 混批 attention 对拍探针：同一批「span A(128 token) + span B(64 token)」，
//! 比较三条路径与 host reference 的一致性。
//!
//!   cargo run --release -p wein-kernel --example attn_mix_probe
//!
//! ① 整批 v9（t=192，混 span）      —— 生产 MIX=1 走的路径
//! ② 分两次 v9（各 span 独立调用）  —— 单 span 参照
//! ③ 整批 online（老 tile 路径）    —— 现状 MIX=0
//! ④ host f32 reference
//!
//! 判据：② 应与 ④ 在 v9 容差内（v9 是 Bc 分块，破位级）；
//!       ① 若与 ② 显著不同 ⇒ v9 依赖批形（bug）；① 若与 ② 一致 ⇒ 混批自洽。
use wein_kernel::cuda::{DevBuf, Gpu};
use wein_kernel::ops::OpTable;
use wein_kernel::reference::{bf16_to_f32, f32_to_bf16};

fn bf16_vec(seed_hi: u64, n: usize, scale: f32) -> Vec<u16> {
    (0..n as u64)
        .map(|i| {
            let z = i.wrapping_mul(6364136223846793005).wrapping_add(seed_hi);
            let f = ((z >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * scale;
            f32_to_bf16(f)
        })
        .collect()
}

fn rel2(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut num, mut den, mut mx) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as f64) - (*y as f64);
        num += d * d;
        den += (*y as f64) * (*y as f64);
        mx = mx.max(d.abs());
    }
    (if den > 0.0 { (num / den).sqrt() } else { 0.0 }, mx)
}

/// 参考实现：每行独立（seq/pos 按行），f32 累加，j ≤ pos 因果
#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[u16], k_pool: &[u16], v_pool: &[u16],
    seq_rows: &[(usize, usize, usize)], // (seq, pos, bpr 起始下标)
    bt: &[u32], bpr: usize, bs: usize,
    qh: usize, kvh: usize, hd: usize, q_scale: f32,
) -> Vec<f32> {
    let group = qh / kvh;
    let he = kvh * hd;
    let mut out = vec![0f32; seq_rows.len() * qh * hd];
    for (r, &(seq, pos, _)) in seq_rows.iter().enumerate() {
        let row_bt = &bt[r * bpr..(r + 1) * bpr];
        for h in 0..qh {
            let kv_h = h / group;
            let qv: Vec<f32> = (0..hd).map(|d| bf16_to_f32(q[(r * qh + h) * hd + d])).collect();
            let mut acc = vec![0f32; hd];
            let mut mx = f32::NEG_INFINITY;
            let mut ps: Vec<f32> = Vec::with_capacity(pos + 1);
            let mut kvals: Vec<Vec<f32>> = Vec::with_capacity(pos + 1);
            for j in 0..=pos {
                let (pg, off) = (row_bt[j / bs] as usize, j % bs);
                let base = (pg * bs + off) * he + kv_h * hd;
                let kv: Vec<f32> = (0..hd).map(|d| bf16_to_f32(k_pool[base + d])).collect();
                let s: f32 = qv.iter().zip(kv.iter()).map(|(a, b)| a * b).sum::<f32>() * q_scale
                    + if seq == 0 { 0.0 } else { 0.0 };
                mx = mx.max(s);
                ps.push(s);
                kvals.push(kv);
            }
            let mut den = 0f32;
            for (j, s) in ps.iter().enumerate() {
                let e = (s - mx).exp();
                den += e;
                let (pg, off) = (row_bt[j / bs] as usize, j % bs);
                let base = (pg * bs + off) * he + kv_h * hd;
                for d in 0..hd {
                    acc[d] += e * bf16_to_f32(v_pool[base + d]);
                }
            }
            for d in 0..hd {
                out[(r * qh + h) * hd + d] = acc[d] / den;
            }
        }
    }
    out
}

fn main() {
    let gpu = Gpu::probe().expect("须有 CUDA 设备");
    let profile = wein_platform::HardwareProfile::with_device(wein_platform::Device::Cuda)
        .with_arch("sm_89");
    let t = OpTable::resolve(&gpu, &profile).expect("sm_89 画像须解析成功");
    let stream = gpu.stream();
    let (bs, qh, kvh, hd) = (16usize, 16usize, 4usize, 256usize);
    let he = kvh * hd;
    let q_scale = 1.0 / (hd as f32).sqrt();
    let (n0, n1) = (128usize, 64usize); // span A / span B
    let rows = n0 + n1;
    let bpr = 64usize; // 页表容量（> n0/bs）
    let npages = 256usize;

    // 行表：前 n0 行 = seq0 pos 0..n0-1，后 n1 行 = seq1 pos 0..n1-1
    let mut si = vec![0u32; rows];
    let mut pos = vec![0u32; rows];
    let mut bt = vec![0u32; rows * bpr];
    for i in 0..n0 {
        si[i] = 0;
        pos[i] = i as u32;
    }
    for i in 0..n1 {
        si[n0 + i] = 1;
        pos[n0 + i] = i as u32;
    }
    for r in 0..rows {
        let row_bt = &mut bt[r * bpr..(r + 1) * bpr];
        if r < n0 {
            for (j, e) in row_bt.iter_mut().enumerate().take(bpr) {
                *e = j as u32; // 页 0..bpr
            }
        } else {
            for (j, e) in row_bt.iter_mut().enumerate().take(bpr) {
                *e = 128 + j as u32; // 页 128..
            }
        }
    }
    let q = bf16_vec(0x7a11, rows * qh * hd, 2.0);
    let k_pool = bf16_vec(0x7b33, npages * bs * he, 1.0);
    let v_pool = bf16_vec(0x7c55, npages * bs * he, 1.0);
    let qbytes = qh * hd * 2;
    let btbytes = bpr * 4;

    let dq = gpu.alloc_from_slice(&q);
    let dk = gpu.alloc_from_slice(&k_pool);
    let dv = gpu.alloc_from_slice(&v_pool);
    let dbt = gpu.alloc_from_slice(&bt);
    let dsi = gpu.alloc_from_slice(&si);
    let dpos = gpu.alloc_from_slice(&pos);
    let dout = gpu.alloc_bytes(rows * qbytes);

    // ① 整批 v9
    dout.zero_on(&stream);
    t.paged_attn.run_tile_v9(
        &dq, &dk, &dv, &dbt, &dsi, &dpos, &dout, q_scale,
        bpr as u32, bs as u32, qh as u32, kvh as u32, hd as u32, rows as u32, &stream,
    );
    stream.sync();
    let out_all: Vec<u16> = dout.read_to_vec::<u16>();

    // ② 分两次 v9（单 span 各一次）
    dout.zero_on(&stream);
    for (start, cnt) in [(0usize, n0), (n0, n1)] {
        let qv = dq.narrow(start * qbytes, cnt * qbytes);
        let btv = dbt.narrow(start * btbytes, cnt * btbytes);
        let siv = dsi.narrow(start * 4, cnt * 4);
        let posv = dpos.narrow(start * 4, cnt * 4);
        let outv = dout.narrow(start * qbytes, cnt * qbytes);
        t.paged_attn.run_tile_v9(
            &qv, &dk, &dv, &btv, &siv, &posv, &outv, q_scale,
            bpr as u32, bs as u32, qh as u32, kvh as u32, hd as u32, cnt as u32, &stream,
        );
    }
    stream.sync();
    let out_split: Vec<u16> = dout.read_to_vec::<u16>();

    // ③ 整批 online（老路径）
    dout.zero_on(&stream);
    let cfg = wein_kernel::ops::AttnCfg::baseline();
    t.paged_attn.run_tile_online_cfg(
        cfg, &dq, &dk, &dv, &dbt, &dsi, &dpos, &dout, q_scale,
        bpr as u32, bs as u32, qh as u32, kvh as u32, hd as u32, rows as u32, &stream,
    );
    stream.sync();
    let out_online: Vec<u16> = dout.read_to_vec::<u16>();

    // ④ reference
    let seq_rows: Vec<(usize, usize, usize)> =
        (0..rows).map(|r| (si[r] as usize, pos[r] as usize, r)).collect();
    let rf = reference(&q, &k_pool, &v_pool, &seq_rows, &bt, bpr, bs, qh, kvh, hd, q_scale);

    let to_f32 = |v: &[u16]| -> Vec<f32> { v.iter().map(|&x| bf16_to_f32(x)).collect() };
    let a_all = to_f32(&out_all);
    let a_split = to_f32(&out_split);
    let a_on = to_f32(&out_online);

    let (r_all, p_all) = rel2(&a_all, &rf);
    let (r_split, p_split) = rel2(&a_split, &rf);
    let (r_on, p_on) = rel2(&a_on, &rf);
    let (r_as, p_as) = rel2(&a_all, &a_split);
    let (r_ao, p_ao) = rel2(&a_all, &a_on);
    println!("行数 {rows}（spanA {n0} + spanB {n1}），bpr={bpr} bs={bs}");
    println!("① 整批 v9   vs reference : rel2={r_all:.3e} 峰值={p_all:.3e}");
    println!("② 分两次 v9 vs reference : rel2={r_split:.3e} 峰值={p_split:.3e}");
    println!("③ 整批 online vs reference: rel2={r_on:.3e} 峰值={p_on:.3e}");
    println!("① vs ②（混批自洽性）     : rel2={r_as:.3e} 峰值={p_as:.3e}");
    println!("① vs ③（与现状路径）     : rel2={r_ao:.3e} 峰值={p_ao:.3e}");
}
