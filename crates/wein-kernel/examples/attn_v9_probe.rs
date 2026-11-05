//! v9 档（手写 PTX：LDSM + mma.sync + C→A 直用 + 99KB 动态 smem）探针。
//!
//!   cargo run --release -p wein-kernel --example attn_v9_probe [-- 6009]
//!
//! ① 正确性 spot：t=512 / t=1152 / t=2048 v9 vs v5（容差档：Bc 16→32 破
//!    位级，判据 rel2 ≤ 3e-3 + 峰值归一 + 双跑 diff 查竞态）。
//! ② 单发计时 @T=3009/6009/8009/32000：v5 vs v9 + v9_meta。
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
}

/// dump 模式（配 WEIN_V9_DBG=1 构建）：读 CTA0/tile0 的 S 片段转储，
/// 重组 lane→(row,j) 后与 host 参考对拍，定位 fragment 布局错位。
/// 布局合同：lane l、warp w ⇒ d0/d1 行 w*16+l/4、列 ni*8+2*(l%4)+{0,1}；
/// d2/d3 行 +8。转储 22 词 @ tid*128B；S 为掩码/q_scale 前的裸点积。
fn dump_mode() {
    use wein_kernel::reference::bf16_to_f32;
    let gpu = Gpu::probe().expect("须有 CUDA 设备");
    let profile = wein_platform::HardwareProfile::with_device(wein_platform::Device::Cuda)
        .with_arch("sm_89");
    let t = OpTable::resolve(&gpu, &profile).expect("sm_89 画像须解析成功");
    let stream = gpu.stream();
    let (bs, q_heads, kv_heads, hd) = (16usize, 16usize, 4usize, 256usize);
    let q_scale = 1.0 / (hd as f32).sqrt();
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
    };
    let poison = vec![0xdeadbeefu32; tokens * q_heads * hd / 2];
    d.out.upload_on(&poison, &stream);
    stream.sync();
    {
        let chk = d.out.read_to_vec::<u32>();
        let ok = chk.iter().take(64).all(|&w| w == 0xdeadbeef);
        let ok_all = chk.iter().all(|&w| w == 0xdeadbeef);
        println!(
            "## 毒回读：前 64 词 ok={ok} 全量 ok={ok_all} 首词 {:08x}",
            chk[0]
        );
    }
    let t64 = tokens as u32;
    t.paged_attn.run_tile_v9(&d.q, &d.k, &d.v, &d.bt, &d.si, &d.pos, &d.out, q_scale, c.bpr as u32, bs as u32, q_heads as u32, kv_heads as u32, hd as u32, t64, &stream);
    stream.sync();
    let raw = d.out.read_to_vec::<u32>();
    let get = |tid: usize, slot: usize| raw[tid * 32 + slot];

    // host 参考：S[row][j] = Σ_d Q[row][d]·K[j][d]（qh=0/kvh=0/CTA0）
    let sref = |row: usize, j: usize| -> f32 {
        let mut acc = 0f32;
        for dd in 0..hd {
            let q = bf16_to_f32(c.q[row * q_heads * hd + dd]) as f32;
            let k = bf16_to_f32(c.k[j * kv_heads * hd + dd]) as f32;
            acc = (q * k + acc) as f32;
        }
        acc
    };

    println!("## CTA0 tile0 元数据（每 warp 首个 lane）：");
    for w in 0..4 {
        let tid = w * 32;
        let meta: Vec<u32> = (16..22).map(|s| get(tid, s)).collect();
        println!(
            "  warp{w}: pos_g={} pos_g8={} max_pos={} n_tile={} tile={} j0={}",
            meta[0], meta[1], meta[2], meta[3], meta[4], meta[5]
        );
    }
    let mut worst_rel = 0f64;
    let mut worst = String::new();
    let mut bad = 0usize;
    let mut total = 0usize;
    for w in 0..4usize {
        for l in 0..32usize {
            let tid = w * 32 + l;
            let g = l / 4;
            let c2 = 2 * (l % 4);
            for ni in 0..4usize {
                for (word, (dr, dc)) in [(0usize, (0usize, 0usize)), (1, (0, 1)), (2, (8, 0)), (3, (8, 1))] {
                    let s = f32::from_bits(get(tid, ni * 4 + word));
                    let row = w * 16 + g + dr;
                    let j = ni * 8 + c2 + dc;
                    let want = sref(row, j);
                    let rel = ((s - want) as f64 / want.abs().max(1e-6) as f64).abs();
                    total += 1;
                    if rel > 2e-2 {
                        bad += 1;
                        if rel > worst_rel {
                            worst_rel = rel;
                            worst = format!("S[{row}][{j}] got {s:.5} want {want:.5} (w{w} l{l} ni{ni} d{word})");
                        }
                    }
                }
            }
        }
    }
    println!("## S 对拍：{}/{} 超差（rel>2e-2），最坏 rel={worst_rel:.3e}", bad, total);
    if !worst.is_empty() {
        println!("   最坏样例：{worst}");
    }
    // 坏样本分布模式：按 (ni,word) 与 (warp,g) 直方图 + 全部样例（前 40）
    let mut by_nw = std::collections::BTreeMap::new();
    let mut by_wg = std::collections::BTreeMap::new();
    let mut samples = Vec::new();
    for w in 0..4usize {
        for l in 0..32usize {
            let tid = w * 32 + l;
            let g = l / 4;
            let c2 = 2 * (l % 4);
            for ni in 0..4usize {
                for (word, (dr, dc)) in [(0usize, (0usize, 0usize)), (1, (0, 1)), (2, (8, 0)), (3, (8, 1))] {
                    let s = f32::from_bits(get(tid, ni * 4 + word));
                    let row = w * 16 + g + dr;
                    let j = ni * 8 + c2 + dc;
                    let want = sref(row, j);
                    let rel = ((s - want) as f64 / want.abs().max(1e-6) as f64).abs();
                    if rel > 2e-2 {
                        *by_nw.entry((ni, word)).or_insert(0usize) += 1;
                        *by_wg.entry((w, g)).or_insert(0usize) += 1;
                        if samples.len() < 40 {
                            samples.push(format!("S[{row}][{j}]={s:.4} want {want:.4} (w{w} l{l} ni{ni} d{word})"));
                        }
                    }
                }
            }
        }
    }
    println!("   按 (ni,word)：{by_nw:?}");
    println!("   按 (warp,g)：{by_wg:?}");
    // 暴力反查：错值在 64×32 窗口穷举找真实来源 (row',j')
    let mut lut = std::collections::BTreeMap::new();
    for r in 0..64usize {
        for j in 0..32usize {
            let v = sref(r, j);
            lut.insert(v.to_bits(), (r, j));
        }
    }
    let mut mapped = 0usize;
    let mut map_fail = 0usize;
    for w in 0..4usize {
        for l in 0..32usize {
            let tid = w * 32 + l;
            let g = l / 4;
            let c2 = 2 * (l % 4);
            for ni in 0..4usize {
                for (word, (dr, dc)) in [(0usize, (0usize, 0usize)), (1, (0, 1)), (2, (8, 0)), (3, (8, 1))] {
                    let s = f32::from_bits(get(tid, ni * 4 + word));
                    let row = w * 16 + g + dr;
                    let j = ni * 8 + c2 + dc;
                    let want = sref(row, j);
                    let rel = ((s - want) as f64 / want.abs().max(1e-6) as f64).abs();
                    if rel > 2e-2 {
                        // bf16 舍入容差穷举：|s - cand| < 1e-3
                        let hit = lut
                            .range(s.to_bits().saturating_sub(4)..=s.to_bits().saturating_add(4))
                            .map(|(_, &p)| p)
                            .next();
                        match hit {
                            Some((r2, j2)) => {
                                mapped += 1;
                                if mapped <= 12 {
                                    println!("   错位映射：以为 S[{row}][{j}] ← 实际来自 S[{r2}][{j2}]（w{w} l{l} ni{ni} d{word}）");
                                }
                            }
                            None => map_fail += 1,
                        }
                    }
                }
            }
        }
    }
    println!("   反查：{mapped} 个找到来源映射，{map_fail} 个找不到");
    // 原始字检查：lane0/lane4 的 22 词 + poison 是否残留
    print!("   raw lane0: ");
    for s in 0..22 {
        print!("{:08x} ", raw[s]);
    }
    println!();
    print!("   raw lane4: ");
    for s in 0..22 {
        print!("{:08x} ", raw[4 * 32 + s]);
    }
    println!();
    let poison_cnt = raw[..4096].iter().filter(|&&w| w == 0xdeadbeef).count();
    println!("   前 16KB poison 残留：{poison_cnt} 词");
    // tid 区域分类：N=全 0x7fff7fff / Z=槽16-17 为 0（真 store 落盘）/ P=poison / ?=其他
    let mut classes = Vec::new();
    for tid in 0..128usize {
        let w: Vec<u32> = (0..22).map(|s| raw[tid * 32 + s]).collect();
        let c = if w.iter().all(|&x| x == 0x7fff7fff) {
            'N'
        } else if w[16] == 0 && w[17] == 0 {
            'Z'
        } else if w.iter().all(|&x| x == 0xdeadbeef) {
            'P'
        } else {
            '?'
        };
        classes.push(c);
    }
    for w in 0..4 {
        let row: String = classes[w * 32..w * 32 + 32].iter().collect();
        println!("   warp{w}: {row}");
    }
    let nq = raw[..4096].iter().filter(|&&x| x != 0x7fff7fff && x != 0xdeadbeef).count();
    println!("   非 N 非 P 词数：{nq}");
    for s in &samples {
        println!("   {s}");
    }
    // 全 128 lane 元数据（查 warp1/3 垃圾范围）
    print!("   元数据 pos_g：");
    for w in 0..4 {
        let v: Vec<String> = (0..32).map(|l| format!("{}", get(w * 32 + l, 16))).collect();
        print!("w{w}[{}] ", v.join(","));
    }
    println!();
    println!("{}", if bad == 0 { "✅ S 相位布局正确" } else { "❌ 布局有错位" });
}

/// nanmap 模式（正常构建，无 WEIN_V9_DBG）：跑完整 v9，按 (row, qh) 画 NaN 分布，
/// 定位哪些 CTA/行算出 NaN（转储实验已证非 CTA0 尾写 = 0x7fff7fff NaN 对）。
fn nanmap_mode() {
    use wein_kernel::reference::bf16_to_f32;
    let gpu = Gpu::probe().expect("须有 CUDA 设备");
    let profile = wein_platform::HardwareProfile::with_device(wein_platform::Device::Cuda)
        .with_arch("sm_89");
    let t = OpTable::resolve(&gpu, &profile).expect("sm_89 画像须解析成功");
    let stream = gpu.stream();
    let (bs, q_heads, kv_heads, hd) = (16usize, 16usize, 4usize, 256usize);
    let q_scale = 1.0 / (hd as f32).sqrt();
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
    };
    let t64 = tokens as u32;
    let poison: Vec<u16> = vec![0x7c7cu16; tokens * q_heads * hd];
    d.out.upload_on(&poison, &stream);
    stream.sync();
    t.paged_attn.run_tile_v9(&d.q, &d.k, &d.v, &d.bt, &d.si, &d.pos, &d.out, q_scale, c.bpr as u32, bs as u32, q_heads as u32, kv_heads as u32, hd as u32, t64, &stream);
    stream.sync();
    let raw = d.out.read_to_vec::<u16>();
    // 未写（毒 0x7c7c）vs 真写：分类打印
    let unwritten = raw.iter().filter(|&&w| w == 0x7c7c).count();
    println!("## v9 未写（毒残留）：{unwritten}/{}", raw.len());
    // out 布局 [row][qh][hd]：NaN 图按 (row, qh) → 256 值里 NaN 个数
    let mut grid = vec![0u16; tokens * q_heads];
    let mut total_nan = 0usize;
    for (i, &w) in raw.iter().enumerate() {
        let f = bf16_to_f32(w);
        if f.is_nan() {
            grid[i / hd] += 1;
            total_nan += 1;
        }
    }
    println!("## v9 全量 NaN 词：{total_nan}/{}", raw.len());
    // dim 级：对 (row, qh)=(0,0)/(2,0)/(4,0) 打 NaN 的 dim 下标集合
    for probe_row in [0usize, 1, 2, 3, 4, 5] {
        let mut dims = Vec::new();
        for dd in 0..hd {
            if bf16_to_f32(raw[probe_row * q_heads * hd + dd]).is_nan() {
                dims.push(dd);
            }
        }
        if !dims.is_empty() {
            let blocks: Vec<usize> = {
                let mut b = Vec::new();
                for (i, d) in dims.iter().enumerate() {
                    let blk = d / 8;
                    if i == 0 || dims[i - 1] / 8 != blk {
                        b.push(blk);
                    }
                }
                b
            };
            println!(
                "   row{probe_row} qh0：NaN {} 个，dim 块(÷8) {blocks:?} 前 12 dims {dims:?}",
                dims.len()
            );
            let _ = dims.clone();
        }
    }
    // 行块图：每 64 行一组（=一个 row0 CTA × 16 qh），打印 64×16 网格缩略（NaN 全/半/无）
    for row0 in (0..tokens).step_by(64) {
        println!("   row0={row0}（ctaid {}..{}）：", row0 / 64 * 16, row0 / 64 * 16 + 15);
        for hh in 0..q_heads {
            let pat: String = (0..64)
                .map(|r| {
                    let n = grid[(row0 + r) * q_heads + hh];
                    if n == 0 {
                        '.'
                    } else if n == hd as u16 {
                        'N'
                    } else {
                        '?'
                    }
                })
                .collect();
            println!("     qh{hh:2}: {pat}");
        }
    }
}

fn rel2(a: &[u16], b: &[u16]) -> (f64, f64) {
    let (mut num, mut den, mut pk) = (0f64, 0f64, 0f64);
    let mut bmax = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let (x, y) = (
            wein_kernel::reference::bf16_to_f32(*x) as f64,
            wein_kernel::reference::bf16_to_f32(*y) as f64,
        );
        num += (x - y) * (x - y);
        den += y * y;
        pk = pk.max((x - y).abs());
        bmax = bmax.max(y.abs());
    }
    ((num / den).sqrt(), pk / bmax)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "dump") {
        return dump_mode();
    }
    if args.iter().any(|a| a == "nanmap") {
        return nanmap_mode();
    }
    let only_t: Option<usize> = args.get(1).and_then(|s| s.parse().ok());
    let gpu = Gpu::probe().expect("须有 CUDA 设备");
    let profile = wein_platform::HardwareProfile::with_device(wein_platform::Device::Cuda)
        .with_arch("sm_89");
    let t = OpTable::resolve(&gpu, &profile).expect("sm_89 画像须解析成功");
    let stream = gpu.stream();
    let (bs, q_heads, kv_heads, hd) = (16usize, 16usize, 4usize, 256usize);
    let q_scale = 1.0 / (hd as f32).sqrt();

    let (r9, l9, s9) = t.paged_attn.v9_meta();
    println!("## v9 编译属性：{r9} regs/{l9}B local/{s9}B 静态 smem（动态 99KB 另计）");

    // ── ① 正确性 spot：v9 vs v5 容差 + 双跑查竞态 ──
    for tokens in [512usize, 1152, 2048] {
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
        };
        let poison = vec![0x7f7fu16; tokens * q_heads * hd];
        let run = |v9: bool| {
            d.out.upload_on(&poison, &stream);
            let t64 = (tokens & !63) as u32;
            if v9 {
                t.paged_attn.run_tile_v9(&d.q, &d.k, &d.v, &d.bt, &d.si, &d.pos, &d.out, q_scale, c.bpr as u32, bs as u32, q_heads as u32, kv_heads as u32, hd as u32, t64, &stream);
            } else {
                t.paged_attn.run_tile_v5(&d.q, &d.k, &d.v, &d.bt, &d.si, &d.pos, &d.out, q_scale, c.bpr as u32, bs as u32, q_heads as u32, kv_heads as u32, hd as u32, t64, &stream);
            }
            stream.sync();
            d.out.read_to_vec::<u16>()
        };
        let v5 = run(false);
        let v9 = run(true);
        let v9b = run(true);
        let (r, pk) = rel2(&v9, &v5);
        let race = v9.iter().zip(v9b.iter()).filter(|(a, b)| a != b).count();
        let flag = if r < 3e-3 && race == 0 { "✅" } else { "❌" };
        println!("## t={tokens}：v9 vs v5 rel2={r:.3e} 峰值={pk:.3e} 双跑 diff={race} {flag}");
        drop(d);
    }

    // ── ② 计时 @3009/6009/8009/32000：v5 vs v9（各取对齐主段）──
    println!("\n## 单发计时（10 发均摊，ms/单层）");
    println!("| T | v5 | v9 | v5/v9 | v9 8 层 |");
    println!("|---|---|---|---|---|");
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
        let t64 = (tt & !63) as u32;
        let v5_ms = timeit(&|| {
            t.paged_attn.run_tile_v5(&d.q, &d.k, &d.v, &d.bt, &d.si, &d.pos, &d.out, q_scale, c.bpr as u32, bs as u32, q_heads as u32, kv_heads as u32, hd as u32, t64, &stream)
        });
        let v9_ms = timeit(&|| {
            t.paged_attn.run_tile_v9(&d.q, &d.k, &d.v, &d.bt, &d.si, &d.pos, &d.out, q_scale, c.bpr as u32, bs as u32, q_heads as u32, kv_heads as u32, hd as u32, t64, &stream)
        });
        println!(
            "| {tt} | {v5_ms:.3} | {v9_ms:.3} | {:.2}x | {:.1} |",
            v5_ms / v9_ms,
            v9_ms * 8.0
        );
        drop(d);
    }
}
