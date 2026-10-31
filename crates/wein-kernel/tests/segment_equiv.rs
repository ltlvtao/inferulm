//! 段等价性质测试（I1 竖切状态接力）：任意切段 + 状态接力 ≡ 整段一次算。
//!
//! 独立真值 = 同输入整段单算，不与 roll/tap/递推的状态存值约定共享路径。
//! 教训两条（vs vLLM 对拍与解码递推偏差定位战）：
//! - GQA 头映射 qh%kv_heads：测试形状 (2,1) 两映射恒等，钉不住；
//! - conv 状态存值：exec 曾把 conv+silu 后的激活值存进状态，而段内 tap
//!   用原始 qkv——预填单段永不触状态故全绿，解码一触即炸。
//! 两者都属于"段内自洽、跨段露馅"，只有段等价性质能分对错。
//!
//! - o15 · conv1d_dw + conv_state_roll 段等价（含 count=1 < st 解码形态）
//! - o16 · deltanet 段等价（lin_state 接力对任意切段不可见）
#![allow(dead_code)]

use wein_kernel::cuda::Gpu;
use wein_kernel::ops::{Conv1dDwOp, ConvStateRollOp, DeltanetOp, OpTable};
use wein_kernel::reference;
use wein_platform::{Device, HardwareProfile};

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

fn assert_bits(label: &str, got: &[u16], expect: &[u16]) {
    assert_eq!(got.len(), expect.len(), "{label} 长度不符");
    let bad: Vec<(usize, u16, u16)> = got
        .iter()
        .zip(expect)
        .enumerate()
        .filter(|(_, (g, e))| g != e)
        .map(|(i, (g, e))| (i, *g, *e))
        .take(4)
        .collect();
    assert!(bad.is_empty(), "{label} 位不对齐，首批差异（idx, got, expect）：{bad:?}");
}

fn vec_bf16(seed_hi: u64, n: usize, scale: f32) -> Vec<u16> {
    (0..n as u64).map(|i| reference::f32_to_bf16(reference::splitmix_f32(i ^ seed_hi) * scale)).collect()
}

/// 三批切段布局（两请求 × 7 token）：预填批 → 解码批（row0 单 token，
/// count=1 < conv st——解码形态）→ 尾段批。元素 = (row, seg_start, seg_count)
const SEG_BATCHES: [&[(usize, usize, usize)]; 3] =
    [&[(0, 0, 3), (1, 0, 2)], &[(0, 3, 1), (1, 2, 2)], &[(0, 4, 3), (1, 4, 3)]];

/// o15 · conv 段等价：conv_state 接力后任意切段的 conv 输出 ≡ 整段单算。
/// 参考侧（性质本体）与 GPU 侧（kernel 位对齐）同跑；批尾 GPU 状态与
/// 参考状态对拍（接力落点也钉住）
#[test]
fn o15_conv_segment_equivalence() {
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let (cd, ks, st) = (64usize, 4usize, 3usize);
    let (rows, ntok) = (2usize, 7usize);
    let w: Vec<u16> = vec_bf16(0x2545_f491, cd * ks, 1.0);
    let x_full: Vec<Vec<u16>> =
        (0..rows).map(|r| vec_bf16(0x9e37_79b9 + r as u64 * 77, ntok * cd, 2.0)).collect();

    // 整段真值（零状态，逐请求一次算）
    let zero_state = vec![0u16; rows * cd * st];
    let truth: Vec<Vec<u16>> = (0..rows)
        .map(|r| {
            let (tr, tp): (Vec<u32>, Vec<u32>) =
                ((0..ntok).map(|_| r as u32).collect(), (0..ntok as u32).collect());
            reference::conv1d_dw(&x_full[r], &w, &zero_state, &tr, &tp, cd, ks)
        })
        .collect();

    // 分段 + 状态接力
    let mut state_ref = zero_state.clone();
    let dstate = gpu.alloc_from_slice(&zero_state);
    let dw = gpu.alloc_from_slice(&w);
    for (bi, segs_b) in SEG_BATCHES.iter().enumerate() {
        let (mut xb, mut tok_row, mut tok_pos): (Vec<u16>, Vec<u32>, Vec<u32>) = (vec![], vec![], vec![]);
        let (mut rs, mut rc, mut rid): (Vec<u32>, Vec<u32>, Vec<u32>) = (vec![], vec![], vec![]);
        let mut expb: Vec<u16> = vec![];
        for &(r, s, c) in segs_b.iter() {
            rs.push((xb.len() / cd) as u32); // 本段在批内的 token 轴起点
            xb.extend_from_slice(&x_full[r][s * cd..(s + c) * cd]);
            expb.extend_from_slice(&truth[r][s * cd..(s + c) * cd]);
            rc.push(c as u32);
            rid.push(r as u32);
            for i in 0..c {
                tok_row.push(r as u32);
                tok_pos.push(i as u32); // 段内局部位
            }
        }
        let out_ref = reference::conv1d_dw(&xb, &w, &state_ref, &tok_row, &tok_pos, cd, ks);
        assert_bits(&format!("o15 批{bi}：参考分段 vs 整段真值"), &out_ref, &expb);
        state_ref = reference::conv_state_roll(&xb, &state_ref, &rs, &rc, &rid, cd, ks, &vec![0u32; rows]);

        let (dx, dtr, dtp) = (
            gpu.alloc_from_slice(&xb),
            gpu.alloc_from_slice(&tok_row),
            gpu.alloc_from_slice(&tok_pos),
        );
        let dout = gpu.alloc_bytes(xb.len() * 2);
        t.conv1d_dw.run(
            &dx, &dw, &dstate, &dout, &dtr, &dtp,
            cd as u32, ks as u32, (xb.len() / cd) as u32, &stream,
        );
        let (drs, drc, drid) = (
            gpu.alloc_from_slice(&rs),
            gpu.alloc_from_slice(&rc),
            gpu.alloc_from_slice(&rid),
        );
        let dspec0 = gpu.alloc_from_slice(&vec![0u32; rows]); // 非投机路径：全 0 常量数组
        t.conv_state_roll.run(&dx, &dstate, &drs, &drc, &drid, cd as u32, ks as u32, rows as u32, &dspec0, &stream);
        stream.sync();
        assert_bits(&format!("o15 批{bi}：GPU 分段 vs 整段真值"), &dout.read_to_vec(), &expb);
    }
    assert_bits("o15 末态：GPU 状态 vs 参考状态", &dstate.read_to_vec(), &state_ref);
}

/// o16 · deltanet 段等价：lin_state 接力后任意切段的递推输出 ≡ 整段单算。
/// 与 o15 同批布局；GPU core 输出与末态 bf16 状态分别位对齐（状态池
/// bf16 化后段等价**天然保持**——段间接力传递的就是落池的 bf16 状态）
#[test]
fn o16_deltanet_segment_equivalence() {
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let (kh_n, vh_n, kd, vd) = (2usize, 4usize, 16usize, 16usize);
    let (keyd, vald, adim) = (kh_n * kd, vh_n * vd, vh_n);
    let (rows, ntok) = (2usize, 7usize);
    let seed = |hi: u64, n: usize| vec_bf16(hi, n, 2.0);
    let (q_full, k_full): (Vec<Vec<u16>>, Vec<Vec<u16>>) = (
        (0..rows).map(|r| seed(0x9e37_79b9 + r as u64 * 31, ntok * keyd)).collect(),
        (0..rows).map(|r| seed(0xbf58_476d + r as u64 * 31, ntok * keyd)).collect(),
    );
    let v_full: Vec<Vec<u16>> =
        (0..rows).map(|r| seed(0x94d0_49bb + r as u64 * 31, ntok * vald)).collect();
    let (a_full, b_full): (Vec<Vec<u16>>, Vec<Vec<u16>>) = (
        (0..rows).map(|r| seed(0x2545_f491 + r as u64 * 31, ntok * adim)).collect(),
        (0..rows).map(|r| seed(0x1142_5bb3 + r as u64 * 31, ntok * adim)).collect(),
    );
    let a_log: Vec<f32> = (0..vh_n).map(|h| 0.3 + 0.2 * h as f32).collect();
    let dtb: Vec<u16> = vec_bf16(0x27d4_eb2f, adim, 1.0);
    let eps = 1e-6f32;

    // 整段真值（零状态，逐请求一次算）
    let mut truth: Vec<Vec<u16>> = vec![];
    for r in 0..rows {
        let mut s0 = vec![0u16; vh_n * vd * kd];
        let out = reference::deltanet(
            &q_full[r], &k_full[r], &v_full[r], &a_full[r], &b_full[r], &a_log, &dtb,
            &mut s0, &[0], &[ntok as u32], &[0], kh_n, vh_n, kd, vd, eps, &[0],
        );
        truth.push(out);
    }

    // 分段 + 状态接力
    let mut state_ref = vec![0u16; rows * vh_n * vd * kd];
    let dstate = gpu.alloc_from_slice(&state_ref);
    let (da_log, ddtb) = (gpu.alloc_from_slice(&a_log), gpu.alloc_from_slice(&dtb));
    for (bi, segs_b) in SEG_BATCHES.iter().enumerate() {
        let (mut qb, mut kb, mut vb, mut ab, mut bb): (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) =
            (vec![], vec![], vec![], vec![], vec![]);
        let (mut rs, mut rc, mut rid): (Vec<u32>, Vec<u32>, Vec<u32>) = (vec![], vec![], vec![]);
        let mut expb: Vec<u16> = vec![];
        for &(r, s, c) in segs_b.iter() {
            rs.push((qb.len() / keyd) as u32);
            qb.extend_from_slice(&q_full[r][s * keyd..(s + c) * keyd]);
            kb.extend_from_slice(&k_full[r][s * keyd..(s + c) * keyd]);
            vb.extend_from_slice(&v_full[r][s * vald..(s + c) * vald]);
            ab.extend_from_slice(&a_full[r][s * adim..(s + c) * adim]);
            bb.extend_from_slice(&b_full[r][s * adim..(s + c) * adim]);
            expb.extend_from_slice(&truth[r][s * vald..(s + c) * vald]);
            rc.push(c as u32);
            rid.push(r as u32);
        }
        let out_ref = reference::deltanet(
            &qb, &kb, &vb, &ab, &bb, &a_log, &dtb, &mut state_ref, &rs, &rc, &rid,
            kh_n, vh_n, kd, vd, eps, &vec![0u32; rows],
        );
        // 参考输出按批内全局 token 轴铺开——重排回 (row, 段内序) 与真值对拍
        let mut gotb: Vec<u16> = vec![];
        for &(r, s, c) in segs_b.iter() {
            let base = rs[segs_b.iter().position(|&(rr, _, _)| rr == r).unwrap()] as usize;
            gotb.extend_from_slice(&out_ref[(base * vald)..(base + c) * vald]);
        }
        assert_bits(&format!("o16 批{bi}：参考分段 vs 整段真值"), &gotb, &expb);

        let (dq, dk_, dv, da, db) = (
            gpu.alloc_from_slice(&qb),
            gpu.alloc_from_slice(&kb),
            gpu.alloc_from_slice(&vb),
            gpu.alloc_from_slice(&ab),
            gpu.alloc_from_slice(&bb),
        );
        let (drs, drc, drid) = (
            gpu.alloc_from_slice(&rs),
            gpu.alloc_from_slice(&rc),
            gpu.alloc_from_slice(&rid),
        );
        let tokens = qb.len() / keyd;
        let dout = gpu.alloc_bytes(tokens * vald * 2);
        let dspec0 = gpu.alloc_from_slice(&vec![0u32; rows]); // 非投机路径：全 0 常量数组
        t.deltanet.run(
            &dq, &dk_, &dv, &da, &db, &da_log, &ddtb, &dstate, &dout, &drs, &drc, &drid,
            rows as u32, kh_n as u32, vh_n as u32, kd as u32, vd as u32, eps, &dspec0, 1, &stream,
        );
        stream.sync();
        // GPU 输出同为批内全局 token 轴——同法重排
        let raw = dout.read_to_vec();
        let mut gpu_b: Vec<u16> = vec![];
        for &(r, s, c) in segs_b.iter() {
            let base = rs[segs_b.iter().position(|&(rr, _, _)| rr == r).unwrap()] as usize;
            gpu_b.extend_from_slice(&raw[(base * vald)..(base + c) * vald]);
        }
        assert_bits(&format!("o16 批{bi}：GPU 分段 vs 整段真值"), &gpu_b, &expb);
    }
    let got_state: Vec<u16> = dstate.read_to_vec();
    let bad = got_state.iter().zip(state_ref.iter()).filter(|(a, b)| a != b).count();
    assert_eq!(bad, 0, "o16 末态：GPU bf16 状态与参考不一致（{bad} 个元素）");
}
