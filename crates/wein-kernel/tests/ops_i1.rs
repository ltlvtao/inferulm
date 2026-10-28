//! I1 竖切算子验收（Qwen3.5 文本骨干）——8 新入口逐个位对齐 + gemm_wt 容差。
//!
//! 数值纪律与 ops_m0 同守（o4/k6 同源）：只正确舍入原语、行内顺序累加、
//! bf16→f32 精确位左移；exp/log 走自控多项式（两侧同字面量同 Horner 序）。
//! deltanet 的门控链首次把对数域（softplus）拉进位对齐射程——log2f 是
//! kernels/reference 双侧同式的自控实现（禁 MUFU lg2 近似）。
//!
//! - o15 · 注册表长全：resolve 一次 → 18 槽位齐（+8 新），名字可审
//! - o16 · rms_norm_zc + rms_norm_gated：零中心 norm（(1+w) 形态）与
//!         带门 norm（xb 唯一中间 round、xw 不收、末级单 round）
//! - o17 · rope_pt：partial RoPE 单张量（前 rot_dim 旋转、其余原样）、
//!         位置含空隙（表寻址真值）
//! - o18 · sigmul：sigmoid 门控乘（q_proj 2× 输出的门路径）
//! - o19 · conv1d_dw + conv_state_roll：varlen 因果卷积（tap 跨段取
//!         state）与状态滚动（count<kernel-1 的移位路径）
//! - o20 · deltanet：递推位对齐（初始状态非零——in/out 状态语义 +
//!         最终状态回读对拍）；GQA 型 v/k 头比 2:1
//! - o21 · argmax_rows：全序归约（等值取小下标——真值构造非随机）
//! - g7 · gemm_wt：TRANSB 权重直读形态，容差模式（|got−ref64| ≤
//!         max(2⁻⁷·|ref|, 2⁻¹¹)）+ 单位阵真值锚（恒等时位精确）
#![allow(dead_code)]
use wein_kernel::cuda::Gpu;
use wein_kernel::ops::{
    ArgmaxRowsOp, Conv1dDwOp, ConvStateRollOp, DeltanetOp, GemmOp, OpTable, RmsNormGatedOp,
    RmsNormZcOp, RopePtOp, SigmulOp,
};
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

fn profile() -> HardwareProfile {
    HardwareProfile::with_device(Device::Cuda).with_arch("sm_89")
}

fn table(gpu: &Gpu) -> OpTable<'_> {
    OpTable::resolve(gpu, &profile()).expect("sm_89 画像须解析成功")
}

fn assert_bits(label: &str, got: &[u16], expect: &[u16]) {
    assert_eq!(got.len(), expect.len(), "{label} 长度");
    for i in 0..expect.len() {
        assert_eq!(got[i], expect[i], "{label}[{i}] 位不对齐（GPU {:#06x} vs 参考 {:#06x}）",
            got[i], expect[i]);
    }
}

fn vec_bf16(seed_hi: u64, n: usize, scale: f32) -> Vec<u16> {
    (0..n as u64).map(|i| reference::f32_to_bf16(reference::splitmix_f32(i ^ seed_hi) * scale)).collect()
}

fn vec_f32(seed_hi: u64, n: usize, scale: f32) -> Vec<f32> {
    (0..n as u64).map(|i| reference::splitmix_f32(i ^ seed_hi) * scale).collect()
}

/// o15 · 注册表长全：+8 槽位、名字可审（档位标识入日志）
#[test]
fn o15_op_table_all_i1_slots() {
    let Some(gpu) = gpu() else { return };
    let t = table(&gpu);
    assert_eq!(t.rms_norm_zc.name(), "rms_norm_zc/cuda");
    assert_eq!(t.rope_pt.name(), "rope_pt/cuda");
    assert_eq!(t.sigmul.name(), "sigmul/cuda");
    assert_eq!(t.conv1d_dw.name(), "conv1d_dw/cuda");
    assert_eq!(t.conv_state_roll.name(), "conv_state_roll/cuda");
    assert_eq!(t.deltanet.name(), "deltanet/cuda");
    assert_eq!(t.rms_norm_gated.name(), "rms_norm_gated/cuda");
    assert_eq!(t.argmax_rows.name(), "argmax_rows/cuda");
}

/// o16 · 两种新 norm 位对齐：零中心（input/post/final/q/k 共用形态）与
/// 带门（deltanet 输出路径；w 是 F32 checkpoint 原生权重）
#[test]
fn o16_norms_golden_bitwise() {
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    for (rows, hidden) in [(3usize, 65usize), (17, 256)] {
        // 零中心 RMSNorm：out = x·rsqrt(mean+eps)·(1+w)
        let x = vec_bf16(0x1001, rows * hidden, 3.0);
        let w = vec_bf16(0x2002, hidden, 0.5);
        let expect = reference::rms_norm_zc(&x, &w, rows, hidden, 1e-6);
        let (dx, dw, dout) = (
            gpu.alloc_from_slice(&x),
            gpu.alloc_from_slice(&w),
            gpu.alloc_bytes(x.len() * 2),
        );
        t.rms_norm_zc.run(&dx, &dw, &dout, rows as u32, hidden as u32, 1e-6, &stream);
        stream.sync();
        assert_bits(&format!("rms_norm_zc {rows}x{hidden}"), &dout.read_to_vec(), &expect);

        // 带门 RMSNorm：w F32；z 是 in_proj_z 投影输出
        let z = vec_bf16(0x3003, rows * hidden, 3.0);
        let wf = vec_f32(0x4004, hidden, 0.5);
        let expect = reference::rms_norm_gated(&x, &z, &wf, rows, hidden, 1e-6);
        let (dz, dwf, dout2) = (
            gpu.alloc_from_slice(&z),
            gpu.alloc_from_slice(&wf),
            gpu.alloc_bytes(x.len() * 2),
        );
        t.rms_norm_gated.run(&dx, &dz, &dwf, &dout2, rows as u32, hidden as u32, 1e-6, &stream);
        stream.sync();
        assert_bits(&format!("rms_norm_gated {rows}x{hidden}"), &dout2.read_to_vec(), &expect);
    }
}

/// o17 · partial RoPE 位对齐：前 rot_dim 维 rotate_half、其余不触碰。
/// out 缓冲先填哨兵——pass-through 维若被 kernel 写坏立刻现形
#[test]
fn o17_rope_pt_golden_bitwise() {
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    // head_dim 16、rot_dim 8（partial！）；pos 稀疏证表寻址
    let (tokens, heads, head_dim, rot_dim, max_pos) = (5usize, 3usize, 16usize, 8usize, 64usize);
    let half = rot_dim / 2;
    let pos: Vec<u32> = (0..tokens).map(|i| (i * 29 + 7) as u32 % max_pos as u32).collect();
    let (cos, sin) = reference::rope_table(max_pos, half, 1e7);
    let x = vec_bf16(0xa11ce, tokens * heads * head_dim, 4.0);
    let expect = reference::rope_pt(&x, &cos, &sin, &pos, rot_dim, head_dim, heads);

    let (dx, dc, ds, dp) = (
        gpu.alloc_from_slice(&x),
        gpu.alloc_from_slice(&cos),
        gpu.alloc_from_slice(&sin),
        gpu.alloc_from_slice(&pos),
    );
    // 哨兵底色：非旋转维必须原样（0xdead 是 x 里不存在的位型概率极高）
    let sentinel = vec![0xdeadu16; x.len()];
    let dout = gpu.alloc_from_slice(&sentinel);
    t.rope_pt.run(&dx, &dc, &ds, &dp, &dout, rot_dim as u32, head_dim as u32, heads as u32, tokens as u32, &stream);
    stream.sync();
    // pass-through 真值：非旋转维必须仍是哨兵（kernel 不写这些维——原地合同）
    let got: Vec<u16> = dout.read_to_vec();
    for tk in 0..tokens {
        for h in 0..heads {
            let base = tk * heads * head_dim + h * head_dim;
            for e in rot_dim..head_dim {
                assert_eq!(got[base + e], 0xdead,
                    "pass-through 维被写坏（token {tk} head {h} dim {e}）");
            }
        }
    }
    // 旋转维单独对拍（expect 的 pass-through 维是 x 原值，与哨兵不同底色）
    for tk in 0..tokens {
        for h in 0..heads {
            let base = tk * heads * head_dim + h * head_dim;
            for p in 0..half {
                assert_eq!(got[base + p], expect[base + p], "rope_pt 前半（token {tk} head {h} p {p}）");
                assert_eq!(got[base + half + p], expect[base + half + p], "rope_pt 后半（token {tk} head {h} p {p}）");
            }
        }
    }
}

/// o18 · sigmul 位对齐：out = x·sigmoid(g)；n 含非 block 整数倍
#[test]
fn o18_sigmul_golden_bitwise() {
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    for n in [1000usize, 4097] {
        let x = vec_bf16(0x51de, n, 3.0);
        let g = vec_bf16(0x600d, n, 2.0);
        let expect = reference::sigmul(&x, &g);
        let (dx, dg, dout) = (gpu.alloc_from_slice(&x), gpu.alloc_from_slice(&g), gpu.alloc_bytes(n * 2));
        t.sigmul.run(&dx, &dg, &dout, n as u32, &stream);
        stream.sync();
        assert_bits(&format!("sigmul n={n}"), &dout.read_to_vec(), &expect);
    }
}

/// o19 · 因果深度卷积 + 状态滚动位对齐：varlen 两段（count 5 与 2——
/// 后者走 state 移位路径），tap 跨段取 state、滚动后状态对拍。
/// 状态行经 row_id 非平凡间接（span 0→行 2、span 1→行 0）——池按请求行
/// 分账，若 kernel 拿 span 序号当行号写，行 2 留零、行 0 被串写，本条现形
#[test]
fn o19_conv_golden_bitwise() {
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let (conv_dim, kernel) = (16usize, 4usize);
    let (row_start, row_count): (Vec<u32>, Vec<u32>) = (vec![0, 5], vec![5, 2]);
    let row_id: Vec<u32> = vec![2, 0]; // 状态池 3 行，两段各占非本征行
    let pool_rows = 3usize;
    let tokens: usize = row_count.iter().sum::<u32>() as usize; // 7
    // tok_row/tok_pos 展开两段（executor 同款：tok_row = 请求状态行）
    let mut tok_row = Vec::new();
    let mut tok_pos = Vec::new();
    for (r, &c) in row_count.iter().enumerate() {
        for p in 0..c {
            tok_row.push(row_id[r]);
            tok_pos.push(p);
        }
    }
    let x = vec_bf16(0xc0ffee, tokens * conv_dim, 2.0);
    let w = vec_bf16(0xd0d0, conv_dim * kernel, 0.5);
    // 旧状态（span 1 只 2 步——滚动走读旧槽位的移位路径）
    let st_old = vec_bf16(0xe77e, pool_rows * conv_dim * (kernel - 1), 2.0);
    let expect_conv = reference::conv1d_dw(&x, &w, &st_old, &tok_row, &tok_pos, conv_dim, kernel);
    let expect_state = reference::conv_state_roll(&x, &st_old, &row_start, &row_count, &row_id, conv_dim, kernel);
    // 未触行（行 1）必须原样
    let untouched = &st_old[conv_dim * (kernel - 1)..2 * conv_dim * (kernel - 1)];
    assert_eq!(&expect_state[conv_dim * (kernel - 1)..2 * conv_dim * (kernel - 1)], untouched);

    let (dx, dw, ds_in) = (gpu.alloc_from_slice(&x), gpu.alloc_from_slice(&w), gpu.alloc_from_slice(&st_old));
    let (dtr, dtp) = (gpu.alloc_from_slice(&tok_row), gpu.alloc_from_slice(&tok_pos));
    let dout = gpu.alloc_bytes(tokens * conv_dim * 2);
    t.conv1d_dw.run(&dx, &dw, &ds_in, &dout, &dtr, &dtp, conv_dim as u32, kernel as u32, tokens as u32, &stream);
    stream.sync();
    assert_bits("conv1d_dw", &dout.read_to_vec(), &expect_conv);

    // 滚动：dst=旧状态上传（原地语义），跑完回读对拍
    let (drs, drc, dri) = (
        gpu.alloc_from_slice(&row_start),
        gpu.alloc_from_slice(&row_count),
        gpu.alloc_from_slice(&row_id),
    );
    let dstate = gpu.alloc_from_slice(&st_old);
    t.conv_state_roll.run(&dx, &dstate, &drs, &drc, &dri, conv_dim as u32, kernel as u32, row_start.len() as u32, &stream);
    stream.sync();
    assert_bits("conv_state_roll", &dstate.read_to_vec(), &expect_state);
}

/// o20 · gated deltanet 位对齐：初始状态非零（in/out 状态语义）、
/// v/k 头 2:1（repeat_interleave 映射）、最终状态回读对拍
#[test]
fn o20_deltanet_golden_bitwise() {
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let (k_heads, v_heads, k_dim, v_dim) = (2usize, 4usize, 8usize, 8usize);
    let (row_start, row_count): (Vec<u32>, Vec<u32>) = (vec![0, 6], vec![6, 3]);
    let row_id: Vec<u32> = vec![0, 1];
    let tokens: usize = 9;
    let n_rows = row_start.len();
    let q = vec_bf16(0x7a11, tokens * k_heads * k_dim, 2.0);
    let k = vec_bf16(0x7b22, tokens * k_heads * k_dim, 2.0);
    let v = vec_bf16(0x7c33, tokens * v_heads * v_dim, 2.0);
    let a = vec_bf16(0x7d44, tokens * v_heads, 1.0);
    let b = vec_bf16(0x7e55, tokens * v_heads, 1.0);
    let a_log: Vec<f32> = vec_f32(0x7f66, v_heads, 1.5).iter().map(|x| x - 0.5).collect();
    let dt_bias = vec_bf16(0x8077, v_heads, 0.5);
    // 非零初始状态：证明 state 是 in/out
    let mut state: Vec<f32> = vec_f32(0x9088, n_rows * v_heads * v_dim * k_dim, 0.5);

    let expect_out = reference::deltanet(&q, &k, &v, &a, &b, &a_log, &dt_bias, &mut state,
        &row_start, &row_count, &row_id, k_heads, v_heads, k_dim, v_dim, 1e-6);
    let expect_state = state.clone();

    let (dq, dk, dv, da, db) = (gpu.alloc_from_slice(&q), gpu.alloc_from_slice(&k), gpu.alloc_from_slice(&v),
        gpu.alloc_from_slice(&a), gpu.alloc_from_slice(&b));
    let (da_log, ddtb) = (gpu.alloc_from_slice(&a_log), gpu.alloc_from_slice(&dt_bias));
    // GPU 侧上传同一初值（mirror 上面已原地更新 state——用同种子重造）
    let init: Vec<f32> = vec_f32(0x9088, n_rows * v_heads * v_dim * k_dim, 0.5);
    let dstate = gpu.alloc_from_slice(&init);
    let dout = gpu.alloc_bytes(tokens * v_heads * v_dim * 2);
    let (drs, drc, dri) = (gpu.alloc_from_slice(&row_start), gpu.alloc_from_slice(&row_count), gpu.alloc_from_slice(&row_id));
    t.deltanet.run(&dq, &dk, &dv, &da, &db, &da_log, &ddtb, &dstate, &dout, &drs, &drc, &dri,
        n_rows as u32, k_heads as u32, v_heads as u32, k_dim as u32, v_dim as u32, 1e-6, &stream);
    stream.sync();
    assert_bits("deltanet core_out", &dout.read_to_vec(), &expect_out);
    let got_state: Vec<f32> = dstate.read_to_vec();
    assert_eq!(got_state.len(), expect_state.len(), "state 长度");
    for i in 0..expect_state.len() {
        assert_eq!(got_state[i].to_bits(), expect_state[i].to_bits(),
            "state[{i}] 位不对齐（GPU {:e} vs 参考 {:e}）", got_state[i], expect_state[i]);
    }
}

/// o21 · 行 argmax：全序归约确定——真值构造（等值取小下标、全负行、
/// 首/末位置极值），非随机
#[test]
fn o21_argmax_rows_golden_bitwise() {
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let n = 1000usize;
    // 3 行采样（总行 4）：0 平局取小下标；1 全负；2 末位极大值；3 不采样。
    // 注：splitmix_f32 ×3 的实际值域 ~[-3,15)（24 位尾数拼图溢出到指数域），
    // 真值极值必须放在值域外（50/-50）才谈得上"构造"
    let mut rows: Vec<Vec<u16>> = (0..4).map(|r| vec_bf16(0xab00 + r as u64, n, 3.0)).collect();
    let hi = reference::f32_to_bf16(50.0);
    let lo = reference::f32_to_bf16(-50.0);
    // 行 0：极值出现在 700 与 123（平局）→ 123
    rows[0][700] = hi;
    rows[0][123] = hi;
    // 行 1：全负（splitmix*scale 可能为正——显式全负重铺）
    for x in rows[1].iter_mut() {
        *x = lo;
    }
    rows[1][999] = reference::f32_to_bf16(-0.5); // 最大者唯一
    // 行 2：末位是极大
    rows[2][999] = hi;
    let flat: Vec<u16> = rows.concat();
    let picked: Vec<u32> = vec![0, 1, 2];
    let expect = reference::argmax_rows(&flat, &picked, n);
    assert_eq!(expect, vec![123, 999, 999], "参考端先钉真值（平局/全负/末位）");

    let (dx, dr) = (gpu.alloc_from_slice(&flat), gpu.alloc_from_slice(&picked));
    let scratch = gpu.alloc_bytes(picked.len() * 512 * 4);
    let dout = gpu.alloc_bytes(picked.len() * 4);
    t.argmax_rows.run(&dx, &dr, &scratch, &dout, n as u32, picked.len() as u32, &stream);
    stream.sync();
    let got: Vec<u32> = dout.read_to_vec();
    assert_eq!(got, expect, "argmax_rows 对拍（等值取小下标）");
}

/// g7 · gemm_wt 容差：C = A·Wᵀ（W 按存储 [n,k] 直读）。随机形状容差
/// （2⁻⁷ 相对 / 2⁻¹¹ 绝对）+ 单位阵真值锚（W=I 时恒等乘积位精确）
#[test]
fn g7_gemm_wt_tolerance() {
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    for (m, n, k) in [(33usize, 65usize, 48usize), (128, 2560, 2560)] {
        let a = vec_bf16(0x9a11, m * k, 1.0);
        let w = vec_bf16(0x9b22, n * k, 1.0); // 存储 [n,k]
        let ref64 = reference::gemm_wt(&a, &w, m, n, k);

        let (da, dw) = (gpu.alloc_from_slice(&a), gpu.alloc_from_slice(&w));
        let dc = gpu.alloc_bytes(m * n * 2);
        t.gemm.run_wt(&da, &dw, &dc, m as u32, n as u32, k as u32, &stream);
        stream.sync();
        let got: Vec<u16> = dc.read_to_vec();
        for i in 0..m * n {
            let (g, r) = (reference::bf16_to_f32(got[i]) as f64, ref64[i]);
            let tol = (2f64.powi(-7) * r.abs()).max(2f64.powi(-11));
            assert!(
                (g - r).abs() <= tol,
                "gemm_wt[{i}]（{m}×{n}×{k}）got {g} vs ref {r}，超容差 {tol}"
            );
        }
    }
    // 真值锚：W = I（n==k）→ C 逐位等于 A（单积精确）
    let (m, d) = (17usize, 33usize);
    let a = vec_bf16(0x9c33, m * d, 1.0);
    let mut w = vec![0u16; d * d];
    for (i, x) in w.iter_mut().enumerate() {
        *x = if i % (d + 1) == 0 { reference::f32_to_bf16(1.0) } else { reference::f32_to_bf16(0.0) };
    }
    let (da, dw) = (gpu.alloc_from_slice(&a), gpu.alloc_from_slice(&w));
    let dc = gpu.alloc_bytes(m * d * 2);
    t.gemm.run_wt(&da, &dw, &dc, m as u32, d as u32, d as u32, &stream);
    stream.sync();
    assert_bits("gemm_wt identity", &dc.read_to_vec(), &a);
}
