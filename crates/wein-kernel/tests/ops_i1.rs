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
    AddNormZcOp, AddOp, ArgmaxRowsOp, Conv1dDwOp, Conv1dDwRollOp, ConvStateRollOp, DeltanetChunkedOp, DeltanetOp,
    DnSplitQkvOp, EmbedOp, GemmOp, GemvOp, MulOp, OpTable, RmsNormGatedOp, RmsNormZcOp, RopePtOp,
    SigmulOp, SiluMulOp, SiluOp, ZeroRowsOp,
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

/// GPU 串行锁（graphs.rs 同款）：cargo test 并行线程的 kernel 会互相
/// 闯入事件计时窗——µs 级门禁（o23 的 60µs）对并发负载敏感，全文件持锁
fn guard() -> std::sync::MutexGuard<'static, ()> {
    static L: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    L.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
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
    let _g = guard();
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
    let _g = guard();
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
    let _g = guard();
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
    let _g = guard();
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
    let _g = guard();
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
    let expect_state = reference::conv_state_roll(&x, &st_old, &row_start, &row_count, &row_id, conv_dim, kernel, &vec![0u32; row_start.len()]);
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
    let dspec0 = gpu.alloc_from_slice(&vec![0u32; row_start.len()]); // 非投机路径：全 0 常量数组（一次分配，不每步重传）
    t.conv_state_roll.run(&dx, &dstate, &drs, &drc, &dri, conv_dim as u32, kernel as u32, row_start.len() as u32, &dspec0, &stream);
    stream.sync();
    assert_bits("conv_state_roll", &dstate.read_to_vec(), &expect_state);
}

/// o20 · gated deltanet 位对齐：初始状态非零（in/out 状态语义）、
/// v/k 头 2:1（repeat_interleave 映射）、最终状态回读对拍。
/// 状态池 bf16 化后：kernel↔reference 位对齐**保持**（reference 同步镜像
/// 「存储即收 bf16」语义）——重开的是对外合同（vs vLLM 字节全同），
/// o25 另钉量化漂移预算
#[test]
fn o20_deltanet_golden_bitwise() {
    // kd=8 走通用 u64 车道路径；kd=128 走行驻留快路径（local depot/寄存器）
    o20_case(2, 4, 8, 8, 1);
    o20_case(2, 4, 128, 8, 1);
    o20_case(4, 8, 128, 128, 1); // E2 路径：kd=vd=128（块=(行,v 头)×4 warp，kvm/o 两级折叠）
    // B 档 chunked：同一形状按 v 行切 4 块（块=(行, v 头, v 行块)，grid ×4）
    // ——逐位必须同上（切块只改"哪个块负责哪几行"，行内算术一个字不动）
    o20_case(4, 8, 128, 128, 4);
    o20_case(4, 8, 128, 128, 2);
    // 8 块 = 上一档生产 prefill 几何（16 行/块 × 2 warp × 8 行 = 批 8）
    o20_case(4, 8, 128, 128, 8);
    // 16 块 = invq/invk 去重后的生产 prefill 几何（8 行/块 × 2 warp × 4 行 =
    // 批 4；单 span 长 prefill 的"块数 ≈ 512"配比就落在这档）
    o20_case(4, 8, 128, 128, 16);
}

fn o20_case(k_heads: usize, v_heads: usize, k_dim: usize, v_dim: usize, vsplit: u32) {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let tag = format!("deltanet kd={k_dim} vd={v_dim} vsplit={vsplit}");
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
    // 非零初始状态（bf16 存储——初值即收，两侧同值）：证明 state 是 in/out
    let mut state: Vec<u16> = vec_f32(0x9088, n_rows * v_heads * v_dim * k_dim, 0.5)
        .iter()
        .map(|&f| reference::f32_to_bf16(f))
        .collect();

    let expect_out = reference::deltanet(&q, &k, &v, &a, &b, &a_log, &dt_bias, &mut state,
        &row_start, &row_count, &row_id, k_heads, v_heads, k_dim, v_dim, 1e-6, &vec![0u32; n_rows]);
    let expect_state = state.clone();

    let (dq, dk, dv, da, db) = (gpu.alloc_from_slice(&q), gpu.alloc_from_slice(&k), gpu.alloc_from_slice(&v),
        gpu.alloc_from_slice(&a), gpu.alloc_from_slice(&b));
    let (da_log, ddtb) = (gpu.alloc_from_slice(&a_log), gpu.alloc_from_slice(&dt_bias));
    // GPU 侧上传同一初值（mirror 上面已原地更新 state——用同种子重造）
    let init: Vec<u16> = vec_f32(0x9088, n_rows * v_heads * v_dim * k_dim, 0.5)
        .iter()
        .map(|&f| reference::f32_to_bf16(f))
        .collect();
    let dstate = gpu.alloc_from_slice(&init);
    let dout = gpu.alloc_bytes(tokens * v_heads * v_dim * 2);
    let (drs, drc, dri) = (gpu.alloc_from_slice(&row_start), gpu.alloc_from_slice(&row_count), gpu.alloc_from_slice(&row_id));
    let dspec0 = gpu.alloc_from_slice(&vec![0u32; n_rows]); // 非投机路径：全 0 常量数组
    t.deltanet.run(&dq, &dk, &dv, &da, &db, &da_log, &ddtb, &dstate, &dout, &drs, &drc, &dri,
        n_rows as u32, k_heads as u32, v_heads as u32, k_dim as u32, v_dim as u32,
        tokens as u32, 1e-6, &dspec0, vsplit, &stream);
    stream.sync();
    assert_bits(&format!("{tag} core_out"), &dout.read_to_vec(), &expect_out);
    assert_bits(&format!("{tag} state"), &dstate.read_to_vec(), &expect_state);
}

/// o25 · 状态池 bf16 化漂移预算：同输入同门同 l2norm 下，bf16 存态递推
/// （现行语义——存储即收，o/递推全吃收过的值）vs f32 存态递推（本刀前
/// 语义）。这是对外合同重开的量化账：vs vLLM 逐字节全同 → 容差对照，
/// 本测试钉"递推层自身引入多少漂移"。真形状 kd=vd=128、T=256 长递推，
/// 输出端逐 token 相对误差 max + 末态相对误差 max，门 = 实测 × 裕量
#[test]
fn o25_deltanet_bf16_state_drift_budget() {
    let (k_heads, v_heads, k_dim, v_dim) = (1usize, 2usize, 128usize, 128usize);
    let tokens = 256usize;
    let q = vec_bf16(0xa5a5, tokens * k_heads * k_dim, 2.0);
    let k = vec_bf16(0xa6b6, tokens * k_heads * k_dim, 2.0);
    let v = vec_bf16(0xa7c7, tokens * v_heads * v_dim, 2.0);
    let a = vec_bf16(0xa8d8, tokens * v_heads, 1.0);
    let b = vec_bf16(0xa9e9, tokens * v_heads, 1.0);
    let a_log: Vec<f32> = (0..v_heads).map(|h| 0.3 + 0.2 * h as f32).collect();
    let dt_bias = vec_bf16(0xaaf0, v_heads, 0.5);
    let eps = 1e-6f32;
    let kd_sqrt = (k_dim as f32).sqrt();

    // f32 存态真值：与 reference::deltanet 逐句同构（含 E2 两级折叠归约
    // 树），仅缺"落池收 bf16"那一跳（s 保持 f32 不收）——即本刀前的递推语义
    let mut s: Vec<f32> = vec![0.0; v_heads * v_dim * k_dim];
    let mut out_f32 = vec![0u16; tokens * v_heads * v_dim];
    for h in 0..v_heads {
        let kh = h * k_heads / v_heads;
        let big_a = reference::expf(a_log[h]);
        let dtb = reference::bf16_to_f32(dt_bias[h]);
        let sbase = h * v_dim * k_dim;
        for g in 0..tokens {
            let decay = reference::expf(0.0 - big_a * reference::softplus(reference::bf16_to_f32(a[g * v_heads + h]) + dtb));
            let beta = reference::bf16_to_f32(reference::f32_to_bf16(
                1.0f32 / (1.0f32 + reference::expf(0.0 - reference::bf16_to_f32(b[g * v_heads + h]))),
            ));
            let qb = (g * k_heads + kh) * k_dim;
            let (mut sq, mut sk) = (0.0f32, 0.0f32);
            for j in 0..k_dim {
                let qv = reference::bf16_to_f32(q[qb + j]);
                let kv = reference::bf16_to_f32(k[qb + j]);
                sq = qv * qv + sq;
                sk = kv * kv + sk;
            }
            let invq = 1.0f32 / ((sq + eps).sqrt());
            let invk = 1.0f32 / ((sk + eps).sqrt());
            for vv in 0..v_dim {
                let srow = &mut s[sbase + vv * k_dim..][..k_dim];
                for j in 0..k_dim {
                    srow[j] *= decay;
                }
                let mut kvm = 0.0f32;
                for sl in 0..k_dim / 4 {
                    let mut p = 0.0f32;
                    for i in 0..4 {
                        p = srow[sl * 4 + i] * (reference::bf16_to_f32(k[qb + sl * 4 + i]) * invk) + p;
                    }
                    kvm = p + kvm;
                }
                let vin = reference::bf16_to_f32(v[(g * v_heads + h) * v_dim + vv]);
                let delta = (vin - kvm) * beta;
                for j in 0..k_dim {
                    srow[j] = (reference::bf16_to_f32(k[qb + j]) * invk) * delta + srow[j];
                }
                let mut o = 0.0f32;
                for sl in 0..k_dim / 4 {
                    let mut p = 0.0f32;
                    for i in 0..4 {
                        p = srow[sl * 4 + i] * ((reference::bf16_to_f32(q[qb + sl * 4 + i]) * invq) / kd_sqrt) + p;
                    }
                    o = p + o;
                }
                out_f32[(g * v_heads + h) * v_dim + vv] = reference::f32_to_bf16(o);
            }
        }
    }

    // bf16 存态（现行 reference——存储即收）
    let mut s_bf = vec![0u16; v_heads * v_dim * k_dim];
    let out_bf = reference::deltanet(
        &q, &k, &v, &a, &b, &a_log, &dt_bias, &mut s_bf, &[0], &[tokens as u32], &[0],
        k_heads, v_heads, k_dim, v_dim, eps, &[0],
    );

    // 度量：单元素相对误差在近零输出上无意义（分母→0），用归一化账——
    // Frobenius 相对误差（整体能量占比）+ 峰值归一（max|Δ|/max|y|）+ 位差计数
    fn norm_err(xs: impl Iterator<Item = (f32, f32)>) -> (f64, f64) {
        let (mut sse, mut sy, mut max_d, mut max_y) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (x, y) in xs {
            sse += ((x - y) * (x - y)) as f64;
            sy += (y * y) as f64;
            max_d = max_d.max((x - y).abs() as f64);
            max_y = max_y.max(y.abs() as f64);
        }
        ((sse / sy).sqrt(), max_d / max_y)
    }
    let (rel2_o, maxn_o) = norm_err(
        out_bf.iter().zip(out_f32.iter())
            .map(|(&b, &f)| (reference::bf16_to_f32(b), reference::bf16_to_f32(f))),
    );
    let (rel2_s, maxn_s) = norm_err(
        s_bf.iter().zip(s.iter()).map(|(&b, &f)| (reference::bf16_to_f32(b), f)),
    );
    let diff_cnt = out_bf.iter().zip(out_f32.iter()).filter(|(a, c)| a != c).count();
    eprintln!(
        "o25 bf16 池漂移（T=256, kd=vd=128）：输出 rel2 {rel2_o:.3e}/峰值归一 {maxn_o:.3e}（{diff_cnt}/{} 位不同），末态 rel2 {rel2_s:.3e}/峰值归一 {maxn_s:.3e}",
        out_bf.len()
    );
    // 门 = 实测 × ~4 裕量（实测 rel2 1.3e-3/峰值 4.2e-3——decay 收缩使量化
    // 误差稳态不累计，峰值 ≈ 1 个 bf16 量子）。rel2 是能量占比（对近零元素
    // 稳健），峰值归一钉最坏单点。越门 = 存/读舍入位变动（kernel↔reference
    // 镜像破坏），非量化噪声
    assert!(rel2_o <= 5.0e-3, "o25 输出 rel2 漂移越门：{rel2_o:.3e} > 5.0e-3");
    assert!(maxn_o <= 2.0e-2, "o25 输出峰值漂移越门：{maxn_o:.3e} > 2.0e-2");
    assert!(rel2_s <= 5.0e-3, "o25 末态 rel2 漂移越门：{rel2_s:.3e} > 5.0e-3");
    assert!(maxn_s <= 2.0e-2, "o25 末态峰值漂移越门：{maxn_s:.3e} > 2.0e-2");
}

/// o22 · deltanet 时延金丝雀：满负荷 decode 形状（m=100 行 × 32 v 头 ×
/// kd=vd=128，每行 1 token）单 launch 事件计时。E2 形状（一块=(行,v 头)
/// ×4 warp、lane 持 j 切片 u64——warp 指令 256B 全合并）后预期 ~0.26-0.31ms
/// （状态流 205MB @ ~700-850GB/s）；门禁 0.36ms——防散射形态回归（带宽
/// 探针实证：u64 跨步散射真 DRAM 仅 ~280-500GB/s，全合并 ~880）
#[test]
fn o22_deltanet_latency_canary() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let (n_rows, k_heads, v_heads, k_dim, v_dim) = (100usize, 16usize, 32usize, 128usize, 128usize);
    let tokens = n_rows; // decode：每行 1 token
    let q = vec_bf16(0x6a12, tokens * k_heads * k_dim, 1.0);
    let k = vec_bf16(0x6b34, tokens * k_heads * k_dim, 1.0);
    let v = vec_bf16(0x6c56, tokens * v_heads * v_dim, 1.0);
    let a = vec_bf16(0x6d78, tokens * v_heads, 1.0);
    let b = vec_bf16(0x6e9a, tokens * v_heads, 1.0);
    let a_log: Vec<f32> = (0..v_heads).map(|h| 0.3 + 0.2 * h as f32).collect();
    let dt_bias = vec_bf16(0x6fbc, v_heads, 0.5);
    let state_len = n_rows * v_heads * v_dim * k_dim;
    let state = vec![0u16; state_len]; // 状态池 bf16——时延与数据值无关
    let row_start: Vec<u32> = (0..n_rows as u32).collect();
    let row_count = vec![1u32; n_rows];
    let row_id: Vec<u32> = (0..n_rows as u32).collect();

    let (dq, dk, dv, da, db) = (
        gpu.alloc_from_slice(&q),
        gpu.alloc_from_slice(&k),
        gpu.alloc_from_slice(&v),
        gpu.alloc_from_slice(&a),
        gpu.alloc_from_slice(&b),
    );
    let (da_log, ddtb) = (gpu.alloc_from_slice(&a_log), gpu.alloc_from_slice(&dt_bias));
    let dstate = gpu.alloc_from_slice(&state);
    let dout = gpu.alloc_bytes(tokens * v_heads * v_dim * 2);
    let (drs, drc, dri) = (
        gpu.alloc_from_slice(&row_start),
        gpu.alloc_from_slice(&row_count),
        gpu.alloc_from_slice(&row_id),
    );
    // 非投机路径的全 0 常量数组：循环外一次分配（计时循环内禁止分配）
    let dspec0 = gpu.alloc_from_slice(&vec![0u32; n_rows]);
    for _ in 0..3 {
        // 热身（JIT/缓存就位）；状态每拍原地演化——形状与访存量不变
        t.deltanet.run(&dq, &dk, &dv, &da, &db, &da_log, &ddtb, &dstate, &dout, &drs, &drc, &dri,
            n_rows as u32, k_heads as u32, v_heads as u32, k_dim as u32, v_dim as u32,
            tokens as u32, 1e-6, &dspec0, 1, &stream);
    }
    stream.sync();
    let (e0, e1) = (gpu.event(), gpu.event());
    e0.record_on(&stream);
    for _ in 0..20 {
        t.deltanet.run(&dq, &dk, &dv, &da, &db, &da_log, &ddtb, &dstate, &dout, &drs, &drc, &dri,
            n_rows as u32, k_heads as u32, v_heads as u32, k_dim as u32, v_dim as u32,
            tokens as u32, 1e-6, &dspec0, 1, &stream);
    }
    e1.record_on(&stream);
    stream.sync();
    let per_launch = e1.elapsed_ms(&e0) / 20.0;
    eprintln!("o22 deltanet m=100×32×128²：{per_launch:.3} ms/launch（20 拍均值）");
    assert!(per_launch < 0.36, "deltanet {per_launch:.3}ms 超 0.36ms 门禁——状态访存形态回归（u64 散射跨步）");
}

/// o23 · rms 系时延金丝雀：满负荷 decode 两真实形状——主 norm
/// （rows=100 × hidden=2560）与 q/k norm（rows=1600 × hidden=128）。
/// 病灶：row-per-thread 串行扫整行（rows=100 时全 GPU 仅 100 线程在跑，
/// 3-4 warp 喂不饱 DRAM 延迟——372µs/拍）。门禁 60µs 逼出 block-per-row
/// 共享内存归约（地板 ~6µs + launch 4.7µs）
#[test]
fn o23_rms_norm_latency_canary() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    for &(rows, hidden) in &[(100usize, 2560usize), (1600usize, 128usize)] {
        let x = vec_bf16(0x7e11, rows * hidden, 1.0);
        let w = vec_bf16(0x7f22, hidden, 1.0);
        let z = vec_bf16(0x7033, rows * hidden, 1.0);
        let wf: Vec<f32> = (0..hidden).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
        let (dx, dw, dz, dwf) = (
            gpu.alloc_from_slice(&x),
            gpu.alloc_from_slice(&w),
            gpu.alloc_from_slice(&z),
            gpu.alloc_from_slice(&wf),
        );
        let dout = gpu.alloc_bytes(rows * hidden * 2);
        for _ in 0..3 {
            t.rms_norm_zc.run(&dx, &dw, &dout, rows as u32, hidden as u32, 1e-6, &stream);
            t.rms_norm_gated.run(&dx, &dz, &dwf, &dout, rows as u32, hidden as u32, 1e-6, &stream);
        }
        stream.sync();
        let (e0, e1) = (gpu.event(), gpu.event());
        e0.record_on(&stream);
        for _ in 0..50 {
            t.rms_norm_zc.run(&dx, &dw, &dout, rows as u32, hidden as u32, 1e-6, &stream);
            t.rms_norm_gated.run(&dx, &dz, &dwf, &dout, rows as u32, hidden as u32, 1e-6, &stream);
        }
        e1.record_on(&stream);
        stream.sync();
        let per_launch = e1.elapsed_ms(&e0) / 100.0; // 两 kernel × 50 拍均摊
        eprintln!("o23 rms rows={rows} hidden={hidden}：{:.1} µs/launch（均摊）", per_launch * 1000.0);
        assert!(per_launch * 1000.0 < 60.0,
            "rms({rows}×{hidden}) {per_launch:.1}µs 超 60µs 门禁——row-per-thread 串行扫行形态回归");
    }
}

/// o21 · 行 argmax：全序归约确定——真值构造（等值取小下标、全负行、
/// 首/末位置极值），非随机
#[test]
fn o21_argmax_rows_golden_bitwise() {
    let _g = guard();
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
    let scratch = gpu.alloc_bytes(wein_kernel::ops::argmax_scratch_words(picked.len() as u32) as usize * 4);
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
    let _g = guard();
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

/// o40 · gemv_bf16 / gemv_bf16_row（m=1 直路 GEMV）容差：out[n] = W[n,k]·x[k]。
/// 双档几何（n≤GEMV_ROW_N 走 block-per-row，否则 warp-per-row）与齐尾/非齐尾
/// k（2056 = 8·257，主循环 256 步 + 8 元素余尾）各覆盖；容差口径同 g7。
#[test]
fn o40_gemv_tolerance() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    for (n, k) in [(32usize, 2560usize), (1024, 2560), (2560, 4096), (9216, 2056)] {
        let x = vec_bf16(0xad11, k, 1.0);
        let w = vec_bf16(0xad22, n * k, 1.0); // 存储 [n,k]
        let ref64 = reference::gemm_wt(&x, &w, 1, n, k);
        let (dx, dw) = (gpu.alloc_from_slice(&x), gpu.alloc_from_slice(&w));
        let dout = gpu.alloc_bytes(n * 2);
        t.gemv.run(&dw, &dx, &dout, n as u32, k as u32, &stream);
        stream.sync();
        let got: Vec<u16> = dout.read_to_vec();
        for i in 0..n {
            let (g, r) = (reference::bf16_to_f32(got[i]) as f64, ref64[i]);
            let tol = (2f64.powi(-7) * r.abs()).max(2f64.powi(-11));
            assert!(
                (g - r).abs() <= tol,
                "gemv[{i}]（n={n} k={k}）got {g} vs ref {r}，超容差 {tol}"
            );
        }
    }
    // 与 cuBLASLt 同题面互拍（真步面：m=1 全形状）：报 rel2 / 峰值归一，
    // 门 5e-3 = 一个 bf16 量子档（换累加树的固有扰动，非实现错）
    for (n, k) in [(32usize, 2560usize), (1024, 2560), (2560, 4096), (4096, 2560), (9216, 2560), (8192, 2560)] {
        let x = vec_bf16(0xaf11, k, 1.0);
        let w = vec_bf16(0xaf22, n * k, 1.0);
        let (dx, dw) = (gpu.alloc_from_slice(&x), gpu.alloc_from_slice(&w));
        let (da, db) = (gpu.alloc_bytes(n * 2), gpu.alloc_bytes(n * 2));
        t.gemm.run_wt(&dx, &dw, &da, 1, n as u32, k as u32, &stream);
        t.gemv.run(&dw, &dx, &db, n as u32, k as u32, &stream);
        stream.sync();
        let (u, v) = (da.read_to_vec(), db.read_to_vec());
        let (mut num, mut den, mut peak) = (0f64, 0f64, 0f64);
        for (&a, &b) in u.iter().zip(&v) {
            let (x, y) = (reference::bf16_to_f32(a) as f64, reference::bf16_to_f32(b) as f64);
            num += (x - y) * (x - y);
            den += x * x;
            peak = peak.max(x.abs());
        }
        let (rel2, pn) = (num / den.max(1e-30), (num / u.len() as f64).sqrt() / peak.max(1e-30));
        eprintln!("gemv vs lt n={n} k={k}：rel2 {rel2:.3e} 峰值归一 {pn:.3e}");
        assert!(rel2 <= 5.0e-3 && pn <= 5.0e-3, "gemv ≢ lt（n={n} k={k}）：rel2 {rel2:.3e}/{pn:.3e}");
    }
}

/// o43 · gemm_skinny（满载 skinny-m 自写 GEMM）容差验收：m∈[16,128] 覆盖
/// 奇尾行（33/100 的尾行钳位）、双 bdim 档（m≤64 用 128 线程 / >64 用 256）、
/// n 非 64 整倍（192 = 3×64）与真步面全形状（含 lm_head 的 n=248320 规模
/// 用 8192 代表）。块=64 列 n 片、A 片从 L2 复读、wmma m16n16k16 累加树与
/// cuBLASLt 不同 ⇒ 容差档（rel2 ≤ 3e-3 且峰值归一 ≤ 3e-3，o25 口径）。
/// 另留一条 `run_wt` 回退锚：n 非 64 整倍时默认路（cuBLASLt）不受扰动，
/// W=I 时逐位等于 A。
#[test]
fn o43_skinny_gemm_tolerance() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    for (m, n, k) in
        [(33usize, 192usize, 128usize), (16, 256, 2560), (64, 1024, 2560), (100, 2560, 2560), (128, 8192, 2560)]
    {
        let a = vec_bf16(0xb311, m * k, 1.0);
        let w = vec_bf16(0xb322, n * k, 1.0); // 存储 [n,k]
        let ref64 = reference::gemm_wt(&a, &w, m, n, k);
        let (da, dw) = (gpu.alloc_from_slice(&a), gpu.alloc_from_slice(&w));
        let dc = gpu.alloc_bytes(m * n * 2);
        t.gemm.run_skinny(&da, &dw, &dc, m as u32, n as u32, k as u32, &stream);
        stream.sync();
        let got: Vec<u16> = dc.read_to_vec();
        let (mut num, mut den, mut peak) = (0f64, 0f64, 0f64);
        for i in 0..m * n {
            let (g, r) = (reference::bf16_to_f32(got[i]) as f64, ref64[i]);
            let tol = (2f64.powi(-7) * r.abs()).max(2f64.powi(-11));
            assert!(
                (g - r).abs() <= tol,
                "gemm_skinny[{i}]（{m}×{n}×{k}）got {g} vs ref {r}，超容差 {tol}"
            );
            num += (g - r) * (g - r);
            den += r * r;
            peak = peak.max(r.abs());
        }
        let (rel2, pn) = (num / den.max(1e-30), (num / (m * n) as f64).sqrt() / peak.max(1e-30));
        eprintln!("gemm_skinny {m}×{n}×{k}：rel2 {rel2:.3e} 峰值归一 {pn:.3e}");
        assert!(rel2 <= 3.0e-3 && pn <= 3.0e-3, "gemm_skinny 超容差（{m}×{n}×{k}）：rel2 {rel2:.3e}/{pn:.3e}");
    }
    // 回退锚：n=96 非 64 整倍 ⇒ 默认 `run_wt` 仍走 cuBLASLt，W=I 时逐位等于 A
    let (m, d) = (17usize, 96usize);
    let a = vec_bf16(0xb433, m * d, 1.0);
    let mut w = vec![0u16; d * d];
    for (i, x) in w.iter_mut().enumerate() {
        *x = if i % (d + 1) == 0 { reference::f32_to_bf16(1.0) } else { reference::f32_to_bf16(0.0) };
    }
    let (da, dw) = (gpu.alloc_from_slice(&a), gpu.alloc_from_slice(&w));
    let dc = gpu.alloc_bytes(m * d * 2);
    t.gemm.run_wt(&da, &dw, &dc, m as u32, d as u32, d as u32, &stream);
    stream.sync();
    assert_bits("gemm_wt 回退锚 identity", &dc.read_to_vec(), &a);
}

/// o35 · deltanet chunked（BT 分块递推，FLA 同构）容差验收：随机形状 ×
/// BT 档（32/64/128）× 双行非整块尾 vs reference::deltanet（bf16 池老递推，
/// 每 token 落池收一次）。分块把"每 token 读写一遍状态"换成块间串行 +
/// 块内 WY/UT（(I+A)⁻¹ 联立前代 + 块边界快照）——递推代数变形破位级合同，
/// 走容差档：**rel2 ≤ 3e-3 且峰值归一 ≤ 3e-3**（o25 口径），core_out 与
/// 末态池都验。误差源 = reference 逐 token bf16 收 vs 本路 f32 链 + 块边界
/// 收（o25 同类实测 rel2 1.26e-3）——越门说明是代数/实现错，不是舍入。
fn o35_case(k_heads: usize, v_heads: usize, row_start: &[u32], row_count: &[u32], bt: u32, vsplit: u32) {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let (k_dim, v_dim) = (128usize, 128usize);
    let n_rows = row_start.len();
    let tokens: usize = row_count.iter().map(|&c| c as usize).sum();
    let row_id: Vec<u32> = (0..n_rows as u32).collect();
    let mut snap_off = vec![0u32];
    for &c in row_count {
        let last = *snap_off.last().unwrap();
        snap_off.push(last + c.div_ceil(bt));
    }
    let chunks = *snap_off.last().unwrap();
    let tag = format!("dn_chunked vh={v_heads} rows={row_count:?} bt={bt} vs={vsplit}");
    let q = vec_bf16(0xb1a1, tokens * k_heads * k_dim, 2.0);
    let k = vec_bf16(0xb2b2, tokens * k_heads * k_dim, 2.0);
    let v = vec_bf16(0xb3c3, tokens * v_heads * v_dim, 2.0);
    let a = vec_bf16(0xb4d4, tokens * v_heads, 1.0);
    let b = vec_bf16(0xb5e5, tokens * v_heads, 1.0);
    let a_log: Vec<f32> = vec_f32(0xb6f6, v_heads, 1.5).iter().map(|x| x - 0.5).collect();
    let dt_bias = vec_bf16(0xb707, v_heads, 0.5);
    // 非零初始状态（bf16——两侧同值起步）
    let init: Vec<u16> = vec_f32(0xb818, n_rows * v_heads * v_dim * k_dim, 0.5)
        .iter()
        .map(|&f| reference::f32_to_bf16(f))
        .collect();
    let mut state = init.clone();
    let expect_out = reference::deltanet(&q, &k, &v, &a, &b, &a_log, &dt_bias, &mut state,
        row_start, row_count, &row_id, k_heads, v_heads, k_dim, v_dim, 1e-6, &vec![0u32; n_rows]);
    let expect_state = state.clone();

    // f32 存态真值锚（o25 同构，多行独立链）：reference 逐 token 落池收 bf16，
    // 本链全程 f32 不收——两侧与锚的三方差分定位误差归属（见尾部度量）
    let kd_sqrt = (k_dim as f32).sqrt();
    let mut s_ex: Vec<f32> = init.iter().map(|&b| reference::bf16_to_f32(b)).collect();
    let mut out_ex = vec![0u16; tokens * v_heads * v_dim];
    for r in 0..n_rows {
        for h in 0..v_heads {
            let kh = h * k_heads / v_heads;
            let big_a = reference::expf(a_log[h]);
            let dtb = reference::bf16_to_f32(dt_bias[h]);
            let sbase = (row_id[r] as usize * v_heads + h) * v_dim * k_dim;
            for g in row_start[r] as usize..row_start[r] as usize + row_count[r] as usize {
                let decay = reference::expf(0.0
                    - big_a * reference::softplus(reference::bf16_to_f32(a[g * v_heads + h]) + dtb));
                let beta = reference::bf16_to_f32(reference::f32_to_bf16(
                    1.0f32 / (1.0f32 + reference::expf(0.0 - reference::bf16_to_f32(b[g * v_heads + h]))),
                ));
                let qb = (g * k_heads + kh) * k_dim;
                let (mut sq, mut sk) = (0.0f32, 0.0f32);
                for j in 0..k_dim {
                    let qv = reference::bf16_to_f32(q[qb + j]);
                    let kv = reference::bf16_to_f32(k[qb + j]);
                    sq = qv * qv + sq;
                    sk = kv * kv + sk;
                }
                let invq = 1.0f32 / ((sq + 1e-6).sqrt());
                let invk = 1.0f32 / ((sk + 1e-6).sqrt());
                for vv in 0..v_dim {
                    let srow = &mut s_ex[sbase + vv * k_dim..][..k_dim];
                    for j in 0..k_dim {
                        srow[j] *= decay;
                    }
                    let mut kvm = 0.0f32;
                    for j in 0..k_dim {
                        kvm = srow[j] * (reference::bf16_to_f32(k[qb + j]) * invk) + kvm;
                    }
                    let delta = (reference::bf16_to_f32(v[(g * v_heads + h) * v_dim + vv]) - kvm) * beta;
                    for j in 0..k_dim {
                        srow[j] = (reference::bf16_to_f32(k[qb + j]) * invk) * delta + srow[j];
                    }
                    let mut o = 0.0f32;
                    for j in 0..k_dim {
                        o = srow[j] * ((reference::bf16_to_f32(q[qb + j]) * invq) / kd_sqrt) + o;
                    }
                    out_ex[(g * v_heads + h) * v_dim + vv] = reference::f32_to_bf16(o);
                }
            }
        }
    }

    let (dq, dk, dv, da, db) = (gpu.alloc_from_slice(&q), gpu.alloc_from_slice(&k),
        gpu.alloc_from_slice(&v), gpu.alloc_from_slice(&a), gpu.alloc_from_slice(&b));
    let (da_log, ddtb) = (gpu.alloc_from_slice(&a_log), gpu.alloc_from_slice(&dt_bias));
    let dstate = gpu.alloc_from_slice(&init);
    let dout = gpu.alloc_bytes(tokens * v_heads * v_dim * 2);
    let (drs, drc, dri) = (gpu.alloc_from_slice(&row_start), gpu.alloc_from_slice(&row_count),
        gpu.alloc_from_slice(&row_id));
    let dsn = gpu.alloc_from_slice(&snap_off);
    let sp = t.deltanet_chunked.run(&dq, &dk, &dv, &da, &db, &da_log, &ddtb, &dstate, &dout,
        &drs, &drc, &dri, &dsn, n_rows as u32, chunks, k_heads as u32, v_heads as u32,
        k_dim as u32, v_dim as u32, tokens as u32, 1e-6, bt, vsplit, &stream);
    stream.sync();
    let (got_out, got_state): (Vec<u16>, Vec<u16>) = (dout.read_to_vec(), dstate.read_to_vec());

    fn norm_err(xs: impl Iterator<Item = (f32, f32)>) -> (f64, f64) {
        let (mut sse, mut sy, mut max_d, mut max_y) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (x, y) in xs {
            sse += ((x - y) * (x - y)) as f64;
            sy += (y * y) as f64;
            max_d = max_d.max((x - y).abs() as f64);
            max_y = max_y.max(y.abs() as f64);
        }
        ((sse / sy).sqrt(), max_d / max_y)
    }
    let f32s = |x: &[u16]| x.iter().map(|&b| reference::bf16_to_f32(b)).collect::<Vec<_>>();
    let ex_o = f32s(&out_ex);
    let ex_s: Vec<f32> = s_ex.clone();
    let (rel2_cr_o, peak_cr_o) = norm_err(f32s(&got_out).iter().zip(&f32s(&expect_out)).map(|(&x, &y)| (x, y)));
    let (rel2_cr_s, peak_cr_s) = norm_err(f32s(&got_state).iter().zip(&f32s(&expect_state)).map(|(&x, &y)| (x, y)));
    let (rel2_ce_o, peak_ce_o) = norm_err(f32s(&got_out).iter().zip(&ex_o).map(|(&x, &y)| (x, y)));
    let (rel2_ce_s, peak_ce_s) = norm_err(f32s(&got_state).iter().zip(&ex_s).map(|(&x, &y)| (x, y)));
    let (_, peak_re_o) = norm_err(f32s(&expect_out).iter().zip(&ex_o).map(|(&x, &y)| (x, y)));
    let (_, peak_re_s) = norm_err(f32s(&expect_state).iter().zip(&ex_s).map(|(&x, &y)| (x, y)));
    if peak_ce_o > 4.0e-3 {
        // 离群元素解剖：got/锚/ref 三方逐元素对质（锚=全程 f32 只在末尾收一次
        // bf16；ref=每 token 落池收 bf16）
        let (gf, ef) = (f32s(&got_out), f32s(&expect_out));
        let mut idx: Vec<usize> = (0..gf.len()).filter(|&i| (gf[i] - ex_o[i]).abs() > 3.9e-3).collect();
        idx.sort_by(|&a, &b| (gf[a] - ex_o[a]).abs().total_cmp(&(gf[b] - ex_o[b]).abs()));
        for &i in idx.iter().rev().take(4) {
            eprintln!(
                "  锚离群[{i}] tok={} h={} vv={}：got {:.6} / 锚 {:.6} / ref {:.6}（|got−锚| {:.3e}，|got−ref| {:.3e}，|锚−ref| {:.3e}）",
                i / (v_heads * v_dim), (i / v_dim) % v_heads, i % v_dim,
                gf[i], ex_o[i], ef[i], (gf[i] - ex_o[i]).abs(), (gf[i] - ef[i]).abs(), (ex_o[i] - ef[i]).abs(),
            );
        }
    }
    eprintln!(
        "o35 {tag}（钳后 vsplit={sp}，chunks={chunks}）：\n    vs reference  out rel2 {rel2_cr_o:.3e}/峰 {peak_cr_o:.3e}，state rel2 {rel2_cr_s:.3e}/峰 {peak_cr_s:.3e}\n    vs f32 锚     out rel2 {rel2_ce_o:.3e}/峰 {peak_ce_o:.3e}，state rel2 {rel2_ce_s:.3e}/峰 {peak_ce_s:.3e}\n    (锚 vs ref   out 峰 {peak_re_o:.3e}，state 峰 {peak_re_s:.3e}——reference 自身逐 token bf16 收的噪声地板)",
    );
    // 合同门：rel2（能量占比）vs reference ≤ 3e-3——代数正确的判据（实测
    // 1.28e-3，与 o25 的 bf16 池噪声同量级、2.3× 裕量）。
    // 峰值门 3e-3 vs reference **不可满足**：reference 每 token 落池收 bf16，
    // 其自身与 f32 锚的峰值差（上行实测 6.29e-3）已越过 3e-3——任何非逐
    // token 位镜像的实现（含精确 f32 计算）都无法在峰值度量上进 3e-3；
    // 本路 vs reference 实测峰 ≈ 锚噪声地板 + 路径单 ULP。峰值判据改锚到
    // f32 真值：≤ 4e-3 = 单个 bf16 ULP（2^-8）——分块路径自身与精确值至多
    // 差一次输出舍入（实测 rel2 1.12e-4，比老路对锚近 10×）；vs reference
    // 峰值 ≤ 8e-3 钉上界（实测 ~1.6 ULP）
    assert!(rel2_cr_o <= 3.0e-3, "o35 {tag} 输出 rel2 越门：{rel2_cr_o:.3e} > 3e-3");
    assert!(rel2_cr_s <= 3.0e-3, "o35 {tag} 末态 rel2 越门：{rel2_cr_s:.3e} > 3e-3");
    // 锚门 = kernel 自身预算（max 幅值处 1 bf16 ULP = 4e-3）+ 锚实测噪声地板
    // （锚=全程 f32 只末尾收一次，对逐 token 收 bf16 的 reference 自身可差
    // ~1.6 ULP——实测离群元素 got≡ref、锚独偏，见上方离群打印）。
    assert!(peak_ce_o <= 4.0e-3 + peak_re_o, "o35 {tag} 输出 vs f32 锚峰值越门：{peak_ce_o:.3e} > {max:.3e}（超 1 ULP+锚噪声——查代数）", max = 4.0e-3 + peak_re_o);
    assert!(peak_ce_s <= 4.0e-3, "o35 {tag} 末态 vs f32 锚峰值越门：{peak_ce_s:.3e} > 4e-3（末态池单次 bf16 收的量子）");
    assert!(peak_cr_o <= 8.0e-3, "o35 {tag} 输出 vs reference 峰值越门：{peak_cr_o:.3e} > 8e-3（超舍入预算——查代数）");
    assert!(peak_cr_s <= 8.0e-3, "o35 {tag} 末态 vs reference 峰值越门：{peak_cr_s:.3e} > 8e-3");
}

#[test]
fn o35_deltanet_chunked_tolerance() {
    // 单行 500 token（4 块尾 500%64=52）× BT 三档；vsplit 扫 1/4/8
    o35_case(2, 4, &[0], &[500], 32, 1);
    o35_case(2, 4, &[0], &[500], 64, 4);
    o35_case(2, 4, &[0], &[500], 128, 8);
    // ② smem staged 专路（生产几何 bt64/sp8：vrows=16、bdim=128、VW=4）
    o35_case(2, 4, &[0], &[64], 64, 8);
    o35_case(2, 4, &[0], &[128], 64, 8);
    o35_case(2, 4, &[0], &[500], 64, 8);
    o35_case(2, 4, &[0, 300], &[300, 133], 64, 8);
    // 双行 300+133（非整块尾 ×2、行间状态独立、snap_off 前缀）
    o35_case(1, 2, &[0, 300], &[300, 133], 64, 8);
}

/// o36 · 三融合算子 vs 两发分离位级对拍（WEIN_FUSED 接线的位级合同）：
/// add_norm_zc / silu_mul / dn_split_qkv 与旧 kernel 序列逐位 ==
#[test]
fn o36_fused_ops_bitwise() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);

    // ── add_norm_zc vs add → rms_norm_zc（含 hidden<bdim 的闲线程树）──
    for (rows, hidden) in [(3usize, 65usize), (17, 256), (29, 2560)] {
        let n = rows * hidden;
        let a = vec_bf16(0x1101, n, 3.0);
        let b = vec_bf16(0x2202, n, 3.0);
        let w = vec_bf16(0x3303, hidden, 0.5);
        let (da, db, dw) = (
            gpu.alloc_from_slice(&a),
            gpu.alloc_from_slice(&b),
            gpu.alloc_from_slice(&w),
        );
        let dsum = gpu.alloc_bytes(n * 2);
        t.add.run(&da, &db, &dsum, n as u32, &stream);
        let dnorm = gpu.alloc_bytes(n * 2);
        t.rms_norm_zc.run(&dsum, &dw, &dnorm, rows as u32, hidden as u32, 1e-6, &stream);
        let (dsum2, dnorm2) = (gpu.alloc_bytes(n * 2), gpu.alloc_bytes(n * 2));
        t.add_norm_zc.run(
            &da, &db, &dw, &dsum2, &dnorm2, rows as u32, hidden as u32, 1e-6, &stream,
        );
        stream.sync();
        assert_bits(
            &format!("add_norm_zc sum {rows}x{hidden}"),
            &dsum2.read_to_vec(),
            &dsum.read_to_vec(),
        );
        assert_bits(
            &format!("add_norm_zc norm {rows}x{hidden}"),
            &dnorm2.read_to_vec(),
            &dnorm.read_to_vec(),
        );
    }

    // ── silu_mul vs silu → mul（act 的 bf16 中间舍入两侧同在）──
    for n in [1000usize, 9216 * 3] {
        let g = vec_bf16(0x4404, n, 3.0);
        let u = vec_bf16(0x5505, n, 2.0);
        let (dg, du) = (gpu.alloc_from_slice(&g), gpu.alloc_from_slice(&u));
        let dact = gpu.alloc_bytes(n * 2);
        t.silu.run(&dg, &dact, n as u32, &stream);
        let dout = gpu.alloc_bytes(n * 2);
        t.mul.run(&dact, &du, &dout, n as u32, &stream);
        let dout2 = gpu.alloc_bytes(n * 2);
        t.silu_mul.run(&dg, &du, &dout2, n as u32, &stream);
        stream.sync();
        assert_bits(&format!("silu_mul n={n}"), &dout2.read_to_vec(), &dout.read_to_vec());
    }

    // ── dn_split_qkv vs 3× embed gather（exec 同款下标表；vd=2·kd 档）──
    for (tokens, kd, vd) in [(4usize, 256usize, 512usize), (3, 2048, 4096)] {
        let cd = 2 * kd + vd;
        let src = vec_bf16(0x6606, tokens * cd, 4.0);
        let ilq: Vec<u32> = (0..tokens).map(|tt| (tt * 4) as u32).collect();
        let ilk: Vec<u32> = (0..tokens).map(|tt| (tt * 4 + 1) as u32).collect();
        let ilv: Vec<u32> = (0..tokens).map(|tt| (tt * 2 + 1) as u32).collect();
        let dsrc = gpu.alloc_from_slice(&src);
        let (dilq, dilk, dilv) = (
            gpu.alloc_from_slice(&ilq),
            gpu.alloc_from_slice(&ilk),
            gpu.alloc_from_slice(&ilv),
        );
        let (dq, dk, dv) = (
            gpu.alloc_bytes(tokens * kd * 2),
            gpu.alloc_bytes(tokens * kd * 2),
            gpu.alloc_bytes(tokens * vd * 2),
        );
        t.embed.run(&dilq, &dsrc, &dq, kd as u32, tokens as u32, &stream);
        t.embed.run(&dilk, &dsrc, &dk, kd as u32, tokens as u32, &stream);
        t.embed.run(&dilv, &dsrc, &dv, vd as u32, tokens as u32, &stream);
        let (dq2, dk2, dv2) = (
            gpu.alloc_bytes(tokens * kd * 2),
            gpu.alloc_bytes(tokens * kd * 2),
            gpu.alloc_bytes(tokens * vd * 2),
        );
        t.dn_split_qkv.run(
            &dsrc, &dq2, &dk2, &dv2, kd as u32, vd as u32, tokens as u32, &stream,
        );
        stream.sync();
        assert_bits(
            &format!("dn_split_qkv q t={tokens} kd={kd}"),
            &dq2.read_to_vec(),
            &dq.read_to_vec(),
        );
        assert_bits(
            &format!("dn_split_qkv k t={tokens} kd={kd}"),
            &dk2.read_to_vec(),
            &dk.read_to_vec(),
        );
        assert_bits(
            &format!("dn_split_qkv v t={tokens} vd={vd}"),
            &dv2.read_to_vec(),
            &dv.read_to_vec(),
        );
    }

    // ── conv1d_dw_roll（合一档）vs conv1d_dw → conv_state_roll（两发档）──
    // 纯 decode 段形：每段恰 1 token（row_count 全 1、specs 全 0）。逐位比
    // **输出**与**滚动后的状态**；状态池按 row_id 非平凡间接（打乱行序，
    // 若融合版拿 token 序号当行号写会立刻现形）。
    for (tokens, conv_dim, kernel) in [
        (7usize, 16usize, 4usize),
        (5, 128, 4),
        (4, 96, 2), // kernel=2：st=1（窗口只 1 槽，边界读 j+1 不存在）
        (3, 64, 3),
    ] {
        let st = kernel - 1;
        let pool_rows = tokens + 2;
        // 行号是**非本征排列**（倒序），但每段必须独占一行——纯 decode 的
        // 段形（executor 一请求一行）。重复行号在两发档里本就是两块写同一槽
        // 的未定义序，不属本 kernel 的语义域
        let row_id: Vec<u32> = (0..tokens as u32).map(|i| (pool_rows - 1 - i as usize) as u32).collect();
        let row_start: Vec<u32> = (0..tokens as u32).collect();
        let row_count: Vec<u32> = vec![1; tokens];
        let specs0: Vec<u32> = vec![0; tokens];
        let x = vec_bf16(0x7707, tokens * conv_dim, 2.0);
        let w = vec_bf16(0x8808, conv_dim * kernel, 0.5);
        let st_old = vec_bf16(0x9909, pool_rows * conv_dim * st, 2.0);

        // 两发档
        let (dx, dw, ds1, dtrow) = (
            gpu.alloc_from_slice(&x),
            gpu.alloc_from_slice(&w),
            gpu.alloc_from_slice(&st_old),
            gpu.alloc_from_slice(&row_id),
        );
        let dtp = gpu.alloc_from_slice(&vec![0u32; tokens]);
        let dout1 = gpu.alloc_bytes(tokens * conv_dim * 2);
        t.conv1d_dw.run(&dx, &dw, &ds1, &dout1, &dtrow, &dtp, conv_dim as u32, kernel as u32, tokens as u32, &stream);
        let (drs, drc, dri, dsp) = (
            gpu.alloc_from_slice(&row_start),
            gpu.alloc_from_slice(&row_count),
            gpu.alloc_from_slice(&row_id),
            gpu.alloc_from_slice(&specs0),
        );
        t.conv_state_roll.run(&dx, &ds1, &drs, &drc, &dri, conv_dim as u32, kernel as u32, tokens as u32, &dsp, &stream);

        // 合一档
        let ds2 = gpu.alloc_from_slice(&st_old);
        let dout2 = gpu.alloc_bytes(tokens * conv_dim * 2);
        t.conv1d_dw_roll.run(&dx, &dw, &ds2, &dout2, &dtrow, conv_dim as u32, kernel as u32, tokens as u32, &stream);
        stream.sync();
        let tag = format!("conv1d_dw_roll t={tokens} cd={conv_dim} k={kernel}");
        assert_bits(&format!("{tag} out"), &dout2.read_to_vec(), &dout1.read_to_vec());
        assert_bits(&format!("{tag} state"), &ds2.read_to_vec(), &ds1.read_to_vec());
        // 真值锚：结果同时要与 reference 两条一致（不是两个错的对齐）
        let expect_conv = reference::conv1d_dw(&x, &w, &st_old, &row_id, &vec![0u32; tokens], conv_dim, kernel);
        assert_bits(&format!("{tag} out vs reference"), &dout2.read_to_vec(), &expect_conv);
        let expect_state =
            reference::conv_state_roll(&x, &st_old, &row_start, &row_count, &row_id, conv_dim, kernel, &specs0);
        assert_bits(&format!("{tag} state vs reference"), &ds2.read_to_vec(), &expect_state);
    }
}

/// o39 · 批量行零化（`zero_rows`）：表里每个池的第 `row` 行整行清零、其余行
/// **逐字节不动**。三池行距/行长各异（4096/1024/512 字节），nblk=1 与 4 两档
/// 都验——写序无关（全零），终态与逐池 `cuMemsetD8Async` 相同，故可整替代
/// 行回收/领用的 48 发零化。
#[test]
fn o39_zero_rows_bitwise() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let cfg = [(4096usize, 4usize), (1024, 4), (512, 4)];
    let mut pats: Vec<Vec<u64>> = Vec::new();
    let mut pools = Vec::new();
    for (i, (stride, rows)) in cfg.iter().enumerate() {
        let n = stride * rows / 8;
        let pat: Vec<u64> =
            (0..n).map(|j| 0xdead_0000_0000_0000u64 ^ ((j as u64) << 1) ^ i as u64).collect();
        pools.push(gpu.alloc_from_slice(&pat));
        pats.push(pat);
    }
    let mut tbl: Vec<u64> = Vec::new();
    for (p, (stride, _)) in pools.iter().zip(cfg.iter()) {
        tbl.push(p.device_ptr());
        tbl.push(*stride as u64 | ((*stride as u64) << 32));
    }
    let tbl_buf = gpu.alloc_from_slice(&tbl);
    for nblk in [1u32, 4] {
        for (p, pat) in pools.iter().zip(pats.iter()) {
            p.upload_on(pat, &stream);
        }
        t.zero_rows.run(&tbl_buf, 2, cfg.len() as u32, nblk, &stream);
        stream.sync();
        for (i, (stride, rows)) in cfg.iter().enumerate() {
            let back: Vec<u64> = pools[i].read_to_vec();
            let per = stride / 8;
            for r in 0..*rows {
                for j in 0..per {
                    let got = back[r * per + j];
                    if r == 2 {
                        assert_eq!(got, 0, "nblk={nblk} 池{i} 第 2 行未清零 @{j}");
                    } else {
                        assert_eq!(got, pats[i][r * per + j], "nblk={nblk} 池{i} 行 {r} 被误写 @{j}");
                    }
                }
            }
        }
    }
}
