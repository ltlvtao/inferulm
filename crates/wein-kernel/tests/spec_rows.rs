//! 投机验证链的逐位置 state 行（MTP 基础原语）——kernel 级位对齐验收。
//!
//! 合同（与 `wein-exec::Qwen35Executor::spec_verify` 同构）：一条草稿链占连续
//! state 行 base..=base+cnt，第 i 个 token 读行 base+i、写行 base+i+1
//! （deltanet/conv 的 `spec==1`）。于是 cnt 个 token 跑完后行 base+j 就是
//! "只跑了前 j 个 token"的递推状态——这就是零拷贝回滚的位级判据：
//!
//! - o26 · conv_state_roll：行 base+j ≡ 独立跑前 j 个 token（spec==0，从
//!   base 起算、前 j 个 token 的输入）；spec==0 与 reference 仍位对齐
//! - o27 · deltanet（kd=vd=128 E2 路径）：同上（state 行 + 前 j 个 token 的
//!   core_out 都比），spec==0 与 reference 仍位对齐
//!
//! 全部 GPU + 串行锁门控（与 ops_i1 同式：一次只跑一个进程）。
use wein_kernel::cuda::Gpu;
use wein_kernel::ops::{ConvStateRollOp, DeltanetOp, OpTable};
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

fn guard() -> std::sync::MutexGuard<'static, ()> {
    static L: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    L.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

fn table(gpu: &Gpu) -> OpTable<'_> {
    OpTable::resolve(gpu, &HardwareProfile::with_device(Device::Cuda).with_arch("sm_89"))
        .expect("sm_89 画像须解析成功")
}

fn assert_bits(label: &str, got: &[u16], expect: &[u16]) {
    assert_eq!(got.len(), expect.len(), "{label} 长度");
    for i in 0..expect.len() {
        assert_eq!(
            got[i], expect[i],
            "{label}[{i}] 位不对齐（GPU {:#06x} vs 参考 {:#06x}）",
            got[i], expect[i]
        );
    }
}

fn vec_bf16(seed_hi: u64, n: usize, scale: f32) -> Vec<u16> {
    (0..n as u64)
        .map(|i| reference::f32_to_bf16(reference::splitmix_f32(i ^ seed_hi) * scale))
        .collect()
}

/// 行切片（bf16 位）
fn row(v: &[u16], r: usize, row_len: usize) -> &[u16] {
    &v[r * row_len..(r + 1) * row_len]
}

/// o26 · conv 逐位置行：第 i 个 token 的窗口落行 base+i+1；行 base+j ≡ 独立
/// 跑前 j 个 token 的结果。spec==0 与 reference 位对齐不变（回归锚）
#[test]
fn o26_conv_state_roll_spec_rows() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let (cd, kernel, cnt, base) = (48usize, 4usize, 3usize, 1u32);
    let st = kernel - 1;
    let pool_rows = cnt + 3;
    let x = vec_bf16(0xa11ce, cnt * cd, 2.0);
    let init = vec_bf16(0xb0b, pool_rows * cd * st, 2.0);
    let (dx, dstate) = (gpu.alloc_from_slice(&x), gpu.alloc_from_slice(&init));
    let (drs, drc, drid) = (
        gpu.alloc_from_slice(&[0u32]),
        gpu.alloc_from_slice(&[cnt as u32]),
        gpu.alloc_from_slice(&[base]),
    );

    // spec==0 回归：单段滚动 ≡ reference
    let dspec0 = gpu.alloc_from_slice(&[0u32]);
    t.conv_state_roll.run(&dx, &dstate, &drs, &drc, &drid, cd as u32, kernel as u32, 1, &dspec0, &stream);
    stream.sync();
    let got0: Vec<u16> = dstate.read_to_vec();
    let want0 =
        reference::conv_state_roll(&x, &init, &[0], &[cnt as u32], &[base], cd, kernel, &[0]);
    assert_bits("conv_state_roll spec==0", &got0, &want0);

    // spec==1：逐位置落盘
    let dstate1 = gpu.alloc_from_slice(&init);
    let dspec1 = gpu.alloc_from_slice(&[1u32]);
    t.conv_state_roll.run(&dx, &dstate1, &drs, &drc, &drid, cd as u32, kernel as u32, 1, &dspec1, &stream);
    stream.sync();
    let got: Vec<u16> = dstate1.read_to_vec();
    let rl = cd * st;
    for j in 1..=cnt {
        let want = reference::conv_state_roll(
            &x, &init, &[0], &[j as u32], &[base], cd, kernel, &[0],
        );
        assert_bits(
            &format!("conv spec 行 base+{j} ≡ 前 {j} token 的独立结果"),
            row(&got, base as usize + j, rl),
            row(&want, base as usize, rl),
        );
    }
    // 链首行（旧状态）不被覆写：deltanet/conv 的读行语义依赖它
    assert_bits("conv 链首行保持旧状态", row(&got, base as usize, rl), row(&init, base as usize, rl));
}

/// o27 · deltanet 逐位置行（E2 kd=vd=128）：行 base+j ≡ 独立跑前 j 个 token
/// 的递推状态；前 j 个 token 的 core_out 亦相同；spec==0 位对齐不变
#[test]
fn o27_deltanet_spec_rows() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let (kh, vh, kd, vd, cnt, base) = (2usize, 2usize, 128usize, 128usize, 3usize, 1u32);
    let pool_rows = cnt + 3;
    let eps = 1e-6f32;
    let (q, k, v) = (
        vec_bf16(0x11, cnt * kh * kd, 2.0),
        vec_bf16(0x22, cnt * kh * kd, 2.0),
        vec_bf16(0x33, cnt * vh * vd, 2.0),
    );
    let (a, b) = (vec_bf16(0x44, cnt * vh, 1.0), vec_bf16(0x55, cnt * vh, 1.0));
    let a_log: Vec<f32> = (0..vh as u64)
        .map(|i| reference::splitmix_f32(i ^ 0x66) * 0.5 - 0.25)
        .collect();
    let dt_bias = vec_bf16(0x77, vh, 0.5);
    let init = vec_bf16(0x8888, pool_rows * vh * vd * kd, 0.5);
    let row_len = vh * vd * kd;

    let (dq, dk, dv) = (
        gpu.alloc_from_slice(&q),
        gpu.alloc_from_slice(&k),
        gpu.alloc_from_slice(&v),
    );
    let (da, db) = (gpu.alloc_from_slice(&a), gpu.alloc_from_slice(&b));
    let (da_log, ddtb) = (gpu.alloc_from_slice(&a_log), gpu.alloc_from_slice(&dt_bias));
    let (drs, drc, drid) = (
        gpu.alloc_from_slice(&[0u32]),
        gpu.alloc_from_slice(&[cnt as u32]),
        gpu.alloc_from_slice(&[base]),
    );
    let dout = gpu.alloc_bytes(cnt * vh * vd * 2);
    let launch = |dstate: &wein_kernel::cuda::DevBuf,
                  specs: &wein_kernel::cuda::DevBuf,
                  split: u32| {
        t.deltanet.run(
            &dq, &dk, &dv, &da, &db, &da_log, &ddtb, dstate, &dout, &drs, &drc, &drid, 1,
            kh as u32, vh as u32, kd as u32, vd as u32, eps, specs, split, &stream,
        );
    };

    // spec==0 回归：单段单行递推 ≡ reference（位对齐合同）
    let ds0 = gpu.alloc_from_slice(&init);
    let dspec0 = gpu.alloc_from_slice(&[0u32]);
    launch(&ds0, &dspec0, 1);
    stream.sync();
    let got0: Vec<u16> = ds0.read_to_vec();
    let out0: Vec<u16> = dout.read_to_vec();
    let mut st_ref = init.clone();
    let want0 = reference::deltanet(
        &q, &k, &v, &a, &b, &a_log, &dt_bias, &mut st_ref, &[0], &[cnt as u32], &[base], kh, vh,
        kd, vd, eps, &[0],
    );
    assert_bits("deltanet spec==0 state", &got0, &st_ref);
    assert_bits("deltanet spec==0 core_out", &out0, &want0);

    // spec==1：逐位置行 —— 行 base+j ≡ 前 j 个 token 的独立结果
    let ds1 = gpu.alloc_from_slice(&init);
    let dspec1 = gpu.alloc_from_slice(&[1u32]);
    launch(&ds1, &dspec1, 1);
    stream.sync();
    let got: Vec<u16> = ds1.read_to_vec();
    let out1: Vec<u16> = dout.read_to_vec();
    for j in 1..=cnt {
        let mut st_j = init.clone();
        let out_j = reference::deltanet(
            &q, &k, &v, &a, &b, &a_log, &dt_bias, &mut st_j, &[0], &[j as u32], &[base], kh, vh,
            kd, vd, eps, &[0],
        );
        assert_bits(
            &format!("deltanet spec 行 base+{j} ≡ 前 {j} token 的独立结果"),
            row(&got, base as usize + j, row_len),
            row(&st_j, base as usize, row_len),
        );
        let per_tok = vh * vd;
        assert_bits(
            &format!("deltanet spec 前 {j} token 的 core_out"),
            &out1[..j * per_tok],
            &out_j[..j * per_tok],
        );
    }
    // B 档 chunked 不变性：同一 spec==1 形状按 v 行切 4 块 ⇒ 逐位同上
    let ds1c = gpu.alloc_from_slice(&init);
    launch(&ds1c, &dspec1, 4);
    stream.sync();
    assert_bits("deltanet spec==1 切 4 块 ≡ 不切块", &ds1c.read_to_vec(), &got);
    // 链首行仍是旧状态（spec 分支读它、不写它）
    assert_bits(
        "deltanet 链首行保持旧状态",
        row(&got, base as usize, row_len),
        row(&init, base as usize, row_len),
    );
}
