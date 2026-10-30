//! M0 算子清单验收（02-architecture §4）——rms_norm 之外五个算子逐个入表。
//!
//! 每算子同套路：trait 家族 + 具体类型槽位 + CPU 参考（独立代码路径）+
//! golden 位对齐。数值对齐纪律与 ops.rs 同守（k6/o4 同源）：只正确舍入
//! 原语、行内顺序累加、bf16→f32 精确位左移；exp 类函数**不走 MUFU 近似**
//! ——exp2 多项式（自控系数，两侧同字面量同 Horner 序）是可达位对齐的
//! A 计划答案（与 bf16 软实现同理：不指望工具链 intrinsic）。
//!
//! - o7 · 注册表长全：resolve 一次 → 六槽位齐（rms + 5 新算子），名字可审
//! - o8 · rope：cos/sin 表为输入（host 生成上传，图捕获友好），half-split
//!        （Llama/Qwen 形态），位置索引含空隙（证明表寻址正确）
//! - o9 · silu：sigmoid 用自控 exp2 路径；n 含非 block 整数倍（4097）
//! - o10 · W8A8 动态量化：逐 token amax → scale=amax/127 → RNE 半偶取整
//!        （自控 RNE，非默认 round），i8 + f32 scale 双输出位对齐
//! - o11 · KV scatter write：页表寻址（block_table + pos 跨块）写入
//!        paged KV；未触及槽位原样保留
//! - o12 · varlen paged attention：GQA（qh//group）、pos 0 只看自身
//!        （单点 softmax 恒等 v —— 独立真值校验）、33 长跨 3 块；
//!        页槽寻址与 wein-memory plan_kv 的 token 记账同构
//!        （[page][block_size][heads][head_dim]）
//! - o13 · golden 文件框架扩展：每算子一文件，缺失生成/回读对拍/
//!        与参考漂移须显式重生成（asc 接入复用同一文件，§4）
#![allow(dead_code)]
//!
//! M0 档全部一维几何、线程间无归约——launch 几何不影响数值序（o5 已证）；
//! W2 flash 分块 / 多头并行时 golden 不变，重对拍即可。
use wein_kernel::cuda::Gpu;
use wein_kernel::ops::{
    AddOp, EmbedOp, GemmOp, KvScatterOp, MulOp, OpTable, PagedAttnOp, RopeOp,
    RmsNormOp, SiluOp, W8A8QuantOp,
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

/// GPU 串行锁（ops_i1/graphs 同款）：cargo test 并行线程的 kernel 会互相
/// 闯入事件计时窗——时延金丝雀（o24）对并发负载敏感，全文件持锁
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

/// o7 · 注册表长全：六槽位齐、名字可审（档位标识入日志）
#[test]
fn o7_op_table_all_m0_slots() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let t = table(&gpu);
    assert_eq!(t.rms_norm.name(), "rms_norm/cuda-blocked");
    assert_eq!(t.rope.name(), "rope/cuda");
    assert_eq!(t.silu.name(), "silu/cuda");
    assert_eq!(t.w8a8_quant.name(), "w8a8_quant/cuda");
    assert_eq!(t.kv_scatter.name(), "kv_scatter/cuda");
    assert_eq!(t.paged_attn.name(), "paged_attn/cuda");
    assert_eq!(t.gemm.name(), "gemm/cuda-lt");
    assert_eq!(t.embed.name(), "embed/cuda");
    assert_eq!(t.add.name(), "add/cuda");
    assert_eq!(t.mul.name(), "mul/cuda");
}

/// o8 · rope 位对齐：half-split（Llama/Qwen），位置含空隙（表寻址真值）
#[test]
fn o8_rope_golden_bitwise() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    // (tokens, rot_dim, max_pos)：b 组 pos 稀疏——证明 cos/sin 表按 pos 索引
    for (tokens, rot_dim, max_pos) in [(8, 64, 64), (17, 128, 200)] {
        let pos: Vec<u32> = (0..tokens).map(|i| (i * 37 + 11) as u32 % max_pos as u32).collect();
        let (cos, sin) = reference::rope_table(max_pos, rot_dim / 2, 10000.0);
        let n = tokens * rot_dim;
        let q: Vec<u16> = (0..n as u64).map(|i| reference::f32_to_bf16(reference::splitmix_f32(i) * 4.0)).collect();
        let k: Vec<u16> = (0..n as u64).map(|i| reference::f32_to_bf16(reference::splitmix_f32(i ^ 0xdead_beef) * 4.0)).collect();
        let (eq, ek) = reference::rope(&q, &k, &cos, &sin, &pos, rot_dim);

        let (dq, dk) = (gpu.alloc_from_slice(&q), gpu.alloc_from_slice(&k));
        let (dc, ds) = (gpu.alloc_from_slice(&cos), gpu.alloc_from_slice(&sin));
        let dp = gpu.alloc_from_slice(&pos);
        let (oq, ok) = (gpu.alloc_bytes(n * 2), gpu.alloc_bytes(n * 2));
        t.rope.run(&dq, &dk, &dc, &ds, &dp, &oq, &ok, tokens as u32, rot_dim as u32, &stream);
        stream.sync();
        assert_bits(&format!("rope q {tokens}x{rot_dim}"), &oq.read_to_vec(), &eq);
        assert_bits(&format!("rope k {tokens}x{rot_dim}"), &ok.read_to_vec(), &ek);
    }
}

/// o9 · silu 位对齐：自控 exp2 路径；n 含非 block 整数倍
#[test]
fn o9_silu_golden_bitwise() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    for n in [512usize, 4097] {
        let x: Vec<u16> = (0..n as u64).map(|i| reference::f32_to_bf16(reference::splitmix_f32(i) * 4.0)).collect();
        let expect = reference::silu(&x);
        let (dx, dout) = (gpu.alloc_from_slice(&x), gpu.alloc_bytes(n * 2));
        t.silu.run(&dx, &dout, n as u32, &stream);
        stream.sync();
        assert_bits(&format!("silu n={n}"), &dout.read_to_vec(), &expect);
    }
}

/// o10 · W8A8 动态量化位对齐：i8 + f32 scale 双输出；RNE 半偶自控实现
#[test]
fn o10_w8a8_quant_golden_bitwise() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    for (tokens, hidden) in [(1usize, 64), (33, 256)] {
        let n = tokens * hidden;
        let x: Vec<u16> = (0..n as u64).map(|i| reference::f32_to_bf16(reference::splitmix_f32(i) * 4.0)).collect();
        let (eq, es) = reference::w8a8_quant(&x, tokens, hidden);

        let dx = gpu.alloc_from_slice(&x);
        let (dq, ds) = (gpu.alloc_bytes(n), gpu.alloc_bytes(tokens * 4));
        t.w8a8_quant.run(&dx, &dq, &ds, tokens as u32, hidden as u32, &stream);
        stream.sync();
        let (gq, gs): (Vec<i8>, Vec<f32>) = (dq.read_to_vec(), ds.read_to_vec());

        assert_eq!(gq.len(), eq.len(), "q 长度 {tokens}x{hidden}");
        for i in 0..eq.len() {
            assert_eq!(gq[i], eq[i], "q[{i}]（GPU {} vs 参考 {}）", gq[i], eq[i]);
        }
        for i in 0..es.len() {
            assert_eq!(gs[i], es[i], "scale[{i}]（GPU {:e} vs 参考 {:e}）", gs[i], es[i]);
        }
        // 语义真值（非位对齐侧）：amax 元素必映射 ±127；反量化误差 ≤ scale
        for r in 0..tokens {
            assert_eq!(gs[r].abs(), gs[r].abs());
            let amax = (0..hidden).map(|c| reference::bf16_to_f32(x[r * hidden + c]).abs())
                .fold(0.0f32, f32::max);
            assert_eq!(es[r], amax / 127.0, "scale[{r}] 应为 amax/127");
            for c in 0..hidden {
                let v = reference::bf16_to_f32(x[r * hidden + c]);
                let err = (v - gq[r * hidden + c] as f32 * gs[r]).abs();
                assert!(err <= es[r], "[{r}][{c}] 反量化误差 {err} 超 scale {}", es[r]);
            }
        }
    }
}

/// o11 · KV scatter 位对齐：页表寻址（pos 跨块边界 15/16/17），未触及槽位保留
#[test]
fn o11_kv_scatter_golden_bitwise() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    // block_size 16（CUDA 档，§13-3）；bpr=4 页表行宽
    let (block_size, bpr, head_elems) = (16usize, 4usize, 64usize);
    // s0：pos 0,1,15,16,17（跨块）；s1：pos 3,20（跨块）
    let seq_ids: Vec<u32> = vec![0, 0, 0, 0, 0, 1, 1];
    let positions: Vec<u32> = vec![0, 1, 15, 16, 17, 3, 20];
    let tokens = seq_ids.len();
    // 页表：s0 块 {5,9,3}；s1 块 {7,1}——故意乱序（页池分配不连续是常态）
    let block_table: Vec<u32> = vec![5, 9, 3, 0, 7, 1, 0, 0];
    let n = tokens * head_elems;
    let sk: Vec<u16> = (0..n as u64).map(|i| reference::f32_to_bf16(reference::splitmix_f32(i))).collect();
    let sv: Vec<u16> = (0..n as u64).map(|i| reference::f32_to_bf16(reference::splitmix_f32(i ^ 0x517c_c1b3))).collect();
    let slots = 10 * block_size; // 最大块 id 9 → 10 块
    let (ek, ev) = reference::kv_scatter(&sk, &sv, &block_table, &seq_ids, &positions, slots, bpr, block_size, head_elems);
    // 部分写算子的对拍前提：dst 先零初始化上传（cuMemAlloc 不清零，
    // 未触及槽位是垃圾——"没写坏别人的槽"语义必须在确定的底色上看）
    let zero = || vec![0u16; slots * head_elems];
    // 块 0 不在页表有效列——槽 0 必然未被触及
    assert_eq!(ek[0], 0, "槽 0 应未被写");

    let (dsk, dsv) = (gpu.alloc_from_slice(&sk), gpu.alloc_from_slice(&sv));
    let (dbt, dsq, dpo) = (gpu.alloc_from_slice(&block_table), gpu.alloc_from_slice(&seq_ids), gpu.alloc_from_slice(&positions));
    let (dk, dv) = (gpu.alloc_from_slice(&zero()), gpu.alloc_from_slice(&zero()));
    t.kv_scatter.run(&dsk, &dsv, &dbt, &dsq, &dpo, &dk, &dv, bpr as u32, block_size as u32, head_elems as u32, tokens as u32, &stream);
    stream.sync();
    assert_bits("kv_scatter k", &dk.read_to_vec(), &ek);
    assert_bits("kv_scatter v", &dv.read_to_vec(), &ev);
}

/// o12 · varlen paged attention 位对齐：GQA、单点 softmax 恒等、跨块
#[test]
fn o12_paged_attn_golden_bitwise() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    // 3 序列：s0 len 5（1 query，pos 4）；s1 len 33（2 query：pos 0 与 32）
    // —— pos 0 只看自身（softmax 单点 = 恒等）；33 长跨 3 块
    let (block_size, bpr) = (16usize, 3usize);
    // 非平凡 GQA（group=4）：qh//group 与 qh%kv_heads 在头 1,3,4,6 分叉——
    // （2,1) 的 MQA 形状两映射恒等，钉不住头映射（vs vLLM 对拍的教训）
    let (q_heads, kv_heads, head_dim) = (8usize, 2usize, 64usize);
    let q_scale: f32 = 0.125; // 1/sqrt(64) 恰为 2 的幂——输入侧不给舍入留余地
    let seq_ids: Vec<u32> = vec![0, 1, 1];
    let positions: Vec<u32> = vec![4, 0, 32];
    let tokens = seq_ids.len();
    let block_table: Vec<u32> = vec![4, 0, 0, 9, 2, 6];
    let slots = 10 * block_size;
    let he = kv_heads * head_dim;
    let slot_elems = slots * he;

    let k_cache: Vec<u16> = (0..slot_elems as u64).map(|i| reference::f32_to_bf16(reference::splitmix_f32(i ^ 0x9e37_79b9))).collect();
    let v_cache: Vec<u16> = (0..slot_elems as u64).map(|i| reference::f32_to_bf16(reference::splitmix_f32(i ^ 0xbf58_476d))).collect();
    let qn = tokens * q_heads * head_dim;
    let q: Vec<u16> = (0..qn as u64).map(|i| reference::f32_to_bf16(reference::splitmix_f32(i) * 2.0)).collect();

    let expect = reference::paged_attn(&q, &k_cache, &v_cache, &block_table, &seq_ids, &positions, q_scale, bpr, block_size, q_heads, kv_heads, head_dim);

    let (dq, dkc, dvc) = (gpu.alloc_from_slice(&q), gpu.alloc_from_slice(&k_cache), gpu.alloc_from_slice(&v_cache));
    let (dbt, dsi, dpo) = (gpu.alloc_from_slice(&block_table), gpu.alloc_from_slice(&seq_ids), gpu.alloc_from_slice(&positions));
    let dout = gpu.alloc_bytes(qn * 2);
    t.paged_attn.run(&dq, &dkc, &dvc, &dbt, &dsi, &dpo, &dout, q_scale,
        bpr as u32, block_size as u32, q_heads as u32, kv_heads as u32, head_dim as u32, tokens as u32, &stream);
    stream.sync();
    assert_bits("paged_attn", &dout.read_to_vec(), &expect);

    // 独立真值：pos 0 的 query，softmax 单点 → out 恒等于 v[slot0]（bf16 原样）。
    // kv 头 = h // group（HF repeat_kv 语义：kv 头 k 服务 q 头 [k*group,(k+1)*group)）——
    // 此断言由 softmax 恒等 + GQA 语义直接构造，不经过 reference 公式，
    // 参考侧映射错（qh%kv_heads）时必红
    let s1_q0 = 1; // seq 1 第一个 query（token 下标 1）
    let group = q_heads / kv_heads;
    for h in 0..q_heads {
        let kh = h / group;
        for e in 0..head_dim {
            let slot = block_table[1 * bpr] * block_size as u32; // seq1 块 9，pos 0
            let v = v_cache[(slot as usize * he) + kh * head_dim + e];
            let got = expect[(s1_q0 * q_heads + h) * head_dim + e];
            assert_eq!(got, v, "单点 softmax 应恒等 v[kv 头 {kh}][{e}]（GQA = h//group）");
        }
    }
}

/// o13 · golden 文件框架扩展：每算子一文件，缺失生成/回读对拍/漂移须重生成
#[test]
fn o13_golden_files_roundtrip_m0() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);

    macro_rules! roundtrip {
        ($op:literal, $gen:expr, $run:expr, $read:expr, $eq:expr) => {{
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(concat!("tests/golden/", $op, ".json"));
            if !path.exists() {
                let file = $gen();
                std::fs::write(&path, serde_json::to_string_pretty(&file).unwrap()).unwrap();
                eprintln!("golden 首次生成：{}", path.display());
            }
            let text = std::fs::read_to_string(&path).unwrap();
            let file = serde_json::from_str::<serde_json::Value>(&text).expect("golden schema");
            assert_eq!(file["op"], $op, "文件 op 名漂移");
            let fresh = $gen();
            assert!(serde_json::to_value(&fresh).unwrap() == file,
                "golden 文件与参考实现漂移（重生成须有意为之）");
            $run(&file, &$run);
            let _ = ($read, $eq);
        }};
    }

    // 每算子：结构生成 + GPU 重跑对拍（读文件输入，跑 GPU，比对文件输出）
    // rope
    {
        let path = golden_path("rope");
        if !path.exists() {
            std::fs::write(&path, serde_json::to_string_pretty(&RopeGolden::build()).unwrap()).unwrap();
            eprintln!("golden 首次生成：{}", path.display());
        }
        let g: RopeGolden = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(g.op, "rope");
        assert_eq!(serde_json::to_value(&RopeGolden::build()).unwrap(),
            serde_json::to_value(&g).unwrap(), "rope golden 与参考漂移");
        let (dc, ds, dp) = (gpu.alloc_from_slice(&g.cos), gpu.alloc_from_slice(&g.sin), gpu.alloc_from_slice(&g.pos));
        let (dq, dk) = (gpu.alloc_from_slice(&g.q), gpu.alloc_from_slice(&g.k));
        let (oq, ok) = (gpu.alloc_bytes(g.q.len() * 2), gpu.alloc_bytes(g.k.len() * 2));
        t.rope.run(&dq, &dk, &dc, &ds, &dp, &oq, &ok, g.pos.len() as u32, g.rot_dim as u32, &stream);
        stream.sync();
        assert_bits("rope golden", &oq.read_to_vec(), &g.out_q);
        assert_bits("rope golden", &ok.read_to_vec(), &g.out_k);
    }
    // silu
    {
        let path = golden_path("silu");
        if !path.exists() {
            std::fs::write(&path, serde_json::to_string_pretty(&SiluGolden::build()).unwrap()).unwrap();
            eprintln!("golden 首次生成：{}", path.display());
        }
        let g: SiluGolden = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(g.op, "silu");
        assert_eq!(serde_json::to_value(&SiluGolden::build()).unwrap(),
            serde_json::to_value(&g).unwrap(), "silu golden 与参考漂移");
        let (dx, dout) = (gpu.alloc_from_slice(&g.x), gpu.alloc_bytes(g.x.len() * 2));
        t.silu.run(&dx, &dout, g.x.len() as u32, &stream);
        stream.sync();
        assert_bits("silu golden", &dout.read_to_vec(), &g.out);
    }
    // w8a8_quant
    {
        let path = golden_path("w8a8_quant");
        if !path.exists() {
            std::fs::write(&path, serde_json::to_string_pretty(&QuantGolden::build()).unwrap()).unwrap();
            eprintln!("golden 首次生成：{}", path.display());
        }
        let g: QuantGolden = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(g.op, "w8a8_quant");
        assert_eq!(serde_json::to_value(&QuantGolden::build()).unwrap(),
            serde_json::to_value(&g).unwrap(), "w8a8_quant golden 与参考漂移");
        let tokens = g.scale.len();
        let hidden = g.x.len() / tokens;
        let dx = gpu.alloc_from_slice(&g.x);
        let (dq, ds) = (gpu.alloc_bytes(g.x.len()), gpu.alloc_bytes(tokens * 4));
        t.w8a8_quant.run(&dx, &dq, &ds, tokens as u32, hidden as u32, &stream);
        stream.sync();
        let (q, s): (Vec<i8>, Vec<f32>) = (dq.read_to_vec(), ds.read_to_vec());
        assert_eq!(&q, &g.q, "w8a8_quant golden q 对拍失位");
        assert_eq!(&s, &g.scale, "w8a8_quant golden scale 对拍失位");
    }
    // kv_scatter
    {
        let path = golden_path("kv_scatter");
        if !path.exists() {
            std::fs::write(&path, serde_json::to_string_pretty(&ScatterGolden::build()).unwrap()).unwrap();
            eprintln!("golden 首次生成：{}", path.display());
        }
        let g: ScatterGolden = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(g.op, "kv_scatter");
        assert_eq!(serde_json::to_value(&ScatterGolden::build()).unwrap(),
            serde_json::to_value(&g).unwrap(), "kv_scatter golden 与参考漂移");
        let tokens = g.seq_ids.len();
        let (dsk, dsv) = (gpu.alloc_from_slice(&g.src_k), gpu.alloc_from_slice(&g.src_v));
        let (dbt, dsi, dpo) = (gpu.alloc_from_slice(&g.block_table), gpu.alloc_from_slice(&g.seq_ids), gpu.alloc_from_slice(&g.positions));
        // 部分写算子：dst 零初始化上传（golden 的未触及槽位是 0）
        let z = || vec![0u16; g.out_k.len()];
        let (dk, dv) = (gpu.alloc_from_slice(&z()), gpu.alloc_from_slice(&z()));
        t.kv_scatter.run(&dsk, &dsv, &dbt, &dsi, &dpo, &dk, &dv, g.bpr as u32, g.block_size as u32, g.head_elems as u32, tokens as u32, &stream);
        stream.sync();
        assert_bits("kv_scatter golden k", &dk.read_to_vec(), &g.out_k);
        assert_bits("kv_scatter golden v", &dv.read_to_vec(), &g.out_v);
    }
    // paged_attn
    {
        let path = golden_path("paged_attn");
        if !path.exists() {
            std::fs::write(&path, serde_json::to_string_pretty(&AttnGolden::build()).unwrap()).unwrap();
            eprintln!("golden 首次生成：{}", path.display());
        }
        let g: AttnGolden = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(g.op, "paged_attn");
        assert_eq!(serde_json::to_value(&AttnGolden::build()).unwrap(),
            serde_json::to_value(&g).unwrap(), "paged_attn golden 与参考漂移");
        let tokens = g.seq_ids.len();
        let (dq, dkc, dvc) = (gpu.alloc_from_slice(&g.q), gpu.alloc_from_slice(&g.k_cache), gpu.alloc_from_slice(&g.v_cache));
        let (dbt, dsi, dpo) = (gpu.alloc_from_slice(&g.block_table), gpu.alloc_from_slice(&g.seq_ids), gpu.alloc_from_slice(&g.positions));
        let dout = gpu.alloc_bytes(g.out.len() * 2);
        t.paged_attn.run(&dq, &dkc, &dvc, &dbt, &dsi, &dpo, &dout, g.q_scale,
            g.bpr as u32, g.block_size as u32, g.q_heads as u32, g.kv_heads as u32, g.head_dim as u32, tokens as u32, &stream);
        stream.sync();
        assert_bits("paged_attn golden", &dout.read_to_vec(), &g.out);
    }
}

/// o14 · exec 骨架三算子位对齐：embed gather（路由查表）/ add（残差）/
/// mul（silu×up）——全逐元素无归约，位对齐模式（自控 kernel 侧）
#[test]
fn o14_embed_add_mul_golden_bitwise() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    // embed：vocab 256 × hidden 128，取乱序 id（含重复 id——gather 纯读）
    let (vocab, hidden) = (256usize, 128usize);
    let w: Vec<u16> = (0..(vocab * hidden) as u64)
        .map(|i| reference::f32_to_bf16(reference::splitmix_f32(i)))
        .collect();
    let ids: Vec<u32> = vec![0, 5, 255, 17, 5];
    let mut expect = vec![0u16; ids.len() * hidden];
    for (t_, &id) in ids.iter().enumerate() {
        expect[t_ * hidden..(t_ + 1) * hidden].copy_from_slice(&w[id as usize * hidden..][..hidden]);
    }
    let (dw, did) = (gpu.alloc_from_slice(&w), gpu.alloc_from_slice(&ids));
    let dout = gpu.alloc_bytes(ids.len() * hidden * 2);
    t.embed.run(&did, &dw, &dout, hidden as u32, ids.len() as u32, &stream);
    stream.sync();
    assert_bits("embed", &dout.read_to_vec(), &expect);

    // add / mul：n=1000（非 block 整数倍之外再取 n=4097 一组）
    for n in [1000usize, 4097] {
        let a: Vec<u16> = (0..n as u64).map(|i| reference::f32_to_bf16(reference::splitmix_f32(i ^ 0x1111) * 3.0)).collect();
        let b: Vec<u16> = (0..n as u64).map(|i| reference::f32_to_bf16(reference::splitmix_f32(i ^ 0x2222) * 3.0)).collect();
        let (ea, em): (Vec<u16>, Vec<u16>) = (
            (0..n).map(|i| reference::f32_to_bf16(reference::bf16_to_f32(a[i]) + reference::bf16_to_f32(b[i]))).collect(),
            (0..n).map(|i| reference::f32_to_bf16(reference::bf16_to_f32(a[i]) * reference::bf16_to_f32(b[i]))).collect(),
        );
        let (da, db) = (gpu.alloc_from_slice(&a), gpu.alloc_from_slice(&b));
        let (o1, o2) = (gpu.alloc_bytes(n * 2), gpu.alloc_bytes(n * 2));
        t.add.run(&da, &db, &o1, n as u32, &stream);
        t.mul.run(&da, &db, &o2, n as u32, &stream);
        stream.sync();
        assert_bits(&format!("add n={n}"), &o1.read_to_vec(), &ea);
        assert_bits(&format!("mul n={n}"), &o2.read_to_vec(), &em);
    }
}

// ───────────────────────── golden 文件契约（o6 同式） ─────────────────────────

fn golden_path(op: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/golden/{op}.json"))
}

fn vec_bf16(seed_hi: u64, n: usize, scale: f32) -> Vec<u16> {
    (0..n as u64).map(|i| reference::f32_to_bf16(reference::splitmix_f32(i ^ seed_hi) * scale)).collect()
}

#[derive(serde::Serialize, serde::Deserialize)]
struct RopeGolden {
    op: String,
    note: String,
    rot_dim: usize,
    pos: Vec<u32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    q: Vec<u16>,
    k: Vec<u16>,
    out_q: Vec<u16>,
    out_k: Vec<u16>,
}
impl RopeGolden {
    fn build() -> Self {
        let (rot_dim, max_pos, tokens) = (128usize, 200usize, 17usize);
        let pos: Vec<u32> = (0..tokens).map(|i| (i * 37 + 11) as u32 % max_pos as u32).collect();
        let (cos, sin) = reference::rope_table(max_pos, rot_dim / 2, 10000.0);
        let (q, k) = (vec_bf16(0, tokens * rot_dim, 4.0), vec_bf16(0xdead_beef, tokens * rot_dim, 4.0));
        let (out_q, out_k) = reference::rope(&q, &k, &cos, &sin, &pos, rot_dim);
        Self { op: "rope".into(), note: NOTE.into(), rot_dim, pos, cos, sin, q, k, out_q, out_k }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SiluGolden {
    op: String,
    note: String,
    x: Vec<u16>,
    out: Vec<u16>,
}
impl SiluGolden {
    fn build() -> Self {
        let x = vec_bf16(0, 512, 4.0);
        let out = reference::silu(&x);
        Self { op: "silu".into(), note: NOTE.into(), x, out }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct QuantGolden {
    op: String,
    note: String,
    x: Vec<u16>,
    q: Vec<i8>,
    scale: Vec<f32>,
}
impl QuantGolden {
    fn build() -> Self {
        let (tokens, hidden) = (33usize, 256usize);
        let x = vec_bf16(0, tokens * hidden, 4.0);
        let (q, scale) = reference::w8a8_quant(&x, tokens, hidden);
        Self { op: "w8a8_quant".into(), note: NOTE.into(), x, q, scale }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ScatterGolden {
    op: String,
    note: String,
    bpr: usize,
    block_size: usize,
    head_elems: usize,
    block_table: Vec<u32>,
    seq_ids: Vec<u32>,
    positions: Vec<u32>,
    src_k: Vec<u16>,
    src_v: Vec<u16>,
    out_k: Vec<u16>,
    out_v: Vec<u16>,
}
impl ScatterGolden {
    fn build() -> Self {
        let (bpr, block_size, head_elems) = (4usize, 16usize, 64usize);
        let block_table = vec![5, 9, 3, 0, 7, 1, 0, 0];
        let (seq_ids, positions): (Vec<u32>, Vec<u32>) =
            (vec![0, 0, 0, 0, 0, 1, 1], vec![0, 1, 15, 16, 17, 3, 20]);
        let n = seq_ids.len() * head_elems;
        let (src_k, src_v) = (vec_bf16(0, n, 1.0), vec_bf16(0x517c_c1b3, n, 1.0));
        let (out_k, out_v) = reference::kv_scatter(&src_k, &src_v, &block_table, &seq_ids, &positions, 10 * block_size, bpr, block_size, head_elems);
        Self { op: "kv_scatter".into(), note: NOTE.into(), bpr, block_size, head_elems,
            block_table, seq_ids, positions, src_k, src_v, out_k, out_v }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct AttnGolden {
    op: String,
    note: String,
    bpr: usize,
    block_size: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    q_scale: f32,
    block_table: Vec<u32>,
    seq_ids: Vec<u32>,
    positions: Vec<u32>,
    q: Vec<u16>,
    k_cache: Vec<u16>,
    v_cache: Vec<u16>,
    out: Vec<u16>,
}
impl AttnGolden {
    fn build() -> Self {
        let (bpr, block_size, q_heads, kv_heads, head_dim) = (3usize, 16usize, 8usize, 2usize, 64usize); // 非平凡 GQA（group=4）
        let block_table = vec![4, 0, 0, 9, 2, 6];
        let (seq_ids, positions): (Vec<u32>, Vec<u32>) = (vec![0, 1, 1], vec![4, 0, 32]);
        let tokens = seq_ids.len();
        let slots = 10 * block_size;
        let q = vec_bf16(0, tokens * q_heads * head_dim, 2.0);
        let k_cache = vec_bf16(0x9e37_79b9, slots * kv_heads * head_dim, 1.0);
        let v_cache = vec_bf16(0xbf58_476d, slots * kv_heads * head_dim, 1.0);
        let out = reference::paged_attn(&q, &k_cache, &v_cache, &block_table, &seq_ids, &positions, 0.125, bpr, block_size, q_heads, kv_heads, head_dim);
        Self { op: "paged_attn".into(), note: NOTE.into(), bpr, block_size, q_heads, kv_heads, head_dim,
            q_scale: 0.125, block_table, seq_ids, positions, q, k_cache, v_cache, out }
    }
}

const NOTE: &str = "cuda/asc 共享 golden（02-architecture §4）；out 由 wein-kernel::reference 生成";

/// o24 · paged_attn 时延金丝雀：满负荷 decode 形状（tokens=100 × qh 32 /
/// kv 16 / hd 128——Qwen3.5-4B full-attn 层的工作形状；pos=23 对位
/// qwen_step_profile m=100 attn 段工作点，pos=63 深序列）。门禁按
/// 带宽地板（pos23：19.7MB≈25µs / pos63：52.4MB≈65µs）× 余量——
/// 防"一线程一 (t,qh) 串行两遍扫 + 每 e 重走页表"形态回归
#[test]
fn o24_paged_attn_latency_canary() {
    let _g = guard();
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let t = table(&gpu);
    let (q_heads, kv_heads, head_dim, block_size, tokens) =
        (32usize, 16usize, 128usize, 16usize, 100usize);
    let he = kv_heads * head_dim;
    let q_scale: f32 = 1.0 / (head_dim as f32).sqrt();
    let qn = tokens * q_heads * head_dim;
    let q = vec_bf16(0x7a11, qn, 2.0);
    let dq = gpu.alloc_from_slice(&q);
    let dout = gpu.alloc_bytes(qn * 2);
    for &pos in &[23usize, 63usize] {
        let bpr = (pos + 1).div_ceil(block_size);
        let npages = tokens * bpr;
        let slots = npages * block_size;
        // 每行独立序列（s=t），页 id 顺排——页表间接寻址真实
        let block_table: Vec<u32> = (0..npages as u32).collect();
        let seq_ids: Vec<u32> = (0..tokens as u32).collect();
        let positions: Vec<u32> = vec![pos as u32; tokens];
        let slot_elems = slots * he;
        let k_cache = vec_bf16(0x7b33, slot_elems, 1.0);
        let v_cache = vec_bf16(0x7c55, slot_elems, 1.0);
        let dkc = gpu.alloc_from_slice(&k_cache);
        let dvc = gpu.alloc_from_slice(&v_cache);
        let dbt = gpu.alloc_from_slice(&block_table);
        let dsi = gpu.alloc_from_slice(&seq_ids);
        let dpo = gpu.alloc_from_slice(&positions);
        for _ in 0..3 {
            t.paged_attn.run(&dq, &dkc, &dvc, &dbt, &dsi, &dpo, &dout, q_scale,
                bpr as u32, block_size as u32, q_heads as u32, kv_heads as u32,
                head_dim as u32, tokens as u32, &stream);
        }
        stream.sync();
        let (e0, e1) = (gpu.event(), gpu.event());
        e0.record_on(&stream);
        for _ in 0..20 {
            t.paged_attn.run(&dq, &dkc, &dvc, &dbt, &dsi, &dpo, &dout, q_scale,
                bpr as u32, block_size as u32, q_heads as u32, kv_heads as u32,
                head_dim as u32, tokens as u32, &stream);
        }
        e1.record_on(&stream);
        stream.sync();
        let per = e1.elapsed_ms(&e0) / 20.0;
        eprintln!("o24 paged_attn tokens={tokens} pos={pos}：{per:.3} ms/launch（20 拍均值）");
        let gate = if pos == 23 { 0.30 } else { 0.60 };
        assert!(per < gate,
            "paged_attn({tokens}×pos{pos}) {per:.3}ms 超 {gate}ms 门禁——单线程两遍扫形态回归");
        drop((dkc, dvc, dbt, dsi, dpo));
    }
}
