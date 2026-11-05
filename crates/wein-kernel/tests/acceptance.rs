//! kernel 层冒烟验收（02-architecture §13-2 的三条验收线 + 驱动路径本身）。
//!
//! 冒烟实验回答"cuda crate 路线（纯 Rust→PTX，无 nvcc）能不能走"：
//! - k1/k2：PTX 产出物结构与 intrinsic 降落（**无 GPU 也能跑**——工具链线）
//! - k3-k6：driver API 探测/显存/launch/数值逐位对齐（需 GPU，无卡则 SKIP）
//! - k7：PTX JIT 缓存落盘可复用（§13-2 ①；子进程跨进程验证）
//! - k8：launch 开销金丝雀（给基线制度的第一格 kernel 侧数字）
//!
//! 无 GPU 环境：k3-k8 打印 SKIP 后通过（设备行为没有硬件无从断言）；
//! 本机（RTX 4090）全量跑。W2 真 attention kernel 出来后，k6 的位对齐
//! 断言原样升格为 golden vector 框架的第一条。
//!
//! 逐位对齐的可达性依据：bf16→f32 精确（位左移）；fma.rn.f32 与 x86 FMA3
//! 都是 IEEE 正确舍入的单次乘加——同操作序列必同位；RNE 公式两侧同一公式。

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use wein_kernel::cuda::{Gpu, Stream};
use wein_kernel::kernel_args;

fn ptx_path() -> PathBuf {
    let p = std::env::var("WEIN_PTX").expect("build.rs 须注入 WEIN_PTX");
    let p = PathBuf::from(p);
    assert!(p.exists(), "PTX 产物缺失：{p:?}（build.rs 未产出？）");
    p
}

fn ptx_text() -> String {
    std::fs::read_to_string(ptx_path()).expect("读 PTX 文本")
}

/// 无卡环境：打印 SKIP 并让测试通过（硬件行为无硬件无从断言）
fn gpu() -> Option<Gpu> {
    match Gpu::probe() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("SKIP（无可用 CUDA 设备：{e}）");
            None
        }
    }
}

// ───────────────────────── 工具链线（无 GPU） ─────────────────────────

/// k1 · PTX 产出物：rustc（无 nvcc）吐出的 PTX 文本结构成立——
/// 版本/目标头在，两个导出 kernel 各有一个 `.visible .entry`，名字未被修饰。
/// 这一条钉死"纯 Rust→PTX"路线的最小事实。
#[test]
fn k1_ptx_artifact_structure() {
    let ptx = ptx_text();
    assert!(ptx.contains(".version "), "缺 PTX 版本头");
    assert!(ptx.contains(".target "), "缺 target 头");
    assert!(ptx.contains(".entry vec_add_f32("), "vec_add_f32 入口缺失/被改名");
    assert!(ptx.contains(".entry bf16_dot("), "bf16_dot 入口缺失/被改名");
    assert!(ptx.contains(".entry rms_norm("), "rms_norm 入口缺失/被改名");
    assert!(ptx.contains(".entry rope("), "rope 入口缺失/被改名");
    assert!(ptx.contains(".entry silu("), "silu 入口缺失/被改名");
    assert!(ptx.contains(".entry w8a8_quant("), "w8a8_quant 入口缺失/被改名");
    assert!(ptx.contains(".entry kv_scatter("), "kv_scatter 入口缺失/被改名");
    assert!(ptx.contains(".entry paged_attn("), "paged_attn 入口缺失/被改名");
    assert!(ptx.contains(".entry embed("), "embed 入口缺失/被改名");
    assert!(ptx.contains(".entry add("), "add 入口缺失/被改名");
    assert!(ptx.contains(".entry mul("), "mul 入口缺失/被改名");
    // I1 竖切（Qwen3.5 文本骨干）八入口
    assert!(ptx.contains(".entry rms_norm_zc("), "rms_norm_zc 入口缺失/被改名");
    assert!(ptx.contains(".entry add_norm_zc("), "add_norm_zc 入口缺失/被改名");
    assert!(ptx.contains(".entry silu_mul("), "silu_mul 入口缺失/被改名");
    assert!(ptx.contains(".entry dn_split_qkv("), "dn_split_qkv 入口缺失/被改名");
    assert!(ptx.contains(".entry rope_pt("), "rope_pt 入口缺失/被改名");
    assert!(ptx.contains(".entry sigmul("), "sigmul 入口缺失/被改名");
    assert!(ptx.contains(".entry conv1d_dw("), "conv1d_dw 入口缺失/被改名");
    assert!(ptx.contains(".entry conv_state_roll("), "conv_state_roll 入口缺失/被改名");
    assert!(ptx.contains(".entry deltanet("), "deltanet 入口缺失/被改名");
    assert!(ptx.contains(".entry rms_norm_gated("), "rms_norm_gated 入口缺失/被改名");
    assert!(ptx.contains(".entry argmax_rows("), "argmax_rows 入口缺失/被改名");
    // 设备端采样器（需求刀 1）
    assert!(ptx.contains(".entry sample_rows("), "sample_rows 入口缺失/被改名");
    // 长序列分块 attention（KV 复用几何）三入口：两遍 + 单遍在线 + 分片合并
    assert!(ptx.contains(".entry paged_attn_tile("), "paged_attn_tile 入口缺失/被改名");
    assert!(ptx.contains(".entry paged_attn_online("), "paged_attn_online 入口缺失/被改名");
    assert!(ptx.contains(".entry attn_combine("), "attn_combine 入口缺失/被改名");
    // deltanet 的 q/k l2norm 前置小 kernel（invq/invk 去重）
    assert!(ptx.contains(".entry dn_qk_l2norm("), "dn_qk_l2norm 入口缺失/被改名");
    // attention A/B 档（几何 × 池 dtype；见 `ops::AttnCfg`）：单遍 9 档 +
    // 两遍 f32 + kv_scatter f32。每档一份单态实体（nd/rpt/wide 是编译期常量）
    for k in [
        "paged_attn_online_n8",
        "paged_attn_online_n16",
        "paged_attn_online_r2",
        "paged_attn_online_r4",
        "paged_attn_online_n8_r4",
        "paged_attn_online_n16_r4",
        "paged_attn_online_n16_r2",
        "paged_attn_online_f32",
        "paged_attn_online_f32_n8",
        "paged_attn_online_f32_r2",
        "paged_attn_online_f32_r4",
        "paged_attn_tile_f32",
        "kv_scatter_f32",
        // 块档（Br 参数化、O/P 落块私有 scratch）：prefill 长序列
        "paged_attn_blk",
        "paged_attn_blk_f32",
        // qblk 档（K/V 行片驻寄存器供全 q 行）：探针/库增量，未接线
        "paged_attn_qblk_n16r4j2",
        "paged_attn_qblk_n16r4j4",
        "paged_attn_qblk_n16r8j2",
        "paged_attn_qblk_n16r8j4",
        "paged_attn_qblk_n8r8j4",
        "paged_attn_qblk_n8r16j2",
        // v3 档（smem K/V tile + shfl 归约）：prefill 长序列（WEIN_ATTN_V3 接线）
        "paged_attn_v3_n16k16",
        "paged_attn_v3_n16k8",
        "paged_attn_v3_n8k16",
        "paged_attn_v3_n8k8",
        "paged_attn_v3_d32k16",
        "paged_attn_v3_d32k8",
        "paged_attn_v3_n32k16",
        "paged_attn_v3_n32k8",
        // v4 档（wmma tensor core）：长 prefill（WEIN_ATTN_V4 接线）
        "paged_attn_v4",
        // wmma 能力探针（v4 前置：fragment 映射/累加验证）
        "wmma_probe",
    ] {
        let head = format!(".entry {k}(");
        assert!(ptx.contains(&head), "{k} 入口缺失/被改名");
    }
    // attention v5 档（4 warp × 64 行 CTA 共享 K/V smem）
    assert!(ptx.contains(".entry paged_attn_v5("), "paged_attn_v5 入口缺失/被改名");
    // attention v9 档（手写 PTX：LDSM + mma.sync + 99KB 动态 smem）
    assert!(ptx.contains(".entry paged_attn_v9("), "paged_attn_v9 入口缺失/被改名");
    assert_eq!(
        ptx.matches(".visible .entry").count(),
        65,
        "导出 kernel 数应为 65（64 + v9 档 1）"
    );
}

/// k2 · intrinsic 降落（§13-2 ③）：线程索引 intrinsic 落到特殊寄存器
/// （%tid.x / %ctaid.x），浮点算术落到正确舍入的 PTX 原语，且没有宿主调用
/// shim（no_std kernel 里出现 call = 有东西没编译干净）。
///
/// FMA 落地后的三段断言：
/// ① **字面量不收缩**（原意保留）：`a*b + acc` 字面量与乘/除仍落
///    mul.rn.f32 / add.rn.f32——收缩只发生在显式 `fma()`（`mul_add`），
///    不是编译器自作主张；`bf16_dot` 就是那条活证据。
/// ② **fma.rn.f32 存在**（全模块计数下界）。
/// ③ **热路径真的吃了 fma**：逐 kernel 段内计数下界——只数全模块会被别的
///    kernel 蒙混过关，这里按 `.visible .entry NAME(` 切段数。下界留一倍
///    裕量（实测值见注释），形态回归（谁把 `fma()` 改回 `a*b+c`）必现形。
#[test]
fn k2_intrinsics_lowered_to_ptx() {
    let ptx = ptx_text();
    assert!(ptx.contains("%tid.x"), "thread_idx 未降落特殊寄存器");
    assert!(ptx.contains("%ctaid.x"), "block_idx 未降落特殊寄存器");
    assert!(ptx.contains("mul.rn.f32"), "乘法未降落 mul.rn.f32（字面量被收缩？）");
    assert!(ptx.contains("add.rn.f32"), "加法未降落 add.rn.f32（字面量被收缩？）");
    let n_fma = ptx.matches("fma.rn.f32").count();
    assert!(n_fma >= 300, "fma.rn.f32 计数 {n_fma} < 300：热路径的 mul_add 没落成 fma");
    // 逐 kernel 段：热路径的 fma 指令数下界（实测 26/38/55/2/646/9/1/1）
    let seg = |name: &str| -> String {
        let head = format!(".visible .entry {name}(");
        let i = ptx.find(&head).unwrap_or_else(|| panic!("{name} 入口缺失"));
        let rest = &ptx[i + head.len()..];
        let end = rest.find(".visible .entry ").unwrap_or(rest.len());
        rest[..end].to_string()
    };
    for (name, floor, note) in [
        ("paged_attn_online", 12, "attn_dot + chunk 偏量 4 维 + 在线合并 4 维（实测 26）"),
        ("paged_attn_tile", 12, "attn_dot + 4 维加权 V（jsplit 合并另计）"),
        ("paged_attn", 20, "attn_dot + 加权 V（流式/直路）"),
        ("attn_combine", 2, "jsplit 合并 l/acc"),
        ("deltanet", 200, "kvm/o 折叠（kvm4*/dn_e2*/upd4* 各级 fma）"),
        ("rms_norm", 4, "行平方和"),
        ("rms_norm_zc", 1, "块归约平方和"),
        ("rms_norm_gated", 1, "块归约平方和"),
    ] {
        let got = seg(name).matches("fma.rn.f32").count();
        assert!(got >= floor, "{name} 段内 fma.rn.f32 = {got} < {floor}（{note} 未走 fma）");
    }
    let host_calls: Vec<&str> = ptx
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            // 白名单：cp.async 占位函数的设备内调用（build.rs patch_cp_async
            // 替换体，ptxas 内联）——不是宿主调用
            t.starts_with("call") && !t.starts_with("call.uni cp_async_")
        })
        .collect();
    assert!(
        host_calls.is_empty(),
        "PTX 出现宿主调用：{host_calls:?}（no_std kernel 不应有 call）"
    );
    // cp.async 能力门：占位函数体替换旁路（patch_cp_async）经 attn_v5_probe
    // 验收后钉死——① 三条真指令落模块 ② v5 K staging 调用点在位
    // ③ 占位 volatile 体不得残留（未替换就上线会写坏 smem）。
    for needle in [
        "cp.async.cg.shared.global",
        "cp.async.commit_group",
        "cp.async.wait_all",
    ] {
        assert!(ptx.contains(needle), "{needle} 未落 PTX（patch_cp_async 回归）");
    }
    assert!(
        ptx.matches("call.uni cp_async_cg16").count() >= 8,
        "v5 K staging 调用点缺失（期望 ≥8：双 stage 调用点 ×4 展开轮）"
    );
    assert!(!ptx.contains("st.volatile"), "cp.async 占位 volatile 体残留");
    // shfl/smem 能力门：旁路（link_llvm_intrinsics + build.rs 段改写）经
    // shfl_probe 验收后钉死——① shfl.sync 出现 ② SMEM_PROBE 以 .shared 声明
    // ③ 产物内不得再留 ld.global/st.global（改写全函数体扫过一遍）。
    assert!(ptx.matches("shfl.sync").count() >= 6, "shfl.sync 未落 PTX（旁路回归）");
    assert!(
        ptx.lines().any(|l| l.contains("SMEM_PROBE") && l.contains(".shared")),
        "SMEM_PROBE 未以 .shared 声明（build.rs 改写回归）"
    );
    assert!(
        !ptx.contains("st.global") && !ptx.contains("ld.global"),
        "产物仍含 ld/st.global：cvta 泛型化未全扫（可能仍有 generic 语义漏洞）"
    );
}

// ───────────────────────── 驱动线（需 GPU） ─────────────────────────

/// k3 · driver 探测：dlopen(libcuda.so.1) → cuInit → 设备枚举/名字/算力，
/// arch 字串与算力自洽（"sm_{maj}{min}"）。三层能力模型的身份层数据源。
#[test]
fn k3_driver_probe() {
    let Some(gpu) = gpu() else { return };
    assert!(gpu.device_count() >= 1);
    assert!(gpu.name().contains("NVIDIA"), "设备名异常：{}", gpu.name());
    let (maj, min) = gpu.compute_capability();
    assert!((maj, min) >= (7, 0), "算力过低：{maj}.{min}");
    assert_eq!(gpu.arch(), format!("sm_{maj}{min}"), "arch 字串须与算力自洽");
}

/// k4 · 显存往返：HtoD/DtoH 逐字节保持（第一次真·显存分配）。
#[test]
fn k4_device_memory_roundtrip() {
    let Some(gpu) = gpu() else { return };
    let host: Vec<f32> = vec![1.5, -2.25, 3.75, 0.0, f32::MAX, f32::MIN_POSITIVE];
    let dev = gpu.alloc_from_slice(&host);
    let back: Vec<f32> = dev.read_to_vec();
    assert_eq!(back.len(), host.len());
    for (a, b) in host.iter().zip(back.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "显存往返位级失真");
    }
    // 大块分配也成立（一次 cuMemAlloc 的规模感）
    let big = gpu.alloc_bytes(64 << 20);
    assert!(big.device_ptr() != 0);
}

/// k5 · 向量加（CUDA hello 的 Rust 版）：1M 元素、多 block 多线程，
/// f32 加法 IEEE 确定 → 逐位对齐。
#[test]
fn k5_vec_add_bitwise() {
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let n: usize = 1 << 20;
    let a: Vec<f32> = (0..n).map(|i| splitmix(i as u64) as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| splitmix(i as u64 ^ 0x9e37) as f32).collect();
    let expect: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

    let da = gpu.alloc_from_slice(&a);
    let db = gpu.alloc_from_slice(&b);
    let dc = gpu.alloc_bytes(n * 4);

    let module = gpu.load_module(&ptx_text());
    let f = module.func("vec_add_f32");
    let mut nn = n as u32;
    let args = kernel_args![da, db, dc, nn];
    gpu.launch(&f, (n as u32 + 255) / 256, 256, &args, &stream);
    stream.sync();

    let got: Vec<f32> = dc.read_to_vec();
    for i in 0..n {
        assert_eq!(got[i].to_bits(), expect[i].to_bits(), "vec_add[{i}] 位级失真");
    }
}

/// k6 · bf16 点积逐位对齐（§13-2 ② 冒烟化身）：load bf16→f32、fma 链、
/// RNE 收尾——attention QK 数值路径的最小完整形态。
/// 512 线程 × 64 段长，GPU 与 CPU 参考实现输出位位相等。
#[test]
fn k6_bf16_dot_bitwise_alignment() {
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let seg: usize = 64;
    let m: usize = 512; // 线程数（各算一段）
    let n = seg * m;

    // 确定性 bf16 输入（先造 f32 再舍入，两侧都只依赖舍入后的值）
    let x: Vec<u16> = (0..n).map(|i| host_f32_to_bf16(splitmix(i as u64) as f32)).collect();
    let y: Vec<u16> = (0..n).map(|i| host_f32_to_bf16(splitmix(i as u64 ^ 0x517c) as f32)).collect();

    // CPU 参考实现（独立代码路径，不与 kernel 共享实现）
    let expect: Vec<u16> = (0..m)
        .map(|t| {
            let mut acc = 0.0f32;
            for j in 0..seg {
                let a = host_bf16_to_f32(x[t * seg + j]);
                let b = host_bf16_to_f32(y[t * seg + j]);
                acc = a.mul_add(b, acc);
            }
            host_f32_to_bf16(acc)
        })
        .collect();

    let dx = gpu.alloc_from_slice(&x);
    let dy = gpu.alloc_from_slice(&y);
    let dout = gpu.alloc_bytes(m * 2);

    let module = gpu.load_module(&ptx_text());
    let f = module.func("bf16_dot");
    let (mut seg_a, mut m_a) = (seg as u32, m as u32);
    let args = kernel_args![dx, dy, dout, seg_a, m_a];
    gpu.launch(&f, (m as u32 + 255) / 256, 256, &args, &stream);
    stream.sync();

    let got: Vec<u16> = dout.read_to_vec();
    assert_eq!(got.len(), m);
    for t in 0..m {
        assert_eq!(got[t], expect[t], "bf16_dot[{t}] 位不对齐（GPU {:#06x} vs 参考 {:#06x}）", got[t], expect[t]);
    }
}

/// k7 · PTX JIT 缓存落盘可复用（§13-2 ①）：子进程加载同一 PTX 两次
/// （同一 CUDA_CACHE_PATH），第一次后缓存目录非空（落盘），第二次成功且
/// 不慢于第一次（复用）。免 JIT 的价值 = 冷启动延迟，量级打印进报告。
#[test]
fn k7_jit_cache_persists_across_processes() {
    // 此测试自身不需要 GPU 在本进程探测成功——子进程才是被测对象；
    // 但无卡机器子进程必失败，先探测门控
    if gpu().is_none() {
        return;
    }
    let cache_dir = std::env::temp_dir().join(format!("wein-jit-cache-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&cache_dir);
    std::fs::create_dir_all(&cache_dir).unwrap();
    let ptx = ptx_path();

    let run = || {
        let t0 = Instant::now();
        let st = Command::new(env!("CARGO_BIN_EXE_jit-cache-probe"))
            .env("CUDA_CACHE_PATH", &cache_dir)
            .env("CUDA_CACHE_DISABLE", "0")
            .env("WEIN_PTX", &ptx)
            .status()
            .expect("启动 jit-cache-probe 子进程");
        (st.success(), t0.elapsed())
    };

    let (ok1, t1) = run();
    assert!(ok1, "首次加载子进程失败");
    let entries: Vec<_> = std::fs::read_dir(&cache_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    assert!(
        !entries.is_empty(),
        "JIT 缓存未落盘：{cache_dir:?} 为空（复用无从谈起）"
    );

    let (ok2, t2) = run();
    assert!(ok2, "二次加载子进程失败");
    eprintln!(
        "JIT 缓存复用：冷 {t1:?} → 热 {t2:?}（缓存文件 {} 个）",
        entries.len()
    );
    assert!(t2 <= t1, "热加载反而更慢：冷 {t1:?} 热 {t2:?}");
}

/// k8 · launch 开销金丝雀：1 万次连续小 kernel 提交 + 一次同步的均摊。
/// 给基线制度第一格 kernel 侧数字；上限放得很宽（50µs），真值预期 2-8µs（WSL）。
#[test]
fn k8_launch_overhead_canary() {
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let module = gpu.load_module(&ptx_text());
    let f = module.func("vec_add_f32");
    let buf = gpu.alloc_bytes(4);
    let mut n: u32 = 1;
    let args = kernel_args![buf, buf, buf, n];

    // 预热（首次含 JIT/模块初始化，不算数）
    gpu.launch(&f, 1, 1, &args, &stream);
    stream.sync();

    let rounds = 10_000u32;
    let t0 = Instant::now();
    for _ in 0..rounds {
        gpu.launch(&f, 1, 1, &args, &stream);
    }
    stream.sync();
    let per = t0.elapsed().as_nanos() as f64 / rounds as f64 / 1000.0;
    eprintln!("launch 开销：{per:.2} µs/次（{rounds} 次均摊，含一次末尾同步）");
    assert!(per < 50.0, "launch 均摊 {per:.2}µs 超金丝雀上限 50µs");
}

/// k9 · 事件计时链路：成对 cuEvent 流上记录 + ElapsedTime——步时分解
/// 插桩（满负荷 L3q 每行账定位：GPU 时间按算子类切开）的根基。
/// WSL 实测事件时间轴有 ~20µs 量化（同点两事件可差 0.02ms）——插桩
/// 只按**粗段边界**打点（段是 ms 级），本测试同理：kernel 加大跨过
/// 量化 floor，同点断言只钉"重复 record 可用 + 时距有界"。
#[test]
fn k9_event_timing() {
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let module = gpu.load_module(&ptx_text());
    let f = module.func("vec_add_f32");
    let n: u32 = 1 << 24;
    let buf = gpu.alloc_bytes(4 * n as usize);
    let args = kernel_args![buf, buf, buf, n];
    gpu.launch(&f, 256, 256, &args, &stream);
    stream.sync();

    let (e0, e1) = (gpu.event(), gpu.event());
    e0.record_on(&stream);
    // 多拍 kernel 跨在两事件间（插桩的真实形态：段内一串 kernel）——
    // vec_add 每线程一元素，4096×256 恰覆 1M；200 拍把时距抬过量化 floor
    for _ in 0..200 {
        gpu.launch(&f, 4096, 256, &args, &stream);
    }
    e1.record_on(&stream);
    stream.sync();
    let work_ms = e1.elapsed_ms(&e0);
    eprintln!("事件计时：200 拍 1M vec_add = {work_ms:.3} ms");
    assert!(work_ms >= 0.1, "kernel 时距 {work_ms:.3}ms 低于预期——计时链路不通");

    // 同点两事件：可重复 record + 时距有界（≤ 量化上界 ~0.1ms）
    let e2 = gpu.event();
    e0.record_on(&stream);
    e2.record_on(&stream);
    stream.sync();
    let zero = e2.elapsed_ms(&e0);
    assert!(zero < 0.1, "同点事件时距 {zero:.3}ms 超量化上界");
}

/// k10 · D2D 区间拷贝位保持：copy_range_on 是状态快照原语的物理底座——
/// 区间内逐位同源、区间外原值不动（未写坏别人的槽）、非零双向偏移成立。
/// 快照语义的数值合同由此钉死：存/恢复就是位搬运，无重算。
#[test]
fn k10_dtod_range_copy_bitwise() {
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    let n = 1024usize;
    let src: Vec<f32> = (0..n as u32).map(|i| splitmix(i as u64)).collect();
    let src = gpu.alloc_from_slice(&src);
    let dst = gpu.alloc_bytes(n * 4);
    dst.zero_on(&stream);
    stream.sync();

    // 中段拷贝：dst[256..768) ← src[128..640)
    dst.copy_range_on(256 * 4, &src, 128 * 4, 512 * 4, &stream);
    stream.sync();
    let back: Vec<f32> = dst.read_to_vec();
    for i in 0..256 {
        assert_eq!(back[i].to_bits(), 0, "区间外被污染（前段）");
    }
    let want: Vec<f32> = src.read_to_vec();
    for i in 0..512 {
        assert_eq!(back[256 + i].to_bits(), want[128 + i].to_bits(), "区间内位失真 @{i}");
    }
    for i in 768..n {
        assert_eq!(back[i].to_bits(), 0, "区间外被污染（后段）");
    }
}

/// k11 · 批量 D2D 位一致：多段（异源异靶、含区间偏移）一次 cuMemcpyBatchAsync
/// 搬完，段内逐位同、段外不受污染——快照段表的正确性底座
#[test]
fn k11_dtod_batch_copy_bitwise() {
    let Some(gpu) = gpu() else { return };
    let stream = gpu.stream();
    use wein_kernel::cuda::driver::CuBatchOp;
    let n = 512usize;
    let mk = |seed: u64, n: usize| -> Vec<f32> {
        (0..n as u32).map(|i| splitmix(seed + i as u64)).collect()
    };
    let s1 = gpu.alloc_from_slice(&mk(11, n));
    let s2 = gpu.alloc_from_slice(&mk(22, n));
    let d1 = gpu.alloc_bytes(n * 4);
    let d2 = gpu.alloc_bytes(n * 4);
    d1.zero_on(&stream);
    d2.zero_on(&stream);
    stream.sync();
    // 段 1：d1[0..256) ← s1[128..384)；段 2：d2[64..320) ← s2[0..256)
    let ops = [
        CuBatchOp { dst: d1.device_ptr(), src: s1.device_ptr() + 128 * 4, size: 256 * 4 },
        CuBatchOp { dst: d2.device_ptr() + 64 * 4, src: s2.device_ptr(), size: 256 * 4 },
    ];
    stream.memcpy_batch(&ops);
    stream.sync();
    let (b1, b2): (Vec<f32>, Vec<f32>) = (d1.read_to_vec(), d2.read_to_vec());
    let (w1, w2): (Vec<f32>, Vec<f32>) = (s1.read_to_vec(), s2.read_to_vec());
    for i in 0..256 {
        assert_eq!(b1[i].to_bits(), w1[128 + i].to_bits(), "段 1 位失真 @{i}");
        assert_eq!(b2[64 + i].to_bits(), w2[i].to_bits(), "段 2 位失真 @{i}");
    }
    for i in 256..n {
        assert_eq!(b1[i].to_bits(), 0, "段 1 区间外被污染 @{i}");
    }
    for i in 0..64 {
        assert_eq!(b2[i].to_bits(), 0, "段 2 区间外被污染（前）@{i}");
    }
    for i in 320..n {
        assert_eq!(b2[i].to_bits(), 0, "段 2 区间外被污染（后）@{i}");
    }
}

// ───────────────────────── 测试侧工具 ─────────────────────────

/// splitmix32——确定性 f32 源（不用 rand，冒烟测试要自含）
fn splitmix(mut z: u64) -> f32 {
    z = z.wrapping_add(0x9e3779b97f4a7c15);
    let mut x = z ^ (z >> 30);
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x = x.wrapping_mul(0x94d049bb133111eb);
    let u = (x >> 40) as u32; // 24 bit 尾数域
    f32::from_bits((u & 0xffffff) | 0x40000000) - 3.0 // [-1, 1)
}

/// 参考实现侧的 bf16 转换（与 kernel 各自独立实现——对齐才有意义）
fn host_bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

fn host_f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits + 0x7fff + lsb) >> 16) as u16
}
