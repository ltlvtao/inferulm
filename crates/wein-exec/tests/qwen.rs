//! wein-exec I1 验收（q 系列）——Qwen3.5-4B 真权重竖切。
//!
//! Qwen35Executor：24 linear_attention（gated deltanet + 深度wise conv）+
//! 8 full_attention（GQA + 零中心 q/k norm + 输出门）混合骨干，真 safetensors
//! 权重上载，分页 KV / 递推状态池 / conv 状态池全走 exec 真路径；对照端
//! qwen_ref 是独立 CPU 镜像（kref 原语 + f64 GEMM）。
//!
//! 全部 GPU + 模型目录门控（无卡/无模型 SKIP——与 x/o/g 系列同式）。
//! 每测试各开执行器（权重 8.4GB + 状态池 ≈ 15GB 显存）——静态锁串行，
//! harness 并行线程下同时开两个必 OOM。
//!
//! - q5 · 前向数值对拍：逐步 logits vs CPU 镜像（容差 + 贪心决策逐位一致）
//! - q6 · 引擎集成：spawn_with 组装真模型，事件流与直跑零漂移
//! - q7 · 真文本生成质量："The capital of France is" → 含 "Paris"
//! - q8 · 状态隔离：交错请求互不污染（linear/conv/KV 状态按行分账）
//! - q9 · 步时延金丝雀：批 4 纯 decode 中位步时延（权重全量读/步的账）
//! - q10 · 行回收零化：回收行重用输出与全新行逐 token 相同
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use wein_engine::{CoreConfig, EngineCore, EngineEvent, Executor, FinishReason, SamplingParams};
use wein_exec::qwen_ref::QwenMirror;
use wein_exec::{argmax, Qwen35Executor};
use wein_model::{model_dir, HfTokenizer};
use wein_sched::StepPlan;

/// 显存串行锁——测试全程持有（不是只护 open：两个活执行器就 OOM）。
/// 中毒穿透（into_inner）：一个测试 panic 不级联毒杀其余测试的排队
fn guard() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

/// 模型目录（不在场 → SKIP）
fn mdir() -> Option<std::path::PathBuf> {
    let d = model_dir();
    if d.is_none() {
        eprintln!("SKIP（模型目录不在场：WEIN_MODEL_DIR 或 ~/.cache/wein/Qwen3.5-4B）");
    }
    d
}

/// 开执行器（GPU + 模型双门控）
fn exec(dir: &std::path::Path) -> Option<Qwen35Executor> {
    match Qwen35Executor::open(dir) {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("SKIP（Qwen35Executor 装配失败：{e}）");
            None
        }
    }
}

/// q5 · 前向数值对拍：prompt 逐步前向 + 2 个贪心自续位，每步全词表 logits
/// 对独立 CPU 镜像。容差 = 6% 相对 + 0.3 绝对底（32 层厂商库 TC 累加 vs
/// f64 镜像的复合漂移上界；结构错一处——权重路由/状态序/round 位置——
/// logits 偏差以数量级爆掉，容差不会兜住）。贪心决策逐步一致是硬断言
#[test]
fn q5_forward_matches_cpu_mirror() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let Some(mut ex) = exec(&dir) else { return };
    let tok = HfTokenizer::open(&dir).unwrap();
    let Ok(mut mi) = QwenMirror::open(&dir) else {
        panic!("CPU 镜像装配失败（模型目录损坏？）");
    };
    let prompt = tok.encode("The capital of France is");
    assert!(prompt.len() >= 4, "tokenizer 异常：{prompt:?}");
    let mut seq = prompt.clone();
    ex.admitted(1, &prompt);
    let mut worst = 0f64;
    for step in 0..(prompt.len() + 2) {
        let pos = step as u32;
        let t_in = seq[step];
        let got = ex.forward_one(1, t_in, pos);
        let want = mi.forward_token(t_in);
        assert_eq!(got.len(), want.len(), "logits 维度 = 词表");
        assert_eq!(argmax(&got), argmax(&want), "第 {step} 步贪心决策不一致（got {} vs 镜像 {}）", argmax(&got), argmax(&want));
        let mut row_worst = 0f64;
        for i in 0..got.len() {
            let (g, r) = (got[i] as f64, want[i] as f64);
            let tol = (0.06 * r.abs()).max(0.3);
            let err = (g - r).abs();
            assert!(err <= tol, "logits[{i}] 偏差 {err} 超容差 {tol}（got {g} vs 镜像 {r}）");
            row_worst = row_worst.max(err / tol);
        }
        worst = worst.max(row_worst);
        seq.push(argmax(&got));
    }
    eprintln!("q5 最差元素容差占用 {:.0}%", worst * 100.0);
    // 序列确有推进（不是平凡复读）
    assert!(seq[prompt.len()..].windows(2).any(|w| w[0] != w[1]), "贪心自续不该恒平");
}

/// q6 · 引擎集成：EngineCore::spawn_with 组装真模型执行器，8 token 输出与
/// 直跑 generate 零漂移。直跑对照先做先放（两份 15GB 显存装不下，串行）
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn q6_engine_integration() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let prompt = HfTokenizer::open(&dir).unwrap().encode("The capital of France is");
    let direct = {
        let Some(mut d) = exec(&dir) else { return };
        d.generate(&prompt, 8)
    }; // 对照执行器在此析构（显存归还），再开引擎侧
    let Some(ex) = exec(&dir) else { return };
    let (handle, stats) =
        EngineCore::spawn_with(CoreConfig::scripted(vec![], Duration::ZERO), Box::new(ex));
    let (_id, mut rx) = handle.submit(prompt.clone(), 8, SamplingParams::default());
    let mut toks = Vec::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            EngineEvent::Token(t) => toks.push(t),
            EngineEvent::Finished(FinishReason::Length) => break,
            EngineEvent::Finished(r) => panic!("8 token 内不该以 {r:?} 终结"),
            EngineEvent::Accepted => {}
        }
    }
    assert_eq!(toks.len(), 8, "max_tokens=8 → 8 个 Token 事件");
    assert_eq!(toks, direct, "引擎路径与直跑路径输出须逐 token 一致");
    assert_eq!(stats.finished_count(), 1);
}

/// q7 · 真文本生成质量：贪心 12 token 续写事实性陈述。这是 I1 的真值面
/// ——数值对拍（q5）证明"和镜像一致"，本条证明"镜像外的东西没接错"
/// （权重路由/层序/rope 表/tokenizer 任一错都产不出 Paris）
#[test]
fn q7_real_text_generation() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let Some(mut ex) = exec(&dir) else { return };
    let tok = HfTokenizer::open(&dir).unwrap();
    let prompt = tok.encode("The capital of France is");
    let toks = ex.generate(&prompt, 12);
    assert_eq!(toks.len(), 12, "贪心 12 token 不提前停");
    let mut full = prompt.clone();
    full.extend_from_slice(&toks);
    let text = tok.decode(&full);
    eprintln!("q7 续写：{text:?}");
    assert!(text.contains("Paris"), "续写须含 Paris（got {text:?}）");
}

/// q8 · 状态隔离：批伴的**值**不得渗入他人的数值路径。同型对拍设计：
/// 运行一 A+A′（同 prompt 异状态行）联合 decode——A′ 与 A 输入全同，
/// token 须逐位相同（行独立 + 状态行不串）；运行二换批伴 B（异 prompt），
/// A 的 token 须与运行一全同（同 m=2 形状，唯批伴值不同——任何 deltanet/
/// conv/KV/块表串线都在这里现形）。
///
/// 不与 m=1 单跑对拍：cuBLASLt 按 m 选 algo，批组成不同则 GEMM 累加序
/// 不同，ulp 级漂移可翻转低边际贪心位（真发现：A 单跑 96230 vs 联合 15，
/// 32 层复合后 top-2 差 < 漂移）——批组成数值依赖记录在案（W2 每形状
/// 钉 algo 后可收紧到跨 m 对拍）
#[test]
fn q8_state_isolation_interleaved() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let tok = HfTokenizer::open(&dir).unwrap();
    let pa = tok.encode("The capital of France is");
    let pb = tok.encode("The tallest mountain on Earth is");
    // 运行一：A + A′（同 prompt，各自状态行）
    let base = {
        let Some(mut ex) = exec(&dir) else { return };
        ex.admitted(1, &pa);
        ex.admitted(2, &pa);
        let t0 = ex.step(&StepPlan { prefill: vec![(1, pa.len())], decode: vec![1], batch: vec![] });
        let t1 = ex.step(&StepPlan { prefill: vec![(2, pa.len())], decode: vec![2], batch: vec![] });
        assert_eq!(t0.len(), 1);
        assert_eq!(t1.len(), 1);
        assert_eq!(t0[0].1, t1[0].1, "同 prompt 同形状（m=1 追平）→ 首 token 须相同");
        let mut a = vec![t0[0].1];
        for _ in 0..5 {
            let t = ex.step(&StepPlan { prefill: vec![], decode: vec![1, 2], batch: vec![] });
            let got: Vec<u32> = t.iter().map(|(_, t)| *t).collect();
            assert_eq!(got.len(), 2, "联合步各出一个 token");
            assert_eq!(got[0], got[1], "联合步两行输入全同 → token 须相同（行独立）");
            a.push(got[0]);
        }
        a
    }; // 执行器在此析构——运行二全新行号，唯一变量是批伴的值
    // 运行二：A + B（异 prompt）——A 的 token 与运行一逐位相同
    let Some(mut ex) = exec(&dir) else { return };
    ex.admitted(1, &pa);
    ex.admitted(2, &pb);
    let t = ex.step(&StepPlan { prefill: vec![(1, pa.len())], decode: vec![1], batch: vec![] });
    assert_eq!(t[0].1, base[0], "A 追平首 token 与运行一一致");
    let _ = ex.step(&StepPlan { prefill: vec![(2, pb.len())], decode: vec![2], batch: vec![] });
    for k in 1..6 {
        let t = ex.step(&StepPlan { prefill: vec![], decode: vec![1, 2], batch: vec![] });
        let got: Vec<u32> = t.iter().filter(|(i, _)| *i == 1).map(|(_, t)| *t).collect();
        assert_eq!(got, vec![base[k]], "第 {k} 步 A 的 token 与运行一一致（批伴值不得渗入）");
    }
    // B 退场回收（行/页/状态零化），A 独走一步不炸
    ex.retain(&[1]);
    let t = ex.step(&StepPlan { prefill: vec![], decode: vec![1], batch: vec![] });
    assert_eq!(t.len(), 1);
    eprintln!("q8：同型六步全同 + 换批伴六步全同 + 回收后独走存活");
}

/// q9 · 步时延金丝雀：批 4 纯 decode 30 步中位（真权重全量读 8.4GB/步的
/// 内存带宽账；对照 x7 toy 的 µs 量级）。200ms 是 Sanity 上限不是性能承诺
#[test]
fn q9_decode_step_latency_canary() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let Some(mut ex) = exec(&dir) else { return };
    let ids: Vec<u64> = (1..=4).collect();
    let prompt = vec![5u32, 6, 7];
    for &id in &ids {
        ex.admitted(id, &prompt);
    }
    let plan = StepPlan {
        prefill: ids.iter().map(|&i| (i, prompt.len())).collect(),
        decode: ids.clone(),
        batch: vec![],
    };
    assert_eq!(ex.step(&plan).len(), 4, "追平步各出一个 token");
    let mut us = Vec::new();
    for _ in 0..30 {
        let t0 = Instant::now();
        let plan = StepPlan { prefill: vec![], decode: ids.clone(), batch: vec![] };
        let out = ex.step(&plan);
        us.push(t0.elapsed().as_secs_f64() * 1e6);
        assert_eq!(out.len(), 4, "纯 decode 每请求每步一个 token");
    }
    us.sort_by(|a, b| a.total_cmp(b));
    let med = us[us.len() / 2];
    eprintln!(
        "q9 批 4 decode 步时延：中位 {med:.0}µs（p10 {:.0} / p90 {:.0}）→ 单流 ~{:.0} tps",
        us[3],
        us[us.len() - 3],
        1e6 / med
    );
    assert!(med < 200_000.0, "步时延 {med:.0}µs 超 Sanity 上限 200ms");
}

/// q10 · 行回收零化：回收行重用（free_rows 路径）后 prefill 首 token 与
/// 全新行一致——linear/conv 状态行的回收零化在 retain/preempted 落地；
/// 漏零化时残留递推状态直接改写 prefill 输出，本条现形
#[test]
fn q10_row_recycling_zeroed_state() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let Some(mut ex) = exec(&dir) else { return };
    let tok = HfTokenizer::open(&dir).unwrap();
    let pb = tok.encode("The tallest mountain on Earth is");
    let base = ex.generate(&pb, 6);
    // 回收循环：admit → 追平 → decode 一步 → retain([])（行回收+零化）
    for k in 1u64..=3 {
        ex.admitted(k, &pb);
        let t0 = ex.step(&StepPlan {
            prefill: vec![(k, pb.len())],
            decode: vec![k],
            batch: vec![],
        });
        assert_eq!(t0.len(), 1, "第 {k} 轮追平步出一个 token");
        assert_eq!(t0[0].1, base[0], "第 {k} 轮回收行 prefill 首 token 与全新行一致（状态零化）");
        let t1 = ex.step(&StepPlan { prefill: vec![], decode: vec![k], batch: vec![] });
        assert_eq!(t1[0].1, base[1], "第 {k} 轮回收行第二 token 与全新行一致");
        ex.retain(&[]);
    }
    // reset 全量零化路径（generate 内部）同样一致
    assert_eq!(ex.generate(&pb, 6), base, "reset 后完整生成须与基线逐 token 一致");
}

/// q13 · 步时分解插桩：WEIN_STEP_PROFILE 下 eager 步按算子类切段
/// （gemm/gather/norm/eltw/conv/delta/attn/rope/kv/embed/logits/argmax）。
/// 结构合同（非时延阈值）：报告存在、段集完备——漏一类 = 插桩盲区，
/// 满负荷的账会错分；打点数远超步数（层循环内多段）且 GPU 段时距非负。
#[test]
fn q13_step_profile_sections() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    std::env::set_var("WEIN_STEP_PROFILE", "1");
    let ex = exec(&dir);
    std::env::remove_var("WEIN_STEP_PROFILE");
    let Some(mut ex) = ex else { return };
    let ids: Vec<u64> = (1..=4).collect();
    let prompt = vec![5u32, 6, 7];
    for &id in &ids {
        ex.admitted(id, &prompt);
    }
    let plan = StepPlan {
        prefill: ids.iter().map(|&i| (i, prompt.len())).collect(),
        decode: ids.clone(),
        batch: vec![],
    };
    assert_eq!(ex.step(&plan).len(), 4, "追平步各出一个 token");
    for _ in 0..3 {
        let out = ex.step(&StepPlan { prefill: vec![], decode: ids.clone(), batch: vec![] });
        assert_eq!(out.len(), 4, "纯 decode 每请求每步一个 token");
    }
    let Some(rep) = ex.step_profile_report() else {
        panic!("WEIN_STEP_PROFILE 开启但无报告——插桩未激活");
    };
    for sect in ["embed", "gemm", "norm", "gather", "eltw", "logits", "argmax", "conv", "delta", "attn", "rope", "kv"] {
        assert!(rep.contains(sect), "段表缺 {sect}——插桩盲区（{rep}）");
    }
    eprintln!("{rep}");
}

// ── qs 系列：radix 前缀缓存 + 状态快照（命中恢复位级合同 / 末 token
//    重算规则 / 抢占去共享 / 关闭态基线）──

/// 33-token prompt（2 整块 + 1）：跨块界落快照 → 完成移交 → 第二请求
/// 命中 32 恢复状态，重算末 token（m=1 同批形状）后续生成逐 token 位级
/// 一致——快照是 D2D 位拷贝，恢复后的递推 ≡ 从未中断（本刀数值合同）
#[test]
fn qs1_radix_hit_restore_bitwise() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let Some(mut ex) = exec(&dir) else { return };
    let tok = HfTokenizer::open(&dir).unwrap();
    let mut p = tok.encode(
        "The quick brown fox jumps over the lazy dog while the sun sets behind the distant mountains, and the river flows quietly through the ancient forest where birds sing their evening songs before night falls over the sleeping village below the quiet stars",
    );
    p.truncate(33);
    assert_eq!(p.len(), 33, "截断后 33 token（2 整块 + 1 尾）");
    // 请求一：32+1 分段前向——首段不采样（请求二无 pos31 采样位，对齐
    // 两者的喂入流），快照落在 32 界，末 token 以 m=1 批形状重放
    ex.admitted(1, &p);
    let _ = ex.step(&StepPlan { prefill: vec![(1, 32)], decode: vec![], batch: vec![] });
    let t2 = ex.step(&StepPlan { prefill: vec![(1, 1)], decode: vec![1], batch: vec![] });
    let mut base = vec![t2[0].1];
    for _ in 0..4 {
        let d = ex.step(&StepPlan { prefill: vec![], decode: vec![1], batch: vec![] });
        base.push(d[0].1);
    }
    ex.retain(&[]); // 移交：2 整块挂树，末块挂快照槽
    // 请求二：同 prompt 命中 32（恢复点边界 32 < 33，留末 token 作采样位）
    let hit = ex.admitted(2, &p);
    assert_eq!(hit, 32, "须命中 2 整块");
    let t = ex.step(&StepPlan { prefill: vec![(2, 1)], decode: vec![2], batch: vec![] });
    assert_eq!(t[0].1, base[0], "恢复后末 token 重放（m=1 同形状）→ 位级一致");
    for k in 1..base.len() {
        let d = ex.step(&StepPlan { prefill: vec![], decode: vec![2], batch: vec![] });
        assert_eq!(d[0].1, base[k], "恢复后续生成第 {k} token 位级一致（快照位拷贝合同）");
    }
}

/// 精确块对齐 prompt（恰 32）不命中——恢复到 prompt 末尾 = 无 token 可
/// 喂 = 无采样位（末 token 重算规则）；长一 token 的 prompt 共享前 32
/// 块正常命中
#[test]
fn qs2_exact_multiple_rule() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let Some(mut ex) = exec(&dir) else { return };
    let tok = HfTokenizer::open(&dir).unwrap();
    let mut p32 = tok.encode(
        "The quick brown fox jumps over the lazy dog while the sun sets behind the distant mountains, and the river flows quietly through the ancient forest where birds sing their evening songs before night falls over the sleeping village below the quiet stars",
    );
    p32.truncate(32);
    ex.admitted(1, &p32);
    let t = ex.step(&StepPlan { prefill: vec![(1, 32)], decode: vec![1], batch: vec![] });
    let mut base = vec![t[0].1];
    for _ in 0..3 {
        let d = ex.step(&StepPlan { prefill: vec![], decode: vec![1], batch: vec![] });
        base.push(d[0].1);
    }
    ex.retain(&[]); // 捐 2 块，槽在末块
    let hit = ex.admitted(2, &p32);
    assert_eq!(hit, 0, "边界 32 == len 32 → 不命中（末 token 重算规则）");
    ex.retain(&[]); // 请求二未消费任何块，清退
    let mut p33 = p32.clone();
    p33.push(7u32);
    let hit = ex.admitted(3, &p33);
    assert_eq!(hit, 32, "长一 token 的 prompt 命中 2 整块");
    let t = ex.step(&StepPlan { prefill: vec![(3, 1)], decode: vec![3], batch: vec![] });
    assert_eq!(t.len(), 1, "命中路径重算末 token 采样正常");
}

/// 抢占去共享：命中恢复的请求被抢占（共享页换私有页 + 状态零化 + 游标
/// 归零）后同批形状全量重放位级复原；radix 树无恙——第三请求仍命中
#[test]
fn qs3_preempt_deshare_replay() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let Some(mut ex) = exec(&dir) else { return };
    let tok = HfTokenizer::open(&dir).unwrap();
    let mut p = tok.encode(
        "The quick brown fox jumps over the lazy dog while the sun sets behind the distant mountains, and the river flows quietly through the ancient forest where birds sing their evening songs before night falls over the sleeping village below the quiet stars",
    );
    p.truncate(33);
    // 请求一：32+1 驱动 → 基线 6 token；移交
    ex.admitted(1, &p);
    let t = ex.step(&StepPlan { prefill: vec![(1, 32)], decode: vec![1], batch: vec![] });
    let t2 = ex.step(&StepPlan { prefill: vec![(1, 1)], decode: vec![1], batch: vec![] });
    let mut base = vec![t[0].1, t2[0].1];
    for _ in 0..4 {
        let d = ex.step(&StepPlan { prefill: vec![], decode: vec![1], batch: vec![] });
        base.push(d[0].1);
    }
    ex.retain(&[]);
    // 请求二：命中 32 → 抢占 → 全量重放（32+1 同批形状）
    assert_eq!(ex.admitted(2, &p), 32);
    ex.preempted(2);
    let t = ex.step(&StepPlan { prefill: vec![(2, 32)], decode: vec![2], batch: vec![] });
    assert_eq!(t[0].1, base[0], "抢占重放 32 段（同形状）→ 位级一致");
    let t2 = ex.step(&StepPlan { prefill: vec![(2, 1)], decode: vec![2], batch: vec![] });
    assert_eq!(t2[0].1, base[1], "重放末 token 位级一致");
    for k in 2..base.len() {
        let d = ex.step(&StepPlan { prefill: vec![], decode: vec![2], batch: vec![] });
        assert_eq!(d[0].1, base[k], "重放续生成第 {k} token 位级一致");
    }
    ex.retain(&[]); // 重放块与树重叠 → 冗余释放（coverage 路径）
    // 请求三：树无恙，仍命中
    assert_eq!(ex.admitted(3, &p), 32, "抢占/重放后 radix 树仍可命中");
}

/// WEIN_RADIX=0 关闭态：重复 prompt 不命中（旧语义），生成正常——A/B 基线
#[test]
fn qs4_radix_off_baseline() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    std::env::set_var("WEIN_RADIX", "0");
    let ex = exec(&dir);
    std::env::remove_var("WEIN_RADIX");
    let Some(mut ex) = ex else { return };
    let tok = HfTokenizer::open(&dir).unwrap();
    let mut p = tok.encode(
        "The quick brown fox jumps over the lazy dog while the sun sets behind the distant mountains, and the river flows quietly through the ancient forest where birds sing their evening songs before night falls over the sleeping village below the quiet stars",
    );
    p.truncate(33);
    ex.admitted(1, &p);
    let t = ex.step(&StepPlan { prefill: vec![(1, 33)], decode: vec![1], batch: vec![] });
    assert_eq!(t.len(), 1, "关闭态全量前向正常");
    for _ in 0..3 {
        let d = ex.step(&StepPlan { prefill: vec![], decode: vec![1], batch: vec![] });
        assert_eq!(d.len(), 1);
    }
    ex.retain(&[]);
    assert_eq!(ex.admitted(2, &p), 0, "关闭态重复 prompt 不命中（旧语义）");
}

/// 槽耗尽 → 驱逐腾槽（块与槽联动回收）+ 之后的命中恢复仍位级正确：
/// 借槽/还槽记账一旦双借或漏还，两个块共用一槽 → 恢复出别人的行状态 →
/// 采样 token 必偏离基线（快照位拷贝合同不让步）。N = SNAP_SLOTS(64) + 6
/// 个首 token 各异（无共享前缀）的 33-token prompt，每个捐 2 块 1 槽
#[test]
fn qs5_snap_slot_exhaustion_recycles() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let Some(mut ex) = exec(&dir) else { return };
    let tok = HfTokenizer::open(&dir).unwrap();
    let mut p0 = tok.encode(
        "The quick brown fox jumps over the lazy dog while the sun sets behind the distant mountains, and the river flows quietly through the ancient forest where birds sing their evening songs before night falls over the sleeping village below the quiet stars",
    );
    p0.truncate(33);
    const N: usize = 70;
    let id = |i: usize| i as u64 + 1;
    let mut base: Vec<[u32; 3]> = Vec::with_capacity(N);
    for i in 0..N {
        let mut p = p0.clone();
        p[0] = 100 + i as u32; // 首 token 各异 → 互不共享前缀（各自独立捐块）
        assert_eq!(ex.admitted(id(i), &p), 0, "#{i} 不命中");
        let _ = ex.step(&StepPlan { prefill: vec![(id(i), 32)], decode: vec![], batch: vec![] });
        let t = ex.step(&StepPlan { prefill: vec![(id(i), 1)], decode: vec![id(i)], batch: vec![] });
        let d1 = ex.step(&StepPlan { prefill: vec![], decode: vec![id(i)], batch: vec![] });
        let d2 = ex.step(&StepPlan { prefill: vec![], decode: vec![id(i)], batch: vec![] });
        base.push([t[0].1, d1[0].1, d2[0].1]);
        ex.retain(&[]); // 移交：块进树、槽随块
    }
    // 复验：先探最老 5（大概率已被驱逐 → 全量重放），再探最新 5（LRU 应留
    // → 命中 32）。两条路径都须与基线逐 token 位级一致
    let mut hits = 0;
    for i in (0..5).chain(N - 5..N) {
        let mut p = p0.clone();
        p[0] = 100 + i as u32;
        let hit = ex.admitted(id(i) + 1000, &p);
        if hit == 0 {
            let _ =
                ex.step(&StepPlan { prefill: vec![(id(i) + 1000, 32)], decode: vec![], batch: vec![] });
        } else {
            assert_eq!(hit, 32, "#{i} 命中粒度须为整块（2 块）");
            hits += 1;
        }
        let t = ex.step(&StepPlan {
            prefill: vec![(id(i) + 1000, 33 - hit)],
            decode: vec![id(i) + 1000],
            batch: vec![],
        });
        assert_eq!(t[0].1, base[i][0], "#{i} 恢复后重算末 token 位级一致（hit={hit}）");
        for k in 1..3 {
            let d = ex.step(&StepPlan {
                prefill: vec![],
                decode: vec![id(i) + 1000],
                batch: vec![],
            });
            assert_eq!(d[0].1, base[i][k], "#{i} 续生成第 {k} token 位级一致（hit={hit}）");
        }
        ex.retain(&[]);
    }
    assert!(hits >= 3, "驱逐须按 LRU 保最新条目（最新 5 中命中 {hits}）");
}
