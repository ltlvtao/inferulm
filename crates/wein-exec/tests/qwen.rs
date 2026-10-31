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
//! - qs14 · 设备端采样复现性：同 prompt + 同 seed ⇒ 逐 token 相同；异 seed ⇒ 不同
//! - qs15 · greedy 零影响：`set_sampling(greedy)` 下 24 token 与原 argmax 逐位相同
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use wein_engine::{
    CoreConfig, EngineCore, EngineEvent, Executor, FinishReason, SampleParams, SamplingParams,
};
use wein_exec::qwen_ref::QwenMirror;
use wein_exec::{argmax, MtpItem, Qwen35Executor};
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

// ── qs6/qs7：Agent 形负载（多轮增长前缀 + 工具结果追加 + 多会话共享系统提示）
//    ——前缀缓存的价值全在这里，块级用例（qs1-qs5）覆盖不到 ──

/// 多轮会话增长：系统提示 + 用户问 → 助手输出 ‖ 工具结果 ‖ 新指示 逐轮追加
/// （工具结果非模型生成，正是 Agent 的 prompt 形态）。断言命中落在已就绪
/// 块界且随轮次加深，且**命中恢复后的续算与全量重放逐 token 位级一致**——
/// 冷参考走抢占回滚（qs3 已钉：抢占=从 0 全量重放），两侧以同样 16-token
/// 块驱动 ⇒ 产样本的步同形（末块余数在 hit 为 16 倍数时相同），位级可比
#[test]
fn qs6_agent_session_growth_hit_equals_replay() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let Some(mut ex) = exec(&dir) else { return };
    let tok = HfTokenizer::open(&dir).unwrap();
    let sys = tok.encode(
        "You are a coding agent with access to tools: read_file, write_file, run_shell, search. \
         Always inspect the relevant code before editing, keep changes minimal, and report exactly \
         which files changed. Prefer small diffs, never delete user data, and ask before any \
         destructive action. If a test fails, read the assertion message before guessing.",
    );
    let q1 = tok.encode("The parser rejects leading whitespace. Find the root cause and fix it.");
    let tool = tok.encode(
        "tool: read_file(path=\"parser.rs\") -> fn parse(input: &str) -> Result<Ast> { \
         let s = input.trim_start_matches(' '); scan(s) }",
    );
    let q2 = tok.encode("Now apply the fix and run the test suite.");
    let q3 = tok.encode("Summarize the diff in one line.");
    let mut convo: Vec<u32> = sys.clone();
    convo.extend_from_slice(&q1);
    assert!(convo.len() > 64 && convo.len() % 16 != 0, "首轮 prompt 须跨多块且非整块（{}）", convo.len());
    let (mut prev, mut last_hit, mut last_len) = (Vec::new(), 0usize, 0usize);
    for turn in 1..=3u32 {
        if turn > 1 {
            convo.extend_from_slice(&prev); // 上一轮助手输出回到上下文
            convo.extend_from_slice(&tool); // 工具结果（非模型生成）
            convo.extend_from_slice(if turn == 2 { &q2 } else { &q3 });
        }
        let l = convo.len();
        assert!(l <= 240, "第 {turn} 轮 prompt 须在 MAX_POS 内（{l}）");
        let hit = ex.admitted(1, &convo);
        assert_eq!(hit % 16, 0, "第 {turn} 轮命中须为整块（hit={hit}）");
        assert!(hit < l, "命中边界须严格小于 prompt 长（hit={hit} l={l}）");
        if turn == 1 {
            assert_eq!(hit, 0, "首轮无共享前缀");
        } else {
            assert!(hit > last_hit, "会话增长 → 命中加深（{last_hit} → {hit}）");
            assert!(hit >= last_len / 16 * 16, "上一轮已就绪的整块须全命中（hit={hit} last={last_len}）");
        }
        // 暖路径：从 hit 起 16-token 块驱动到末块（末块带采样），再 3 步 decode
        let mut rem = l - hit;
        while rem > 16 {
            let _ = ex.step(&StepPlan { prefill: vec![(1, 16)], decode: vec![], batch: vec![] });
            rem -= 16;
        }
        let t = ex.step(&StepPlan { prefill: vec![(1, rem)], decode: vec![1], batch: vec![] });
        let mut warm = vec![t[0].1];
        for _ in 0..3 {
            let d = ex.step(&StepPlan { prefill: vec![], decode: vec![1], batch: vec![] });
            warm.push(d[0].1);
        }
        // 冷参考：抢占回滚 → 从 0 逐块全量重放（同形）
        ex.preempted(1);
        let mut rem = l;
        while rem > 16 {
            let _ = ex.step(&StepPlan { prefill: vec![(1, 16)], decode: vec![], batch: vec![] });
            rem -= 16;
        }
        let t = ex.step(&StepPlan { prefill: vec![(1, rem)], decode: vec![1], batch: vec![] });
        let mut cold = vec![t[0].1];
        for _ in 0..3 {
            let d = ex.step(&StepPlan { prefill: vec![], decode: vec![1], batch: vec![] });
            cold.push(d[0].1);
        }
        assert_eq!(warm, cold, "第 {turn} 轮：命中恢复 ≡ 全量重放（同形驱动，hit={hit}）");
        prev = warm;
        last_hit = hit;
        last_len = l;
        ex.retain(&[]); // 移交：本轮全部块（命中段 unpin、新段进树）
    }
}

/// 多会话共享系统提示（Agent 部署的典型形态：同一 system + tools，不同问）：
/// 系统提示段须整段命中；反复 admit/移交不得漏页漏槽——页池 512 页是硬上限，
/// 记账漏一页就随轮次单调涨，故钉 pages_used 的稳态
#[test]
fn qs7_shared_system_prompt_many_sessions() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let Some(mut ex) = exec(&dir) else { return };
    let tok = HfTokenizer::open(&dir).unwrap();
    let mut sys = tok.encode(
        "You are a coding agent with access to tools: read_file, write_file, run_shell, search. \
         Always inspect the relevant code before editing, keep changes minimal, and report exactly \
         which files changed. Prefer small diffs, never delete user data, and ask before any \
         destructive action. If a test fails, read the assertion message before guessing.",
    );
    let blocks = sys.len() / 16;
    sys.truncate(blocks * 16); // 系统提示整块对齐 → 命中深度可精确钉
    assert!(blocks >= 4, "系统提示须跨多块（{}）", blocks);
    let qs: Vec<Vec<u32>> = [
        "Why does the parser reject leading whitespace?",
        "Fix the failing test in ci/lint.rs",
        "Add a regression test for the parser fix",
        "Explain the diff to the user in two lines",
    ]
    .iter()
    .map(|s| tok.encode(s))
    .collect();
    let mut steady: Option<usize> = None;
    for round in 0..16u32 {
        for (s, q) in qs.iter().enumerate() {
            let id = (round as u64 * 8) + s as u64 + 1;
            let mut p = sys.clone();
            p.extend_from_slice(q);
            let hit = ex.admitted(id, &p);
            assert_eq!(hit % 16, 0, "命中须为整块（hit={hit}）");
            // 首个请求（round 0 / 会话 0）建树时无前缀可命中；此后每个会话
            // 都共享同一段系统提示，须整段命中
            let shared_prefix = round > 0 || s > 0;
            if shared_prefix {
                assert!(hit >= blocks * 16, "第 {round} 轮会话 {s}：系统提示段须全命中（hit={hit}）");
            }
            let mut rem = p.len() - hit;
            while rem > 16 {
                let _ = ex.step(&StepPlan { prefill: vec![(id, 16)], decode: vec![], batch: vec![] });
                rem -= 16;
            }
            let t = ex.step(&StepPlan { prefill: vec![(id, rem)], decode: vec![id], batch: vec![] });
            assert_eq!(t.len(), 1, "每会话产出 1 token");
            ex.retain(&[]);
        }
        let used = ex.pages_used();
        eprintln!("[qs7] round {round}: pages_used={used}");
        match steady {
            // 首轮建树 + 次轮钉稳态；此后不许再涨（漏页/漏 unpin 会单调累积）
            None => {}
            Some(u) if round == 1 => {
                assert_eq!(used, u, "第 2 轮页占用须与第 1 轮持平（round {round}）");
            }
            Some(u) => assert_eq!(used, u, "页占用稳态（round {round}: {used} vs {u}）"),
        }
        if round == 1 {
            steady = Some(used);
        }
    }
}

// ── qs8：MTP 草稿层（多 token 预测）教师强制一致率 ──

/// MTP 草稿层单层前向的 schema 证据：给定位置 p 的目标末层 hidden `h_p` 与
/// 位置 p+1 的 token `t_{p+1}`，草稿层应预测 `t_{p+2}`。一条真文本上逐位
/// 教师强制（target 全在上下文里，草稿层学到的就是"下一个 token"）的一致率
/// 就是 schema 是否逐项对齐的判据：结构错一处（拼接序 / q‖gate 交错 / pre-norm
/// 漏重 / rope 位置偏移）一致率掉到个位数百分比（随机量级，见下实测）。
///
/// 驱动方式：chunk 16 的前向把整段 prompt 喂完（模拟服务侧 chunked prefill，
/// 同时让 hidden 行落在每步的上限内），**每喂完一个 chunk 就对这 16 行发一次
/// `mtp_draft`**——MTP 的 KV 链（chain 0）跨 chunk 累积，项 (p, h 行, t_{p+1})
/// 的因果上下文 = 链内 0..=p，与真实草稿解码逐位同构。注意 `last_hidden`
/// 只保上一步的行（hidden 不跨步留在设备上，主前向的 scratch 是逐级覆写的），
/// 故教师强制的 h 行号 = 该 chunk 内的行号（绝对位置另由 item.pos 给出）。
///
/// 实测（本机 Qwen3.5-4B / RTX 4090 / WSL，2026-10-31，prompt = 下面这段
/// 122 token，n_cmp = T-2 = 120）：
/// - **正确组合**（cat = [embed ‖ hidden]，q‖gate 按头交错）：**45.0%**（54/120）
///   分段 [0,16)44%｜[16,32)44%｜[32,48)31%｜[48,64)50%｜[64,80)50%｜[80,96)44%
///   ｜[96,112)44%｜[112,120)62%——后期上下文更充分略高，整体与位置弱相关
/// - 仅拼接序反（消融：临时把 cat 改成 [hidden ‖ embed]）：**0.0%**（0/120，随机量级）
/// - 仅 q‖gate 错读法（消融：前 16 行 q / 后 16 行 gate）：
///   **4.2%**（5/120，随机量级）
/// - 两者同时错：**0.0%**（0/120）
/// 即两侧各自都是"全对/全错"的开关（45% ↔ ≈0%），拼接序与 q‖gate 交错这两处
/// 正是 schema 最易踩错的地方，本条即证据。另做了一次自注意消融（每项独立
/// 链、pos=0 → 只看得见自己）：总一致率仍 45.0% 但逐位输出全变——草稿头主要
/// 吃 target hidden + next-token embed，草稿链自注意对 top-1 一致率贡献在噪声内。
/// pre-norm 语义的现场旁证：mtp.pre_fc_norm_* 权重全负（min -0.55 / -0.43，
/// 均 -0.41 / -0.25），只有零中心 (1+w) 读法才给出合理的 ≈0.5-1.5 缩放；
/// 普通 RMSNorm（w 直乘）语义下这层是"整向量反号"，不可能训练出来。
#[test]
fn qs8_mtp_teacher_forcing_agreement() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let Some(mut ex) = exec(&dir) else { return };
    if !ex.has_mtp() {
        eprintln!("SKIP（checkpoint 无 MTP 草稿层权重 mtp.*）");
        return;
    }
    let tok = HfTokenizer::open(&dir).unwrap();
    let mut p = tok.encode(
        "The quick brown fox jumps over the lazy dog while the sun sets behind the distant mountains, \
         and the river flows quietly through the ancient forest where birds sing their evening songs \
         before night falls over the sleeping village. In the spring, farmers plant wheat and barley \
         in the wide fields, and by late summer the harvest fills the barns with golden grain. The \
         village school has one room, a wooden stove, and a shelf of books that the teacher brought \
         from the city many years ago. Every winter the children walk through the snow to reach it, \
         carrying their lunch in small metal boxes and laughing at the cold wind.",
    );
    assert!(p.len() >= 120 && p.len() <= 200, "prompt 须 120-200 token（实得 {}）", p.len());
    p.truncate(200);
    let t_len = p.len();
    let n_cmp = t_len - 2; // p = 0..=T-3（要 t_{p+1} 与 t_{p+2} 都在上下文里）
    ex.admitted(1, &p);

    const CHUNK: usize = 16;
    let mut draft = vec![0u32; n_cmp];
    let mut rows_seen = 0usize;
    let mut start = 0usize;
    while start < t_len {
        let n = CHUNK.min(t_len - start);
        let st = ex.step(&StepPlan { prefill: vec![(1, n)], decode: vec![], batch: vec![] });
        assert!(st.is_empty(), "chunked prefill 步不采样");
        let (t, hid) = ex.last_hidden();
        assert_eq!(t, n, "last_hidden 行数 = 本 chunk 行数");
        assert_eq!(hid.len(), n * 2560, "hidden 宽度 = 2560");
        rows_seen += t;
        // 本 chunk 覆盖的 p：绝对位置 start..min(start+n, T-2)（上界留 t_{p+2}）
        let hi = (start + n).min(n_cmp);
        let items: Vec<MtpItem> = (start..hi)
            .map(|q| MtpItem {
                kv_page_row: 0,
                pos: q as u32,
                h: (q - start) as u32,
                next_tok: p[q + 1],
            })
            .collect();
        if !items.is_empty() {
            let out = ex.mtp_draft(&items);
            assert_eq!(out.len(), items.len(), "每项一个草稿 token");
            draft[start..hi].copy_from_slice(&out);
        }
        start += n;
    }
    assert_eq!(rows_seen, t_len, "chunked prefill 走完 → hidden 行数 = T");

    let hits = (0..n_cmp).filter(|&q| draft[q] == p[q + 2]).count();
    let acc = hits as f64 / n_cmp as f64;
    eprintln!("qs8 MTP 教师强制一致率 {:.1}%（{hits}/{n_cmp}）", acc * 100.0);
    // 分段一致率（自注意上下文长度 = pos，前几段上下文极短——分段读数既是
    // schema 证据也是"短上下文先天弱"的归档）
    let mut segs = Vec::new();
    let mut q = 0usize;
    while q < n_cmp {
        let hi = (q + 16).min(n_cmp);
        let h = (q..hi).filter(|&i| draft[i] == p[i + 2]).count();
        segs.push(format!("[{q},{hi}) {:.0}%", h as f64 / (hi - q) as f64 * 100.0));
        q = hi;
    }
    eprintln!("  分段：{}", segs.join("｜"));
    for q in [0usize, 31, 63, 95, n_cmp - 1] {
        eprintln!(
            "  样本 p={q}：期望 {:?} / 草稿 {:?}（{}）",
            tok.decode(&[p[q + 2]]),
            tok.decode(&[draft[q]]),
            if draft[q] == p[q + 2] { "命中" } else { "未中" }
        );
    }
    ex.mtp_reset();
    assert!(acc >= 0.40, "MTP 教师强制一致率 {:.1}% < 40%（schema 未对齐？）", acc * 100.0);
}

// ── qs9：投机解码基础原语——逐位置 state 行（零拷贝回滚）+ spec_verify ──

/// A 的 state 行（接受位落定行）与 B 的普通步 state 行逐位比（u16 位级）。
/// 只读 2 个 lin 层（1MB/行）+ 全部 24 个 conv 层（48KB/行）——整池回读是
/// 128MB×24 层，禁
fn assert_state_rows_bitwise(ex: &Qwen35Executor, a_id: u64, b_id: u64) {
    let ra = ex.state_row_of(a_id).expect("A 无状态行");
    let rb = ex.state_row_of(b_id).expect("B 无状态行");
    for (li, _) in [(0usize, 0usize), (12, 0)] {
        let (va, vb) = (ex.lin_state_row_bits(li, ra), ex.lin_state_row_bits(li, rb));
        assert_eq!(va.len(), vb.len(), "lin{li} 行宽");
        for i in 0..va.len() {
            assert_eq!(
                va[i], vb[i],
                "lin{li} 行位不一致 @{i}（A 行 {ra} {:#06x} vs B 行 {rb} {:#06x}）",
                va[i], vb[i]
            );
        }
    }
    for li in 0..24usize {
        let (va, vb) = (ex.conv_state_row_bits(li, ra), ex.conv_state_row_bits(li, rb));
        for i in 0..va.len() {
            assert_eq!(va[i], vb[i], "conv{li} 行位不一致 @{i}（A 行 {ra} vs B 行 {rb}）");
        }
    }
}

/// qs9 · 投机验证的逐位置 state 行（MTP 基础原语）：请求 A 真草稿链一轮
/// `spec_verify(id, &[d1,d2,d3])`（链占连续 cnt+1 = 5 行，token i 读行 +i、
/// 写 +i+1），接受 a 个草稿 ⇒ 落定行 base+1+a 拷回 A 的状态行。断言：
///
/// 1. 草稿取真贪心续写（由同 prompt 的请求 C 用普通步产出）⇒ 全接受
///    （a = 3）——验证"链式一次前向 ≡ 逐 token 前向"的位级等价；
/// 2. A 的 `target_tokens[k]` = B 普通步第 k 个采样结果（位级，k = 0..=a）
///    ——"验证看到的就是真前向"；
/// 3. A 的状态行 ≡ B 推进 1+a 个普通步后的状态行（2 lin + 24 conv 层逐位）。
///
/// 批形状对齐（位级合同的前提）：A 的验证前向 = 一条链 4 行一次前向（t=4）；
/// B/C 的普通步用 3 个旁路请求凑满 4 行批（t=4）。cuBLASLt 跨 M 换 algo 会
/// 破位级合同（临时实验：m=1 与 m=4 同一行有 1/2048 位不同），所以两侧必须
/// 同 t。旁路请求只负责占位（状态/采样丢弃）。
#[test]
fn qs9_spec_state_rows_bitwise() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let Some(mut ex) = exec(&dir) else { return };
    let tok = HfTokenizer::open(&dir).unwrap();
    let mut p = tok.encode(
        "The quick brown fox jumps over the lazy dog while the sun sets behind the distant \
         mountains, and the river flows quietly through the ancient forest where birds sing \
         to one another in the cool evening air, and every leaf on every branch trembles \
         softly as the wind moves through the tall trees toward the open meadow beyond",
    );
    assert!(p.len() >= 40, "prompt 须 ≥ 40 token（实得 {}）", p.len());
    p.truncate(40);
    let l = p.len();

    // chunked prefill（16+16+8，末段带采样位）——A/B/C 同形，状态逐位同
    let prefill = |ex: &mut Qwen35Executor, id: u64| -> u32 {
        ex.admitted(id, &p);
        let mut rem = l;
        while rem > 16 {
            let st = ex.step(&StepPlan { prefill: vec![(id, 16)], decode: vec![], batch: vec![] });
            assert!(st.is_empty(), "chunked prefill 步不采样");
            rem -= 16;
        }
        let t = ex.step(&StepPlan { prefill: vec![(id, rem)], decode: vec![id], batch: vec![] });
        t[0].1
    };
    let (ta, tb, tc) = (prefill(&mut ex, 1), prefill(&mut ex, 2), prefill(&mut ex, 3));
    assert_eq!(ta, tb, "A/B 同形 prefill 的末位采样须一致");
    assert_eq!(ta, tc, "A/C 同形 prefill 的末位采样须一致");
    // 旁路请求（凑批形状用）：短 prompt，自持 token 流（每步推 1 个、采 1 个）
    for id in 4u64..=6 {
        ex.admitted(id, &p[..16]);
        let _ = ex.step(&StepPlan { prefill: vec![(id, 16)], decode: vec![id], batch: vec![] });
    }
    let pad = [4u64, 5, 6];
    // 普通步驱动：目标请求 1 行 + 3 旁路行 = t=4（与 A 的链批同形）
    let plain_step = |ex: &mut Qwen35Executor, id: u64| -> u32 {
        let mut dec = vec![id];
        dec.extend_from_slice(&pad);
        let out = ex.step(&StepPlan { prefill: vec![(id, 1)], decode: dec, batch: vec![] });
        out.iter().find(|(r, _)| *r == id).expect("目标请求须采样").1
    };
    // C 先走 3 个普通步 → 真贪心续写 d1..d3（作 A 的草稿）
    let drafts: Vec<u32> = (0..3).map(|_| plain_step(&mut ex, 3)).collect();
    // A：投机验证链（cnt = 4 行一次前向，spec==1 逐位置落盘）
    let r = ex.spec_verify(1, &drafts).expect("链行分配失败");
    assert_eq!(r.target_tokens.len(), drafts.len() + 1, "每链位置一个 argmax");
    assert_eq!(r.draft_tokens.len(), r.accepted, "draft_tokens = 接受的草稿");
    assert_eq!(r.accepted, 3, "真贪心续写作草稿须全接受（实得 {}）", r.accepted);
    for (k, &d) in drafts.iter().enumerate() {
        assert_eq!(r.target_tokens[k], d, "链位 {k} 的目标 token 须等于草稿 {d}");
    }
    // B：普通步推进 1+a 个 token（每步 t=4，与 A 的链批同形）
    let b_toks: Vec<u32> = (0..=r.accepted).map(|_| plain_step(&mut ex, 2)).collect();
    // 2. target_tokens（每位置 argmax）= B 逐 token 采样（位级）
    assert_eq!(
        r.target_tokens, b_toks,
        "A 的逐位置 argmax 须与 B 的普通步采样逐 token 相同"
    );
    // 3. state 行逐位：A 的接受位落定行 ≡ B 推进 1+a 个普通步后的行
    assert_state_rows_bitwise(&ex, 1, 2);
    eprintln!(
        "qs9 投机验证：drafts={:?} accepted={} target={:?}（A 行 {} ≡ B 行 {}）",
        drafts,
        r.accepted,
        r.target_tokens,
        ex.state_row_of(1).unwrap(),
        ex.state_row_of(2).unwrap()
    );
    // 记账自洽：A 的待喂游标 = 旧游标 + 1 + accepted（下一步喂奖励 token）
    let nxt = ex.spec_verify(1, &[]).expect("空草稿一轮（单 token 验证）");
    assert_eq!(nxt.accepted, 0);
    assert_eq!(nxt.draft_tokens.len(), 0);
    eprintln!("qs9 后续单 token 验证：target={:?}", nxt.target_tokens);
}

// ── qs10/qs11：投机解码端到端（step 接线 + per-span spec 隔离性）──

/// 一次 step 产出的 token 按 id 归并（同一请求一步可多个 token——投机步）
fn collect(out: &[(u64, u32)], got: &mut Vec<(u64, Vec<u32>)>) {
    for &(id, tok) in out {
        match got.iter_mut().find(|(r, _)| *r == id) {
            Some((_, v)) => v.push(tok),
            None => got.push((id, vec![tok])),
        }
    }
}

fn toks_of(got: &[(u64, Vec<u32>)], id: u64) -> Vec<u32> {
    got.iter().find(|(r, _)| *r == id).map(|(_, v)| v.clone()).unwrap_or_default()
}

/// qs10 · 投机解码的正确性合同：同一 prompt 两个请求 A(spec_k=4) / B(spec_k=0)
/// 各自贪心生成 24 token，**逐 token 相同**——草稿只是猜测，验证 + 逐位置
/// state 回滚保证输出与普通 decode 逐位一致（分布不变）。
///
/// 批形状说明：A 的每步是"一条链 5 行"的 eager 前向，B 的每步是 1 行 eager
/// 前向；cuBLASLt 跨 M 换 algo 会有 ~1/2048 的位差，所以此条只断言 token
/// 序列（argmax 决策）相同，不拿 logits 位级对拍。radix 关掉（`set_radix`）：
/// 同 prompt 的前缀缓存命中会让 B 的 prefill 形状与 A 不同。
#[test]
fn qs10_spec_greedy_matches_plain_bitwise() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let Some(mut ex) = exec(&dir) else { return };
    if !ex.has_mtp() {
        eprintln!("SKIP（checkpoint 无 MTP 草稿层权重 mtp.*）");
        return;
    }
    ex.set_radix(false);
    let tok = HfTokenizer::open(&dir).unwrap();
    let mut p = tok.encode(
        "The quick brown fox jumps over the lazy dog while the sun sets behind the distant \
         mountains and the river flows quietly through the ancient forest",
    );
    p.truncate(24);
    let l = p.len();
    const N: usize = 24;

    // A = 投机（k=4），B = 普通
    ex.admitted(1, &p);
    ex.admitted(2, &p);
    // 固定 k 合同：自适应策略（`set_spec` 默认开）会让 k 随 EMA 每轮变，
    // 批形状（链行数 k+1）随之变——跨 M 的 cuBLASLt algo 差异会破位级合同。
    // 策略本身由探针（[4] 表）测，这里关掉
    ex.set_spec_policy(false);
    ex.set_spec(1, 4);
    ex.spec_stats_reset();
    let ta = ex.step(&StepPlan { prefill: vec![(1, l)], decode: vec![1], batch: vec![] });
    let tb = ex.step(&StepPlan { prefill: vec![(2, l)], decode: vec![2], batch: vec![] });
    assert_eq!(ta.len(), 1, "prefill 追平步出首 token");
    assert_eq!(tb.len(), 1, "prefill 追平步出首 token");
    let mut got: Vec<(u64, Vec<u32>)> = Vec::new();
    collect(&ta, &mut got);
    collect(&tb, &mut got);
    assert_eq!(toks_of(&got, 1), toks_of(&got, 2), "首 token（prefill 末位采样）须相同");

    // 交替推进：A 每步 1..=k+1 个 token，B 每步 1 个
    let mut per_step: Vec<usize> = Vec::new();
    while toks_of(&got, 1).len() < N || toks_of(&got, 2).len() < N {
        if toks_of(&got, 1).len() < N {
            let out = ex.step(&StepPlan { prefill: vec![], decode: vec![1], batch: vec![] });
            let n = out.iter().filter(|(id, _)| *id == 1).count();
            assert!((1..=5).contains(&n), "投机步须出 1..=k+1 个 token（实得 {n}）");
            per_step.push(n);
            collect(&out, &mut got);
        }
        if toks_of(&got, 2).len() < N {
            let out = ex.step(&StepPlan { prefill: vec![], decode: vec![2], batch: vec![] });
            collect(&out, &mut got);
        }
    }
    let (a, b) = (toks_of(&got, 1), toks_of(&got, 2));
    assert_eq!(&a[..N], &b[..N], "投机序列须与普通贪心逐 token 相同");
    let (rounds, drafts, accepted, fallbacks) = ex.spec_stats();
    let acc = accepted as f64 / drafts.max(1) as f64;
    eprintln!(
        "qs10 投机 vs 普通：{N} token 逐 token 相同；接受率 {:.1}%（{accepted}/{drafts}，{rounds} 轮，退化 {fallbacks}）",
        acc * 100.0
    );
    let mut hist: Vec<(usize, usize)> = Vec::new();
    for &n in &per_step {
        match hist.iter_mut().find(|(k, _)| *k == n) {
            Some((_, c)) => *c += 1,
            None => hist.push((n, 1)),
        }
    }
    hist.sort();
    eprintln!(
        "  每步出 token 分布：{}（共 {} 步 decode）",
        hist.iter().map(|(n, c)| format!("{n}个×{c}")).collect::<Vec<_>>().join("，"),
        per_step.len()
    );
}

/// qs11 · 混合步隔离性（per-span spec）：一步里 2 个投机请求 + 2 个普通请求
/// + 1 个 prefill chunk 同批推进，所有请求的输出仍与其"单独跑（spec 关）"
/// 的贪心序列逐 token 相同。
///
/// 单执行器内两趟：先单独跑参考（spec 关、每请求一行），再同批跑混合步。
/// `set_radix(false)` 让第二趟的准入不做前缀命中（否则参考趟的 prompt 块会
/// 被复用，前向批形状变化）。
#[test]
fn qs11_spec_mixed_batch() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let Some(mut ex) = exec(&dir) else { return };
    if !ex.has_mtp() {
        eprintln!("SKIP（checkpoint 无 MTP 草稿层权重 mtp.*）");
        return;
    }
    ex.set_radix(false);
    let tok = HfTokenizer::open(&dir).unwrap();
    let base = tok.encode(
        "The quick brown fox jumps over the lazy dog while the sun sets behind the distant \
         mountains and the river flows quietly through the ancient forest where birds sing \
         their evening songs before night falls over the sleeping village, and the farmers \
         walk home along the narrow road with lanterns in their hands",
    );
    assert!(base.len() >= 36, "prompt 须 ≥ 36 token（实得 {}）", base.len());
    const N: usize = 6; // 每请求比较的 token 数（形状差 1/2048 的容忍窗）
    // 5 个不同 prompt（长度各异，prefill chunk 只有 E 留尾）
    let prompts: Vec<Vec<u32>> = (0..5u32)
        .map(|i| {
            let mut q = base.clone();
            q.truncate(20 + 4 * i as usize);
            q.rotate_left(i as usize * 3);
            q
        })
        .collect();

    // ── 第一趟：参考序列（5 个请求各自单独跑，spec 关）──
    let mut reference: Vec<Vec<u32>> = Vec::new();
    for (i, q) in prompts.iter().enumerate() {
        let id = 100 + i as u64;
        ex.admitted(id, q);
        let l = q.len();
        let mut got: Vec<(u64, Vec<u32>)> = Vec::new();
        // E（最后一例）按 16+尾 分两段 prefill，与混合趟同形（首段不带采样位）
        if i == 4 {
            let o = ex.step(&StepPlan { prefill: vec![(id, 16)], decode: vec![], batch: vec![] });
            assert!(o.is_empty(), "chunked prefill 中间段不采样");
            let o = ex.step(&StepPlan {
                prefill: vec![(id, l - 16)],
                decode: vec![id],
                batch: vec![],
            });
            collect(&o, &mut got);
        } else {
            let o =
                ex.step(&StepPlan { prefill: vec![(id, l)], decode: vec![id], batch: vec![] });
            collect(&o, &mut got);
        }
        while toks_of(&got, id).len() < N + 1 {
            let o = ex.step(&StepPlan { prefill: vec![], decode: vec![id], batch: vec![] });
            collect(&o, &mut got);
        }
        reference.push(toks_of(&got, id)[..N].to_vec());
    }
    ex.retain(&[]); // 行/状态/页全回收
    ex.spec_stats_reset();

    // ── 第二趟：混合批（A/B 投机 k=3，C/D 普通，E 带 16 token 的 prefill 尾段）──
    let ids: Vec<u64> = (1..=5).collect();
    for (i, q) in prompts.iter().enumerate() {
        ex.admitted(ids[i], q);
    }
    ex.set_spec_policy(false); // 固定 k=3（理由同 qs10）
    ex.set_spec(1, 3);
    ex.set_spec(2, 3);
    // A..D 各自 prefill 追平并出首 token（E 只喂前 16 token，尾段留给混合步）
    let mut got: Vec<(u64, Vec<u32>)> = Vec::new();
    for i in 0..4 {
        let (id, l) = (ids[i], prompts[i].len());
        let o = ex.step(&StepPlan { prefill: vec![(id, l)], decode: vec![id], batch: vec![] });
        collect(&o, &mut got);
    }
    let oe = ex.step(&StepPlan { prefill: vec![(5, 16)], decode: vec![], batch: vec![] });
    assert!(oe.is_empty(), "chunked prefill 中间段不采样");
    // 混合步 1：2 投机 + 2 普通 + 1 prefill 尾段
    let e_tail = prompts[4].len() - 16;
    let o = ex.step(&StepPlan {
        prefill: vec![(5, e_tail)],
        decode: ids.clone(),
        batch: ids.clone(),
    });
    collect(&o, &mut got);
    // 继续纯 decode 混合步（投机与普通同批）
    while ids.iter().any(|&id| toks_of(&got, id).len() < N) {
        let o = ex.step(&StepPlan { prefill: vec![], decode: ids.clone(), batch: ids.clone() });
        collect(&o, &mut got);
    }
    for (i, id) in ids.iter().enumerate() {
        let want = &reference[i];
        let have = toks_of(&got, *id);
        assert_eq!(
            &have[..N],
            &want[..],
            "请求 {id}（{}）的输出须与单独跑（spec 关）逐 token 相同",
            if i < 2 { "投机 k=3" } else if i < 4 { "普通" } else { "prefill 尾段" }
        );
    }
    let (rounds, drafts, accepted, fallbacks) = ex.spec_stats();
    eprintln!(
        "qs11 混合批：5 请求 × {N} token 全部与单独跑逐 token 相同；投机接受率 {:.1}%（{accepted}/{drafts}，{rounds} 轮，退化 {fallbacks}）",
        accepted as f64 / drafts.max(1) as f64 * 100.0
    );
}

/// qs12 · **多链共享一次前向**（批化投机的核心合同）：一步里 ≥4 条投机链
/// 同时起草（k=3）与验证（一条目标前向覆盖 4×4 = 16 行），每条链的输出仍
/// 与其"单独跑、spec 关"的贪心序列逐 token 相同。
///
/// 三件事各钉一处：
/// 1. **链 KV 不互踩**：4 条链的草稿槽（`mtp_chain` = `mtp_bt` 页行）必须两两
///    不同——v1 全部写死 `SPEC_CHAIN_ROW = 0`，多链会互相覆盖草稿 KV；
/// 2. **验证批的 per-span 切分**：4 条链各自用真贪心续写作草稿，一次
///    `spec_commit_batch` 把 4 条链放进**同一条前向**，每条链都须 3/3 全接受、
///    逐位置 argmax = 参考序列（段切错/链内 pos 写错就会掉接受）；
/// 3. **step 级批化**：4 个 `spec_k = 3` 请求同一步 decode，`spec_chain_max`
///    记到 4，输出与单独跑逐 token 相同。
#[test]
fn qs12_spec_multi_chain_batch() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let Some(mut ex) = exec(&dir) else { return };
    if !ex.has_mtp() {
        eprintln!("SKIP（checkpoint 无 MTP 草稿层权重 mtp.*）");
        return;
    }
    ex.set_radix(false);
    let tok = HfTokenizer::open(&dir).unwrap();
    let base = tok.encode(
        "The quick brown fox jumps over the lazy dog while the sun sets behind the distant \
         mountains and the river flows quietly through the ancient forest where birds sing \
         their evening songs before night falls over the sleeping village, and the farmers \
         walk home along the narrow road with lanterns in their hands",
    );
    assert!(base.len() >= 36, "prompt 须 ≥ 36 token（实得 {}）", base.len());
    const R: usize = 4; // 同一步里同时投机的链数
    const K: u32 = 3; // 每条链的草稿数
    const N: usize = 6; // 每请求比较的 token 数
    // 4 个不同 prompt（长度/相位各异 ⇒ 4 条互不相干的链）
    let prompts: Vec<Vec<u32>> = (0..R)
        .map(|i| {
            let mut q = base.clone();
            q.truncate(20 + 4 * i);
            q.rotate_left(i * 5);
            q
        })
        .collect();

    // ── 第一趟：参考序列（4 个请求各自单独跑，spec 关）──
    let mut reference: Vec<Vec<u32>> = Vec::new();
    for (i, q) in prompts.iter().enumerate() {
        let id = 100 + i as u64;
        ex.admitted(id, q);
        let l = q.len();
        let mut got: Vec<(u64, Vec<u32>)> = Vec::new();
        let o = ex.step(&StepPlan { prefill: vec![(id, l)], decode: vec![id], batch: vec![] });
        collect(&o, &mut got);
        while toks_of(&got, id).len() < N + 1 {
            let o = ex.step(&StepPlan { prefill: vec![], decode: vec![id], batch: vec![] });
            collect(&o, &mut got);
        }
        reference.push(toks_of(&got, id)[..N + 1].to_vec());
    }
    ex.retain(&[]);
    ex.spec_stats_reset();

    // ── 第二趟 A：验证批（4 条链一条前向，真贪心续写作草稿 ⇒ 须全接受）──
    let ids: Vec<u64> = (1..=R as u64).collect();
    for (i, q) in prompts.iter().enumerate() {
        ex.admitted(ids[i], q);
    }
    for i in 0..R {
        let (id, l) = (ids[i], prompts[i].len());
        let _ = ex.step(&StepPlan { prefill: vec![(id, l)], decode: vec![id], batch: vec![] });
    }
    let items: Vec<(u64, Vec<u32>)> =
        ids.iter().enumerate().map(|(i, &id)| (id, reference[i][1..4].to_vec())).collect();
    let rounds = ex.spec_commit_batch(&items);
    assert_eq!(rounds.len(), R, "每条链一个结果");
    for (i, r) in rounds.iter().enumerate() {
        let r = r.as_ref().expect("链行分配失败（池里须有连续行）");
        assert_eq!(
            r.accepted,
            K as usize,
            "链 {i} 用真贪心续写作草稿须 {K}/{K} 全接受（实得 {} ⇒ 4 链共享前向的段切分/链内 pos 错）",
            r.accepted
        );
        assert_eq!(
            r.target_tokens,
            reference[i][1..5].to_vec(),
            "链 {i} 的逐位置 argmax 须等于它自己单独跑的贪心序列"
        );
    }
    assert!(
        ex.spec_chain_batch_max() >= R as u32,
        "一条前向须验证 ≥{R} 条链（实得 {}）",
        ex.spec_chain_batch_max()
    );
    ex.retain(&[]);
    ex.spec_stats_reset();

    // ── 第二趟 B：step 级批化（4 个 spec_k=3 请求同步 decode）──
    ex.set_spec_policy(false); // 固定 k=K（理由同 qs10）
    for (i, q) in prompts.iter().enumerate() {
        ex.admitted(ids[i], q);
        ex.set_spec(ids[i], K);
    }
    // 链槽（MTP KV 页行）两两不同：多链同批前向的隔离前提
    let mut slots: Vec<u32> = ids.iter().filter_map(|&id| ex.spec_chain_of(id)).collect();
    assert_eq!(slots.len(), R, "每条投机请求须领到一个草稿链槽（实得 {}）", slots.len());
    slots.sort_unstable();
    slots.dedup();
    assert_eq!(slots.len(), R, "草稿链槽须两两不同——否则多链草稿 KV 互相覆盖");

    let mut got: Vec<(u64, Vec<u32>)> = Vec::new();
    for i in 0..R {
        let (id, l) = (ids[i], prompts[i].len());
        let o = ex.step(&StepPlan { prefill: vec![(id, l)], decode: vec![id], batch: vec![] });
        collect(&o, &mut got);
    }
    // 一条步里 4 条链同时起草 + 验证
    let o = ex.step(&StepPlan { prefill: vec![], decode: ids.clone(), batch: ids.clone() });
    assert!(
        ex.spec_chain_batch_max() >= R as u32,
        "同一步须验证 ≥{R} 条链（实得 {}）",
        ex.spec_chain_batch_max()
    );
    assert!(
        (R..=R * (K as usize + 1)).contains(&o.len()),
        "4 条链一步须出 {R}..={} 个 token（实得 {}）",
        R * (K as usize + 1),
        o.len()
    );
    collect(&o, &mut got);
    for &id in &ids {
        assert!(toks_of(&got, id).len() >= 2, "请求 {id} 的首个多链步须出 token");
    }
    while ids.iter().any(|&id| toks_of(&got, id).len() < N) {
        let o = ex.step(&StepPlan { prefill: vec![], decode: ids.clone(), batch: ids.clone() });
        collect(&o, &mut got);
    }
    for (i, &id) in ids.iter().enumerate() {
        let have = toks_of(&got, id);
        assert_eq!(
            &have[..N],
            &reference[i][..N],
            "链 {i}（{id}）的输出须与单独跑（spec 关）逐 token 相同"
        );
    }
    let (r, drafts, accepted, fallbacks) = ex.spec_stats();
    eprintln!(
        "qs12 多链同批：{R} 链 × {N} token 与单独跑逐 token 相同；槽 {:?}；接受率 {:.1}%（{accepted}/{drafts}，{r} 轮，退化 {fallbacks}，峰值链数 {}）",
        ex.spec_chain_of(1),
        accepted as f64 / drafts.max(1) as f64 * 100.0,
        ex.spec_chain_batch_max()
    );
}

// ── qs13：换行零拷贝的多轮合同（行账不漂移 / 不串行 / 不泄漏）──

/// 行账守恒：`[0, next_row)` 恰好 = 活请求行（块表行 ∪ 状态行）∪ free_rows，
/// 且两者不相交、free 表无重复。任何"归还了还占着""同一行发给两个请求"
/// "行凭空消失"都会在这里炸。
fn assert_row_accounting(ex: &Qwen35Executor, live: &[u64]) {
    let (_, _, _, next_row, flen) = ex.spec_row_stats();
    let free = ex.spec_free_rows();
    assert_eq!(free.len(), flen, "spec_row_stats 的 free_rows 长度须与快照一致");
    let mut fset: Vec<u32> = free.clone();
    fset.sort_unstable();
    assert!(fset.windows(2).all(|w| w[0] != w[1]), "free_rows 有重复行：{fset:?}");
    let mut owned: Vec<u32> = Vec::new();
    for &id in live {
        if let Some(r) = ex.block_row_of(id) {
            owned.push(r);
        }
        if let Some(r) = ex.state_row_of(id) {
            owned.push(r);
        }
    }
    owned.sort_unstable();
    owned.dedup();
    for &r in &owned {
        assert!(!fset.contains(&r), "行 {r} 同时在 free_rows 与活请求名下（串行/双发）");
    }
    // 并集覆盖：每个已发过的行号（< next_row）要么属于活请求、要么在空闲表
    let mut bag: Vec<u32> = owned.clone();
    bag.extend_from_slice(&fset);
    bag.sort_unstable();
    bag.dedup();
    assert_eq!(bag.len(), owned.len() + fset.len(), "活请求行与空闲行重叠");
    assert_eq!(
        bag.len(),
        next_row as usize,
        "行账不守恒：活 {} + 空闲 {} ≠ next_row {next_row}（有行泄漏）",
        owned.len(),
        fset.len()
    );
    assert_eq!(*bag.last().unwrap_or(&0) + 1, next_row, "行号越界（≥ next_row）");
}

/// qs13 · **多轮换行落定**：A 连续 3 轮 `spec_verify`（每轮 k=3，真贪心续写
/// 作草稿 ⇒ 全接受），B 走普通步。每轮断言：
///
/// 1. A 的逐位置 argmax = B 的 4 个普通步采样（逐 token）；
/// 2. A 的落定状态行 ≡ B 推进 4 个普通步后的状态行（2 lin + 24 conv 逐 u16）
///    ——换行（`state_row = head+1+accepted`）在多轮下不漂移；
/// 3. `state_row` 每轮前进 1+accepted 行、块表行全程不动，且拷入次数为 0
///    （稳态零拷贝：链首恒等于自己的 state_row）；
/// 4. 行账守恒（`assert_row_accounting`）：free_rows 无重复、不与活请求
///    重叠、并集 = `[0, next_row)`；3 轮之间不泄漏。
///
/// 批形状对齐（位级前提，同 qs9）：A 的链 = 4 行（t=4），B 的普通步用 3 个
/// 旁路请求凑成 t=4。
#[test]
fn qs13_spec_row_swap_multiround() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let Some(mut ex) = exec(&dir) else { return };
    if !ex.has_mtp() {
        eprintln!("SKIP（checkpoint 无 MTP 草稿层权重 mtp.*）");
        return;
    }
    ex.set_radix(false);
    let tok = HfTokenizer::open(&dir).unwrap();
    let mut p = tok.encode(
        "The quick brown fox jumps over the lazy dog while the sun sets behind the distant \
         mountains, and the river flows quietly through the ancient forest where birds sing \
         to one another in the cool evening air, and every leaf on every branch trembles \
         softly as the wind moves through the tall trees toward the open meadow beyond",
    );
    assert!(p.len() >= 40, "prompt 须 ≥ 40 token（实得 {}）", p.len());
    p.truncate(40);
    let l = p.len();
    const K: usize = 3; // 每轮草稿数
    const ROUNDS: usize = 3; // ≥3 轮（漂移/泄漏都在轮间暴露）

    // A/B 同形 prefill（16+16+8，末段带采样位）；旁路 4..6 只占位凑 t=4
    let prefill = |ex: &mut Qwen35Executor, id: u64| -> u32 {
        ex.admitted(id, &p);
        let mut rem = l;
        while rem > 16 {
            let st = ex.step(&StepPlan { prefill: vec![(id, 16)], decode: vec![], batch: vec![] });
            assert!(st.is_empty(), "chunked prefill 步不采样");
            rem -= 16;
        }
        ex.step(&StepPlan { prefill: vec![(id, rem)], decode: vec![id], batch: vec![] })[0].1
    };
    // 准入顺序：先 B/旁路、最后 A——A 的块表行落在已发行号之后（行号后面
    // 就是未分配前沿），链的 case 1（自己的 state 行当链首、零拷入）才成立；
    // 否则 A 的行夹在别人中间，头几轮会走 case 2/3 的 25.2MB 拷入
    let tb = prefill(&mut ex, 2);
    for id in 4u64..=6 {
        ex.admitted(id, &p[..16]);
        let _ = ex.step(&StepPlan { prefill: vec![(id, 16)], decode: vec![id], batch: vec![] });
    }
    let ta = prefill(&mut ex, 1);
    assert_eq!(ta, tb, "A/B 同形 prefill 的末位采样须一致");
    let pad = [4u64, 5, 6];
    // B 的普通步：目标请求 1 行 + 3 旁路行 = t=4（与 A 的链批同形）
    let plain_step = |ex: &mut Qwen35Executor, id: u64| -> u32 {
        let mut dec = vec![id];
        dec.extend_from_slice(&pad);
        let out = ex.step(&StepPlan { prefill: vec![(id, 1)], decode: dec, batch: vec![] });
        out.iter().find(|(r, _)| *r == id).expect("目标请求须采样").1
    };
    let live = [1u64, 2, 4, 5, 6];
    let (bt0, sr0) = (ex.block_row_of(1).unwrap(), ex.state_row_of(1).unwrap());
    assert_eq!(bt0, sr0, "投机前 state_row 恒等块表行");
    ex.spec_stats_reset();
    let mut sr_prev = sr0;
    for round in 0..ROUNDS {
        // B 先走 K 个普通步 → 真贪心续写（作 A 的草稿）
        let drafts: Vec<u32> = (0..K).map(|_| plain_step(&mut ex, 2)).collect();
        let r = ex.spec_verify(1, &drafts).expect("链行分配失败");
        assert_eq!(r.accepted, K, "第 {round} 轮：真贪心续写作草稿须全接受（实得 {}）", r.accepted);
        // B 再走 1 步（= A 的奖励 token 位），凑满 1+accepted 个普通步
        let reward = plain_step(&mut ex, 2);
        let mut want = drafts.clone();
        want.push(reward);
        assert_eq!(r.target_tokens, want, "第 {round} 轮：逐位置 argmax 须 = B 的普通步采样");
        // 状态行逐位（换行落定之后的行 ≡ B 推进 1+a 步后的行）
        assert_state_rows_bitwise(&ex, 1, 2);
        // 换行：state_row 前进 1+accepted，块表行不动
        let (bt, sr) = (ex.block_row_of(1).unwrap(), ex.state_row_of(1).unwrap());
        assert_eq!(bt, bt0, "第 {round} 轮：块表行全程不动（KV 页账与 state 行分离）");
        assert_eq!(
            sr,
            sr_prev + 1 + r.accepted as u32,
            "第 {round} 轮：落定行 = 上一轮状态行 + 1 + accepted（换行零拷贝）"
        );
        sr_prev = sr;
        assert_row_accounting(&ex, &live);
    }
    let (allocs, copyins, _cap, next_row, flen) = ex.spec_row_stats();
    assert_eq!(allocs, ROUNDS as u64, "每轮一次链行分配（实得 {allocs}）");
    assert_eq!(copyins, 0, "稳态零拷入（链首恒 = 自己的 state_row，实得 {copyins} 次）");
    assert_eq!(ex.spec_chain_batch_max(), 1, "单链测试：每轮一条链");
    assert_eq!(ex.spec_stats().0, ROUNDS as u64, "轮次记账 = 实跑轮数");
    eprintln!(
        "qs13 多轮换行：{ROUNDS} 轮 × (1+{K}) token 逐 token + 逐 u16 与普通步相同；\
         state_row {sr0} → {sr_prev}，块表行 {bt0} 不动，拷入 0，next_row {next_row}，free_rows {flen}"
    );
    // 全部退场后行必须全数回收（泄漏会在"空闲表长度 ≠ next_row"上暴露）
    ex.retain(&[]);
    let (_, _, _, next_row, flen) = ex.spec_row_stats();
    assert_eq!(flen, next_row as usize, "退场后所有行须回到空闲表（实得 {flen}/{next_row}）");
    assert_row_accounting(&ex, &[]);
}

// ── qs14/qs15：设备端采样（top-k/top-p）接入 —— 复现性与 greedy 零影响 ──

/// qs14 · 采样复现性：**同 prompt + 同 seed 的两个请求逐 token 相同**，
/// 不同 seed 则不同。
///
/// 四个请求同批推进（同一 GEMM 调用、同一批形状），故 logits 逐位一致——
/// 剩下唯一能影响结果的输入就是 RNG 流。RNG 状态在设备缓冲（槽 = 请求块表
/// 行，准入时零化），图内前进、主机不碰；`seed` 缺省时由全局递增计数器派生。
/// 采样参数 `top_k = 64`（设备端最宽支持集）、`top_p = 1`（不截断）：
/// 这是纯温度采样，24 个 token 撞车的概率可忽略。
#[test]
fn qs14_sampling_seed_reproducible() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let Some(mut ex) = exec(&dir) else { return };
    ex.set_radix(false);
    let tok = HfTokenizer::open(&dir).unwrap();
    let mut p = tok.encode(
        "The quick brown fox jumps over the lazy dog while the sun sets behind the distant mountains",
    );
    p.truncate(24);
    let l = p.len();
    const N: usize = 24;

    let samp = |seed: u64| SampleParams {
        temperature: 1.0,
        top_k: 64,
        top_p: 1.0,
        seed: Some(seed),
    };
    // 采样参数在**准入之前**下发（engine 提交路径的顺序：Submit 时校验、
    // 准入时搬进请求态）
    for (id, seed) in [(1u64, 777u64), (2, 777), (3, 888), (4, 999)] {
        ex.set_sampling(id, samp(seed)).expect("合法采样参数须被接受");
    }
    // 边界：全词表（top_k=0）必须**明确报错**而不是静默退化
    let bad = SampleParams { temperature: 1.0, top_k: 0, top_p: 0.9, seed: None };
    let err = ex.set_sampling(9, bad).expect_err("top_k=0 须被拒绝");
    assert!(err.contains("全词表"), "错误文案须点明全词表：{err}");
    eprintln!("qs14 拒绝全词表采样：{err}");

    for id in 1u64..=4 {
        ex.admitted(id, &p);
    }
    let out = ex.step(&StepPlan {
        prefill: vec![(1, l), (2, l), (3, l), (4, l)],
        decode: vec![1, 2, 3, 4],
        batch: vec![],
    });
    let mut got: Vec<(u64, Vec<u32>)> = Vec::new();
    collect(&out, &mut got);
    assert_eq!(out.len(), 4, "同批 4 个请求各出 1 token（实得 {}）", out.len());

    while (1u64..=4).any(|id| toks_of(&got, id).len() < N) {
        let out = ex.step(&StepPlan { prefill: vec![], decode: vec![1, 2, 3, 4], batch: vec![] });
        collect(&out, &mut got);
    }
    let (a, b, c, d) = (toks_of(&got, 1), toks_of(&got, 2), toks_of(&got, 3), toks_of(&got, 4));
    assert_eq!(&a[..N], &b[..N], "同 seed（777）同 logits ⇒ 序列须逐位相同");
    assert_ne!(&a[..N], &c[..N], "不同 seed（888）序列须不同");
    assert_ne!(&a[..N], &d[..N], "不同 seed（999）序列须不同");
    // RNG 槽 ≠ 流的身份：槽 1 与槽 2 各自独立前进，互不影响
    eprintln!(
        "qs14 采样复现：同 seed 24 token 逐位相同；seed777/888/999 差异 {} / {} 项\n  seq777 = {:?}",
        (0..N).filter(|&i| a[i] != c[i]).count(),
        (0..N).filter(|&i| a[i] != d[i]).count(),
        &a[..N]
    );

    // 第二轮：全部退场后重新准入 —— 行/RNG 槽**被上一批用脏了**（四个槽
    // 分别推进过 777/777/888/999）。准入时零化 RNG 槽 + 同 seed 派生 ⇒ 同批
    // 四个同 seed 请求的序列必须彼此逐位相同。这道用例专门钉"复用行不继承
    // 旧 RNG 位置"（不零化的话四个槽停在不同的前进位置，序列会散开）
    ex.retain(&[]);
    for id in 1u64..=4 {
        ex.set_sampling(id, samp(777)).expect("重下发");
        ex.admitted(id, &p);
    }
    let mut got2: Vec<(u64, Vec<u32>)> = Vec::new();
    collect(
        &ex.step(&StepPlan {
            prefill: vec![(1, l), (2, l), (3, l), (4, l)],
            decode: vec![1, 2, 3, 4],
            batch: vec![],
        }),
        &mut got2,
    );
    while (1u64..=4).any(|id| toks_of(&got2, id).len() < N) {
        let out = ex.step(&StepPlan { prefill: vec![], decode: vec![1, 2, 3, 4], batch: vec![] });
        collect(&out, &mut got2);
    }
    let s1 = toks_of(&got2, 1);
    for id in 2u64..=4 {
        assert_eq!(
            &s1[..N],
            &toks_of(&got2, id)[..N],
            "行回收（RNG 槽用脏）后重新准入：同 seed 的请求 {id} 须与请求 1 逐位相同"
        );
    }
    eprintln!("qs14 行回收后重跑：4 个同 seed 请求 24 token 彼此逐位相同（RNG 槽准入零化）");
}

/// qs15 · greedy 零影响：`set_sampling(greedy)` 与"根本不下发采样参数"
/// 的 24 token **逐位相同**——接入设备端采样不得改动 argmax 路径一个 bit。
///
/// 两个请求同批推进（同一 GEMM 调用），逐 token 比对；断言也覆盖"greedy
/// 不下发参数"与"greedy 显式下发"两条路径的结构等价。这是 qs10 的
/// 24-token 纯 decode 序列在接入后的回归钉子。
#[test]
fn qs15_greedy_unchanged() {
    let _g = guard();
    let Some(dir) = mdir() else { return };
    let Some(mut ex) = exec(&dir) else { return };
    ex.set_radix(false);
    let tok = HfTokenizer::open(&dir).unwrap();
    let mut p = tok.encode(
        "The quick brown fox jumps over the lazy dog while the sun sets behind the distant \
         mountains and the river flows quietly through the ancient forest",
    );
    p.truncate(24);
    let l = p.len();
    const N: usize = 24;

    ex.admitted(1, &p);
    ex.admitted(2, &p);
    // 请求 1：完全不下发（历史路径）；请求 2：显式 greedy（temp = 0）
    ex.set_sampling(2, SampleParams::greedy()).expect("greedy 须被接受");
    assert!(ex.sampling_of(2).expect("请求 2 已准入").is_greedy(), "greedy 语义须可回读");
    ex.set_sampling(2, SampleParams { temperature: -1.0, top_k: 3, top_p: 0.5, seed: Some(5) })
        .expect("temp < 0 也按 greedy 处理");

    let out = ex.step(&StepPlan { prefill: vec![(1, l), (2, l)], decode: vec![1, 2], batch: vec![] });
    let mut got: Vec<(u64, Vec<u32>)> = Vec::new();
    collect(&out, &mut got);
    while toks_of(&got, 1).len() < N || toks_of(&got, 2).len() < N {
        let out = ex.step(&StepPlan { prefill: vec![], decode: vec![1, 2], batch: vec![] });
        collect(&out, &mut got);
    }
    let (a, b) = (toks_of(&got, 1), toks_of(&got, 2));
    assert_eq!(&a[..N], &b[..N], "greedy 路径（含显式 set_sampling）须与原 argmax 逐位相同");
    eprintln!("qs15 greedy 零影响：24 token 逐位相同（seq = {:?}）", &a[..N]);
}
