//! DRAM 分层（W-Agent-5）端到端：驱逐下沉 → 同前缀重进 → 接力换入 →
//! 位级恢复。独立测试进程（ctx 配置是进程级 OnceLock——env 旋钮只在
//! 本文件首测初始化时生效，两测试共用同一套 env）
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use wein_engine::Executor;
use wein_exec::Qwen35Executor;
use wein_model::{model_dir, HfTokenizer};
use wein_sched::StepPlan;

fn guard() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

/// 统一 env（两测试一致——ctx OnceLock 取首测快照）：
/// 池 16 页 = 256 token（撑穿用多会话堆压）、槽 4（竞争快）、DRAM 开
fn env() {
    std::env::set_var("WEIN_DRAM", "1");
    std::env::set_var("WEIN_KV_PAGES", "16");
    std::env::set_var("WEIN_SNAP_SLOTS", "4");
}

fn mdir() -> Option<std::path::PathBuf> {
    let d = model_dir();
    if d.is_none() {
        eprintln!("SKIP（模型目录不在场）");
    }
    d
}

const TEXT: &str = "The quick brown fox jumps over the lazy dog while the sun sets behind the distant mountains, and the river flows quietly through the ancient forest where birds sing their evening songs before night falls over the sleeping village below the quiet stars";

/// A 会话 33-token prompt（2 整块 + 尾）：32+1 分段 → 6 token 基线
fn session_a(ex: &mut Qwen35Executor, tok: &HfTokenizer) -> Vec<u32> {
    let mut p = tok.encode(TEXT);
    p.truncate(33);
    ex.admitted(1, &p);
    let _ = ex.step(&StepPlan { prefill: vec![(1, 32)], decode: vec![], batch: vec![] });
    let t2 = ex.step(&StepPlan { prefill: vec![(1, 1)], decode: vec![1], batch: vec![] });
    let mut base = vec![t2[0].1];
    for _ in 0..5 {
        let d = ex.step(&StepPlan { prefill: vec![], decode: vec![1], batch: vec![] });
        base.push(d[0].1);
    }
    ex.retain(&[]);
    base
}

/// 共享 A 前缀的挤压会话：A 段在其在飞锁（match_prefix lock_ref）保护下
/// 存活——force_release 只解挂表 hold，动不了它。240 token 自建 13 块
fn session_sq(ex: &mut Qwen35Executor, tok: &HfTokenizer, id: u64, salt: u32) {
    let mut p = tok.encode(TEXT);
    p.truncate(33);
    for i in 0..207u32 {
        p.push(salt * 977 + i % 9000);
    }
    assert_eq!(p.len(), 240);
    ex.admitted(id, &p);
    let _ = ex.step(&StepPlan { prefill: vec![(id, 240)], decode: vec![], batch: vec![] });
    let _ = ex.step(&StepPlan { prefill: vec![], decode: vec![id], batch: vec![] });
    ex.retain(&[]);
}

/// 冷前缀挤压会话：首 token 破坏前缀匹配（树零命中）⇒ 15 新页硬压穿
/// 16 页池，触发 force_release：最老槽（A 的）下沉 DRAM（KV+状态）→
/// 解 hold → A 链连根逐出
fn session_cold(ex: &mut Qwen35Executor, tok: &HfTokenizer, id: u64, salt: u32) {
    let mut p = tok.encode(TEXT);
    p.truncate(33);
    p[0] = p[0].wrapping_add(salt * 7);
    for i in 0..207u32 {
        p.push(salt * 977 + i % 9000);
    }
    assert_eq!(p.len(), 240);
    ex.admitted(id, &p);
    let _ = ex.step(&StepPlan { prefill: vec![(id, 240)], decode: vec![], batch: vec![] });
    let _ = ex.step(&StepPlan { prefill: vec![], decode: vec![id], batch: vec![] });
    ex.retain(&[]);
}

/// 换入位级合同：A 挂树 → 共享前缀会话穿过（A 在飞锁保护）→ 冷前缀
/// 会话压穿池把 A 的块+状态下沉 DRAM 并逐出 → 同前缀重进 → DRAM 接力
/// 命中 2 整块 → 生成与基线逐 token 相同
#[test]
fn qsd1_dram_tier_roundtrip_bitwise() {
    let _g = guard();
    env();
    let Some(dir) = mdir() else { return };
    let Some(mut ex) = (|| match Qwen35Executor::open(&dir) {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("SKIP（open 失败：{e}）");
            None
        }
    })() else { return };
    let tok = HfTokenizer::open(&dir).unwrap();
    let mut p33 = tok.encode(TEXT);
    p33.truncate(33);
    let base = session_a(&mut ex, &tok);
    session_sq(&mut ex, &tok, 2, 1); // 池到 15/16：A 段在飞锁保护下存活
    session_cold(&mut ex, &tok, 3, 2); // 冷压穿：A 槽 force 下沉 + A 链逐出
    session_cold(&mut ex, &tok, 4, 3); // 再压一轮：上条冷链同样下沉逐出
    let (sb, ss, _, _) = ex.dram_stats();
    eprintln!("qsd1 下沉：blocks={sb} states={ss}");
    assert!(ss >= 1, "强制逐槽须把状态下沉（实得 {ss}）");
    assert!(sb >= 2, "块 KV 须下沉（实得 {sb}）");
    // 同前缀重进：树上的 A 段已逐空 ⇒ 命中只能来自 DRAM 接力
    let hit = ex.admitted(5, &p33);
    assert_eq!(hit, 32, "DRAM 接力须命中 2 整块（实得 {hit}）");
    let (_, _, fb, fs) = ex.dram_stats();
    assert!(fs >= 1, "状态须换入（实得 {fs}）");
    assert!(fb >= 2, "块 KV 须换入 ≥2（实得 {fb}）");
    let t = ex.step(&StepPlan { prefill: vec![(5, 1)], decode: vec![5], batch: vec![] });
    assert_eq!(t[0].1, base[0], "DRAM 恢复后末 token 重放位级一致");
    for k in 1..base.len() {
        let d = ex.step(&StepPlan { prefill: vec![], decode: vec![5], batch: vec![] });
        assert_eq!(d[0].1, base[k], "DRAM 恢复后续生成第 {k} token 位级一致");
    }
}

/// 槽竞争不死锁：4 槽全被挂表块 hold 后，新请求仍能拿槽前向。
/// 步尾必须落在块边界（p=31）才会采快照——单步 33-token 的步尾 p=32
/// 在栅格外，4 会话全都不挂槽（前提不成立）
#[test]
fn qsd2_force_release_no_deadlock() {
    let _g = guard();
    env();
    let Some(dir) = mdir() else { return };
    let Some(mut ex) = (|| match Qwen35Executor::open(&dir) {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("SKIP（open 失败：{e}）");
            None
        }
    })() else { return };
    let tok = HfTokenizer::open(&dir).unwrap();
    let mut p = tok.encode(TEXT);
    p.truncate(33);
    // 4 个不同前缀会话各占一槽（32+1 分段：首步尾 p=31 采快照 → retain 挂表）
    for id in 1u64..=4 {
        let mut q = p.clone();
        q[0] = q[0].wrapping_add(id as u32);
        ex.admitted(id, &q);
        let _ = ex.step(&StepPlan { prefill: vec![(id, 32)], decode: vec![], batch: vec![] });
        let _ = ex.step(&StepPlan { prefill: vec![(id, 1)], decode: vec![id], batch: vec![] });
        ex.retain(&[]);
    }
    // 第五会话：槽全占 ⇒ acquire_slot 强制下沉最老槽
    let mut w = p.clone();
    w[0] = w[0].wrapping_add(99);
    ex.admitted(5, &w);
    let _ = ex.step(&StepPlan { prefill: vec![(5, 32)], decode: vec![], batch: vec![] });
    let t2 = ex.step(&StepPlan { prefill: vec![(5, 1)], decode: vec![5], batch: vec![] });
    assert_eq!(t2.len(), 1, "槽竞争下新请求仍能前向（不死锁）");
}

/// 树常驻段接力（W-Agent-5 修复的缺口）：共享前缀块在树上（他链
/// hold 保护、从未下沉），A 的状态+尾块只在 DRAM——接力须把树内段
/// 当零搬运覆盖、只从 DRAM 补尾块。旧条件（恢复点前全链在 DRAM）
/// 在此场景永不触发（热系统 prompt 常驻即失败）
#[test]
fn qsd3_relay_with_tree_resident_prefix() {
    let _g = guard();
    env();
    let Some(dir) = mdir() else { return };
    let Some(mut ex) = (|| match Qwen35Executor::open(&dir) {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("SKIP（open 失败：{e}）");
            None
        }
    })() else { return };
    let tok = HfTokenizer::open(&dir).unwrap();
    let mut p33 = tok.encode(TEXT);
    p33.truncate(33);
    // A 链 = [共享块 b1][A 尾块 b2]，挂槽于 b2（snap_nb=2）
    let base = session_a(&mut ex, &tok);
    // S2：共享 b1（前 16 token）、尾块不同 → b1 变热且被 S2 链 hold 保护
    let mut q = p33.clone();
    q[16] = q[16].wrapping_add(7);
    ex.admitted(2, &q);
    let _ = ex.step(&StepPlan { prefill: vec![(2, 32)], decode: vec![], batch: vec![] });
    let _ = ex.step(&StepPlan { prefill: vec![(2, 1)], decode: vec![2], batch: vec![] });
    ex.retain(&[]);
    // 冷会话 14 块压穿（自由 13 页）：force 下沉 A 槽（状态 n_tok=32 +
    // b2 KV，靠 snap_nb 才能对上 token 数）→ 逐出 A 尾块；b1 因 S2
    // 链 hold 存活
    let mut c = tok.encode(TEXT);
    c.truncate(33);
    c[0] = c[0].wrapping_add(500);
    for i in 0..191u32 {
        c.push(9000 + i % 500);
    }
    assert_eq!(c.len(), 224);
    ex.admitted(3, &c);
    let _ = ex.step(&StepPlan { prefill: vec![(3, 224)], decode: vec![], batch: vec![] });
    ex.retain(&[]);
    let (sb, ss, _, _) = ex.dram_stats();
    eprintln!("qsd3 下沉：blocks={sb} states={ss}");
    assert!(ss >= 1, "A 槽让位须下沉状态（实得 {ss}）");
    // A 重进：树只认 b1（无挂槽 → keep=0），接力 = 树内 b1 + DRAM 补
    // b2 与状态——旧代码此场景 hit=0
    let hit = ex.admitted(4, &p33);
    assert_eq!(hit, 32, "树常驻段接力须命中 2 整块（旧代码此场景为 0）");
    let (_, _, fb, fs) = ex.dram_stats();
    assert!(fs >= 1, "状态须换入（实得 {fs}）");
    assert!(fb >= 1, "尾块 KV 须换入（实得 {fb}）");
    let t = ex.step(&StepPlan { prefill: vec![(4, 1)], decode: vec![4], batch: vec![] });
    assert_eq!(t[0].1, base[0], "接力恢复后末 token 重放位级一致");
    for k in 1..base.len() {
        let d = ex.step(&StepPlan { prefill: vec![], decode: vec![4], batch: vec![] });
        assert_eq!(d[0].1, base[k], "接力恢复后续生成第 {k} token 位级一致");
    }
}
