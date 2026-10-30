//! 执行器行容量接入准入（e5）：exec 每请求占一行（块表/递推/conv 状态池），
//! 调度器 max_running 默认 256 与 exec 行上限脱钩——超额准入会在 exec 内部
//! assert 崩掉引擎任务（I1 满负荷批界：c>128 直接炸流）。engine 须把
//! `Executor::row_capacity()` 钳进 max_running：超容请求排队等行释放，
//! 不进批、不炸执行器、不饿死。
//!
//! e6：前缀命中（exec.admitted 返回已恢复 token 数）必须在**规划之前**
//! 进账。晚一步的账：本步预算按未命中的债切 chunk，exec 按真实游标少喂
//! → 调度器认为 prefill 已追平并逐 token 记账，exec 却无 token 可喂、采样
//! 全丢——请求以 0 个 token 静默终结（真实流量里表现为"命中即空回"）。

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use wein_engine::{CoreConfig, EngineCore, EngineEvent, EngineStats, Executor, FinishReason};
use wein_sched::{ReqId, StepPlan};

/// 行占用探针：admitted 建行、retain 按活跃集裁剪——max_active 是执行器
/// 视角的峰值行数（镜像真实 exec：行集 = 已准入 ∖ 离场，与 keep 集对账）
struct RowProbe {
    rows: std::collections::HashSet<ReqId>,
    max_active: Arc<AtomicUsize>,
}

impl Executor for RowProbe {
    fn step(&mut self, plan: &StepPlan) -> Vec<(ReqId, u32)> {
        plan.decode.iter().map(|&id| (id, 7u32)).collect()
    }
    fn retain(&mut self, active: &[ReqId]) {
        self.rows.retain(|k| active.contains(k));
        self.max_active.fetch_max(self.rows.len(), Ordering::SeqCst);
    }
    fn admitted(&mut self, id: ReqId, _prompt: &[u32]) -> usize {
        self.rows.insert(id);
        self.max_active.fetch_max(self.rows.len(), Ordering::SeqCst);
        0
    }
    fn row_capacity(&self) -> usize {
        2
    }
}

/// 命中探针：镜像真实 exec 的游标语义——prefill 只喂到 seq 尾，采样才长
/// seq；decode 步没有可喂 token 就不出 token（真实 exec 的 `forwarded <
/// seq.len()` 分支）。前缀命中在 admitted 时把游标推到 hit。
struct HitProbe {
    fwd: std::collections::HashMap<ReqId, usize>,
    seq: std::collections::HashMap<ReqId, usize>,
    hit: usize,
    fed: Arc<AtomicUsize>,
}

impl Executor for HitProbe {
    fn step(&mut self, plan: &StepPlan) -> Vec<(ReqId, u32)> {
        for &(id, n) in &plan.prefill {
            let Some(&seq) = self.seq.get(&id) else { continue };
            let fwd = self.fwd[&id];
            let take = n.min(seq.saturating_sub(fwd));
            self.fwd.insert(id, fwd + take);
            self.fed.fetch_add(take, Ordering::SeqCst);
            if take > 0 && plan.decode.contains(&id) {
                *self.seq.get_mut(&id).unwrap() += 1; // 追平步采样
            }
        }
        let mut out = Vec::new();
        for &id in &plan.decode {
            let is_pf = plan.prefill.iter().any(|&(p, _)| p == id);
            if is_pf {
                out.push((id, 7u32));
                continue;
            }
            let (fwd, seq) = (self.fwd[&id], self.seq[&id]);
            if fwd < seq {
                self.fwd.insert(id, fwd + 1);
            } else {
                return out; // 无可喂 token：不出 token（真实 exec 的静默空转）
            }
            *self.seq.get_mut(&id).unwrap() += 1;
            out.push((id, 7u32));
        }
        out
    }
    fn retain(&mut self, active: &[ReqId]) {
        self.fwd.retain(|k, _| active.contains(k));
        self.seq.retain(|k, _| active.contains(k));
    }
    fn admitted(&mut self, id: ReqId, prompt: &[u32]) -> usize {
        // vLLM 末 token 重算规则：不能把 prompt 末尾也"恢复"掉（否则无采样位）
        let hit = self.hit.min(prompt.len().saturating_sub(1));
        self.fwd.insert(id, hit);
        self.seq.insert(id, prompt.len());
        hit
    }
}

async fn next(stream: &mut wein_engine::EventStream) -> EngineEvent {
    tokio::time::timeout(Duration::from_secs(5), stream.recv())
        .await
        .expect("5s 内必须有事件（不能死等）")
        .expect("事件流必须仍然打开")
}

/// e5 · 5 个并发请求对 2 行执行器：峰值并发行 ≤ 2，全部正常完成（Length），
/// 完成者释放行后排队者照常服务
#[tokio::test]
async fn e5_exec_row_capacity_clamps_admission() {
    let max_active = Arc::new(AtomicUsize::new(0));
    let (_engine, _stats): (wein_engine::EngineHandle, Arc<EngineStats>) =
        EngineCore::spawn_with(
            CoreConfig::scripted(vec![7], Duration::ZERO),
            Box::new(RowProbe { rows: Default::default(), max_active: max_active.clone() }),
        );

    let mut streams = Vec::new();
    for _ in 0..5 {
        let (_id, s) = _engine.submit(vec![1], 2, Default::default());
        streams.push(s);
    }
    let mut finished = 0;
    for mut s in streams {
        assert!(matches!(next(&mut s).await, EngineEvent::Accepted));
        loop {
            match next(&mut s).await {
                EngineEvent::Token(t) => assert_eq!(t, 7),
                EngineEvent::Finished(FinishReason::Length) => {
                    finished += 1;
                    break;
                }
                ev => panic!("意外事件 {ev:?}"),
            }
        }
        assert!(s.recv().await.is_none(), "Finished 之后必须关流");
    }
    assert_eq!(finished, 5, "超容请求排队等行，不得饿死");
    let peak = max_active.load(Ordering::SeqCst);
    assert!(peak <= 2, "执行器峰值并发行 {peak} 超 row_capacity 2——准入未钳制");
}

/// e6 · 前缀命中在规划之前进账：命中请求不得少喂/重喂——每请求恰好吃
/// prompt_len - hit 个 prefill token，且正常出满 max_tokens 个 token
#[tokio::test]
async fn e6_prefix_hit_applies_before_planning() {
    const HIT: usize = 16;
    let fed = Arc::new(AtomicUsize::new(0));
    let mut cfg = CoreConfig::scripted(vec![7], Duration::ZERO);
    cfg.sched.chunk_size = 8; // 多步 prefill：暴露"按旧债切 chunk"的漂移
    let (engine, _stats) = EngineCore::spawn_with(
        cfg,
        Box::new(HitProbe {
            fwd: Default::default(),
            seq: Default::default(),
            hit: HIT,
            fed: fed.clone(),
        }),
    );
    let prompt_len = 40usize;
    let max_tokens = 5u32;
    let mut streams = Vec::new();
    for _ in 0..3 {
        let (_id, s) = engine.submit(vec![1; prompt_len], max_tokens, Default::default());
        streams.push(s);
    }
    let mut tokens = 0;
    for mut s in streams {
        assert!(matches!(next(&mut s).await, EngineEvent::Accepted));
        loop {
            match next(&mut s).await {
                EngineEvent::Token(_) => tokens += 1,
                EngineEvent::Finished(FinishReason::Length) => break,
                ev => panic!("意外事件 {ev:?}"),
            }
        }
    }
    assert_eq!(tokens, 3 * max_tokens as usize, "命中后请求不得空转/少出 token");
    assert_eq!(
        fed.load(Ordering::SeqCst),
        3 * (prompt_len - HIT),
        "prefill 只喂命中之后的段——多喂/少喂都是账漂移"
    );
}
