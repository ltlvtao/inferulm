//! 执行器行容量接入准入（e5）：exec 每请求占一行（块表/递推/conv 状态池），
//! 调度器 max_running 默认 256 与 exec 行上限脱钩——超额准入会在 exec 内部
//! assert 崩掉引擎任务（I1 满负荷批界：c>128 直接炸流）。engine 须把
//! `Executor::row_capacity()` 钳进 max_running：超容请求排队等行释放，
//! 不进批、不炸执行器、不饿死。

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
    fn admitted(&mut self, id: ReqId, _prompt: &[u32]) {
        self.rows.insert(id);
        self.max_active.fetch_max(self.rows.len(), Ordering::SeqCst);
    }
    fn row_capacity(&self) -> usize {
        2
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
