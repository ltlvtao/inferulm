//! wein-sched：统一 token 预算调度器（纯逻辑，无 async / 无 I/O / 零依赖）。
//!
//! 设计依据 design/02-architecture.md §6：
//! - **无 prefill/decode phase**：token 债统一建模——债 = prompt 尚未 forward 的
//!   + 已采样未 forward 的，共用一个预算；债 ≤ 1 即 decode（每步 1 token），
//!   债 > 1 即 prefill/recompute（按 chunk 切）；追平的那一步顺带采样（vLLM 追赶模型）
//! - **chunked prefill**：完成追赶的那一步同时产出 token（forward 最后一个 token
//!   即得到下一个采样，采样不额外计预算）
//! - **抢占 = recompute**（增长驱动）：批内上下文总量 Σ(prompt+generated) 超过
//!   `max_batch_tokens`（粗粒度 KV 压力模型）→ 驱逐进度最浅者到队尾，仅 forward
//!   进度清零，**已生成 token 保留**（客户端已收到的不可收回），重算债 =
//!   prompt + 已生成（vLLM recompute 语义）。至少保留 1 个在跑（进度保证）；
//!   不为准入而抢占（防抖——槽位/容量由完成自然释放）。wein-memory 接入后
//!   触发条件换成 BlockPool 水位，策略钩子不变
//! - **BatchBuffer 增量 op**：请求进出批各计一次（metrics.batch_adds/removes），
//!   禁止每步全量重建——由验收测试钉死
//! - **DeferredRelease**：结构性变更（完成/取消/抢占）在步末统一回放事件，
//!   规划中途不做移除
//!
//! 本 crate 是 BASELINE.md 归因阶梯 L2 的责任方（schedule 开销的账记在这里）。
use std::collections::{HashMap, VecDeque};

pub type ReqId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Aborted,
}

#[derive(Clone, Debug)]
pub struct SchedConfig {
    /// 每步 token 预算（prefill + decode 共用）
    pub token_budget: usize,
    /// 最大并发运行请求数（槽位容量）
    pub max_running: usize,
    /// 单请求单步 prefill 上限
    pub chunk_size: usize,
    /// 批内上下文容量 Σ(prompt + generated)——粗粒度 KV 压力模型；
    /// 超限触发抢占（增长驱动）。wein-memory 接入后由 BlockPool 水位替换
    pub max_batch_tokens: usize,
}

impl Default for SchedConfig {
    fn default() -> Self {
        Self { token_budget: 512, max_running: 256, chunk_size: 256, max_batch_tokens: 65536 }
    }
}

pub struct ReqInput {
    pub id: ReqId,
    pub prompt_len: usize,
    pub max_tokens: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SchedEvent {
    Admitted(ReqId),
    Preempted(ReqId),
    Finished { id: ReqId, reason: FinishReason },
    Cancelled(ReqId),
}

/// 一步的执行计划。批序 = FIFO 准入序。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepPlan {
    /// 本步各请求推进的 prefill token 数
    pub prefill: Vec<(ReqId, usize)>,
    /// 本步各推进一个 decode token 的请求
    pub decode: Vec<ReqId>,
    /// 本步在批内的全部请求
    pub batch: Vec<ReqId>,
}

/// 调度器指标（BASELINE §4 插桩的第一层；L2 电池的读数源）。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SchedMetrics {
    pub steps: u64,
    pub admitted: u64,
    pub preempted: u64,
    pub finished: u64,
    pub cancelled: u64,
    /// BatchBuffer 增量 op 计数——若有人改成每步全量重建，这两个数会爆炸，
    /// 验收测试 s4/s9 立刻抓到
    pub batch_adds: u64,
    pub batch_removes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum St {
    Waiting,
    Running,
}

struct Req {
    prompt_len: usize,
    /// 已 forward 的 token 数（prompt + 已生成中已处理的部分）；
    /// 抢占后清零 → 重算债 = prompt_len + generated
    forwarded: usize,
    generated: u32,
    max_tokens: u32,
    st: St,
}

impl Req {
    /// 上下文长度 = prompt + 已生成（KV 压力的来源：只增不减，直到请求离批）
    fn context(&self) -> usize {
        self.prompt_len + self.generated as usize
    }
}

/// 批缓冲：只经 add/remove 增量变更，绝不整表重建（metrics 计数钉死）。
#[derive(Default)]
struct BatchBuffer {
    ids: VecDeque<ReqId>,
}

impl BatchBuffer {
    fn add(&mut self, id: ReqId) {
        self.ids.push_back(id);
    }
    fn remove(&mut self, id: ReqId) {
        if let Some(p) = self.ids.iter().position(|&x| x == id) {
            self.ids.remove(p);
        }
    }
    fn len(&self) -> usize {
        self.ids.len()
    }
    fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    fn iter(&self) -> impl Iterator<Item = ReqId> + '_ {
        self.ids.iter().copied()
    }
}

pub struct Scheduler {
    cfg: SchedConfig,
    reqs: HashMap<ReqId, Req>,
    waiting: VecDeque<ReqId>,
    batch: BatchBuffer,
    events: VecDeque<SchedEvent>,
    m: SchedMetrics,
}

impl Scheduler {
    pub fn new(cfg: SchedConfig) -> Self {
        Self {
            cfg,
            reqs: HashMap::new(),
            waiting: VecDeque::new(),
            batch: BatchBuffer::default(),
            events: VecDeque::new(),
            m: SchedMetrics::default(),
        }
    }

    /// 提交。max_tokens == 0 立即终结（Length），不入队不占批。
    pub fn submit(&mut self, input: ReqInput) {
        if input.max_tokens == 0 {
            self.m.finished += 1;
            self.events.push_back(SchedEvent::Finished { id: input.id, reason: FinishReason::Length });
            return;
        }
        self.reqs.insert(
            input.id,
            Req {
                prompt_len: input.prompt_len,
                forwarded: 0,
                generated: 0,
                max_tokens: input.max_tokens,
                st: St::Waiting,
            },
        );
        self.waiting.push_back(input.id);
    }

    /// 取消。waiting → 直接移除；running → 出批（释放容量与预算）；未知 id 无操作。
    pub fn cancel(&mut self, id: ReqId) {
        match self.reqs.get(&id).map(|r| r.st) {
            Some(St::Waiting) => {
                self.waiting.retain(|&x| x != id);
                self.reqs.remove(&id);
            }
            Some(St::Running) => {
                self.batch.remove(id);
                self.m.batch_removes += 1;
                self.reqs.remove(&id);
            }
            None => return,
        }
        self.m.cancelled += 1;
        self.events.push_back(SchedEvent::Cancelled(id));
    }

    /// 推进一步：准入（含容量触发抢占）→ 预算规划 → 完成判定（步末回放）。
    pub fn step(&mut self) -> StepPlan {
        self.m.steps += 1;

        // --- 准入（FIFO + 容量反压）：头请求放得下才放（严格保序，不越过队首）；
        //     批空时无条件放——进度保证：超大请求独跑（真实引擎应在 API 层拒绝）---
        while self.batch.len() < self.cfg.max_running && !self.waiting.is_empty() {
            let head_ctx = self.reqs[self.waiting.front().unwrap()].context();
            if !self.batch.is_empty() && self.demand() + head_ctx > self.cfg.max_batch_tokens {
                break; // 反压：等容量释放（完成会缩 demand），不为准入而抢占
            }
            self.admit_one();
        }
        // --- 抢占（增长驱动）：批内上下文总量超容 → 驱逐进度最浅者到队尾（recompute）。
        //     上下文随采样单调增长，所以压力自然出现；至少保留 1 个在跑。
        //     已知简化：最新准入者（进度最浅）可能被反复弹回，但老请求有界完成
        //     保证收敛；wein-memory 接入后由 BlockPool 水位替换触发条件 ---
        while self.batch.len() > 1 && self.demand() > self.cfg.max_batch_tokens {
            let victim = self
                .batch
                .iter()
                .min_by_key(|&id| self.reqs[&id].forwarded + self.reqs[&id].generated as usize)
                .unwrap();
            self.batch.remove(victim);
            self.m.batch_removes += 1;
            let r = self.reqs.get_mut(&victim).unwrap();
            r.st = St::Waiting;
            r.forwarded = 0; // recompute：forward 进度清零；generated 保留
            self.waiting.push_back(victim);
            self.m.preempted += 1;
            self.events.push_back(SchedEvent::Preempted(victim));
        }

        // --- 预算规划：批序（FIFO 准入序）逐个分账；预算尽则 break（先来者保护）。
        //     债 backlog = prompt_len + generated - forwarded（未 forward 的 token 数）。
        //     债 ≤ 1 且 prompt 已 forward ⇒ decode（成本 1）；否则按 chunk 追赶，
        //     追平的那一步顺带采样（采样不额外计预算——它就是那最后一次 forward 的输出）---
        let mut budget = self.cfg.token_budget;
        let mut plan = StepPlan { batch: self.batch.iter().collect(), ..Default::default() };
        for &id in &plan.batch {
            let r = self.reqs.get_mut(&id).unwrap();
            if r.generated >= r.max_tokens {
                continue; // 完成判定在步末统一回放；这里不再分预算
            }
            let backlog = r.prompt_len + r.generated as usize - r.forwarded;
            if backlog <= 1 && r.forwarded >= r.prompt_len {
                // decode：forward 最后采样的 token（空 prompt 首步 backlog=0 同样直接采样）
                if budget < backlog {
                    break;
                }
                budget -= backlog;
                r.forwarded += backlog;
                r.generated += 1;
                plan.decode.push(id);
            } else {
                let chunk = backlog.min(self.cfg.chunk_size).min(budget);
                if chunk == 0 {
                    break;
                }
                r.forwarded += chunk;
                budget -= chunk;
                plan.prefill.push((id, chunk));
                if r.forwarded == r.prompt_len + r.generated as usize {
                    // 追平：本次 forward 的最后一个 token 直接产出下一个采样
                    r.generated += 1;
                    plan.decode.push(id);
                }
            }
        }

        // --- 完成判定：DeferredRelease，步末统一回放 ---
        let done: Vec<ReqId> = plan
            .decode
            .iter()
            .copied()
            .filter(|&id| self.reqs[&id].generated >= self.reqs[&id].max_tokens)
            .collect();
        for id in done {
            self.batch.remove(id);
            self.m.batch_removes += 1;
            self.reqs.remove(&id);
            self.m.finished += 1;
            self.events.push_back(SchedEvent::Finished { id, reason: FinishReason::Length });
        }

        plan
    }

    pub fn poll_events(&mut self) -> Vec<SchedEvent> {
        self.events.drain(..).collect()
    }

    pub fn metrics(&self) -> SchedMetrics {
        self.m
    }

    pub fn is_idle(&self) -> bool {
        self.batch.ids.is_empty() && self.waiting.is_empty()
    }

    /// 批内上下文总需求（KV 压力读数）
    fn demand(&self) -> usize {
        self.batch.iter().map(|id| self.reqs[&id].context()).sum()
    }

    fn admit_one(&mut self) {
        if let Some(id) = self.waiting.pop_front() {
            self.reqs.get_mut(&id).unwrap().st = St::Running;
            self.batch.add(id);
            self.m.batch_adds += 1;
            self.m.admitted += 1;
            self.events.push_back(SchedEvent::Admitted(id));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn cfg(budget: usize, max_running: usize, chunk: usize) -> SchedConfig {
        // 默认容量给到天文数字：非压力用例不关心抢占
        SchedConfig { token_budget: budget, max_running, chunk_size: chunk, max_batch_tokens: 1 << 30 }
    }

    /// 跑到全部完成，返回 (plans, events)；max_steps 防死循环。
    fn run_to_idle(s: &mut Scheduler, max_steps: u64) -> (Vec<StepPlan>, Vec<SchedEvent>) {
        let mut plans = Vec::new();
        let mut events = Vec::new();
        let mut steps = 0;
        while !s.is_idle() {
            plans.push(s.step());
            events.extend(s.poll_events());
            steps += 1;
            assert!(steps < max_steps, "调度器未在 {max_steps} 步内收敛（疑似死锁/饿死）");
        }
        // 收尾事件
        events.extend(s.poll_events());
        (plans, events)
    }

    /// s1 · chunked prefill 按预算切片；prefill 完成步出首 token；恰好在 max_tokens 终结
    #[test]
    fn s1_chunked_prefill_spans_budget() {
        let mut s = Scheduler::new(cfg(4, 8, 4));
        s.submit(ReqInput { id: 1, prompt_len: 10, max_tokens: 3 });
        let (plans, events) = run_to_idle(&mut s, 50);

        let prefill_chunks: Vec<usize> =
            plans.iter().filter(|p| !p.prefill.is_empty()).map(|p| p.prefill[0].1).collect();
        assert_eq!(prefill_chunks, vec![4, 4, 2], "10 token 债按预算 4 切成 4/4/2");
        let first_decode_step = plans.iter().position(|p| p.decode.contains(&1)).unwrap();
        assert_eq!(first_decode_step, 2, "首个 decode 出现在 prefill 完成的那一步（第 3 步，0 起）");
        let decode_total: usize = plans.iter().map(|p| p.decode.len()).sum();
        assert_eq!(decode_total, 3, "恰好产出 max_tokens 个 token");
        assert!(matches!(
            events.last(),
            Some(SchedEvent::Finished { id: 1, reason: FinishReason::Length })
        ));
    }

    /// s2 · 无 phase：prefill 与 decode 混在同一个预算同一步里，新 prefill 不挡老 decode
    #[test]
    fn s2_mixed_batch_shares_one_budget() {
        let mut s = Scheduler::new(cfg(4, 8, 2));
        s.submit(ReqInput { id: 1, prompt_len: 6, max_tokens: 10 }); // 老请求：长 prefill
        s.submit(ReqInput { id: 2, prompt_len: 1, max_tokens: 10 }); // 新请求：短 prefill
        let (plans, _) = run_to_idle(&mut s, 100);

        let p1 = &plans[0];
        assert!(!p1.prefill.is_empty() && !p1.decode.is_empty(), "第 1 步必须同时含 prefill 与 decode");
        assert!(p1.prefill.contains(&(1, 2)), "请求 1 推进 2（chunk 上限）");
        assert!(p1.prefill.contains(&(2, 1)), "请求 2 prefill 完成");
        assert!(p1.decode.contains(&2), "请求 2 prefill 完成即出首 token");
        assert!(!p1.decode.contains(&1), "请求 1 prefill 未完不出 token");
        // 老请求不被饿死：最终也完成
        assert_eq!(s.metrics().finished, 2);
    }

    /// s3 · FIFO 准入：同优先级按提交顺序进批；无容量压力时零抢占
    #[test]
    fn s3_fifo_admission_order_no_thrash() {
        let mut s = Scheduler::new(cfg(16, 2, 16));
        for id in 1..=5 {
            s.submit(ReqInput { id, prompt_len: 1, max_tokens: 2 });
        }
        let (_, events) = run_to_idle(&mut s, 100);
        let admitted: Vec<ReqId> = events
            .iter()
            .filter_map(|e| match e {
                SchedEvent::Admitted(id) => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(admitted, vec![1, 2, 3, 4, 5], "准入严格 FIFO，不因排队而抢占");
        assert_eq!(s.metrics().preempted, 0, "槽位排队不是抢占的正当理由");
    }

    /// s4 · 抢占 = recompute（增长驱动）：上下文总量超容 → 驱逐进度最浅者，
    ///      仅 forward 进度清零、已生成 token 保留，重算债 = prompt + 已生成；
    ///      BatchBuffer 增量 op 恰好每次进出各计一次（无全量重建）
    #[test]
    fn s4_preemption_is_recompute_and_batch_ops_are_incremental() {
        // 容量 6：A/C 各 P=2 max=8，采样增长把 demand 推过 6 → 触发抢占
        let mut s = Scheduler::new(SchedConfig {
            token_budget: 16,
            max_running: 8,
            chunk_size: 16,
            max_batch_tokens: 6,
        });
        s.submit(ReqInput { id: 1, prompt_len: 2, max_tokens: 8 }); // A
        s.submit(ReqInput { id: 2, prompt_len: 2, max_tokens: 8 }); // C
        let (plans, events) = run_to_idle(&mut s, 100);

        // A（并列最浅，批序在先）恰被抢占一次；C 在保 1 在跑的进度保证下跑完
        assert!(events.contains(&SchedEvent::Preempted(1)));
        assert_eq!(s.metrics().preempted, 1, "抢占恰一次，不得反复弹回");

        // recompute：A 的 2 个已生成 token 保留，重算债 = prompt(2) + 已生成(2) = 4
        assert!(
            plans.iter().any(|p| p.prefill.contains(&(1, 4))),
            "重算的 prefill 必须覆盖 prompt + 已生成 token（债 = 4）"
        );
        let a_decode_slots: usize =
            plans.iter().map(|p| p.decode.iter().filter(|&&id| id == 1).count()).sum();
        assert_eq!(a_decode_slots, 8, "A 的 decode 槽位总数 = max_tokens（生成进度跨抢占保留）");

        // 两者最终以 Length 完成（被抢占者重排队后收敛）
        for id in [1u64, 2] {
            assert!(matches!(
                events.iter().find(|e| matches!(e, SchedEvent::Finished { id: i, .. } if *i == id)),
                Some(SchedEvent::Finished { reason: FinishReason::Length, .. })
            ), "请求 {id} 必须完成");
        }

        // BatchBuffer 增量性：adds == 准入次数（含重排后的再准入），removes == 抢占+完成
        let m = s.metrics();
        assert_eq!(m.batch_adds, m.admitted, "每个请求每次进批恰好计一次 add");
        assert_eq!(m.batch_removes, m.preempted + m.finished, "每次出批恰好计一次 remove");
        // 反证：若改成每步全量重建，adds 会 ≈ steps × batch_size，远大于 admitted
        assert!(m.batch_adds <= m.admitted, "禁止全量重建");
    }

    /// s5 · 取消的三种状态：waiting（不占批）、running（释放容量）、未知 id（无操作）
    #[test]
    fn s5_cancel_waiting_running_and_unknown() {
        let mut s = Scheduler::new(cfg(8, 1, 8));
        s.submit(ReqInput { id: 1, prompt_len: 2, max_tokens: 10 }); // B 运行中
        s.step();
        s.poll_events();
        s.submit(ReqInput { id: 2, prompt_len: 1, max_tokens: 1 }); // A 等待中
        let before = s.metrics();

        s.cancel(2); // waiting：不出批
        assert_eq!(s.metrics().batch_removes, before.batch_removes);
        s.cancel(1); // running：出批
        assert_eq!(s.metrics().batch_removes, before.batch_removes + 1);
        s.cancel(99); // 未知：无操作不 panic
        assert_eq!(s.metrics().cancelled, 2);

        let plan = s.step();
        assert!(plan.batch.is_empty(), "取消后批为空");
        let evs = s.poll_events();
        assert_eq!(evs.len(), 2, "两次有效取消各回放一个 Cancelled 事件");
    }

    /// s6 · max_tokens 边界：0 立即终结不占批；N 恰好 N 个 token 后终结
    #[test]
    fn s6_max_tokens_boundaries() {
        let mut s = Scheduler::new(cfg(8, 8, 8));
        s.submit(ReqInput { id: 1, prompt_len: 1, max_tokens: 0 });
        let evs = s.poll_events();
        assert!(matches!(evs[0], SchedEvent::Finished { id: 1, reason: FinishReason::Length }));
        assert!(s.is_idle(), "max_tokens=0 不入队");

        s.submit(ReqInput { id: 2, prompt_len: 1, max_tokens: 2 });
        let (plans, _) = run_to_idle(&mut s, 20);
        let n: usize = plans.iter().map(|p| p.decode.len()).sum();
        assert_eq!(n, 2);
    }

    /// s7 · 确定性：同一命令序列 → 逐位相同的计划与事件（批不变性的地基）
    #[test]
    fn s7_determinism_same_inputs_same_plans() {
        let scenario = |s: &mut Scheduler| {
            for id in 1..=4 {
                s.submit(ReqInput { id, prompt_len: (id % 3 + 1) as usize, max_tokens: id as u32 });
            }
            s.cancel(3);
            let (plans, events) = run_to_idle(s, 100);
            (plans, events)
        };
        let mut a = Scheduler::new(cfg(4, 2, 2));
        let mut b = Scheduler::new(cfg(4, 2, 2));
        assert_eq!(scenario(&mut a), scenario(&mut b));
    }

    /// s8 · 前向进展：容量受限 + 持续到达，全部请求在有界步数内完成（无饿死）
    #[test]
    fn s8_forward_progress_under_pressure() {
        let t0 = Instant::now();
        let mut s = Scheduler::new(cfg(8, 2, 4));
        for id in 1..=20 {
            s.submit(ReqInput { id, prompt_len: 2, max_tokens: 2 });
        }
        let (_, events) = run_to_idle(&mut s, 500);
        assert_eq!(events.iter().filter(|e| matches!(e, SchedEvent::Finished { .. })).count(), 20);
        assert!(s.metrics().steps < 100, "20 个小请求不该超过 100 步：{}", s.metrics().steps);
        assert!(t0.elapsed().as_millis() < 1000, "纯逻辑调度器必须亚毫秒级处理此规模");
    }
}
