//! wein-sched：统一 token 预算调度器（纯逻辑，无 async / 无 I/O / 零依赖）。
//!
//! 设计依据 design/02-architecture.md §6：
//! - **无 prefill/decode phase**：token 债统一建模——债 = prompt 尚未 forward 的
//!   + 已采样未 forward 的，共用一个预算；债 ≤ 1 即 decode（每步 1 token），
//!   债 > 1 即 prefill/recompute（按 chunk 切）；追平的那一步顺带采样（vLLM 追赶模型）
//! - **chunked prefill**：完成追赶的那一步同时产出 token（forward 最后一个 token
//!   即得到下一个采样，采样不额外计预算）
//! - **混合步封顶**（`mix_cap`）：批内有稳定 decode 时单步总 token 封顶——
//!   重步拆轻，decode 行的 ITL 不被同车的大 prefill chunk 打穿（p99 账）；
//!   纯 prefill 波不受限；decode 占满时下限保 [`PREFILL_MIN`]（进度保证）
//! - **容量模型可插拔**（`CapacityModel`）：准入反压与抢占触发全部经此钩子。
//!   默认 `TokenCapacity`（批内 Σ(prompt+generated) ≤ limit 的聚合模型）；
//!   wein-memory 的 BlockPool 水位模型经 wein-engine 适配器接入，调度策略不变
//! - **抢占 = recompute**（增长驱动）：容量超压 → 驱逐进度最浅者到队尾，仅
//!   forward 进度清零，**已生成 token 保留**（客户端已收到的不可收回），重算债 =
//!   prompt + 已生成（vLLM recompute 语义）。至少保留 1 个在跑（进度保证）；
//!   不为准入而抢占（防抖——容量由完成自然释放）
//! - **BatchBuffer 增量 op**：请求进出批各计一次（metrics.batch_adds/removes），
//!   禁止每步全量重建——由验收测试钉死
//! - **DeferredRelease**：结构性变更（完成/取消/抢占）在事件队列统一回放；
//!   完成者的容量在**下一步首**释放——引擎届时已把最后一个 token 记入内存层，
//!   完整序列方可入 radix 缓存
//!
//! 本 crate 是 BASELINE.md 归因阶梯 L2 的责任方（schedule 开销的账记在这里）。
use std::any::Any;
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
    /// 聚合容量上限 Σ(prompt + generated)——默认容量模型 TokenCapacity 的 limit；
    /// 接入 BlockPool 水位模型后此字段仅作默认回退
    pub max_batch_tokens: usize,
    /// 混合步封顶：批内存在稳定 decode 请求时，单步总 token（decode + prefill）
    /// 不超过此值。满负荷 ITL p99 账：混合步时延 ∝ T，预算 512 会把波次切换/
    /// 爬坡期的 ~400 prefill token 与 ~100 decode 行塞进同一步（T≈512 计算
    /// 受限 ~65ms），搭乘的 decode 行 ITL 被打穿——封顶后重步拆轻。批内无
    /// 稳定 decode（纯 prefill 波）不受限（TTFT/吞吐优先）；decode 行数占满
    /// cap 时至少留 [`PREFILL_MIN`] 给 prefill（进度保证）
    pub mix_cap: usize,
    /// prefill chunk 尾对齐粒度（0/1 = 不对齐）。前缀缓存的递推状态检查点
    /// 只能落在"某次 forward 的最后一 token"上，而共享/恢复点必须是块边界
    /// ——非末块 chunk 尾圆到该粒度，chunk 尾即成为检查点（密度 = chunk，
    /// 成本 = 每 chunk 一次状态位拷贝）。执行器经 `Executor::prefix_align`
    /// 上报（无前缀缓存的执行器恒 1，账与旧版逐位一致）
    pub snap_align: usize,
}

impl Default for SchedConfig {
    fn default() -> Self {
        Self {
            token_budget: 512,
            max_running: 256,
            chunk_size: 256,
            max_batch_tokens: 65536,
            // T≈192 的混合步 ≈ 纯 decode 步的 1.4×（qwen_mixed_probe 曲线），
            // p99 与 TTFT 的平衡点——battery 仲裁
            mix_cap: 192,
            snap_align: 1,
        }
    }
}

/// 混合步封顶的下限保护：decode 行占满 cap 时 prefill 每步至少推进的 token 数
///（~1ms 步时延的代价换等待 prefill 的有界推进）
pub const PREFILL_MIN: usize = 8;

pub struct ReqInput {
    pub id: ReqId,
    /// prompt token 序列（容量模型的前缀缓存匹配需要真实 token，不只是长度）
    pub prompt_token_ids: Vec<u32>,
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
    /// 验收测试 s4 立刻抓到
    pub batch_adds: u64,
    pub batch_removes: u64,
}

/// 请求离批原因（容量模型按此分流回收语义）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Removal {
    /// 自然完成：完整序列入 radix 缓存，块移交缓存
    Finished,
    /// 抢占重排：块释放（deferred），序列保留（重准入时前缀命中 → 廉价重算）
    Preempted,
    /// 取消：块释放（deferred），序列丢弃
    Cancelled,
}

/// 容量模型：调度策略的容量侧决策全部经此（02-architecture §5/§6 的策略钩子）。
/// 调用时序（调度器与引擎共同保证）：
/// - `can_admit` → `on_admitted`：准入检查通过后登记（实现可做前缀缓存匹配与块分配）
/// - `on_sample`：规划中每确定一个采样（token id 未知——纯容量记账）
/// - `on_token`：引擎拿到执行器产出后调用（内存布局推进：跨块分配、序列追加）
/// - `on_removed`：离批回收。Preempted/Cancelled 在变更点同步调用；
///   Finished 在**下一步首**回放——届时最后 token 已 on_token，
///   实现才能把完整序列入 radix 缓存
/// - `over_pressure`：抢占触发判据（增长驱动）
/// - `step_end`：步末回收点（DeferredRelease 的块回收）
pub trait CapacityModel: Send {
    /// 新请求（或重排请求）能否放下。批空时调度器强制准入（进度保证），不调用此方法
    fn can_admit(&mut self, ctx_len: usize) -> bool;
    fn on_admitted(&mut self, req: ReqId, prompt: &[u32], generated: u32);
    fn on_sample(&mut self, req: ReqId);
    fn on_token(&mut self, req: ReqId, token_id: u32);
    fn on_removed(&mut self, req: ReqId, how: Removal);
    fn over_pressure(&mut self) -> bool;
    fn step_end(&mut self) {}
    /// 引擎下行转换钩子：识别具体容量实现（强制抢占等扩展信号）
    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        None
    }
}

/// 聚合 token 容量模型（默认）：批内 Σ(prompt + generated) ≤ limit。
/// 事件记账式——与直接遍历批内请求求和等价（s1-s8 验收钉死该等价性）。
pub struct TokenCapacity {
    limit: usize,
    demand: usize,
    ctx: HashMap<ReqId, usize>,
}

impl TokenCapacity {
    pub fn new(limit: usize) -> Self {
        Self { limit, demand: 0, ctx: HashMap::new() }
    }
}

impl CapacityModel for TokenCapacity {
    fn can_admit(&mut self, ctx_len: usize) -> bool {
        self.demand + ctx_len <= self.limit
    }

    fn on_admitted(&mut self, req: ReqId, prompt: &[u32], generated: u32) {
        let c = prompt.len() + generated as usize;
        self.demand += c;
        self.ctx.insert(req, c);
    }

    fn on_sample(&mut self, req: ReqId) {
        self.demand += 1;
        if let Some(c) = self.ctx.get_mut(&req) {
            *c += 1;
        }
    }

    fn on_token(&mut self, _req: ReqId, _token_id: u32) {}

    fn on_removed(&mut self, req: ReqId, _how: Removal) {
        if let Some(c) = self.ctx.remove(&req) {
            self.demand -= c;
        }
    }

    fn over_pressure(&mut self) -> bool {
        self.demand > self.limit
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum St {
    Waiting,
    Running,
}

struct Req {
    prompt: Vec<u32>,
    /// 已 forward 的 token 数（prompt + 已生成中已处理的部分）；
    /// 抢占后清零 → 重算债 = prompt_len + generated
    forwarded: usize,
    generated: u32,
    max_tokens: u32,
    st: St,
}

impl Req {
    fn prompt_len(&self) -> usize {
        self.prompt.len()
    }
    /// 上下文长度 = prompt + 已生成（KV 压力的来源：只增不减，直到请求离批）
    fn context(&self) -> usize {
        self.prompt_len() + self.generated as usize
    }
    /// 稳定 decode 态：prompt 已追平且债 ≤ 1（每步 1 token 的采样-前向循环）。
    /// 混合步封顶的判据——批内有此态请求时本步总 token 受 mix_cap 约束
    fn is_decode(&self) -> bool {
        self.generated < self.max_tokens
            && self.forwarded >= self.prompt_len()
            && self.prompt_len() + self.generated as usize - self.forwarded <= 1
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
    cap: Box<dyn CapacityModel>,
    reqs: HashMap<ReqId, Req>,
    waiting: VecDeque<ReqId>,
    /// 完成回放队列（DeferredRelease）：本步达标者下一步首释放容量并回放事件
    done: VecDeque<ReqId>,
    batch: BatchBuffer,
    events: VecDeque<SchedEvent>,
    m: SchedMetrics,
}

impl Scheduler {
    pub fn new(cfg: SchedConfig) -> Self {
        let max_batch_tokens = cfg.max_batch_tokens;
        Self::with_capacity(cfg, TokenCapacity::new(max_batch_tokens))
    }

    pub fn with_capacity(cfg: SchedConfig, cap: impl CapacityModel + 'static) -> Self {
        Self {
            cfg,
            cap: Box::new(cap),
            reqs: HashMap::new(),
            waiting: VecDeque::new(),
            done: VecDeque::new(),
            batch: BatchBuffer::default(),
            events: VecDeque::new(),
            m: SchedMetrics::default(),
        }
    }

    /// 容量模型引用（引擎在步循环外调用 on_token / step_end / 下行转换）
    pub fn capacity_mut(&mut self) -> &mut dyn CapacityModel {
        self.cap.as_mut()
    }

    /// 提交。max_tokens == 0 立即终结（Length），不入队不占批。
    pub fn submit(&mut self, input: ReqInput) {
        if input.max_tokens == 0 {
            self.m.finished += 1;
            self.events.push_back(SchedEvent::Finished { id: input.id, reason: FinishReason::Length });
            return;
        }
        let ReqInput { id, prompt_token_ids, max_tokens } = input;
        self.reqs.insert(
            id,
            Req { prompt: prompt_token_ids, forwarded: 0, generated: 0, max_tokens, st: St::Waiting },
        );
        self.waiting.push_back(id);
    }

    /// 取消。waiting → 直接移除；running → 出批（释放容量与预算）；未知 id 无操作。
    /// 返回是否命中已知请求（幂等：重复取消/取消未知 id 不产生事件与计数）
    pub fn cancel(&mut self, id: ReqId) -> bool {
        let was = match self.reqs.get(&id).map(|r| r.st) {
            Some(St::Waiting) => {
                self.waiting.retain(|&x| x != id);
                self.reqs.remove(&id);
                true
            }
            Some(St::Running) => {
                self.batch.remove(id);
                self.m.batch_removes += 1;
                self.reqs.remove(&id);
                true
            }
            None => false,
        };
        if !was {
            return false;
        }
        self.m.cancelled += 1;
        self.cap.on_removed(id, Removal::Cancelled);
        self.events.push_back(SchedEvent::Cancelled(id));
        true
    }

    /// 前缀缓存命中回填：执行器恢复了 k 个 token 的递推/KV 状态（radix
    /// 命中，prefill 无须重算），调度债等量消减——预算切片/mix_cap/追平
    /// 判定全部按剩余前向段计算。与抢占的 forwarded 清零对称：一增一减
    /// 都是"执行器先行、调度器对账"的同步点。超额命中钳到债上界。
    pub fn prefix_hit(&mut self, id: ReqId, tokens: usize) {
        if let Some(r) = self.reqs.get_mut(&id) {
            let bound = r.prompt_len() + r.generated as usize;
            r.forwarded = (r.forwarded + tokens).min(bound);
        }
    }

    /// 投机步的多 token 发射记账：`plan()` 每步只按"1 个 token"记了一次
    /// （generated/forwarded 各 +1、`on_sample` 一次）；当执行器一步真实发出
    /// n > 1 个 token 时，引擎对**第 2 个起**的每个 token 调一次本方法补齐。
    ///
    /// 语义：把 `generated`/`forwarded` 同步前进 1（保持 decode 不变量
    /// `prompt_len + generated - forwarded == 1`），容量侧补一次 `on_sample`。
    /// **额度边界**：`generated >= max_tokens` 时返回 false（不记账）——调用方
    /// 必须丢弃该 token 并让请求以 Length 立即完成（`step` 拿不到剩余额度，
    /// 由发射处兜底：超发部分不外吐，执行器内部序列可能多推进 1..k 个 token，
    /// 但请求已终结，不会影响别的请求或后续输出）。
    ///
    /// 返回 true = 已记账（token 可外吐）。
    pub fn on_emitted(&mut self, id: ReqId) -> bool {
        let Some(r) = self.reqs.get_mut(&id) else { return false };
        if r.generated >= r.max_tokens {
            return false;
        }
        r.generated += 1;
        r.forwarded += 1;
        let fin = r.generated >= r.max_tokens;
        self.cap.on_sample(id);
        if fin {
            // 与 plan() 的完成判定同路：出批 + 回放队列（下一步首释放容量）
            self.batch.remove(id);
            self.m.batch_removes += 1;
            self.reqs.remove(&id);
            self.done.push_back(id);
        }
        true
    }

    /// 引擎强制抢占（容量模型 on_token 失败时调用，如块耗尽）。
    /// 批内仅剩此请求时拒绝（重排后立刻被强制再准入 → 活锁；应在上游拒绝请求）。
    pub fn preempt(&mut self, id: ReqId) -> bool {
        let running = matches!(self.reqs.get(&id).map(|r| r.st), Some(St::Running));
        if running && self.batch.len() > 1 {
            self.preempt_one(id);
            return true;
        }
        false
    }

    /// 推进一步 = 准入相位 + 规划相位（脚本件/测试用；引擎走两段式以在
    /// 两者之间跑执行器的建行与前缀命中对账——见 [`Self::advance`]）
    pub fn step(&mut self) -> StepPlan {
        self.advance();
        self.plan()
    }

    /// 准入相位：完成回放 → 准入（含容量反压）→ 抢占（增长驱动）。
    /// 事件在此产出，引擎须在 [`Self::plan`] 之前回放干净：新准入者的
    /// 前缀命中（执行器先行的状态恢复）会消减本步预算要切的债——晚应用
    /// 一步就会让规划多切 chunk，执行器按真实游标少喂，两侧账漂移。
    pub fn advance(&mut self) {
        self.m.steps += 1;

        // --- 完成回放（DeferredRelease）：上一步达标者此刻释放容量并回放事件；
        //     引擎已对其最后一个 token 调用过 on_token，序列完整 ---
        while let Some(id) = self.done.pop_front() {
            self.cap.on_removed(id, Removal::Finished);
            self.m.finished += 1;
            self.events.push_back(SchedEvent::Finished { id, reason: FinishReason::Length });
        }

        // --- 准入（FIFO + 容量反压）：头请求放得下才放（严格保序，不越过队首）；
        //     批空时无条件放——进度保证：超大请求独跑（真实引擎应在 API 层拒绝）---
        while self.batch.len() < self.cfg.max_running && !self.waiting.is_empty() {
            let head_ctx = self.reqs[self.waiting.front().unwrap()].context();
            if !self.batch.is_empty() && !self.cap.can_admit(head_ctx) {
                break; // 反压：等容量释放（完成回放会缩 demand），不为准入而抢占
            }
            self.admit_one();
        }
        // --- 抢占（增长驱动）：容量超压 → 驱逐进度最浅者到队尾（recompute）。
        //     上下文随采样单调增长，所以压力自然出现；至少保留 1 个在跑。
        //     已知简化：最新准入者（进度最浅）可能被反复弹回，但老请求有界完成
        //     保证收敛 ---
        while self.batch.len() > 1 && self.cap.over_pressure() {
            let victim = self
                .batch
                .iter()
                .min_by_key(|&id| self.reqs[&id].forwarded + self.reqs[&id].generated as usize)
                .unwrap();
            self.preempt_one(victim);
        }

    }

    /// 规划相位：本步预算切片（prefill/decode 两相位 + 混合步封顶）
    pub fn plan(&mut self) -> StepPlan {
        // --- 预算规划：两相位（批序 = FIFO 准入序）。
        //     债 backlog = prompt_len + generated - forwarded（未 forward 的 token 数）。
        //     债 ≤ 1 且 prompt 已 forward ⇒ decode（成本 1）；否则按 chunk 追赶，
        //     追平的那一步顺带采样（采样不额外计预算——它就是那最后一次 forward 的输出）。
        //     相位一 decode 先落位（每请求 1 token，ITL 进度优先）；相位二 prefill
        //     受混合步封顶约束：批内有稳定 decode 时总 token ≤ mix_cap（重步拆轻），
        //     否则全预算；decode 占满 cap 时下限保 PREFILL_MIN（进度保证）---
        let mut budget = self.cfg.token_budget;
        let any_decoder = self.batch.iter().any(|id| self.reqs[&id].is_decode());
        let mut cap = if any_decoder { self.cfg.mix_cap } else { self.cfg.token_budget };
        let mut plan = StepPlan { batch: self.batch.iter().collect(), ..Default::default() };
        for &id in &plan.batch {
            let r = self.reqs.get_mut(&id).unwrap();
            if r.generated >= r.max_tokens {
                continue; // 已达标（等待下一步回放）；这里不再分预算
            }
            let backlog = r.prompt_len() + r.generated as usize - r.forwarded;
            if backlog <= 1 && r.forwarded >= r.prompt_len() {
                // decode：forward 最后采样的 token（空 prompt 首步 backlog=0 同样直接采样）
                if budget < backlog {
                    break;
                }
                budget -= backlog;
                cap = cap.saturating_sub(backlog);
                r.forwarded += backlog;
                r.generated += 1;
                plan.decode.push(id);
                self.cap.on_sample(id);
            }
        }
        // decode 占满 cap（slack 0）时给 prefill 留下限（等待中的 prefill 有界
        // 推进）。仅 slack==0 触发：decode 行数 < mix_cap 时 T ≤ mix_cap 是严格
        // 不变量。有 prefill 工作的请求 = 非 decode 态且未达标
        if any_decoder
            && cap == 0
            && self.batch.iter().any(|id| {
                let r = &self.reqs[&id];
                r.generated < r.max_tokens && !r.is_decode()
            })
        {
            cap = PREFILL_MIN;
        }
        for &id in &plan.batch {
            let r = self.reqs.get_mut(&id).unwrap();
            if r.generated >= r.max_tokens {
                continue;
            }
            let backlog = r.prompt_len() + r.generated as usize - r.forwarded;
            if backlog <= 1 && r.forwarded >= r.prompt_len() {
                continue; // 相位一已落位
            }
            let mut chunk = backlog.min(self.cfg.chunk_size).min(budget).min(cap);
            if chunk == 0 {
                break;
            }
            // 块边界对齐（上方 snap_align 论证）：圆掉不足一个粒度且会归零
            // 时不做（PREFILL_MIN 短推时进度优先，那一步不出检查点）
            if self.cfg.snap_align > 1 {
                let a = chunk - chunk % self.cfg.snap_align;
                if a > 0 {
                    chunk = a;
                }
            }
            r.forwarded += chunk;
            budget -= chunk;
            cap -= chunk;
            plan.prefill.push((id, chunk));
            if r.forwarded == r.prompt_len() + r.generated as usize {
                // 追平：本次 forward 的最后一个 token 直接产出下一个采样
                r.generated += 1;
                plan.decode.push(id);
                self.cap.on_sample(id);
            }
        }

        // --- 完成判定：DeferredRelease，达标者入队，下一步首统一回放 ---
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
            self.done.push_back(id);
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
        self.batch.is_empty() && self.waiting.is_empty() && self.done.is_empty()
    }

    fn admit_one(&mut self) {
        if let Some(id) = self.waiting.pop_front() {
            let r = self.reqs.get_mut(&id).unwrap();
            r.st = St::Running;
            let (prompt, generated) = (r.prompt.clone(), r.generated);
            self.batch.add(id);
            self.m.batch_adds += 1;
            self.m.admitted += 1;
            self.cap.on_admitted(id, &prompt, generated);
            self.events.push_back(SchedEvent::Admitted(id));
        }
    }

    fn preempt_one(&mut self, victim: ReqId) {
        self.batch.remove(victim);
        self.m.batch_removes += 1;
        self.m.preempted += 1;
        let r = self.reqs.get_mut(&victim).unwrap();
        r.st = St::Waiting;
        r.forwarded = 0; // recompute：forward 进度清零；generated 保留
        self.waiting.push_back(victim);
        self.cap.on_removed(victim, Removal::Preempted);
        self.events.push_back(SchedEvent::Preempted(victim));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Instant;

    fn cfg(budget: usize, max_running: usize, chunk: usize) -> SchedConfig {
        // 默认容量给到天文数字：非压力用例不关心抢占；mix_cap 同理不封顶
        SchedConfig {
            token_budget: budget,
            max_running,
            chunk_size: chunk,
            max_batch_tokens: 1 << 30,
            mix_cap: 1 << 30,
            snap_align: 1,
        }
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
        s.submit(ReqInput { id: 1, prompt_token_ids: vec![0; 10], max_tokens: 3 });
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
        s.submit(ReqInput { id: 1, prompt_token_ids: vec![0; 6], max_tokens: 10 }); // 老请求：长 prefill
        s.submit(ReqInput { id: 2, prompt_token_ids: vec![0; 1], max_tokens: 10 }); // 新请求：短 prefill
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
            s.submit(ReqInput { id, prompt_token_ids: vec![0; 1], max_tokens: 2 });
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
            mix_cap: 1 << 30,
            snap_align: 1,
        });
        s.submit(ReqInput { id: 1, prompt_token_ids: vec![0; 2], max_tokens: 8 }); // A
        s.submit(ReqInput { id: 2, prompt_token_ids: vec![0; 2], max_tokens: 8 }); // C
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
        s.submit(ReqInput { id: 1, prompt_token_ids: vec![0; 2], max_tokens: 10 }); // B 运行中
        s.step();
        s.poll_events();
        s.submit(ReqInput { id: 2, prompt_token_ids: vec![0; 1], max_tokens: 1 }); // A 等待中
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
        s.submit(ReqInput { id: 1, prompt_token_ids: vec![0; 1], max_tokens: 0 });
        let evs = s.poll_events();
        assert!(matches!(evs[0], SchedEvent::Finished { id: 1, reason: FinishReason::Length }));
        assert!(s.is_idle(), "max_tokens=0 不入队");

        s.submit(ReqInput { id: 2, prompt_token_ids: vec![0; 1], max_tokens: 2 });
        let (plans, _) = run_to_idle(&mut s, 20);
        let n: usize = plans.iter().map(|p| p.decode.len()).sum();
        assert_eq!(n, 2);
    }

    /// s7 · 确定性：同一命令序列 → 逐位相同的计划与事件（批不变性的地基）
    #[test]
    fn s7_determinism_same_inputs_same_plans() {
        let scenario = |s: &mut Scheduler| {
            for id in 1..=4u64 {
                s.submit(ReqInput {
                    id,
                    prompt_token_ids: vec![0; (id % 3 + 1) as usize],
                    max_tokens: id as u32,
                });
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
        for id in 1..=20u64 {
            s.submit(ReqInput { id, prompt_token_ids: vec![0; 2], max_tokens: 2 });
        }
        let (_, events) = run_to_idle(&mut s, 500);
        assert_eq!(events.iter().filter(|e| matches!(e, SchedEvent::Finished { .. })).count(), 20);
        assert!(s.metrics().steps < 100, "20 个小请求不该超过 100 步：{}", s.metrics().steps);
        assert!(t0.elapsed().as_millis() < 1000, "纯逻辑调度器必须亚毫秒级处理此规模");
    }

    /// s10 · 混合步封顶（mix_cap）：满负荷 ITL p99 账——稳定 decode 在批时
    ///      单步总 token（decode + prefill）≤ mix_cap（波次切换/爬坡期的大
    ///      prefill chunk 不再把搭乘的 decode 行打进计算受限的重步）；批内
    ///      无稳定 decode（纯 prefill 波）不受限（TTFT/吞吐优先）；cap 被
    ///      decode 行占满时至少留 PREFILL_MIN 给 prefill（进度保证）
    #[test]
    fn s10_mixed_step_cap_shields_decoders() {
        // 预算 16 / cap 8：请求 1（prompt 1）步 1 追平后成为稳定 decode，
        // 请求 2（prompt 30）的 prefill 从步 2 起被 cap 切片
        let mut s = Scheduler::new(SchedConfig {
            token_budget: 16,
            max_running: 8,
            chunk_size: 16,
            max_batch_tokens: 1 << 30,
            mix_cap: 8,
            snap_align: 1,
        });
        s.submit(ReqInput { id: 1, prompt_token_ids: vec![0; 1], max_tokens: 10 });
        s.submit(ReqInput { id: 2, prompt_token_ids: vec![0; 30], max_tokens: 10 });
        let (plans, _) = run_to_idle(&mut s, 100);

        // 步 1：批内无稳定 decode（两者都在追平）→ 全预算：1 追平 + 2 推 15
        assert_eq!(plans[0].prefill, vec![(1, 1), (2, 15)], "无 decode 在批 → 不封顶");

        // 步 2+：请求 1 稳定 decode → 混合步 T = chunk + 1 ≤ 8（chunk ≤ 7）
        let mut chunks_after = Vec::new();
        for p in plans.iter().skip(1) {
            if let Some(&(_, n)) = p.prefill.iter().find(|&&(id, _)| id == 2) {
                assert!(!p.decode.contains(&2) || n <= 7, "追平步同样计入 cap");
                if !p.decode.contains(&2) {
                    assert!(p.decode.contains(&1), "稳定 decode 每步在场");
                    assert!(n <= 7, "混合步总 token ≤ cap 8（chunk {n} + decode 1）");
                }
                chunks_after.push(n);
            }
        }
        assert_eq!(
            chunks_after.iter().sum::<usize>(),
            15,
            "cap 切片不丢 token：30 = 15（步 1）+ Σ 后续"
        );
        // 收敛与总量：两请求各 10 token
        assert_eq!(plans.iter().map(|p| p.decode.len()).sum::<usize>(), 20);

        // 下限保护：cap 2 < decode 行数 4 → prefill 仍至少推进 PREFILL_MIN
        let mut s = Scheduler::new(SchedConfig {
            token_budget: 64,
            max_running: 8,
            chunk_size: 64,
            max_batch_tokens: 1 << 30,
            mix_cap: 2,
            snap_align: 1,
        });
        for id in 1..=4u64 {
            s.submit(ReqInput { id, prompt_token_ids: vec![0; 1], max_tokens: 10 });
        }
        s.submit(ReqInput { id: 5, prompt_token_ids: vec![0; 100], max_tokens: 10 });
        s.step(); // 四请求追平 + 5 推 60（步 1 无稳定 decode → 全预算 64）
        let p2 = s.step();
        let n5: usize = p2.prefill.iter().filter(|&&(id, _)| id == 5).map(|(_, n)| n).sum();
        assert!(n5 >= PREFILL_MIN, "decode 行占满 cap 时 prefill 至少推进 {PREFILL_MIN}（得 {n5}）");
    }

    /// s9 · 容量钩子真实可插拔：准入与抢占由容量模型驱动，而非调度器内置聚合
    struct FlipCap {
        admit: Arc<AtomicBool>,
        pressure: Arc<AtomicBool>,
        calls: AtomicUsize,
    }
    impl CapacityModel for FlipCap {
        fn can_admit(&mut self, _ctx_len: usize) -> bool {
            self.admit.load(Ordering::Relaxed)
        }
        fn on_admitted(&mut self, _req: ReqId, _prompt: &[u32], _generated: u32) {
            self.calls.fetch_add(1, Ordering::Relaxed);
        }
        fn on_sample(&mut self, _req: ReqId) {}
        fn on_token(&mut self, _req: ReqId, _token_id: u32) {}
        fn on_removed(&mut self, _req: ReqId, _how: Removal) {}
        fn over_pressure(&mut self) -> bool {
            self.pressure.load(Ordering::Relaxed)
        }
    }

    #[test]
    fn s9_capacity_model_drives_admission_and_preemption() {
        let admit = Arc::new(AtomicBool::new(false));
        let pressure = Arc::new(AtomicBool::new(false));
        let mut s = Scheduler::with_capacity(
            cfg(64, 8, 64),
            FlipCap { admit: admit.clone(), pressure: pressure.clone(), calls: AtomicUsize::new(0) },
        );
        s.submit(ReqInput { id: 1, prompt_token_ids: vec![0; 2], max_tokens: 10 });
        s.submit(ReqInput { id: 2, prompt_token_ids: vec![0; 2], max_tokens: 10 });

        // 容量说"放不下"：槽位充足也反压——但批空时首个请求无条件准入（进度保证：
        // 反压不能演变成全局停车；放不下的超大请求应在 API 层拒绝）
        let plan = s.step();
        assert_eq!(plan.batch.len(), 1, "批空时无条件准入首个请求");
        let evs = s.poll_events();
        assert_eq!(evs.iter().filter(|e| matches!(e, SchedEvent::Admitted(_))).count(), 1);

        // 容量持续说"放不下"：等待者在队首被反压，不越过、不抢占队内请求
        let plan = s.step();
        assert_eq!(plan.batch.len(), 1, "can_admit=false 时头请求停在队首");

        // 容量恢复：等待者立刻进批
        admit.store(true, Ordering::Relaxed);
        let plan = s.step();
        assert_eq!(plan.batch.len(), 2);
        let evs = s.poll_events();
        assert_eq!(evs.iter().filter(|e| matches!(e, SchedEvent::Admitted(_))).count(), 1);

        // 容量报超压：驱逐最浅者（recompute），即使聚合上毫无压力
        pressure.store(true, Ordering::Relaxed);
        s.step();
        let evs = s.poll_events();
        assert_eq!(evs.iter().filter(|e| matches!(e, SchedEvent::Preempted(_))).count(), 1);
        pressure.store(false, Ordering::Relaxed);

        let (_, events) = run_to_idle(&mut s, 100);
        assert_eq!(
            events.iter().filter(|e| matches!(e, SchedEvent::Finished { .. })).count(),
            2,
            "压力解除后全部收敛"
        );
    }

    /// s11 · 前缀命中回填（prefix_hit）：执行器恢复 k token 后调度债等量
    ///      消减——预算按剩余段切片、追平判定前移、超额命中安全钳制
    #[test]
    fn s11_prefix_hit_reduces_debt() {
        let mut s = Scheduler::new(cfg(4, 8, 4));
        s.submit(ReqInput { id: 1, prompt_token_ids: vec![0; 10], max_tokens: 3 });
        s.prefix_hit(1, 8); // 执行器恢复 8 token：债 10 → 2
        let (plans, _) = run_to_idle(&mut s, 50);
        let prefill_chunks: Vec<usize> =
            plans.iter().filter(|p| !p.prefill.is_empty()).map(|p| p.prefill[0].1).collect();
        assert_eq!(prefill_chunks, vec![2], "命中后只切剩余 2 token");
        assert_eq!(plans.iter().map(|p| p.decode.len()).sum::<usize>(), 3);

        // 超额钳制：命中 > 债 → 钳到上界（不产生负债/越界 forwarded）
        let mut s = Scheduler::new(cfg(4, 8, 4));
        s.submit(ReqInput { id: 1, prompt_token_ids: vec![0; 3], max_tokens: 2 });
        s.prefix_hit(1, 100);
        let (plans, _) = run_to_idle(&mut s, 50);
        assert!(plans.iter().all(|p| p.prefill.is_empty()), "全额命中无 prefill 段");
        assert_eq!(plans.iter().map(|p| p.decode.len()).sum::<usize>(), 2);
    }

    /// s12 · prefill chunk 尾对齐（前缀缓存检查点只在"forward 最后一 token
    /// 落在块边界"时存在，故非末块 chunk 尾圆到 snap_align；末块不圆——
    /// 追平步照常采样，TTFT 少付一步；对齐全开与全关的总 token 数一致）
    #[test]
    fn s12_prefill_chunks_align_to_checkpoint_granularity() {
        let run = |align: usize| {
            let mut s = Scheduler::new(SchedConfig {
                token_budget: 16,
                max_running: 8,
                chunk_size: 16,
                max_batch_tokens: 1 << 30,
                mix_cap: 1 << 30,
                snap_align: align,
            });
            s.submit(ReqInput { id: 1, prompt_token_ids: vec![0; 10], max_tokens: 2 });
            let (plans, _) = run_to_idle(&mut s, 50);
            let chunks: Vec<usize> =
                plans.iter().filter(|p| !p.prefill.is_empty()).map(|p| p.prefill[0].1).collect();
            let decodes = plans.iter().map(|p| p.decode.len()).sum::<usize>();
            (chunks, decodes)
        };
        assert_eq!(run(1), (vec![10], 2), "不对齐：prompt 一步追平");
        assert_eq!(run(4), (vec![8, 2], 2), "对齐 4：首块 8（块边界）+ 末块 2");
        // 多块预算切片：仍逐段对齐，末块不圆
        let mut s = Scheduler::new(SchedConfig {
            token_budget: 16,
            max_running: 8,
            chunk_size: 6,
            max_batch_tokens: 1 << 30,
            mix_cap: 1 << 30,
            snap_align: 4,
        });
        s.submit(ReqInput { id: 1, prompt_token_ids: vec![0; 10], max_tokens: 2 });
        let (plans, _) = run_to_idle(&mut s, 50);
        let chunks: Vec<usize> =
            plans.iter().filter(|p| !p.prefill.is_empty()).map(|p| p.prefill[0].1).collect();
        assert_eq!(chunks, vec![4, 4, 2], "chunk 6 圆到 4，末块 2");
        assert_eq!(plans.iter().map(|p| p.decode.len()).sum::<usize>(), 2);
    }
}
