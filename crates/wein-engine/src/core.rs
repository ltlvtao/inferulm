//! EngineCore：真实 engine core 的第一步——统一 token 预算调度器（wein-sched）
//! + 可插拔执行器，组装成与 MockEngine 同协议的引擎任务。
//!
//! 对外只经 EngineHandle（Submit/Cancel + 事件流）——serve 验收套件对引擎实现
//! 完全无感知，MockEngine 可与之互换，这就是 TDD 外→内协议锚的兑现。
//!
//! 三线程拓扑（02-architecture §5）中的 "engine core 线程"：当前为 tokio 任务，
//! 协议是 channel，后续搬到专用线程不动接口。exec 线程（§5 第三线程）以
//! `Executor` trait 预留——真实实现是 CUDA 图/NPU 图；M0 用 ScriptedExecutor。
use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::time::sleep;

use wein_memory::state::{MemConfig, MemState, MemStats};
use wein_sched::{
    CapacityModel, FinishReason as SchedFinish, Removal, ReqId, ReqInput, SchedConfig, SchedEvent,
    Scheduler, StepPlan,
};

use crate::handle::EngineHandle;
use crate::protocol::{EngineCommand, EngineEvent, EngineRequest, FinishReason, RequestId};

pub struct CoreConfig {
    pub sched: SchedConfig,
    /// 脚本执行器的 token 表（循环取用；与 MockScript.pieces 对齐以便 parity 测试）
    pub pieces: Vec<u32>,
    /// 每步墙钟延迟（模拟 kernel 时间；0 = 尽快推进）
    pub step_delay: Duration,
    /// KV 内存层（Some = BlockPool 水位驱动准入/抢占 + radix 前缀缓存；None = 聚合 token 模型）
    pub memory: Option<MemConfig>,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            sched: SchedConfig::default(),
            pieces: vec![0, 1, 2, 3, 4, 5, 6, 7],
            step_delay: Duration::from_millis(1),
            memory: None,
        }
    }
}

impl CoreConfig {
    /// 脚本式配置：调度参数取默认（budget 512 / max_running 256 / chunk 256）。
    pub fn scripted(pieces: Vec<u32>, step_delay: Duration) -> Self {
        Self { pieces, step_delay, ..Default::default() }
    }
}

/// BlockPool 容量模型 ↔ 调度器钩子的适配器（依赖规则：wein-memory 不依赖
/// wein-sched，适配发生在 engine 侧——orphan 规则下适配器类型属于本 crate）。
///
/// on_token 失败（块耗尽）与 on_admitted 失败（批空无条件准入但放不下）都进
/// `forced`，引擎在步循环内 drain：先试 sched.preempt（recompute），批内仅剩
/// 此请求时 preempt 拒绝 → abort（超大请求应在 API 层拒绝，这里是兜底）
struct PoolCapacity {
    mem: MemState,
    forced: Vec<ReqId>,
}

impl CapacityModel for PoolCapacity {
    fn can_admit(&mut self, ctx_len: usize) -> bool {
        self.mem.can_admit(ctx_len)
    }

    fn on_admitted(&mut self, req: ReqId, prompt: &[u32], generated: u32) {
        if !self.mem.admit(req, prompt, generated) {
            self.forced.push(req);
        }
    }

    fn on_sample(&mut self, _req: ReqId) {}

    fn on_token(&mut self, req: ReqId, token_id: u32) {
        if !self.mem.on_token(req, token_id) {
            self.forced.push(req);
        }
    }

    fn on_removed(&mut self, req: ReqId, how: Removal) {
        match how {
            Removal::Finished => self.mem.finish(req),
            Removal::Preempted => self.mem.release(req, true), // keep_seq：重准入全上下文命中
            Removal::Cancelled => self.mem.release(req, false),
        }
    }

    fn over_pressure(&mut self) -> bool {
        self.mem.over_pressure()
    }

    fn step_end(&mut self) {
        self.mem.step_end();
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        Some(self)
    }
}

/// 引擎侧可断言状态（对应 MockState 的角色；BASELINE §4 L2 插桩的载体）。
#[derive(Default)]
pub struct EngineStats {
    accepted: AtomicUsize,
    cancelled: AtomicUsize,
    finished: AtomicUsize,
    preempted: AtomicUsize,
    steps: AtomicU64,
    /// 内存层快照（每步末刷新；未启用内存层时恒为默认值）
    mem: Mutex<MemStats>,
}

impl EngineStats {
    pub fn accepted_count(&self) -> usize {
        self.accepted.load(Ordering::Relaxed)
    }
    pub fn cancelled_count(&self) -> usize {
        self.cancelled.load(Ordering::Relaxed)
    }
    pub fn finished_count(&self) -> usize {
        self.finished.load(Ordering::Relaxed)
    }
    pub fn preempted_count(&self) -> usize {
        self.preempted.load(Ordering::Relaxed)
    }
    pub fn step_count(&self) -> u64 {
        self.steps.load(Ordering::Relaxed)
    }
    pub fn mem(&self) -> MemStats {
        self.mem.lock().expect("mem stats 锁中毒").clone()
    }
}

/// 执行器：engine core 只做决策，一步的"算"由执行器完成（§5 exec 线程的 trait 预留）。
pub trait Executor: Send {
    /// 执行一步，返回每个 decode 槽位产出的 token
    fn step(&mut self, plan: &StepPlan) -> Vec<(ReqId, u32)>;
    /// 步末回收：清理已离批请求的执行器内部状态（DeferredRelease 钩子）
    fn retain(&mut self, active: &[ReqId]);
    /// 准入通知：prompt 在此首次可见（执行器建立自己的序列/KV 记账）。
    /// 空指针不算——空 prompt 由执行器自行决定 BOS 语义
    fn admitted(&mut self, _id: ReqId, _prompt: &[u32]) {}
    /// 抢占通知：调度器会以 recompute 语义重放（forwarded 清零）——
    /// 执行器须把自己的前向游标一并归零，否则重放位置错位（x6 验收钉死）
    fn preempted(&mut self, _id: ReqId) {}
    /// 行容量（并发请求上限）：执行器每请求占一行（块表/递推/conv 状态池），
    /// 超额准入会在执行器内部 assert 崩掉引擎任务。engine 把它钳进
    /// max_running——超容请求在调度器排队等行释放（e5 钉死）。默认无界
    /// （脚本件无行概念）
    fn row_capacity(&self) -> usize {
        usize::MAX
    }
}

/// 确定性执行器：每请求独立循环取 pieces（与 MockEngine 相同的可预测输出，
/// parity 测试因此可以逐字节比对）。
pub struct ScriptedExecutor {
    pieces: Vec<u32>,
    counters: HashMap<ReqId, usize>,
}

impl ScriptedExecutor {
    pub fn new(pieces: Vec<u32>) -> Self {
        Self { pieces, counters: HashMap::new() }
    }
}

impl Executor for ScriptedExecutor {
    fn step(&mut self, plan: &StepPlan) -> Vec<(ReqId, u32)> {
        let n = self.pieces.len().max(1);
        plan.decode
            .iter()
            .map(|&id| {
                let c = self.counters.entry(id).or_insert(0);
                let tok = self.pieces[*c % n];
                *c += 1;
                (id, tok)
            })
            .collect()
    }

    fn retain(&mut self, active: &[ReqId]) {
        self.counters.retain(|k, _| active.contains(k));
    }
}

pub struct EngineCore;

impl EngineCore {
    /// 启动 engine core 任务（内置脚本执行器），返回 (serve 用的句柄, 断言用的状态)。
    pub fn spawn(cfg: CoreConfig) -> (EngineHandle, Arc<EngineStats>) {
        let exec = ScriptedExecutor::new(cfg.pieces.clone());
        Self::spawn_with(cfg, Box::new(exec))
    }

    /// 带执行器启动：真实执行器（wein-exec）经此注入。组装发生在调用方
    /// （serve/cli 层）——engine 不依赖 exec（依赖规则 §2：单向，无环）
    pub fn spawn_with(
        cfg: CoreConfig,
        exec: Box<dyn Executor>,
    ) -> (EngineHandle, Arc<EngineStats>) {
        let (cmd_tx, cmd_rx) = tokio::sync::mpsc::unbounded_channel();
        let stats = Arc::new(EngineStats::default());
        tokio::spawn(Self::run(cfg, cmd_rx, stats.clone(), exec));
        (EngineHandle::new(cmd_tx), stats)
    }

    async fn run(
        cfg: CoreConfig,
        mut cmd_rx: UnboundedReceiver<EngineCommand>,
        stats: Arc<EngineStats>,
        mut exec: Box<dyn Executor>,
    ) {
        // 容量模型二选一：BlockPool 水位（memory）或聚合 token（默认）。
        // 调度策略不变——准入反压与抢占触发全部经 CapacityModel 钩子。
        // 执行器行容量钳进 max_running：超额准入会在 exec 内部 assert 崩掉
        // 引擎任务（I1 满负荷批界）——超容请求排队等行（e5）
        let mut cfg = cfg;
        cfg.sched.max_running = cfg.sched.max_running.min(exec.row_capacity());
        let mut sched = match cfg.memory {
            Some(mem_cfg) => Scheduler::with_capacity(
                cfg.sched,
                PoolCapacity { mem: MemState::new(mem_cfg), forced: Vec::new() },
            ),
            None => Scheduler::new(cfg.sched),
        };
        let mut senders: HashMap<RequestId, UnboundedSender<EngineEvent>> = HashMap::new();
        // 停止词表（chat eos 等）：随请求提交进 来，请求离场即清（W5）
        let mut stops: HashMap<RequestId, Vec<u32>> = HashMap::new();
        // prompt 暂存：exec 建行发生在调度**准入**时（SchedEvent::Admitted），
        // 不是提交时——超容请求在 waiting 排队不占执行器行（e5）。首次准入
        // 即消费；抢占重准入不再触发（exec 保留行/状态走 recompute）
        let mut prompts: HashMap<RequestId, Vec<u32>> = HashMap::new();

        while let Some(cmd) = cmd_rx.recv().await {
            Self::handle(&mut sched, &mut senders, &mut stops, &mut prompts, &stats, &mut exec, cmd);
            // 批量吸收同刻到达的命令再统一推进：一次调度决策管一批提交
            Self::drain(&mut cmd_rx, &mut sched, &mut senders, &mut stops, &mut prompts, &stats, &mut exec);

            // 推进直到空闲；空闲即停车等新命令（不忙转）。
            // 步是原子的：本步 token 全部发出（并记入内存层）→ 完成事件 → 回收；
            // 步间命令（取消/新提交）在下一步首 drain 时生效——
            // 因此取消后客户端至多再收到在飞那一步的 token（e2 验收钉死的边界）
            while !sched.is_idle() {
                Self::drain(&mut cmd_rx, &mut sched, &mut senders, &mut stops, &mut prompts, &stats, &mut exec);
                let plan = sched.step();
                // 事件先于 exec.step：本步新准入者先建行（exec.admitted）
                // 再进前向——否则 step 查不到行会静默空转
                Self::pump(&mut sched, &mut senders, &mut stops, &mut prompts, &stats, &mut exec);
                stats.steps.fetch_add(1, Ordering::Relaxed);
                if !cfg.step_delay.is_zero() {
                    sleep(cfg.step_delay).await;
                }
                for (id, tok) in exec.step(&plan) {
                    // 停止词表命中：token 不外吐，请求以 Stop 终结（vLLM 语义）。
                    // 决策必须在发射处——调度器看到 token 时它已经出去了
                    let hit = stops.get(&id).is_some_and(|s| s.contains(&tok));
                    if !hit {
                        if let Some(tx) = senders.get(&id) {
                            let _ = tx.send(EngineEvent::Token(tok));
                        }
                    }
                    // 内存布局推进（token id 才在这里可知；TokenCapacity 实现为空操作）
                    sched.capacity_mut().on_token(id, tok);
                    if hit {
                        stats.finished.fetch_add(1, Ordering::Relaxed);
                        if let Some(tx) = senders.remove(&id) {
                            let _ = tx.send(EngineEvent::Finished(FinishReason::Stop));
                        }
                        stops.remove(&id);
                        // 出调度器（含内存层释放）；后续 poll 的 Cancelled
                        // 只做 sender 清理（此处已移除，幂等）
                        sched.cancel(id);
                    }
                }
                // 强制抢占：内存层报的失败者（块耗尽/批空放不下的超大请求）。
                // preempt 被拒（批内仅剩此请求）→ abort 兜底
                let forced: Vec<ReqId> = match sched.capacity_mut().as_any_mut() {
                    Some(any) => any
                        .downcast_mut::<PoolCapacity>()
                        .map(|pc| std::mem::take(&mut pc.forced))
                        .unwrap_or_default(),
                    None => Vec::new(),
                };
                for id in forced {
                    if !sched.preempt(id) {
                        Self::abort(&mut sched, &mut senders, &mut stops, &mut prompts, &stats, id);
                    }
                }
                // 内存层记账快照（每步末刷新；供断言与观测）
                if let Some(any) = sched.capacity_mut().as_any_mut() {
                    if let Some(pc) = any.downcast_mut::<PoolCapacity>() {
                        *stats.mem.lock().expect("mem stats 锁中毒") = pc.mem.stats();
                    }
                }
                Self::pump(&mut sched, &mut senders, &mut stops, &mut prompts, &stats, &mut exec);
                // DeferredRelease：步末回收执行器内部状态。保留集合 = 仍有
                // 事件流者——含被抢占的等待者（它们会带着已生成 token 回来，
                // 计数状态不得清；im1 抓到：按本步在批清理会把重算请求的
                // 输出序打回起点）
                let keep: Vec<ReqId> = senders.keys().copied().collect();
                exec.retain(&keep);
                sched.capacity_mut().step_end();
            }
        }
        // cmd_tx 全部 drop = 引擎关闭；senders 随之 drop，事件流自然关闭
    }

    /// 吸收当前排队中的全部命令并回放事件。
    /// 步间必须 drain：取消要及时生效（不能等当前批跑完），新提交也要能进批
    /// （continuous batching 的准入路径——e2 验收钉死的语义）。
    fn drain(
        cmd_rx: &mut UnboundedReceiver<EngineCommand>,
        sched: &mut Scheduler,
        senders: &mut HashMap<RequestId, UnboundedSender<EngineEvent>>,
        stops: &mut HashMap<RequestId, Vec<u32>>,
        prompts: &mut HashMap<RequestId, Vec<u32>>,
        stats: &EngineStats,
        exec: &mut Box<dyn Executor>,
    ) {
        while let Ok(cmd) = cmd_rx.try_recv() {
            Self::handle(sched, senders, stops, prompts, stats, exec, cmd);
        }
        Self::pump(sched, senders, stops, prompts, stats, exec);
    }

    fn handle(
        sched: &mut Scheduler,
        senders: &mut HashMap<RequestId, UnboundedSender<EngineEvent>>,
        stops: &mut HashMap<RequestId, Vec<u32>>,
        prompts: &mut HashMap<RequestId, Vec<u32>>,
        stats: &EngineStats,
        exec: &mut Box<dyn Executor>,
        cmd: EngineCommand,
    ) {
        match cmd {
            EngineCommand::Submit(req) => {
                stats.accepted.fetch_add(1, Ordering::Relaxed);
                let EngineRequest {
                    id, prompt_token_ids, max_tokens, sampling, respond_to,
                } = req;
                senders.insert(id, respond_to);
                if !sampling.stop_token_ids.is_empty() {
                    stops.insert(id, sampling.stop_token_ids);
                }
                if let Some(tx) = senders.get(&id) {
                    let _ = tx.send(EngineEvent::Accepted);
                }
                // prompt 暂存：建行推迟到调度准入（pump 的 Admitted 臂）——
                // 空 prompt 的 BOS 语义由执行器在 admitted 决定
                sched.submit(ReqInput { id, prompt_token_ids: prompt_token_ids.clone(), max_tokens });
                prompts.insert(id, prompt_token_ids);
            }
            EngineCommand::Cancel(id) => Self::abort(sched, senders, stops, prompts, stats, id),
        }
    }

    /// 取消/中止一个请求：出调度器 + best-effort 通知 Aborted。
    /// 幂等：调度器已不认识它（先前已终结/移除）则什么都不做——
    /// forced 列表可能对同一请求重复报告（admit 失败 + 无布局 token 增长失败）
    fn abort(
        sched: &mut Scheduler,
        senders: &mut HashMap<RequestId, UnboundedSender<EngineEvent>>,
        stops: &mut HashMap<RequestId, Vec<u32>>,
        prompts: &mut HashMap<RequestId, Vec<u32>>,
        stats: &EngineStats,
        id: RequestId,
    ) {
        stops.remove(&id);
        prompts.remove(&id);
        if !sched.cancel(id) {
            return;
        }
        stats.cancelled.fetch_add(1, Ordering::Relaxed);
        if let Some(tx) = senders.get(&id) {
            let _ = tx.send(EngineEvent::Finished(FinishReason::Aborted));
        }
    }

    /// 回放调度器事件。抢占对 serve 不可见——它是引擎内部实现细节，
    /// serve 只是看到 token 流稍后到达；Finished 是唯一对外终结事件。
    fn pump(
        sched: &mut Scheduler,
        senders: &mut HashMap<RequestId, UnboundedSender<EngineEvent>>,
        stops: &mut HashMap<RequestId, Vec<u32>>,
        prompts: &mut HashMap<RequestId, Vec<u32>>,
        stats: &EngineStats,
        exec: &mut Box<dyn Executor>,
    ) {
        let mut admitted_ids: Vec<ReqId> = Vec::new();
        for ev in sched.poll_events() {
            match ev {
                // 建行时机：调度准入处。但**先回收后建行**——同轮回放的
                // Finished/Aborted 已从 senders 除名，滞留一步的完成行
                // （DeferredRelease）先归还行池再为准入者分配，峰值行数
                // 不越过 row_capacity（e5）。首见消费暂存 prompt；抢占
                // 重准入查不到 → 不重复建行（exec 保留状态走 recompute）
                SchedEvent::Admitted(id) => admitted_ids.push(id),
                SchedEvent::Preempted(id) => {
                    stats.preempted.fetch_add(1, Ordering::Relaxed);
                    // 重算语义的另一半：调度器清自己的 forwarded，执行器
                    // 也须清前向游标（重放会覆盖写同样的 KV——确定性前向下幂等）
                    exec.preempted(id);
                }
                SchedEvent::Cancelled(id) => {
                    senders.remove(&id);
                    stops.remove(&id);
                    prompts.remove(&id);
                }
                SchedEvent::Finished { id, reason } => {
                    stats.finished.fetch_add(1, Ordering::Relaxed);
                    stops.remove(&id);
                    if let Some(tx) = senders.remove(&id) {
                        let _ = tx.send(EngineEvent::Finished(map_reason(reason)));
                    }
                }
            }
        }
        if !admitted_ids.is_empty() {
            let keep: Vec<ReqId> = senders.keys().copied().collect();
            exec.retain(&keep);
            for id in admitted_ids {
                if let Some(p) = prompts.remove(&id) {
                    exec.admitted(id, &p);
                }
            }
        }
    }
}

fn map_reason(r: SchedFinish) -> FinishReason {
    match r {
        SchedFinish::Stop => FinishReason::Stop,
        SchedFinish::Length => FinishReason::Length,
        SchedFinish::Aborted => FinishReason::Aborted,
    }
}
