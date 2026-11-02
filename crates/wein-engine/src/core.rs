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
use crate::protocol::{
    EngineCommand, EngineEvent, EngineRequest, FinishReason, RequestId, SampleParams,
};

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
/// 全部字段是无锁原子量——serve 的 `/health`、`/metrics` 直接读这里。
pub struct EngineStats {
    accepted: AtomicUsize,
    cancelled: AtomicUsize,
    finished: AtomicUsize,
    preempted: AtomicUsize,
    steps: AtomicU64,
    /// 上一步的批行数（gauge：exec 单步前向的行数 = prefill token + decode 行）
    batch_rows: AtomicUsize,
    /// 前缀缓存命中 token 累计（执行器 `admitted` 上报的恢复量之和）
    prefix_hit_tokens: AtomicU64,
    /// 执行器上下文窗口（serve 前置校验用；`usize::MAX` = 无上限）
    ctx_limit: AtomicUsize,
    /// 内存层快照（每步末刷新；未启用内存层时恒为默认值）
    mem: Mutex<MemStats>,
}

impl Default for EngineStats {
    fn default() -> Self {
        Self {
            accepted: AtomicUsize::new(0),
            cancelled: AtomicUsize::new(0),
            finished: AtomicUsize::new(0),
            preempted: AtomicUsize::new(0),
            steps: AtomicU64::new(0),
            batch_rows: AtomicUsize::new(0),
            prefix_hit_tokens: AtomicU64::new(0),
            // AtomicUsize::default() = 0 会被误读成"上下文窗口为 0"
            ctx_limit: AtomicUsize::new(usize::MAX),
            mem: Mutex::new(MemStats::default()),
        }
    }
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
    /// 上一步的批行数（gauge）
    pub fn batch_rows(&self) -> usize {
        self.batch_rows.load(Ordering::Relaxed)
    }
    pub fn prefix_hit_tokens(&self) -> u64 {
        self.prefix_hit_tokens.load(Ordering::Relaxed)
    }
    pub fn ctx_limit(&self) -> usize {
        self.ctx_limit.load(Ordering::Relaxed)
    }
    pub fn set_ctx_limit(&self, n: usize) {
        self.ctx_limit.store(n, Ordering::Relaxed);
    }
    pub fn mem(&self) -> MemStats {
        self.mem.lock().expect("mem stats 锁中毒").clone()
    }
}

/// 执行器请求级失败：**可恢复**——只终结该请求，引擎任务与同批其他请求不受影响。
///
/// 这是"exec 侧不再 panic"的通道：执行器在 `step`/`admitted` 内部发现页池耗尽、
/// 行/位置上限、vocab 越界时，把该请求记进失败表（`Executor::take_failures`），
/// 引擎在步间 drain 后按类别终结它。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepError {
    /// 容量类（页池耗尽 / 块表行耗尽 / 绝对位置超上限）= "这段装不下/到头了"
    /// → 该请求以 `FinishReason::Length` 结束（serve: finish_reason "length"）
    Capacity(String),
    /// 请求本身非法或执行器真异常（vocab 越界等）
    /// → 该请求以 `FinishReason::Error` 结束（serve: finish_reason "error"）
    Invalid(String),
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StepError::Capacity(m) => write!(f, "容量失败：{m}"),
            StepError::Invalid(m) => write!(f, "请求非法：{m}"),
        }
    }
}

/// 执行器：engine core 只做决策，一步的"算"由执行器完成（§5 exec 线程的 trait 预留）。
pub trait Executor: Send {
    /// 执行一步，返回每个 decode 槽位产出的 token
    fn step(&mut self, plan: &StepPlan) -> Vec<(ReqId, u32)>;
    /// 步末回收：清理已离批请求的执行器内部状态（DeferredRelease 钩子）
    fn retain(&mut self, active: &[ReqId]);
    /// 准入通知：prompt 在此首次可见（执行器建立自己的序列/KV 记账）。
    /// 空指针不算——空 prompt 由执行器自行决定 BOS 语义。
    /// 返回值 = 执行器已就绪的 token 数（前缀缓存命中恢复的部分），
    /// engine 会把它回填进调度债（sched.prefix_hit）——无缓存执行器返回 0
    fn admitted(&mut self, _id: ReqId, _prompt: &[u32]) -> usize { 0 }
    /// 采样参数下发给执行器（提交时一次；执行器可在此做设备侧/静态校验）。
    ///
    /// Err = 该配置**设备侧不可满足**——engine 不把它送进调度器，直接以
    /// `Finished(Aborted)` 终结（等价 serve 侧的 400：客户端拿到明确终结事件，
    /// 不会跑出一个语义错误的序列）。当前唯一的拒绝项是"全词表采样"
    /// （`top_k = 0`，见 `SampleParams`）：设备端采样器只有 top-k（k ≤ 64）
    /// 路径，248K 项降序全排序不可接受。
    ///
    /// 默认实现吞掉参数（脚本件/替身无采样概念，既有 parity 不变）
    fn set_sampling(&mut self, _id: ReqId, _p: SampleParams) -> Result<(), String> {
        Ok(())
    }
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
    /// 前缀缓存的对齐粒度：调度器把非末块 prefill chunk 尾圆到此粒度，
    /// 使 chunk 尾落在执行器的块边界上（= 可共享/可恢复的状态检查点）。
    /// 默认 1（不对齐）——无前缀缓存的执行器账与旧版逐位一致
    fn prefix_align(&self) -> usize {
        1
    }
    /// 上下文窗口（token 数）：prompt + max_tokens 的上限。serve 用它做前置
    /// 校验（越界 → 400），引擎在提交处再校验一次（兜底）。默认无上限
    fn ctx_limit(&self) -> usize {
        usize::MAX
    }
    /// 取走自上次调用以来累积的**请求级失败**（见 [`StepError`]）。
    ///
    /// 执行器一旦遇到页池耗尽 / 行耗尽 / 位置越界 / vocab 越界，记进本表而
    /// **不是 panic**：panic 会让 engine 任务整体消失（在飞请求全部 Aborted、
    /// 之后 submit 全失败 = 服务假死）。引擎在每步之后 drain 并按类别终结
    /// 对应请求，同批其他请求照常推进。
    fn take_failures(&mut self) -> Vec<(ReqId, StepError)> {
        Vec::new()
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
        // 执行器上报的上下文窗口进 stats（serve 经 handle.ctx_limit() 读，
        // 引擎在提交处也用它兜底）——须在执行器 move 进任务之前取
        stats.set_ctx_limit(exec.ctx_limit());
        let handle = EngineHandle::new(cmd_tx).with_stats(stats.clone());
        tokio::spawn(Self::run(cfg, cmd_rx, stats.clone(), exec));
        (handle, stats)
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
        // 前缀缓存对齐：执行器（真模型 radix 开时 = KV 块大小）→ 调度 chunk 尾
        cfg.sched.snap_align = cfg.sched.snap_align.max(exec.prefix_align());
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
                // 两段式：准入相位 → 回放事件（建行 + 前缀命中对账 + 完成
                // 回收）→ 规划相位。命中必须在规划之前进账：晚一步本步预算
                // 会按未命中的债切 chunk，执行器按真实游标少喂 → 两侧漂移
                sched.advance();
                // 事件先于 exec.step：本步新准入者先建行（exec.admitted）
                // 再进前向——否则 step 查不到行会静默空转
                Self::pump(&mut sched, &mut senders, &mut stops, &mut prompts, &stats, &mut exec);
                let plan = sched.plan();
                stats.steps.fetch_add(1, Ordering::Relaxed);
                // gauge：本步前向行数（prefill token 数 + decode 行数）
                stats.batch_rows.store(
                    plan.prefill.iter().map(|&(_, n)| n as usize).sum::<usize>()
                        + plan.decode.len(),
                    Ordering::Relaxed,
                );
                if cfg.step_delay.is_zero() {
                    // 让点：本循环无任何 await（drain 用 try_recv、send 是非阻塞），
                    // 会把所在 worker 线程占满整代——同 worker 上的 SSE 投递任务被饿死，
                    // 客户端在生成结束前一个 token 都收不到（单请求尤其明显）。
                    tokio::task::yield_now().await;
                } else {
                    sleep(cfg.step_delay).await;
                }
                // 同一步内同一请求可发多个 token（投机解码：接受的草稿 + 奖励）。
                // 调度器 plan() 每步只记 1 个；第 2 个起在此按真实发射补账
                // （sched.on_emitted），额度耗尽/已终止则丢弃该 token——请求
                // 随即以 Length 完成（执行器内部序列可能多推进，但已终结）。
                let mut emitted: HashMap<ReqId, u32> = HashMap::new();
                for (id, tok) in exec.step(&plan) {
                    let n = emitted.entry(id).or_insert(0);
                    if *n > 0 && (!senders.contains_key(&id) || !sched.on_emitted(id)) {
                        continue; // 已停止（停止词/取消）或超额：丢弃不外吐
                    }
                    *n += 1;
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
                // 执行器请求级失败（页池耗尽/vocab 越界/位置越界…）：**不 panic**，
                // 只终结该请求。放在本步 token 发射之后：同批其他请求的 token
                // 已经出去了，不受影响
                Self::settle_failures(&mut sched, &mut senders, &mut stops, &mut prompts, &stats, &mut exec);
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
                // 采样参数（须在 stop_token_ids 被 move 之前取）
                let sp = sampling.sample_params();
                if !sampling.stop_token_ids.is_empty() {
                    stops.insert(id, sampling.stop_token_ids);
                }
                if let Some(tx) = senders.get(&id) {
                    let _ = tx.send(EngineEvent::Accepted);
                }
                // 上下文窗口兜底（serve 已前置校验 → 400；这里防直连引擎的
                // 调用方绕过校验）：prompt 本身已吃满窗口 → 一个位置都没有，
                // 直接以 Error 终结。**只查 prompt**：prompt+max_tokens 越界
                // 是容量问题不是协议问题——让它进调度器，跑到窗口尽头由执行器
                // 报 Capacity → finish_reason=length（客户端能拿到部分结果）
                let ctx = stats.ctx_limit();
                if ctx != usize::MAX && prompt_token_ids.len() >= ctx {
                    tracing::warn!(
                        request = id,
                        prompt_tokens = prompt_token_ids.len(),
                        ctx_limit = ctx,
                        "prompt 已吃满上下文窗口，直接以 Error 终结"
                    );
                    stats.cancelled.fetch_add(1, Ordering::Relaxed);
                    if let Some(tx) = senders.remove(&id) {
                        let _ = tx.send(EngineEvent::Finished(FinishReason::Error));
                    }
                    stops.remove(&id);
                    return;
                }
                // 采样参数先过执行器校验（设备侧能力边界：top_k ≤ 64 的
                // top-k/top-p；全词表 top_k=0 不可满足）。不可满足 → 不进
                // 调度器、不建行，直接终结（serve 侧表现为 Aborted/400）：
                // 让它跑起来再中途出错会把错序列吐给客户端
                if let Err(e) = exec.set_sampling(id, sp) {
                    tracing::warn!(request = id, "采样参数不可满足，请求以 Error 终结：{e}");
                    sched.cancel(id);
                    stats.cancelled.fetch_add(1, Ordering::Relaxed);
                    if let Some(tx) = senders.remove(&id) {
                        let _ = tx.send(EngineEvent::Finished(FinishReason::Error));
                    }
                    stops.remove(&id);
                    return;
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
        Self::terminate(sched, senders, stops, prompts, stats, id, FinishReason::Aborted);
    }

    /// 以指定原因终结一个请求（取消 / 执行器失败共用一支路径）。
    /// 幂等：终结过一次后再调无副作用（sender 已除名则不再发事件、不再计数）
    fn terminate(
        sched: &mut Scheduler,
        senders: &mut HashMap<RequestId, UnboundedSender<EngineEvent>>,
        stops: &mut HashMap<RequestId, Vec<u32>>,
        prompts: &mut HashMap<RequestId, Vec<u32>>,
        stats: &EngineStats,
        id: RequestId,
        reason: FinishReason,
    ) {
        stops.remove(&id);
        prompts.remove(&id);
        let known = sched.cancel(id);
        if let Some(tx) = senders.remove(&id) {
            if known {
                stats.cancelled.fetch_add(1, Ordering::Relaxed);
            }
            let _ = tx.send(EngineEvent::Finished(reason));
        }
    }

    /// 执行器请求级失败 → 按类别终结（容量类 = Length，其余 = Error）。
    /// 这是"exec 不再 panic"的落地处：只有失败者离场，引擎任务与同批其他
    /// 请求继续跑（旧实现里一次 assert 就带走整个引擎任务 = 服务假死）
    fn settle_failures(
        sched: &mut Scheduler,
        senders: &mut HashMap<RequestId, UnboundedSender<EngineEvent>>,
        stops: &mut HashMap<RequestId, Vec<u32>>,
        prompts: &mut HashMap<RequestId, Vec<u32>>,
        stats: &EngineStats,
        exec: &mut Box<dyn Executor>,
    ) {
        for (id, err) in exec.take_failures() {
            let reason = match &err {
                StepError::Capacity(m) => {
                    tracing::warn!(request = id, "执行器容量不足 → 该请求以 length 结束：{m}");
                    FinishReason::Length
                }
                StepError::Invalid(m) => {
                    tracing::warn!(request = id, "执行器请求级失败 → 该请求以 error 结束：{m}");
                    FinishReason::Error
                }
            };
            Self::terminate(sched, senders, stops, prompts, stats, id, reason);
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
                    let hit = exec.admitted(id, &p);
                    if hit > 0 {
                        // 前缀缓存命中对账：exec 已恢复的状态让调度债等量消减
                        // （预算切片/mix_cap/追平判定按剩余段计算）
                        sched.prefix_hit(id, hit);
                        // 观测口（/metrics 的 wein_prefix_hit_tokens_total）
                        stats.prefix_hit_tokens.fetch_add(hit as u64, Ordering::Relaxed);
                    }
                }
            }
            // 建行失败（vocab 越界 / 行耗尽）→ 该请求以 Error/Length 结束，
            // 同批其他请求不受影响（绝不 panic）
            Self::settle_failures(sched, senders, stops, prompts, stats, exec);
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
