//! Qwen35Executor：I1 竖切真身——Qwen3.5-4B 文本骨干全 32 层上设备。
//!
//! 混合骨干（transformers 5.21 `modeling_qwen3_5.py` 对齐）：
//! - 24 × linear_attention：in_proj_qkv → 深度wise conv1d(k=4)+silu（跨步
//!   接 conv 状态）→ q/k/v 拆分 → in_proj_z/a/b → gated deltanet 递推
//!   （逐请求状态池）→ RMSNormGated（F32 权重）→ out_proj
//! - 8 × full_attention：q_proj（q‖gate 按头交错）→ 零中心 q/k norm →
//!   partial rope（rot_dim 64，表收 bf16 再回展）→ 分页 KV → GQA paged
//!   attn → sigmoid 门乘 → o_proj
//! - 共通：零中心 RMSNorm / SwiGLU MLP / 残差；lm_head 与 embed 同张量
//!   （tie_word_embeddings，gemm_wt 直用）
//!
//! 状态分账：块表行号同时索引 linear 状态行 [32,128,128] f32、conv 状态行
//! [8192,3] bf16、KV 块表 [bpr]——行回收（retain/preempted）时线性/卷积
//! 状态区间零化（KV 由重算覆盖写，幂等）。
//!
//! 调度合同与 ToyExecutor 同钉（x6/x7 验收）：追平段尾即采样位；批内同
//! 请求行连续（conv/deltanet 的段语义前提——forward_batch 构造处有断言）。
//!
//! 图档位（§7，qg 系验收）：`open_graph` = FULL_DECODE——纯 decode 步的
//! span/gather/采样行表是桶形的纯函数（固化上传一次），数据五缓冲
//! （ids/pos/seqs/rid/bt）常驻每步覆写；argmax 烤进图尾。sink 行（块表
//! 末行 + 专属页 + 专属状态行）吸收桶尾 padding。`open` = NONE（q 系
//! 保持 eager）
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

/// 对拍钩子的 run_forward 调用计数（s1=预填批，s2..=解码步）
static TRACE_STEP: AtomicU32 = AtomicU32::new(0);

use wein_engine::Executor;
use wein_kernel::cuda::driver::CuBatchOp;
use wein_kernel::cuda::{DevBuf, Event, GraphExec, Stream};
use wein_kernel::ops::{
    AddOp, ArgmaxRowsOp, Conv1dDwOp, ConvStateRollOp, DeltanetOp, EmbedOp, GemmOp, KvScatterOp,
    MulOp, PagedAttnOp, RmsNormGatedOp, RmsNormZcOp, RopePtOp, SigmulOp, SiluOp,
};
use wein_kernel::reference;
use wein_memory::pool::BlockPool;
use wein_memory::radix::RadixCache;
use wein_model::{LayerType, Qwen35Config, WeightStore};
use wein_sched::{ReqId, StepPlan};

use crate::device::{pin, DeviceSession};

/// 批前向 scratch 行数上限（对齐调度 token_budget=512）
const T_MAX: usize = 512;
/// 块表/状态行数（串行并发上限；serve 接线时 max_running 须 ≤ 此值）
const MAX_ROWS: usize = 128;
/// 序列长上限（paged_attn M0 单线程两遍扫的栈上分母数组；W2 flash 移除）
const MAX_POS: usize = 256;
/// KV 页池页数（全注意力层共用一个池容量）
const NUM_PAGES: u32 = 512;
const BLOCK: u32 = 16;
/// radix 前缀缓存快照槽数（每槽 24 层 lin 1MB + conv 48KB ≈ 25.2MB；
/// 从约束于 KV 页池——512 页 ÷ 每前缀 ~3 页 ≈ 百级条目，64 槽同量级）。
/// 槽挂 radix 块（snap_of_block）；请求活跃期逐边界各借一槽（前缀树上每块
/// 都要有槽才可能被命中——单活跃槽只保末边界时祖先无槽、命中恒 0）
const SNAP_SLOTS: usize = 64;
/// linear 侧 q/k l2norm eps（FLA 对齐，modeling l2norm）
const L2_EPS: f32 = 1e-6;

/// full_attention 层权重
struct FullDev {
    /// [8192, 2560]：每头 [q(256) ‖ gate(256)] 交错——HF 在 view(T,-1,512)
    /// 上 chunk(2, dim=-1)，拆分用行 gather（宽 256）位对齐
    qg: DevBuf<'static>,
    k: DevBuf<'static>,  // [1024, 2560]
    v: DevBuf<'static>,  // [1024, 2560]
    o: DevBuf<'static>,  // [2560, 4096]
    qn: DevBuf<'static>, // [256] 零中心 norm 权重
    kn: DevBuf<'static>, // [256]
}

/// linear_attention 层权重
struct LinDev {
    qkv: DevBuf<'static>,   // [8192, 2560]
    z: DevBuf<'static>,     // [4096, 2560]
    a: DevBuf<'static>,     // [32, 2560]
    b: DevBuf<'static>,     // [32, 2560]
    conv: DevBuf<'static>,  // [8192, 4]（checkpoint [8192,1,4] 直接平铺同布局）
    a_log: DevBuf<'static>, // [32] F32（checkpoint 原生）
    dtb: DevBuf<'static>,   // [32] bf16
    norm: DevBuf<'static>,  // [128] F32（RMSNormGated 权重）
    out: DevBuf<'static>,   // [2560, 4096]
}

enum AttnDev {
    Full(FullDev),
    Linear(LinDev),
}

struct LayerDev {
    ln1: DevBuf<'static>,
    attn: AttnDev,
    ln2: DevBuf<'static>,
    wg: DevBuf<'static>,
    wu: DevBuf<'static>,
    wd: DevBuf<'static>,
}

/// 前向 scratch（按 T_MAX 一次分配；下标/块表等小数组每步上传）
struct ScrDev {
    x: DevBuf<'static>,      // [T,2560] 主路（embed → 层间传递 → final norm）
    n1: DevBuf<'static>,     // [T,2560]
    qg: DevBuf<'static>,     // [T,8192] full q‖gate
    qh: DevBuf<'static>,     // [T,16,256] full q
    gate: DevBuf<'static>,   // [T,16,256] full 门
    kh: DevBuf<'static>,     // [T,4,256] full k（norm 前）
    vh: DevBuf<'static>,     // [T,4,256] full v
    attn: DevBuf<'static>,   // [T,16,256]
    agated: DevBuf<'static>, // [T,4096]
    qkv: DevBuf<'static>,    // [T,8192] in_proj_qkv
    qkvc: DevBuf<'static>,   // [T,8192] conv+silu 后
    qs: DevBuf<'static>,     // [T,16,128] linear q
    ks: DevBuf<'static>,     // [T,16,128] linear k
    vs: DevBuf<'static>,     // [T,32,128] linear v
    z: DevBuf<'static>,      // [T,4096]
    aa: DevBuf<'static>,     // [T,32]
    bb: DevBuf<'static>,     // [T,32]
    core: DevBuf<'static>,   // [T,4096] deltanet 输出
    normed: DevBuf<'static>, // [T,4096]
    o: DevBuf<'static>,      // [T,2560] attn/out_proj 输出
    h2: DevBuf<'static>,     // [T,2560] 残差中继
    n2: DevBuf<'static>,     // [T,2560]
    g: DevBuf<'static>,      // [T,9216]
    u: DevBuf<'static>,      // [T,9216]
    act: DevBuf<'static>,    // [T,9216]
    m: DevBuf<'static>,      // [T,9216]
    d: DevBuf<'static>,      // [T,2560]
    logits: DevBuf<'static>, // [T, vocab]
    am_scratch: DevBuf<'static>, // [T,512] u32
    am_out: DevBuf<'static>, // [T] u32
}

struct ReqState {
    seq: Vec<u32>,
    forwarded: usize,
    row: u32,
    blocks: Vec<u32>,
    /// blocks[..shared] 是 radix 命中来的共享块（重放/驱逐语义分叉点）
    shared: usize,
    /// 活跃期落过的边界快照 (块, 槽)——每个栅格边界一份，retain 时逐块
    /// 移交（块进树则槽随块，否则槽回池）；抢占重放则全数作废
    snaps: Vec<(u32, u32)>,
}

/// FULL_DECODE 图档（§7；机制由 w 系在 toy 钉死，此处为 32 层真模型装配）。
/// 数据五缓冲每步覆写；固化输入（span/gather/采样行表）是纯 decode 桶形
/// 的纯函数，上传一次永不再动——各桶图录共享缓冲的 narrow view。
/// 声明序先于 session：'static 钉扎（borrow session.stream 的短命借用提为
/// 'static）与 stream/table 同一论证
struct QGraphMode {
    sink_row: u32,
    b_max: u32,
    buckets: Vec<u32>,
    // —— 每步覆写（设备指针烤进图节点，数据 live）——
    in_ids: DevBuf<'static>,
    in_pos: DevBuf<'static>,
    in_seqs: DevBuf<'static>,
    /// 纯 decode 下 row_id 与 tok_row 同内容（每行一段，段主即行主）
    in_rid: DevBuf<'static>,
    /// 全块表常驻镜像（页保证先于上传）
    in_bt: DevBuf<'static>,
    // —— 固化输入（桶形纯函数；图节点指向这些缓冲，必须与 execs 同寿）——
    fx_start: DevBuf<'static>,   // [0..b_max)
    fx_count: DevBuf<'static>,   // [1; b_max]
    fx_tpos: DevBuf<'static>,    // [0; b_max]（段内局部位恒 0）
    fx_rows: DevBuf<'static>,    // argmax 采样行 [0..b_max)
    fx_iq: DevBuf<'static>,
    fx_ig: DevBuf<'static>,
    fx_ilq: DevBuf<'static>,
    fx_ilk: DevBuf<'static>,
    fx_ilv: DevBuf<'static>,
    execs: Vec<GraphExec<'static>>,
    /// 图路径前向计数（qg 系验收：证路径确实被走到）
    replays: u64,
}

/// 前向输入束——eager 临时缓冲与图档常驻缓冲同构喂法（run_forward 共用）
struct FwdIn<'a> {
    ids: &'a DevBuf<'a>,
    pos: &'a DevBuf<'a>,
    seqs: &'a DevBuf<'a>,
    bt: &'a DevBuf<'a>,
    start: &'a DevBuf<'a>,
    count: &'a DevBuf<'a>,
    rid: &'a DevBuf<'a>,
    trow: &'a DevBuf<'a>,
    tpos: &'a DevBuf<'a>,
    iq: &'a DevBuf<'a>,
    ig: &'a DevBuf<'a>,
    ilq: &'a DevBuf<'a>,
    ilk: &'a DevBuf<'a>,
    ilv: &'a DevBuf<'a>,
}

// 独占设备句柄，exec 单线程驱动（§5）——同 DeviceSession 论证
unsafe impl Send for Qwen35Executor {}

// ── 步时分解插桩（WEIN_STEP_PROFILE；满负荷账定位——BASELINE §4 的段级形态）──
// 粗段边界打点：WSL 事件时间轴 ~20µs 量化（k9 实测同点两事件可差 0.02ms），
// 逐 op 打点会把 phantom 时间灌进每段；按算子类切段（段 ms 级）足够定位
// 大项。事件池 open 期建好反复 record（k9 合同）；同类相邻段合并（一层内
// gemm 连发只落一次事件）。仅 eager 激活——图捕获区间禁额外事件。
//
// 打点协议：prof_mark(sect) 在 sect 段**首个 launch 之前**调用——落下的
// 事件把上一段关在其末 kernel 之后（流保序），段时距 = 事件对时差。

/// 事件池容量（单步边界数上界：~440 段/步 @ 32 层同类合并后——超则截断标注）
const PROF_POOL: usize = 1024;

struct StepProf {
    ev: Vec<Event<'static>>,
    /// 本步已录事件数（含 ev[0] 步首）
    idx: usize,
    /// 当前段名（正在跑的算子类；None = 步首未入段）
    cur: Option<&'static str>,
    /// (段名, 段尾事件 idx)——段首 = 上一段尾（首段 = ev[0]）
    ends: Vec<(&'static str, usize)>,
    totals: HashMap<&'static str, f64>,
    hits: HashMap<&'static str, u64>,
    steps: u64,
    wall_ms: f64,
    /// 11 次 alloc+H2D 的 host 账（每步）
    upload_ms: f64,
    /// run_forward 提交（launch 流）的 host 账
    submit_ms: f64,
    /// sync 后采样 D2H 读回的 host 账
    readback_ms: f64,
    /// 单事件记录的 GPU 时间轴成本（步尾校准突发的跨步均值的当前值）
    cal_ms: f64,
    /// 本步校准突发的 (首, 末) 事件 idx（finalize 时查时距）
    cal_pair: Option<(usize, usize)>,
    truncated: bool,
}

fn prof_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

impl StepProf {
    fn new(gpu: &wein_kernel::cuda::Gpu) -> Self {
        StepProf {
            ev: (0..PROF_POOL).map(|_| pin!(gpu.event())).collect(),
            idx: 0,
            cur: None,
            ends: Vec::new(),
            totals: HashMap::new(),
            hits: HashMap::new(),
            steps: 0,
            wall_ms: 0.0,
            upload_ms: 0.0,
            submit_ms: 0.0,
            readback_ms: 0.0,
            cal_ms: 0.0,
            cal_pair: None,
            truncated: false,
        }
    }

    /// 步首：落 ev[0]（uploads 之后——H2D 同步拷贝不进 GPU 段账）
    fn begin(&mut self, stream: &Stream) {
        self.idx = 0;
        self.ends.clear();
        self.cur = None;
        self.ev[0].record_on(stream);
    }

    /// 进入 sect 段（段首个 launch 前调用）
    fn mark(&mut self, sect: &'static str, stream: &Stream) {
        if self.cur == Some(sect) {
            return; // 同类相邻段合并
        }
        if let Some(prev) = self.cur.replace(sect) {
            if self.record(stream) {
                self.ends.push((prev, self.idx));
            }
        }
    }

    /// 步尾：关掉最后一段（最后 launch 之后、sync 之前）
    fn end(&mut self, stream: &Stream) {
        if let Some(prev) = self.cur.take() {
            if self.record(stream) {
                self.ends.push((prev, self.idx));
            }
        }
    }

    fn record(&mut self, stream: &Stream) -> bool {
        if self.idx + 1 >= PROF_POOL {
            self.truncated = true;
            return false;
        }
        self.idx += 1;
        self.ev[self.idx].record_on(stream);
        true
    }

    /// sync 后结算：段时距累计（sync 前调用 end 已把最后事件落定）。
    /// 事件记录本身在 GPU 时间轴上有成本（WSL 实测 ~85µs/次，m=4 步
    /// 440 边界 +37ms）——步尾同点校准突发量出单价，报告按段内边界数扣除
    fn finalize(&mut self, wall: f64) {
        let mut prev = 0usize;
        for &(sect, e) in &self.ends {
            *self.totals.entry(sect).or_insert(0.0) +=
                self.ev[e].elapsed_ms(&self.ev[prev]) as f64;
            *self.hits.entry(sect).or_insert(0) += 1;
            prev = e;
        }
        self.steps += 1;
        self.wall_ms += wall;
        if let Some((a, b)) = self.cal_pair.take() {
            let per = self.ev[b].elapsed_ms(&self.ev[a]) as f64 / (b - a) as f64;
            self.cal_ms = (self.cal_ms * (self.steps - 1) as f64 + per) / self.steps as f64;
        }
    }

    /// 步尾校准突发：连录 K 个无 kernel 事件——相邻时距 = 单记录成本。
    /// 只落记录；时距查询在 finalize（sync 后——事件完成才可查）
    fn calibrate(&mut self, stream: &Stream) {
        const K: usize = 32;
        if self.idx + K >= PROF_POOL {
            return;
        }
        let a = self.idx + 1;
        for _ in 0..K {
            self.idx += 1;
            self.ev[self.idx].record_on(stream);
        }
        self.cal_pair = Some((a, self.idx));
    }

    fn report(&self) -> String {
        let steps = self.steps.max(1) as f64;
        let mut rows: Vec<_> = self.totals.iter().collect();
        rows.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        // 校正段账：raw − 边界数 × 单记录成本（事件落在段尾——段 i 的
        // 边界 = hits[i]，含步首步尾摊销误差，量级 ≈ 单记录成本）
        let adj: HashMap<&str, f64> = rows
            .iter()
            .map(|(&s, &ms)| {
                let hits = self.hits.get(s).copied().unwrap_or(0) as f64 / steps;
                (s, (ms / steps - hits * self.cal_ms).max(0.0))
            })
            .collect();
        let adj_sum: f64 = adj.values().sum();
        let gpu_sum: f64 = self.totals.values().sum();
        let mut s = format!(
            "# step profile（{} 步累计）\n\nwall {:.2} ms/步｜GPU 段和 raw {:.2} → 校正 {:.2}（事件记录 {:.0}µs/次 × {:.0} 打点/步）｜host：上传 {:.2} + 提交 {:.2}（与 GPU 重叠）+ 读回 {:.2}{}\n\n| 段 | 校正 ms/步 | 占比% | raw ms/步 | 打点/步 |\n|---|---|---|---|---|",
            self.steps,
            self.wall_ms / steps,
            gpu_sum / steps,
            adj_sum,
            self.cal_ms * 1e3,
            self.hits.values().sum::<u64>() as f64 / steps,
            self.upload_ms / steps,
            self.submit_ms / steps,
            self.readback_ms / steps,
            if self.truncated { "｜⚠ 事件池截断" } else { "" },
        );
        for (sect, ms) in rows {
            let a = adj.get(sect).copied().unwrap_or(0.0);
            let pct = if adj_sum > 0.0 { a / adj_sum * 100.0 } else { 0.0 };
            let h = self.hits.get(sect).copied().unwrap_or(0) as f64 / steps;
            s.push_str(&format!(
                "\n| {sect} | {a:.2} | {pct:.1} | {:.2} | {h:.0} |",
                *ms / steps
            ));
        }
        s
    }
}

pub struct Qwen35Executor {
    layers: Vec<LayerDev>,
    embed: DevBuf<'static>,
    norm_f: DevBuf<'static>,
    /// 24 个线性层状态池 [MAX_ROWS, 32,128,128] f32（层序）
    lin_state: Vec<DevBuf<'static>>,
    /// 24 个线性层 conv 状态池 [MAX_ROWS, 8192,3] bf16
    conv_state: Vec<DevBuf<'static>>,
    /// 8 个全注意力层 KV 池（k/v 成对，层序）
    k_pool: Vec<DevBuf<'static>>,
    v_pool: Vec<DevBuf<'static>>,
    cos: DevBuf<'static>,
    sin: DevBuf<'static>,
    scr: ScrDev,
    cfg: Qwen35Config,
    pool: BlockPool,
    /// 前缀缓存索引（token 块 → KV 页；与 pool 联动 pin/rc）
    radix: RadixCache,
    /// 快照池：24 层 lin + conv 的槽化副本（radix 命中恢复用——位拷贝无重算）
    snap_lin: Vec<DevBuf<'static>>,
    snap_conv: Vec<DevBuf<'static>>,
    snap_free: Vec<u32>,
    /// 块 → 槽（快照挂块：命中截断到有槽祖先；驱逐联动回收）
    snap_of_block: HashMap<u32, u32>,
    /// 前缀缓存总开关（WEIN_RADIX=0 关——A/B 对比用）
    radix_on: bool,
    /// 命中/移交诊断打印（WEIN_RADIX_DEBUG=1）
    radix_dbg: bool,
    /// decode 侧快照栅格（0 = 不落）。decode 边界快照只在"共享同一续写"
    /// 场景有用（多轮对话的下一轮 prompt 含前轮生成 token），故默认稀疏
    snap_grid_dec: usize,
    reqs: HashMap<ReqId, ReqState>,
    bt_host: Vec<u32>,
    next_row: u32,
    free_rows: Vec<u32>,
    /// 可分配块表行上限（图档让出末行作 sink——NONE 时 = MAX_ROWS）
    row_capacity: u32,
    // 派生常量
    lin_row: usize, // 线性层状态单行字节数
    conv_row: usize, // conv 状态单行字节数
    bpr: usize,
    q_scale: f32,
    /// FULL_DECODE 图档（None = NONE——eager 直发）
    graph: Option<QGraphMode>,
    /// 步时分解插桩（WEIN_STEP_PROFILE；仅 eager——声明序先于 session，
    /// 事件池借用 gpu 与 DevBuf 同一钉扎论证）
    prof: RefCell<Option<StepProf>>,
    session: DeviceSession,
}

impl Qwen35Executor {
    /// gemm autotune 的 m 集（= 图桶默认集；图档开启时被实际桶集替换）：
    /// ramp/drain 的瞬态 m 不调优只钉 heuristic（零调优债），稳态 decode
    /// 形状全在桶集内（battery 实测无门控的债：满负荷 tps 2827→1678）。
    /// `WEIN_TUNE_M=1,2,…` 可覆盖（空串 = 全 heuristic 钉，A/B 用）
    const TUNE_M: [u32; 9] = [1, 2, 4, 8, 16, 32, 64, 100, 128];

    pub fn open(dir: &Path) -> Result<Self, String> {
        let tune = match std::env::var("WEIN_TUNE_M") {
            Ok(s) => s
                .split(',')
                .filter_map(|t| t.trim().parse::<u32>().ok())
                .collect::<Vec<_>>(),
            _ => Self::TUNE_M.to_vec(),
        };
        Self::open_with_tune(dir, &tune)
    }

    /// gemm 调优审计（布局/选型 A/B 的判别仪器：每形状 chosen/best_ms）
    pub fn gemm_tune_report(&self) -> Vec<wein_kernel::cuda::GemmTuneRecord> {
        self.session.table().gemm.tune_report()
    }

    fn open_with_tune(dir: &Path, tune_m: &[u32]) -> Result<Self, String> {
        let session = DeviceSession::open_tuned(tune_m)?;
        let gpu = session.gpu();
        let cfg = Qwen35Config::load(dir).map_err(|e| format!("config：{e}"))?;
        cfg.validate().map_err(|e| format!("config：{e}"))?;
        let store = WeightStore::open(dir).map_err(|e| format!("权重：{e}"))?;
        let n_layers = cfg.num_layers as usize;

        // ---- 权重上载（mmap 切片直传；文本骨干前缀 model.language_model.，
        // visual/mtp 不在文本路径——738 张量里只取 32 层文本集）----
        const P: &str = "model.language_model.";
        let up = |name: &str| -> Result<DevBuf<'static>, String> {
            let t = store.tensor(name).map_err(|e| format!("{e}"))?;
            let bits = t.bf16().map_err(|e| format!("{name}：{e}"))?;
            Ok(pin!(gpu.alloc_from_slice(bits)))
        };
        let up_f32 = |name: &str| -> Result<DevBuf<'static>, String> {
            let t = store.tensor(name).map_err(|e| format!("{e}"))?;
            let v = t.f32().map_err(|e| format!("{name}：{e}"))?;
            Ok(pin!(gpu.alloc_from_slice(v)))
        };
        let mut layers = Vec::with_capacity(n_layers);
        let mut lin_state = Vec::new();
        let mut conv_state = Vec::new();
        let mut k_pool = Vec::new();
        let mut v_pool = Vec::new();
        for i in 0..n_layers {
            let base = format!("{P}layers.{i}.");
            let ln1 = up(&format!("{base}input_layernorm.weight"))?;
            let ln2 = up(&format!("{base}post_attention_layernorm.weight"))?;
            let wg = up(&format!("{base}mlp.gate_proj.weight"))?;
            let wu = up(&format!("{base}mlp.up_proj.weight"))?;
            let wd = up(&format!("{base}mlp.down_proj.weight"))?;
            let attn = match cfg.layer_type(i as u32) {
                LayerType::Full => {
                    let bytes = NUM_PAGES as usize * BLOCK as usize
                        * (cfg.num_kv_heads * cfg.head_dim) as usize
                        * 2;
                    k_pool.push(pin!(gpu.alloc_bytes(bytes)));
                    v_pool.push(pin!(gpu.alloc_bytes(bytes)));
                    AttnDev::Full(FullDev {
                        qg: up(&format!("{base}self_attn.q_proj.weight"))?,
                        k: up(&format!("{base}self_attn.k_proj.weight"))?,
                        v: up(&format!("{base}self_attn.v_proj.weight"))?,
                        o: up(&format!("{base}self_attn.o_proj.weight"))?,
                        qn: up(&format!("{base}self_attn.q_norm.weight"))?,
                        kn: up(&format!("{base}self_attn.k_norm.weight"))?,
                    })
                }
                LayerType::Linear => {
                    // 状态池开局全零（流序清零；回收零化走 zero_range_on）
                    let s: DevBuf<'static> =
                        pin!(gpu.alloc_bytes(lin_row_bytes(&cfg) * MAX_ROWS));
                    s.zero_on(session.stream());
                    lin_state.push(s);
                    let c: DevBuf<'static> =
                        pin!(gpu.alloc_bytes(conv_row_bytes(&cfg) * MAX_ROWS));
                    c.zero_on(session.stream());
                    conv_state.push(c);
                    AttnDev::Linear(LinDev {
                        qkv: up(&format!("{base}linear_attn.in_proj_qkv.weight"))?,
                        z: up(&format!("{base}linear_attn.in_proj_z.weight"))?,
                        a: up(&format!("{base}linear_attn.in_proj_a.weight"))?,
                        b: up(&format!("{base}linear_attn.in_proj_b.weight"))?,
                        conv: up(&format!("{base}linear_attn.conv1d.weight"))?,
                        a_log: up_f32(&format!("{base}linear_attn.A_log"))?,
                        dtb: up(&format!("{base}linear_attn.dt_bias"))?,
                        norm: up_f32(&format!("{base}linear_attn.norm.weight"))?,
                        out: up(&format!("{base}linear_attn.out_proj.weight"))?,
                    })
                }
            };
            layers.push(LayerDev { ln1, attn, ln2, wg, wu, wd });
        }
        let embed = up(&format!("{P}embed_tokens.weight"))?;
        let norm_f = up(&format!("{P}norm.weight"))?;

        // ---- rope 表：HF 收 bf16 再用（.to(x.dtype)）——executor/镜像同表 ----
        let rot_half = (cfg.rot_dim() / 2) as usize;
        let (mut cos, mut sin) = reference::rope_table(MAX_POS, rot_half, cfg.rope_theta);
        for v in cos.iter_mut().chain(sin.iter_mut()) {
            *v = reference::bf16_to_f32(reference::f32_to_bf16(*v));
        }
        let cos = pin!(gpu.alloc_from_slice(&cos));
        let sin = pin!(gpu.alloc_from_slice(&sin));

        // ---- scratch ----
        let (h, ii, voc) =
            (cfg.hidden_size as usize, cfg.inter_size as usize, cfg.vocab_size as usize);
        let bf = |n: usize| n * 2;
        let scr = ScrDev {
            x: pin!(gpu.alloc_bytes(bf(T_MAX * h))),
            n1: pin!(gpu.alloc_bytes(bf(T_MAX * h))),
            qg: pin!(gpu.alloc_bytes(bf(T_MAX * 8192))),
            qh: pin!(gpu.alloc_bytes(bf(T_MAX * 4096))),
            gate: pin!(gpu.alloc_bytes(bf(T_MAX * 4096))),
            kh: pin!(gpu.alloc_bytes(bf(T_MAX * 1024))),
            vh: pin!(gpu.alloc_bytes(bf(T_MAX * 1024))),
            attn: pin!(gpu.alloc_bytes(bf(T_MAX * 4096))),
            agated: pin!(gpu.alloc_bytes(bf(T_MAX * h))),
            qkv: pin!(gpu.alloc_bytes(bf(T_MAX * 8192))),
            qkvc: pin!(gpu.alloc_bytes(bf(T_MAX * 8192))),
            qs: pin!(gpu.alloc_bytes(bf(T_MAX * 2048))),
            ks: pin!(gpu.alloc_bytes(bf(T_MAX * 2048))),
            vs: pin!(gpu.alloc_bytes(bf(T_MAX * 4096))),
            z: pin!(gpu.alloc_bytes(bf(T_MAX * 4096))),
            aa: pin!(gpu.alloc_bytes(bf(T_MAX * 32))),
            bb: pin!(gpu.alloc_bytes(bf(T_MAX * 32))),
            core: pin!(gpu.alloc_bytes(bf(T_MAX * 4096))),
            normed: pin!(gpu.alloc_bytes(bf(T_MAX * 4096))),
            o: pin!(gpu.alloc_bytes(bf(T_MAX * h))),
            h2: pin!(gpu.alloc_bytes(bf(T_MAX * h))),
            n2: pin!(gpu.alloc_bytes(bf(T_MAX * h))),
            g: pin!(gpu.alloc_bytes(bf(T_MAX * ii))),
            u: pin!(gpu.alloc_bytes(bf(T_MAX * ii))),
            act: pin!(gpu.alloc_bytes(bf(T_MAX * ii))),
            m: pin!(gpu.alloc_bytes(bf(T_MAX * ii))),
            d: pin!(gpu.alloc_bytes(bf(T_MAX * h))),
            logits: pin!(gpu.alloc_bytes(bf(T_MAX * voc))),
            am_scratch: pin!(gpu.alloc_bytes(T_MAX * 512 * 4)),
            am_out: pin!(gpu.alloc_bytes(T_MAX * 4)),
        };
        session.stream().sync(); // 状态池开局零化落定
        let (lin_row, conv_row) = (lin_row_bytes(&cfg), conv_row_bytes(&cfg));
        let q_scale = 1.0f32 / (cfg.head_dim as f32).sqrt();
        let prof = if std::env::var("WEIN_STEP_PROFILE").is_ok() {
            Some(StepProf::new(gpu))
        } else {
            None
        };
        // 前缀缓存快照池（WEIN_RADIX=0 关：不分配、三钩子全跳过）。
        // 池深跟状态池实际层数（lin_state=24 linear 层，非 n_layers=32）
        let radix_on = std::env::var("WEIN_RADIX").map_or(true, |v| v != "0");
        let radix_dbg = std::env::var("WEIN_RADIX_DEBUG").is_ok();
        // decode 栅格默认 64（4 块）——快照 25MB/行/份 ≈ 0.23ms，栅格 16 时
        // 满负荷每步 +1.4ms（实测）；decode 侧命中场景远少于 prompt 侧
        let dgrid = std::env::var("WEIN_SNAP_GRID_DEC")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4 * BLOCK as usize);
        let snap_grid_dec = dgrid.div_ceil(BLOCK as usize) * BLOCK as usize;
        let (snap_lin, snap_conv, snap_free) = if radix_on {
            let mut sl = Vec::with_capacity(lin_state.len());
            let mut sc = Vec::with_capacity(conv_state.len());
            for _ in 0..lin_state.len() {
                let l: DevBuf<'static> = pin!(gpu.alloc_bytes(lin_row * SNAP_SLOTS));
                sl.push(l);
            }
            for _ in 0..conv_state.len() {
                let c: DevBuf<'static> = pin!(gpu.alloc_bytes(conv_row * SNAP_SLOTS));
                sc.push(c);
            }
            for s in &sl {
                s.zero_on(session.stream());
            }
            for s in &sc {
                s.zero_on(session.stream());
            }
            (sl, sc, (0..SNAP_SLOTS as u32).rev().collect())
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };

        Ok(Qwen35Executor {
            layers,
            embed,
            norm_f,
            lin_state,
            conv_state,
            k_pool,
            v_pool,
            cos,
            sin,
            scr,
            cfg,
            pool: BlockPool::new(NUM_PAGES, 0.5),
            radix: RadixCache::new(),
            snap_lin,
            snap_conv,
            snap_free,
            snap_of_block: HashMap::new(),
            radix_on,
            radix_dbg,
            snap_grid_dec,
            reqs: HashMap::new(),
            bt_host: vec![0u32; MAX_ROWS * bpr_of()],
            next_row: 0,
            free_rows: Vec::new(),
            row_capacity: MAX_ROWS as u32,
            lin_row,
            conv_row,
            bpr: bpr_of(),
            q_scale,
            graph: None,
            prof: RefCell::new(prof),
            session,
        })
    }

    /// FULL_DECODE 图档开启（NONE 档走 `open`）。桶集合须非空、严格升序、
    /// ≥1 且最大桶 ≤ T_MAX。open 后追加建造：sink 行落位 + 固化输入 +
    /// 各桶「预热→捕获→实例化」——捕获窗口内零隐式同步（§4，gr5 守卫）
    pub fn open_graph(dir: &Path, buckets: Vec<u32>) -> Result<Self, String> {
        if buckets.is_empty() {
            return Err("图档桶集合不能为空（NONE 档用 open）".into());
        }
        if buckets.windows(2).any(|w| w[0] >= w[1]) {
            return Err(format!("桶集合须严格升序：{buckets:?}"));
        }
        if buckets[0] < 1 || *buckets.last().unwrap() as usize > T_MAX {
            return Err(format!("桶集越界（须 ≥1 且最大 ≤ {T_MAX}）：{buckets:?}"));
        }
        let mut ex = Self::open_with_tune(dir, &buckets)?;
        ex.row_capacity = MAX_ROWS as u32 - 1; // sink 让位——须在首个 admit 前定死
        ex.build_graphs(buckets)?;
        Ok(ex)
    }

    fn build_graphs(&mut self, buckets: Vec<u32>) -> Result<(), String> {
        let b_max = *buckets.last().unwrap() as usize;
        let sink_row = (MAX_ROWS - 1) as u32;
        // sink 专属页 + 状态行零化：padding 的 KV/递推垃圾只落自己的页与行
        let sink_page = self.pool.alloc().ok_or("页池耗尽——sink 行需要 1 页")?;
        self.bt_host[sink_row as usize * self.bpr] = sink_page;
        self.zero_row_state(sink_row, self.session.stream());

        let gpu = self.session.gpu();
        let stream = self.session.stream();
        let nqh = self.cfg.num_heads as usize;
        // 每步覆写数据缓冲（b_max 全量——各桶图录 narrow view）
        let in_ids: DevBuf<'static> = pin!(gpu.alloc_bytes(b_max * 4));
        let in_pos: DevBuf<'static> = pin!(gpu.alloc_bytes(b_max * 4));
        let in_seqs: DevBuf<'static> = pin!(gpu.alloc_bytes(b_max * 4));
        let in_rid: DevBuf<'static> = pin!(gpu.alloc_bytes(b_max * 4));
        let in_bt: DevBuf<'static> = pin!(gpu.alloc_bytes(MAX_ROWS * self.bpr * 4));
        // 固化输入：纯 decode 桶形的纯函数
        let start: Vec<u32> = (0..b_max as u32).collect();
        let fx_start: DevBuf<'static> = pin!(gpu.alloc_from_slice(&start));
        let fx_count: DevBuf<'static> = pin!(gpu.alloc_from_slice(&vec![1u32; b_max]));
        let fx_tpos: DevBuf<'static> = pin!(gpu.alloc_from_slice(&vec![0u32; b_max]));
        let fx_rows: DevBuf<'static> = pin!(gpu.alloc_from_slice(&start)); // argmax 采样行
        let ids_q: Vec<u32> =
            (0..b_max).flat_map(|t| (0..nqh).map(move |h| (t * 2 * nqh + 2 * h) as u32)).collect();
        let ids_gate: Vec<u32> =
            (0..b_max).flat_map(|t| (0..nqh).map(move |h| (t * 2 * nqh + 2 * h + 1) as u32)).collect();
        let ids_lq: Vec<u32> = (0..b_max).map(|t| (t * 4) as u32).collect();
        let ids_lk: Vec<u32> = (0..b_max).map(|t| (t * 4 + 1) as u32).collect();
        let ids_lv: Vec<u32> = (0..b_max).map(|t| (t * 2 + 1) as u32).collect();
        let fx_iq: DevBuf<'static> = pin!(gpu.alloc_from_slice(&ids_q));
        let fx_ig: DevBuf<'static> = pin!(gpu.alloc_from_slice(&ids_gate));
        let fx_ilq: DevBuf<'static> = pin!(gpu.alloc_from_slice(&ids_lq));
        let fx_ilk: DevBuf<'static> = pin!(gpu.alloc_from_slice(&ids_lk));
        let fx_ilv: DevBuf<'static> = pin!(gpu.alloc_from_slice(&ids_lv));
        // 捕获期输入全指 sink（tok 0 / pos 0）——真实数据回放前覆写
        in_ids.upload_on(&vec![0u32; b_max], stream);
        in_pos.upload_on(&vec![0u32; b_max], stream);
        in_seqs.upload_on(&vec![sink_row; b_max], stream);
        in_rid.upload_on(&vec![sink_row; b_max], stream);
        in_bt.upload_on(&self.bt_host, stream);
        stream.sync();

        let inp = FwdIn {
            ids: &in_ids,
            pos: &in_pos,
            seqs: &in_seqs,
            bt: &in_bt,
            start: &fx_start,
            count: &fx_count,
            rid: &in_rid,
            trow: &in_rid,
            tpos: &fx_tpos,
            iq: &fx_iq,
            ig: &fx_ig,
            ilq: &fx_ilq,
            ilk: &fx_ilk,
            ilv: &fx_ilv,
        };
        let voc = self.cfg.vocab_size as u32;
        let mut execs = Vec::with_capacity(buckets.len());
        for &b in &buckets {
            // 预热：同形状直发一次（cuBLASLt heuristic/PTX 模块落缓存——gr4）
            self.run_forward(b, b, &inp);
            stream.sync();
            // 捕获 → 实例化；采样烤进图尾：argmax s=B、采样行 [0..B)
            stream.begin_capture();
            self.run_forward(b, b, &inp);
            self.session.table().argmax_rows.run(
                &self.scr.logits, &fx_rows, &self.scr.am_scratch, &self.scr.am_out, voc, b, stream,
            );
            let exec: GraphExec<'static> = pin!(stream.end_capture().instantiate());
            execs.push(exec);
        }
        stream.sync();
        self.graph = Some(QGraphMode {
            sink_row,
            b_max: b_max as u32,
            buckets,
            in_ids,
            in_pos,
            in_seqs,
            in_rid,
            in_bt,
            fx_start,
            fx_count,
            fx_tpos,
            fx_rows,
            fx_iq,
            fx_ig,
            fx_ilq,
            fx_ilk,
            fx_ilv,
            execs,
            replays: 0,
        });
        Ok(())
    }

    pub fn cfg(&self) -> &Qwen35Config {
        &self.cfg
    }

    pub fn device_name(&self) -> &str {
        self.session.gpu().name()
    }

    pub fn pages_used(&self) -> usize {
        (self.pool.total() - self.pool.free_now() as u32) as usize
    }

    pub fn pages_of(&self, id: ReqId) -> usize {
        self.reqs.get(&id).map(|s| s.blocks.len()).unwrap_or(0)
    }

    /// 图回放步数（qg 系验收：证图路径确实被走到；NONE 档恒 0）
    pub fn graph_replays(&self) -> u64 {
        self.graph.as_ref().map(|g| g.replays).unwrap_or(0)
    }

    /// 单 token 前向探针（不动 forwarded——q5 逐步对拍用）：返回该位全词表
    /// logits（bf16 存储 → f32）
    pub fn forward_one(&mut self, id: ReqId, token: u32, pos: u32) -> Vec<f32> {
        let row = self
            .reqs
            .get(&id)
            .map(|s| s.row)
            .unwrap_or_else(|| panic!("forward_one：未知请求 {id}"));
        self.ensure_pages(id, pos as usize);
        self.forward_batch(&[(row, token, pos)], &[]);
        let bits: Vec<u16> = self.scr.logits.read_to_vec();
        let voc = self.cfg.vocab_size as usize;
        bits[..voc].iter().map(|&b| reference::bf16_to_f32(b)).collect()
    }

    /// 直跑完整生成（测试/serve CLI 便利面；不经过调度器）。
    /// 语义与 step 合同一致：追平段尾采样，之后逐 token decode
    pub fn generate(&mut self, prompt: &[u32], max_tokens: u32) -> Vec<u32> {
        self.reset();
        self.admitted(1, prompt);
        if max_tokens == 0 {
            return Vec::new();
        }
        assert!(
            prompt.len() + max_tokens as usize <= MAX_POS,
            "序列长超 paged_attn M0 上限 {MAX_POS}"
        );
        let mut out = Vec::new();
        let mut first_tok = None;
        loop {
            let Some(st) = self.reqs.get(&1) else { break };
            let (start, len) = (st.forwarded, st.seq.len());
            if start == len {
                break;
            }
            let take = (len - start).min(T_MAX);
            let row = st.row;
            let batch: Vec<(u32, u32, u32)> = (0..take)
                .map(|j| (row, self.reqs[&1].seq[start + j], (start + j) as u32))
                .collect();
            self.reqs.get_mut(&1).unwrap().forwarded += take;
            self.ensure_pages(1, start + take - 1);
            if start + take == self.reqs[&1].seq.len() {
                first_tok = Some(self.forward_batch(&batch, &[take as u32 - 1])[0]);
                break;
            }
            self.forward_batch(&batch, &[]);
        }
        let Some(tok) = first_tok else { return out };
        out.push(tok);
        self.reqs.get_mut(&1).unwrap().seq.push(tok);
        while (out.len() as u32) < max_tokens {
            let Some(st) = self.reqs.get(&1) else { break };
            let (row, pos) = (st.row, st.forwarded);
            let tok_in = st.seq[pos];
            let next = {
                self.reqs.get_mut(&1).unwrap().forwarded += 1;
                self.ensure_pages(1, pos);
                self.forward_rows(&[(row, tok_in, pos as u32)], &[0], true)[0]
            };
            out.push(next);
            self.reqs.get_mut(&1).unwrap().seq.push(next);
        }
        out
    }

    /// 清全部请求状态（页归还、行表清零、状态池整体零化）——generate 前提
    fn reset(&mut self) {
        let old = std::mem::take(&mut self.reqs);
        for (_, st) in old {
            for b in st.blocks {
                self.pool.release(b);
            }
        }
        self.bt_host.iter_mut().for_each(|x| *x = 0);
        self.next_row = 0;
        self.free_rows.clear();
        // 行号从 0 重来：全部状态池整体零化（单行零化只走回收路径）
        let stream = self.session.stream();
        for s in &self.lin_state {
            s.zero_on(stream);
        }
        for s in &self.conv_state {
            s.zero_on(stream);
        }
        stream.sync();
    }

    /// 页保证：请求的块表行长到覆盖 `upto_pos`（含）
    fn ensure_pages(&mut self, id: ReqId, upto_pos: usize) {
        let need = upto_pos / BLOCK as usize + 1;
        if !matches!(self.reqs.get(&id), Some(st) if st.blocks.len() < need) {
            return;
        }
        let bpr = self.bpr;
        let row = self.reqs.get(&id).unwrap().row as usize;
        let have = self.reqs.get(&id).unwrap().blocks.len();
        // 先攒本地（alloc_page 可能驱逐 radix，须避开 st 的借用）
        let news: Vec<u32> = (have..need).map(|_| self.alloc_page()).collect();
        let st = self.reqs.get_mut(&id).unwrap();
        for b in news {
            self.bt_host[row * bpr + st.blocks.len()] = b;
            st.blocks.push(b);
        }
    }

    /// 行状态零化（回收/抢占）：linear + conv 状态行的区间清零（流序）
    fn zero_row_state(&self, row: u32, stream: &Stream) {
        let (lr, cr) = (self.lin_row, self.conv_row);
        let off = row as usize;
        for s in &self.lin_state {
            s.zero_range_on(off * lr, lr, stream);
        }
        for s in &self.conv_state {
            s.zero_range_on(off * cr, cr, stream);
        }
    }

    // ── radix 前缀缓存（快照原语：行↔槽位 D2D 位拷贝，无重算——
    //     命中恢复后的递推与"从未中断"逐位一致是本刀的数值合同）──

    /// 行状态 → 快照槽的段表（流序：调用方在本步前向 kernel 之后一批发行）。
    /// 每槽 24 lin + 24 conv 段——逐段 async memcpy 的驱动调用开销在 WSL
    /// 约 4µs/次，每步多次跨块会放大成 ms 级；段表攒齐后一次 cuMemcpyBatchAsync
    fn push_snapshot_ops(&self, row: u32, slot: u32, ops: &mut Vec<CuBatchOp>) {
        let (ro, so) = (row as usize * self.lin_row, slot as usize * self.lin_row);
        for (s, t) in self.snap_lin.iter().zip(&self.lin_state) {
            ops.push(CuBatchOp {
                dst: s.device_ptr() + so as u64,
                src: t.device_ptr() + ro as u64,
                size: self.lin_row,
            });
        }
        let (ro, so) = (row as usize * self.conv_row, slot as usize * self.conv_row);
        for (s, t) in self.snap_conv.iter().zip(&self.conv_state) {
            ops.push(CuBatchOp {
                dst: s.device_ptr() + so as u64,
                src: t.device_ptr() + ro as u64,
                size: self.conv_row,
            });
        }
    }

    /// 快照槽 → 行状态（radix 命中恢复，一次批量 D2D）
    fn restore_row_snapshot(&self, row: u32, slot: u32) {
        let mut ops: Vec<CuBatchOp> =
            Vec::with_capacity(self.lin_state.len() + self.conv_state.len());
        let (ro, so) = (row as usize * self.lin_row, slot as usize * self.lin_row);
        for (t, s) in self.lin_state.iter().zip(&self.snap_lin) {
            ops.push(CuBatchOp {
                dst: t.device_ptr() + ro as u64,
                src: s.device_ptr() + so as u64,
                size: self.lin_row,
            });
        }
        let (ro, so) = (row as usize * self.conv_row, slot as usize * self.conv_row);
        for (t, s) in self.conv_state.iter().zip(&self.snap_conv) {
            ops.push(CuBatchOp {
                dst: t.device_ptr() + ro as u64,
                src: s.device_ptr() + so as u64,
                size: self.conv_row,
            });
        }
        self.session.stream().memcpy_batch(&ops);
    }

    /// 借槽（耗尽时 LRU 驱逐 radix 腾槽；无可驱逐返回 None——本请求降级无快照）
    fn acquire_slot(&mut self) -> Option<u32> {
        while self.snap_free.is_empty() {
            let freed = self.radix.evict_blocks(&mut self.pool, 1);
            if freed.is_empty() {
                return None;
            }
            for b in freed {
                if let Some(s) = self.snap_of_block.remove(&b) {
                    self.snap_free.push(s);
                }
            }
        }
        self.snap_free.pop()
    }

    /// 页分配（耗尽时 LRU 驱逐 radix 腾页——槽随块联动回收）
    fn alloc_page(&mut self) -> u32 {
        if self.pool.free_now() == 0 {
            let freed = self.radix.evict_blocks(&mut self.pool, 8);
            for b in freed {
                if let Some(s) = self.snap_of_block.remove(&b) {
                    self.snap_free.push(s);
                }
            }
            self.pool.recycle();
        }
        self.pool
            .alloc()
            .unwrap_or_else(|| panic!("页池耗尽（{} 页）——I1 配置须保证容量", self.pool.total()))
    }

    /// 批前向 + 设备端采样。rows: (块表行, token, 位置)；samples: 采样行
    /// 全局下标。返回每采样行一个 token。批内同请求行须连续（断言钉——
    /// conv tap/deltanet 递推的段语义前提）
    // ── 插桩辅助（未开启时 borrow 落空即返——热路径成本 ns 级）──

    fn prof_mark(&self, sect: &'static str) {
        if let Some(p) = self.prof.borrow_mut().as_mut() {
            p.mark(sect, self.session.stream());
        }
    }
    fn prof_upload_done(&self, t: Instant) {
        if let Some(p) = self.prof.borrow_mut().as_mut() {
            p.upload_ms += prof_ms(t.elapsed());
            p.begin(self.session.stream());
        }
    }
    fn prof_note_submit(&self, t: Instant) {
        if let Some(p) = self.prof.borrow_mut().as_mut() {
            p.submit_ms += prof_ms(t.elapsed());
        }
    }
    fn prof_end(&self) {
        if let Some(p) = self.prof.borrow_mut().as_mut() {
            p.end(self.session.stream());
            p.calibrate(self.session.stream());
        }
    }
    fn prof_note_readback(&self, t: Instant) {
        if let Some(p) = self.prof.borrow_mut().as_mut() {
            p.readback_ms += prof_ms(t.elapsed());
        }
    }
    fn prof_finalize(&self, wall: Instant) {
        if let Some(p) = self.prof.borrow_mut().as_mut() {
            p.finalize(prof_ms(wall.elapsed()));
        }
    }

    /// 步时分解报告（markdown；插桩未开 → None）。q13 结构合同 + 满负荷
    /// 账定位的读数源（examples/qwen_step_profile 驱动）
    pub fn step_profile_report(&self) -> Option<String> {
        self.prof.borrow().as_ref().map(|p| p.report())
    }

    /// 段表清零（校准单价保留）——example 按 m 分段读数的隔板
    pub fn step_profile_reset(&self) {
        if let Some(p) = self.prof.borrow_mut().as_mut() {
            p.totals.clear();
            p.hits.clear();
            p.steps = 0;
            p.wall_ms = 0.0;
            p.upload_ms = 0.0;
            p.submit_ms = 0.0;
            p.readback_ms = 0.0;
        }
    }

    fn forward_batch(&mut self, rows: &[(u32, u32, u32)], samples: &[u32]) -> Vec<u32> {
        // context 线程局部：engine 任务可在 worker 间迁移，进设备路径先绑
        self.session.gpu().bind_thread();
        let t = rows.len();
        assert!(t > 0 && t <= T_MAX, "批大小 {t} 超前向缓冲上限 {T_MAX}");
        for &(_, _, pos) in rows {
            assert!((pos as usize) < MAX_POS, "位置 {pos} 超 paged_attn M0 上限 {MAX_POS}");
        }
        // ---- span 构造：同请求连续行归一段 ----
        let mut row_start: Vec<u32> = Vec::new();
        let mut row_count: Vec<u32> = Vec::new();
        let mut row_id: Vec<u32> = Vec::new();
        for (j, &(r, _, _)) in rows.iter().enumerate() {
            if j > 0 && rows[j - 1].0 == r {
                *row_count.last_mut().unwrap() += 1;
            } else {
                row_start.push(j as u32);
                row_count.push(1);
                row_id.push(r);
            }
        }
        let n_rows = row_start.len() as u32;
        // 一请求一段：跨段同请求 = 前段 tap/递推上下文与后段断裂——合同违约
        let mut seen = HashSet::new();
        for &r in &row_id {
            assert!(seen.insert(r), "批内请求 {r} 的行不连续——调度合同违约");
        }
        // conv 的 token 侧索引：状态行 + 段内局部位（tap 越段前取 state）
        let mut tok_row = Vec::with_capacity(t);
        let mut tok_pos = Vec::with_capacity(t);
        {
            let mut si = 0usize;
            for (j, &(r, _, _)) in rows.iter().enumerate() {
                while row_id[si] != r {
                    si += 1;
                }
                tok_row.push(r);
                tok_pos.push((j - row_start[si] as usize) as u32);
            }
        }

        let (gpu, stream, tb) = (self.session.gpu(), self.session.stream(), self.session.table());
        let prof_on = self.prof.borrow().is_some();
        let tw = Instant::now();
        let ids: Vec<u32> = rows.iter().map(|r| r.1).collect();
        let pos: Vec<u32> = rows.iter().map(|r| r.2).collect();
        let seqs: Vec<u32> = rows.iter().map(|r| r.0).collect();
        // gather 拆分下标（单位 = gather 宽）：
        // full qg [T, 16×(q256‖gate256)] → q/gate 各 [T,16,256]
        let nqh = self.cfg.num_heads as usize;
        let ids_q: Vec<u32> =
            (0..t).flat_map(|tt| (0..nqh).map(move |h| (tt * 2 * nqh + 2 * h) as u32)).collect();
        let ids_gate: Vec<u32> = (0..t)
            .flat_map(|tt| (0..nqh).map(move |h| (tt * 2 * nqh + 2 * h + 1) as u32))
            .collect();
        // linear qkv [T, 2048|2048|4096] → q/k 各 [T,2048]、v [T,4096]
        let ids_lq: Vec<u32> = (0..t).map(|tt| (tt * 4) as u32).collect();
        let ids_lk: Vec<u32> = (0..t).map(|tt| (tt * 4 + 1) as u32).collect();
        let ids_lv: Vec<u32> = (0..t).map(|tt| (tt * 2 + 1) as u32).collect();
        let tu = Instant::now();
        let (d_ids, d_pos, d_seqs, d_bt) = (
            gpu.alloc_from_slice(&ids),
            gpu.alloc_from_slice(&pos),
            gpu.alloc_from_slice(&seqs),
            gpu.alloc_from_slice(&self.bt_host),
        );
        let (d_start, d_count, d_rid) = (
            gpu.alloc_from_slice(&row_start),
            gpu.alloc_from_slice(&row_count),
            gpu.alloc_from_slice(&row_id),
        );
        let (d_trow, d_tpos) =
            (gpu.alloc_from_slice(&tok_row), gpu.alloc_from_slice(&tok_pos));
        let (d_iq, d_ig) = (gpu.alloc_from_slice(&ids_q), gpu.alloc_from_slice(&ids_gate));
        let (d_ilq, d_ilk, d_ilv) = (
            gpu.alloc_from_slice(&ids_lq),
            gpu.alloc_from_slice(&ids_lk),
            gpu.alloc_from_slice(&ids_lv),
        );
        let inp = FwdIn {
            ids: &d_ids,
            pos: &d_pos,
            seqs: &d_seqs,
            bt: &d_bt,
            start: &d_start,
            count: &d_count,
            rid: &d_rid,
            trow: &d_trow,
            tpos: &d_tpos,
            iq: &d_iq,
            ig: &d_ig,
            ilq: &d_ilq,
            ilk: &d_ilk,
            ilv: &d_ilv,
        };
        if prof_on {
            self.prof_upload_done(tu);
        }
        let tsub = Instant::now();
        self.run_forward(t as u32, n_rows, &inp);
        if prof_on {
            self.prof_note_submit(tsub);
        }

        // ── 采样（设备端 argmax；只回读采样行的 u32）──
        if samples.is_empty() {
            self.prof_end();
            stream.sync();
            if prof_on {
                self.prof_finalize(tw);
            }
            return Vec::new();
        }
        let s = samples.len() as u32;
        assert!((s as usize) <= t, "采样行数超批");
        self.prof_mark("argmax");
        let d_rows = gpu.alloc_from_slice(samples);
        tb.argmax_rows.run(
            &self.scr.logits, &d_rows, &self.scr.am_scratch, &self.scr.am_out,
            self.cfg.vocab_size as u32, s, stream,
        );
        self.prof_end();
        stream.sync();
        let tr = Instant::now();
        let out: Vec<u32> = self.scr.am_out.read_to_vec();
        if prof_on {
            self.prof_note_readback(tr);
            self.prof_finalize(tw);
        }
        out[..s as usize].to_vec()
    }

    /// 前向 kernel 序（embed → 32 层 → final norm + logits；不含采样/
    /// sync/回读）——eager 直发与图捕获共用同一条序列（gr1 位一致合同）
    fn run_forward(&self, t32: u32, n_rows: u32, inp: &FwdIn) {
        let stream = self.session.stream();
        let tb = self.session.table();
        let eps = self.cfg.rms_eps;
        let (h, ii, voc) =
            (self.cfg.hidden_size as u32, self.cfg.inter_size as u32, self.cfg.vocab_size as u32);

        // ── 对拍钩子（WEIN_TRACE_DIR 设定且无图档时激活；图捕获中禁 sync）──
        let step = TRACE_STEP.fetch_add(1, Ordering::Relaxed) + 1;
        let trace = self.graph.is_none() && std::env::var("WEIN_TRACE_DIR").is_ok();
        if trace && step == 1 {
            self.trace_meta();
        }

        // ── embed ──
        self.prof_mark("embed");
        tb.embed.run(inp.ids, &self.embed, &self.scr.x, h, t32, stream);
        if trace {
            self.trace_x(&format!("s{step}-embed"), t32);
        }

        // ── 层循环（x 主路：n1 ← norm(x)；h2 = x + o；x = h2 + mlp）──
        let mut lin_i = 0usize;
        let mut full_i = 0usize;
        for (li, layer) in self.layers.iter().enumerate() {
            self.prof_mark("norm");
            tb.rms_norm_zc.run(&self.scr.x, &layer.ln1, &self.scr.n1, t32, h, eps, stream);
            match &layer.attn {
                AttnDev::Full(w) => {
                    let (qh_n, kvh, hd) = (self.cfg.num_heads, self.cfg.num_kv_heads, self.cfg.head_dim);
                    let he = kvh * hd;
                    self.prof_mark("gemm");
                    tb.gemm.run_wt(&self.scr.n1, &w.qg, &self.scr.qg, t32, 2 * qh_n * hd, h, stream);
                    if trace {
                        self.trace_buf(&format!("s{step}-L{li:02}-qg"), &self.scr.qg, t32 as usize * 2 * qh_n as usize * hd as usize);
                    }
                    self.prof_mark("gather");
                    tb.embed.run(inp.iq, &self.scr.qg, &self.scr.qh, hd, t32 * qh_n, stream);
                    tb.embed.run(inp.ig, &self.scr.qg, &self.scr.gate, hd, t32 * qh_n, stream);
                    // 零中心 norm 在 head_dim 上（原地：行私有读写，安全）
                    self.prof_mark("norm");
                    tb.rms_norm_zc.run(&self.scr.qh, &w.qn, &self.scr.qh, t32 * qh_n, hd, eps, stream);
                    self.prof_mark("gemm");
                    tb.gemm.run_wt(&self.scr.n1, &w.k, &self.scr.kh, t32, he, h, stream);
                    tb.gemm.run_wt(&self.scr.n1, &w.v, &self.scr.vh, t32, he, h, stream);
                    if trace {
                        self.trace_buf(&format!("s{step}-L{li:02}-vh"), &self.scr.vh, t32 as usize * he as usize);
                    }
                    self.prof_mark("norm");
                    tb.rms_norm_zc.run(&self.scr.kh, &w.kn, &self.scr.kh, t32 * kvh, hd, eps, stream);
                    self.prof_mark("rope");
                    let rd = self.cfg.rot_dim();
                    tb.rope_pt.run(
                        &self.scr.qh, &self.cos, &self.sin, inp.pos, &self.scr.qh, rd, hd, qh_n,
                        t32, stream,
                    );
                    tb.rope_pt.run(
                        &self.scr.kh, &self.cos, &self.sin, inp.pos, &self.scr.kh, rd, hd, kvh,
                        t32, stream,
                    );
                    self.prof_mark("kv");
                    tb.kv_scatter.run(
                        &self.scr.kh, &self.scr.vh, inp.bt, inp.seqs, inp.pos, &self.k_pool[full_i],
                        &self.v_pool[full_i], self.bpr as u32, BLOCK, he, t32, stream,
                    );
                    self.prof_mark("attn");
                    tb.paged_attn.run(
                        &self.scr.qh, &self.k_pool[full_i], &self.v_pool[full_i], inp.bt, inp.seqs,
                        inp.pos, &self.scr.attn, self.q_scale, self.bpr as u32, BLOCK, qh_n, kvh,
                        hd, t32, stream,
                    );
                    self.prof_mark("eltw");
                    tb.sigmul.run(
                        &self.scr.attn, &self.scr.gate, &self.scr.agated, t32 * qh_n * hd, stream,
                    );
                    if trace {
                        self.trace_buf(&format!("s{step}-L{li:02}-attn"), &self.scr.attn, t32 as usize * qh_n as usize * hd as usize);
                        self.trace_buf(&format!("s{step}-L{li:02}-gate"), &self.scr.gate, t32 as usize * qh_n as usize * hd as usize);
                        self.trace_buf(&format!("s{step}-L{li:02}-agated"), &self.scr.agated, t32 as usize * qh_n as usize * hd as usize);
                    }
                    self.prof_mark("gemm");
                    tb.gemm.run_wt(&self.scr.agated, &w.o, &self.scr.o, t32, h, qh_n * hd, stream);
                    if trace {
                        self.trace_buf(&format!("s{step}-L{li:02}-o"), &self.scr.o, t32 as usize * h as usize);
                    }
                    full_i += 1;
                }
                AttnDev::Linear(w) => {
                    let (kh_n, vh_n, kd, vd) = (
                        self.cfg.linear_num_k_heads,
                        self.cfg.linear_num_v_heads,
                        self.cfg.linear_k_dim,
                        self.cfg.linear_v_dim,
                    );
                    let (keyd, vald) = (kh_n * kd, vh_n * vd);
                    let cd = 2 * keyd + vald;
                    self.prof_mark("gemm");
                    tb.gemm.run_wt(&self.scr.n1, &w.qkv, &self.scr.qkv, t32, cd, h, stream);
                    self.prof_mark("conv");
                    tb.conv1d_dw.run(
                        &self.scr.qkv, &w.conv, &self.conv_state[lin_i], &self.scr.qkvc, inp.trow,
                        inp.tpos, cd, self.cfg.conv_kernel, t32, stream,
                    );
                    tb.conv_state_roll.run(
                        // 状态存**原始 qkv**（conv 前）——tap 与段内同源；
                        // 存 qkvc（conv+silu 后）是解码递推偏差的根源
                        &self.scr.qkv, &self.conv_state[lin_i], inp.start, inp.count, inp.rid, cd,
                        self.cfg.conv_kernel, n_rows, stream,
                    );
                    self.prof_mark("gather");
                    tb.embed.run(inp.ilq, &self.scr.qkvc, &self.scr.qs, keyd, t32, stream);
                    tb.embed.run(inp.ilk, &self.scr.qkvc, &self.scr.ks, keyd, t32, stream);
                    tb.embed.run(inp.ilv, &self.scr.qkvc, &self.scr.vs, vald, t32, stream);
                    self.prof_mark("gemm");
                    tb.gemm.run_wt(&self.scr.n1, &w.z, &self.scr.z, t32, vald, h, stream);
                    tb.gemm.run_wt(&self.scr.n1, &w.a, &self.scr.aa, t32, 32, h, stream);
                    tb.gemm.run_wt(&self.scr.n1, &w.b, &self.scr.bb, t32, 32, h, stream);
                    self.prof_mark("delta");
                    tb.deltanet.run(
                        &self.scr.qs, &self.scr.ks, &self.scr.vs, &self.scr.aa, &self.scr.bb,
                        &w.a_log, &w.dtb, &self.lin_state[lin_i], &self.scr.core, inp.start,
                        inp.count, inp.rid, n_rows, kh_n, vh_n, kd, vd, L2_EPS, stream,
                    );
                    if trace {
                        // 定位解码递推偏差：conv 输出与 deltanet 输出（decode vs
                        // 整段预填的同 token 对拍——分叉首组件即病灶）
                        self.trace_buf(&format!("s{step}-L{li:02}-qkvc"), &self.scr.qkvc, t32 as usize * cd as usize);
                        self.trace_buf(&format!("s{step}-L{li:02}-core"), &self.scr.core, t32 as usize * vald as usize);
                    }
                    self.prof_mark("norm");
                    tb.rms_norm_gated.run(
                        &self.scr.core, &self.scr.z, &w.norm, &self.scr.normed, t32 * vh_n, vd,
                        eps, stream,
                    );
                    self.prof_mark("gemm");
                    tb.gemm.run_wt(&self.scr.normed, &w.out, &self.scr.o, t32, h, vald, stream);
                    lin_i += 1;
                }
            }
            self.prof_mark("eltw");
            tb.add.run(&self.scr.x, &self.scr.o, &self.scr.h2, t32 * h, stream);
            self.prof_mark("norm");
            tb.rms_norm_zc.run(&self.scr.h2, &layer.ln2, &self.scr.n2, t32, h, eps, stream);
            self.prof_mark("gemm");
            tb.gemm.run_wt(&self.scr.n2, &layer.wg, &self.scr.g, t32, ii, h, stream);
            tb.gemm.run_wt(&self.scr.n2, &layer.wu, &self.scr.u, t32, ii, h, stream);
            self.prof_mark("eltw");
            tb.silu.run(&self.scr.g, &self.scr.act, t32 * ii, stream);
            tb.mul.run(&self.scr.act, &self.scr.u, &self.scr.m, t32 * ii, stream);
            self.prof_mark("gemm");
            tb.gemm.run_wt(&self.scr.m, &layer.wd, &self.scr.d, t32, h, ii, stream);
            self.prof_mark("eltw");
            tb.add.run(&self.scr.h2, &self.scr.d, &self.scr.x, t32 * h, stream);
            if trace {
                self.trace_x(&format!("s{step}-layer-{li:02}"), t32);
            }
        }
        // ── final norm + logits（tie：embed 兼作 lm_head 的 W[n,k]）──
        self.prof_mark("norm");
        tb.rms_norm_zc.run(&self.scr.x, &self.norm_f, &self.scr.n2, t32, h, eps, stream);
        self.prof_mark("logits");
        tb.gemm.run_wt(&self.scr.n2, &self.embed, &self.scr.logits, t32, voc, h, stream);
        if trace {
            self.trace_logits(step, t32);
        }
    }

    /// 通用缓冲落盘（bf16 n 元素 → f32 LE）——full-attn 组件级对拍用
    fn trace_buf(&self, tag: &str, buf: &DevBuf, n: usize) {
        let Ok(dir) = std::env::var("WEIN_TRACE_DIR") else { return };
        if self.graph.is_some() {
            return;
        }
        self.session.stream().sync();
        let bits: Vec<u16> = buf.read_to_vec();
        let vals: Vec<f32> = bits[..n].iter().map(|&b| reference::bf16_to_f32(b)).collect();
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("{dir}/{tag}.f32"), bytes).expect("trace 写盘");
    }

    /// 逐层对拍落盘：残差流 scr.x（bf16）→ f32 LE，文件 `<dir>/<tag>.f32`
    /// 行主序 [t, hidden]。仅 WEIN_TRACE_DIR 设定时有副作用（读回含 sync，
    /// 图档下不得触发——调用侧已挡，此处再挡一层）
    fn trace_x(&self, tag: &str, t: u32) {
        let Ok(dir) = std::env::var("WEIN_TRACE_DIR") else { return };
        if self.graph.is_some() {
            return;
        }
        self.session.stream().sync();
        let h = self.cfg.hidden_size as usize;
        let bits: Vec<u16> = self.scr.x.read_to_vec();
        let vals: Vec<f32> =
            bits[..t as usize * h].iter().map(|&b| reference::bf16_to_f32(b)).collect();
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("{dir}/{tag}.f32"), bytes).expect("trace 写盘");
    }

    /// logits 落盘（[t, vocab] f32；对拍终局用）
    fn trace_logits(&self, step: u32, t: u32) {
        let Ok(dir) = std::env::var("WEIN_TRACE_DIR") else { return };
        if self.graph.is_some() {
            return;
        }
        self.session.stream().sync();
        let voc = self.cfg.vocab_size as usize;
        let bits: Vec<u16> = self.scr.logits.read_to_vec();
        let vals: Vec<f32> =
            bits[..t as usize * voc].iter().map(|&b| reference::bf16_to_f32(b)).collect();
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("{dir}/s{step}-logits.f32"), bytes).expect("trace 写盘");
    }

    /// 首步写 meta：hidden/vocab/eps + 层类型表（对拍侧定位层用）
    fn trace_meta(&self) {
        let Ok(dir) = std::env::var("WEIN_TRACE_DIR") else { return };
        let kinds: Vec<&str> = self
            .layers
            .iter()
            .map(|l| match &l.attn {
                AttnDev::Full(_) => "full",
                AttnDev::Linear(_) => "linear",
            })
            .collect();
        let meta = format!(
            "{{\"hidden\":{},\"vocab\":{},\"eps\":{},\"layers\":[{}]}}\n",
            self.cfg.hidden_size,
            self.cfg.vocab_size,
            self.cfg.rms_eps,
            kinds.iter().map(|k| format!("\"{k}\"")).collect::<Vec<_>>().join(",")
        );
        std::fs::write(format!("{dir}/meta.json"), meta).expect("trace meta 写盘");
    }

    /// 步前向统一入口：pure_decode（每行独立请求的 1-token 步，采样行
    /// = 全部行）且桶覆盖 → 图回放；否则 eager 直发。混合步/prefill 永远
    /// eager（图只在纯 decode 档——批形状固定，§4 入图安全规则）
    fn forward_rows(&mut self, rows: &[(u32, u32, u32)], samples: &[u32], pure_decode: bool) -> Vec<u32> {
        if pure_decode && !rows.is_empty() && samples.len() == rows.len() {
            if let Some(bi) = self.bucket_for(rows.len()) {
                return self.forward_graph(bi, rows);
            }
        }
        self.forward_batch(rows, samples)
    }

    /// ≥m 的最小桶下标；超最大桶 → None（回落 eager）
    fn bucket_for(&self, m: usize) -> Option<usize> {
        let g = self.graph.as_ref()?;
        if m > g.b_max as usize {
            return None;
        }
        g.buckets.iter().position(|&b| b as usize >= m)
    }

    /// 图回放前向：前 m 行真数据，其余行由 sink 吸收（tok 0 / pos 0 /
    /// seq=sink——KV/递推垃圾只落 sink 专属页与状态行）。数据五缓冲按
    /// b_max 整段覆写，一次 graph launch 重放全 kernel 序（含图尾 argmax），
    /// 只回读前 m 个采样 u32
    fn forward_graph(&mut self, b_idx: usize, rows: &[(u32, u32, u32)]) -> Vec<u32> {
        self.session.gpu().bind_thread();
        let (b, cap, m, sink) = {
            let g = self.graph.as_ref().expect("forward_graph 须图档");
            (g.buckets[b_idx] as usize, g.b_max as usize, rows.len(), g.sink_row)
        };
        assert!(m > 0 && m <= b, "桶路由合同：m {m} ≤ 桶 {b}");
        for &(_, _, pos) in rows {
            assert!((pos as usize) < MAX_POS, "位置 {pos} 超 paged_attn M0 上限 {MAX_POS}");
        }
        // host 组装：b_max 全量（默认值 = sink 吸收行），前 m 行写真数据
        let mut ids = vec![0u32; cap];
        let mut pos = vec![0u32; cap];
        let mut seqs = vec![sink; cap];
        let mut rid = vec![sink; cap];
        for (j, &(r, tok, p)) in rows.iter().enumerate() {
            ids[j] = tok;
            pos[j] = p;
            seqs[j] = r;
            rid[j] = r;
        }
        let stream = self.session.stream();
        {
            let g = self.graph.as_ref().expect("forward_graph 须图档");
            g.in_ids.upload_on(&ids, stream);
            g.in_pos.upload_on(&pos, stream);
            g.in_seqs.upload_on(&seqs, stream);
            g.in_rid.upload_on(&rid, stream);
            // 页保证已先于本调用完成（step/generate 合同）——bt_host 是最新账
            g.in_bt.upload_on(&self.bt_host, stream);
            g.execs[b_idx].launch(stream);
        }
        stream.sync();
        self.graph.as_mut().unwrap().replays += 1;
        let out: Vec<u32> = self.scr.am_out.read_to_vec();
        out[..m].to_vec()
    }
}

// ---- 布局常量辅助（open 期与运行期共用）----

fn bpr_of() -> usize {
    MAX_POS.div_ceil(BLOCK as usize)
}

/// 线性层递推状态**单行**字节数 [v_heads, v_dim, k_dim] f32（池 = ×MAX_ROWS）
fn lin_row_bytes(cfg: &Qwen35Config) -> usize {
    let vh = cfg.linear_num_v_heads as usize;
    let kd = cfg.linear_k_dim as usize;
    let vd = cfg.linear_v_dim as usize;
    vh * kd * vd * 2 // 状态池 bf16（寄存器 f32 累加、落池收 bf16——reference 同式镜像）
}

/// conv 状态**单行**字节数 [conv_dim, kernel-1] bf16（池 = ×MAX_ROWS）
fn conv_row_bytes(cfg: &Qwen35Config) -> usize {
    let cd = cfg.conv_dim() as usize;
    cd * (cfg.conv_kernel as usize - 1) * 2
}

impl Executor for Qwen35Executor {
    fn row_capacity(&self) -> usize {
        self.row_capacity as usize
    }

    /// 块边界（= 快照检查点/前缀共享粒度）。radix 关时上报 1：调度账与
    /// 旧版逐位一致（A/B 基线）
    fn prefix_align(&self) -> usize {
        if self.radix_on {
            BLOCK as usize
        } else {
            1
        }
    }

    fn admitted(&mut self, id: ReqId, prompt: &[u32]) -> usize {
        assert!(
            prompt.iter().all(|&t| t < self.cfg.vocab_size),
            "token 越界（vocab {}）",
            self.cfg.vocab_size
        );
        let seq = if prompt.is_empty() { vec![0u32] } else { prompt.to_vec() };
        let row = self.free_rows.pop().unwrap_or_else(|| {
            assert!(
                self.next_row < self.row_capacity,
                "块表行耗尽（{}，图档含 sink 让位 1 行）——I1 并发上限",
                self.row_capacity
            );
            let r = self.next_row;
            self.next_row += 1;
            r
        });
        // radix 前缀缓存命中：最长匹配后截断到有快照槽的祖先（无槽尾块
        // 不共享——恢复点之后的重算覆盖写会以 ulp 漂移污染共享页），
        // 槽→行状态恢复 + 块表继承 + 前向游标前移。边界还须严格小于
        // seq.len()：恢复到 prompt 末尾 = 无 token 可喂 = 无采样位
        // （vLLM 同款"末 token 重算"规则——精确块对齐 prompt 不命中）
        let mut blocks = Vec::new();
        let mut shared = 0usize;
        if self.radix_on {
            let m = self.radix.match_prefix(&mut self.pool, &seq);
            if self.radix_dbg {
                eprintln!(
                    "[radix] match blocks={} tokens={} seq={} snap_keys={:?}",
                    m.blocks.len(),
                    m.tokens,
                    seq.len(),
                    self.snap_of_block.keys().take(8).collect::<Vec<_>>()
                );
            }
            if !m.blocks.is_empty() {
                let keep = (0..m.blocks.len())
                    .rev()
                    .find(|&i| {
                        ((i + 1) * BLOCK as usize) < seq.len()
                            && self.snap_of_block.contains_key(&m.blocks[i])
                    })
                    .map_or(0, |i| i + 1);
                if keep < m.blocks.len() {
                    self.radix.unpin_range(&mut self.pool, &seq, keep, m.blocks.len());
                }
                if keep > 0 {
                    let slot = self.snap_of_block[&m.blocks[keep - 1]];
                    self.restore_row_snapshot(row, slot);
                    let base = row as usize * self.bpr;
                    for (j, &b) in m.blocks[..keep].iter().enumerate() {
                        self.bt_host[base + j] = b;
                    }
                    blocks = m.blocks[..keep].to_vec();
                    shared = keep;
                }
            }
        }
        if self.radix_dbg {
            eprintln!(
                "[radix] admit seq={} keep_blocks={} hit_tokens={}",
                seq.len(),
                shared,
                shared * BLOCK as usize
            );
        }
        let forwarded = shared * BLOCK as usize;
        self.reqs.insert(
            id,
            ReqState { seq, forwarded, row, blocks, shared, snaps: Vec::new() },
        );
        forwarded
    }

    fn step(&mut self, plan: &StepPlan) -> Vec<(ReqId, u32)> {
        struct RowIn {
            id: ReqId,
            row: u32,
            tok: u32,
            pos: u32,
            /// 该行是 prefill（prompt）token——决定用哪条快照栅格
            pref: bool,
        }
        let mut batch: Vec<RowIn> = Vec::new();
        let mut samples: Vec<(ReqId, usize)> = Vec::new();
        // 1. prefill 段（追平步段尾即采样位）
        for &(id, n) in &plan.prefill {
            let Some(st) = self.reqs.get_mut(&id) else { continue };
            let start = st.forwarded;
            let take = (n as usize).min(st.seq.len().saturating_sub(start));
            for j in 0..take {
                batch.push(RowIn {
                    id,
                    row: st.row,
                    tok: st.seq[start + j],
                    pos: (start + j) as u32,
                    pref: true,
                });
            }
            st.forwarded += take;
            if take > 0 && plan.decode.contains(&id) {
                samples.push((id, batch.len() - 1));
            }
        }
        // 2. 纯 decode：前向 seq 尾
        for &id in &plan.decode {
            if plan.prefill.iter().any(|&(p, _)| p == id) {
                continue;
            }
            let Some(st) = self.reqs.get_mut(&id) else { continue };
            if st.forwarded < st.seq.len() {
                let pos = st.forwarded;
                batch.push(RowIn {
                    id,
                    row: st.row,
                    tok: st.seq[pos],
                    pos: pos as u32,
                    pref: false,
                });
                st.forwarded += 1;
                samples.push((id, batch.len() - 1));
            }
        }
        if batch.is_empty() {
            return Vec::new();
        }
        // 3. 页保证（每请求最大 pos）——先于 scatter
        let mut need: HashMap<ReqId, usize> = HashMap::new();
        for r in &batch {
            let e = need.entry(r.id).or_insert(0);
            *e = (*e).max(r.pos as usize);
        }
        for (id, upto) in need {
            self.ensure_pages(id, upto);
        }
        // 4. 前向 + 设备端贪心。prefill 空 = 纯 decode 步（每行独立请求、
        //    采样行 = 全部行）→ 图档候选；否则 eager
        let rows: Vec<(u32, u32, u32)> = batch.iter().map(|r| (r.row, r.tok, r.pos)).collect();
        let sample_rows: Vec<u32> = samples.iter().map(|s| s.1 as u32).collect();
        let toks = self.forward_rows(&rows, &sample_rows, plan.prefill.is_empty());
        // 边界快照：本步把行推过硬栅格边界时落一份位拷贝（radix 命中恢复
        // 点；每请求逐边界各借一槽、完成时随块移交）。流序：批
        // 量 memcpy 排在前向 kernel 之后，读到的是落定状态；全步段表攒齐
        // 一次发行（逐段 async 在 WSL 约 4µs/次驱动开销）。栅格按相位分：
        // prompt 侧 = BLOCK（命中粒度 = 块），decode 侧默认 64（快照 25MB/
        // 行/份，16 栅格在 c=100 满负荷 +1.4ms/步，decode 命中场景稀疏）
        if self.radix_on {
            let mut last_pos: HashMap<ReqId, (usize, bool)> = HashMap::new();
            for r in &batch {
                let e = last_pos.entry(r.id).or_insert((0, r.pref));
                if r.pos as usize >= e.0 {
                    *e = (r.pos as usize, r.pref);
                }
            }
            // 槽位决策先做完（耗尽驱逐会动 radix/snap_free），再攒段表
            let mut snaps: Vec<(u32, u32)> = Vec::new();
            for (id, (p, pref)) in last_pos {
                let nb = (p + 1) / BLOCK as usize;
                // prompt 侧恒按 BLOCK（块边界才可共享/可恢复；密度由调度
                // 侧 chunk 对齐到块边界决定）；decode 侧按 snap_grid_dec
                let grid = if pref { BLOCK as usize } else { self.snap_grid_dec };
                if nb == 0 || grid == 0 || (p + 1) % grid != 0 {
                    continue;
                }
                let (row, blk) = match self.reqs.get(&id) {
                    Some(st) if st.blocks.len() == nb => (st.row, st.blocks[nb - 1]),
                    _ => continue, // 页未盖满（防御）或请求已不在
                };
                let Some(slot) = self.acquire_slot() else {
                    continue; // 槽耗尽：降级无快照
                };
                if let Some(st) = self.reqs.get_mut(&id) {
                    st.snaps.push((blk, slot));
                }
                snaps.push((row, slot));
            }
            if !snaps.is_empty() {
                let per = self.snap_lin.len() + self.snap_conv.len();
                let mut ops: Vec<CuBatchOp> = Vec::with_capacity(snaps.len() * per);
                for (row, slot) in snaps {
                    self.push_snapshot_ops(row, slot, &mut ops);
                }
                self.session.stream().memcpy_batch(&ops);
            }
        }
        let mut out = Vec::new();
        for ((id, _), tok) in samples.iter().zip(toks) {
            if let Some(st) = self.reqs.get_mut(id) {
                st.seq.push(tok);
            }
            out.push((*id, tok));
        }
        out
    }

    fn retain(&mut self, active: &[ReqId]) {
        let stream = self.session.stream();
        let gone: Vec<ReqId> =
            self.reqs.keys().copied().filter(|k| !active.contains(k)).collect();
        for id in gone {
            if let Some(st) = self.reqs.remove(&id) {
                // 前缀缓存移交：本请求写过的完整私有块串挂 radix（所有权
                // 转移，rc 不变）；沿途边界快照随块挂（槽内容 = 该块末的
                // 位拷贝——恢复点与共享前缀严格对齐才有效）
                let full = (st.forwarded / BLOCK as usize).min(st.blocks.len());
                let shared = st.shared.min(full);
                // 逐块移交：块进树则槽随块（snap_of_block），未进树/冗余段的
                // 槽回池。多槽是常态——prompt 侧每个 chunk 尾各一份
                let mut kept: Vec<u32> = Vec::new();
                if self.radix_on && full > shared {
                    let tok_end = full * BLOCK as usize;
                    // 树既有覆盖深度（>= shared：命中段被 pin 不可逐）。
                    // [shared..d) 私有块与树重叠 = 冗余页，释放；只把分叉
                    // 后新段 [d..full) 交 insert（blocks 契约 = 分叉点对齐）
                    let d = self.radix.coverage(&st.seq[..tok_end]);
                    debug_assert!(d >= shared, "命中段被 pin，覆盖不可能更浅");
                    for &b in &st.blocks[shared..d] {
                        self.pool.release(b);
                    }
                    if self.radix_dbg {
                        eprintln!(
                            "[radix] retain seq={} fwd={} full={} shared={} cover={} snaps={:?}",
                            st.seq.len(),
                            st.forwarded,
                            full,
                            shared,
                            d,
                            st.snaps
                        );
                    }
                    if d < full {
                        kept = st.blocks[d..full].to_vec();
                        self.radix.insert(&st.seq[..tok_end], &kept);
                    }
                }
                for &(b, slot) in &st.snaps {
                    if kept.contains(&b) {
                        self.snap_of_block.insert(b, slot);
                    } else {
                        self.snap_free.push(slot);
                    }
                }
                // 页记账：命中共享块 unpin（请求份释放）；已移交私有块保留
                // （radix 份）；未移交的完整块与部分尾块 release。
                // radix 关闭态 shared 恒 0 → 全量 release = 旧语义
                if self.radix_on && shared > 0 {
                    self.radix.unpin(&mut self.pool, &st.seq, shared);
                }
                let released_from = if self.radix_on { full } else { 0 };
                for &b in &st.blocks[released_from..] {
                    self.pool.release(b);
                }
                let base = st.row as usize * self.bpr;
                for j in 0..st.blocks.len() {
                    self.bt_host[base + j] = 0;
                }
                self.zero_row_state(st.row, stream); // 回收行状态零化（q10 钉）
                self.free_rows.push(st.row);
            }
        }
        self.pool.recycle();
    }

    fn preempted(&mut self, id: ReqId) {
        if let Some(st) = self.reqs.get_mut(&id) {
            let row = st.row;
            // 快照槽全数作废回池（行即将零化；重放重新落槽）
            for (_, slot) in st.snaps.drain(..) {
                self.snap_free.push(slot);
            }
            // 去共享：命中块换新私有页——重放覆盖写有 ulp 漂移（跨批
            // 形状 GEMM algo 差），共享页不可重写（污染缓存/他人）
            let shared = st.shared;
            if self.radix_on && shared > 0 {
                let head: Vec<u32> = st.seq[..shared * BLOCK as usize].to_vec();
                self.radix.unpin(&mut self.pool, &head, shared);
                let bpr = self.bpr;
                let base = row as usize * bpr;
                let news: Vec<u32> = (0..shared).map(|_| self.alloc_page()).collect();
                let st2 = self.reqs.get_mut(&id).unwrap();
                for (j, b) in news.into_iter().enumerate() {
                    self.bt_host[base + j] = b;
                    st2.blocks[j] = b;
                }
                st2.shared = 0;
            }
            let stream = self.session.stream();
            self.zero_row_state(row, stream); // 重放从 pos 0 起步：递推状态必须归零
            self.reqs.get_mut(&id).unwrap().forwarded = 0;
        }
    }
}
