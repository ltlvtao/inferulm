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
//! - MTP 草稿层（可选，checkpoint 带 `mtp.*` 张量时装配）：单层 full
//!   attention 草稿头，入 = [pre_fc_norm_embedding(embed(t_{p+1})) ‖
//!   pre_fc_norm_hidden(h_p)] 过 fc，自有 KV 池 + 因果自注意（见 MtpDev）
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

use wein_engine::{Executor, SampleParams, StepError};
use wein_kernel::cuda::driver::CuBatchOp;
use wein_kernel::cuda::{DevBuf, Event, GraphExec, Stream};
use wein_kernel::ops::{
    AddOp, ArgmaxRowsOp, Conv1dDwOp, ConvStateRollOp, DeltanetOp, EmbedOp, GemmOp, KvScatterOp,
    MulOp, PagedAttnOp, RmsNormGatedOp, RmsNormZcOp, RopePtOp, SampleRowsOp, SigmulOp, SiluOp,
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
/// 投机草稿的单轮上限：k 步草稿 + 1 个待喂位必须落在绝对位置上限内
/// （MAX_POS=256）且 MTP 前向批 ≤ T_MAX；k ≤ 8 是两者下的保守上界
/// （每次草稿 = 一次 MTP kernel 序，k=8 = 8 次 launch，仍远小于主步成本）
const SPEC_K_MAX: u32 = 8;

// ── 自适应 k 策略参数（`WEIN_SPEC_ADAPTIVE=0` 关；见 `spec_policy_step`）──
// 阈值来自 R=16 的 k 扫描（qwen_spec_probe）：接受率 39%/30%/22% 对应
// k=1/2/3 的 tok/s 0.70/0.72/0.65 —— 即 k=2 是"接受率 ~30% 时的最优档"，
// k=1 与 k=3 在其两侧几乎并列。于是：
//   EMA ≥ 0.45  → k=3（接受率高过扫描点时，第 3 个草稿的边际收益翻正：
//                       期望出 token 1+3a vs 1+2a，a=0.45 即 2.35 vs 2.0）
//   0.30..0.45   → k=2（扫描实测的最优带 ≈30%）
//   < 0.30       → k=1（低接受时草稿成本主导，只赌第一个）
//   连续 3 轮 < 0.15 → k=0（这些请求的草稿纯亏：连第一个都基本不中，
//                       继续投机等于每步多付 k 次草稿前向 + 一条链行验证）
const SPEC_EMA_ALPHA: f32 = 0.3;
const SPEC_K_ON: f32 = 0.45;
const SPEC_K_MID: f32 = 0.30;
const SPEC_OFF_RATIO: f32 = 0.15;
const SPEC_OFF_RUNS: u32 = 3;
/// 关投机后的再探间隔（步）：k=0 期间不产生轮次 ⇒ EMA 冻结，必须有探针
/// 才能"回升重开"（用 k=1 试一轮，仍不行就再关）。首探间隔取小：短请求
/// （R=1 一轮 2~3 token，整段 15~30 步）等不起 8 步——实测固定 8 步的探针
/// 让 R=1 的自适应从 1.43× 掉到 1.06×（大半个段都在 k=0 空转）。反复被判
/// "该关"的请求按 2→4→8→16 退避（见 `spec_policy_step` 的 `spec_off_tries`）
const SPEC_PROBE_STEPS: u32 = 2;

/// 冷启动 k：EMA 未知时先按中档探（k=1 出 token 太少、k=3 草稿成本最高）
const SPEC_K_COLD: u32 = 2;
/// 关投机后"搬回块表行"的耐性（连续 k=0 的步数）：搬一次 25.2MB，只为让
/// 该请求的纯 decode 步重回图档。抖动中的请求（刚关又开）永远等不到这个
/// 步数 ⇒ 不白搬；稳定关掉的请求 ~8 步后搬一次
const SPEC_ANCHOR_IDLE: u32 = 8;

/// KV 页池页数（全注意力层共用一个池容量）
const NUM_PAGES: u32 = 512;
const BLOCK: u32 = 16;

/// 对外可查的 KV 池几何（块大小 / 页数）：cli/serve 侧要按同一把尺子装配
/// 水位准入模型——这里导出让"exec 的池"与"engine 的容量模型"不各写一份魔数
pub const KV_BLOCK: u32 = BLOCK;
pub const KV_PAGES: u32 = NUM_PAGES;
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

/// MTP 草稿链条数上限（块表行 = 并发草稿会话数；`MtpItem::kv_page_row` 索引）。
/// 每条活跃投机请求在 `set_spec` 时领一个槽（`ReqState::mtp_chain`），
/// 请求完成/`mtp_reset` 时归还——多链同批前向的 KV 页必须按槽分账
const MTP_CHAINS: usize = 16;

/// MTP（多 token 预测）草稿层权重 + 专属 scratch。checkpoint 无 `mtp.*`
/// 张量时整体 None，主链路逐位不变。schema 对齐 vLLM `qwen3_5_mtp.py`：
/// `e = pre_fc_norm_embedding(embed(t_{p+1}))`、`hh = pre_fc_norm_hidden(h_p)`、
/// `x = fc([e ‖ hh])`、`y = x + attn(ln1(x))`、`z = y + mlp(ln2(y))`、
/// `logits = rms_norm_zc(z, norm) @ embedᵀ`（tie 的 embed 兼作 lm_head）
struct MtpDev {
    /// [2560, 5120]：入序 = [pre_fc_norm_embedding ‖ pre_fc_norm_hidden]
    fc: DevBuf<'static>,
    pre_e: DevBuf<'static>, // [2560] pre_fc_norm_embedding
    pre_h: DevBuf<'static>, // [2560] pre_fc_norm_hidden
    /// 单层 full attention + SwiGLU MLP（与主模型层同构，复用 LayerDev）
    layer: LayerDev,
    norm: DevBuf<'static>, // [2560] mtp.norm
    /// 草稿层 K/V 池在 k_pool/v_pool 中的下标（主模型 8 个 full 层之后）
    kv: usize,
    /// 上一步 hidden 的行 gather 落点 [T_MAX,2560]（fc 输入的 hidden 半）
    hid: DevBuf<'static>,
    /// fc 输入拼接 [T_MAX,5120]（行 = [e ‖ hh]）
    cat: DevBuf<'static>,
}

/// MTP 草稿前向的一项输入：链行 + 链内位置 + 上一步 hidden 行 + 下一 token
#[derive(Debug, Clone, Copy)]
pub struct MtpItem {
    /// 草稿链的块表行（每链独立 KV 页表；因果上下文 = 该链 0..=pos）
    pub kv_page_row: u32,
    /// 链内位置：page_attn/scatter 的 KV 槽位 + 因果上下文上界。
    /// 主模型里它与 rope 位置同尺；草稿链（`spec_draft`）里两者分开——
    /// 槽位用链内局部下标，rope 用绝对位置（见 `mtp_draft_rope`）
    pub pos: u32,
    /// 上一步前向的 hidden 行号（`last_hidden` 的切片尺）。草稿批里 =
    /// 该链的槽号（`scr.x` 的第 `mtp_chain` 行：本轮显式写入的 hidden）
    pub h: u32,
    /// 位置 p+1 的 token（embed 输入）
    pub next_tok: u32,
}

/// MTP 前向输入束（每项一行的 H2D 小表 + 草稿链块表）
struct MtpIn<'a> {
    ids: &'a DevBuf<'a>,
    /// KV 槽位/因果位置（scatter + paged_attn）
    pos: &'a DevBuf<'a>,
    /// rope 角度位置（可与 `pos` 不同：草稿链槽位局部、rope 绝对）
    rope_pos: &'a DevBuf<'a>,
    seqs: &'a DevBuf<'a>,
    hrows: &'a DevBuf<'a>,
    bt: &'a DevBuf<'a>,
    iq: &'a DevBuf<'a>,
    ig: &'a DevBuf<'a>,
}

/// 混合批里的一条投机验证链段（`forward_batch_mixed` 的输入）：批内
/// [row_start, row_start+cnt) 覆盖该链的 cnt = 草稿数+1 个位置，`head` =
/// 链首递推状态行（deltanet/conv 的 `spec==1` 逐位置落盘起点）。与普通
/// decode 行/prefill 行同批一次前向——一次 8.4GB 权重读覆盖全部链
#[derive(Debug, Clone, Copy)]
struct ChainSeg {
    /// 批内首行（rows 下标）
    row_start: u32,
    /// 行数 = 草稿数 + 1
    cnt: u32,
    /// 链首 state 行（= 该 span 的 row_id）
    head: u32,
}

/// 一步内一条投机链的批内落点（草稿已定、链行已占、前向未发）
struct ChainJob {
    id: ReqId,
    /// 块表行（KV 页账；= 请求的 row）
    bt: u32,
    /// 链首 state 行
    head: u32,
    /// 链内 token 数 = 草稿数 + 1
    cnt: u32,
    /// 验证前向的批内首行（由 step 在拼批时填）
    row: u32,
    /// 绝对位置起点（= 请求的 forwarded）
    start: u32,
    drafts: Vec<u32>,
    /// 借出的 arena 行（连续段；轮末零化归还）
    borrowed: Vec<u32>,
}

/// q‖gate 拆分下标（单位 = head_dim 宽的 gather 行）：HF 在 view(T,-1,2·hd)
/// 上 chunk(2) ⇒ 每头 [q ‖ gate] 交错，q 取偶数行、gate 取奇数行
/// （qs8 消融：按"前 nqh 行 q / 后 nqh 行 gate"错读 → 一致率 45%→4.2%）
fn qg_split_ids(t: usize, nqh: usize) -> (Vec<u32>, Vec<u32>) {
    let mut q = Vec::with_capacity(t * nqh);
    let mut g = Vec::with_capacity(t * nqh);
    for tt in 0..t {
        let base = tt * 2 * nqh;
        for hh in 0..nqh {
            q.push((base + 2 * hh) as u32);
            g.push((base + 2 * hh + 1) as u32);
        }
    }
    (q, g)
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
    // —— 设备端采样器（top-k/top-p）：逐行参数 + RNG 槽状态，全部常驻 ——
    /// 逐行温度 [T_MAX] f32（0 = greedy → kernel 内走 argmax 分支）
    sm_temp: DevBuf<'static>,
    /// 逐行 top-k [T_MAX] u32（1..=64；0 = 全词表，主机已拒）
    sm_topk: DevBuf<'static>,
    /// 逐行 top-p [T_MAX] f32（(0,1]；(0,1) 之外按 1.0）
    sm_topp: DevBuf<'static>,
    /// 逐行种子 [T_MAX] u64（RNG 槽未初始化时由它派生流）
    sm_seed: DevBuf<'static>,
    /// 逐行 RNG 槽号 [T_MAX] u32（= 请求块表行；槽独占整条请求生命期）
    sm_slots: DevBuf<'static>,
    /// RNG 槽状态 [MAX_ROWS] u64：**图内读写**（图尾 sample_rows 前进它），
    /// 0 = 未初始化哨兵 → 由 seed 派生；准入新请求时按行零化（同 seed 可复现）
    sm_rng: DevBuf<'static>,
    /// 采样输出 [T_MAX] u32（0xFFFFFFFF = 参数非法兜底，正常不可达）
    sm_out: DevBuf<'static>,
    /// 逐 span 投机标志的常量缓冲（长度 T_MAX，覆盖任何 n_rows ≤ T_MAX）：
    /// 非投机路径全 0、投机链段全 1。**不每步分配/上传**——kernel 只按
    /// 段号取 `specs[r]`，常量内容一次写定（open 期 zero/upload）
    specs_zero: DevBuf<'static>,
    specs_one: DevBuf<'static>,
    /// 混合标志（普通段 0 + 链段 1 同现）的落点：全 T_MAX 覆写便于用
    /// `upload_on`（只前 n_rows 项被 kernel 读）
    specs_mix: DevBuf<'static>,
}


/// 投机验证一轮的结果（`Qwen35Executor::spec_verify`）。
/// - `accepted`：从第一个草稿起**连续**被接受的草稿数（贪心验证）
/// - `draft_tokens`：被接受的草稿（长度 = accepted；其后为拒绝段）
/// - `target_tokens`：链上每个位置的 argmax（长度 = drafts.len()+1）。
///   `target_tokens[i]` 是**位置 forwarded+i+1** 的预测 ⇒ `target_tokens[j]`
///   与草稿 drafts[j] 比位即验证；`target_tokens[accepted]` 是接受后的
///   奖励 token（已推进为新的待喂位）
pub struct SpecRound {
    pub accepted: usize,
    pub draft_tokens: Vec<u32>,
    pub target_tokens: Vec<u32>,
}

struct ReqState {
    seq: Vec<u32>,
    forwarded: usize,
    /// 块表行（KV 页账）
    row: u32,
    /// 递推状态行（deltanet/conv 状态池）——默认 = `row`；投机验证链的行
    /// 从 arena 借（见 `spec_verify`），两者分离后不得混用
    state_row: u32,
    blocks: Vec<u32>,
    /// blocks[..shared] 是 radix 命中来的共享块（重放/驱逐语义分叉点）
    shared: usize,
    /// 活跃期落过的边界快照 (块, 槽)——每个栅格边界一份，retain 时逐块
    /// 移交（块进树则槽随块，否则槽回池）；抢占重放则全数作废
    snaps: Vec<(u32, u32)>,
    /// 投机草稿步数（0 = 关；`set_spec` 开）。默认 0 ⇒ 现有测试逐位不受影响
    spec_k: u32,
    /// 自适应 k（`set_spec` 默认开，`WEIN_SPEC_ADAPTIVE=0` 关）：策略接管
    /// `spec_k` 的取值，见 `spec_policy_step`
    spec_auto: bool,
    /// 接受率 EMA（accepted/drafts，α=0.3）。None = 冷启动（首轮实测直接当 EMA）
    spec_ema: Option<f32>,
    /// 连续低接受轮数（< `SPEC_OFF_RATIO` 计数到 `SPEC_OFF_RUNS` → 本段关投机）
    spec_low_run: u32,
    /// 关投机（k=0）后的再探倒计时（解一步减一；到 0 用 k=1 重试）
    spec_probe: u32,
    /// 连续 k=0 的 decode 步数（到 `SPEC_ANCHOR_IDLE` 才把状态搬回块表行）
    spec_idle: u32,
    /// 本段已被判"关投机"的次数——再探间隔按它退避（2/4/8/16 步），
    /// 免得在一个注定亏的请求上每 2 步烧一轮链验证 + 草稿
    spec_off_tries: u32,
    /// 草稿链槽（0..MTP_CHAINS，与 `mtp_bt` 的行一一对应；None = 未领）。
    /// 槽号 = `MtpItem::kv_page_row` + 草稿 hidden 在 `scr.x` 的行号：
    /// 同一批里多条链各走自己的 KV 页与 hidden 行，互不踩
    mtp_chain: Option<u32>,
    /// 位置 `forwarded-1` 的目标末层 hidden（一行 2560 bf16）：草稿循环的
    /// 起点 h_p。由产生该位置的 forward 现场落盘（prefill 步/普通步/spec_verify
    /// 的落定行）——主 scratch 的行空间逐级覆写，必须单独存
    hid: Option<DevBuf<'static>>,
    /// 设备端采样参数（默认 = greedy：temp ≤ 0 走原 argmax，不进采样 kernel）。
    /// seed 在 `set_sampling` 时定死（None → 全局递增计数器派生），同 seed
    /// 同 logits ⇒ 同 token 序列（qs14）
    smp: SampleParams,
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
    /// 采样版图（图尾 = argmax + sample_rows）：只在"本步有非 greedy 行"时
    /// 回放。greedy 步仍回放 `execs`——原图拓扑逐位不变，也零额外开销
    execs_smp: Vec<GraphExec<'static>>,
    /// 图路径前向计数（qg 系验收：证路径确实被走到）
    replays: u64,
    /// 其中走采样版图的次数（采样图确实被选中/复用的证据）
    smp_replays: u64,
}

/// eager 混合批的常驻输入表（T_MAX 容量；每步只上传用到的前缀）。
///
/// 为什么常驻：`forward_batch_mixed` 每步要喂 ids/pos/seqs/段表/token 侧
/// 索引/gather 拆分下标/块表共十几张小表。原先每张 `alloc_from_slice` =
/// 一次 cuMemAlloc + H2D + 步末 cuMemFree——R=16 k=3 实测 host 上传段
/// 0.27→2.60ms/步（`WEIN_STEP_PROFILE`），驱动分配开销主导，跟流量无关。
/// 换成常驻 T_MAX 缓冲 + `upload_prefix_on`（只搬前缀），分配账归零。
///
/// 与图档无关：FULL_DECODE 图走自己那套 `fx_*` 固化表 + `in_*` 覆写缓冲
/// （`QGraphMode`），本结构只在 eager 路径出现——图捕获区间内不得分配，
/// 这些缓冲也必须 open 期就位（见 `open_with_tune`）
struct MixIn {
    ids: DevBuf<'static>,   // [T_MAX]
    pos: DevBuf<'static>,   // [T_MAX]
    seqs: DevBuf<'static>,  // [T_MAX]
    start: DevBuf<'static>, // [T_MAX] 段表（段数 ≤ 行数）
    count: DevBuf<'static>, // [T_MAX]
    rid: DevBuf<'static>,   // [T_MAX]
    trow: DevBuf<'static>,  // [T_MAX] conv tap 源状态行
    tpos: DevBuf<'static>,  // [T_MAX] conv tap 段内局部位
    /// gather 拆分下标（full q‖gate）：[T_MAX × nqh]
    iq: DevBuf<'static>,
    ig: DevBuf<'static>,
    /// linear qkv 拆分下标：[T_MAX]
    ilq: DevBuf<'static>,
    ilk: DevBuf<'static>,
    ilv: DevBuf<'static>,
    /// 全块表镜像 [MAX_ROWS × bpr]（页保证先于上传）
    bt: DevBuf<'static>,
    /// argmax 采样行表 [T_MAX]
    samples: DevBuf<'static>,
    /// 设备端采样器的采样行表 [T_MAX]（只含非 greedy 行——每条按行下标）
    sm_rows: DevBuf<'static>,
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
    /// 逐 span 投机标志（长度 = 段数）：0 = 普通单行状态滚动；1 = 投机
    /// 验证链（逐位置落盘，见 `forward_batch_spec`）。常量缓冲由 exec 常驻
    /// （非投机全 0 / 链段全 1）——每步零分配、零上传
    specs: &'a DevBuf<'a>,
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
    /// MTP 草稿层（checkpoint 无 `mtp.*` 张量 → None，主链路逐位不变）
    mtp: Option<MtpDev>,
    /// MTP 草稿链块表（行 = MtpItem::kv_page_row，与主模型 bt_host 分账）
    mtp_bt: Vec<u32>,
    /// 已分配页的草稿链（链行 → 块串）；mtp_reset 全数归还页池
    mtp_chains: HashMap<u32, Vec<u32>>,
    /// 空闲草稿链槽（0..MTP_CHAINS）——`set_spec` 领、请求退场/`mtp_reset` 还
    mtp_free: Vec<u32>,
    /// 上一步前向的批行数（last_hidden 的行数；mtp_draft 的 h 行上界）
    last_t: usize,
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
    /// eager 混合批的常驻输入表（每步前缀上传；与图档的 `in_*` 分开）
    mix: MixIn,
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
    /// 投机读数（测试/探针）：轮数、提出的草稿总数、接受总数、退化次数
    spec_rounds: u64,
    spec_drafts: u64,
    spec_accepted: u64,
    spec_fallbacks: u64,
    /// 一步里同时验证的最大链数（批化证据：≥2 即"多链共享一次前向"）
    spec_chain_max: u32,
    /// 自适应 k 策略总开关（open 期取 `WEIN_SPEC_ADAPTIVE`，默认开；
    /// `set_spec_policy` 运行期可翻）
    spec_policy: bool,
    /// 自适应 k 的切换次数（读数：策略确实在动）
    spec_k_switches: u64,
    /// 换行零拷贝的行账读数：链行分配次数 / 其中需要拷入的（链首 ≠ state 行）
    spec_chain_allocs: u64,
    spec_copyins: u64,
    /// 步内耗时分解（探针读数，总毫秒）：草稿（全部链合计，k 次 MTP 前向）、
    /// 混合前向（含普通行 + 链行）、落定（接受 + state 拷贝 + arena 零化）
    spec_ms_draft: f64,
    spec_ms_forward: f64,
    spec_ms_finish: f64,
    /// 逐请求采样参数（提交时下发，准入时搬进 `ReqState`；退场即清）。
    /// 默认不在表里 = greedy（历史逐位行为）
    sample_cfg: HashMap<ReqId, SampleParams>,
    /// seed 缺省时的派生计数器（单调递增）：不同请求互不相关，同请求的
    /// seed 一旦定死就不再变（同 seed ⇒ 同序列）
    seed_seq: u64,
    /// 请求级失败表（`Executor::take_failures` 的载体）：页池耗尽、行耗尽、
    /// 位置/vocab 越界都是**记录**而不是 panic——panic 会让 engine 任务整体
    /// 消失（在飞请求全 Aborted、之后 submit 全失败 = 服务假死）
    failures: Vec<(ReqId, StepError)>,
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

        // ---- MTP 草稿层（顶级前缀 `mtp.`，不带 model.language_model.）----
        // 15 张量；checkpoint 无 `mtp.fc.weight` → None（主链路逐位不变）。
        // K/V 池是第 9 对（索引 = 主模型 8 个 full 层之后），full_i 语义不动
        let mtp = if store.names().iter().any(|n| *n == "mtp.fc.weight") {
            if cfg.mtp_num_hidden_layers != 1 {
                return Err(format!(
                    "checkpoint 带 MTP 张量，config mtp_num_hidden_layers={}（须为 1）",
                    cfg.mtp_num_hidden_layers
                ));
            }
            let ml = "mtp.layers.0.";
            let layer = LayerDev {
                ln1: up(&format!("{ml}input_layernorm.weight"))?,
                attn: AttnDev::Full(FullDev {
                    qg: up(&format!("{ml}self_attn.q_proj.weight"))?,
                    k: up(&format!("{ml}self_attn.k_proj.weight"))?,
                    v: up(&format!("{ml}self_attn.v_proj.weight"))?,
                    o: up(&format!("{ml}self_attn.o_proj.weight"))?,
                    qn: up(&format!("{ml}self_attn.q_norm.weight"))?,
                    kn: up(&format!("{ml}self_attn.k_norm.weight"))?,
                }),
                ln2: up(&format!("{ml}post_attention_layernorm.weight"))?,
                wg: up(&format!("{ml}mlp.gate_proj.weight"))?,
                wu: up(&format!("{ml}mlp.up_proj.weight"))?,
                wd: up(&format!("{ml}mlp.down_proj.weight"))?,
            };
            let kv = k_pool.len();
            let bytes = NUM_PAGES as usize * BLOCK as usize
                * (cfg.num_kv_heads * cfg.head_dim) as usize
                * 2;
            k_pool.push(pin!(gpu.alloc_bytes(bytes)));
            v_pool.push(pin!(gpu.alloc_bytes(bytes)));
            let hidden_bytes = cfg.hidden_size as usize * 2;
            Some(MtpDev {
                fc: up("mtp.fc.weight")?,
                pre_e: up("mtp.pre_fc_norm_embedding.weight")?,
                pre_h: up("mtp.pre_fc_norm_hidden.weight")?,
                layer,
                norm: up("mtp.norm.weight")?,
                kv,
                hid: pin!(gpu.alloc_bytes(hidden_bytes * T_MAX)),
                cat: pin!(gpu.alloc_bytes(hidden_bytes * 2 * T_MAX)),
            })
        } else {
            None
        };

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
            // 采样器逐行参数 + RNG 槽：**open 期一次分配**（图捕获区间内
            // 不许分配——gr5 守卫会 panic）。初值全 greedy（temp=0），
            // 图捕获就是"全 greedy 版"采样图，回放前主机覆写参数
            sm_temp: pin!(gpu.alloc_bytes(T_MAX * 4)),
            sm_topk: pin!(gpu.alloc_bytes(T_MAX * 4)),
            sm_topp: pin!(gpu.alloc_bytes(T_MAX * 4)),
            sm_seed: pin!(gpu.alloc_bytes(T_MAX * 8)),
            sm_slots: pin!(gpu.alloc_bytes(T_MAX * 4)),
            sm_rng: pin!(gpu.alloc_bytes(MAX_ROWS * 8)),
            sm_out: pin!(gpu.alloc_bytes(T_MAX * 4)),
            specs_zero: pin!(gpu.alloc_bytes(T_MAX * 4)),
            specs_one: pin!(gpu.alloc_bytes(T_MAX * 4)),
            specs_mix: pin!(gpu.alloc_bytes(T_MAX * 4)),
        };
        // 逐 span 标志常量：一次写定（非投机 = 全 0；投机链段 = 全 1）
        scr.specs_zero.zero_on(session.stream());
        scr.specs_one.upload_on(&vec![1u32; T_MAX], session.stream());
        // 采样器开局态：temp=0（greedy）、top_k=64、top_p=1、槽号 0、种子 0、
        // RNG 槽全 0（= 未初始化哨兵 → 由图尾 kernel 用 seed 派生）。图捕获
        // 期不写这些缓冲（回放前主机才覆写），这里只是给"无采样步"一个
        // 确定初值：temp=0 ⇒ 图尾采样节点等价 argmax，且**不碰 RNG**
        scr.sm_temp.zero_on(session.stream());
        scr.sm_topk.upload_on(&vec![64u32; T_MAX], session.stream());
        scr.sm_topp.upload_on(&vec![1.0f32; T_MAX], session.stream());
        scr.sm_seed.zero_on(session.stream());
        scr.sm_slots.zero_on(session.stream());
        scr.sm_rng.zero_on(session.stream());
        scr.sm_out.zero_on(session.stream());
        // eager 混合批常驻输入表（open 期一次分配；运行期只前缀覆写）
        let nqh = cfg.num_heads as usize;
        let mix = MixIn {
            ids: pin!(gpu.alloc_bytes(u32b(T_MAX))),
            pos: pin!(gpu.alloc_bytes(u32b(T_MAX))),
            seqs: pin!(gpu.alloc_bytes(u32b(T_MAX))),
            start: pin!(gpu.alloc_bytes(u32b(T_MAX))),
            count: pin!(gpu.alloc_bytes(u32b(T_MAX))),
            rid: pin!(gpu.alloc_bytes(u32b(T_MAX))),
            trow: pin!(gpu.alloc_bytes(u32b(T_MAX))),
            tpos: pin!(gpu.alloc_bytes(u32b(T_MAX))),
            iq: pin!(gpu.alloc_bytes(u32b(T_MAX * nqh))),
            ig: pin!(gpu.alloc_bytes(u32b(T_MAX * nqh))),
            ilq: pin!(gpu.alloc_bytes(u32b(T_MAX))),
            ilk: pin!(gpu.alloc_bytes(u32b(T_MAX))),
            ilv: pin!(gpu.alloc_bytes(u32b(T_MAX))),
            bt: pin!(gpu.alloc_bytes(u32b(MAX_ROWS * bpr_of()))),
            samples: pin!(gpu.alloc_bytes(u32b(T_MAX))),
            sm_rows: pin!(gpu.alloc_bytes(u32b(T_MAX))),
        };
        session.stream().sync(); // 状态池开局零化落定
        let (lin_row, conv_row) = (lin_row_bytes(&cfg), conv_row_bytes(&cfg));
        // 行状态零化的常驻零源（arena 行归还是一批 memcpy 的源头）
        session.stream().sync();
        let q_scale = 1.0f32 / (cfg.head_dim as f32).sqrt();
        let prof = if std::env::var("WEIN_STEP_PROFILE").is_ok() {
            Some(StepProf::new(gpu))
        } else {
            None
        };
        // 自适应 k 策略默认开（WEIN_SPEC_ADAPTIVE=0 关：固定 k 的 A/B 基线）
        let spec_policy = std::env::var("WEIN_SPEC_ADAPTIVE").map_or(true, |v| v != "0");
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
            mtp,
            mtp_bt: vec![0u32; MTP_CHAINS * bpr_of()],
            mtp_chains: HashMap::new(),
            mtp_free: (0..MTP_CHAINS as u32).rev().collect(),
            last_t: 0,
            lin_state,
            conv_state,
            k_pool,
            v_pool,
            cos,
            sin,
            scr,
            mix,
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
            spec_rounds: 0,
            spec_drafts: 0,
            spec_accepted: 0,
            spec_fallbacks: 0,
            spec_chain_max: 0,
            spec_policy,
            spec_k_switches: 0,
            spec_chain_allocs: 0,
            spec_copyins: 0,
            spec_ms_draft: 0.0,
            spec_ms_forward: 0.0,
            spec_ms_finish: 0.0,
            // 采样参数表默认空 = 全 greedy；派生种子从 1 起（0 留作"未设"）
            sample_cfg: HashMap::new(),
            seed_seq: 1,
            failures: Vec::new(),
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
            // 图档恒 spec==0：图内 rid 与块表行同命名空间，投机链不进图
            // （spec_verify 直走 eager，见其注释）——常驻全 0 标志缓冲即可
            specs: &self.scr.specs_zero,
        };
        let voc = self.cfg.vocab_size as u32;
        let mut execs = Vec::with_capacity(buckets.len());
        let mut execs_smp = Vec::with_capacity(buckets.len());
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
            // 采样版图：同一条前向 + argmax + 图内 sample_rows（RNG 状态常驻
            // 设备缓冲、图内前进；参数缓冲的初值全 greedy ⇒ 捕获期就是一次
            // 等价 argmax 的采样，回放前主机整桶覆写参数）。
            // 捕获区间内零分配/零同步（gr5 守卫）——sample_rows 只是一次
            // launch，参数/scratch/输出缓冲全部 open 期就位
            stream.begin_capture();
            self.run_forward(b, b, &inp);
            self.session.table().argmax_rows.run(
                &self.scr.logits, &fx_rows, &self.scr.am_scratch, &self.scr.am_out, voc, b, stream,
            );
            self.session.table().sample_rows.run(
                &self.scr.logits,
                &fx_rows,
                &self.scr.sm_slots,
                &self.scr.sm_temp,
                &self.scr.sm_topk,
                &self.scr.sm_topp,
                &self.scr.sm_seed,
                &self.scr.sm_rng,
                &self.scr.sm_out,
                voc,
                b,
                stream,
            );
            let exec_smp: GraphExec<'static> = pin!(stream.end_capture().instantiate());
            execs_smp.push(exec_smp);
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
            execs_smp,
            replays: 0,
            smp_replays: 0,
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

    /// 其中走**采样版图**（图尾含 sample_rows）的回放次数——greedy 步不进它
    pub fn graph_smp_replays(&self) -> u64 {
        self.graph.as_ref().map(|g| g.smp_replays).unwrap_or(0)
    }

    /// 单 token 前向探针（不动 forwarded——q5 逐步对拍用）：返回该位全词表
    /// logits（bf16 存储 → f32）。页池耗尽返回空表（探针不 panic）
    pub fn forward_one(&mut self, id: ReqId, token: u32, pos: u32) -> Vec<f32> {
        let row = self
            .reqs
            .get(&id)
            .map(|s| s.row)
            .unwrap_or_else(|| panic!("forward_one：未知请求 {id}"));
        let state_row = self.reqs.get(&id).map(|s| s.state_row).unwrap();
        if self.ensure_pages(id, pos as usize).is_err() {
            return Vec::new();
        }
        self.forward_batch(&[(row, state_row, token, pos)], &[]);
        let bits: Vec<u16> = self.scr.logits.read_to_vec();
        let voc = self.cfg.vocab_size as usize;
        bits[..voc].iter().map(|&b| reference::bf16_to_f32(b)).collect()
    }

    /// 直跑完整生成（测试/CLI 便利面；不经过调度器）。
    /// 语义与 step 合同一致：追尾段采样，之后逐 token decode。
    /// 容量类失败（超位置上限/页池耗尽）→ **提前收尾返回已生成部分**，
    /// 不 panic（serve 路径不经过这里，但同一把纪律：执行器不炸进程）
    pub fn generate(&mut self, prompt: &[u32], max_tokens: u32) -> Vec<u32> {
        self.reset();
        self.admitted(1, prompt);
        // admitted 的失败（vocab 越界/行耗尽）记在失败表里——直跑路径自己消化
        self.failures.clear();
        if max_tokens == 0 || !self.reqs.contains_key(&1) {
            return Vec::new();
        }
        if prompt.len() >= MAX_POS {
            return Vec::new();
        }
        let max_tokens = max_tokens.min((MAX_POS - prompt.len()) as u32);
        let mut out = Vec::new();
        let mut first_tok = None;
        loop {
            let Some(st) = self.reqs.get(&1) else { break };
            let (start, len) = (st.forwarded, st.seq.len());
            if start == len {
                break;
            }
            let take = (len - start).min(T_MAX);
            let (row, srow) = (st.row, st.state_row);
            let batch: Vec<(u32, u32, u32, u32)> = (0..take)
                .map(|j| (row, srow, self.reqs[&1].seq[start + j], (start + j) as u32))
                .collect();
            self.reqs.get_mut(&1).unwrap().forwarded += take;
            if self.ensure_pages(1, start + take - 1).is_err() {
                break; // 页池耗尽：提前收尾（不 panic）
            }
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
            let (row, srow, pos) = (st.row, st.state_row, st.forwarded);
            let tok_in = st.seq[pos];
            self.reqs.get_mut(&1).unwrap().forwarded += 1;
            if self.ensure_pages(1, pos).is_err() {
                break; // 页池耗尽：提前收尾（不 panic）
            }
            let next = self.forward_rows(&[(row, srow, tok_in, pos as u32)], &[0], true, &[])[0];
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
        // 草稿链页一并归还（MTP 链属请求态——reset 后从零开始）
        self.mtp_reset();
    }

    /// 投机验证一轮（MTP 草稿验证的基础原语）。**eager only**：图档里 rid 与
    /// 块表行同命名空间（图固化的输入表），表达不出逐位置链——spec 不进图，
    /// 这里直走 `forward_batch_mixed`（`pure_decode` 分流在图档里，不受影响）。
    ///
    /// 位置对应：待喂 token = `seq[forwarded]`（位置 forwarded）。本轮前向
    /// cnt = drafts.len()+1 个链位置 i ⇒ 绝对位置 forwarded+i、token
    /// `[seq[forwarded], drafts[0..]]`。**第 i 行的 argmax 是位置
    /// forwarded+i+1 的预测**，即 `target_tokens[i]`；草稿 drafts[j] 是给位置
    /// forwarded+1+j 的，于是贪心验证 = `drafts[j] == target_tokens[j]` 从
    /// j=0 起连续成立 → accepted = a；消费链 token 0..=a（1+a 个），新待喂
    /// token = `target_tokens[a]`（被接受位之后的奖励 token）。
    ///
    /// state 行：链占连续 cnt+1 行 head..=head+cnt，第 i 个 token 读 head+i、
    /// 写 head+i+1（deltanet/conv kernel 的 spec==1）⇒ 行 head+j = "处理完前
    /// j 个 token"的递推状态。接受 a 个 ⟺ 落定行 head+1+a——**零拷贝回滚**
    /// （不撤销/不重算/不拍快照），该行 D2D 拷回请求的 `state_row`。
    ///
    /// 容量：链需 cnt+1 连续行（K=3 ⇒ 5 行）。优先拿请求自己的 state 行当
    /// 链首（省一次 25.2MB 拷入），否则从 `next_row` 尾部连续切、再退化到
    /// `free_rows` 连续段。128 行池在 K=3 下 = 8 条并发链 × 5 行 + 88 行普通
    /// 请求；链行轮末零化归还。分配不到连续段 → None（调用方退化为普通步）。
    ///
    /// KV 语义：本轮把位置 forwarded..forwarded+cnt-1 的 KV 全写进请求块表
    /// 页（页已 ensure 到该范围）。被拒位（forwarded+1+a..）的 KV 是垃圾但
    /// **无害**：因果注意力只读 pos' ≤ 当前 pos 的槽，后续步从
    /// forwarded+a+1 起算并覆写同一槽位；块账户（blocks）只多不少，无需回缩。
    pub fn spec_verify(&mut self, id: ReqId, drafts: &[u32]) -> Option<SpecRound> {
        // 越界草稿钳到词表内（投机只是提示：被钳的位必然验证不过，输出不受
        // 影响）；不再 assert——一次构造错误不该炸掉整个 engine 任务
        let voc = self.cfg.vocab_size;
        debug_assert!(
            drafts.iter().all(|&t| t < voc),
            "草稿 token 越界（vocab {voc}）"
        );
        let drafts: Vec<u32> = drafts.iter().map(|&t| t.min(voc - 1)).collect();
        self.spec_commit_batch(&[(id, drafts)]).pop().flatten()
    }

    /// 一批链的验证 + 接受（批化的 `spec_verify`）：**一条目标前向**覆盖全部
    /// 链的 cnt 行（每条链一个 spec==1 段），argmax 回读后按链切分、逐链算
    /// 接受、逐链拷回落定 state 行。返回与 `items` 对齐的结果；None = 该链
    /// 占不到连续 state 行（调用方退化，见 `step`）。
    pub fn spec_commit_batch(
        &mut self,
        items: &[(ReqId, Vec<u32>)],
    ) -> Vec<Option<SpecRound>> {
        let (mut jobs, fails) = self.spec_chain_alloc(items);
        if jobs.is_empty() {
            return items.iter().map(|_| None).collect();
        }
        let mut rows: Vec<(u32, u32, u32, u32)> = Vec::new();
        let mut samples: Vec<u32> = Vec::new();
        let segs = self.spec_chain_fill(&mut jobs, &mut rows, &mut samples);
        let target = self.forward_batch_mixed(&rows, &samples, &segs, &[]);
        let rounds = self.spec_finish(jobs, &target);
        let mut it = rounds.into_iter();
        items
            .iter()
            .map(|(id, _)| {
                if fails.contains(id) {
                    return None;
                }
                it.next().map(|(_, r)| r)
            })
            .collect()
    }

    /// 把链作业摊成批行（chain 首行 = 待喂 token，其余 = 草稿 token），并
    /// 同时产出段表；`job.row` 被写成该链在 `rows` 里的首行号（`stash_hidden`
    /// 要从这条链自己的 scratch 行取 hidden）。
    fn spec_chain_fill(
        &self,
        jobs: &mut [ChainJob],
        rows: &mut Vec<(u32, u32, u32, u32)>,
        samples: &mut Vec<u32>,
    ) -> Vec<ChainSeg> {
        let mut segs: Vec<ChainSeg> = Vec::with_capacity(jobs.len());
        for job in jobs.iter_mut() {
            let st = &self.reqs[&job.id];
            let pending = st.seq[st.forwarded];
            job.row = rows.len() as u32;
            rows.push((job.bt, job.head, pending, job.start));
            for i in 1..job.cnt {
                rows.push((job.bt, job.head + i, job.drafts[i as usize - 1], job.start + i));
            }
            segs.push(ChainSeg { row_start: job.row, cnt: job.cnt, head: job.head });
            samples.extend((0..job.cnt).map(|i| job.row + i));
        }
        segs
    }

    /// 链行分配：逐条链占 cnt+1 连续 state 行（链首 = 请求自己的 state 行
    /// 优先，免 25.2MB 拷入）。返回 (链作业, 失败请求)——失败 = 请求不在 /
    /// 尾位置超 `MAX_POS` / 池里无连续行。
    fn spec_chain_alloc(
        &mut self,
        items: &[(ReqId, Vec<u32>)],
    ) -> (Vec<ChainJob>, Vec<ReqId>) {
        let mut jobs = Vec::with_capacity(items.len());
        let mut fails = Vec::new();
        // 本步所有漂移链的拷入段攒在一起（一次发行，见
        // `push_copy_row_state_ops`）
        let mut copy_ops: Vec<CuBatchOp> = Vec::new();
        for (id, drafts) in items {
            let cnt = drafts.len() as u32 + 1;
            let Some(st) = self.reqs.get(id) else {
                fails.push(*id);
                continue;
            };
            let (bt, mut state_row, start) = (st.row, st.state_row, st.forwarded);
            if cnt as usize > T_MAX || start + cnt as usize > MAX_POS {
                fails.push(*id);
                continue;
            }
            // 漂移再锚定：换行落定让 state_row 每轮往前滑（1+accepted 行），
            // 滑过的旧行都还进了 free_rows——落在新 state_row **后面**，自己
            // 用不上（case 1 只借前面）。于是行空间一边被头前进吃掉、一边
            // 堆着身后的空闲行，几轮后 case 1 必落空 → 每轮退 case 2/3 的
            // 25.2MB 拷入（R=16 实测 4.97 次/步 ≈ 126MB/步，把换行零拷贝
            // 省下的钱又花回去）。这里先试"块表行当链首"：一次 25.2MB 把
            // 状态搬回 `row`，之后若干轮又能零拷入（顺带恢复
            // `state_row == row` ⇒ 该请求的纯 decode 步重回图档）
            let n = cnt + 1;
            if state_row != bt
                && !self.chain_head_free(state_row, n)
                && self.chain_head_free(bt, n)
            {
                self.spec_reanchor(*id);
                state_row = bt;
            }
            // 页保证先做（失败 → 该链本步退化，行还没借，不会泄漏）
            if self.ensure_pages(*id, start + cnt as usize - 1).is_err() {
                fails.push(*id);
                continue;
            }
            let Some((head, borrowed)) = self.spec_chain_rows(state_row, n) else {
                fails.push(*id);
                continue;
            };
            self.spec_chain_allocs += 1;
            if head != state_row {
                // 拷入：链首不是请求自己的 state 行（前沿/空闲段分配）——每链
                // 25.2MB D2D。换行零拷贝下这只是"漂移到池边"时的再锚定，不再
                // 是每步必付的账（旧实现是每步每链一次拷出）
                self.spec_copyins += 1;
                self.push_copy_row_state_ops(state_row, head, &mut copy_ops);
            }
            jobs.push(ChainJob {
                id: *id,
                bt,
                head,
                cnt,
                row: 0,
                start: start as u32,
                drafts: drafts.clone(),
                borrowed,
            });
        }
        if !copy_ops.is_empty() {
            // 流序：拷入排在本步前向之前（前向的链首行读的就是这份状态）
            self.session.stream().memcpy_batch(&copy_ops);
        }
        (jobs, fails)
    }

    /// 投机段的落定：一条前向的 argmax 按链切分 → 贪心接受 → **换行**
    /// 落定 → 发射簿记（seq/forwarded）→ arena 行归还。返回每链的
    /// SpecRound（与 `jobs` 同序）。
    ///
    /// **落定 = 换行零拷贝（不搬 25.2MB）**：链行空间 `[head, head+cnt]` 里
    /// 第 `head+j` 行 = "处理完前 j 个链 token 后的递推状态"（deltanet/conv
    /// kernel 的 `spec==1` 逐位置落盘），于是接受 a 个草稿后的落定行
    /// `head+1+a` **本来就是**请求要的状态行 ⇒ 直接把 `state_row` 指过去，
    /// 把它从 `job.borrowed` 里剔除（留在请求名下），其余借出行连同旧
    /// `state_row` 一起还给 `free_rows`。
    ///
    /// 旧实现是 24 lin + 24 conv 段 `memcpy_batch` 拷回旧 state_row：R=16 时
    /// 768 段/步、实测只跑到 82GB/s（4.85ms/步，驱动开销主导）——投机负收益
    /// 的第二大项。块表行 `st.row` 完全不动（KV 页账与 state 行早已分离）。
    ///
    /// 零化纪律：还回 `free_rows` 的行**不在这里零化**（`admitted` 领用时零，
    /// 见那里的论证）——链内只有链首行会被**读**，而链首恒等于请求自己的
    /// `state_row`（`spec_chain_rows` case 1）或已拷入状态的行（case 2/3），
    /// 其余行 deltanet/conv 都是先写后读，不需要零态起点。
    fn spec_finish(
        &mut self,
        jobs: Vec<ChainJob>,
        targets: &[u32],
    ) -> Vec<(ReqId, SpecRound)> {
        let t_fin = Instant::now();
        let n_jobs = jobs.len() as u32;
        self.spec_chain_max = self.spec_chain_max.max(n_jobs);
        let mut out = Vec::with_capacity(jobs.len());
        let mut cursor = 0usize;
        let mut give_back: Vec<u32> = Vec::new();
        for job in jobs {
            let target: Vec<u32> = targets[cursor..cursor + job.cnt as usize].to_vec();
            cursor += job.cnt as usize;
            // 贪心验证：连续命中即接受（target[j] = 位置 forwarded+1+j 的预测）
            let mut accepted = 0usize;
            while accepted < job.drafts.len() && job.drafts[accepted] == target[accepted] {
                accepted += 1;
            }
            // 草稿循环起点：接受位链行 = 位置 start+accepted = 新的 forwarded-1。
            // 该位置的 hidden 在本轮批的第 row+accepted 行（第 j 行 = 位置
            // start+j；**不是 row+1+accepted**——多一个 +1 就是把下一位置的
            // hidden 喂给草稿层，语义错位一拍，接受率 43%→16%）。
            // scr.x 此刻仍是本轮链行的 hidden（后续任何 forward 都会覆写）
            self.stash_hidden(job.id, job.row as usize + accepted);
            // ── 落定：认领落定行（零拷贝）──
            let landed = job.head + 1 + accepted as u32;
            let mut freed_state = None;
            if let Some(st) = self.reqs.get_mut(&job.id) {
                if st.state_row != landed {
                    // 旧 state 行归还——但请求首次投机时 state_row 就是块表行
                    // （KV 页账的行号），那一行**不能**还：它仍归本请求的
                    // 块表用，还了会被别的请求领走（页表串台/批内行号撞车）
                    if st.state_row != st.row {
                        freed_state = Some(st.state_row);
                    }
                    st.state_row = landed;
                }
                st.seq.truncate(job.start as usize + 1);
                st.seq.extend_from_slice(&job.drafts[..accepted]);
                st.seq.push(target[accepted]); // 奖励 token（位置 forwarded+a+1）
                st.forwarded = job.start as usize + 1 + accepted;
            }
            if std::env::var("WEIN_SPEC_DEBUG").is_ok() {
                let hits: Vec<u8> =
                    (0..job.drafts.len()).map(|j| (job.drafts[j] == target[j]) as u8).collect();
                eprintln!("[spec] drafts={:?} target={:?} hits={:?} accepted={accepted}", job.drafts, target, hits);
            }
            // 读数（接受率账）：本轮草稿数 / 接受数
            self.spec_rounds += 1;
            self.spec_drafts += job.drafts.len() as u64;
            self.spec_accepted += accepted as u64;
            // 自适应 k：本轮实测接受率喂 EMA，策略改下一轮的 k
            self.spec_policy_step(job.id, accepted, job.drafts.len());
            // 归还 arena 行（**不在这里零化**）：借出行恒为连续段，且只有链首
            // 行会被读（= 请求自己的旧 state 行或拷入的状态行；其余行
            // deltanet/conv 都是先写后读）。零化改在 `admitted` 领到
            // free_rows 行时做——链每步都要借同一批行，归还即零化等于每步
            // 白扫 R×cnt×25.2MB（R=16 时 1.6GB/步）。
            // 落定行（`landed`）留下当请求的 state_row，不进归还表
            if let Some(r) = freed_state {
                give_back.push(r); // 旧 state 行：已被 landed 取代
            }
            for &r in &job.borrowed {
                if r != landed {
                    give_back.push(r);
                }
            }
            out.push((
                job.id,
                SpecRound {
                    accepted,
                    draft_tokens: job.drafts[..accepted].to_vec(),
                    target_tokens: target,
                },
            ));
        }
        self.free_rows.extend(give_back);
        // 无拷贝 ⇒ 无需 sync：混合前向已在 argmax 回读处同步过，落定行由
        // 那条前向的 kernel 写好；后续任何对该行的读（D2H 对拍/下一步
        // kernel）都在同一流上保序
        self.spec_ms_finish += prof_ms(t_fin.elapsed());
        out
    }

    /// 自适应 k 策略一拍：喂一轮 (accepted, drafts) 进 EMA，重算本请求的
    /// `spec_k`（阈值与理由见文件头 `SPEC_*` 常量）。只对 `spec_auto` 的
    /// 请求生效（普通 `spec_k` 固定档 / 测试用的 `spec_verify` 不受影响）。
    ///
    /// 关闭（k=0）时顺手把状态行搬回块表行（`spec_reanchor`）：关投机后
    /// 请求回到"state_row == row"的不变量，纯 decode 步才能继续走图档。
    fn spec_policy_step(&mut self, id: ReqId, accepted: usize, drafts: usize) {
        if drafts == 0 {
            return;
        }
        let ratio = accepted as f32 / drafts as f32;
        let (new_k, changed) = {
            let Some(st) = self.reqs.get_mut(&id) else { return };
            if !st.spec_auto {
                return;
            }
            st.spec_ema = Some(match st.spec_ema {
                Some(e) => SPEC_EMA_ALPHA * ratio + (1.0 - SPEC_EMA_ALPHA) * e,
                None => ratio, // 冷启动：首轮实测直接当 EMA
            });
            let ema = st.spec_ema.unwrap_or(0.0);
            // 关闭判据要"瞬时踩低"且"EMA 也在 k=1 档以下"两条同时成立：
            // k=1 时 ratio 只有 0/1 两档，接受率 ~0.4 的请求连丢 3 个的概率
            // 也有 ~18%（实测 R=1 的自适应因此 14 步里关投机 8 次、从
            // 1.43× 掉到 1.06×）；EMA 在 0.30 以上说明"只是运气差"，不关
            st.spec_low_run =
                if ratio < SPEC_OFF_RATIO && ema < SPEC_K_MID { st.spec_low_run + 1 } else { 0 };
            let mut k = if ema >= SPEC_K_ON {
                3
            } else if ema >= SPEC_K_MID {
                2
            } else {
                1
            };
            if st.spec_low_run >= SPEC_OFF_RUNS {
                k = 0; // 本段关投机（草稿纯亏），由 `step` 的探针定时重开
                st.spec_off_tries = st.spec_off_tries.saturating_add(1);
                st.spec_low_run = 0;
                // 再探退避：2 → 4 → 8 → 16 步。反复关=这个请求确实不该投机
                // （R=16 的 4B 模型就是：主前向已被 16 行摊薄，草稿纯亏），
                // 每 2 步一探等于每步都在付一轮链验证的钱
                st.spec_probe = SPEC_PROBE_STEPS << (st.spec_off_tries - 1).min(3);
            } else {
                st.spec_off_tries = 0;
            }
            let k = k.min(SPEC_K_MAX);
            let changed = k != st.spec_k;
            st.spec_k = k;
            (k, changed)
        };
        if changed {
            self.spec_k_switches += 1;
        }
        if std::env::var("WEIN_SPEC_POLICY_DEBUG").is_ok() {
            let (ema, low) = self
                .reqs
                .get(&id)
                .map(|st| (st.spec_ema.unwrap_or(0.0), st.spec_low_run))
                .unwrap_or((0.0, 0));
            eprintln!(
                "[spec-policy] id={id} accepted={accepted}/{drafts} ema={ema:.3} low_run={low} → k={new_k}{}",
                if changed { "（切换）" } else { "" }
            );
        }
    }

    /// 把请求的递推状态从投机链借来的行搬回块表行（`state_row → row`），
    /// 恢复 `state_row == row` 不变量——图档路径（`forward_graph`）把 rid 与
    /// 块表行当同一套固化表，表达不出分离的 state 行。只在关投机时搬一次
    /// （25.2MB），之后该请求的纯 decode 步照旧可回放图。
    fn spec_reanchor(&mut self, id: ReqId) {
        let Some(st) = self.reqs.get(&id) else { return };
        if st.state_row == st.row {
            return;
        }
        let (src, dst) = (st.state_row, st.row);
        self.copy_row_state(src, dst);
        if let Some(st) = self.reqs.get_mut(&id) {
            st.state_row = dst;
        }
        // 归还的行不零化（`admitted` 领用时零）；拷贝与后续任何零化同流保序
        self.free_rows.push(src);
    }

    /// 一步里同时验证的最大链数（批化读数：≥2 = 多条链共享一次前向）
    pub fn spec_chain_batch_max(&self) -> u32 {
        self.spec_chain_max
    }

    /// 测试/诊断：请求的草稿链槽号（= 它 MTP KV 的 `mtp_bt` 页行，0..MTP_CHAINS）。
    /// 多链同批前向时每条链必须占不同槽，否则草稿 KV 互相覆盖
    pub fn spec_chain_of(&self, id: ReqId) -> Option<u32> {
        self.reqs.get(&id).and_then(|st| st.mtp_chain)
    }

    /// 投机草稿开关（`k` = 单轮草稿数，0 = 关；> SPEC_K_MAX 时 clamp）。
    /// 默认关闭 ⇒ 现有测试/路径逐位不变。hidden 落点（一行 2560 bf16）按需
    /// 分配一次；显存分配不得落在图捕获区间（本方法只在运行期调用）。
    /// `k > 0` 同时领一个草稿链槽（`MTP_CHAINS` 个池子——多条链同批前向时
    /// 各走自己的 `mtp_bt` 行与 KV 页）；槽耗尽 → 不领，该请求本步退化。
    pub fn set_spec(&mut self, id: ReqId, k: u32) {
        let k = k.min(SPEC_K_MAX);
        let h = self.cfg.hidden_size as usize;
        if !self.reqs.contains_key(&id) {
            return;
        }
        // 采样与投机互斥：链验证逐位置断言 greedy（`spec_verify`），采样行
        // 进链会让"草稿接受"判定失真。非 greedy 请求开投机 = 无操作
        // （`set_sampling` 侧对反向顺序报错，两侧都挡住）
        if k > 0 && self.reqs.get(&id).is_some_and(|st| !st.smp.is_greedy()) {
            return;
        }
        if k == 0 {
            self.mtp_release(id);
        }
        // 只在**真要用**槽时才领（k=0 或已有槽时 pop 会漏槽：MTP_CHAINS 个
        // 槽耗尽后所有投机请求都退化）
        let need_slot = k > 0 && self.reqs[&id].mtp_chain.is_none();
        let slot = if need_slot { self.mtp_free.pop() } else { None };
        let policy = self.spec_policy;
        let buf = self.session.gpu();
        let st = self.reqs.get_mut(&id).unwrap();
        st.spec_auto = policy && k > 0;
        st.spec_ema = None;
        st.spec_low_run = 0;
        st.spec_probe = 0;
        st.spec_idle = 0;
        st.spec_off_tries = 0;
        // 策略接管时本参数只是"上界"：冷启动按中档探（`SPEC_K_COLD`），
        // 之后由 `spec_policy_step` 依 EMA 逐轮定档
        st.spec_k = if st.spec_auto { k.min(SPEC_K_COLD) } else { k };
        if k > 0 {
            if st.hid.is_none() {
                st.hid = Some(pin!(buf.alloc_bytes(h * 2)));
            }
            if st.mtp_chain.is_none() {
                st.mtp_chain = slot;
            }
        }
        // 关投机：把状态行搬回块表行，恢复图档可回放的形态
        if k == 0 {
            self.spec_reanchor(id);
        }
    }

    /// 自适应 k 策略开关（`WEIN_SPEC_ADAPTIVE` 的运行期覆盖；探针用它做
    /// "固定 k vs 策略自选"的同进程 A/B）。关掉后新开的投机请求走固定 k，
    /// 已在跑的请求保留当前档位
    pub fn set_spec_policy(&mut self, on: bool) {
        self.spec_policy = on;
    }

    /// 测试/探针：请求当前的投机草稿数（策略下 = 自适应选出的档）
    pub fn spec_k_of(&self, id: ReqId) -> Option<u32> {
        self.reqs.get(&id).map(|st| st.spec_k)
    }

    /// 行账读数（换行零拷贝的漂移/复用证据）：(链行分配次数, 其中拷入次数,
    /// 状态池行数, 当前 next_row, 当前 free_rows 长度)。
    /// 拷入次数 → 0 即稳态零拷贝；`free_rows` 长度在多轮之间不涨即不泄漏
    pub fn spec_row_stats(&self) -> (u64, u64, usize, u32, usize) {
        (self.spec_chain_allocs, self.spec_copyins, self.row_capacity as usize, self.next_row, self.free_rows.len())
    }

    /// 自适应 k 读数：(当前每请求 k 的分布（升序）, 切换次数)。
    /// 分布按活跃请求计——0 = 被策略关掉的、1/2/3 = 策略当前所选档
    pub fn spec_k_stats(&self) -> (Vec<(u32, usize)>, u64) {
        let mut hist: Vec<(u32, usize)> = Vec::new();
        for st in self.reqs.values() {
            if !st.spec_auto {
                continue;
            }
            match hist.iter_mut().find(|(k, _)| *k == st.spec_k) {
                Some((_, c)) => *c += 1,
                None => hist.push((st.spec_k, 1)),
            }
        }
        hist.sort_unstable();
        (hist, self.spec_k_switches)
    }

    /// 草稿链槽归还（页回池、块表行清零、槽回空闲表）。请求退场/关投机/
    /// `mtp_reset` 三条路径共用——槽属请求态，与 `mtp_chains` 一一对应
    fn mtp_release(&mut self, id: ReqId) {
        let Some(st) = self.reqs.get_mut(&id) else { return };
        let Some(slot) = st.mtp_chain.take() else { return };
        self.mtp_drop_chain(slot);
        self.mtp_free.push(slot);
    }

    /// 丢掉槽 `slot` 的草稿链 KV（页回池 + 该 `mtp_bt` 行清零），槽本身不动。
    /// 抢占走这条：请求重放要从 pos 0 重来，旧链 KV 全部作废，但槽继续归它
    fn mtp_drop_chain(&mut self, slot: u32) {
        if let Some(blocks) = self.mtp_chains.remove(&slot) {
            for b in blocks {
                self.pool.release(b);
            }
            let base = slot as usize * self.bpr;
            for j in 0..self.bpr {
                self.mtp_bt[base + j] = 0;
            }
            self.pool.recycle();
        }
    }

    /// 单请求草稿（`spec_draft_batch` 的一元情形；qs9/探针的单链读数用）
    pub fn spec_draft(&mut self, id: ReqId, k: u32) -> Vec<u32> {
        self.spec_draft_batch(&[(id, k)]).pop().unwrap_or_default()
    }

    /// 一批链的草稿：**第 i 个草稿步 = 一次 MTP 前向覆盖全部活跃链**（items
    /// 每链一行），返回与 `reqs` 同序的草稿表；空 vec = 该请求本轮不可投机
    /// （无 MTP 权重 / 无链槽 / hidden 未落盘 / 链尾越界 / k=0）。
    ///
    /// 语义（MTP 的正确偏移，qs8 已验）：给定位置 p 的目标末层 hidden `h_p`
    /// 与位置 p+1 的 token，草稿层预测 p+2 的 token。于是
    /// - 起点 h = 位置 `forwarded-1` 的末层 hidden（`ReqState::hid`，由产生该
    ///   位置的 forward 现场落盘），tok = `seq[forwarded]`（待喂 token，位置
    ///   forwarded）⇒ 第 1 个草稿 = 位置 forwarded+1 的预测，正是
    ///   `spec_verify` 需要的 `drafts[0]`；
    /// - 自回归：第 i+1 步用第 i 步草稿层的**输出 hidden**（`run_mtp_forward`
    ///   里 `mtp.norm` 之后、logits 之前写进 `scr.n2` 的那份）当下一行 hidden，
    ///   token = 第 i 个草稿。
    ///
    /// 批化后成本 = k 次前向（每次只读一遍 MTP 层 + logits 权重），**与链数
    /// 无关**——v1 的"每请求 k 次单行 MTP 前向"在 R=16 时是 48 次权重读，
    /// 这是投机负收益的主项。
    ///
    /// **v1 明确不做草稿链前缀 KV**：每轮一条干净链（`mtp_reset` 语义 + 链内
    /// 槽位 0..k-1），paged_attn/scatter 的位置用链内局部下标、rope 用绝对位置
    /// （`run_mtp_forward` 的两个入参本就分开传，见 `mtp_draft_rope`）。理由 +
    /// 证据：qs8 的自注意消融（每项独立链、pos=0，只看得见自己）总一致率仍
    /// 45.0%，但逐位输出全变——草稿头主要吃 target hidden + next-token embed，
    /// 草稿链自注意对 top-1 一致率的贡献在噪声内。前缀 KV（把主链 KV 也给草稿
    /// 层看）是后续提升项，会抬高接受率。
    /// **这是近似：影响的是接受率，不是正确性**——草稿只是猜测，`spec_verify`
    /// 的贪心验证 + 逐位置 state 回滚保证输出分布与普通 decode 逐位相同。
    ///
    /// k 上限：`MAX_POS`（绝对位置）与 `T_MAX`（批）约束下取 ≤ SPEC_K_MAX = 8，
    /// 越界直接 clamp。链尾越界（`start + k + 1 > MAX_POS`）→ 返回空（不投机，
    /// 调用方退化为普通步）。
    ///
    /// 链间隔离：槽 c 的 hidden 落 `scr.x` 第 c 行、KV 走 `mtp_bt` 第 c 行
    /// （`kv_page_row = c`）——两条链共用一个槽就会互相踩 KV/页表。
    pub fn spec_draft_batch(&mut self, reqs: &[(ReqId, u32)]) -> Vec<Vec<u32>> {
        let t0 = Instant::now();
        let out = self.spec_draft_batch_impl(reqs);
        self.spec_ms_draft += prof_ms(t0.elapsed());
        out
    }

    fn spec_draft_batch_impl(&mut self, reqs: &[(ReqId, u32)]) -> Vec<Vec<u32>> {
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); reqs.len()];
        if self.mtp.is_none() || reqs.is_empty() {
            return out;
        }
        let h = self.cfg.hidden_size as usize;
        // 活跃链：槽 + 待喂 token + hidden + 位置余量都齐
        struct Act {
            idx: usize,
            slot: u32,
            start: u32,
            tok: u32,
            k: u32,
        }
        let mut act: Vec<Act> = Vec::with_capacity(reqs.len());
        for (j, &(id, k)) in reqs.iter().enumerate() {
            let k = k.min(SPEC_K_MAX);
            if k == 0 {
                continue;
            }
            let Some(st) = self.reqs.get(&id) else { continue };
            let (Some(slot), true) = (st.mtp_chain, st.hid.is_some()) else { continue };
            let Some(&tok) = st.seq.get(st.forwarded) else { continue };
            let start = st.forwarded as u32;
            let k = k.min(MAX_POS.saturating_sub(start as usize + 1) as u32);
            if k == 0 {
                continue;
            }
            act.push(Act { idx: j, slot, start, tok, k });
        }
        if act.is_empty() {
            return out;
        }
        // 各链 hidden 落自己的槽行（一次批量 D2D：R 行 × 2560×2B）
        {
            let mut ops: Vec<CuBatchOp> = Vec::with_capacity(act.len());
            for a in &act {
                let st = self.reqs.get(&reqs[a.idx].0).expect("活跃链的请求必在");
                let hid = st.hid.as_ref().expect("hid 已判存在");
                ops.push(CuBatchOp {
                    dst: self.scr.x.device_ptr() + (a.slot as usize * h * 2) as u64,
                    src: hid.device_ptr(),
                    size: h * 2,
                });
            }
            self.session.stream().memcpy_batch(&ops);
        }
        let kmax = act.iter().map(|a| a.k).max().unwrap_or(0);
        let mut cur: Vec<u32> = act.iter().map(|a| a.tok).collect();
        let mut got: Vec<Vec<u32>> = vec![Vec::new(); act.len()];
        for i in 0..kmax {
            // 本轮活跃：还没跑满 k 步的链
            let live: Vec<usize> = (0..act.len()).filter(|&j| act[j].k > i).collect();
            if live.is_empty() {
                break;
            }
            let items: Vec<MtpItem> = live
                .iter()
                .map(|&j| MtpItem {
                    kv_page_row: act[j].slot,
                    pos: i,                       // 链内槽位（干净链）
                    h: act[j].slot,               // 本轮写进 scr.x 第槽号行的 hidden
                    next_tok: cur[j],
                })
                .collect();
            // 绝对位置（旋转角）：与 v1 逐位同式（start+i，qs8 的 pos 语义）
            let rope: Vec<u32> = live.iter().map(|&j| act[j].start + i).collect();
            let res = self.mtp_draft_rope(&items, &rope);
            for (t, &j) in live.iter().enumerate() {
                got[j].push(res[t]);
                cur[j] = res[t];
            }
            if i + 1 == kmax {
                break;
            }
            // 下一轮 hidden = 本步草稿层的输出（`mtp.norm` 之后、logits 之前
            // 写进 `scr.n2` 的那份；A/B 过残差流 `scr.x`：接受率 21.6% vs
            // 19.7%，在噪声内——next 层进来还要过一遍 pre_h 归一化，缩放量
            // 被归一化吃掉）
            let mut ops: Vec<CuBatchOp> = Vec::new();
            for (t, &j) in live.iter().enumerate() {
                if act[j].k <= i + 1 {
                    continue;
                }
                ops.push(CuBatchOp {
                    dst: self.scr.x.device_ptr() + (act[j].slot as usize * h * 2) as u64,
                    src: self.scr.n2.device_ptr() + (t * h * 2) as u64,
                    size: h * 2,
                });
            }
            if !ops.is_empty() {
                self.session.stream().memcpy_batch(&ops);
            }
        }
        for (j, a) in act.iter().enumerate() {
            out[a.idx] = std::mem::take(&mut got[j]);
        }
        out
    }

    /// 位置 `forwarded-1` 的 hidden 落盘（草稿循环起点）：`row` = 该位置在
    /// `scr.x` 里的批行号（prefill/普通步的采样行，或 spec_verify 的接受位链行）
    fn stash_hidden(&self, id: ReqId, row: usize) {
        let h = self.cfg.hidden_size as usize;
        let Some(st) = self.reqs.get(&id) else { return };
        let Some(hid) = st.hid.as_ref() else { return };
        if row >= self.last_t {
            return;
        }
        hid.copy_range_on(0, &self.scr.x, row * h * 2, h * 2, self.session.stream());
    }

    /// 链行空间：要 n = cnt+1 连续状态行。优先用请求自己的 state 行当链首
    /// （免拷入），否则从 `next_row` 尾部连续切（未分配区恒空），再退化到
    /// `free_rows` 里找连续段。返回 (链首, 借出行表)——借出行轮末零化归还，
    /// 链首若就是 state 行则不在借出表里。None = 池里找不到连续 n 行。
    /// case 1 的判据（"`head` 当链首、后面 n-1 行都空闲"）：`spec_chain_rows`
    /// 的零拷入分配与 `spec_chain_alloc` 的再锚定决策共用同一套语义
    fn chain_head_free(&self, head: u32, n: u32) -> bool {
        if head + n > self.row_capacity {
            return false;
        }
        let owned: HashSet<u32> = self.reqs.values().map(|s| s.state_row).collect();
        let free = |r: u32| !owned.contains(&r) && (r >= self.next_row || self.free_rows.contains(&r));
        (1..n).all(|k| free(head + k))
    }

    fn spec_chain_rows(&mut self, state_row: u32, n: u32) -> Option<(u32, Vec<u32>)> {
        let cap = self.row_capacity;
        // 1) 请求自己的状态行当链首：占地连续、行数不变（最省，零拷入）
        if self.chain_head_free(state_row, n) {
            let mut borrowed = Vec::with_capacity(n as usize - 1);
            for r in state_row + 1..state_row + n {
                if let Some(p) = self.free_rows.iter().position(|&x| x == r) {
                    self.free_rows.swap_remove(p);
                }
                borrowed.push(r);
            }
            // 借来的行可能越过分配前沿（r >= next_row 判空的前提是"前沿外
            // 从未发放"）——前沿必须推过去，否则轮末归还进 free_rows 后，
            // 后续 admitted 的长路径会把同一行再发一次（双发=状态串台）
            self.next_row = self.next_row.max(state_row + n);
            return Some((state_row, borrowed));
        }
        // 2) 尾部连续切（next_row 之后从未分配）
        if self.next_row + n <= cap {
            let head = self.next_row;
            self.next_row += n;
            return Some((head, (head..head + n).collect()));
        }
        // 3) free_rows 连续段（排序扫）
        let mut fr = self.free_rows.clone();
        fr.sort_unstable();
        let mut lo = 0usize;
        while lo + n as usize <= fr.len() {
            let seg = &fr[lo..lo + n as usize];
            if seg[0] + n <= cap && seg.windows(2).all(|w| w[1] == w[0] + 1) {
                for &r in seg {
                    if let Some(p) = self.free_rows.iter().position(|&x| x == r) {
                        self.free_rows.swap_remove(p);
                    }
                }
                return Some((seg[0], seg.to_vec()));
            }
            lo += 1;
        }
        None
    }

    /// 行状态整行 D2D 位拷贝（24 lin + 24 conv 段一次批量发行；25.2MB/次）。
    /// 段表写法同 `restore_row_snapshot`：逐层小段 + 一次 cuMemcpyBatchAsync
    /// 攒一条"状态行 src → dst"的段表（24 lin + 24 conv = 48 段）。跨链攒进
    /// **同一个**段表再一次发行：逐链发行是 48 段 × 链数 次驱动调用（R=16
    /// 漂移步实测 ~4.2 链/步 ≈ 200 次调用 ≈ 1.3ms/步，纯驱动开销；D2D 数据
    /// 本身只要 ~0.07ms）。`spec_chain_alloc` 用它把一步内所有拷入合并成一次
    fn push_copy_row_state_ops(&self, src: u32, dst: u32, ops: &mut Vec<CuBatchOp>) {
        let (ro, wo) = (src as usize * self.lin_row, dst as usize * self.lin_row);
        for t in &self.lin_state {
            ops.push(CuBatchOp {
                dst: t.device_ptr() + wo as u64,
                src: t.device_ptr() + ro as u64,
                size: self.lin_row,
            });
        }
        let (ro, wo) = (src as usize * self.conv_row, dst as usize * self.conv_row);
        for t in &self.conv_state {
            ops.push(CuBatchOp {
                dst: t.device_ptr() + wo as u64,
                src: t.device_ptr() + ro as u64,
                size: self.conv_row,
            });
        }
    }

    /// 单条状态行拷贝（一次发行）——再锚定/测试路径用
    fn copy_row_state(&self, src: u32, dst: u32) {
        let mut ops: Vec<CuBatchOp> =
            Vec::with_capacity(self.lin_state.len() + self.conv_state.len());
        self.push_copy_row_state_ops(src, dst, &mut ops);
        self.session.stream().memcpy_batch(&ops);
    }

    /// 测试/诊断：请求的递推状态行号（spec_verify 之后 = 接受位落定行）
    pub fn state_row_of(&self, id: ReqId) -> Option<u32> {
        self.reqs.get(&id).map(|s| s.state_row)
    }

    /// 测试/诊断：请求的块表行号（KV 页账）；投机跑过之后与 `state_row`
    /// 分离——换行零拷贝只动 state_row，块表行全程不动
    pub fn block_row_of(&self, id: ReqId) -> Option<u32> {
        self.reqs.get(&id).map(|s| s.row)
    }

    /// 测试/诊断：空闲行表快照。行账守恒判据：
    /// `[0, next_row)` = 活请求行（块表行 ∪ 状态行）∪ free_rows，两者不相交
    pub fn spec_free_rows(&self) -> Vec<u32> {
        self.free_rows.clone()
    }

    /// 测试/诊断：读回第 `layer` 个线性层的状态行 `row`（bf16 位，u16）。
    /// 单行 1MB——按行区间拷临时缓冲再回读（整池回读是 128MB，禁）
    pub fn lin_state_row_bits(&self, layer: usize, row: u32) -> Vec<u16> {
        self.state_row_bits(&self.lin_state[layer], row, self.lin_row)
    }

    /// 测试/诊断：conv 层状态行（48KB/行）
    pub fn conv_state_row_bits(&self, layer: usize, row: u32) -> Vec<u16> {
        self.state_row_bits(&self.conv_state[layer], row, self.conv_row)
    }

    fn state_row_bits(&self, pool: &DevBuf, row: u32, row_bytes: usize) -> Vec<u16> {
        let tmp = self.session.gpu().alloc_bytes(row_bytes);
        let stream = self.session.stream();
        tmp.copy_range_on(0, pool, row as usize * row_bytes, row_bytes, stream);
        stream.sync();
        tmp.read_to_vec()
    }

    /// 页保证：请求的块表行长到覆盖 `upto_pos`（含）。
    ///
    /// 页池耗尽 = **可恢复失败**（旧实现是 `panic!("页池耗尽")` —— 一次就带走
    /// engine 任务）：已分配的新页照常挂上（请求随即终结，`retain` 统一归还），
    /// 返回 `Err(Capacity)` 由引擎把该请求以 `length` 结束。
    fn ensure_pages(&mut self, id: ReqId, upto_pos: usize) -> Result<(), StepError> {
        let need = upto_pos / BLOCK as usize + 1;
        if !matches!(self.reqs.get(&id), Some(st) if st.blocks.len() < need) {
            return Ok(());
        }
        let bpr = self.bpr;
        let row = self.reqs.get(&id).unwrap().row as usize;
        let have = self.reqs.get(&id).unwrap().blocks.len();
        // 先攒本地（alloc_page 可能驱逐 radix，须避开 st 的借用）
        let mut news: Vec<u32> = Vec::with_capacity(need - have);
        let mut short = false;
        for _ in have..need {
            match self.alloc_page() {
                Some(b) => news.push(b),
                None => {
                    short = true;
                    break;
                }
            }
        }
        let st = self.reqs.get_mut(&id).unwrap();
        for b in news {
            self.bt_host[row * bpr + st.blocks.len()] = b;
            st.blocks.push(b);
        }
        if short {
            return Err(StepError::Capacity(format!(
                "页池耗尽（{} 页，请求需 {} 页）",
                self.pool.total(),
                need
            )));
        }
        Ok(())
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

    /// 页分配（耗尽时 LRU 驱逐 radix 腾页——槽随块联动回收）。
    /// 仍无页 → `None`（调用方转成 `StepError::Capacity`，**不 panic**）
    fn alloc_page(&mut self) -> Option<u32> {
        if self.pool.free_now() == 0 {
            let freed = self.radix.evict_blocks(&mut self.pool, 8);
            for b in freed {
                if let Some(s) = self.snap_of_block.remove(&b) {
                    self.snap_free.push(s);
                }
            }
            self.pool.recycle();
        }
        self.pool.alloc()
    }

    /// 记一次请求级失败（去重：同一请求同一步只留第一条——后续处理都基于
    /// "该请求即将离场"的前提）
    fn fail(&mut self, id: ReqId, e: StepError) {
        if !self.failures.iter().any(|(k, _)| *k == id) {
            self.failures.push((id, e));
        }
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

    /// 批前向（eager）：rows = (块表行, 状态行, token, pos)——块表行管 KV 页，
    /// 状态行管 deltanet/conv 状态池（两者默认同值，投机链下分离）
    fn forward_batch(&mut self, rows: &[(u32, u32, u32, u32)], samples: &[u32]) -> Vec<u32> {
        self.forward_batch_mixed(rows, samples, &[], &[])
    }

    /// 混合批前向（**一次 launch 覆盖普通行 + 任意条投机验证链**）：
    /// - 普通行/prefill 行：同请求连续行归一段（rid = 状态行，`specs = 0`）；
    /// - 链行（`chains`）：每条链一个段（`specs = 1`），rid = 链首 state 行，
    ///   token i 读链行 +i、写 +i+1（deltanet/conv kernel 的逐位置落盘）。
    ///   conv 的 tap 源行 = 链首行、段内局部位 = i（x 索引 `g-p+idx` 还原
    ///   成链内行）——写成"head+i / 0"会读错 tap 窗口（qs9 踩过）。
    ///
    /// 投机标志按**逐 span 数组**喂（长度 = 段数）：全 0/全 1 用 exec 常驻的
    /// 常量缓冲 `specs_zero`/`specs_one`（零分配零上传）；混合时拼
    /// `[0,0,..,1,..]` 覆写 `scr.specs_mix` 一次（2KB H2D）。
    ///
    /// 一请求一段的断言保留：链段内部 cnt 行共享同一块表行（同一段内共享
    /// 合法），不同段之间仍不许重复——否则 tap/递推上下文与后段断裂。
    fn forward_batch_mixed(
        &mut self,
        rows: &[(u32, u32, u32, u32)],
        samples: &[u32],
        chains: &[ChainSeg],
        smp: &[SampleParams],
    ) -> Vec<u32> {
        // context 线程局部：engine 任务可在 worker 间迁移，进设备路径先绑
        self.session.gpu().bind_thread();
        let t = rows.len();
        // 批形与位置上界是**内部账目合同**（serve 侧 ctx_limit 前置校验 + `step`
        // 的 killed 过滤保证不违约）。违约时宁可本步空转也不 panic 带走引擎：
        // 空步由调度器下轮重试（`step` 已把越界请求标成失败，不会死循环）
        if t == 0 || t > T_MAX || rows.iter().any(|&(_, _, _, pos)| (pos as usize) >= MAX_POS) {
            eprintln!("[exec] 前向批合同违约（t={t}，上限 {T_MAX}/{MAX_POS}），本步作废");
            return Vec::new();
        }
        self.last_t = t;
        // ---- span 构造：普通段（一请求一段）+ 链段（一链一段）----
        let mut row_start: Vec<u32> = Vec::new();
        let mut row_count: Vec<u32> = Vec::new();
        let mut row_id: Vec<u32> = Vec::new();
        let mut span_spec: Vec<u32> = Vec::new();
        let mut span_bt: Vec<u32> = Vec::new();
        // conv 的 token 侧索引：tap 源状态行 + 段内局部位（tap 越段前取 state）
        let mut tok_row = Vec::with_capacity(t);
        let mut tok_pos = Vec::with_capacity(t);
        let (mut j, mut ci) = (0usize, 0usize);
        while j < t {
            if ci < chains.len() && chains[ci].row_start as usize == j {
                let ch = chains[ci];
                let n = ch.cnt as usize;
                assert!(n > 0 && j + n <= t, "链段越界（行 {j}+{n} > 批 {t}）");
                row_start.push(j as u32);
                row_count.push(ch.cnt);
                row_id.push(ch.head);
                span_spec.push(1);
                span_bt.push(rows[j].0);
                for i in 0..n {
                    tok_row.push(ch.head);
                    tok_pos.push(i as u32);
                }
                j += n;
                ci += 1;
                continue;
            }
            // 普通段：同请求（块表行相同）连续行——链段边界打断合并
            let bt = rows[j].0;
            let start = j;
            j += 1;
            while j < t
                && rows[j].0 == bt
                && !(ci < chains.len() && chains[ci].row_start as usize == j)
            {
                j += 1;
            }
            row_start.push(start as u32);
            row_count.push((j - start) as u32);
            row_id.push(rows[start].1);
            span_spec.push(0);
            span_bt.push(bt);
            for k in start..j {
                tok_row.push(rows[start].1);
                tok_pos.push((k - start) as u32);
            }
        }
        assert!(ci == chains.len(), "链段未对齐批行（余 {}）", chains.len() - ci);
        // 一请求一段：跨段同请求 = 前段 tap/递推上下文与后段断裂——合同违约
        let mut seen = HashSet::new();
        for &bt in &span_bt {
            assert!(seen.insert(bt), "批内请求 {bt} 的行不连续——调度合同违约");
        }
        let n_rows = row_start.len() as u32;
        self.spec_chain_max = self.spec_chain_max.max(chains.len() as u32);

        let (stream, tb) = (self.session.stream(), self.session.table());
        let prof_on = self.prof.borrow().is_some();
        let tw = Instant::now();
        let ids: Vec<u32> = rows.iter().map(|r| r.2).collect();
        let pos: Vec<u32> = rows.iter().map(|r| r.3).collect();
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
        // 常驻输入表前缀覆写（零分配：open 期一次分配，见 MixIn 注释）
        let mi = &self.mix;
        mi.ids.upload_prefix_on(&ids, stream);
        mi.pos.upload_prefix_on(&pos, stream);
        mi.seqs.upload_prefix_on(&seqs, stream);
        mi.bt.upload_prefix_on(&self.bt_host, stream);
        mi.start.upload_prefix_on(&row_start, stream);
        mi.count.upload_prefix_on(&row_count, stream);
        mi.rid.upload_prefix_on(&row_id, stream);
        mi.trow.upload_prefix_on(&tok_row, stream);
        mi.tpos.upload_prefix_on(&tok_pos, stream);
        mi.iq.upload_prefix_on(&ids_q, stream);
        mi.ig.upload_prefix_on(&ids_gate, stream);
        mi.ilq.upload_prefix_on(&ids_lq, stream);
        mi.ilk.upload_prefix_on(&ids_lk, stream);
        mi.ilv.upload_prefix_on(&ids_lv, stream);
        // 逐 span 标志：全 0/全 1 走常驻常量，混合时拼一次 T_MAX 长的表覆写
        let all_zero = span_spec.iter().all(|&s| s == 0);
        let all_one = span_spec.iter().all(|&s| s == 1);
        let specs_buf = if all_zero {
            &self.scr.specs_zero
        } else if all_one {
            &self.scr.specs_one
        } else {
            let mut mix = vec![0u32; T_MAX];
            mix[..span_spec.len()].copy_from_slice(&span_spec);
            self.scr.specs_mix.upload_on(&mix, stream);
            &self.scr.specs_mix
        };
        let inp = FwdIn {
            ids: &self.mix.ids,
            pos: &self.mix.pos,
            seqs: &self.mix.seqs,
            bt: &self.mix.bt,
            start: &self.mix.start,
            count: &self.mix.count,
            rid: &self.mix.rid,
            trow: &self.mix.trow,
            tpos: &self.mix.tpos,
            iq: &self.mix.iq,
            ig: &self.mix.ig,
            ilq: &self.mix.ilq,
            ilk: &self.mix.ilk,
            ilv: &self.mix.ilv,
            specs: specs_buf,
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
        self.mix.samples.upload_prefix_on(samples, stream);
        tb.argmax_rows.run(
            &self.scr.logits, &self.mix.samples, &self.scr.am_scratch, &self.scr.am_out,
            self.cfg.vocab_size as u32, s, stream,
        );
        // 采样行（temp > 0）另发 sample_rows；全 greedy 步**不发**（argmax
        // 结果原样用——接入前后的 greedy 逐位行为由此结构性不变）
        let smp_sel = self.launch_sample_rows(rows, samples, smp, stream);
        self.prof_end();
        stream.sync();
        let tr = Instant::now();
        let mut out: Vec<u32> = self.scr.am_out.read_to_vec();
        if let Some(sel) = &smp_sel {
            // 采样行覆盖 argmax 位（顺序 = `samples` 序，恒等映射）
            let sm: Vec<u32> = self.scr.sm_out.read_to_vec();
            for (j, &si) in sel.iter().enumerate() {
                assert_ne!(
                    sm[j], 0xffff_ffff,
                    "sample_rows 参数非法（top_k=0 / 候选为空）——set_sampling 应已拒绝"
                );
                out[si as usize] = sm[j];
            }
        }
        if prof_on {
            self.prof_note_readback(tr);
            self.prof_finalize(tw);
        }
        out[..s as usize].to_vec()
    }

    /// 设备端采样尾巴（eager）：把非 greedy 采样行的参数/种子/RNG 槽逐行
    /// 前缀上传后发一次 `sample_rows`。返回 `Some(选中行在 samples 里的下标)`
    /// ——调用方在 `stream.sync()` 之后按它把 `sm_out` 覆盖进 argmax 结果；
    /// 全 greedy → `None`（零上传、零 kernel，逐位不变）。
    ///
    /// RNG 槽 = 该请求的块表行（准入时零化、图内前进、退场即废）。
    fn launch_sample_rows(
        &self,
        rows: &[(u32, u32, u32, u32)],
        samples: &[u32],
        smp: &[SampleParams],
        stream: &Stream,
    ) -> Option<Vec<u32>> {
        let mut sel: Vec<(u32, SampleParams, u32)> = Vec::new();
        for (i, &r) in samples.iter().enumerate() {
            let p = smp.get(i).copied().unwrap_or_default();
            if p.is_greedy() {
                continue;
            }
            sel.push((i as u32, p, rows[r as usize].0));
        }
        if sel.is_empty() {
            return None;
        }
        let s = sel.len() as u32;
        let idx: Vec<u32> = sel.iter().map(|x| samples[x.0 as usize]).collect();
        let temp: Vec<f32> = sel.iter().map(|x| x.1.temperature as f32).collect();
        let topk: Vec<u32> = sel.iter().map(|x| x.1.top_k).collect();
        let topp: Vec<f32> = sel.iter().map(|x| x.1.top_p as f32).collect();
        let seed: Vec<u64> = sel.iter().map(|x| x.1.seed.unwrap_or(0)).collect();
        let slots: Vec<u32> = sel.iter().map(|x| x.2).collect();
        self.mix.sm_rows.upload_prefix_on(&idx, stream);
        self.scr.sm_temp.upload_prefix_on(&temp, stream);
        self.scr.sm_topk.upload_prefix_on(&topk, stream);
        self.scr.sm_topp.upload_prefix_on(&topp, stream);
        self.scr.sm_seed.upload_prefix_on(&seed, stream);
        self.scr.sm_slots.upload_prefix_on(&slots, stream);
        self.session.table().sample_rows.run(
            &self.scr.logits,
            &self.mix.sm_rows,
            &self.scr.sm_slots,
            &self.scr.sm_temp,
            &self.scr.sm_topk,
            &self.scr.sm_topp,
            &self.scr.sm_seed,
            &self.scr.sm_rng,
            &self.scr.sm_out,
            self.cfg.vocab_size as u32,
            s,
            stream,
        );
        Some(sel.into_iter().map(|x| x.0).collect())
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
                        self.cfg.conv_kernel, n_rows, inp.specs, stream,
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
                        inp.count, inp.rid, n_rows, kh_n, vh_n, kd, vd, L2_EPS, inp.specs, stream,
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

    // ── MTP 草稿层（可选能力：checkpoint 带 `mtp.*` 张量时激活）──
    //
    // 接口分三层：
    //   `last_hidden`  上一步前向的末层 hidden（final norm 之前）→ 教师强制的 h_p
    //   `mtp_draft`    一批 (链行, 位置, hidden 行, 下一 token) 一次前向出每项 argmax
    //   `mtp_reset`    草稿链页归还（请求生命周期收尾）
    // 因果性靠 paged_attn 的 pos 语义 + 草稿链自有块表：项 (row, p) 只看该链
    // 位置 0..=p——同批先行项（更小 pos）的 KV 已在本前向的 scatter 落池。
    // 步时插桩不覆盖 MTP（profile 的"步"= 主模型前向；草稿层是另一条账）

    /// 上一步前向的末层 hidden（final norm 之前）：(行数, [T, hidden] bf16 位)。
    /// 行号与该步批内行号同尺——`MtpItem::h` 直接引用
    pub fn last_hidden(&self) -> (usize, Vec<u16>) {
        let h = self.cfg.hidden_size as usize;
        let t = self.last_t;
        let bits: Vec<u16> = self.scr.x.read_to_vec();
        (t, bits[..t * h].to_vec())
    }

    /// MTP 是否在场（checkpoint 带草稿层权重）
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// 前缀缓存开关（运行期覆盖 `WEIN_RADIX` 的 open 期默认）。关掉后后续
    /// 准入不做 match_prefix（无快照恢复 ⇒ 前向批形状可预测，便于逐位/逐
    /// token 对拍：qs10/qs11 与性能探针用）
    pub fn set_radix(&mut self, on: bool) {
        self.radix_on = on;
    }

    /// 采样参数校验 + 归一化（engine 提交路径与直连调用共用，纯函数）。
    ///
    /// - greedy（`temp ≤ 0` / NaN）→ 原样返回：top_k/top_p/seed 都不参与
    ///   （图尾走原 `argmax_rows`，**不进采样 kernel**——逐位合同不变）
    /// - `temperature` 非有限（+inf）→ Err（除以 inf 会把分布抹平成均匀，
    ///   静默接受等于给出一个用户没要求的分布）
    /// - `top_k = 0`（全词表）→ Err，**绝不静默退化成"取前 64"**：
    ///   设备端只有 top-k 路径，248K 词表要做 top-p 就得全序（等价全排序），
    ///   单请求单步的排序开销 >= 全词表扫描 × log，不可接受。`top_k=0 且
    ///   top_p<1`（"全词表 top-p"）是同一个坑的另一张皮，一并拒。
    /// - `top_k` 夹到 ≤ 64（kernel 的块内梯子深度上限）；`top_p` 落在
    ///   [0,1] 之外/0 → 1.0（与 kernel 兜底同语义：全保留）
    fn normalize_sampling(mut p: SampleParams) -> Result<SampleParams, String> {
        if p.is_greedy() {
            p.seed = None;
            return Ok(p);
        }
        if !p.temperature.is_finite() {
            return Err(format!("temperature 须为有限正数（实得 {}）", p.temperature));
        }
        if p.top_k == 0 {
            return Err(format!(
                "设备端采样器不支持全词表采样（top_k=0，top_p={}）：248K 词表全排序不可接受\
                 （top-p 需对全词表降序累加）；请给 1..=64 的 top_k",
                p.top_p
            ));
        }
        p.top_k = p.top_k.min(64);
        if !(p.top_p > 0.0 && p.top_p <= 1.0) {
            p.top_p = 1.0;
        }
        Ok(p)
    }

    /// 采样参数下发（`Executor::set_sampling` 的真身；提交时一次）。
    ///
    /// 入库两处：`sample_cfg`（准入前下发 ⇒ 建行时取用）与已建行的
    /// `ReqState::smp`（准入后改参直接生效）。改参/换 seed 会**归零该请求的
    /// RNG 槽**——新流只由新 seed 决定，不继承旧位置（同 seed ⇒ 同序列）。
    pub fn set_sampling(&mut self, id: ReqId, p: SampleParams) -> Result<(), String> {
        let mut p = Self::normalize_sampling(p)?;
        if p.seed.is_none() {
            p.seed = Some(self.seed_seq);
            self.seed_seq += 1;
        }
        if let Some(st) = self.reqs.get_mut(&id) {
            if !p.is_greedy() && st.spec_k > 0 {
                // 投机链验证是逐位置 argmax（`spec_verify` 的接受判定），
                // 采样行混进链=接受语义失效。拒绝而不是偷偷关掉投机
                return Err(format!(
                    "请求 {id} 在跑投机草稿链（spec_k={}）——采样与链验证互斥，先 set_spec(id, 0)",
                    st.spec_k
                ));
            }
            st.smp = p;
            if !p.is_greedy() {
                let row = st.row as usize;
                self.scr.sm_rng.zero_range_on(row * 8, 8, &self.session.stream());
            }
        }
        self.sample_cfg.insert(id, p);
        Ok(())
    }

    /// 请求当前的采样参数（测试/探针；未下发 = greedy）
    pub fn sampling_of(&self, id: ReqId) -> Option<SampleParams> {
        self.reqs.get(&id).map(|st| st.smp).or_else(|| self.sample_cfg.get(&id).copied())
    }

    /// 投机读数：(轮数, 提出的草稿总数, 接受总数, 退化次数)。接受率 =
    /// accepted / drafts（草稿越多、越准，一步出 token 越多）
    pub fn spec_stats(&self) -> (u64, u64, u64, u64) {
        (self.spec_rounds, self.spec_drafts, self.spec_accepted, self.spec_fallbacks)
    }

    pub fn spec_stats_reset(&mut self) {
        self.spec_rounds = 0;
        self.spec_drafts = 0;
        self.spec_accepted = 0;
        self.spec_fallbacks = 0;
        self.spec_chain_max = 0;
        self.spec_k_switches = 0;
        self.spec_chain_allocs = 0;
        self.spec_copyins = 0;
        self.spec_ms_draft = 0.0;
        self.spec_ms_forward = 0.0;
        self.spec_ms_finish = 0.0;
    }

    /// 测试/探针：步内耗时分解（总毫秒）——草稿（全部链合计）/ 混合前向 /
    /// 落定（接受 + state 拷贝 + arena 零化）。累计自 `spec_stats_reset`
    pub fn spec_timing_ms(&self) -> (f64, f64, f64) {
        (self.spec_ms_draft, self.spec_ms_forward, self.spec_ms_finish)
    }

    /// 草稿链页归还（行清零；页回池；全部槽回空闲表）。链属请求态——请求
    /// 退场/重放时调；`reset` 后从零开始（请求的 `mtp_chain` 一并摘掉）
    pub fn mtp_reset(&mut self) {
        for (_, blocks) in std::mem::take(&mut self.mtp_chains) {
            for b in blocks {
                self.pool.release(b);
            }
        }
        self.mtp_bt.iter_mut().for_each(|x| *x = 0);
        for st in self.reqs.values_mut() {
            st.mtp_chain = None;
        }
        self.mtp_free = (0..MTP_CHAINS as u32).rev().collect();
        self.pool.recycle();
    }

    /// 页保证：草稿链 `row` 的块表行长到覆盖 `pos`（含）。
    /// 页池耗尽 → `Err(Capacity)`（调用方把该链的草稿作废，本步退化为普通
    /// decode 行；绝不 panic）
    fn ensure_mtp_pages(&mut self, row: u32, pos: u32) -> Result<(), StepError> {
        let need = pos as usize / BLOCK as usize + 1;
        let have = self.mtp_chains.get(&row).map_or(0, |v| v.len());
        if have >= need {
            return Ok(());
        }
        // 先攒本地（alloc_page 可能驱逐 radix，须避开链表借用）
        let mut news: Vec<u32> = Vec::with_capacity(need - have);
        let mut short = false;
        for _ in have..need {
            match self.alloc_page() {
                Some(b) => news.push(b),
                None => {
                    short = true;
                    break;
                }
            }
        }
        let base = row as usize * self.bpr;
        let chain = self.mtp_chains.entry(row).or_default();
        for b in news {
            self.mtp_bt[base + chain.len()] = b;
            chain.push(b);
        }
        if short {
            return Err(StepError::Capacity(format!(
                "草稿链页耗尽（{} 页）",
                self.pool.total()
            )));
        }
        Ok(())
    }

    /// 草稿前向：一批项一次前向，返回每项 argmax（预测位置 p+2 的 token）。
    /// hidden 行号取自**上一步前向**（`last_hidden` 同尺）；同链项按 pos
    /// 升序给出即可（因果自注意不依赖批内序——scatter 先于 attn 落池）
    pub fn mtp_draft(&mut self, items: &[MtpItem]) -> Vec<u32> {
        let pos: Vec<u32> = items.iter().map(|i| i.pos).collect();
        self.mtp_draft_rope(items, &pos)
    }

    /// 同 `mtp_draft`，但 rope 位置单给：`MtpItem::pos` 管 KV 槽位/因果
    /// （草稿循环用链内局部下标 0..k-1），`rope_pos[i]` 管旋转角（绝对位置
    /// forwarded+i）。主模型前向里两者同尺，草稿链里必须分开——链 KV 是
    /// 每轮重写的干净段（局部槽位），而 token 的真实位置是绝对的。
    fn mtp_draft_rope(&mut self, items: &[MtpItem], rope_pos: &[u32]) -> Vec<u32> {
        let t = items.len();
        assert_eq!(rope_pos.len(), t, "rope 位置表须与项数同长");
        assert!(t > 0, "mtp_draft：空批");
        assert!(t <= T_MAX, "MTP 批 {t} 超前向缓冲上限 {T_MAX}");
        let voc = self.cfg.vocab_size;
        assert!(self.mtp.is_some(), "mtp_draft：checkpoint 无 MTP 权重");
        // 逐项上限检查从 assert 改为**钳制 + debug_assert**：投机只是提示，
        // 越界的草稿会被下一轮贪心验证拒绝（不产生错误输出），但绝不能因为
        // 一次越界 panic 掉整个 engine 任务（= 所有在飞请求一起死）
        for it in items {
            debug_assert!(
                (it.kv_page_row as usize) < MTP_CHAINS,
                "草稿链行 {} 超上限 {MTP_CHAINS}",
                it.kv_page_row
            );
            debug_assert!(
                (it.pos as usize) < MAX_POS,
                "MTP 位置 {} 超 paged_attn 上限 {MAX_POS}",
                it.pos
            );
            debug_assert!(
                (it.h as usize) < T_MAX,
                "hidden 行 {} 超 scratch 行数 {T_MAX}",
                it.h
            );
            debug_assert!(it.next_tok < voc, "MTP token {} 越界（vocab {voc}）", it.next_tok);
            // 链页耗尽 → 本轮草稿直接作废（返回全 0 占位，必然被验证拒绝）；
            // 该请求本步退化成普通 decode 行，同批其他请求不受影响
            if self.ensure_mtp_pages(it.kv_page_row, it.pos).is_err() {
                return vec![0u32; t];
            }
        }
        self.session.gpu().bind_thread();
        let gpu = self.session.gpu();
        let stream = self.session.stream();
        let ids: Vec<u32> = items.iter().map(|i| i.next_tok.min(voc - 1)).collect();
        let pos: Vec<u32> = items.iter().map(|i| i.pos.min(MAX_POS as u32 - 1)).collect();
        let seqs: Vec<u32> = items.iter().map(|i| i.kv_page_row).collect();
        let hrows: Vec<u32> = items.iter().map(|i| i.h.min(T_MAX as u32 - 1)).collect();
        let (iq, ig) = qg_split_ids(t, self.cfg.num_heads as usize);
        let (d_ids, d_pos, d_rope_pos, d_seqs, d_hrows, d_bt, d_iq, d_ig) = (
            gpu.alloc_from_slice(&ids),
            gpu.alloc_from_slice(&pos),
            gpu.alloc_from_slice(rope_pos),
            gpu.alloc_from_slice(&seqs),
            gpu.alloc_from_slice(&hrows),
            gpu.alloc_from_slice(&self.mtp_bt),
            gpu.alloc_from_slice(&iq),
            gpu.alloc_from_slice(&ig),
        );
        let inp = MtpIn {
            ids: &d_ids,
            pos: &d_pos,
            rope_pos: &d_rope_pos,
            seqs: &d_seqs,
            hrows: &d_hrows,
            bt: &d_bt,
            iq: &d_iq,
            ig: &d_ig,
        };
        self.run_mtp_forward(t as u32, &inp);
        // 每项一个草稿 token（设备端 argmax，只回读 t 个 u32）
        let rows: Vec<u32> = (0..t as u32).collect();
        let d_rows = gpu.alloc_from_slice(&rows);
        self.session.table().argmax_rows.run(
            &self.scr.logits,
            &d_rows,
            &self.scr.am_scratch,
            &self.scr.am_out,
            voc,
            t as u32,
            stream,
        );
        stream.sync();
        let out: Vec<u32> = self.scr.am_out.read_to_vec();
        out[..t].to_vec()
    }

    /// MTP 单层前向 kernel 序（embed gather → 双 pre-norm → cat → fc →
    /// 单层 full attention（自有池 + 因果）→ SwiGLU MLP → final norm + logits）。
    /// 主模型 scratch 复用：n1/o/h2/x/n2 + g/u/act/m/d + attn/agated/qg/qh/gate/kh/vh
    fn run_mtp_forward(&self, t32: u32, inp: &MtpIn) {
        let stream = self.session.stream();
        let tb = self.session.table();
        let eps = self.cfg.rms_eps;
        let (h, ii, voc) =
            (self.cfg.hidden_size, self.cfg.inter_size, self.cfg.vocab_size);
        let w = self.mtp.as_ref().expect("run_mtp_forward：无 MTP 权重");
        let AttnDev::Full(a) = &w.layer.attn else { panic!("MTP 层须为 full attention") };
        let (qh_n, kvh, hd) =
            (self.cfg.num_heads, self.cfg.num_kv_heads, self.cfg.head_dim);
        let he = kvh * hd;

        // ① 两个 pre-norm（各 2560）：e = norm_embed(embed(t_{p+1}))，
        //    hh = norm_hidden(h_p)——hidden 行先按行 gather 成紧排 [t,2560]
        tb.embed.run(inp.hrows, &self.scr.x, &w.hid, h, t32, stream);
        tb.embed.run(inp.ids, &self.embed, &self.scr.n1, h, t32, stream);
        tb.rms_norm_zc.run(&self.scr.n1, &w.pre_e, &self.scr.o, t32, h, eps, stream);
        tb.rms_norm_zc.run(&w.hid, &w.pre_h, &self.scr.h2, t32, h, eps, stream);

        // ② cat 行 = [e ‖ hh]：rms_norm_zc 只写紧排行、embed gather 的宽与源
        //    行宽同尺——跨缓冲拼半行只能逐行 D2D（段表一次发行，非逐项 launch）
        let (lower, upper) = (&self.scr.o, &self.scr.h2);
        let half = h as usize * 2;
        let mut ops: Vec<CuBatchOp> = Vec::with_capacity(2 * t32 as usize);
        for j in 0..t32 as usize {
            let dst = w.cat.device_ptr() + (j * 2 * half) as u64;
            let src = (j * half) as u64;
            ops.push(CuBatchOp { dst, src: lower.device_ptr() + src, size: half });
            ops.push(CuBatchOp { dst: dst + half as u64, src: upper.device_ptr() + src, size: half });
        }
        stream.memcpy_batch(&ops);
        tb.gemm.run_wt(&w.cat, &w.fc, &self.scr.x, t32, h, 2 * h, stream);

        // ③ 单层 full attention（q‖gate 交错投影 → 零中心 q/k norm →
        //    partial rope（位置 = 链内 p）→ 自有池 scatter → 因果 GQA →
        //    门乘 → o_proj）
        tb.rms_norm_zc.run(&self.scr.x, &w.layer.ln1, &self.scr.n1, t32, h, eps, stream);
        tb.gemm.run_wt(&self.scr.n1, &a.qg, &self.scr.qg, t32, 2 * qh_n * hd, h, stream);
        tb.embed.run(inp.iq, &self.scr.qg, &self.scr.qh, hd, t32 * qh_n, stream);
        tb.embed.run(inp.ig, &self.scr.qg, &self.scr.gate, hd, t32 * qh_n, stream);
        tb.rms_norm_zc.run(&self.scr.qh, &a.qn, &self.scr.qh, t32 * qh_n, hd, eps, stream);
        tb.gemm.run_wt(&self.scr.n1, &a.k, &self.scr.kh, t32, he, h, stream);
        tb.gemm.run_wt(&self.scr.n1, &a.v, &self.scr.vh, t32, he, h, stream);
        tb.rms_norm_zc.run(&self.scr.kh, &a.kn, &self.scr.kh, t32 * kvh, hd, eps, stream);
        let rd = self.cfg.rot_dim();
        tb.rope_pt.run(&self.scr.qh, &self.cos, &self.sin, inp.rope_pos, &self.scr.qh, rd, hd, qh_n, t32, stream);
        tb.rope_pt.run(&self.scr.kh, &self.cos, &self.sin, inp.rope_pos, &self.scr.kh, rd, hd, kvh, t32, stream);
        tb.kv_scatter.run(
            &self.scr.kh, &self.scr.vh, inp.bt, inp.seqs, inp.pos, &self.k_pool[w.kv],
            &self.v_pool[w.kv], self.bpr as u32, BLOCK, he, t32, stream,
        );
        tb.paged_attn.run(
            &self.scr.qh, &self.k_pool[w.kv], &self.v_pool[w.kv], inp.bt, inp.seqs, inp.pos,
            &self.scr.attn, self.q_scale, self.bpr as u32, BLOCK, qh_n, kvh, hd, t32, stream,
        );
        tb.sigmul.run(&self.scr.attn, &self.scr.gate, &self.scr.agated, t32 * qh_n * hd, stream);
        tb.gemm.run_wt(&self.scr.agated, &a.o, &self.scr.o, t32, h, qh_n * hd, stream);

        // ④ 残差 + SwiGLU MLP（h2 此前是 hh 的临时落点，此处起作 y）
        tb.add.run(&self.scr.x, &self.scr.o, &self.scr.h2, t32 * h, stream);
        tb.rms_norm_zc.run(&self.scr.h2, &w.layer.ln2, &self.scr.n2, t32, h, eps, stream);
        tb.gemm.run_wt(&self.scr.n2, &w.layer.wg, &self.scr.g, t32, ii, h, stream);
        tb.gemm.run_wt(&self.scr.n2, &w.layer.wu, &self.scr.u, t32, ii, h, stream);
        tb.silu.run(&self.scr.g, &self.scr.act, t32 * ii, stream);
        tb.mul.run(&self.scr.act, &self.scr.u, &self.scr.m, t32 * ii, stream);
        tb.gemm.run_wt(&self.scr.m, &w.layer.wd, &self.scr.d, t32, h, ii, stream);
        tb.add.run(&self.scr.h2, &self.scr.d, &self.scr.x, t32 * h, stream);

        // ⑤ final norm + logits（tie：embed 兼作 lm_head 的 W[n,k]）
        tb.rms_norm_zc.run(&self.scr.x, &w.norm, &self.scr.n2, t32, h, eps, stream);
        tb.gemm.run_wt(&self.scr.n2, &self.embed, &self.scr.logits, t32, voc, h, stream);
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
    fn forward_rows(
        &mut self,
        rows: &[(u32, u32, u32, u32)],
        samples: &[u32],
        pure_decode: bool,
        smp: &[SampleParams],
    ) -> Vec<u32> {
        // 护栏（同 `step` 的 pure_decode 判定）：带分离 state 行的批不许进图档
        let pure_decode = pure_decode && rows.iter().all(|r| r.1 == r.0);
        if pure_decode && !rows.is_empty() && samples.len() == rows.len() {
            if let Some(bi) = self.bucket_for(rows.len()) {
                return self.forward_graph(bi, rows, smp);
            }
        }
        self.forward_batch_mixed(rows, samples, &[], smp)
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
    ///
    /// **采样步选另一张图**：图尾有两个版本——`execs`（只 argmax，greedy 的
    /// 原图，逐位/零额外开销不变）与 `execs_smp`（argmax + 图内 sample_rows）。
    /// 本步有任一非 greedy 行 → 回放后者并**整桶**覆写逐行参数（尾部 = greedy
    /// 填充：sink 行绝不偷用别的请求的 RNG 槽）；全 greedy → 回放前者，
    /// 一次上传都不多、一次回读都不多。
    fn forward_graph(
        &mut self,
        b_idx: usize,
        rows: &[(u32, u32, u32, u32)],
        smp: &[SampleParams],
    ) -> Vec<u32> {
        self.session.gpu().bind_thread();
        let (b, cap, m, sink) = {
            let g = self.graph.as_ref().expect("forward_graph 须图档");
            (g.buckets[b_idx] as usize, g.b_max as usize, rows.len(), g.sink_row)
        };
        // 同 `forward_batch_mixed`：合同违约不 panic（图档桶路由 + 位置上界）
        if m == 0 || m > b || rows.iter().any(|&(_, _, _, pos)| (pos as usize) >= MAX_POS) {
            eprintln!("[exec] 图档合同违约（m={m}/桶 {b}，位置上限 {MAX_POS}），本步作废");
            return Vec::new();
        }
        self.last_t = m;
        // 本步是否有采样行（逐行参数表可能短于样本行——缺省 = greedy）
        let row_smp: Vec<SampleParams> =
            (0..m).map(|j| smp.get(j).copied().unwrap_or_default()).collect();
        let any_smp = row_smp.iter().any(|p| !p.is_greedy());
        // host 组装：b_max 全量（默认值 = sink 吸收行），前 m 行写真数据
        let mut ids = vec![0u32; cap];
        let mut pos = vec![0u32; cap];
        let mut seqs = vec![sink; cap];
        let mut rid = vec![sink; cap];
        for (j, &(bt, sr, tok, p)) in rows.iter().enumerate() {
            // 图档的 rid/trow 与块表行同缓冲：仅当 state_row == 块表行时成立。
            // 投机链的逐位置 state 行（`spec_finish` 换行落定后 ≠ 块表行）由
            // `step`/`forward_rows` 的 pure_decode 护栏挡在入口，走到这里还
            // 不等就是护栏失效——宁可炸也不许把借来的行烤进图
            assert_eq!(sr, bt, "图档要求 state_row == 块表行（分离行须走 eager，见 step 的 pure_decode 护栏）");
            ids[j] = tok;
            pos[j] = p;
            seqs[j] = bt;
            rid[j] = sr;
        }
        // 采样参数整桶覆写（尾部 greedy：temp=0 的 sink 行不进 RNG、不采样）
        let stream = self.session.stream();
        if any_smp {
            let mut st = vec![0f32; cap];
            let mut sk = vec![64u32; cap];
            let mut sp = vec![1f32; cap];
            let mut sd = vec![0u64; cap];
            let mut ss = vec![sink; cap];
            for j in 0..m {
                let p = row_smp[j];
                st[j] = p.temperature as f32; // greedy 已由 is_greedy 归零语义覆盖
                sk[j] = p.top_k.clamp(1, 64);
                sp[j] = if p.top_p > 0.0 && p.top_p <= 1.0 { p.top_p as f32 } else { 1.0 };
                sd[j] = p.seed.unwrap_or(0);
                ss[j] = rows[j].0; // RNG 槽 = 请求块表行
            }
            // 前缀 = 整个桶（cap 项）；缓冲尾部（> cap）kernel 不读（s = b）
            self.scr.sm_temp.upload_prefix_on(&st, stream);
            self.scr.sm_topk.upload_prefix_on(&sk, stream);
            self.scr.sm_topp.upload_prefix_on(&sp, stream);
            self.scr.sm_seed.upload_prefix_on(&sd, stream);
            self.scr.sm_slots.upload_prefix_on(&ss, stream);
        }
        {
            let g = self.graph.as_ref().expect("forward_graph 须图档");
            g.in_ids.upload_on(&ids, stream);
            g.in_pos.upload_on(&pos, stream);
            g.in_seqs.upload_on(&seqs, stream);
            g.in_rid.upload_on(&rid, stream);
            // 页保证已先于本调用完成（step/generate 合同）——bt_host 是最新账
            g.in_bt.upload_on(&self.bt_host, stream);
            if any_smp {
                g.execs_smp[b_idx].launch(stream);
            } else {
                g.execs[b_idx].launch(stream);
            }
        }
        stream.sync();
        {
            let g = self.graph.as_mut().unwrap();
            g.replays += 1;
            if any_smp {
                g.smp_replays += 1;
            }
        }
        let mut out: Vec<u32> = self.scr.am_out.read_to_vec();
        if any_smp {
            // 采样行覆盖：greedy 行仍用 am_out（图尾 argmax 的输出逐位不变）
            let sm: Vec<u32> = self.scr.sm_out.read_to_vec();
            for j in 0..m {
                if !row_smp[j].is_greedy() {
                    assert_ne!(
                        sm[j], 0xffff_ffff,
                        "sample_rows 参数非法（top_k=0 / 候选为空）——set_sampling 应已拒绝"
                    );
                    out[j] = sm[j];
                }
            }
        }
        out[..m].to_vec()
    }
}

// ---- 布局常量辅助（open 期与运行期共用）----

/// n 个 u32 的字节数（eager 混合批常驻输入表）
fn u32b(n: usize) -> usize {
    n * 4
}

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

    /// 采样参数下发（见 `Qwen35Executor::set_sampling`）：Err = 设备侧不可
    /// 满足（全词表 top_k=0 / 与投机链冲突），engine 直接终结该请求
    fn set_sampling(&mut self, id: ReqId, p: SampleParams) -> Result<(), String> {
        Qwen35Executor::set_sampling(self, id, p)
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

    /// 硬上下文长度（serve 侧 `max_tokens` 缺省与越界前置校验的唯一来源）。
    /// = paged_attn 的绝对位置上界 `MAX_POS`（KV 池几何 512 页 × 16 只是容量，
    /// 位置表与图档缓冲按 MAX_POS 分配）
    fn ctx_limit(&self) -> usize {
        MAX_POS
    }

    /// 取走本步的非 panic 失败（页池/行耗尽 = Capacity → length；token 越界等
    /// = Invalid → error）。engine 的 `settle_failures` 据此终结该请求
    fn take_failures(&mut self) -> Vec<(ReqId, StepError)> {
        std::mem::take(&mut self.failures)
    }

    fn admitted(&mut self, id: ReqId, prompt: &[u32]) -> usize {
        // 越界/耗尽从 assert（panic 会带走整个引擎任务）改成**可恢复失败**：
        // 记进 failures，pump 后续 settle_failures 让该请求以 Error/Length 结束
        if let Some(&bad) = prompt.iter().find(|&&t| t >= self.cfg.vocab_size) {
            self.fail(
                id,
                StepError::Invalid(format!("token {bad} 越界（vocab {}）", self.cfg.vocab_size)),
            );
            return 0;
        }
        let seq = if prompt.is_empty() { vec![0u32] } else { prompt.to_vec() };
        let row = match self.free_rows.pop() {
            // 回收行零化挪到领用时做（链的 arena 借出行归还时不再扫一遍）：
            // 新请求从 pos 0 起跑，递推状态必须是零行（q10 合同）
            Some(r) => {
                self.zero_row_state(r, &self.session.stream());
                r
            }
            None => {
                if self.next_row >= self.row_capacity {
                    self.fail(
                        id,
                        StepError::Capacity(format!(
                            "块表行耗尽（{}，图档含 sink 让位 1 行）",
                            self.row_capacity
                        )),
                    );
                    return 0;
                }
                let r = self.next_row;
                self.next_row += 1;
                r
            }
        };
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
        // 采样 RNG 槽（= 请求行）回到"未初始化"：新请求的随机流只由 seed 决定，
        // 不继承上一个占用该行的请求的 RNG 位置（同 seed ⇒ 同序列的前提，
        // qs14 钉死）。抢占重准入不走这里（prompt 已消费），流得以续接
        self.scr.sm_rng.zero_range_on(row as usize * 8, 8, &self.session.stream());
        let smp = self.sample_cfg.get(&id).copied().unwrap_or_default();
        self.reqs.insert(
            id,
            // state_row 默认 = 块表行（现有语义不变；投机链另从 arena 借行）
            ReqState {
                seq,
                forwarded,
                row,
                state_row: row,
                blocks,
                shared,
                snaps: Vec::new(),
                spec_k: 0,
                spec_auto: false,
                spec_ema: None,
                spec_low_run: 0,
                spec_probe: 0,
                spec_idle: 0,
                spec_off_tries: 0,
                mtp_chain: None,
                hid: None,
                smp,
            },
        );
        forwarded
    }

    fn step(&mut self, plan: &StepPlan) -> Vec<(ReqId, u32)> {
        struct RowIn {
            id: ReqId,
            row: u32,
            /// 递推状态行（deltanet/conv；默认 = row）
            state_row: u32,
            tok: u32,
            pos: u32,
            /// 该行是 prefill（prompt）token——决定用哪条快照栅格
            pref: bool,
        }
        /// 把失败请求的行从本步批里摘掉，并重映射采样行下标（表/行的一致性）
        fn prune_rows(
            batch: Vec<RowIn>,
            samples: Vec<(ReqId, usize)>,
            killed: &HashSet<ReqId>,
        ) -> (Vec<RowIn>, Vec<(ReqId, usize)>) {
            let mut remap = vec![usize::MAX; batch.len()];
            let mut out: Vec<RowIn> = Vec::with_capacity(batch.len());
            for (i, r) in batch.into_iter().enumerate() {
                if killed.contains(&r.id) {
                    continue;
                }
                remap[i] = out.len();
                out.push(r);
            }
            let keep = samples
                .into_iter()
                .filter_map(|(id, i)| (remap[i] != usize::MAX).then_some((id, remap[i])))
                .collect();
            (out, keep)
        }
        let mut batch: Vec<RowIn> = Vec::new();
        let mut samples: Vec<(ReqId, usize)> = Vec::new();
        // 自适应 k 的再探：k=0（策略判定本段关投机）的请求每解一步倒计时，
        // 到点用 k=1 试一轮——k=0 期间不产生轮次 ⇒ EMA 冻结，靠这只探针
        // 才能"由 EMA 回升重新开"（见 `SPEC_PROBE_STEPS`）
        let mut anchor_now: Vec<ReqId> = Vec::new();
        for id in plan.decode.iter().copied() {
            let Some(st) = self.reqs.get_mut(&id) else { continue };
            if !st.spec_auto {
                continue;
            }
            if st.spec_k != 0 {
                st.spec_idle = 0;
                continue;
            }
            if st.spec_probe > 0 {
                st.spec_probe -= 1;
                st.spec_idle += 1;
            } else {
                st.spec_k = 1;
            }
            if st.spec_idle >= SPEC_ANCHOR_IDLE && st.state_row != st.row {
                anchor_now.push(id);
            }
        }
        // 关投机够久：把递推状态搬回块表行（一次 25.2MB），该请求的纯 decode
        // 步重回图档；k=0 期间没有任何轮次 ⇒ 不搬就永远 eEager
        for id in anchor_now {
            self.spec_reanchor(id);
            if let Some(st) = self.reqs.get_mut(&id) {
                st.spec_idle = 0;
            }
        }
        // 投机候选（纯 decode 且 spec_k > 0）：先整批起草——第 i 个草稿步 = 一次
        // MTP 前向覆盖全部链，成本与链数无关。起草必须在主前向之前（要读
        // `st.hid`，写 `scr.x`/`scr.n2`）；草稿拿不到（无 MTP 权重 / 无链槽 /
        // hidden 未落盘 / 链尾越界）的请求本步退化成普通 decode 行，混进同一批
        // （不再单独发一次前向）。
        let spec_cands: Vec<(ReqId, u32)> = plan
            .decode
            .iter()
            .copied()
            .filter(|id| !plan.prefill.iter().any(|&(p, _)| p == *id))
            .filter(|id| {
                self.reqs.get(id).is_some_and(|st| {
                    st.spec_k > 0 && st.forwarded < st.seq.len() && st.smp.is_greedy()
                })
            })
            .map(|id| (id, self.reqs[&id].spec_k))
            .collect();
        let drafts = self.spec_draft_batch(&spec_cands);
        let mut spec_ready: Vec<(ReqId, Vec<u32>)> = Vec::with_capacity(spec_cands.len());
        let mut degraded: Vec<ReqId> = Vec::new();
        for ((id, _), d) in spec_cands.iter().zip(drafts) {
            if d.is_empty() {
                degraded.push(*id);
            } else {
                spec_ready.push((*id, d));
            }
        }
        self.spec_fallbacks += degraded.len() as u64;
        let spec_ids: Vec<ReqId> = spec_ready.iter().map(|(id, _)| *id).collect();
        // 1. prefill 段（追平步段尾即采样位）
        for &(id, n) in &plan.prefill {
            let Some(st) = self.reqs.get_mut(&id) else { continue };
            let start = st.forwarded;
            let take = (n as usize).min(st.seq.len().saturating_sub(start));
            for j in 0..take {
                batch.push(RowIn {
                    id,
                    row: st.row,
                    state_row: st.state_row,
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
        // 2. 纯 decode：前向 seq 尾（含草稿失败的退化请求——同一批一行）
        for &id in &plan.decode {
            if plan.prefill.iter().any(|&(p, _)| p == id) {
                continue;
            }
            if spec_ids.contains(&id) {
                continue; // 投机链走下方链行（草稿已起草完）
            }
            let Some(st) = self.reqs.get_mut(&id) else { continue };
            if st.forwarded < st.seq.len() {
                let pos = st.forwarded;
                batch.push(RowIn {
                    id,
                    row: st.row,
                    state_row: st.state_row,
                    tok: st.seq[pos],
                    pos: pos as u32,
                    pref: false,
                });
                st.forwarded += 1;
                samples.push((id, batch.len() - 1));
            }
        }
        if batch.is_empty() && spec_ready.is_empty() {
            return Vec::new();
        }
        // 3. 页保证（每请求最大 pos）——先于 scatter。页池耗尽 / 位置上界从
        //    panic 改成**可恢复失败**：该请求退出本步批、以 length 结束，
        //    同批其他请求照常（绝不 panic 带走引擎任务）
        let mut killed: HashSet<ReqId> = HashSet::new();
        for r in &batch {
            if (r.pos as usize) >= MAX_POS {
                killed.insert(r.id);
            }
        }
        for id in killed.iter().copied().collect::<Vec<_>>() {
            self.fail(
                id,
                StepError::Capacity(format!("前向位置越界（paged_attn 上限 {MAX_POS}）")),
            );
        }
        if !killed.is_empty() {
            let (b2, s2) = prune_rows(batch, samples, &killed);
            batch = b2;
            samples = s2;
            spec_ready.retain(|(id, _)| !killed.contains(id));
        }
        let mut need: HashMap<ReqId, usize> = HashMap::new();
        for r in &batch {
            let e = need.entry(r.id).or_insert(0);
            *e = (*e).max(r.pos as usize);
        }
        let mark = self.failures.len();
        for (id, upto) in need {
            if let Err(e) = self.ensure_pages(id, upto) {
                self.fail(id, e);
            }
        }
        if self.failures.len() > mark {
            let fresh: HashSet<ReqId> = self.failures[mark..].iter().map(|(id, _)| *id).collect();
            killed.extend(fresh.iter().copied());
            let (b2, s2) = prune_rows(batch, samples, &fresh);
            batch = b2;
            samples = s2;
            spec_ready.retain(|(id, _)| !fresh.contains(id));
        }
        // 4. 批组装：普通行（prefill / decode）+ 链行（每链 cnt 行）同一条前向。
        //    链的 state 行在此分配（在主批行之后：段表 row_start 指进同一 rows）
        let (mut jobs, ch_fails) = self.spec_chain_alloc(&spec_ready);
        // 链行分配失败（池里没有连续 n 行）→ 该请求本步退化为普通 decode 行，
        // 混进同一批（不再单独发一次前向），只喂 pending token、出 1 个 token
        for id in ch_fails {
            self.spec_fallbacks += 1;
            let (forwarded, seqlen) = match self.reqs.get(&id) {
                Some(st) => (st.forwarded, st.seq.len()),
                None => continue,
            };
            if forwarded >= seqlen {
                continue;
            }
            if forwarded >= MAX_POS {
                self.fail(
                    id,
                    StepError::Capacity(format!("前向位置越界（paged_attn 上限 {MAX_POS}）")),
                );
                continue;
            }
            let pos = forwarded;
            let Some(st) = self.reqs.get_mut(&id) else { continue };
            let (row, state_row, tok) = (st.row, st.state_row, st.seq[pos]);
            st.forwarded += 1;
            batch.push(RowIn {
                id,
                row,
                state_row,
                tok,
                pos: pos as u32,
                pref: false,
            });
            samples.push((id, batch.len() - 1));
        }
        let mut rows: Vec<(u32, u32, u32, u32)> =
            batch.iter().map(|r| (r.row, r.state_row, r.tok, r.pos)).collect();
        let mut sample_rows: Vec<u32> = samples.iter().map(|s| s.1 as u32).collect();
        // 采样参数与 `sample_rows` 一一对齐（greedy 请求 = 默认值 → 不发 kernel）
        let mut smp: Vec<SampleParams> = samples
            .iter()
            .map(|(id, _)| self.reqs.get(id).map(|st| st.smp).unwrap_or_default())
            .collect();
        let segs = self.spec_chain_fill(&mut jobs, &mut rows, &mut sample_rows);
        // 链行（投机验证）一律 greedy：逐位置 argmax 是接受判定的前提
        // （`set_spec`/`set_sampling` 两侧已挡非 greedy 请求进链）
        smp.resize(sample_rows.len(), SampleParams::greedy());
        // 含投机链的步一律 eager：链行不进图（forward_graph 有 sr==bt 断言），
        // 且图捕获的采样行表是纯 decode 桶形的纯函数，混进链批会错位。
        // 链全退化 / 无候选的步仍是纯 decode → 走图档。
        //
        // **护栏**：跑过投机的请求 `state_row ≠ row`（换行落定把它指到了链行），
        // 而图档的 rid/trow 与块表行是同一套固化表（`forward_graph` 的
        // `assert_eq!(sr, bt)`）——混进去必 panic。这里把"有没有分离行"放进
        // 纯 decode 的判定：只要有借来的 state 行就退 eager（关投机时会
        // `spec_reanchor` 搬回，请求随即恢复图档资格）
        let pure_decode =
            plan.prefill.is_empty() && jobs.is_empty() && batch.iter().all(|r| r.state_row == r.row);
        let t_fwd = Instant::now();
        let toks = if rows.is_empty() {
            Vec::new()
        } else if pure_decode {
            self.forward_rows(&rows, &sample_rows, true, &smp)
        } else {
            self.forward_batch_mixed(&rows, &sample_rows, &segs, &smp)
        };
        if !jobs.is_empty() {
            self.spec_ms_forward += prof_ms(t_fwd.elapsed());
        }
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
                // nb <= blocks.len() 即该边界所在页已分配（防御：请求已不在）。
                // 快照源行取 `state_row`（不是块表行）：跑过投机的请求两者
                // 分离，递推状态在 state_row 上——拿块表行会存一份过期状态，
                // 将来 radix 命中恢复出错位（静默污染）
                let (srow, blk) = match self.reqs.get(&id) {
                    Some(st) if st.blocks.len() >= nb => (st.state_row, st.blocks[nb - 1]),
                    _ => continue,
                };
                let Some(slot) = self.acquire_slot() else {
                    continue; // 槽耗尽：降级无快照
                };
                if let Some(st) = self.reqs.get_mut(&id) {
                    st.snaps.push((blk, slot));
                }
                snaps.push((srow, slot));
            }
            if !snaps.is_empty() {
                let per = self.snap_lin.len() + self.snap_conv.len();
                let mut ops: Vec<CuBatchOp> = Vec::with_capacity(snaps.len() * per);
                for (srow, slot) in snaps {
                    self.push_snapshot_ops(srow, slot, &mut ops);
                }
                self.session.stream().memcpy_batch(&ops);
            }
        }
        // hidden 落盘：投机请求在本批内时（prefill 末位 / 退化普通行），其
        // 采样行的 hidden 就是位置 forwarded-1 的末层 hidden = 下一轮草稿起点
        for (id, r) in &samples {
            if self.reqs.get(id).is_some_and(|st| st.spec_k > 0) {
                self.stash_hidden(*id, *r);
            }
        }
        let mut out = Vec::new();
        for ((id, _), tok) in samples.iter().zip(&toks) {
            if let Some(st) = self.reqs.get_mut(id) {
                st.seq.push(*tok);
            }
            out.push((*id, *tok));
        }
        // 4.5 链的接受 + 落定：同一条前向的 argmax 按链切分（每链 cnt 个），
        //     逐链贪心接受 → 落定行拷回 → 按序发射 1..=k+1 个 token
        if !jobs.is_empty() {
            let chain_toks = &toks[samples.len()..];
            for (id, round) in self.spec_finish(jobs, chain_toks) {
                // 发射序：接受的草稿（位置 forwarded_old+1..）→ 奖励 token
                // （target_tokens[accepted]，位置 forwarded_old+1+accepted）。
                // 同一 ReqId 多个条目——引擎侧按序外吐（core.rs 循环天然容忍）
                for &tok in &round.draft_tokens {
                    out.push((id, tok));
                }
                out.push((id, round.target_tokens[round.accepted]));
            }
        }
        out
    }

    fn retain(&mut self, active: &[ReqId]) {
        // 排序退场：`reqs` 是 HashMap，迭代序每进程随机——退场顺序决定
        // 行/页/槽回池的顺序，进而决定后续 `admitted`/`spec_chain_rows` 领到
        // 哪个行号。行号变 ⇒ 状态行地址对齐变 ⇒ cuBLASLt 选别的 algo（跨 M
        // ulp 漂移，见 q5/qs9 注释）⇒ 探针吃掉两档不同的接受率（~43%/25%）。
        // 排序让"同一串操作 → 同一套行号"成立，读数可比
        let mut gone: Vec<ReqId> =
            self.reqs.keys().copied().filter(|k| !active.contains(k)).collect();
        gone.sort_unstable();
        for id in gone {
            self.mtp_release(id); // 草稿链槽 + 链 KV 归还（在移除请求态之前）
            self.sample_cfg.remove(&id); // 采样参数随请求退场（行可能被复用）
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
                // 回收行零化（q10 钉）。投机跑过的请求有两个行号：块表行管 KV 页、
                // 换行落定后的 state_row 管递推状态——都零化、都归还（`admitted`
                // 领用时还会再零一次，幂等）
                self.zero_row_state(st.state_row, &self.session.stream());
                self.free_rows.push(st.state_row);
                if st.row != st.state_row {
                    self.zero_row_state(st.row, &self.session.stream());
                    self.free_rows.push(st.row);
                }
            }
        }
        self.pool.recycle();
    }

    fn preempted(&mut self, id: ReqId) {
        // 重放从 pos 0 起步：草稿链 KV 是"位置相对 forwarded"的干净段，必须
        // 丢掉（槽保留，请求继续用自己的链槽，下一轮 `ensure_mtp_pages` 重建）
        if let Some(slot) = self.reqs.get(&id).and_then(|st| st.mtp_chain) {
            self.mtp_drop_chain(slot);
        }
        // 状态搬出表：下面要 `alloc_page`（借 `self`），留着 `get_mut` 会撞借用；
        // 换页失败时原样放回——请求由 engine 以 length 终结，不再重放
        let mut st = match self.reqs.remove(&id) {
            Some(st) => st,
            None => return,
        };
        let (row, srow) = (st.row, st.state_row);
        // 快照槽全数作废回池（行即将零化；重放重新落槽）
        for (_, slot) in st.snaps.drain(..) {
            self.snap_free.push(slot);
        }
        // 去共享：命中块换新私有页——重放覆盖写有 ulp 漂移（跨批
        // 形状 GEMM algo 差），共享页不可重写（污染缓存/他人）
        let shared = st.shared;
        if self.radix_on && shared > 0 {
            // 先凑齐替换页再拆共享：页池不够就整笔退回，避免块表半途出现空洞
            let mut news: Vec<u32> = Vec::with_capacity(shared);
            while news.len() < shared {
                match self.alloc_page() {
                    Some(b) => news.push(b),
                    None => {
                        for b in news.drain(..) {
                            self.pool.release(b);
                        }
                        self.fail(
                            id,
                            StepError::Capacity(format!(
                                "重放换私页失败：页池耗尽（{} 页）",
                                self.pool.total()
                            )),
                        );
                        self.reqs.insert(id, st);
                        return;
                    }
                }
            }
            let head: Vec<u32> = st.seq[..shared * BLOCK as usize].to_vec();
            self.radix.unpin(&mut self.pool, &head, shared);
            let base = row as usize * self.bpr;
            for (j, b) in news.into_iter().enumerate() {
                self.bt_host[base + j] = b;
                st.blocks[j] = b;
            }
            st.shared = 0;
        }
        let stream = self.session.stream();
        self.zero_row_state(srow, stream); // 重放从 pos 0 起步：递推状态必须归零
        st.forwarded = 0;
        self.reqs.insert(id, st);
    }
}
