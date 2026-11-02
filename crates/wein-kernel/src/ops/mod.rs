//! 算子注册表（02-architecture §4）。
//!
//! **启动时解析一次 → 具体类型，热路径零动态分发**：
//! - 槽位是具体 enum 字段（无 Box<dyn>、无字符串查表、无每步分配）
//! - trait（Op 家族）定义算子 ABI；OpTable 存具体类型——分支在 resolve
//!   后冻结，编译器可内联
//! - 解析失败 = Degrade 三元组列表（声明式降级，wein-platform §3）——
//!   启动期裁决，绝不带病运行也绝不隐式硬改
use crate::cuda::{DevBuf, Func, Gpu, Stream};
use wein_platform::{Device, Degrade, HardwareProfile};

/// rms_norm 的家族 trait（§4 "trait Op 家族"首个成员）。
/// trait 钉 ABI；热路径走 OpTable 的具体类型字段（不 dyn）
pub trait RmsNormOp {
    /// x/out: [rows, hidden] bf16 行主序；w: [hidden] bf16；eps: f32
    fn run(
        &self,
        x: &DevBuf,
        w: &DevBuf,
        out: &DevBuf,
        rows: u32,
        hidden: u32,
        eps: f32,
        stream: &Stream,
    );
    /// 解析结果标识（测试/日志；含实现档位）
    fn name(&self) -> &'static str;
}

/// rms_norm 的实现档位（同一 kernel、不同 launch 几何——W2 真 kernel
/// 落地时在此扩档并按 profile 选择，调用方代码不变）
pub enum RmsNormImpl<'a> {
    /// 每行一 block×1 线程——老架构保守档（< sm_80）
    Row { gpu: &'a Gpu, f: Func },
    /// block=256 承载行——sm_80+ 档（本机 sm_89 走这档）
    Blocked { gpu: &'a Gpu, f: Func, block: u32 },
}

/// 算子表：启动时 resolve 一次，全程只读
pub struct OpTable<'a> {
    pub rms_norm: RmsNormImpl<'a>,
    pub rope: RopeCuda<'a>,
    pub silu: SiluCuda<'a>,
    pub w8a8_quant: QuantCuda<'a>,
    pub kv_scatter: KvScatterCuda<'a>,
    pub paged_attn: PagedAttnCuda<'a>,
    pub gemm: GemmLt<'a>,
    pub embed: EmbedCuda<'a>,
    pub add: AddCuda<'a>,
    pub mul: MulCuda<'a>,
    // I1 竖切（Qwen3.5 文本骨干）
    pub rms_norm_zc: RmsNormZcCuda<'a>,
    pub rope_pt: RopePtCuda<'a>,
    pub sigmul: SigmulCuda<'a>,
    pub conv1d_dw: Conv1dDwCuda<'a>,
    pub conv_state_roll: ConvStateRollCuda<'a>,
    pub deltanet: DeltanetCuda<'a>,
    pub rms_norm_gated: RmsNormGatedCuda<'a>,
    pub argmax_rows: ArgmaxRowsCuda<'a>,
    pub sample_rows: SampleRowsCuda<'a>,
}

impl OpTable<'_> {
    /// 按 HardwareProfile 解析。设备族不匹配 → Degrade 列表（启动期裁决）。
    /// PTX 模块不显式卸载——算子表即进程生命周期，CUfunction 句柄常驻。
    /// gemm 全 heuristic 钉（瞬态形状路径；稳态形状调 `resolve_tuned`）
    pub fn resolve<'a>(
        gpu: &'a Gpu,
        profile: &HardwareProfile,
    ) -> Result<OpTable<'a>, Vec<Degrade>> {
        Self::resolve_tuned(gpu, profile, &[])
    }

    /// 同 `resolve`，另指定 gemm autotune 的 m 集（exec 稳态 decode 桶集
    /// 走此口；见 `BlasLt::open_tuned`）
    pub fn resolve_tuned<'a>(
        gpu: &'a Gpu,
        profile: &HardwareProfile,
        tune_m: &[u32],
    ) -> Result<OpTable<'a>, Vec<Degrade>> {
        if profile.device != Device::Cuda {
            return Err(vec![Degrade {
                key: "kernel.backend",
                reason: format!("画像设备为 {:?}，CUDA 算子表无法解析", profile.device),
                impact: "无可用 rms_norm 实现，拒绝带病启动——须显式选择正确后端".into(),
            }]);
        }
        let arch = profile.arch.clone().unwrap_or_else(|| gpu.arch().to_string());
        let (maj, _min) = parse_arch(&arch).ok_or_else(|| {
            vec![Degrade {
                key: "kernel.arch",
                reason: format!("arch 字串不可解析：{arch}"),
                impact: "无法裁决 launch 档位（row/blocked）".into(),
            }]
        })?;
        let ptx = std::fs::read_to_string(crate::ptx_path()).map_err(|e| {
            vec![Degrade {
                key: "kernel.ptx",
                reason: format!("PTX 产物读取失败：{e}"),
                impact: "rms_norm 无法加载（先构建 kernels 子 crate）".into(),
            }]
        })?;
        let f = gpu.load_module(&ptx).func("rms_norm");
        // 档位裁决：sm_80+ 走 blocked（block=256 承载行）；老架构保守 row 档
        let rms_norm = if maj >= 8 {
            RmsNormImpl::Blocked { gpu, f, block: 256 }
        } else {
            RmsNormImpl::Row { gpu, f }
        };
        // 其余 M0 算子同模块取入口（几何无档位之分——一维、线程间无归约）
        let m = gpu.load_module(&ptx);
        let rope = RopeCuda { gpu, f: m.func("rope") };
        let silu = SiluCuda { gpu, f: m.func("silu") };
        let w8a8_quant = QuantCuda { gpu, f: m.func("w8a8_quant") };
        let kv_scatter = KvScatterCuda {
            gpu,
            f: m.func("kv_scatter"),
            f32v: m.func("kv_scatter_f32"),
        };
        let paged_attn = PagedAttnCuda {
            gpu,
            f: m.func("paged_attn"),
            scratch: gpu.alloc_bytes(ATTN_BLOCK_CAP * (ATTN_SLOT_CAP + 2) * 4),
            slot_cap: ATTN_SLOT_CAP as u32,
            ft: m.func("paged_attn_tile"),
            ftf32: m.func("paged_attn_tile_f32"),
            fo: m.func("paged_attn_online"),
            f_nd8: m.func("paged_attn_online_n8"),
            f_nd16: m.func("paged_attn_online_n16"),
            f_r2: m.func("paged_attn_online_r2"),
            f_r4: m.func("paged_attn_online_r4"),
            f_nd8_r4: m.func("paged_attn_online_n8_r4"),
            f_n16_r4: m.func("paged_attn_online_n16_r4"),
            f_n16_r2: m.func("paged_attn_online_n16_r2"),
            f_f32: m.func("paged_attn_online_f32"),
            f_f32_nd8: m.func("paged_attn_online_f32_n8"),
            f_f32_r4: m.func("paged_attn_online_f32_r4"),
            f_f32_r2: m.func("paged_attn_online_f32_r2"),
            fc: m.func("attn_combine"),
            f_blk: m.func("paged_attn_blk"),
            f_blk_f32: m.func("paged_attn_blk_f32"),
            tscratch: gpu.alloc_bytes(ATTN_TILE_BLOCK_CAP * ATTN_TILE_SLOTS * 4),
            pacc: gpu.alloc_bytes(ATTN_TILE_PART_ROWS * ATTN_TILE_SPLIT_CAP * ATTN_TILE_HD_CAP * 4),
            pml: gpu.alloc_bytes(ATTN_TILE_PART_ROWS * ATTN_TILE_SPLIT_CAP * 2 * 4),
            bscratch: gpu.alloc_bytes(BLK_SCRATCH_FLOATS * 4),
            pacc_b: gpu.alloc_bytes(BLK_PART_ROWS * BLK_SPLIT_CAP * ATTN_TILE_HD_CAP * 4),
            pml_b: gpu.alloc_bytes(BLK_PART_ROWS * BLK_SPLIT_CAP * 2 * 4),
        };
        let embed = EmbedCuda { gpu, f: m.func("embed") };
        let add = AddCuda { gpu, f: m.func("add") };
        let mul = MulCuda { gpu, f: m.func("mul") };
        // I1 竖切八入口（同模块；deltanet 自带块几何——注册时零假设）
        let rms_norm_zc = RmsNormZcCuda {
            gpu,
            f: m.func("rms_norm_zc"),
            scratch: gpu.alloc_bytes(REDUCE_ROWS_CAP * 256 * 4),
        };
        let rope_pt = RopePtCuda { gpu, f: m.func("rope_pt") };
        let sigmul = SigmulCuda { gpu, f: m.func("sigmul") };
        let conv1d_dw = Conv1dDwCuda { gpu, f: m.func("conv1d_dw") };
        let conv_state_roll = ConvStateRollCuda { gpu, f: m.func("conv_state_roll") };
        let deltanet = DeltanetCuda {
            gpu,
            f: m.func("deltanet"),
            l2f: m.func("dn_qk_l2norm"),
            scratch: gpu.alloc_bytes(DELTA_BLOCK_CAP * DELTA_SLOTS_PER_BLOCK * 4),
            inv: gpu.alloc_bytes(DELTA_L2_TOKENS_CAP * DELTA_L2_KHEADS_CAP * 2 * 4),
        };
        let rms_norm_gated = RmsNormGatedCuda {
            gpu,
            f: m.func("rms_norm_gated"),
            scratch: gpu.alloc_bytes(REDUCE_ROWS_CAP * 256 * 4),
        };
        let argmax_rows = ArgmaxRowsCuda { gpu, f: m.func("argmax_rows") };
        let sample_rows = SampleRowsCuda {
            gpu,
            f: m.func("sample_rows"),
            scratch: gpu.alloc_bytes(SAMPLE_ROWS_CAP * SAMPLE_SLOT * 4),
        };
        // 标准算子走厂商库（§4）：库缺 = 声明式降级，不带病运行
        let gemm = GemmLt { lt: Box::new(
            crate::cuda::BlasLt::open_tuned(gpu, tune_m).map_err(|e| vec![Degrade {
                key: "kernel.gemm",
                reason: format!("libcublasLt 打不开：{e}"),
                impact: "GEMM 无法解析（标准算子走厂商库，M0 无回退实现）".into(),
            }])?,
        )};
        Ok(OpTable { rms_norm, rope, silu, w8a8_quant, kv_scatter, paged_attn, gemm, embed, add, mul,
            rms_norm_zc, rope_pt, sigmul, conv1d_dw, conv_state_roll, deltanet, rms_norm_gated,
            argmax_rows, sample_rows })
    }
}

// ───────────────────── exec 骨架三算子（§4）：embed 路由查表 / 残差加 / 门控乘 ─────────────────────

/// embed gather：out[t·hidden+e] = w[id[t]·hidden+e]（位拷贝无数值）
pub trait EmbedOp {
    /// ids: [tokens] u32；w: [vocab, hidden] bf16；out: [tokens, hidden] bf16
    fn run(&self, ids: &DevBuf, w: &DevBuf, out: &DevBuf, hidden: u32, tokens: u32, stream: &Stream);
    fn name(&self) -> &'static str;
}

pub struct EmbedCuda<'a> {
    gpu: &'a Gpu,
    f: Func,
}

impl EmbedOp for EmbedCuda<'_> {
    fn run(&self, ids: &DevBuf, w: &DevBuf, out: &DevBuf, hidden: u32, tokens: u32, stream: &Stream) {
        let n = tokens * hidden;
        let args = crate::kernel_args![*ids, *w, *out, hidden, tokens];
        self.gpu.launch(&self.f, n.div_ceil(256), 256, &args, stream);
    }
    fn name(&self) -> &'static str {
        "embed/cuda"
    }
}

/// bf16 残差加：out = a + b（f32 域 RNE 收 bf16）
pub trait AddOp {
    fn run(&self, a: &DevBuf, b: &DevBuf, out: &DevBuf, n: u32, stream: &Stream);
    fn name(&self) -> &'static str;
}

pub struct AddCuda<'a> {
    gpu: &'a Gpu,
    f: Func,
}

impl AddOp for AddCuda<'_> {
    fn run(&self, a: &DevBuf, b: &DevBuf, out: &DevBuf, n: u32, stream: &Stream) {
        let args = crate::kernel_args![*a, *b, *out, n];
        self.gpu.launch(&self.f, n.div_ceil(256), 256, &args, stream);
    }
    fn name(&self) -> &'static str {
        "add/cuda"
    }
}

/// bf16 门控乘：out = a · b（f32 域 RNE 收 bf16）
pub trait MulOp {
    fn run(&self, a: &DevBuf, b: &DevBuf, out: &DevBuf, n: u32, stream: &Stream);
    fn name(&self) -> &'static str;
}

pub struct MulCuda<'a> {
    gpu: &'a Gpu,
    f: Func,
}

impl MulOp for MulCuda<'_> {
    fn run(&self, a: &DevBuf, b: &DevBuf, out: &DevBuf, n: u32, stream: &Stream) {
        let args = crate::kernel_args![*a, *b, *out, n];
        self.gpu.launch(&self.f, n.div_ceil(256), 256, &args, stream);
    }
    fn name(&self) -> &'static str {
        "mul/cuda"
    }
}

// ───────────────────── I1 竖切算子（Qwen3.5 文本骨干） ─────────────────────
// 与 kernels/ 新入口一一对应；deltanet 是唯一的非 256 块几何（block=v_dim）。

/// 零中心 RMSNorm（Qwen3_5RMSNorm 语义）：out = x·rsqrt(mean(x²)+eps)·(1+w)
pub trait RmsNormZcOp {
    /// x/out: [rows, hidden] bf16；w: [hidden] bf16
    fn run(&self, x: &DevBuf, w: &DevBuf, out: &DevBuf, rows: u32, hidden: u32, eps: f32, stream: &Stream);
    fn name(&self) -> &'static str;
}

/// 块归约 scratch 行容量：满批 q-norm 128×16=2048 行，×4 余量
const REDUCE_ROWS_CAP: usize = 8192;

/// E2 warp 归约块容量：满批 rows(128)×v_heads(32)
const DELTA_BLOCK_CAP: usize = 4096;
/// E2 的 block 宽（线程数）：32 的倍数 = warp 数，vd=128 行按 warp 等分。
/// 窄 = 4 warp × 32 行（批 16 行 ⇒ 2 批/token、4 道屏障）；
/// 宽 = 8 warp × 16 行（批 16 行 ⇒ 1 批/token、2 道屏障，并行 warp 翻倍）。
/// 行内算术与折叠树两者全同（v 行彼此独立）——只有"哪条 warp 吃哪几行"变。
const DELTA_BLOCK_DIM: usize = 128;
const DELTA_WIDE_BLOCK_DIM: usize = 256;
/// 宽块的门：块数少 = 并行度饥饿（单请求预填 32 块，全卡只 128 warp）时加宽；
/// 块数多（满负荷 decode 3200 块）时窄块给 3 块/SM × 4 warp = 12 warp/SM 的
/// 占用率，宽块掉到 1 块/SM × 8 warp。实测：预填 delta −23%（宽），
/// 满负荷 decode 走窄块不变
const DELTA_WIDE_GRID_MAX: usize = 64;
/// 每块 scratch 槽数 = 2 区 × warp 数 × 16 行批 × 32 lane（与 kernel 侧
/// `bid * nwarps * 1024` 同构）；按宽块（warp 多）预留
const DELTA_SLOTS_PER_BLOCK: usize = (DELTA_WIDE_BLOCK_DIM / 32) * 1024;
/// B 档 chunked（保守档）的 v 行切分数上限：块从 (行, v 头) 扩成
/// (行, v 头, v 行块)，每块只吃 vd/vsplit 个连续 v 行 ⇒ 块数 ×vsplit、每块
/// 依赖链 ÷vsplit。**具体档位由 `DELTA_SPLIT_TARGET_BLOCKS` 按块数配比**
/// （invq/invk 去重后 16 才吃到肉：早先每块重复算 l2norm，切 16 的重复账
/// 把收益吃掉了）。
/// **行内算术一个字不动**：切块只改"哪个块负责哪几行"（v 行彼此独立、归约在
/// warp 内 32 lane 上），故位级输出与不切块同（o20/o25 钉住）。
const DELTA_SPLIT: usize = 16;
/// 切块的目标块数（= 128 SM × 4 块/SM）：`s = min(目标/base, 上限)`，
/// base = 行数 × v 头。实测（250/256 token 的 prefill，每 token 每层 µs，
/// invq/invk 去重后）：1 行(s=16) 2.70、2 行(s=8) 3.52、4 行(s=4) 6.85、
/// 8 行(s=2) 11.8、100 行(s=1) 310µs/launch——最优点恒在"块数 ≈ 512"附近，
/// 行数再多就回到"铺满即不切"
const DELTA_SPLIT_TARGET_BLOCKS: usize = 512;
/// 切块时的块宽（线程数）：64 = 2 warp（行/warp 加倍、每 token 的屏障减半）；
/// 128 = 4 warp。行内算术全同，只改 warp 数与批大小（屏障密度）；实测同 split
/// 下两者的 delta 反向不定（split4：128 线程 23.9 vs 64 线程 27.5），故靠测量选
const DELTA_SPLIT_BLOCK_DIM: usize = 64;
/// 切块门 ②：块数已达此值即视为铺满（128 SM × 4 块）——此时再切只加重复读
/// （每块每 token 各载一遍 q/k 切片），不加并行
const DELTA_SPLIT_MAX_BLOCKS: usize = 512;
/// 前置 l2norm kernel（`dn_qk_l2norm`）的缓冲容量：tokens × k 头。
/// tokens 取 exec 的 T_MAX（512），k 头留到 64（真模型 16）——用不满的段
/// 不碰（缓冲 512×64×2 f32 = 256KB，一次性分配）。超容即断言（不静默截）
const DELTA_L2_TOKENS_CAP: usize = 512;
const DELTA_L2_KHEADS_CAP: usize = 64;
/// 前置 l2norm kernel 的块宽（一线程一 (token, k 头)，各持一条 j 升序
/// 平方和链——无跨线程归约，宽窄只影响占用率）
const DELTA_L2_BLOCK_DIM: u32 = 128;

pub struct RmsNormZcCuda<'a> {
    gpu: &'a Gpu,
    f: Func,
    /// 块归约 scratch：[REDUCE_ROWS_CAP, 256] f32，block 按 blockIdx 私有槽
    /// （resolve 一次预分配——capture 期不触 alloc_guard，运行时零分配）
    scratch: DevBuf<'a>,
}

impl RmsNormZcOp for RmsNormZcCuda<'_> {
    fn run(&self, x: &DevBuf, w: &DevBuf, out: &DevBuf, rows: u32, hidden: u32, eps: f32, stream: &Stream) {
        let args = crate::kernel_args![*x, *w, *out, hidden, eps, rows, self.scratch];
        // block-per-row：块宽 = min(hidden,256) 上取 2 幂（kernel 的
        // scratch 块归约按 blockDim 同构——reference.rs 镜像同式）
        let b = hidden.min(256).next_power_of_two();
        assert!(
            rows as usize * b as usize * 4 <= self.scratch.bytes(),
            "rms_norm_zc rows={rows} 超 scratch 容量 {} 槽",
            self.scratch.bytes() / 4
        );
        self.gpu.launch(&self.f, rows, b, &args, stream);
    }
    fn name(&self) -> &'static str {
        "rms_norm_zc/cuda"
    }
}

/// partial RoPE：单张量 [tokens, heads, head_dim]，前 rot_dim 维 rotate_half，
/// 其余 pass-through（kernel 不触碰——out 可与 x 同缓冲原地）
pub trait RopePtOp {
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        x: &DevBuf,
        cos: &DevBuf,
        sin: &DevBuf,
        pos: &DevBuf,
        out: &DevBuf,
        rot_dim: u32,
        head_dim: u32,
        heads: u32,
        tokens: u32,
        stream: &Stream,
    );
    fn name(&self) -> &'static str;
}

pub struct RopePtCuda<'a> {
    gpu: &'a Gpu,
    f: Func,
}

impl RopePtOp for RopePtCuda<'_> {
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        x: &DevBuf,
        cos: &DevBuf,
        sin: &DevBuf,
        pos: &DevBuf,
        out: &DevBuf,
        rot_dim: u32,
        head_dim: u32,
        heads: u32,
        tokens: u32,
        stream: &Stream,
    ) {
        let n = tokens * heads * (rot_dim / 2);
        let args = crate::kernel_args![*x, *cos, *sin, *pos, *out, rot_dim, head_dim, heads, tokens];
        self.gpu.launch(&self.f, n.div_ceil(256), 256, &args, stream);
    }
    fn name(&self) -> &'static str {
        "rope_pt/cuda"
    }
}

/// 门控乘：out = x·sigmoid(g)（注意力输出门；与 mul 的区别在 sigmoid 在内）
pub trait SigmulOp {
    fn run(&self, x: &DevBuf, g: &DevBuf, out: &DevBuf, n: u32, stream: &Stream);
    fn name(&self) -> &'static str;
}

pub struct SigmulCuda<'a> {
    gpu: &'a Gpu,
    f: Func,
}

impl SigmulOp for SigmulCuda<'_> {
    fn run(&self, x: &DevBuf, g: &DevBuf, out: &DevBuf, n: u32, stream: &Stream) {
        let args = crate::kernel_args![*x, *g, *out, n];
        self.gpu.launch(&self.f, n.div_ceil(256), 256, &args, stream);
    }
    fn name(&self) -> &'static str {
        "sigmul/cuda"
    }
}

/// 深度wise 因果 conv1d + silu：varlen 批内同请求 token 段连续（executor
/// 合同），不足 kernel-1 步的 tap 取 conv 状态
pub trait Conv1dDwOp {
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        x: &DevBuf,
        w: &DevBuf,
        state_in: &DevBuf,
        out: &DevBuf,
        tok_row: &DevBuf,
        tok_pos: &DevBuf,
        conv_dim: u32,
        kernel: u32,
        tokens: u32,
        stream: &Stream,
    );
    fn name(&self) -> &'static str;
}

pub struct Conv1dDwCuda<'a> {
    gpu: &'a Gpu,
    f: Func,
}

impl Conv1dDwOp for Conv1dDwCuda<'_> {
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        x: &DevBuf,
        w: &DevBuf,
        state_in: &DevBuf,
        out: &DevBuf,
        tok_row: &DevBuf,
        tok_pos: &DevBuf,
        conv_dim: u32,
        kernel: u32,
        tokens: u32,
        stream: &Stream,
    ) {
        let n = tokens * conv_dim;
        let args = crate::kernel_args![*x, *w, *state_in, *out, *tok_row, *tok_pos, conv_dim, kernel, tokens];
        self.gpu.launch(&self.f, n.div_ceil(256), 256, &args, stream);
    }
    fn name(&self) -> &'static str {
        "conv1d_dw/cuda"
    }
}

/// conv 状态滚动：state ←（旧 state ++ 本段 x）末 kernel-1 个（原地）。
/// 状态池按请求行分账——span r 写 **row_id[r]** 段（与 deltanet 同间接）
pub trait ConvStateRollOp {
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        x: &DevBuf,
        state: &DevBuf,
        row_start: &DevBuf,
        row_count: &DevBuf,
        row_id: &DevBuf,
        conv_dim: u32,
        kernel: u32,
        rows: u32,
        // [rows] 逐 span 标志（长度 = rows）：0 = 逐段滚动（现有语义）；
        // 1 = 逐位置落盘（第 t 个 token 写行 row_id[r]+t+1——MTP 验证链；
        // 该段 specs[r]==0 时的路径逐字节不变）。逐 span 而非整批标量：
        // 真实一步里普通 decode 行/prefill 行/投机链行同批
        specs: &DevBuf,
        stream: &Stream,
    );
    fn name(&self) -> &'static str;
}

pub struct ConvStateRollCuda<'a> {
    gpu: &'a Gpu,
    f: Func,
}

impl ConvStateRollOp for ConvStateRollCuda<'_> {
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        x: &DevBuf,
        state: &DevBuf,
        row_start: &DevBuf,
        row_count: &DevBuf,
        row_id: &DevBuf,
        conv_dim: u32,
        kernel: u32,
        rows: u32,
        specs: &DevBuf,
        stream: &Stream,
    ) {
        let n = rows * conv_dim;
        let args = crate::kernel_args![*x, *state, *row_start, *row_count, *row_id, conv_dim,
            kernel, rows, *specs];
        self.gpu.launch(&self.f, n.div_ceil(256), 256, &args, stream);
    }
    fn name(&self) -> &'static str {
        "conv_state_roll/cuda"
    }
}

/// gated deltanet 递推：grid = rows×v_heads×vsplit；block = v_dim（kd=vd=128 时
/// 128 线程走 E2 warp 重映射）。状态池属 executor（[rows, v_heads, v_dim,
/// k_dim] bf16，行号经 row_id 间接）。
///
/// 前置一小 kernel `dn_qk_l2norm`（q/k 的逐 (token, k 头) l2norm 倒数，
/// 主 kernel 只读）——早先每个 (行, 头) 的 vsplit 个块 × 每块 nwarps 个 warp
/// 都在重复算同一份 invq/invk，k[j]/q[j] 却对整条 v 维共享。**一次 run 两次
/// launch**（前置小 kernel 依赖本层的 q/k，且主 kernel 必须等它写完 ⇒ 跨块
/// 同步在 nvptx 后端无安全机制，只能分 launch）。
pub trait DeltanetOp {
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        q: &DevBuf,
        k: &DevBuf,
        v: &DevBuf,
        a: &DevBuf,
        b: &DevBuf,
        a_log: &DevBuf,
        dt_bias: &DevBuf,
        state: &DevBuf,
        core_out: &DevBuf,
        row_start: &DevBuf,
        row_count: &DevBuf,
        row_id: &DevBuf,
        n_rows: u32,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        // 本步 token 总数（q/k 的行数，= Σ row_count）：前置 l2norm kernel 按
        // [tokens, k_heads] 逐对算倒数；deltanet 的 g 就是该缓冲的行号
        tokens: u32,
        l2_eps: f32,
        // [rows] 逐 span 标志：0 = 逐段单行状态（现有语义）；1 = 逐位置行
        // （token i 读行 row_id[r]+i、写行 +i+1——MTP 验证链的零拷贝回滚；
        // 仅 kd=vd=128）。逐 span 而非整批标量：链段与普通行同批
        specs: &DevBuf,
        // v 行切分数（B 档 chunked；1 = 不切）。调用方按批形给建议值
        // （同 span token 数 t>1 才值得切），run 内按容量/几何上限钳制
        vsplit: u32,
        stream: &Stream,
    );
    fn name(&self) -> &'static str;
}

pub struct DeltanetCuda<'a> {
    gpu: &'a Gpu,
    f: Func,
    /// 前置 l2norm kernel（`dn_qk_l2norm`）
    l2f: Func,
    /// E2 warp 归约 scratch：[DELTA_BLOCK_CAP, 4096] f32（kvm/o 两区各 2048），
    /// block 按 blockIdx 私有槽——resolve 一次预分配，capture 期零分配
    scratch: DevBuf<'a>,
    /// invq/invk 缓冲：[DELTA_L2_TOKENS_CAP × DELTA_L2_KHEADS_CAP, 2] f32
    /// （= 每 token 每层 2×k_heads 个 f32：kh=16 时 128 B/token/层，250 token
    /// 一个 prefill 步 = 32KB；24 层复用同一块，不按层乘）。resolve 一次
    /// 预分配——capture 期零分配，跨层复用（前置 kernel 每层全量覆写本层
    /// tokens×k_heads 段后才被主 kernel 读，无别名风险）
    inv: DevBuf<'a>,
}

impl DeltanetOp for DeltanetCuda<'_> {
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        q: &DevBuf,
        k: &DevBuf,
        v: &DevBuf,
        a: &DevBuf,
        b: &DevBuf,
        a_log: &DevBuf,
        dt_bias: &DevBuf,
        state: &DevBuf,
        core_out: &DevBuf,
        row_start: &DevBuf,
        row_count: &DevBuf,
        row_id: &DevBuf,
        n_rows: u32,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        tokens: u32,
        l2_eps: f32,
        specs: &DevBuf,
        vsplit: u32,
        stream: &Stream,
    ) {
        // ── 前置：q/k 的逐 (token, k 头) l2norm 倒数（一次 launch）──
        // 主 kernel 的每个 (行, 头) 有 vsplit 个块、每块 nwarps 个 warp，
        // 早先各自把 q/k 的 k_dim 维扫一遍求同一份 invq/invk——k[j]/q[j]
        // 对整条 v 维共享，这份重复是切块后剩下的主项。归约树（j 升序单链）
        // 原样搬进前置 kernel，位级不变。
        assert!(
            tokens as usize <= DELTA_L2_TOKENS_CAP
                && k_heads as usize <= DELTA_L2_KHEADS_CAP,
            "deltanet l2norm 前置：tokens={tokens} × k_heads={k_heads} 超缓冲容量 \
             {DELTA_L2_TOKENS_CAP}×{DELTA_L2_KHEADS_CAP}"
        );
        let n = tokens * k_heads;
        let l2args = crate::kernel_args![*q, *k, self.inv, tokens, k_heads, k_dim, l2_eps];
        self.gpu.launch(&self.l2f, n.div_ceil(DELTA_L2_BLOCK_DIM), DELTA_L2_BLOCK_DIM, &l2args, stream);
        // 块几何自带：kd=vd=128 走 E2（一块=(行, v 头)×128 线程=4 warp，
        // warp 顺序吃 v 行、lane 持 j 切片 u64——warp 指令 256B 全合并；
        // kvm/o 归约走 scratch 两级折叠）；其余一块 = (行, v 头)、
        // block 宽 = v_dim（kernel 侧同条件分支）
        let e2 = k_dim == 128 && v_dim == 128;
        // ── B 档 chunked 的自适应门（仅 E2）──
        // ① 容量：块数 rows×v_heads×vsplit ≤ DELTA_BLOCK_CAP（块私有 scratch 槽）
        // ② 铺满：块数已 ≥ DELTA_SPLIT_MAX_BLOCKS 时切块只加重复读（每块每
        //    token 各载一遍 q/k 切片），不加并行
        // ③ 几何：vd 可被 vsplit 整除，且每 warp 行数 = vd/(vsplit·nwarps) 是
        //    kernel 侧批 4 的整数倍（rows_pw % 4 == 0）
        let sp = if !e2 {
            1
        } else {
            let base = (n_rows as usize * v_heads as usize).max(1);
            let cap = DELTA_BLOCK_CAP / base;
            // 档位 = min(目标块数/base, 调用方给的上限)：`vsplit` 是**上限**
            // （解码门控给 1 即永远不切），实际切多少看铺满程度
            let mut s = (DELTA_SPLIT_TARGET_BLOCKS / base)
                .min(vsplit as usize)
                .clamp(1, DELTA_SPLIT);
            // kernel 侧批大小 = min(16, rows_pw)、步进按 4 的整数倍 ⇒ rows_pw ∈
            // {4, 8, 16, 32}（vd=128 与 2 幂 warp 数下必为 2 幂）
            let ok = |s: usize| {
                if v_dim as usize % s != 0 {
                    return false;
                }
                let rows_pw = (v_dim as usize / s) / (DELTA_SPLIT_BLOCK_DIM / 32);
                rows_pw >= 4 && rows_pw <= 32 && rows_pw.is_power_of_two()
            };
            if base >= DELTA_SPLIT_MAX_BLOCKS {
                s = 1;
            }
            while s > 1 && (s > cap || !ok(s)) {
                s -= 1;
            }
            s
        };
        let sp = sp as u32; // 活过 launch 的 place（kernel_args! 取地址，临时值会被栈槽复用）
        let args = crate::kernel_args![*q, *k, *v, *a, *b, *a_log, *dt_bias, *state, *core_out,
            *row_start, *row_count, *row_id, n_rows, k_heads, v_heads, k_dim, v_dim, self.inv,
            self.scratch, *specs, sp];
        let grid = n_rows * v_heads * sp;
        if e2 {
            assert!((grid as usize) <= DELTA_BLOCK_CAP,
                "deltanet 块数 {grid} 超 scratch 容量 {DELTA_BLOCK_CAP}");
        }
        let bdim = if !e2 {
            v_dim
        } else if sp > 1 {
            DELTA_SPLIT_BLOCK_DIM as u32
        } else if (grid as usize) <= DELTA_WIDE_GRID_MAX {
            DELTA_WIDE_BLOCK_DIM as u32
        } else {
            DELTA_BLOCK_DIM as u32
        };
        self.gpu.launch(&self.f, grid, bdim, &args, stream);
    }
    fn name(&self) -> &'static str {
        "deltanet/cuda"
    }
}

/// 带门 RMSNorm（deltanet 输出）：w 是 F32 权重（Qwen3.5 checkpoint 原生）
pub trait RmsNormGatedOp {
    fn run(&self, x: &DevBuf, z: &DevBuf, w: &DevBuf, out: &DevBuf, rows: u32, hidden: u32, eps: f32, stream: &Stream);
    fn name(&self) -> &'static str;
}

pub struct RmsNormGatedCuda<'a> {
    gpu: &'a Gpu,
    f: Func,
    /// 块归约 scratch（同 RmsNormZcCuda）
    scratch: DevBuf<'a>,
}

impl RmsNormGatedOp for RmsNormGatedCuda<'_> {
    fn run(&self, x: &DevBuf, z: &DevBuf, w: &DevBuf, out: &DevBuf, rows: u32, hidden: u32, eps: f32, stream: &Stream) {
        let args = crate::kernel_args![*x, *z, *w, *out, hidden, eps, rows, self.scratch];
        let b = hidden.min(256).next_power_of_two(); // block-per-row（同 rms_norm_zc）
        assert!(
            rows as usize * b as usize * 4 <= self.scratch.bytes(),
            "rms_norm_gated rows={rows} 超 scratch 容量 {} 槽",
            self.scratch.bytes() / 4
        );
        self.gpu.launch(&self.f, rows, b, &args, stream);
    }
    fn name(&self) -> &'static str {
        "rms_norm_gated/cuda"
    }
}

/// 设备端行 argmax（greedy 采样）：全序归约确定，等值取小下标
pub trait ArgmaxRowsOp {
    /// x: [row_total, n] bf16；rows: [s] 采样的全局行号；scratch: [s,512] u32
    fn run(&self, x: &DevBuf, rows: &DevBuf, scratch: &DevBuf, out: &DevBuf, n: u32, s: u32, stream: &Stream);
    fn name(&self) -> &'static str;
}

pub struct ArgmaxRowsCuda<'a> {
    gpu: &'a Gpu,
    f: Func,
}

impl ArgmaxRowsOp for ArgmaxRowsCuda<'_> {
    fn run(&self, x: &DevBuf, rows: &DevBuf, scratch: &DevBuf, out: &DevBuf, n: u32, s: u32, stream: &Stream) {
        let args = crate::kernel_args![*x, *rows, *scratch, *out, n, s];
        self.gpu.launch(&self.f, s, 256, &args, stream);
    }
    fn name(&self) -> &'static str {
        "argmax_rows/cuda"
    }
}

/// 设备端采样（top-k / top-p）：逐行 temp/top_k/top_p/seed + 设备 RNG 状态。
/// temp ≤ 0 走 kernel 内的 argmax 分支（与 `argmax_rows` 同算法同位，位同）；
/// top_k ≤ 0（全词表）**不在设备侧支持**——248K 项全排序不可接受（top-p
/// 需要降序全序），主机侧在 `set_sampling`/`step` 就拒，kernel 落错误标记
/// 0xFFFFFFFF 兜底，绝不静默退化成"取前 64"。
pub trait SampleRowsOp {
    /// x: [row_total, n] bf16；rows: [s] 采样行号；slots: [s] RNG 槽（= 请求行号）；
    /// temp/top_k/top_p/seed: [s]；rng: [槽数] u64（0 = 未初始化，由 seed 派生）；
    /// out: [s]（0xFFFFFFFF = 参数非法）
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        x: &DevBuf,
        rows: &DevBuf,
        slots: &DevBuf,
        temp: &DevBuf,
        top_k: &DevBuf,
        top_p: &DevBuf,
        seed: &DevBuf,
        rng: &DevBuf,
        out: &DevBuf,
        n: u32,
        s: u32,
        stream: &Stream,
    );
    fn name(&self) -> &'static str;
}

/// 采样器 scratch 行容量：= exec 的 logits 行上限（T_MAX=512）× 每行槽字数
/// （34MB：block=1024 lane × 8 键 u64 池 = 64KB/行；open 期一次分配，捕获期零分配）
const SAMPLE_ROWS_CAP: usize = 512;
/// 每行 scratch 字数——与 kernels/ 的 `SMP_SLOT` 一一对应（改动须两侧同改）
const SAMPLE_SLOT: usize = 4352;
/// block 宽度——与 kernels/ 的 `SMP_T` 一一对应（= 每行一级归约的 lane 数）
const SAMPLE_T: u32 = 256;

pub struct SampleRowsCuda<'a> {
    gpu: &'a Gpu,
    f: Func,
    scratch: DevBuf<'a>,
}

impl SampleRowsOp for SampleRowsCuda<'_> {
    fn run(
        &self,
        x: &DevBuf,
        rows: &DevBuf,
        slots: &DevBuf,
        temp: &DevBuf,
        top_k: &DevBuf,
        top_p: &DevBuf,
        seed: &DevBuf,
        rng: &DevBuf,
        out: &DevBuf,
        n: u32,
        s: u32,
        stream: &Stream,
    ) {
        let need = s as usize * SAMPLE_SLOT * 4;
        assert!(
            need <= self.scratch.bytes(),
            "sample_rows s={s} 超 scratch 容量 {SAMPLE_ROWS_CAP} 行（{need} > {} 字节）",
            self.scratch.bytes()
        );
        let args = crate::kernel_args![
            *x, *rows, *slots, *temp, *top_k, *top_p, *seed, *rng, self.scratch, *out, n, s
        ];
        self.gpu.launch(&self.f, s, SAMPLE_T, &args, stream);
    }
    fn name(&self) -> &'static str {
        "sample_rows/cuda"
    }
}

/// "sm_89" → (8, 9)；"sm_120" → (12, 0)——PTX 命名法 minor 恒一位，
/// 末位拆 minor 其余为 major；非法输入 → None
pub fn parse_arch(s: &str) -> Option<(u32, u32)> {
    let digits = s.strip_prefix("sm_")?;
    let mut chars = digits.chars();
    let min = chars.next_back()?.to_digit(10)?;
    let maj_str: String = chars.collect();
    if maj_str.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some((maj_str.parse().ok()?, min))
}

impl RmsNormOp for RmsNormImpl<'_> {
    fn run(
        &self,
        x: &DevBuf,
        w: &DevBuf,
        out: &DevBuf,
        rows: u32,
        hidden: u32,
        eps: f32,
        stream: &Stream,
    ) {
        // launch 几何分档，kernel 本体同一份（数值序不变——o5 对拍的依据）
        let (gpu, f, grid, block) = match self {
            RmsNormImpl::Row { gpu, f } => (*gpu, f, rows, 1),
            RmsNormImpl::Blocked { gpu, f, block } => {
                (*gpu, f, rows.div_ceil(*block), *block)
            }
        };
        // x/w/out 形参是 &DevBuf，解引用成 place 再取址（宏按 &$arg 取位）
        let args = crate::kernel_args![*x, *w, *out, hidden, eps, rows];
        gpu.launch(f, grid, block, &args, stream);
    }

    fn name(&self) -> &'static str {
        match self {
            RmsNormImpl::Row { .. } => "rms_norm/cuda-row",
            RmsNormImpl::Blocked { .. } => "rms_norm/cuda-blocked",
        }
    }
}

// ───────────────────── M0 清单其余算子（§4）：一维几何，槽位预留档位演化 ─────────────────────

/// rope：half-split（Llama/Qwen 形态）。cos/sin 表是**输入**（host 生成上传，
/// 图捕获友好）——kernel 只做 apply，表生成在 reference::rope_table
pub trait RopeOp {
    /// q/k: [tokens, rot_dim] bf16；cos/sin: [max_pos, rot_dim/2] f32；
    /// pos: [tokens] u32；out 同 q/k 形状
    fn run(
        &self,
        q: &DevBuf,
        k: &DevBuf,
        cos: &DevBuf,
        sin: &DevBuf,
        pos: &DevBuf,
        out_q: &DevBuf,
        out_k: &DevBuf,
        tokens: u32,
        rot_dim: u32,
        stream: &Stream,
    );
    fn name(&self) -> &'static str;
}

pub struct RopeCuda<'a> {
    gpu: &'a Gpu,
    f: Func,
}

impl RopeOp for RopeCuda<'_> {
    fn run(
        &self,
        q: &DevBuf,
        k: &DevBuf,
        cos: &DevBuf,
        sin: &DevBuf,
        pos: &DevBuf,
        out_q: &DevBuf,
        out_k: &DevBuf,
        tokens: u32,
        rot_dim: u32,
        stream: &Stream,
    ) {
        let half = rot_dim / 2;
        let n = tokens * half;
        let args = crate::kernel_args![*q, *k, *cos, *sin, *pos, *out_q, *out_k, rot_dim, tokens];
        self.gpu.launch(&self.f, n.div_ceil(256), 256, &args, stream);
    }
    fn name(&self) -> &'static str {
        "rope/cuda"
    }
}

/// silu：out = x·sigmoid(x)。sigmoid 的 exp 走自控 exp2 多项式
/// （kernels/reference 同字面量同 Horner 序——禁 MUFU 近似，对齐可达）
pub trait SiluOp {
    fn run(&self, x: &DevBuf, out: &DevBuf, n: u32, stream: &Stream);
    fn name(&self) -> &'static str;
}

pub struct SiluCuda<'a> {
    gpu: &'a Gpu,
    f: Func,
}

impl SiluOp for SiluCuda<'_> {
    fn run(&self, x: &DevBuf, out: &DevBuf, n: u32, stream: &Stream) {
        let args = crate::kernel_args![*x, *out, n];
        self.gpu.launch(&self.f, n.div_ceil(256), 256, &args, stream);
    }
    fn name(&self) -> &'static str {
        "silu/cuda"
    }
}

/// W8A8 动态量化：逐 token amax → scale=amax/127 → RNE 半偶取整。
/// i8 q + f32 scale 双输出（weight-only 之外，激活量化喂 W8A8 GEMM）
pub trait W8A8QuantOp {
    /// x: [tokens, hidden] bf16；q: [tokens, hidden] i8；scale: [tokens] f32
    fn run(&self, x: &DevBuf, q: &DevBuf, scale: &DevBuf, tokens: u32, hidden: u32, stream: &Stream);
    fn name(&self) -> &'static str;
}

pub struct QuantCuda<'a> {
    gpu: &'a Gpu,
    f: Func,
}

impl W8A8QuantOp for QuantCuda<'_> {
    fn run(&self, x: &DevBuf, q: &DevBuf, scale: &DevBuf, tokens: u32, hidden: u32, stream: &Stream) {
        // 一线程一 token（amax 顺序扫在行内完成——与参考同序）
        let args = crate::kernel_args![*x, *q, *scale, hidden, tokens];
        self.gpu.launch(&self.f, tokens.div_ceil(256), 256, &args, stream);
    }
    fn name(&self) -> &'static str {
        "w8a8_quant/cuda"
    }
}

/// KV scatter write：新 token 的 K/V 写入 paged KV。
/// 槽位寻址 slot = block_table[seq·bpr + pos/bs]·bs + pos%bs，槽内
/// [heads, head_dim] —— 与 wein-memory plan_kv 的 token 记账同构
pub trait KvScatterOp {
    /// src_k/src_v: [tokens, head_elems] bf16；k_dst/v_dst: paged 池
    /// （未触及槽位原样保留——池属调度层所有，kernel 只写自己的槽）
    fn run(
        &self,
        src_k: &DevBuf,
        src_v: &DevBuf,
        block_table: &DevBuf,
        seq_ids: &DevBuf,
        positions: &DevBuf,
        k_dst: &DevBuf,
        v_dst: &DevBuf,
        bpr: u32,
        block_size: u32,
        head_elems: u32,
        tokens: u32,
        stream: &Stream,
    );
    /// 同 `run`，但写入 **f32 池**（源仍 bf16，写入端精确加宽——
    /// 池里每一位都是原 bf16 值的精确表示，故不改任何数值）
    #[allow(clippy::too_many_arguments)]
    fn run_f32(
        &self,
        src_k: &DevBuf,
        src_v: &DevBuf,
        block_table: &DevBuf,
        seq_ids: &DevBuf,
        positions: &DevBuf,
        k_dst: &DevBuf,
        v_dst: &DevBuf,
        bpr: u32,
        block_size: u32,
        head_elems: u32,
        tokens: u32,
        stream: &Stream,
    );
    fn name(&self) -> &'static str;
}

pub struct KvScatterCuda<'a> {
    gpu: &'a Gpu,
    f: Func,
    /// f32 池写入口（`kv_scatter_f32`）
    f32v: Func,
}

impl KvScatterOp for KvScatterCuda<'_> {
    fn run(
        &self,
        src_k: &DevBuf,
        src_v: &DevBuf,
        block_table: &DevBuf,
        seq_ids: &DevBuf,
        positions: &DevBuf,
        k_dst: &DevBuf,
        v_dst: &DevBuf,
        bpr: u32,
        block_size: u32,
        head_elems: u32,
        tokens: u32,
        stream: &Stream,
    ) {
        let n = tokens * head_elems;
        let args = crate::kernel_args![*src_k, *src_v, *block_table, *seq_ids, *positions,
            *k_dst, *v_dst, bpr, block_size, head_elems, tokens];
        self.gpu.launch(&self.f, n.div_ceil(256), 256, &args, stream);
    }
    fn run_f32(
        &self,
        src_k: &DevBuf,
        src_v: &DevBuf,
        block_table: &DevBuf,
        seq_ids: &DevBuf,
        positions: &DevBuf,
        k_dst: &DevBuf,
        v_dst: &DevBuf,
        bpr: u32,
        block_size: u32,
        head_elems: u32,
        tokens: u32,
        stream: &Stream,
    ) {
        let n = tokens * head_elems;
        let args = crate::kernel_args![*src_k, *src_v, *block_table, *seq_ids, *positions,
            *k_dst, *v_dst, bpr, block_size, head_elems, tokens];
        self.gpu.launch(&self.f32v, n.div_ceil(256), 256, &args, stream);
    }
    fn name(&self) -> &'static str {
        "kv_scatter/cuda"
    }
}

/// varlen paged attention：一 block 一 (token, q_head)（block=head_dim
/// 线程四相位协作，逐元素累加序与 reference 全同——o12 位对齐零漂移）。
/// pos 位置的 query 因果只看 0..=pos；GQA：kv 头 = qh // group
pub trait PagedAttnOp {
    /// q: [tokens, q_heads, head_dim] bf16；k/v_cache: paged 池；out 同 q。
    /// q_scale 为输入（host 算 1/sqrt(head_dim)）——kernel 不做 sqrt
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        q: &DevBuf,
        k_cache: &DevBuf,
        v_cache: &DevBuf,
        block_table: &DevBuf,
        seq_ids: &DevBuf,
        positions: &DevBuf,
        out: &DevBuf,
        q_scale: f32,
        bpr: u32,
        block_size: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        tokens: u32,
        stream: &Stream,
    );
    fn name(&self) -> &'static str;
}

pub struct PagedAttnCuda<'a> {
    gpu: &'a Gpu,
    f: Func,
    /// 块归约 scratch：[ATTN_BLOCK_CAP, ATTN_SLOT_CAP+2] f32（分值/p + 分母槽），
    /// block 按 blockIdx 私有槽——resolve 一次预分配，capture 期零分配
    scratch: DevBuf<'a>,
    /// 每块分值槽宽（kernel 的 `slot_cap`）：pos+1 ≤ 此值走直路（四相位），
    /// 超出走位级同的流式路——容量只影响 scratch 用量，不再是序列长上限
    slot_cap: u32,
    /// 长序列分块档（`paged_attn_tile` + `attn_combine`）：按 (kv 头,
    /// token 片, j 分片) 组织，一块内 `ntok×group` 个 q 行共享同一段 KV。
    /// 默认路径（ctx ≤ 256）不碰它——见 `run_tile`
    ft: Func,
    /// 两遍档 f32 池入口（`paged_attn_tile_f32`；回退口 + f32 池用）
    ftf32: Func,
    /// 单遍在线 softmax 档（`paged_attn_online`）：与 `ft` **同几何同 scratch
    /// 布局**，只把"两遍（先求 max 再复算 dot）"换成一遍在线合并。长序列默认
    /// 走它（`run_tile_online`）；`run_tile` 仍是两遍档（位级资产 + 回退口，
    /// exec 侧 `WEIN_ATTN_ONLINE=0`）
    fo: Func,
    /// 单遍档的几何/dtype 变体（见 `AttnCfg`）：ND 维每线程 × RPT 行每线程 ×
    /// 池 dtype。基线与变体共用同一 ABI 与 scratch，故只是入口不同。
    f_nd8: Func,
    f_nd16: Func,
    f_r2: Func,
    f_r4: Func,
    f_nd8_r4: Func,
    f_n16_r4: Func,
    f_n16_r2: Func,
    f_f32: Func,
    f_f32_nd8: Func,
    f_f32_r4: Func,
    f_f32_r2: Func,
    fc: Func,
    /// 块档（**O/P 落块私有 scratch**、Br 参数化）：prefill 长序列专用入口。
    /// bf16/f32 池两档，ABI 与单遍档只差一个 `stride`（每块 scratch 槽数）
    f_blk: Func,
    f_blk_f32: Func,
    /// 分块档块私有 scratch：[ATTN_TILE_BLOCK_CAP, ATTN_TILE_SLOTS] f32
    tscratch: DevBuf<'a>,
    /// j 分片偏结果（jsplit>1 档）：acc [rows, split, hd] f32 + (m,l) [rows, split, 2]
    pacc: DevBuf<'a>,
    pml: DevBuf<'a>,
    /// 块档块私有 scratch（[BLK_BLOCK_CAP, stride] f32，stride 由几何裁决）
    bscratch: DevBuf<'a>,
    /// 块档 j 分片偏结果（rows 上限 BLK_PART_ROWS，比旧档大 4 倍）
    pacc_b: DevBuf<'a>,
    pml_b: DevBuf<'a>,
}

/// attn 块数容量上限：T_MAX(512) × q_heads(32)——超容显式 assert
const ATTN_BLOCK_CAP: usize = 16384;
/// 分值槽宽（每块 f32 槽数）：256 保 M0 默认路径逐字节不变；更长的位置
/// 由流式路承担。抬到 `max_pos` 可让长上下文也走直路（scratch = 块数 ×
/// (cap+2) × 4B：cap=4096 时 ≈ 256MB——默认不付这份显存）
pub const ATTN_SLOT_CAP: usize = 256;

// ── 分块档（长序列）容量 ──
/// 块数上限：kv 头 × token 片 × j 分片。两档都放得下：Qwen3.5-4B 形
/// (4 kv × 128 片 × 1) = 512；(32q/16kv/128) 形的 256-token chunk 是
/// 16 × 64 = 1024。scratch = 1024 × ATTN_TILE_SLOTS × 4B ≈ 21MB（一次分配）
const ATTN_TILE_BLOCK_CAP: usize = 1024;
/// 一块内 q 行数上限 = ntok(即 **BR**，一块覆盖的 q token 数) × group。
/// 32 → 128 是 BR 维度（前缀 KV 的块内复用）的容量前提：group=4 时 BR 可到
/// 32（rows=128）。几何仍由 `online_geom` 的 bdim/寄存器账夹住
/// （映射用：bdim = rows/rpt × hd/nd ≤ 512）
const ATTN_TILE_ROWS_CAP: usize = 128;
/// p 暂存的 j chunk 宽（每 chunk 一次屏障）
const ATTN_TILE_CHUNK: usize = 256;
/// 每块 scratch 槽数上限：max 归约(rows_nom×nv = bdim×rpt ≤ 512×4 = 2048) +
/// p(chunk×rows≤32768) + den(rows≤128) + 槽位表(ntok≤ROWS_CAP × chunk≤512
/// = 65536)——与 kernel 侧 `TILE_SLOT` 同值。首项按 RPT>1 的 rows_nom×nv
/// 定尺（不是 bdim）
const ATTN_TILE_SLOTS: usize = 2048
    + ATTN_TILE_CHUNK * ATTN_TILE_ROWS_CAP
    + ATTN_TILE_ROWS_CAP
    + ATTN_TILE_SLOT_TAB;
/// 槽位表定尺 = ntok(BR) 上限 × chunk 上限（与 kernel 侧 `SLOT_TAB` 同值）
const ATTN_TILE_SLOT_TAB: usize = 32 * 512;
/// j 分片数上限（decode 形：批小、块不够用时按 j 切片换并行度）
pub const ATTN_TILE_SPLIT_CAP: usize = 16;
/// 分片档的 q 行容量（jsplit>1 只在小批发生：rows = tokens×q_heads ≤ 1024）
const ATTN_TILE_PART_ROWS: usize = 1024;
/// 分片档 head_dim 容量（偏结果表按它定尺）
const ATTN_TILE_HD_CAP: usize = 256;
/// 分块档最小并行目标：块数低于它就往上加 j 分片
const ATTN_TILE_TARGET_BLOCKS: usize = 128;
/// 分块档线程上限的**兜底**值：真正的夹子是寄存器账（`attn_bdim_cap` =
/// 65536/regs 向下取 32 的整）。512 = tile(96 regs)/tile_f32(102 regs) 的
/// 65536/regs 都不足 640（向上取到 1024 线程会撞 "too many resources"）。
/// 单遍各档 regs ≥86 ⇒ 1024 线程（需 ≤64 regs）在所有档上都不可达
/// （见 `attn_regs` 与 A2 的 maxrregcount 实测：64 档 162 spill stores）。
const ATTN_TILE_BDIM_CAP: usize = 512;

// ── 块档（Br 参数化；O/P 落块私有 scratch）的容量账 ──
/// 块档块私有 scratch 池上限（f32 槽数）= 256MB。定尺依据：单块 slot =
/// rows×hd(O) + rows×cap(P) + br×cap(槽位表) + 5×rows；T=256 的 prefill chunk
/// 上最贵的一档（Br=128、cap=1024）= 3.16MB/块，8 块 = 25MB；j 分片补并行度
/// 时块数乘上去，池按 256MB 封顶（余量给 cap=1024 × 16 片形）
pub const BLK_SCRATCH_FLOATS: usize = 64 << 20;
/// 块档 j 分片偏结果的 q 行容量：prefill chunk 形 tokens×q_heads（256×16）
pub const BLK_PART_ROWS: usize = 4096;
/// 块档 j 分片数上限（每片至少一个 j 段；块数目标见 `BLK_TARGET_BLOCKS`）
pub const BLK_SPLIT_CAP: usize = 16;
/// 块档并行目标块数：Br 大 ⇒ 块少，靠 j 分片补。定尺依据是实测——128 regs
/// ⇒ 2 块/SM（512 线程/SM）；实测 128 块（1 块/SM、8 warp）比 256 块
/// （2 块/SM、16 warp）慢 19%（br=64/cap=256：21.2 vs 17.8 ms/步）
pub const BLK_TARGET_BLOCKS: usize = 256;
/// 块档线程上限：`paged_attn_blk` 实测 128 regs（0 spill）⇒ 65536/128 = 512
const BLK_BDIM_CAP: usize = 512;

// ── 单遍档的几何 / dtype 旋钮（A/B 判别仪器） ──

/// KV 池 dtype。**只改载入形状，不改数值**：写入端 bf16→f32 是精确加宽，
/// 读出端算术序一字不改 ⇒ 两种池的 attention 输出逐位相同。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KvDtype {
    Bf16,
    F32,
}

/// 单遍 attention 的几何配置：
/// · `rpt`：**每线程 q 行数**＝"每 KV 字节服务的 q 行数"，K/V 的 L2 载入次数
///   直接 ÷rpt（唯一有效的流量杠杆；受 group 整除与寄存器预算约束）
/// · `nd`：每线程维数（4/8/16）＝V 段切分与载入粒度
/// · `ntok_max`：单块 token 片上限，即 **BR**（一块覆盖的 q token 数，
///   行数 = BR×group）。KV 前缀的块内重读次数 ∝ 1/BR——这是"KV 前缀要流
///   几遍"的唯一杠杆。但它同时把块数除以 BR（块数 = kv 头 × tokens/BR），
///   而本内核的墙是"每 SM 的常驻线程数"（寄存器夹在 65536/regs）⇒ BR 抬过
///   `SM 数 /(kv 头 × 32/group)` 之后并行度掉得比流量快，实测净负（见
///   `bench/attn_br_sweep.sh` 与提交说明的表）
/// · `bdim_max`：单块线程上限（实际再被寄存器账夹住）
/// · `kv`：池 dtype
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AttnCfg {
    pub nd: usize,
    pub rpt: usize,
    pub ntok_max: usize,
    pub bdim_max: usize,
    pub kv: KvDtype,
}

impl AttnCfg {
    /// 现值（W2 单遍档首发形态）：4 维/线程、1 行/线程、bf16 池
    pub fn baseline() -> Self {
        Self { nd: 4, rpt: 1, ntok_max: 4, bdim_max: ATTN_TILE_BDIM_CAP, kv: KvDtype::Bf16 }
    }
}

/// 单遍档的落地几何（`online_geom` 的返回值；测试/探针打印用）
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct OnlineGeom {
    pub ntok: usize,
    pub nd: usize,
    pub rpt: usize,
    pub nv: usize,
    pub rows_nom: usize,
    pub bdim: usize,
    pub ntile: usize,
    pub jsplit: usize,
}

/// 块档（Br 参数化、O/P 落块私有 scratch）的几何旋钮。
/// 与单遍档的区别：累加器不再占寄存器，故 Br 只受 **scratch 与并行度** 约束。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlkCfg {
    /// 一块覆盖的 q token 数（Br）；行数 = br × group
    pub br: usize,
    /// j 片宽（每片一次 rescale + 每片一次 O 往返）
    pub jchunk: usize,
    /// 线程/块：ngrp = bdim/nv 就是**每块内 KV 的重读次数**
    pub bdim: usize,
    /// j 分片数；0 = 按 `BLK_TARGET_BLOCKS` 自动补并行度
    pub jsplit: usize,
    pub kv: KvDtype,
}

impl BlkCfg {
    /// 默认形态：Br=32（256 行）、j 片 256、256 线程（ngrp=4 ⇒ 4 次重读）
    pub fn baseline() -> Self {
        Self { br: 32, jchunk: 256, bdim: 256, jsplit: 0, kv: KvDtype::Bf16 }
    }
}

/// 块档的落地几何（`blk_geom` 的返回值；探针/报告口径）
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlkGeom {
    pub br: usize,
    pub nd: usize,
    pub group: usize,
    pub nv: usize,
    /// 线程组数 = bdim/nv = **块内 KV 重读次数**
    pub ngrp: usize,
    /// 每组线程负责的 pass 数（1 pass = 1 个 token 的 group 行）
    pub npass: usize,
    pub rows_nom: usize,
    pub bdim: usize,
    pub ntile: usize,
    pub jsplit: usize,
    /// j 片宽（= kernel 的 `chunk`）
    pub cap: usize,
    /// 每块 scratch 槽数（f32）：O + P + 5×rows + 槽位表
    pub stride: usize,
}

impl BlkGeom {
    pub fn nblk(&self, kv_heads: u32) -> usize {
        kv_heads as usize * self.ntile * self.jsplit
    }
    /// 块档 scratch 实付字节（块数 × 每块字节）
    pub fn scratch_bytes(&self, kv_heads: u32) -> usize {
        self.nblk(kv_heads) * self.stride * 4
    }
}

/// 每档 kernel 的寄存器用量（ptxas -v @sm_89 实测；改 kernel 后须重测）。
/// 作用：把 bdim 夹在 65536/regs 之内——超了就 "too many resources" 起不来。
const fn attn_regs(nd: usize, rpt: usize, f32_kv: bool) -> usize {
    match (nd, rpt, f32_kv) {
        (4, 1, false) => 88,
        (8, 1, false) => 86,
        (16, 1, false) => 100,
        (4, 2, false) => 128,
        // qb 由 `[usize; RPT]` 改基址+常量偏移后 r4 从 168 落到 128
        (4, 4, false) => 128,
        (8, 4, false) => 168,
        (16, 4, false) => 167,
        (16, 2, false) => 159,
        (4, 1, true) => 96,
        (8, 1, true) => 96,
        (4, 2, true) => 134,
        (4, 4, true) => 168,
        _ => 128,
    }
}

/// 某档的寄存器用量（探针/报告口径；见 `attn_regs`）
pub fn attn_regs_of(cfg: AttnCfg) -> usize {
    attn_regs(cfg.nd, cfg.rpt, cfg.kv == KvDtype::F32)
}

/// 该档的线程上限 = min(全局上限, 65536/regs 向下取 32 的整)
const fn attn_bdim_cap(nd: usize, rpt: usize, f32_kv: bool) -> usize {
    let cap = 65536 / attn_regs(nd, rpt, f32_kv);
    let cap = cap / 32 * 32;
    if cap < ATTN_TILE_BDIM_CAP {
        cap
    } else {
        ATTN_TILE_BDIM_CAP
    }
}

impl PagedAttnCuda<'_> {
    /// 测试用：偏结果表设备指针（pacc, pml, tscratch）
    pub fn part_ptrs(&self) -> (u64, u64, u64) {
        (self.pacc.device_ptr(), self.pml.device_ptr(), self.tscratch.device_ptr())
    }

    /// 分值槽宽覆盖（测试/容量档口用；只能缩小——分配按 ATTN_SLOT_CAP 一次）
    pub fn set_slot_cap(&mut self, cap: usize) {
        assert!(cap + 2 <= ATTN_SLOT_CAP + 2, "slot_cap {cap} 超分配容量 {ATTN_SLOT_CAP}");
        self.slot_cap = cap as u32;
    }

    /// 分块档几何裁决（纯函数：测试与 exec 共用同一份算式，附图性质）
    /// 返回 (ntok, nv, rows_nom, bdim, ntile, jsplit)。
    pub fn tile_geom(q_heads: u32, kv_heads: u32, head_dim: u32, tokens: u32)
        -> (usize, usize, usize, usize, usize, usize)
    {
        let group = (q_heads / kv_heads) as usize;
        let hd = head_dim as usize;
        let nv = hd / 4; // 4 维/线程（u64 载 4×bf16）
        // token 片：最多 4 个 token，且 rows = ntok×group ≤ ROWS_CAP，
        // 线程数 rows×hd/4 ≤ BDIM_CAP（寄存器上限，见常量注释）
        // 寄存器账：两遍档 tile=96 / tile_f32=102 regs，取更紧的一档做夹子
        let bdim_cap = ATTN_TILE_BDIM_CAP.min(attn_bdim_cap(4, 1, true));
        let mut ntok = 4.min(tokens as usize).max(1);
        while ntok > 1
            && (ntok * group > ATTN_TILE_ROWS_CAP || ntok * group * nv > bdim_cap)
        {
            ntok -= 1;
        }
        let rows_nom = ntok * group;
        let bdim = rows_nom * nv;
        let ntile = (tokens as usize + ntok - 1) / ntok;
        // j 分片：块数不足时按 j 切（decode 形）；prefill 形落 jsplit=1（位级同）
        let base = kv_heads as usize * ntile;
        // j 分片只在**小批**（偏结果表装得下 tokens×q_heads 行）时开：大 chunk
        // 的块数本就够，开分片只会把行数顶穿 PART_ROWS（曾实测 6009 的 121
        // token chunk：ntok=4 ⇒ ntile=31 ⇒ base=124 <128 触发 jsplit=2，
        // 行数 121×16=1936 > 1024）
        let mut jsplit = 1usize;
        if tokens as usize * q_heads as usize <= ATTN_TILE_PART_ROWS {
            while jsplit * base < ATTN_TILE_TARGET_BLOCKS && jsplit < ATTN_TILE_SPLIT_CAP {
                jsplit += 1;
            }
        }
        (ntok, nv, rows_nom, bdim, ntile, jsplit)
    }

    /// 长序列分块档（两遍）：block = (kv 头, token 片, j 分片)。
    /// 与 `run` 的 ABI 同出参（out 同形），调用方无需知道几何。
    /// jsplit=1 → 直接落 out（与 `run` **逐位相同**）；jsplit>1 → 偏结果 +
    /// `attn_combine` 合并（容差档：分段求和改结合律）。
    /// 语义冻结：这是位级资产那一条（o30）；新代码默认走 `run_tile_online`，
    /// 本函数保留为回退口（exec 侧 `WEIN_ATTN_ONLINE=0`）。
    #[allow(clippy::too_many_arguments)]
    pub fn run_tile(
        &self,
        q: &DevBuf,
        k_cache: &DevBuf,
        v_cache: &DevBuf,
        block_table: &DevBuf,
        seq_ids: &DevBuf,
        positions: &DevBuf,
        out: &DevBuf,
        q_scale: f32,
        bpr: u32,
        block_size: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        tokens: u32,
        stream: &Stream,
    ) {
        let g = Self::tile_geom(q_heads, kv_heads, head_dim, tokens);
        self.launch_tile(
            &self.ft, g.0, g.5, g.3, q, k_cache, v_cache, block_table, seq_ids, positions, out,
            q_scale, bpr, block_size, q_heads, kv_heads, head_dim, tokens, stream,
        );
    }

    /// 两遍档（回退口）跑 **f32 池**：与 `run_tile` 同几何同算术，只有 K/V
    /// 载入形状不同（与 bf16 逐位同值）
    #[allow(clippy::too_many_arguments)]
    pub fn run_tile_f32(
        &self,
        q: &DevBuf,
        k_cache: &DevBuf,
        v_cache: &DevBuf,
        block_table: &DevBuf,
        seq_ids: &DevBuf,
        positions: &DevBuf,
        out: &DevBuf,
        q_scale: f32,
        bpr: u32,
        block_size: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        tokens: u32,
        stream: &Stream,
    ) {
        let g = Self::tile_geom(q_heads, kv_heads, head_dim, tokens);
        self.launch_tile(
            &self.ftf32, g.0, g.5, g.3, q, k_cache, v_cache, block_table, seq_ids, positions, out,
            q_scale, bpr, block_size, q_heads, kv_heads, head_dim, tokens, stream,
        );
    }

    /// 长序列分块档（**单遍在线 softmax**）：几何、scratch、出参与 `run_tile`
    /// 逐字相同，kernel 内把"遍 1 求 max + 遍 2 复算 dot"并成一遍
    /// （K/V 各读一遍：流量 3 单位 → 2 单位）。合同落到**容差档**（跨 chunk
    /// rescale 改结合律，O33 钉 rel2 + 峰值归一）；端到端逐 token 一致由
    /// 长/短两路的对拍守着。
    /// jsplit>1 时偏量语义与两遍档一致（m_s = 本分片 max），`attn_combine`
    /// 一字不改直接合并。
    #[allow(clippy::too_many_arguments)]
    pub fn run_tile_online(
        &self,
        q: &DevBuf,
        k_cache: &DevBuf,
        v_cache: &DevBuf,
        block_table: &DevBuf,
        seq_ids: &DevBuf,
        positions: &DevBuf,
        out: &DevBuf,
        q_scale: f32,
        bpr: u32,
        block_size: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        tokens: u32,
        stream: &Stream,
    ) {
        self.run_tile_online_cfg(
            AttnCfg::baseline(), q, k_cache, v_cache, block_table, seq_ids, positions, out,
            q_scale, bpr, block_size, q_heads, kv_heads, head_dim, tokens, stream,
        );
    }

    /// 单遍档（**指定几何/dtype 档**）：`cfg` 决定进哪个 kernel 变体与落地
    /// 几何（`online_geom`）。同一 ABI/scratch/出参，故调用方只需换 cfg。
    #[allow(clippy::too_many_arguments)]
    pub fn run_tile_online_cfg(
        &self,
        cfg: AttnCfg,
        q: &DevBuf,
        k_cache: &DevBuf,
        v_cache: &DevBuf,
        block_table: &DevBuf,
        seq_ids: &DevBuf,
        positions: &DevBuf,
        out: &DevBuf,
        q_scale: f32,
        bpr: u32,
        block_size: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        tokens: u32,
        stream: &Stream,
    ) {
        let g = Self::online_geom(cfg, q_heads, kv_heads, head_dim, tokens);
        // 实体按**落地几何**选（rpt 由 online_geom 保证与 cfg 一致，此处仍按
        // g.rpt 取，避免将来加回落时再次错配）
        let f = self.online_func(AttnCfg { rpt: g.rpt, ..cfg });
        self.launch_tile(
            f, g.ntok, g.jsplit, g.bdim, q, k_cache, v_cache, block_table,
            seq_ids, positions, out, q_scale, bpr, block_size, q_heads, kv_heads, head_dim,
            tokens, stream,
        );
    }

    /// cfg → kernel 入口（每档一份单态实体，见 kernels/src 的导出表）
    fn online_func(&self, cfg: AttnCfg) -> &Func {
        match (cfg.nd, cfg.rpt, cfg.kv) {
            (4, 1, KvDtype::Bf16) => &self.fo,
            (8, 1, KvDtype::Bf16) => &self.f_nd8,
            (16, 1, KvDtype::Bf16) => &self.f_nd16,
            (4, 2, KvDtype::Bf16) => &self.f_r2,
            (4, 4, KvDtype::Bf16) => &self.f_r4,
            (8, 4, KvDtype::Bf16) => &self.f_nd8_r4,
            (16, 4, KvDtype::Bf16) => &self.f_n16_r4,
            (16, 2, KvDtype::Bf16) => &self.f_n16_r2,
            (4, 1, KvDtype::F32) => &self.f_f32,
            (8, 1, KvDtype::F32) => &self.f_f32_nd8,
            (4, 4, KvDtype::F32) => &self.f_f32_r4,
            (4, 2, KvDtype::F32) => &self.f_f32_r2,
            (nd, rpt, kv) => panic!("无此单遍档：nd={nd} rpt={rpt} kv={kv:?}"),
        }
    }

    /// 单遍档几何裁决（纯函数：探针/exec/测试共用一份算式）。
    ///
    /// 选法：**先要 rpt 大**（流量 ÷rpt），再要 ntok 大（块少、屏障少），
    /// 夹在 `rows_nom ≤ ROWS_CAP`、`bdim ≤ min(cfg.bdim_max, 寄存器账)` 之内。
    /// rpt 还须整除 group（行组不跨 token）。
    pub fn online_geom(
        cfg: AttnCfg,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        tokens: u32,
    ) -> OnlineGeom {
        let group = (q_heads / kv_heads) as usize;
        let hd = head_dim as usize;
        let nd = cfg.nd;
        assert!(nd == 4 || nd == 8 || nd == 16, "nd 须 4/8/16");
        assert!(hd % nd == 0, "head_dim {hd} 须整除 nd {nd}");
        let nv = hd / nd;
        let bdim_cap = cfg.bdim_max.min(attn_bdim_cap(nd, cfg.rpt, cfg.kv == KvDtype::F32));
        let ntok_cap = cfg
            .ntok_max
            .min(tokens as usize)
            .min(ATTN_TILE_ROWS_CAP / group)
            .max(1);
        // rpt **不回落**：kernel 实体是按 (nd, rpt) 单态的，几何若偷换成别的
        // rpt 就会拿 RPT 不符的实体去跑（实测 n8r4 在 decode 形回落 rpt=2 ⇒
        // 输出全错）。故 rpt 必须由 cfg 给定并可落地，落不了就是配置错。
        let rpt = cfg.rpt;
        assert!(
            rpt >= 1 && group % rpt == 0 && nd * rpt <= 64,
            "rpt 须整除 group({group}) 且 nd×rpt ≤ 64：cfg={cfg:?}"
        );
        // bdim 下限 32 = 一个 warp：decode 形（tokens=1、大 nd、大 rpt）只剩
        // 一组线程，1 warp/块也是合法几何（块数由 j 分片补回来）
        let ntok = (1..=ntok_cap)
            .rev()
            .find(|&ntok| {
                let bdim = (ntok * group / rpt) * nv;
                ntok * group % rpt == 0 && bdim <= bdim_cap && bdim % 32 == 0 && bdim >= 32
            })
            .unwrap_or_else(|| {
                panic!(
                    "无可用单遍档几何：cfg={cfg:?} qh={q_heads} kvh={kv_heads} hd={hd} t={tokens}"
                )
            });
        let (rows_nom, bdim) = (ntok * group, (ntok * group / rpt) * nv);
        // scratch 定尺（与 kernel 侧 `TILE_SLOT` 的三项一一对应）：p 区
        // chunk×rows、槽位表 ntok×chunk、行内 max 表 bdim×rpt。任一越界都是
        // 块间互踩（静默错值），故在这里显式挡死
        assert!(
            rows_nom <= ATTN_TILE_ROWS_CAP
                && ntok * ATTN_TILE_CHUNK <= ATTN_TILE_SLOT_TAB
                && bdim * rpt <= 2048,
            "单遍档 scratch 越界：ntok={ntok} rows={rows_nom} bdim={bdim} rpt={rpt}"
        );
        let ntile = (tokens as usize + ntok - 1) / ntok;
        let base = kv_heads as usize * ntile;
        // 同 `tile_geom`：偏结果表按 tokens×q_heads 行定尺，大 chunk 不开分片
        let mut jsplit = 1usize;
        if tokens as usize * q_heads as usize <= ATTN_TILE_PART_ROWS {
            while jsplit * base < ATTN_TILE_TARGET_BLOCKS && jsplit < ATTN_TILE_SPLIT_CAP {
                jsplit += 1;
            }
        }
        OnlineGeom { ntok, nd, rpt, nv, rows_nom, bdim, ntile, jsplit }
    }

    /// 块档几何裁决（纯函数：探针/exec/测试共用一份算式）。
    ///
    /// 与 `online_geom` 的差别：累加器在块私有 scratch 里，寄存器只装
    /// "一个 pass 的 group×nd 个 f32" ⇒ Br 不再被 65536/regs 顶死。
    /// 约束变成三条：
    /// · `br % ngrp == 0`（每线程组的 br/ngrp 个 token 整份）
    /// · `jchunk % nv == 0`（j 升序拼接的前提，与旧档同）
    /// · 块数 × 每块 slot ≤ `BLK_SCRATCH_FLOATS`（超了就**降 j 分片**）
    pub fn blk_geom(
        cfg: BlkCfg,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        tokens: u32,
    ) -> BlkGeom {
        let group = (q_heads / kv_heads) as usize;
        let hd = head_dim as usize;
        let nd = 4usize; // 块档导出只有 nd=4（`blk_export!` 的 G/ND 定态）
        // GQA 比也是定态（`blk_export!` 的 G=4）：别的 group 要另导实体
        assert!(group == 4, "块档导出按 group=4 单态（Qwen3.5-4B 16q/4kv），当前 group={group}");
        assert!(hd % nd == 0 && hd <= ATTN_TILE_HD_CAP, "块档要求 head_dim%4==0 且 ≤{ATTN_TILE_HD_CAP}");
        let nv = hd / nd;
        let bdim = (cfg.bdim.min(BLK_BDIM_CAP).max(nv) / nv) * nv;
        let ngrp = bdim / nv;
        assert!(bdim % 32 == 0, "块档线程数 {bdim} 须为 32 的倍数");
        // br 必须是 ngrp 的整倍（组内 token 整份），且 ≤ tokens
        let want = cfg.br.min(tokens as usize).max(ngrp);
        let br = (want / ngrp).max(1) * ngrp;
        let rows_nom = br * group;
        let npass = br / ngrp;
        let cap = cfg.jchunk;
        assert!(cap % nv == 0 && cap >= nv, "块档 j 片宽 {cap} 须是 nv={nv} 的整倍数");
        let ntile = (tokens as usize + br - 1) / br;
        let base = kv_heads as usize * ntile;
        let stride = Self::blk_stride(rows_nom, hd, br, cap);
        // j 分片：偏结果表装不下 tokens×q_heads 时只能单片（大 chunk 的块数够）
        let part_ok = tokens as usize * q_heads as usize <= BLK_PART_ROWS;
        let mut jsplit = if !part_ok {
            1
        } else if cfg.jsplit > 0 {
            cfg.jsplit.min(BLK_SPLIT_CAP)
        } else {
            ((BLK_TARGET_BLOCKS + base - 1) / base).clamp(1, BLK_SPLIT_CAP)
        };
        while jsplit > 1 && base * jsplit * stride > BLK_SCRATCH_FLOATS {
            jsplit -= 1;
        }
        assert!(
            base * jsplit * stride <= BLK_SCRATCH_FLOATS,
            "块档 scratch 越界：{} 块 × {stride} 槽 > {BLK_SCRATCH_FLOATS}（降 br/jchunk）",
            base * jsplit
        );
        assert!(
            rows_nom <= BLK_PART_ROWS && br * cap <= BLK_SCRATCH_FLOATS,
            "块档行列越界：br={br} rows={rows_nom} cap={cap}"
        );
        BlkGeom { br, nd, group, nv, ngrp, npass, rows_nom, bdim, ntile, jsplit, cap, stride }
    }

    /// 每块 scratch 槽数（f32）：与 kernel 侧 `blk_body` 的布局**逐项同式**
    /// O[rows×hd] + P[rows×cap] + mch/mru/lru/scl/ech[5×rows] + sb[br×cap]
    pub const fn blk_stride(rows_nom: usize, hd: usize, br: usize, cap: usize) -> usize {
        rows_nom * hd + rows_nom * cap + 5 * rows_nom + br * cap
    }

    /// 块档（**O/P 落块私有 scratch、Br 参数化**）：prefill 长序列专用。
    /// 与单遍档同一套在线 softmax 代数（每 j 片一次 rescale + 片内 max 归一），
    /// 只是累加器从寄存器搬到 `bscratch + blk*stride`。
    #[allow(clippy::too_many_arguments)]
    pub fn run_tile_blk(
        &self,
        cfg: BlkCfg,
        q: &DevBuf,
        k_cache: &DevBuf,
        v_cache: &DevBuf,
        block_table: &DevBuf,
        seq_ids: &DevBuf,
        positions: &DevBuf,
        out: &DevBuf,
        q_scale: f32,
        bpr: u32,
        block_size: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        tokens: u32,
        stream: &Stream,
    ) {
        let g = Self::blk_geom(cfg, q_heads, kv_heads, head_dim, tokens);
        let f = match cfg.kv {
            KvDtype::F32 => &self.f_blk_f32,
            KvDtype::Bf16 => &self.f_blk,
        };
        self.launch_blk(
            f, &g, q, k_cache, v_cache, block_table, seq_ids, positions, out, q_scale, bpr,
            block_size, q_heads, kv_heads, head_dim, tokens, stream,
        );
    }

    /// 块档装订：ABI = 单遍档 + `stride`（插在 `chunk` 之后）
    #[allow(clippy::too_many_arguments)]
    fn launch_blk(
        &self,
        f: &Func,
        g: &BlkGeom,
        q: &DevBuf,
        k_cache: &DevBuf,
        v_cache: &DevBuf,
        block_table: &DevBuf,
        seq_ids: &DevBuf,
        positions: &DevBuf,
        out: &DevBuf,
        q_scale: f32,
        bpr: u32,
        block_size: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        tokens: u32,
        stream: &Stream,
    ) {
        assert!(tokens > 0, "paged_attn_blk: tokens=0");
        let nblk = g.nblk(kv_heads);
        assert!(nblk * g.stride <= BLK_SCRATCH_FLOATS, "块档 scratch 越界：{nblk} 块");
        if g.jsplit > 1 {
            assert!(
                tokens as usize * q_heads as usize <= BLK_PART_ROWS,
                "块档分片行数 {} 超偏结果表 {BLK_PART_ROWS}",
                tokens as usize * q_heads as usize
            );
        }
        let (br_u, js_u, chunk_u, st_u) =
            (g.br as u32, g.jsplit as u32, g.cap as u32, g.stride as u32);
        let args = crate::kernel_args![
            *q, *k_cache, *v_cache, *block_table, *seq_ids, *positions, *out,
            self.pacc_b, self.pml_b, q_scale, bpr, block_size, q_heads, kv_heads, head_dim,
            tokens, br_u, js_u, chunk_u, st_u, self.bscratch
        ];
        self.gpu.launch(f, nblk as u32, g.bdim as u32, &args, stream);
        if g.jsplit > 1 {
            let (nrows, hd_u) = (tokens * q_heads, head_dim);
            let args = crate::kernel_args![self.pacc_b, self.pml_b, *out, nrows, js_u, hd_u];
            self.gpu.launch(&self.fc, nrows, 256, &args, stream);
        }
    }

    /// 分块档公共装订（两遍/单遍同一份 launch 账）：
    /// `f` 决定进哪个 kernel、`ntok/js/bdim` 是几何裁决的结果，其余逐字相同
    #[allow(clippy::too_many_arguments)]
    fn launch_tile(
        &self,
        f: &Func,
        ntok: usize,
        jsplit: usize,
        bdim: usize,
        q: &DevBuf,
        k_cache: &DevBuf,
        v_cache: &DevBuf,
        block_table: &DevBuf,
        seq_ids: &DevBuf,
        positions: &DevBuf,
        out: &DevBuf,
        q_scale: f32,
        bpr: u32,
        block_size: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        tokens: u32,
        stream: &Stream,
    ) {
        assert!(tokens > 0, "paged_attn_tile: tokens=0");
        assert!(head_dim % 4 == 0 && head_dim >= 4, "分块档要求 head_dim % 4 == 0");
        assert!(head_dim as usize <= ATTN_TILE_HD_CAP, "head_dim 超分块档表容量");
        // j 跨步 = nv = head_dim/4 必须整除 chunk：否则 chunk 升序 ≠ 全局 j 升序
        assert!(
            ATTN_TILE_CHUNK % (head_dim as usize / 4) == 0,
            "chunk 宽须是 head_dim/4 的整数倍（j 升序拼接的前提）"
        );
        assert!(
            bdim <= ATTN_TILE_BDIM_CAP && bdim % 32 == 0,
            "分块档线程数 {bdim} 非法（须 ≤{ATTN_TILE_BDIM_CAP} 且 32 的倍数）"
        );
        let ntile = (tokens as usize + ntok - 1) / ntok;
        let nblk = kv_heads as usize * ntile * jsplit;
        assert!(
            nblk <= ATTN_TILE_BLOCK_CAP,
            "分块档块数 {nblk} 超 scratch 容量 {ATTN_TILE_BLOCK_CAP}"
        );
        if jsplit > 1 {
            let nrows = tokens as usize * q_heads as usize;
            assert!(
                nrows <= ATTN_TILE_PART_ROWS,
                "分片档行数 {nrows} 超偏结果表容量 {ATTN_TILE_PART_ROWS}"
            );
        }
        // 注意：`kernel_args!` 取的是实参的**地址**，实参必须是活过 launch 的
        // place（局部变量/字段）——`jsplit as u32` 这类临时值在语句结束即失效，
        // launch 时读到的是被覆盖的栈槽（非法地址的老来源）。先落局部变量。
        let (ntok_u, js_u, chunk_u) = (ntok as u32, jsplit as u32, ATTN_TILE_CHUNK as u32);
        let args = crate::kernel_args![
            *q, *k_cache, *v_cache, *block_table, *seq_ids, *positions, *out,
            self.pacc, self.pml, q_scale, bpr, block_size, q_heads, kv_heads, head_dim,
            tokens, ntok_u, js_u, chunk_u, self.tscratch
        ];
        self.gpu.launch(f, nblk as u32, bdim as u32, &args, stream);
        if jsplit > 1 {
            let (nrows, hd_u) = (tokens * q_heads, head_dim);
            let args = crate::kernel_args![
                self.pacc, self.pml, *out, nrows, js_u, hd_u
            ];
            self.gpu.launch(&self.fc, nrows, 256, &args, stream);
        }
    }
}

impl PagedAttnOp for PagedAttnCuda<'_> {
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        q: &DevBuf,
        k_cache: &DevBuf,
        v_cache: &DevBuf,
        block_table: &DevBuf,
        seq_ids: &DevBuf,
        positions: &DevBuf,
        out: &DevBuf,
        q_scale: f32,
        bpr: u32,
        block_size: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        tokens: u32,
        stream: &Stream,
    ) {
        let n = tokens * q_heads;
        assert!(n as usize <= ATTN_BLOCK_CAP,
            "paged_attn 块数 {n} 超 scratch 容量 {ATTN_BLOCK_CAP}");
        let args = crate::kernel_args![*q, *k_cache, *v_cache, *block_table, *seq_ids, *positions,
            *out, q_scale, bpr, block_size, q_heads, kv_heads, head_dim, tokens, self.slot_cap,
            self.scratch];
        // 一 block 一 (t,qh)、block=head_dim 线程（四相位块协作——kernel 同构）
        self.gpu.launch(&self.f, n, head_dim, &args, stream);
    }
    fn name(&self) -> &'static str {
        "paged_attn/cuda"
    }
}

// ───────────────────── GEMM：标准算子走厂商库（§4） ─────────────────────

/// GEMM 家族 trait：C = A·B。数值合同 bf16 入出 + f32 累加（tensor core）；
/// 累加序在厂商库内——对齐标准是容差（golden 两模式的厂商库侧，g2）
pub trait GemmOp {
    /// a: [m,k] / b: [k,n] / c: [m,n]，bf16 行主序
    fn run(&self, a: &DevBuf, b: &DevBuf, c: &DevBuf, m: u32, n: u32, k: u32, stream: &Stream);
    /// C = A·Wᵀ：w 按存储 [n,k] 直读（nn.Linear 权重布局，I1 稳态形态）
    fn run_wt(&self, a: &DevBuf, w: &DevBuf, c: &DevBuf, m: u32, n: u32, k: u32, stream: &Stream);
    fn name(&self) -> &'static str;
}

/// cuBLASLt 档（Box 持有：BlasLt 带工作区与符号库生命周期）
pub struct GemmLt<'a> {
    pub(crate) lt: Box<crate::cuda::BlasLt<'a>>,
}

impl GemmLt<'_> {
    /// gemm autotune 审计（诊断出口；见 `BlasLt::tune_report`）
    pub fn tune_report(&self) -> Vec<crate::cuda::GemmTuneRecord> {
        self.lt.tune_report()
    }
}

impl GemmOp for GemmLt<'_> {
    fn run(&self, a: &DevBuf, b: &DevBuf, c: &DevBuf, m: u32, n: u32, k: u32, stream: &Stream) {
        self.lt.gemm(a, b, c, m, n, k, stream)
    }
    fn run_wt(&self, a: &DevBuf, w: &DevBuf, c: &DevBuf, m: u32, n: u32, k: u32, stream: &Stream) {
        self.lt.gemm_wt(a, w, c, m, n, k, stream)
    }
    fn name(&self) -> &'static str {
        "gemm/cuda-lt"
    }
}
