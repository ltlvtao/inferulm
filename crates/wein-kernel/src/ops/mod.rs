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
}

impl OpTable<'_> {
    /// 按 HardwareProfile 解析。设备族不匹配 → Degrade 列表（启动期裁决）。
    /// PTX 模块不显式卸载——算子表即进程生命周期，CUfunction 句柄常驻
    pub fn resolve<'a>(
        gpu: &'a Gpu,
        profile: &HardwareProfile,
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
        let kv_scatter = KvScatterCuda { gpu, f: m.func("kv_scatter") };
        let paged_attn = PagedAttnCuda { gpu, f: m.func("paged_attn") };
        // 标准算子走厂商库（§4）：库缺 = 声明式降级，不带病运行
        let gemm = GemmLt { lt: Box::new(
            crate::cuda::BlasLt::open(gpu).map_err(|e| vec![Degrade {
                key: "kernel.gemm",
                reason: format!("libcublasLt 打不开：{e}"),
                impact: "GEMM 无法解析（标准算子走厂商库，M0 无回退实现）".into(),
            }])?,
        )};
        Ok(OpTable { rms_norm, rope, silu, w8a8_quant, kv_scatter, paged_attn, gemm })
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
    fn name(&self) -> &'static str;
}

pub struct KvScatterCuda<'a> {
    gpu: &'a Gpu,
    f: Func,
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
    fn name(&self) -> &'static str {
        "kv_scatter/cuda"
    }
}

/// varlen paged attention（correctness-first：单线程两遍扫 KV，W2 再 flash 分块）。
/// 每 (token, q_head) 一线程：pos 位置的 query 因果只看 0..=pos；
/// GQA：kv 头 = qh % kv_heads
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
        let args = crate::kernel_args![*q, *k_cache, *v_cache, *block_table, *seq_ids, *positions,
            *out, q_scale, bpr, block_size, q_heads, kv_heads, head_dim, tokens];
        self.gpu.launch(&self.f, n.div_ceil(256), 256, &args, stream);
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
    fn name(&self) -> &'static str;
}

/// cuBLASLt 档（Box 持有：BlasLt 带工作区与符号库生命周期）
pub struct GemmLt<'a> {
    pub(crate) lt: Box<crate::cuda::BlasLt<'a>>,
}

impl GemmOp for GemmLt<'_> {
    fn run(&self, a: &DevBuf, b: &DevBuf, c: &DevBuf, m: u32, n: u32, k: u32, stream: &Stream) {
        self.lt.gemm(a, b, c, m, n, k, stream)
    }
    fn name(&self) -> &'static str {
        "gemm/cuda-lt"
    }
}
