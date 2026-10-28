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
use std::collections::{HashMap, HashSet};
use std::path::Path;

use wein_engine::Executor;
use wein_kernel::cuda::{DevBuf, Stream};
use wein_kernel::ops::{
    AddOp, ArgmaxRowsOp, Conv1dDwOp, ConvStateRollOp, DeltanetOp, EmbedOp, GemmOp, KvScatterOp,
    MulOp, PagedAttnOp, RmsNormGatedOp, RmsNormZcOp, RopePtOp, SigmulOp, SiluOp,
};
use wein_kernel::reference;
use wein_memory::pool::BlockPool;
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
}

// 独占设备句柄，exec 单线程驱动（§5）——同 DeviceSession 论证
unsafe impl Send for Qwen35Executor {}

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
    reqs: HashMap<ReqId, ReqState>,
    bt_host: Vec<u32>,
    next_row: u32,
    free_rows: Vec<u32>,
    // 派生常量
    lin_row: usize, // 线性层状态单行字节数
    conv_row: usize, // conv 状态单行字节数
    bpr: usize,
    q_scale: f32,
    session: DeviceSession,
}

impl Qwen35Executor {
    pub fn open(dir: &Path) -> Result<Self, String> {
        let session = DeviceSession::open()?;
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
            reqs: HashMap::new(),
            bt_host: vec![0u32; MAX_ROWS * bpr_of()],
            next_row: 0,
            free_rows: Vec::new(),
            lin_row,
            conv_row,
            bpr: bpr_of(),
            q_scale,
            session,
        })
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
                self.forward_batch(&[(row, tok_in, pos as u32)], &[0])[0]
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
        let row = match self.reqs.get_mut(&id) {
            Some(st) if st.blocks.len() < need => st.row as usize,
            _ => return,
        };
        let bpr = self.bpr;
        let st = self.reqs.get_mut(&id).unwrap();
        while st.blocks.len() < need {
            let b = self
                .pool
                .alloc()
                .unwrap_or_else(|| panic!("页池耗尽（{} 页）——I1 配置须保证容量", self.pool.total()));
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

    /// 批前向 + 设备端采样。rows: (块表行, token, 位置)；samples: 采样行
    /// 全局下标。返回每采样行一个 token。批内同请求行须连续（断言钉——
    /// conv tap/deltanet 递推的段语义前提）
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
        let eps = self.cfg.rms_eps;
        let (h, ii, voc) =
            (self.cfg.hidden_size as u32, self.cfg.inter_size as u32, self.cfg.vocab_size as u32);
        let t32 = t as u32;

        // ── embed ──
        tb.embed.run(&d_ids, &self.embed, &self.scr.x, h, t32, stream);

        // ── 层循环（x 主路：n1 ← norm(x)；h2 = x + o；x = h2 + mlp）──
        let mut lin_i = 0usize;
        let mut full_i = 0usize;
        for layer in &self.layers {
            tb.rms_norm_zc.run(&self.scr.x, &layer.ln1, &self.scr.n1, t32, h, eps, stream);
            match &layer.attn {
                AttnDev::Full(w) => {
                    let (qh_n, kvh, hd) = (self.cfg.num_heads, self.cfg.num_kv_heads, self.cfg.head_dim);
                    let he = kvh * hd;
                    tb.gemm.run_wt(&self.scr.n1, &w.qg, &self.scr.qg, t32, 2 * qh_n * hd, h, stream);
                    tb.embed.run(&d_iq, &self.scr.qg, &self.scr.qh, hd, t32 * qh_n, stream);
                    tb.embed.run(&d_ig, &self.scr.qg, &self.scr.gate, hd, t32 * qh_n, stream);
                    // 零中心 norm 在 head_dim 上（原地：行私有读写，安全）
                    tb.rms_norm_zc.run(&self.scr.qh, &w.qn, &self.scr.qh, t32 * qh_n, hd, eps, stream);
                    tb.gemm.run_wt(&self.scr.n1, &w.k, &self.scr.kh, t32, he, h, stream);
                    tb.gemm.run_wt(&self.scr.n1, &w.v, &self.scr.vh, t32, he, h, stream);
                    tb.rms_norm_zc.run(&self.scr.kh, &w.kn, &self.scr.kh, t32 * kvh, hd, eps, stream);
                    let rd = self.cfg.rot_dim();
                    tb.rope_pt.run(
                        &self.scr.qh, &self.cos, &self.sin, &d_pos, &self.scr.qh, rd, hd, qh_n,
                        t32, stream,
                    );
                    tb.rope_pt.run(
                        &self.scr.kh, &self.cos, &self.sin, &d_pos, &self.scr.kh, rd, hd, kvh,
                        t32, stream,
                    );
                    tb.kv_scatter.run(
                        &self.scr.kh, &self.scr.vh, &d_bt, &d_seqs, &d_pos, &self.k_pool[full_i],
                        &self.v_pool[full_i], self.bpr as u32, BLOCK, he, t32, stream,
                    );
                    tb.paged_attn.run(
                        &self.scr.qh, &self.k_pool[full_i], &self.v_pool[full_i], &d_bt, &d_seqs,
                        &d_pos, &self.scr.attn, self.q_scale, self.bpr as u32, BLOCK, qh_n, kvh,
                        hd, t32, stream,
                    );
                    tb.sigmul.run(
                        &self.scr.attn, &self.scr.gate, &self.scr.agated, t32 * qh_n * hd, stream,
                    );
                    tb.gemm.run_wt(&self.scr.agated, &w.o, &self.scr.o, t32, h, qh_n * hd, stream);
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
                    tb.gemm.run_wt(&self.scr.n1, &w.qkv, &self.scr.qkv, t32, cd, h, stream);
                    tb.conv1d_dw.run(
                        &self.scr.qkv, &w.conv, &self.conv_state[lin_i], &self.scr.qkvc, &d_trow,
                        &d_tpos, cd, self.cfg.conv_kernel, t32, stream,
                    );
                    tb.conv_state_roll.run(
                        &self.scr.qkvc, &self.conv_state[lin_i], &d_start, &d_count, &d_rid, cd,
                        self.cfg.conv_kernel, n_rows, stream,
                    );
                    tb.embed.run(&d_ilq, &self.scr.qkvc, &self.scr.qs, keyd, t32, stream);
                    tb.embed.run(&d_ilk, &self.scr.qkvc, &self.scr.ks, keyd, t32, stream);
                    tb.embed.run(&d_ilv, &self.scr.qkvc, &self.scr.vs, vald, t32, stream);
                    tb.gemm.run_wt(&self.scr.n1, &w.z, &self.scr.z, t32, vald, h, stream);
                    tb.gemm.run_wt(&self.scr.n1, &w.a, &self.scr.aa, t32, 32, h, stream);
                    tb.gemm.run_wt(&self.scr.n1, &w.b, &self.scr.bb, t32, 32, h, stream);
                    tb.deltanet.run(
                        &self.scr.qs, &self.scr.ks, &self.scr.vs, &self.scr.aa, &self.scr.bb,
                        &w.a_log, &w.dtb, &self.lin_state[lin_i], &self.scr.core, &d_start,
                        &d_count, &d_rid, n_rows, kh_n, vh_n, kd, vd, L2_EPS, stream,
                    );
                    tb.rms_norm_gated.run(
                        &self.scr.core, &self.scr.z, &w.norm, &self.scr.normed, t32 * vh_n, vd,
                        eps, stream,
                    );
                    tb.gemm.run_wt(&self.scr.normed, &w.out, &self.scr.o, t32, h, vald, stream);
                    lin_i += 1;
                }
            }
            tb.add.run(&self.scr.x, &self.scr.o, &self.scr.h2, t32 * h, stream);
            tb.rms_norm_zc.run(&self.scr.h2, &layer.ln2, &self.scr.n2, t32, h, eps, stream);
            tb.gemm.run_wt(&self.scr.n2, &layer.wg, &self.scr.g, t32, ii, h, stream);
            tb.gemm.run_wt(&self.scr.n2, &layer.wu, &self.scr.u, t32, ii, h, stream);
            tb.silu.run(&self.scr.g, &self.scr.act, t32 * ii, stream);
            tb.mul.run(&self.scr.act, &self.scr.u, &self.scr.m, t32 * ii, stream);
            tb.gemm.run_wt(&self.scr.m, &layer.wd, &self.scr.d, t32, h, ii, stream);
            tb.add.run(&self.scr.h2, &self.scr.d, &self.scr.x, t32 * h, stream);
        }
        // ── final norm + logits（tie：embed 兼作 lm_head 的 W[n,k]）──
        tb.rms_norm_zc.run(&self.scr.x, &self.norm_f, &self.scr.n2, t32, h, eps, stream);
        tb.gemm.run_wt(&self.scr.n2, &self.embed, &self.scr.logits, t32, voc, h, stream);

        // ── 采样（设备端 argmax；只回读采样行的 u32）──
        if samples.is_empty() {
            stream.sync();
            return Vec::new();
        }
        let s = samples.len() as u32;
        assert!((s as usize) <= t, "采样行数超批");
        let d_rows = gpu.alloc_from_slice(samples);
        tb.argmax_rows.run(
            &self.scr.logits, &d_rows, &self.scr.am_scratch, &self.scr.am_out, voc, s, stream,
        );
        stream.sync();
        let out: Vec<u32> = self.scr.am_out.read_to_vec();
        out[..s as usize].to_vec()
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
    vh * kd * vd * 4
}

/// conv 状态**单行**字节数 [conv_dim, kernel-1] bf16（池 = ×MAX_ROWS）
fn conv_row_bytes(cfg: &Qwen35Config) -> usize {
    let cd = cfg.conv_dim() as usize;
    cd * (cfg.conv_kernel as usize - 1) * 2
}

impl Executor for Qwen35Executor {
    fn admitted(&mut self, id: ReqId, prompt: &[u32]) {
        assert!(
            prompt.iter().all(|&t| t < self.cfg.vocab_size),
            "token 越界（vocab {}）",
            self.cfg.vocab_size
        );
        let seq = if prompt.is_empty() { vec![0u32] } else { prompt.to_vec() };
        let row = self.free_rows.pop().unwrap_or_else(|| {
            assert!(
                (self.next_row as usize) < MAX_ROWS,
                "块表行耗尽（{MAX_ROWS}）——I1 并发上限"
            );
            let r = self.next_row;
            self.next_row += 1;
            r
        });
        self.reqs.insert(id, ReqState { seq, forwarded: 0, row, blocks: Vec::new() });
    }

    fn step(&mut self, plan: &StepPlan) -> Vec<(ReqId, u32)> {
        struct RowIn {
            id: ReqId,
            row: u32,
            tok: u32,
            pos: u32,
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
                batch.push(RowIn { id, row: st.row, tok: st.seq[pos], pos: pos as u32 });
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
        // 4. 前向 + 设备端贪心
        let rows: Vec<(u32, u32, u32)> = batch.iter().map(|r| (r.row, r.tok, r.pos)).collect();
        let sample_rows: Vec<u32> = samples.iter().map(|s| s.1 as u32).collect();
        let toks = self.forward_batch(&rows, &sample_rows);
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
                let base = st.row as usize * self.bpr;
                for (j, b) in st.blocks.iter().enumerate() {
                    self.bt_host[base + j] = 0;
                    self.pool.release(*b);
                }
                self.zero_row_state(st.row, stream); // 回收行状态零化（q10 钉）
                self.free_rows.push(st.row);
            }
        }
        self.pool.recycle();
    }

    fn preempted(&mut self, id: ReqId) {
        let stream = self.session.stream();
        if let Some(st) = self.reqs.get(&id) {
            let row = st.row;
            self.zero_row_state(row, stream); // 重放从 pos 0 起步：递推状态必须归零
            self.reqs.get_mut(&id).unwrap().forwarded = 0;
        }
    }
}
