//! ToyExecutor：单块 toy 模型 + 真算子装配——M0 exec 的"真身"。
//!
//! 模型（形状为把 M0 算子清单全部过一遍，数值无关）：
//! embed → rms1 → {q,k,v} GEMM → rope（每头一行）→ KV scatter（分页池）
//! → paged attn（因果/GQA）→ o 投影 → 残差 → rms2 → {gate,up} GEMM →
//! silu×up → down GEMM → 残差 → lm_head GEMM → host 贪心 argmax
//!
//! 调度合同（与 wein-sched step 语义钉死，x6/x7 验收）：
//! - 追平步：`plan.prefill` 段覆盖到采样位（同列 `plan.decode`）——段尾行即样本
//! - 纯 decode：前向 seq 尾（上步采样追加、尚未前向的那个 token）
//! - 空 prompt：`admitted` 时落 BOS(0)——调度器看长度 0 直接走 decode
//! - 抢占重入：`preempted` 清前向游标（重放对同样 KV 覆盖写，确定性下幂等）
//!
//! 图档位（§7，w 系验收）：`graph_buckets` 非空 = FULL_DECODE——桶集合
//! 共用单输入存储（常驻 DevBuf，每步整段覆写），纯 decode 步路由到 ≥m
//! 的最小桶图回放，批尾 sink 行（块表末行 + 专属页）吸收 padding；混合步
//! 走 eager。空桶集 = NONE（x 系保持每步分配的 eager 直发）
use std::collections::HashMap;

use wein_engine::Executor;
use wein_kernel::cuda::{DevBuf, GraphExec};
use wein_kernel::ops::{
    AddOp, EmbedOp, GemmOp, KvScatterOp, MulOp, PagedAttnOp, RmsNormOp, RopeOp, SiluOp,
};
use wein_kernel::reference;
use wein_memory::pool::BlockPool;
use wein_sched::{ReqId, StepPlan};

use crate::device::{pin, DeviceSession};

/// toy 形状（默认 = 一次 4090 上的轻量全路径冒烟）
#[derive(Debug, Clone)]
pub struct ToyConfig {
    pub vocab: u32,
    pub hidden: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub inter: u32,
    pub num_pages: u32,
    pub block_size: u32,
    pub max_pos: u32,
    pub seed: u64,
    pub eps: f32,
    /// FULL_DECODE 图档桶集合（§7 图档位谱系）。空 = NONE（默认，eager
    /// 直发）；非空须严格升序、≥1、最大桶 ≤ t_max(1024)。m 超最大桶的步
    /// 回落 eager。M0 不实现 exec update 的桶间迁移——每桶独立捕获
    pub graph_buckets: Vec<u32>,
}

impl Default for ToyConfig {
    fn default() -> Self {
        Self {
            vocab: 256,
            hidden: 128,
            q_heads: 2,
            kv_heads: 2,
            head_dim: 64,
            inter: 256,
            // 页/行容量按 bench 并发给宽裕（c=100 × 40 槽 ≈ 250 页；max_running 256）——
            // 量 exec 的账，不制造池耗尽噪声
            num_pages: 1024,
            block_size: 16,
            max_pos: 256, // paged_attn M0 上限（序列长 ≤ 256，W2 flash 分块移除）
            seed: 42,
            eps: 1e-6,
            graph_buckets: Vec::new(),
        }
    }
}

impl ToyConfig {
    fn validate(&self) -> Result<(), String> {
        if self.q_heads != self.kv_heads {
            // rope kernel 一次调用同拍处理 q/k 两侧行——MHA（qh==kvh）即可单拍；
            // GQA 展开留给 W2 真 attention 装配
            return Err(format!("M0 toy 须 MHA（q_heads {} != kv_heads {}）", self.q_heads, self.kv_heads));
        }
        if self.head_dim % 2 != 0 {
            return Err("head_dim 须为偶数（rope half-split）".into());
        }
        if self.max_pos == 0 || self.max_pos > 256 {
            return Err(format!("max_pos {} 超出 paged_attn M0 上限 256", self.max_pos));
        }
        if self.vocab < 2 {
            return Err("vocab 至少 2".into());
        }
        let b = &self.graph_buckets;
        if !b.is_empty() {
            if b[0] < 1 {
                return Err(format!("graph_buckets 须 ≥ 1（首桶 {}）", b[0]));
            }
            if b.windows(2).any(|w| w[0] >= w[1]) {
                return Err(format!("graph_buckets 须严格升序：{b:?}"));
            }
            const T_MAX: u32 = 1024;
            if *b.last().unwrap() > T_MAX {
                return Err(format!("最大桶 {} 超 t_max {T_MAX}", b.last().unwrap()));
            }
        }
        Ok(())
    }
}

/// host 端 bf16 权重（设备上传源；CPU 镜像同源生成——权重是输入，不是被测路径）
pub struct ToyWeights {
    pub embed: Vec<u16>, // [vocab, hidden]
    pub rms1: Vec<u16>,  // [hidden]
    pub wq: Vec<u16>,    // [hidden, qE]
    pub wk: Vec<u16>,    // [hidden, kvE]
    pub wv: Vec<u16>,    // [hidden, kvE]
    pub wo: Vec<u16>,    // [qE, hidden]
    pub rms2: Vec<u16>,  // [hidden]
    pub wg: Vec<u16>,    // [hidden, inter]
    pub wu: Vec<u16>,    // [hidden, inter]
    pub wd: Vec<u16>,    // [inter, hidden]
    pub lm: Vec<u16>,    // [hidden, vocab]
}

impl ToyWeights {
    /// splitmix 定长扫（确定性）；缩放 1/sqrt(扇入)——不训不推理，只求激活不爆
    pub fn synthesize(cfg: &ToyConfig) -> Self {
        let (v, h, i) = (cfg.vocab as usize, cfg.hidden as usize, cfg.inter as usize);
        let (qe, ke) = (
            cfg.q_heads as usize * cfg.head_dim as usize,
            cfg.kv_heads as usize * cfg.head_dim as usize,
        );
        let mat = |rows: usize, cols: usize, seed: u64| -> Vec<u16> {
            let s = (0.5 / (cols as f64).sqrt()) as f32;
            (0..(rows * cols) as u64)
                .map(|j| reference::f32_to_bf16(reference::splitmix_f32(j ^ seed) * s))
                .collect()
        };
        let s = cfg.seed;
        Self {
            embed: mat(v, h, s ^ 0xE11),
            rms1: mat(1, h, s ^ 0xE12),
            wq: mat(h, qe, s ^ 0xE13),
            wk: mat(h, ke, s ^ 0xE14),
            wv: mat(h, ke, s ^ 0xE15),
            wo: mat(qe, h, s ^ 0xE16),
            rms2: mat(1, h, s ^ 0xE17),
            wg: mat(h, i, s ^ 0xE18),
            wu: mat(h, i, s ^ 0xE19),
            wd: mat(i, h, s ^ 0xE1A),
            lm: mat(h, v, s ^ 0xE1B),
        }
    }
}

/// 设备端权重束（全部借用 session.gpu——钉扎见 device 模块）
struct WDev {
    embed: DevBuf<'static>,
    rms1: DevBuf<'static>,
    wq: DevBuf<'static>,
    wk: DevBuf<'static>,
    wv: DevBuf<'static>,
    wo: DevBuf<'static>,
    rms2: DevBuf<'static>,
    wg: DevBuf<'static>,
    wu: DevBuf<'static>,
    wd: DevBuf<'static>,
    lm: DevBuf<'static>,
}

/// 前向 scratch（按 T_MAX 一次分配；eager 上传缓冲每步分配，图档输入常驻覆写）
struct ScrDev {
    x: DevBuf<'static>,
    n1: DevBuf<'static>,
    q: DevBuf<'static>,
    k: DevBuf<'static>,
    v: DevBuf<'static>,
    attn: DevBuf<'static>,
    o: DevBuf<'static>,
    h2: DevBuf<'static>,
    n2: DevBuf<'static>,
    g: DevBuf<'static>,
    u: DevBuf<'static>,
    act: DevBuf<'static>,
    m: DevBuf<'static>,
    d: DevBuf<'static>,
    h3: DevBuf<'static>,
    logits: DevBuf<'static>,
}

struct ReqState {
    /// prompt + 自采样追加（exec 自己的序列真相源）
    seq: Vec<u32>,
    /// 已前向 token 数（抢占清零——recompute 重放）
    forwarded: usize,
    /// 块表行号（dense 分配，M0 不回收）
    row: u32,
    /// 该请求持有的页（BlockPool 记账镜像）
    blocks: Vec<u32>,
}

/// FULL_DECODE 图档（§7）：桶集合共用**单输入存储**（常驻 DevBuf——设备
/// 指针烤进图节点，数据每步 upload_on 整段覆写），sink 行（块表末行 +
/// 专属页）吸收桶尾 padding。execs 与 buckets 下标对齐。
/// 声明序即析构序：本结构先于 session 析构——GraphExec 的 'static 钉扎
/// （borrow session.stream 的短命借用提为 'static）与 stream/table 同一论证
struct GraphMode {
    sink_row: u32,
    b_max: u32,
    buckets: Vec<u32>,
    in_ids: DevBuf<'static>,
    in_pos: DevBuf<'static>,
    in_seq: DevBuf<'static>,
    /// rope 以 (token,头) 为行——pos 按头数展开
    in_posq: DevBuf<'static>,
    /// 全块表常驻镜像（每步覆写——页保证先于上传）
    in_bt: DevBuf<'static>,
    execs: Vec<GraphExec<'static>>,
    /// 图路径前向计数（w 系验收：证路径确实被走到）
    replays: u64,
}

// 独占持有全部设备句柄；exec 单线程驱动（§5）——同 DeviceSession 的论证
unsafe impl Send for ToyExecutor {}

pub struct ToyExecutor {
    // 借用 session.gpu 的设备缓冲：声明序在前 = 先于 session 析构
    w: WDev,
    k_pool: DevBuf<'static>,
    v_pool: DevBuf<'static>,
    cos: DevBuf<'static>,
    sin: DevBuf<'static>,
    scr: ScrDev,
    // host 记账
    cfg: ToyConfig,
    pool: BlockPool,
    reqs: HashMap<ReqId, ReqState>,
    /// 块表 host 镜像 [max_rows × bpr]（每步全量上传——M0 简化）
    bt_host: Vec<u32>,
    next_row: u32,
    /// 已回收块表行（admitted 取用——行必须可回收，否则服务器在第
    /// max_rows 次 admit 后死掉：battery run2/3 全空抓到的长寿性缺陷）
    free_rows: Vec<u32>,
    /// 可分配块表行上限（图档让出末行作 sink——NONE 时 = max_rows）
    row_capacity: u32,
    // 派生常量（validate 后缓存）
    q_elems: usize,
    kv_elems: usize,
    t_max: usize,
    max_rows: usize,
    bpr: usize,
    q_scale: f32,
    /// FULL_DECODE 图档（None = NONE——每步分配的 eager 直发）
    graph: Option<GraphMode>,
    // 会话最后声明 = 最后析构（graph 的 exec/buffer 钉扎依赖此序）
    session: DeviceSession,
}

impl ToyExecutor {
    pub fn open(cfg: ToyConfig) -> Result<Self, String> {
        cfg.validate()?;
        let session = DeviceSession::open()?;
        let gpu = session.gpu();
        let w = ToyWeights::synthesize(&cfg);
        let (h, i, v) = (cfg.hidden as usize, cfg.inter as usize, cfg.vocab as usize);
        let (qe, ke) = (
            cfg.q_heads as usize * cfg.head_dim as usize,
            cfg.kv_heads as usize * cfg.head_dim as usize,
        );
        // t_max 对齐调度步预算上限（512）留倍余——步内 prefill+decode 总行数不超过
        let (t_max, max_rows) = (1024usize, 512usize);
        let bpr = cfg.max_pos as usize / cfg.block_size as usize
            + usize::from(cfg.max_pos as usize % cfg.block_size as usize != 0);
        let (cos, sin) =
            reference::rope_table(cfg.max_pos as usize, cfg.head_dim as usize / 2, 10000.0);
        let kv_bytes = cfg.num_pages as usize * cfg.block_size as usize * ke * 2;
        let bf = |n: usize| n * 2; // bf16 字节数
        let row_capacity = max_rows as u32 - u32::from(!cfg.graph_buckets.is_empty());
        let mut ex = ToyExecutor {
            w: WDev {
                embed: pin!(gpu.alloc_from_slice(&w.embed)),
                rms1: pin!(gpu.alloc_from_slice(&w.rms1)),
                wq: pin!(gpu.alloc_from_slice(&w.wq)),
                wk: pin!(gpu.alloc_from_slice(&w.wk)),
                wv: pin!(gpu.alloc_from_slice(&w.wv)),
                wo: pin!(gpu.alloc_from_slice(&w.wo)),
                rms2: pin!(gpu.alloc_from_slice(&w.rms2)),
                wg: pin!(gpu.alloc_from_slice(&w.wg)),
                wu: pin!(gpu.alloc_from_slice(&w.wu)),
                wd: pin!(gpu.alloc_from_slice(&w.wd)),
                lm: pin!(gpu.alloc_from_slice(&w.lm)),
            },
            k_pool: pin!(gpu.alloc_bytes(kv_bytes)),
            v_pool: pin!(gpu.alloc_bytes(kv_bytes)),
            cos: pin!(gpu.alloc_from_slice(&cos)),
            sin: pin!(gpu.alloc_from_slice(&sin)),
            scr: ScrDev {
                x: pin!(gpu.alloc_bytes(bf(t_max * h))),
                n1: pin!(gpu.alloc_bytes(bf(t_max * h))),
                q: pin!(gpu.alloc_bytes(bf(t_max * qe))),
                k: pin!(gpu.alloc_bytes(bf(t_max * ke))),
                v: pin!(gpu.alloc_bytes(bf(t_max * ke))),
                attn: pin!(gpu.alloc_bytes(bf(t_max * qe))),
                o: pin!(gpu.alloc_bytes(bf(t_max * h))),
                h2: pin!(gpu.alloc_bytes(bf(t_max * h))),
                n2: pin!(gpu.alloc_bytes(bf(t_max * h))),
                g: pin!(gpu.alloc_bytes(bf(t_max * i))),
                u: pin!(gpu.alloc_bytes(bf(t_max * i))),
                act: pin!(gpu.alloc_bytes(bf(t_max * i))),
                m: pin!(gpu.alloc_bytes(bf(t_max * i))),
                d: pin!(gpu.alloc_bytes(bf(t_max * h))),
                h3: pin!(gpu.alloc_bytes(bf(t_max * h))),
                logits: pin!(gpu.alloc_bytes(bf(t_max * v))),
            },
            pool: BlockPool::new(cfg.num_pages, 0.5),
            reqs: HashMap::new(),
            bt_host: vec![0u32; max_rows * bpr],
            next_row: 0,
            free_rows: Vec::new(),
            row_capacity,
            q_elems: qe,
            kv_elems: ke,
            t_max,
            max_rows,
            bpr,
            q_scale: 1.0f32 / (cfg.head_dim as f32).sqrt(),
            graph: None,
            cfg,
            session,
        };
        if !ex.cfg.graph_buckets.is_empty() {
            ex.build_graphs()?;
        }
        Ok(ex)
    }

    /// FULL_DECODE 图建造（open 期一次）：sink 行落位 + 常驻输入存储 +
    /// 各桶「预热→捕获→实例化」。此后纯 decode 步按 ≥m 最小桶回放。
    /// 捕获窗口内零隐式同步（§4 安全规则，gr5 守卫兜底）——上传与预热
    /// 全部在 begin_capture 之前完成
    fn build_graphs(&mut self) -> Result<(), String> {
        let buckets = self.cfg.graph_buckets.clone();
        let b_max = *buckets.last().unwrap() as usize;
        let sink_row = (self.max_rows - 1) as u32;
        // sink 专属页：padding 行的 KV 写垃圾只落此页，永不分配给请求
        let sink_page = self.pool.alloc().ok_or("页池耗尽——sink 行需要 1 页")?;
        self.bt_host[sink_row as usize * self.bpr] = sink_page;
        let gpu = self.session.gpu();
        let stream = self.session.stream();
        let qh = self.cfg.q_heads as usize;
        let (in_ids, in_pos, in_seq, in_posq, in_bt): (DevBuf<'static>, DevBuf<'static>, DevBuf<'static>, DevBuf<'static>, DevBuf<'static>) = (
            pin!(gpu.alloc_bytes(b_max * 4)),
            pin!(gpu.alloc_bytes(b_max * 4)),
            pin!(gpu.alloc_bytes(b_max * 4)),
            pin!(gpu.alloc_bytes(b_max * qh * 4)),
            pin!(gpu.alloc_bytes(self.max_rows * self.bpr * 4)),
        );
        // 捕获期输入：全行指向 sink（tok 0 / pos 0）——真实数据回放前覆写
        in_ids.upload_on(&vec![0u32; b_max], stream);
        in_pos.upload_on(&vec![0u32; b_max], stream);
        in_seq.upload_on(&vec![sink_row; b_max], stream);
        in_posq.upload_on(&vec![0u32; b_max * qh], stream);
        in_bt.upload_on(&self.bt_host, stream);
        stream.sync();
        let mut execs = Vec::with_capacity(buckets.len());
        for &b in &buckets {
            // 预热：同形状直发一次（cuBLASLt heuristic/PTX 模块落缓存——
            // gr4 合同：缓存命中后 gemm 是纯 launch，捕获安全）
            self.run_forward(b, &in_ids, &in_pos, &in_seq, &in_posq, &in_bt);
            stream.sync();
            // 捕获 → 实例化：录的是 kernel 序，不是数据快照（gr2）
            stream.begin_capture();
            self.run_forward(b, &in_ids, &in_pos, &in_seq, &in_posq, &in_bt);
            let exec: GraphExec<'static> = pin!(stream.end_capture().instantiate());
            execs.push(exec);
        }
        self.graph = Some(GraphMode {
            sink_row,
            b_max: b_max as u32,
            buckets,
            in_ids,
            in_pos,
            in_seq,
            in_posq,
            in_bt,
            execs,
            replays: 0,
        });
        Ok(())
    }

    pub fn cfg(&self) -> &ToyConfig {
        &self.cfg
    }

    pub fn device_name(&self) -> &str {
        self.session.gpu().name()
    }

    /// BlockPool 已占用页数（x5 与块表记账对照）
    pub fn pages_used(&self) -> usize {
        (self.pool.total() - self.pool.free_now() as u32) as usize
    }

    /// 某请求持有的页数
    pub fn pages_of(&self, id: ReqId) -> usize {
        self.reqs.get(&id).map(|s| s.blocks.len()).unwrap_or(0)
    }

    /// 图回放步数（w 系验收：证图路径确实被走到；NONE 档恒 0）
    pub fn graph_replays(&self) -> u64 {
        self.graph.as_ref().map(|g| g.replays).unwrap_or(0)
    }

    /// 前向一个 token（纯原语，不动 forwarded——x2/x4 逐步探针用）。
    /// 单行即纯 decode 形状——图档下走桶回放（w1 用它做位级对拍）。
    /// 返回该位置 logits（bf16 存储 → f32）
    pub fn forward_one(&mut self, id: ReqId, token: u32, pos: u32) -> Vec<f32> {
        let row = self.reqs.get(&id).map(|s| s.row).unwrap_or_else(|| panic!("forward_one：未知请求 {id}"));
        self.ensure_pages(id, pos as usize);
        self.forward_rows(&[(row, token, pos)], true)
    }

    /// 直跑完整生成（测试/CLI 便利面；不经过调度器）。
    /// 语义与 step 合同一致：追平段尾采样，之后逐 token decode。
    /// 内部 reset 后用请求 id=1
    pub fn generate(&mut self, prompt: &[u32], max_tokens: u32) -> Vec<u32> {
        self.reset();
        self.admitted(1, prompt);
        if max_tokens == 0 {
            return Vec::new();
        }
        let mut out = Vec::new();
        // 追平：整段前向（分块 ≤ t_max），段尾即首个采样位
        let mut first_tok = None;
        loop {
            let Some(st) = self.reqs.get(&1) else { break };
            let (start, len) = (st.forwarded, st.seq.len());
            if start == len {
                break;
            }
            let take = (len - start).min(self.t_max);
            let row = st.row;
            let batch: Vec<(u32, u32, u32)> = (0..take)
                .map(|j| (row, self.reqs[&1].seq[start + j], (start + j) as u32))
                .collect();
            self.reqs.get_mut(&1).unwrap().forwarded += take;
            self.ensure_pages(1, (start + take - 1) as usize);
            let logits = self.forward_batch(&batch);
            if start + take == self.reqs[&1].seq.len() {
                let last = &logits[(take - 1) * self.cfg.vocab as usize..];
                first_tok = Some(crate::argmax(last));
                break;
            }
        }
        let Some(tok) = first_tok else {
            return out; // 空 prompt 且无前向位——不该发生（admitted 已落 BOS）
        };
        out.push(tok);
        self.reqs.get_mut(&1).unwrap().seq.push(tok);
        // 纯 decode：逐 token 前向 → 采样 → 追加
        while (out.len() as u32) < max_tokens {
            let Some(st) = self.reqs.get(&1) else { break };
            let (row, pos) = (st.row, st.forwarded);
            let tok_in = st.seq[pos];
            let next = {
                self.reqs.get_mut(&1).unwrap().forwarded += 1;
                self.ensure_pages(1, pos);
                let logits = self.forward_rows(&[(row, tok_in, pos as u32)], true);
                crate::argmax(&logits)
            };
            out.push(next);
            self.reqs.get_mut(&1).unwrap().seq.push(next);
        }
        out
    }

    /// 清全部请求状态（页归还、块表行清零）——generate 的独立直跑前提
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
    }

    /// 页保证：请求的块表行长到覆盖 `upto_pos`（含）
    fn ensure_pages(&mut self, id: ReqId, upto_pos: usize) {
        let need = upto_pos / self.cfg.block_size as usize + 1;
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
                .unwrap_or_else(|| panic!("页池耗尽（{} 页）——toy 配置须保证容量", self.pool.total()));
            self.bt_host[row * bpr + st.blocks.len()] = b;
            st.blocks.push(b);
        }
    }

    /// 批前向（全部 M0 算子真路径，eager 直发）。rows: (块表行, token, 位置)。
    /// 返回 [T, vocab] f32（bf16 存储 → f32）。步内一次 sync（读 logits）
    fn forward_batch(&mut self, rows: &[(u32, u32, u32)]) -> Vec<f32> {
        // context 线程局部：engine 任务可在 worker 间迁移，进入设备路径先绑
        self.session.gpu().bind_thread();
        let t = rows.len();
        assert!(t > 0 && t <= self.t_max, "批大小 {t} 超前向缓冲上限 {}", self.t_max);
        for &(_, _, pos) in rows {
            assert!((pos as usize) < 256, "位置 {pos} 超 paged_attn M0 上限 256");
        }
        let gpu = self.session.gpu();
        let stream = self.session.stream();
        let qh = self.cfg.q_heads as usize;
        let ids: Vec<u32> = rows.iter().map(|r| r.1).collect();
        let pos: Vec<u32> = rows.iter().map(|r| r.2).collect();
        let seqs: Vec<u32> = rows.iter().map(|r| r.0).collect();
        // rope 以 (token,头) 为行——pos 按头数展开（每头同 pos）
        let posq: Vec<u32> = pos.iter().flat_map(|&p| std::iter::repeat_n(p, qh)).collect();
        let (d_ids, d_pos, d_seq, d_posq, d_bt) = (
            gpu.alloc_from_slice(&ids),
            gpu.alloc_from_slice(&pos),
            gpu.alloc_from_slice(&seqs),
            gpu.alloc_from_slice(&posq),
            gpu.alloc_from_slice(&self.bt_host),
        );
        self.run_forward(t as u32, &d_ids, &d_pos, &d_seq, &d_posq, &d_bt);
        stream.sync();
        // 上传缓冲在此之后才析构（sync 已过，无在飞 kernel）
        let v = self.cfg.vocab as usize;
        let bits: Vec<u16> = self.scr.logits.read_to_vec();
        bits[..t * v].iter().map(|&b| reference::bf16_to_f32(b)).collect()
    }

    /// 前向 kernel 序（模型序，不含上传/sync/回读）——eager 直发与图捕获
    /// 共用同一条序列：图是录制不是重写，位一致合同的实现前提（gr1）
    fn run_forward(
        &self,
        t: u32,
        d_ids: &DevBuf,
        d_pos: &DevBuf,
        d_seq: &DevBuf,
        d_posq: &DevBuf,
        d_bt: &DevBuf,
    ) {
        let stream = self.session.stream();
        let tb = self.session.table();
        let (h, i, v) =
            (self.cfg.hidden as usize, self.cfg.inter as usize, self.cfg.vocab as usize);
        let (qe, ke) = (self.q_elems, self.kv_elems);
        let (qh, kvh, hd) =
            (self.cfg.q_heads as u32, self.cfg.kv_heads as u32, self.cfg.head_dim as u32);
        let eps = self.cfg.eps;
        // ── 前向（顺序 = 调度无关的模型序；rope 原地（线程读写自分元素对））──
        tb.embed.run(d_ids, &self.w.embed, &self.scr.x, h as u32, t, stream);
        tb.rms_norm.run(&self.scr.x, &self.w.rms1, &self.scr.n1, t, h as u32, eps, stream);
        tb.gemm.run(&self.scr.n1, &self.w.wq, &self.scr.q, t, qe as u32, h as u32, stream);
        tb.gemm.run(&self.scr.n1, &self.w.wk, &self.scr.k, t, ke as u32, h as u32, stream);
        tb.gemm.run(&self.scr.n1, &self.w.wv, &self.scr.v, t, ke as u32, h as u32, stream);
        tb.rope.run(
            &self.scr.q, &self.scr.k, &self.cos, &self.sin, d_posq,
            &self.scr.q, &self.scr.k, t * qh, hd, stream,
        );
        tb.kv_scatter.run(
            &self.scr.k, &self.scr.v, d_bt, d_seq, d_pos,
            &self.k_pool, &self.v_pool, self.bpr as u32, self.cfg.block_size, ke as u32, t,
            stream,
        );
        tb.paged_attn.run(
            &self.scr.q, &self.k_pool, &self.v_pool, d_bt, d_seq, d_pos,
            &self.scr.attn, self.q_scale, self.bpr as u32, self.cfg.block_size,
            qh, kvh, hd, t, stream,
        );
        tb.gemm.run(&self.scr.attn, &self.w.wo, &self.scr.o, t, h as u32, qe as u32, stream);
        tb.add.run(&self.scr.x, &self.scr.o, &self.scr.h2, t * h as u32, stream);
        tb.rms_norm.run(&self.scr.h2, &self.w.rms2, &self.scr.n2, t, h as u32, eps, stream);
        tb.gemm.run(&self.scr.n2, &self.w.wg, &self.scr.g, t, i as u32, h as u32, stream);
        tb.gemm.run(&self.scr.n2, &self.w.wu, &self.scr.u, t, i as u32, h as u32, stream);
        tb.silu.run(&self.scr.g, &self.scr.act, t * i as u32, stream);
        tb.mul.run(&self.scr.act, &self.scr.u, &self.scr.m, t * i as u32, stream);
        tb.gemm.run(&self.scr.m, &self.w.wd, &self.scr.d, t, h as u32, i as u32, stream);
        tb.add.run(&self.scr.h2, &self.scr.d, &self.scr.h3, t * h as u32, stream);
        tb.gemm.run(&self.scr.h3, &self.w.lm, &self.scr.logits, t, v as u32, h as u32, stream);
    }

    /// 步前向统一入口：pure_decode（每行独立请求的 1-token 步）且桶覆盖 →
    /// 图回放；否则 eager 直发。混合步/prefill 永远 eager——图只在纯
    /// decode 档（批形状固定，无 host 侧数据依赖控制流，§4 入图安全规则）
    fn forward_rows(&mut self, rows: &[(u32, u32, u32)], pure_decode: bool) -> Vec<f32> {
        if pure_decode && !rows.is_empty() {
            if let Some(bi) = self.bucket_for(rows.len()) {
                return self.forward_graph(bi, rows);
            }
        }
        self.forward_batch(rows)
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
    /// seq=sink——KV 垃圾只落 sink 专属页）。输入按 **b_max 整段覆写**上传
    /// （桶共享单输入存储：各桶图录共享缓冲的 narrow view），一次 graph
    /// launch 重放全 kernel 序，回读前 m 行 logits
    fn forward_graph(&mut self, b_idx: usize, rows: &[(u32, u32, u32)]) -> Vec<f32> {
        self.session.gpu().bind_thread();
        let (b, cap, m, sink, qh) = {
            let g = self.graph.as_ref().expect("forward_graph 须图档");
            (g.buckets[b_idx] as usize, g.b_max as usize, rows.len(), g.sink_row, self.cfg.q_heads as usize)
        };
        assert!(m > 0 && m <= b, "桶路由合同：m {m} ≤ 桶 {b}");
        for &(_, _, pos) in rows {
            assert!((pos as usize) < 256, "位置 {pos} 超 paged_attn M0 上限 256");
        }
        // host 组装：b_max 全量（默认值 = sink 吸收行），前 m 行写真数据
        let mut ids = vec![0u32; cap];
        let mut pos = vec![0u32; cap];
        let mut seq = vec![sink; cap];
        let mut posq = vec![0u32; cap * qh];
        for (j, &(r, tok, p)) in rows.iter().enumerate() {
            ids[j] = tok;
            pos[j] = p;
            seq[j] = r;
            for k in 0..qh {
                posq[j * qh + k] = p;
            }
        }
        let stream = self.session.stream();
        {
            let g = self.graph.as_ref().expect("forward_graph 须图档");
            g.in_ids.upload_on(&ids, stream);
            g.in_pos.upload_on(&pos, stream);
            g.in_seq.upload_on(&seq, stream);
            g.in_posq.upload_on(&posq, stream);
            // 页保证已先于本调用完成（step/generate 合同）——bt_host 是最新账
            g.in_bt.upload_on(&self.bt_host, stream);
            g.execs[b_idx].launch(stream);
        }
        stream.sync();
        self.graph.as_mut().unwrap().replays += 1;
        let v = self.cfg.vocab as usize;
        let bits: Vec<u16> = self.scr.logits.read_to_vec();
        bits[..m * v].iter().map(|&x| reference::bf16_to_f32(x)).collect()
    }
}

impl Executor for ToyExecutor {
    fn admitted(&mut self, id: ReqId, prompt: &[u32]) {
        assert!(
            prompt.iter().all(|&t| t < self.cfg.vocab),
            "toy 词表越界（vocab {}）",
            self.cfg.vocab
        );
        // 空 prompt → BOS(0)：调度器看长度 0 直接 decode，首步前向 BOS
        let seq = if prompt.is_empty() { vec![0u32] } else { prompt.to_vec() };
        let row = self.free_rows.pop().unwrap_or_else(|| {
            assert!(
                self.next_row < self.row_capacity,
                "块表行耗尽（{}，图档含 sink 让位 1 行）——M0 toy 上限",
                self.row_capacity
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
        let mut samples: Vec<(ReqId, usize)> = Vec::new(); // (请求, 采样行)
        // 1. prefill 段（含追平：段尾行即采样位）
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
        // 2. 纯 decode：前向 seq 尾（上步采样追加、尚未前向的 token）
        for &id in &plan.decode {
            if plan.prefill.iter().any(|&(p, _)| p == id) {
                continue; // 追平步——采样位已在 prefill 段尾
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
        // 4. 前向 + 贪心（采样行各自的 logits 行）。prefill 空 = 纯 decode
        //    步（每行独立请求）→ 图档候选；否则 eager
        let rows: Vec<(u32, u32, u32)> = batch.iter().map(|r| (r.row, r.tok, r.pos)).collect();
        let logits = self.forward_rows(&rows, plan.prefill.is_empty());
        let v = self.cfg.vocab as usize;
        let mut out = Vec::new();
        for (id, row_i) in samples {
            let tok = crate::argmax(&logits[row_i * v..(row_i + 1) * v]);
            if let Some(st) = self.reqs.get_mut(&id) {
                st.seq.push(tok);
            }
            out.push((id, tok));
        }
        out
    }

    fn retain(&mut self, active: &[ReqId]) {
        let gone: Vec<ReqId> =
            self.reqs.keys().copied().filter(|k| !active.contains(k)).collect();
        for id in gone {
            if let Some(st) = self.reqs.remove(&id) {
                let base = st.row as usize * self.bpr;
                for (j, b) in st.blocks.iter().enumerate() {
                    self.bt_host[base + j] = 0;
                    self.pool.release(*b);
                }
                self.free_rows.push(st.row); // 行回收（bt 行已清零，复用安全）
            }
        }
        // deferred → free-list（步末回收点，m3 语义；toy 无 radix 共享，可急收）
        self.pool.recycle();
    }

    fn preempted(&mut self, id: ReqId) {
        if let Some(st) = self.reqs.get_mut(&id) {
            st.forwarded = 0; // recompute 重放：游标归零（KV 覆盖写幂等）
        }
    }
}
