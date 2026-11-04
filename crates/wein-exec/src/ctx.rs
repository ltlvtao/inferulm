//! 上下文容量配置（W2 B2）：把 paged_attn 那套"编译期栅格"
//! （`MAX_POS=256 / NUM_PAGES=512 / BLOCK=16`）变成**启动期参数**。
//!
//! 默认 { max_pos 256, kv_pages 512, block 16, snap_stride 16, t_max 自动 }
//! ——默认路径逐字节不变（t_max 自动档 = 512 原常量）。来源优先级：显式
//! [`set`]（cli 的 `--ctx-len` / `--kv-pages` / `--snap-stride` /
//! `--t-max`）> 环境变量（`WEIN_CTX_LEN` / `WEIN_KV_PAGES` /
//! `WEIN_CTX_BLOCK` / `WEIN_SNAP_STRIDE` / `WEIN_T_MAX`）> 默认值。
//! `t_max` 未显式指定时按 max_pos 自动取档（≥512，封顶 8192——长上下文
//! ⟹ prefill 大块 lane）。首次取值时懒初始化一次并全程只读（exec 单进程
//! 单引擎；serve 的 `ctx_limit()` 与 cli 的 KV 水位模型读同一份，不各写魔数）。
//!
//! 账目：`kv_pages × block` = 全池 token 数（8 层 × 4 kv 头 × 256 hd ×
//! k/v × 2B = 32KB/token）；`max_pos` 是单行绝对位置上界（= `ctx_limit()`）。
//! 池装不下 `max_pos` 时启动即报错；单条请求超池由 `BlockPool` 的
//! Capacity 失败路径（非 panic）终结。

use std::sync::OnceLock;

/// 一对"token 预算"口径的启动期容量参数
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CtxConfig {
    /// 单行绝对位置上界（paged_attn 的 pos 上限 = `ctx_limit()`）
    pub max_pos: usize,
    /// KV 页数（全注意力层共用一个池）
    pub kv_pages: usize,
    /// 页大小（token/页）：块表宽度 `bpr = max_pos / block`
    pub block: usize,
    /// 快照边界步长（token，须为 block 的整数倍）：默认 = block（逐块快照，
    /// 位级同旧形态）；长上下文抬高它换取"槽够用"
    pub snap_stride: usize,
    /// 前缀快照槽数（每槽 ≈25.2MB 线性层 + conv 状态，与 ctx 无关——
    /// 槽压随 kv_pages/会话数增长）。默认 64 = 旧常量；0 = 关快照池
    pub snap_slots: usize,
    /// 批前向行（token）上限——exec 激活/输入表的定尺（旧编译期 T_MAX=512）。
    /// 0 = 未显式指定（`resolve` 时按 max_pos 自动取档，见 [`Self::resolve`]）
    pub t_max: usize,
}

/// t_max 自动档：长上下文 ⟹ 单请求可整段入批（chunk 受调度侧控制，
/// 缓冲先给足）；上限 8192 是 24GB 卡的激活缓冲显存账（~2.3GB，见
/// qwen.rs 的 ScrDev 注释）。默认 512 = 旧常量，短上下文路径逐字节不变
pub const T_MAX_CEIL: usize = 8192;
pub const T_MAX_FLOOR: usize = 512;

impl CtxConfig {
    /// t_max 的自动档（max_pos ≥ 512 时跟到 min(max_pos, 8192)）
    pub fn auto_t_max(max_pos: usize) -> usize {
        T_MAX_FLOOR.max(max_pos.min(T_MAX_CEIL))
    }
}

impl Default for CtxConfig {
    fn default() -> Self {
        Self {
            max_pos: 256,
            kv_pages: 512,
            block: 16,
            snap_stride: 16,
            snap_slots: 64,
            // 0 = 未指定（`resolve` 填自动档；默认路径 = 512 原值）
            t_max: 0,
        }
    }
}

impl CtxConfig {
    /// 环境变量打底（cli 用：flag 覆盖再走 `set`）。`WEIN_T_MAX` 未设时
    /// t_max 留 0（未指定），由 [`Self::resolve`] 按 max_pos 取自动档
    pub fn from_env() -> Self {
        let d = Self::default();
        let num = |k: &str, dv: usize| -> usize {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(dv)
        };
        Self {
            max_pos: num("WEIN_CTX_LEN", d.max_pos),
            kv_pages: num("WEIN_KV_PAGES", d.kv_pages),
            block: num("WEIN_CTX_BLOCK", d.block),
            snap_stride: num("WEIN_SNAP_STRIDE", d.snap_stride),
            snap_slots: num("WEIN_SNAP_SLOTS", d.snap_slots),
            t_max: num("WEIN_T_MAX", d.t_max),
        }
    }

    /// 显式给定的 t_max（env/flag）；None = 未指定 → 自动档
    pub fn t_max_explicit(&self) -> Option<usize> {
        (self.t_max != 0).then_some(self.t_max)
    }

    /// 未显式指定时按 max_pos 取自动档（cli 在 flag 全部落地后调用一次）
    pub fn resolve(&mut self) {
        if self.t_max == 0 {
            self.t_max = Self::auto_t_max(self.max_pos);
        }
    }

    /// 启动期校验（不可满足的容量组合直接拒绝启动，不带病跑）
    pub fn validate(&self) -> Result<(), String> {
        if self.block == 0 {
            return Err("block（页大小）须 > 0".into());
        }
        if self.max_pos < 2 * self.block {
            return Err(format!(
                "max_pos {} 小于 2×block {}（上下文至少容两个页）",
                self.max_pos, self.block
            ));
        }
        if self.kv_pages == 0 {
            return Err("kv_pages 须 > 0".into());
        }
        if self.max_pos > self.kv_pages * self.block {
            return Err(format!(
                "上下文 {} token 超 KV 池 {} token（kv_pages {} × block {}）——请抬 --kv-pages",
                self.max_pos,
                self.kv_pages * self.block,
                self.kv_pages,
                self.block
            ));
        }
        if self.snap_stride == 0 || self.snap_stride % self.block != 0 {
            return Err(format!(
                "snap_stride {} 须为 block {} 的正整数倍",
                self.snap_stride, self.block
            ));
        }
        if self.t_max == 0 || self.t_max > 65536 {
            return Err(format!("t_max {} 须在 1..=65536（批前向行上限）", self.t_max));
        }
        Ok(())
    }

    /// 单行页表宽度（块数）= max_pos / block 上取整
    pub fn bpr(&self) -> usize {
        self.max_pos.div_ceil(self.block)
    }

    /// 全池 token 数
    pub fn kv_tokens(&self) -> usize {
        self.kv_pages * self.block
    }
}

static CTX: OnceLock<CtxConfig> = OnceLock::new();

/// 启动期装配（cli）。重复 set 且值不同 = 装配冲突（显式报错）
pub fn set(c: CtxConfig) -> Result<(), String> {
    let mut c = c;
    c.resolve(); // t_max 未显式给 → 按 max_pos 自动档
    c.validate()?;
    match CTX.set(c) {
        Ok(()) => Ok(()),
        Err(prev) if prev == c => Ok(()),
        Err(prev) => Err(format!("上下文配置已装配为 {prev:?}，拒绝改成 {c:?}")),
    }
}

/// 当前配置（首次访问时按环境变量初始化——lib/examples 无需显式装配）
pub fn get() -> &'static CtxConfig {
    CTX.get_or_init(|| {
        let mut c = CtxConfig::from_env();
        c.resolve();
        c
    })
}

/// 启动期兜底校验（`Qwen35Executor::open` / `ToyExecutor::open` 调）
pub fn check() -> Result<(), String> {
    get().validate()
}

/// 批前向行（token）上限（旧编译期 T_MAX=512 的启动期参数化）
pub fn t_max() -> usize {
    get().t_max
}

pub fn max_pos() -> usize {
    get().max_pos
}

pub fn kv_pages() -> u32 {
    get().kv_pages as u32
}

pub fn block() -> u32 {
    get().block as u32
}

pub fn snap_stride() -> usize {
    get().snap_stride
}

pub fn snap_slots() -> usize {
    get().snap_slots
}

/// 单行页表宽度（块数）——块表/图档 `in_bt`/host `bt_host` 的列宽
pub fn bpr() -> usize {
    get().bpr()
}
