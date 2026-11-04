//! MemState：把 pool/radix/r2t 组装成**容量模型语义**——wein-engine 侧的
//! 适配器把这些方法接到 wein_sched::CapacityModel 钩子上（本 crate 不依赖 sched）。
//!
//! 生命周期语义（对应 Removal 三态）：
//! - **准入** `admit`：radix 匹配（pin 命中块）→ 驱逐凑块 → 分配私有块 → 绑定。
//!   重准入（抢占保留的序列）以完整上下文匹配 → 廉价重算（m9）
//! - **增长** `on_token`：序列追加 + 跨界分配。false = 真放不下 → 调用方强制抢占
//! - **完成** `finish`：命中部分解 pin；私有整块移交 radix（rc 不变，所有权转移）；
//!   尾块 deferred
//! - **抢占** `release(keep_seq=true)`：块释放（deferred），序列保留
//! - **取消** `release(keep_seq=false)`：块释放，序列丢弃
//!
//! 判定口径（m10 钉死，两者刻意不同）：
//! - `can_admit`：free_now + 可驱逐 ≥ 需求 + 低水位余量（悲观——deferred 在途
//!   不算；余量是防抖：入批后须仍高于水位，否则同 step 内就被增长压力弹回，
//!   准入→抢占→重算乒乓，im1 抓到）
//! - `over_pressure`：free_soon ≤ 水位（含在途——抢占循环同一 step 内多次判定，
//!   被抢占者的块虽在 deferred 也必须算数，否则会过度驱逐）

use std::collections::HashMap;

use crate::pool::BlockPool;
use crate::radix::RadixCache;
use crate::r2t::{Push, ReqToToken};
use crate::ReqId;

#[derive(Clone, Copy, Debug)]
pub struct MemConfig {
    pub block_size: u32,
    pub num_blocks: u32,
    /// 低水位（池比例）：free_soon ≤ max(1, num_blocks × 比例) 报压
    pub low_watermark: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MemStats {
    pub blocks_allocated: u64,
    pub cache_hit_tokens: u64,
    pub evicted_blocks: u64,
    pub deferred_releases: u64,
    pub forced_preempts: u64,
}

pub struct MemState {
    bs: usize,
    pool: BlockPool,
    radix: RadixCache,
    r2t: ReqToToken,
    /// 请求 token 序列（跨抢占保留——重准入的匹配源）
    seqs: HashMap<ReqId, Vec<u32>>,
    stats: MemStats,
}

impl MemState {
    pub fn new(cfg: MemConfig) -> Self {
        let bs = cfg.block_size.max(1) as usize;
        Self {
            bs,
            pool: BlockPool::new(cfg.num_blocks, cfg.low_watermark),
            radix: RadixCache::new(),
            r2t: ReqToToken::new(bs),
            seqs: HashMap::new(),
            stats: MemStats::default(),
        }
    }

    pub fn stats(&self) -> MemStats {
        self.stats
    }

    pub fn free_blocks(&self) -> usize {
        self.pool.free_now()
    }

    fn blocks_needed(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.bs)
    }

    /// 悲观准入判定（不变更状态——真正的驱逐发生在 admit 里）
    pub fn can_admit(&self, ctx_len: usize) -> bool {
        let need = self.blocks_needed(ctx_len);
        // 水位是防振荡余量，不是容量税：整池单请求（如 32K 会话 × 2048 页）
        // 时钳到池余量——否则 need+watermark > total 永远不准入，只能走
        // 批空强制准入 → preempt → abort 的兜底链
        let wm = self
            .pool
            .watermark()
            .min((self.pool.total() as usize).saturating_sub(need));
        self.pool.free_now() + self.radix.evictable_blocks() >= need + wm
    }

    /// 压力判定（free_soon = free + 在途 deferred ≤ 低水位）
    pub fn over_pressure(&self) -> bool {
        self.pool.free_soon() <= self.pool.watermark()
    }

    /// 前缀缓存全清（`/reset_prefix_cache` 的记账侧）：驱逐全部可逐块
    /// （在飞 pin 块天然保留）并立即回收。返回清出的块数。
    pub fn clear_prefix_cache(&mut self) -> usize {
        let n = self.radix.evict(&mut self.pool, usize::MAX);
        self.pool.recycle();
        n
    }

    /// 准入（重准入含抢占保留的完整上下文）。false = 放不下（无副作用残留）
    pub fn admit(&mut self, req: ReqId, prompt: &[u32], generated: u32) -> bool {
        // 重准入：用保留的完整上下文（prompt + 已生成）匹配；首准入 generated 恒 0
        let seq: Vec<u32> = match self.seqs.remove(&req) {
            Some(kept) => kept,
            None => prompt.to_vec(),
        };
        let need = self.blocks_needed(seq.len());

        // 1) radix 匹配：pin 命中块（驱逐免疫 + rc 共享）
        let m = self.radix.match_prefix(&mut self.pool, &seq);
        let hit = m.blocks.len();
        self.stats.cache_hit_tokens += m.tokens as u64;

        // 2) 私有块缺口：先驱逐凑块（被逐块进 deferred → 立即 recycle 可用）
        let priv_need = need.saturating_sub(hit);
        if self.pool.free_now() < priv_need {
            let short = priv_need - self.pool.free_now();
            let evicted = self.radix.evict(&mut self.pool, short);
            self.stats.evicted_blocks += evicted as u64;
            self.pool.recycle();
        }

        // 3) 分配私有块；中途耗尽 → 回滚（解 pin + 放回已分配块 + 序列保留）
        let mut blocks = m.blocks;
        let mut got = 0usize;
        while got < priv_need {
            match self.pool.alloc() {
                Some(b) => {
                    self.stats.blocks_allocated += 1;
                    blocks.push(b);
                    got += 1;
                }
                None => {
                    self.radix.unpin(&mut self.pool, &seq, hit);
                    for &b in &blocks[hit..] {
                        self.pool.release(b);
                        self.stats.deferred_releases += 1;
                    }
                    self.seqs.insert(req, seq);
                    return false;
                }
            }
        }

        debug_assert_eq!(seq.len(), prompt.len() + generated as usize, "重准入序列须与调度器视图一致");
        self.r2t.bind(req, blocks, hit, seq.len());
        self.seqs.insert(req, seq);
        true
    }

    /// token 增长：跨界分配。false = 真的放不下（序列与布局未变更）→ 调用方强制抢占。
    /// 兜底链：在途 deferred 先回收（rc 归零的块随时安全，被抢占者刚释放的块
    /// 恰好可能是增长所需——im1 抓到）→ radix 驱逐一块 → 仍失败才认输。
    /// 上下文不超池容量的在跑请求绝不该失败（悲观口径只约束准入）
    pub fn on_token(&mut self, req: ReqId, token_id: u32) -> bool {
        if !self.seqs.contains_key(&req) {
            return false; // 无布局（admit 失败者）：不得为它驱逐缓存
        }
        match self.r2t.push(req, &mut self.pool) {
            Push::Ok => {}
            Push::OkAlloc(_) => self.stats.blocks_allocated += 1,
            Push::Exhausted => {
                self.pool.recycle(); // 兜底 1：在途块
                let evicted = self.radix.evict(&mut self.pool, 1); // 兜底 2：缓存
                self.stats.evicted_blocks += evicted as u64;
                if evicted > 0 {
                    self.pool.recycle();
                }
                match self.r2t.push(req, &mut self.pool) {
                    Push::Ok => {}
                    Push::OkAlloc(_) => self.stats.blocks_allocated += 1,
                    Push::Exhausted => return false, // 真耗尽
                }
            }
        }
        self.seqs.get_mut(&req).expect("admit 后必有序列").push(token_id);
        true
    }

    /// 完成：私有整块移交 radix（所有权转移，rc 不变），尾块 deferred
    pub fn finish(&mut self, req: ReqId) {
        let seq = match self.seqs.remove(&req) {
            Some(s) => s,
            None => return,
        };
        let layout = match self.r2t.remove(req) {
            Some(l) => l,
            None => return,
        };
        let full = seq.len() / self.bs; // 整块数
        let hit = layout.hit_len_blocks.min(full);

        // 命中部分解 pin（缓存继续持有）
        self.radix.unpin(&mut self.pool, &seq, hit);
        // 尾块（不足一块）释放
        for &b in &layout.blocks[full..] {
            self.pool.release(b);
            self.stats.deferred_releases += 1;
        }
        // 私有整块入缓存
        let new_blocks = &layout.blocks[hit..full];
        self.radix.insert(&seq[..full * self.bs], new_blocks);
    }

    /// 离批释放。keep_seq=true（抢占）：序列保留，重准入时完整上下文再匹配；
    /// keep_seq=false（取消）：序列丢弃。未知请求无操作
    pub fn release(&mut self, req: ReqId, keep_seq: bool) {
        let seq = match self.seqs.remove(&req) {
            Some(s) => s,
            None => return,
        };
        let layout = match self.r2t.remove(req) {
            Some(l) => l,
            None => {
                self.seqs.insert(req, seq);
                return;
            }
        };
        // 命中部分解 pin（缓存继续持有）
        self.radix.unpin(&mut self.pool, &seq, layout.hit_len_blocks);
        // 私有块全部释放
        for &b in &layout.blocks[layout.hit_len_blocks..] {
            self.pool.release(b);
            self.stats.deferred_releases += 1;
        }
        if keep_seq {
            self.seqs.insert(req, seq);
        }
    }

    /// 步末回收点：deferred 回池（DeferredRelease）
    pub fn step_end(&mut self) {
        self.pool.recycle();
    }
}
