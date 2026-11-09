//! DRAM 分层（W-Agent-5）：LRU 驱逐的 KV 块与快照状态下沉页锁定 host 内存，
//! admit 时按块哈希寻址搬回——"驱逐不失忆"。
//!
//! 对标 LMCache / SGLang HiRadix / mooncake 的块级分层，适配混合架构：
//! - **块哈希链**：`h(i) = mix(h(i-1), tokens[block..block+16])`——DRAM 条目
//!   的身份是 token 前缀（内容寻址），与 radix 树节点对齐；同前缀跨会话
//!   天然去重。
//! - **换出单位**：16-token 块的 KV（8 层 × K/V，每段 = 块×kv头×hd×esz）
//!   与恢复点状态（deltanet+conv 单行 ≈ 25.4MB）分开存——状态挂在其末块
//!   哈希上。恢复点换入要求该点之前全部块在（树命中段或 DRAM 补齐）。
//! - **搬运**：pinned slab + 异步逐段拷贝（探针双向 24-26GB/s）。
//!
//! 容量管理：两类 slab 各自定长槽 + FIFO 队列惰性 LRU（读命中重新入队，
//! 队首即最旧）。DramTier 不做策略——"存什么/何时查"由 executor 决定。

use wein_kernel::cuda::{DevBuf, Gpu, HostPinned, Stream};
use std::collections::{HashMap, VecDeque};

/// 块哈希链混合（FNV-1a 变体）：父哈希 + 本块 token
pub fn block_hash(parent: u64, tokens: &[u32]) -> u64 {
    let mut h = parent ^ 0x9E37_79B9_7F4A_7C15;
    for &t in tokens {
        h = (h ^ (t as u64)).wrapping_mul(0x100_0000_01B3);
    }
    h
}

struct BlockEntry {
    slot: u32,
    tick: u64,
}
struct StateEntry {
    slot: u32,
    /// 恢复点覆盖的 token 数（admit 换入时验证恢复点位置）
    n_tokens: usize,
    tick: u64,
}

/// DRAM 分层池：块 KV 与恢复点状态两个 pinned slab。
pub struct DramTier<'g> {
    blk: HostPinned<'g>,
    /// 单层单片 KV 段字节（块×kv头×hd×esz）；每块 16 段（8 层×K/V）
    blk_seg: usize,
    blk_free: Vec<u32>,
    blk_map: HashMap<u64, BlockEntry>,
    blk_fifo: VecDeque<(u64, u64)>,
    st: Option<HostPinned<'g>>,
    /// 状态槽字节 = lin_row + conv_row（构造时定）
    st_stride: usize,
    st_free: Vec<u32>,
    st_map: HashMap<u64, StateEntry>,
    st_fifo: VecDeque<(u64, u64)>,
    tick: u64,
    pub spilled_blocks: u64,
    pub fetched_blocks: u64,
    pub spilled_states: u64,
    pub fetched_states: u64,
}

impl<'g> DramTier<'g> {
    /// `seg_bytes`：单层单片 KV 段字节；`n_blk_slots`：块槽数；
    /// `n_state_slots`（0 = 不分层状态）/`state_bytes`：状态层。
    pub fn new(
        gpu: &'g Gpu,
        seg_bytes: usize,
        n_blk_slots: usize,
        n_state_slots: usize,
        state_bytes: usize,
    ) -> Self {
        let st = (n_state_slots > 0)
            .then(|| gpu.alloc_host_bytes(n_state_slots * state_bytes));
        Self {
            blk: gpu.alloc_host_bytes(n_blk_slots * seg_bytes * 16),
            blk_seg: seg_bytes,
            blk_free: (0..n_blk_slots as u32).rev().collect(),
            blk_map: Default::default(),
            blk_fifo: Default::default(),
            st,
            st_stride: state_bytes,
            st_free: (0..n_state_slots as u32).rev().collect(),
            st_map: Default::default(),
            st_fifo: Default::default(),
            tick: 0,
            spilled_blocks: 0,
            fetched_blocks: 0,
            spilled_states: 0,
            fetched_states: 0,
        }
    }

    /// 换出一块：`segs` = (设备缓冲, 段起始字节)×16（8 层×K/V，每段 blk_seg）
    pub fn spill_block(&mut self, hash: u64, segs: &[(&DevBuf, usize)], stream: &Stream) {
        debug_assert_eq!(segs.len(), 16, "8 层 × K/V");
        if self.blk_map.contains_key(&hash) {
            return; // 共享前缀已在 DRAM——不重复存
        }
        if self.blk_free.is_empty() {
            while let Some((h, tick)) = self.blk_fifo.pop_front() {
                if let Some(e) = self.blk_map.get(&h) {
                    if e.tick == tick {
                        let slot = e.slot;
                        self.blk_map.remove(&h);
                        self.blk_free.push(slot);
                        break;
                    }
                }
            }
            if self.blk_free.is_empty() {
                return;
            }
        }
        let slot = self.blk_free.pop().expect("回收保证");
        for (i, (dev, start)) in segs.iter().enumerate() {
            self.blk.download_seg_on(
                slot as usize * 16 * self.blk_seg + i * self.blk_seg,
                dev,
                *start,
                self.blk_seg,
                stream,
            );
        }
        self.tick += 1;
        self.blk_map.insert(hash, BlockEntry { slot, tick: self.tick });
        self.blk_fifo.push_back((hash, self.tick));
        self.spilled_blocks += 1;
    }

    /// 换出一恢复点状态：`segs`/`seg_bytes` = 快照池各层槽段与其字节
    pub fn spill_state(
        &mut self,
        hash: u64,
        n_tokens: usize,
        segs: &[(&DevBuf, usize)],
        seg_bytes: &[usize],
        stream: &Stream,
    ) -> bool {
        let Some(ref st) = self.st else { return false };
        if self.st_map.contains_key(&hash) {
            return true;
        }
        if self.st_free.is_empty() {
            while let Some((h, tick)) = self.st_fifo.pop_front() {
                if let Some(e) = self.st_map.get(&h) {
                    if e.tick == tick {
                        let slot = e.slot;
                        self.st_map.remove(&h);
                        self.st_free.push(slot);
                        break;
                    }
                }
            }
            if self.st_free.is_empty() {
                return false; // 状态槽尽：本恢复点不存（降级）
            }
        }
        let slot = self.st_free.pop().expect("回收保证");
        let mut off = 0usize;
        for ((dev, start), bytes) in segs.iter().zip(seg_bytes) {
            st.download_seg_on(slot as usize * self.st_stride + off, dev, *start, *bytes, stream);
            off += bytes;
        }
        self.tick += 1;
        self.st_map.insert(hash, StateEntry { slot, n_tokens, tick: self.tick });
        self.st_fifo.push_back((hash, self.tick));
        self.spilled_states += 1;
        true
    }

    pub fn has_block(&self, hash: u64) -> bool {
        self.blk_map.contains_key(&hash)
    }
    pub fn has_state(&self, hash: u64) -> bool {
        self.st_map.contains_key(&hash)
    }
    /// 恢复点覆盖 token 数（换入规划：状态位置 = 快照点）
    pub fn state_tokens(&self, hash: u64) -> Option<usize> {
        self.st_map.get(&hash).map(|e| e.n_tokens)
    }

    /// 换入一块（H2D 到 `segs`；tick 刷新 = 读命中也算使用）
    pub fn fetch_block(&mut self, hash: u64, segs: &[(&DevBuf, usize)], stream: &Stream) -> bool {
        let Some(e) = self.blk_map.get(&hash) else { return false };
        let slot = e.slot;
        for (i, (dev, start)) in segs.iter().enumerate() {
            self.blk.upload_seg_on(
                slot as usize * 16 * self.blk_seg + i * self.blk_seg,
                dev,
                *start,
                self.blk_seg,
                stream,
            );
        }
        self.tick += 1;
        self.blk_map.get_mut(&hash).expect("刚查过").tick = self.tick;
        self.blk_fifo.push_back((hash, self.tick));
        self.fetched_blocks += 1;
        true
    }

    /// 换入恢复点状态（H2D 到快照池槽段）
    pub fn fetch_state(
        &mut self,
        hash: u64,
        segs: &[(&DevBuf, usize)],
        seg_bytes: &[usize],
        stream: &Stream,
    ) -> bool {
        let Some(ref st) = self.st else { return false };
        let Some(e) = self.st_map.get(&hash) else { return false };
        let slot = e.slot;
        let mut off = 0usize;
        for ((dev, start), bytes) in segs.iter().zip(seg_bytes) {
            st.upload_seg_on(slot as usize * self.st_stride + off, dev, *start, *bytes, stream);
            off += bytes;
        }
        self.tick += 1;
        self.st_map.get_mut(&hash).expect("刚查过").tick = self.tick;
        self.st_fifo.push_back((hash, self.tick));
        self.fetched_states += 1;
        true
    }
}
