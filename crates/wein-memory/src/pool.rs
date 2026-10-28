//! BlockPool：块级物理管理。free-list + 引用计数 + DeferredRelease + 水位。
//!
//! - **共享**：radix 命中的块被多请求共享——retain/release 成对，rc 归零才可回收
//! - **DeferredRelease**（02 §5）：rc 归零不立即回 free-list，先进 deferred 队列，
//!   步末 `recycle()` 统一回池。真实实现里这一步对应"步边界才改写设备侧
//!   req_to_token 表"——本步内还在被引用的结构不变更
//! - **水位**：free_now ≤ 低水位即报压（抢占触发判据）。判定用 free_now
//!   （悲观——deferred 在途块不算可用），m4 钉死；free_soon 语义（含在途）
//!   由 MemState::over_pressure 组合给出
use std::collections::HashMap;

pub type BlockId = u32;

pub struct BlockPool {
    free: Vec<BlockId>,
    /// 引用计数：alloc=1、retain+1、release-1；0 = 无主（在 deferred 里）
    rc: HashMap<BlockId, u32>,
    deferred: Vec<BlockId>,
    /// 低水位阈值（块数）：free_now ≤ 此值报压
    watermark: usize,
    total: u32,
}

impl BlockPool {
    pub fn new(num_blocks: u32, low_watermark: f32) -> Self {
        let threshold = ((num_blocks as f32) * low_watermark).round().max(1.0) as usize;
        Self {
            free: (0..num_blocks).rev().collect(), // LIFO：刚释放的块优先复用（局部性）
            rc: HashMap::new(),
            deferred: Vec::new(),
            watermark: threshold,
            total: num_blocks,
        }
    }

    /// 分配一块（rc=1）。耗尽返回 None——上层做准入控制与强制抢占，不 panic（m2）
    pub fn alloc(&mut self) -> Option<BlockId> {
        let id = self.free.pop()?;
        self.rc.insert(id, 1);
        Some(id)
    }

    /// 增加引用（radix 命中共享：缓存节点 + 请求各持一份）
    pub fn retain(&mut self, b: BlockId) {
        *self.rc.entry(b).or_insert(0) += 1;
    }

    /// 减引用。rc 归零 → deferred（跨步回收，不立即回 free——m3）
    pub fn release(&mut self, b: BlockId) {
        if let Some(rc) = self.rc.get_mut(&b) {
            *rc = rc.saturating_sub(1);
            if *rc == 0 {
                self.deferred.push(b);
            }
        }
    }

    /// 步末回收点：deferred 全部回 free-list，返回回收数
    pub fn recycle(&mut self) -> usize {
        let n = self.deferred.len();
        self.free.append(&mut self.deferred);
        n
    }

    pub fn rc_of(&self, b: BlockId) -> u32 {
        self.rc.get(&b).copied().unwrap_or(0)
    }

    pub fn owned(&self, b: BlockId) -> bool {
        self.rc_of(b) > 0
    }

    pub fn free_now(&self) -> usize {
        self.free.len()
    }

    pub fn deferred_len(&self) -> usize {
        self.deferred.len()
    }

    /// free_now + 在途 deferred：同一 step 内已被释放（如被抢占者）但未回收的块
    pub fn free_soon(&self) -> usize {
        self.free.len() + self.deferred.len()
    }

    /// 低水位判定（悲观：只看 free_now）
    pub fn over_watermark(&self) -> bool {
        self.free_now() <= self.watermark
    }

    /// 低水位阈值（块数）
    pub fn watermark(&self) -> usize {
        self.watermark
    }

    pub fn total(&self) -> u32 {
        self.total
    }
}
