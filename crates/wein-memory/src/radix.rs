//! radix 前缀缓存索引：token 序列 → 块的映射（SGLang radix_cache 结构，M0 简化版）。
//!
//! - **索引与数据分离**：节点只持 BlockId，不持 KV 数据（数据在设备显存）
//! - **块对齐**：节点边 = 整块（block_size 个 token）；匹配/插入都只处理整块，
//!   部分尾不属于缓存（m5）
//! - **pin（双计数分账）**：`lock_ref` = 运行中请求持有（match_prefix
//!   加、unpin 减）；`hold` = 挂表/快照链保护（pin_block 加）。两者都
//!   构成驱逐免疫，但只有 hold 可被 DRAM 下沉后强解（force_unhold）——
//!   lock_ref 是活请求的块表页依赖，动了就是正确性事故
//! - **驱逐**：只摘未 pin 的叶子，按 LRU；父节点有子嗣则保留（索引职责）。
//!   M0 无节点分裂：一次 insert 的新块串成一条链（子节点按键整块内容索引
//!   ⇒ 同首 token 的不同块并存——首 token 单键会让后插覆盖先插）
//! - **LRU touch**：命中/插入刷新 last_access——同一前缀反复使用不老化
use std::collections::HashMap;

use crate::pool::{BlockId, BlockPool};

#[derive(Debug, Default)]
struct Node {
    block: Option<BlockId>,
    /// 子节点按下一块的整块 token 内容索引（键即该块的 token，root 恒空）。
    /// 块对齐 ⇒ 键长恒 bs，同首 token 的不同块因此并存
    children: HashMap<Vec<u32>, Node>,
    /// 运行中请求持有数（match_prefix 加、unpin 减）。**只属于活请求**——
    /// 它是块表页的读写依赖，驱逐/强解都会让活请求读到复用后的脏页。
    /// 挂表/快照链保护走独立的 `hold`，两者共同构成驱逐免疫
    lock_ref: u32,
    /// 挂表（快照链）保护数：pin_block 加、unpin_block 减。与 lock_ref
    /// 分账是 DRAM 分层的前提——强解（force_unhold）只清它，永不动
    /// 活请求的 lock_ref
    hold: u32,
    last_access: u64,
}

#[derive(Debug, Default, PartialEq)]
pub struct PrefixMatch {
    pub blocks: Vec<BlockId>,
    pub tokens: usize,
}

pub struct RadixCache {
    root: Node,
    clock: u64,
    /// 块大小，首次 insert 时由 tokens/blocks 比例推断
    bs: Option<usize>,
}

impl RadixCache {
    pub fn new() -> Self {
        Self { root: Node::default(), clock: 0, bs: None }
    }

    /// 最长块对齐前缀匹配。命中块 lock_ref+1 并 pool.retain（pin——驱逐免疫）。
    pub fn match_prefix(&mut self, pool: &mut BlockPool, tokens: &[u32]) -> PrefixMatch {
        let Self { root, clock, bs } = self;
        let bs = match *bs {
            Some(b) => b,
            None => return PrefixMatch::default(), // 空树
        };
        let mut blocks = Vec::new();
        let mut pos = 0;
        let mut cur = root;
        while pos + bs <= tokens.len() {
            let child = match cur.children.get_mut(&tokens[pos..pos + bs]) {
                Some(c) => c,
                None => break,
            };
            *clock += 1;
            child.last_access = *clock;
            child.lock_ref += 1;
            let b = child.block.expect("非 root 节点必有块");
            pool.retain(b);
            blocks.push(b);
            cur = child;
            pos += bs;
        }
        PrefixMatch { blocks, tokens: pos }
    }

    /// 解除前 n_blocks 个命中块的 pin（lock_ref-1 + pool.release）。
    /// 请求离批（完成/抢占/取消）时对其命中部分调用。
    pub fn unpin(&mut self, pool: &mut BlockPool, tokens: &[u32], n_blocks: usize) {
        self.unpin_range(pool, tokens, 0, n_blocks);
    }

    /// 解除第 [from, to) 块的 pin（match 全量命中后部分退回：命中截断到
    /// 有状态快照槽的祖先——尾部无槽块不共享，prefill 重算走私有页，
    /// 避免 ulp 漂移覆盖写污染共享缓存）。
    pub fn unpin_range(&mut self, pool: &mut BlockPool, tokens: &[u32], from: usize, to: usize) {
        let bs = match self.bs {
            Some(b) => b,
            None => return,
        };
        let mut cur = &mut self.root;
        for i in 0..to {
            let pos = i * bs;
            if pos + bs > tokens.len() {
                return;
            }
            let child = match cur.children.get_mut(&tokens[pos..pos + bs]) {
                Some(c) => c,
                None => return,
            };
            if i >= from {
                child.lock_ref = child.lock_ref.saturating_sub(1);
                if let Some(b) = child.block {
                    pool.release(b);
                }
            }
            cur = child;
        }
    }

    /// 沿 tokens 走树返回已覆盖的整块深度（不 pin、不 touch）。移交方
    /// 用它判断私有块与既有覆盖的重叠段——重叠段是冗余页应释放，只把
    /// 分叉后的新段交给 insert（insert 的 blocks 契约 = 分叉点对齐）。
    pub fn coverage(&self, tokens: &[u32]) -> usize {
        let bs = match self.bs {
            Some(b) => b,
            None => return 0,
        };
        let mut cur = &self.root;
        let mut depth = 0;
        let mut pos = 0;
        while pos + bs <= tokens.len() {
            let child = match cur.children.get(&tokens[pos..pos + bs]) {
                Some(c) => c,
                None => break,
            };
            cur = child;
            depth += 1;
            pos += bs;
        }
        depth
    }

    /// 插入：沿既有路径走（该部分已在缓存），在分叉处把新块串成链追加。
    /// blocks 是调用方的私有整块（finish 时从请求移交给缓存，rc 不变——所有权转移）。
    pub fn insert(&mut self, tokens: &[u32], blocks: &[BlockId]) {
        if blocks.is_empty() {
            return;
        }
        if self.bs.is_none() {
            let b = tokens.len() / blocks.len();
            debug_assert!(b > 0 && b * blocks.len() <= tokens.len(), "insert 必须整块对齐");
            self.bs = Some(b);
        }
        let bs = self.bs.unwrap();
        let Self { root, clock, .. } = self;
        Self::insert_rec(root, tokens, blocks, bs, clock);
    }

    /// 下钻既有路径；首处不匹配/缺失即分叉挂链（blocks 只覆盖分叉后的新段）
    fn insert_rec(
        node: &mut Node,
        tokens: &[u32],
        blocks: &[BlockId],
        bs: usize,
        clock: &mut u64,
    ) {
        if blocks.is_empty() || tokens.len() < bs {
            return;
        }
        let matched = match node.children.get_mut(&tokens[..bs]) {
            Some(c) => c,
            None => {
                Self::append_chain(node, tokens, blocks, bs, clock);
                return;
            }
        };
        *clock += 1;
        matched.last_access = *clock;
        Self::insert_rec(matched, &tokens[bs..], blocks, bs, clock);
    }

    fn append_chain(
        node: &mut Node,
        tokens: &[u32],
        blocks: &[BlockId],
        bs: usize,
        clock: &mut u64,
    ) {
        let Some((&b, rest)) = blocks.split_first() else { return };
        if tokens.len() < bs {
            return;
        }
        *clock += 1;
        let child = node.children.entry(tokens[..bs].to_vec()).or_insert_with(|| Node {
            block: Some(b),
            children: HashMap::new(),
            lock_ref: 0,
            hold: 0,
            last_access: *clock,
        });
        child.last_access = *clock;
        Self::append_chain(child, &tokens[bs..], rest, bs, clock);
    }

    /// 活请求份加锁（按块 id）：`lock_ref+1 + pool.retain`。用于 DRAM 挂载
    /// 段——那些块在准入时就被 insert 进树（树份已转移），但请求在飞期间
    /// 仍要读它们，必须与 `match_prefix` 的命中链同款保护；否则池压一上来
    /// LRU 就把它们逐出并复用，`bt_host` 指向别人的 KV（retain 的 coverage
    /// 合同随之炸裂）。退还走 [`Self::unpin`]（沿 token 路径，对称）。
    pub fn lock_block(&mut self, pool: &mut BlockPool, block: u32) -> bool {
        if let Some(node) = Self::find_block_mut(&mut self.root, block) {
            node.lock_ref += 1;
            pool.retain(block);
            true
        } else {
            false
        }
    }

    /// 快照链保护：按块 id 找到树上节点并 hold+1 + pool.retain（驱逐
    /// 免疫）。找不到（已逐/从未插入）返回 false。计数语义与 match_prefix
    /// 的 lock_ref **分账**：hold 属于挂表保护（可 force_unhold 强解），
    /// lock_ref 属于活请求（永不强解）。
    pub fn pin_block(&mut self, pool: &mut BlockPool, block: u32) -> bool {
        if let Some(node) = Self::find_block_mut(&mut self.root, block) {
            node.hold += 1;
            pool.retain(block);
            true
        } else {
            false
        }
    }

    /// pin_block 的解除。不在树上时静默返回（块可能已被整体逐出）。
    pub fn unpin_block(&mut self, pool: &mut BlockPool, block: u32) {
        if let Some(node) = Self::find_block_mut(&mut self.root, block) {
            node.hold = node.hold.saturating_sub(1);
            pool.release(block);
        }
    }

    /// 强解挂表保护：把块的 hold 清零并按次数 release。DRAM 下沉后块
    /// 内容可弃，挂表保护随之让位。**永不动 lock_ref**——那是运行中
    /// 请求的块表页依赖（清了 = 共享页被逐+复用 = 活请求读到别人的 KV，
    /// retain 的 coverage 合同随之炸裂）
    pub fn force_unhold(&mut self, pool: &mut BlockPool, block: u32) {
        if let Some(node) = Self::find_block_mut(&mut self.root, block) {
            for _ in 0..node.hold {
                pool.release(block);
            }
            node.hold = 0;
        }
    }

    fn find_block_mut(node: &mut Node, block: u32) -> Option<&mut Node> {
        if node.block == Some(block) {
            return Some(node);
        }
        for c in node.children.values_mut() {
            if let Some(hit) = Self::find_block_mut(c, block) {
                return Some(hit);
            }
        }
        None
    }

    /// LRU 驱逐：摘 need 块（只摘未 pin 叶子；父节点保留）。被逐块 pool.release
    /// （进 deferred——调用方在准入路径显式 recycle）。返回实际驱逐数。
    pub fn evict(&mut self, pool: &mut BlockPool, need: usize) -> usize {
        self.evict_blocks(pool, need).len()
    }

    /// evict 的块列表版：快照槽挂块的联动钩子——块离开缓存时挂其上的
    /// 槽必须同步回收，否则块 id 复用后映射残留 = 错误命中。
    pub fn evict_blocks(&mut self, pool: &mut BlockPool, need: usize) -> Vec<BlockId> {
        // with_capacity 只是提示：usize::MAX（clear-all 语义）会容量溢出 panic，
        // 钳到 256——真实驱逐数由 find_lru_leaf 的 None 出口决定
        let mut freed = Vec::with_capacity(need.min(256));
        while freed.len() < need {
            let (_, block, path) = match Self::find_lru_leaf(&self.root, &mut Vec::new()) {
                Some(b) => b,
                None => break, // 没有可驱逐的（全被 pin 或空）
            };
            // path = root→叶子的键路径；走到父节点摘除叶子
            let (leaf_key, parent_path) = path.split_last().expect("find 保证非空");
            let mut cur = &mut self.root;
            for k in parent_path {
                cur = cur.children.get_mut(k).expect("find 走过的路径必在");
            }
            let node = cur.children.remove(leaf_key).expect("find 保证叶子存在");
            debug_assert_eq!(node.block, Some(block));
            pool.release(block);
            freed.push(block);
        }
        freed
    }

    /// 驱逐预览：返回接下来 `evict_blocks(need)` 将摘除的块及其**全前缀
    /// tokens**（root→该块），不执行。DRAM 分层在真正驱逐前用它把块 KV
    /// 与挂载快照下沉（前缀 = 块哈希链的身份来源）。
    ///
    /// 模拟等价性：逐出循环每轮摘全局最旧可逐叶（摘叶后父变叶继续可逐），
    /// = 按 last_access 升序取「子树无 pin」节点。与 `evictable_blocks`
    /// 同语义：含 pin 的子树整体不可逐（摘 pin 的祖先会孤立它）。
    pub fn evict_frontier(&self, need: usize) -> Vec<(BlockId, Vec<u32>)> {
        // 键在父的 children 表里（Node 自身不存键）——递归时携带
        fn collect(
            node: &Node,
            key: &[u32],
            prefix: &mut Vec<u32>,
            out: &mut Vec<(u64, BlockId, Vec<u32>)>,
        ) -> bool {
            if node.lock_ref > 0 || node.hold > 0 {
                return false;
            }
            for (k, c) in &node.children {
                if !collect(c, k, prefix, out) {
                    return false; // 某子方向有 pin：本节点作为其祖先也不可逐
                }
            }
            prefix.extend_from_slice(key);
            if let Some(b) = node.block {
                out.push((node.last_access, b, prefix.clone()));
            }
            prefix.truncate(prefix.len() - key.len());
            true
        }
        let mut all = Vec::new();
        for (k, c) in &self.root.children {
            let mut p = Vec::new();
            collect(c, k, &mut p, &mut all);
        }
        all.sort_by_key(|(age, _, _)| *age);
        all.into_iter().take(need).map(|(_, b, pfx)| (b, pfx)).collect()
    }

    /// 可驱逐块数——MemState 悲观准入的组成部分。
    /// 与 evict() 的级联语义一致：摘掉叶子后父节点变叶子可继续摘，
    /// 所以计数 = 子树内无 pin 依赖的节点总数（而非仅当前叶子——
    /// 那会系统性低估准入能力，m10 抓到）。含 pin 的子树整体不可逐
    /// （pin 节点的祖先是它的索引路径）
    pub fn evictable_blocks(&self) -> usize {
        let mut total = 0;
        for c in self.root.children.values() {
            if let Some(n) = Self::subtree_evictable(c) {
                total += n;
            }
        }
        total
    }

    /// Some(n)：子树全部无 pin（lock_ref 与 hold 双清），共 n 节点；
    /// None：子树内含 pin（整体不可逐）
    fn subtree_evictable(node: &Node) -> Option<usize> {
        if node.lock_ref > 0 || node.hold > 0 {
            return None;
        }
        let mut n = 1;
        for c in node.children.values() {
            n += Self::subtree_evictable(c)?;
        }
        Some(n)
    }

    /// DFS 找子树内最旧的未 pin 叶子，返回 (age, block, root→叶子键路径)
    fn find_lru_leaf(
        node: &Node,
        prefix: &mut Vec<Vec<u32>>,
    ) -> Option<(u64, BlockId, Vec<Vec<u32>>)> {
        let mut best: Option<(u64, BlockId, Vec<Vec<u32>>)> = None;
        for (k, c) in &node.children {
            if c.children.is_empty() && c.lock_ref == 0 && c.hold == 0 {
                let mut p = prefix.clone();
                p.push(k.clone());
                let cand = (c.last_access, c.block.expect("叶子必有块"), p);
                if best.as_ref().map_or(true, |b| cand.0 < b.0) {
                    best = Some(cand);
                }
            } else {
                prefix.push(k.clone());
                if let Some(cand) = Self::find_lru_leaf(c, prefix) {
                    if best.as_ref().map_or(true, |b| cand.0 < b.0) {
                        best = Some(cand);
                    }
                }
                prefix.pop();
            }
        }
        best
    }
}
