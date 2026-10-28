//! req_to_token 的宿主镜像：请求 → 块布局（blocks / 命中块数 / token 长度）。
//! 真实实现是设备侧 flat u32 表（req 槽 × max_context 的两级索引，02 §5），
//! kernel 时代接入；M0 先由宿主侧镜像承载语义，接口不再变。
//!
//! push 的边界语义（m7 钉死）：token 跨过 ceil(len/bs) 边界才向 pool 要新块——
//! 半满尾块继续填。池耗尽返回 false（调用方触发强制抢占），绝不 panic。

use std::collections::HashMap;

use crate::pool::{BlockId, BlockPool};
use crate::ReqId;

/// push 的三种结果（分配时机对外可见——MemState 据此记账 blocks_allocated）
#[derive(Debug, PartialEq)]
pub enum Push {
    /// 已追加，未分配新块
    Ok,
    /// 已追加，跨界分配了新块
    OkAlloc(BlockId),
    /// 池耗尽（布局未变更）——调用方触发强制抢占
    Exhausted,
}

#[derive(Debug, Default, PartialEq)]
pub struct ReqLayout {
    /// 块序列：前 hit_len_blocks 块为 radix 命中（缓存共享），其余为私有
    pub blocks: Vec<BlockId>,
    pub hit_len_blocks: usize,
    /// 已绑定的 token 数（prompt + 已生成）
    pub len: usize,
}

pub struct ReqToToken {
    map: HashMap<ReqId, ReqLayout>,
    bs: usize,
}

impl ReqToToken {
    pub fn new(block_size: usize) -> Self {
        Self { map: HashMap::new(), bs: block_size.max(1) }
    }

    pub fn bind(&mut self, req: ReqId, blocks: Vec<BlockId>, hit_len_blocks: usize, len: usize) {
        debug_assert!(len <= blocks.len() * self.bs, "token 数不得超过块容量");
        self.map.insert(req, ReqLayout { blocks, hit_len_blocks, len });
    }

    /// 追加一个 token 的结果（m7 钉死边界语义）：
    /// 跨过 ceil(len/bs) 边界才向 pool 要新块——半满尾块继续填
    pub fn push(&mut self, req: ReqId, pool: &mut BlockPool) -> Push {
        let bs = self.bs;
        let l = match self.map.get_mut(&req) {
            Some(l) => l,
            None => return Push::Exhausted,
        };
        if (l.len + 1).div_ceil(bs) > l.blocks.len() {
            match pool.alloc() {
                Some(b) => {
                    l.blocks.push(b);
                    l.len += 1;
                    Push::OkAlloc(b)
                }
                None => Push::Exhausted, // 布局未变更——调用方触发强制抢占
            }
        } else {
            l.len += 1;
            Push::Ok
        }
    }

    /// 摘除并返回布局（块的池操作由 MemState 语义化地完成）
    pub fn remove(&mut self, req: ReqId) -> Option<ReqLayout> {
        self.map.remove(&req)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
}
