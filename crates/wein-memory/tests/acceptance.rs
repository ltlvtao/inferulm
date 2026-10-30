//! wein-memory 验收套件（m1-m10）——02-architecture §5 内存层的对外承诺。
//!
//! 分层：
//! - m1     布局规划：KvSpec → 字节布局（纯数学，跨平台的第一块基石）
//! - m2-m4  BlockPool：耗尽/引用计数/DeferredRelease/水位
//! - m5-m6  radix 索引：块对齐匹配、pin 免驱逐、LRU 顺序
//! - m7     req_to_token：块边界处的分配时机
//! - m8-m10 MemState：准入（悲观+驱逐）/前缀命中/完成入缓存/抢占保留序列
//!
//! 与调度层的接线（CapacityModel 适配）不在此测——那是 wein-engine 的 im 套件，
//! 依赖规则禁止 wein-memory 依赖 wein-sched。

// ---------- m1 布局规划 ----------

mod m1_layout {
    use wein_memory::layout::{plan_kv, DType, KvSpec};

    /// m1 · 每 token KV 字节数 = 2(K和V) × layers × kv_heads × head_dim × dtype 字宽；
    ///      block 字节数随之推出（block_size 是平台参数：CUDA 16/32、昇腾 128）
    #[test]
    fn bytes_per_token_and_block() {
        let spec = KvSpec { num_layers: 28, kv_heads: 8, head_dim: 128, dtype: DType::Bf16 };
        let kv = plan_kv(&spec, 16);
        // 2 × 28 × 8 × 128 × 2B = 114688 B/token
        assert_eq!(kv.bytes_per_token, 114688);
        assert_eq!(kv.block_size, 16);
        assert_eq!(kv.bytes_per_block, 114688 * 16);

        // W8A8/FP8 路径：dtype 减半字宽 → 字节减半（M0 量化承诺的账要在这里算清）
        let spec8 = KvSpec { dtype: DType::Fp8E4m3, ..spec };
        assert_eq!(plan_kv(&spec8, 16).bytes_per_token, 57344);
        assert_eq!(plan_kv(&spec8, 16).bytes_per_token * 2, kv.bytes_per_token);
    }

    /// m1b · 昇腾形态：block_size 128 平台参数直接生效
    #[test]
    fn ascend_block_size_platform_param() {
        let spec = KvSpec { num_layers: 32, kv_heads: 4, head_dim: 128, dtype: DType::Bf16 };
        let kv = plan_kv(&spec, 128);
        assert_eq!(kv.block_size, 128);
        assert_eq!(kv.bytes_per_block, kv.bytes_per_token * 128);
    }
}

// ---------- m2-m4 BlockPool ----------

mod m2_pool {
    use wein_memory::pool::BlockPool;

    /// m2 · 分配耗尽：N 块池第 N+1 次 alloc 返回 None（不是 panic——上层要做准入控制）
    #[test]
    fn exhaustion_returns_none() {
        let mut pool = BlockPool::new(8, 0.1);
        let ids: Vec<_> = (0..8).map(|_| pool.alloc().expect("前 8 次必成功")).collect();
        assert_eq!(ids.len(), 8, "8 块池分配 8 次");
        assert!(pool.alloc().is_none(), "第 9 次必须 None");
        assert_eq!(pool.free_now(), 0);
    }
}

mod m3_pool {
    use wein_memory::pool::BlockPool;

    /// m3 · 引用计数 + DeferredRelease：共享块 rc 归零不立即回空闲表——
    ///      入 deferred，步末 recycle 才可用（结构性变更跨步生效）
    #[test]
    fn refcount_and_deferred_release() {
        let mut pool = BlockPool::new(4, 0.1);
        let b = pool.alloc().unwrap();
        assert_eq!(pool.rc_of(b), 1);

        pool.retain(b); // radix 命中共享：cache 持有 + 请求持有
        assert_eq!(pool.rc_of(b), 2);

        pool.release(b); // 请求放手，cache 仍在
        assert!(pool.owned(b), "rc>0 仍在占用");
        assert_eq!(pool.deferred_len(), 0, "未归零不进 deferred");

        pool.release(b); // cache 也放手 → rc 0
        assert_eq!(pool.rc_of(b), 0);
        assert_eq!(pool.deferred_len(), 1, "rc 归零进 deferred，不立即回 free");
        assert_eq!(pool.free_now(), 3, "recycle 之前 alloc 看不到它");

        let recycled = pool.recycle(); // 步末回收点
        assert_eq!(recycled, 1);
        assert_eq!(pool.deferred_len(), 0);
        assert_eq!(pool.free_now(), 4);
        assert_eq!(pool.alloc(), Some(b), "回收后同一块可再分配");
    }
}

mod m4_pool {
    use wein_memory::pool::BlockPool;

    /// m4 · 水位：free_now 跌破低水位即报压；deferred 不解压（悲观），
    ///      recycle 后才解除
    #[test]
    fn watermark_pressure_and_deferred_pessimism() {
        let mut pool = BlockPool::new(10, 0.2); // 低水位 = max(1, 2) = 2
        assert!(!pool.over_watermark());

        for _ in 0..7 {
            pool.alloc().unwrap();
        }
        assert_eq!(pool.free_now(), 3);
        assert!(!pool.over_watermark(), "3 > 2 未到水位");

        pool.alloc().unwrap();
        assert!(pool.over_watermark(), "free=2 ≤ 水位 2");

        // 释放一个归零块 → deferred；free_now 不变 → 压力仍在（悲观：
        // 跨步在途的块不当作可用，直到 step_end 回收）
        let b = pool.alloc().unwrap(); // free=1
        pool.release(b);
        assert_eq!(pool.deferred_len(), 1);
        assert!(pool.over_watermark(), "deferred 不解压");

        pool.recycle();
        assert_eq!(pool.free_now(), 2);
        assert!(pool.over_watermark(), "free=2 仍 ≤ 水位（回收不改变水位判定）");
    }
}

// ---------- m5-m6 radix 索引 ----------

mod m5_radix {
    use wein_memory::pool::BlockPool;
    use wein_memory::radix::RadixCache;

    fn bs2_pool(n: u32) -> BlockPool {
        BlockPool::new(n, 0.1)
    }

    /// m5 · 匹配按块对齐：只命中整块；部分尾不算；命中块被 pin（驱逐免疫）+ rc 共享
    #[test]
    fn match_is_block_aligned_and_pins() {
        let mut cache = RadixCache::new();
        let mut pool = bs2_pool(16);

        // 先经 pool 分配 3 块，插入 [1,2,3,4,5,6]
        let blocks: Vec<_> = (0..3).map(|_| pool.alloc().unwrap()).collect();
        cache.insert(&[1, 2, 3, 4, 5, 6], &blocks);

        // 3 token 查询：只匹配 1 整块（[1,2]），第 2 块不完整不算；
        // 命中即 pin：rc+1（cache + 匹配者各持一份）
        let m = cache.match_prefix(&mut pool, &[1, 2, 3]);
        assert_eq!(m.tokens, 2, "部分块不命中");
        assert_eq!(m.blocks.len(), 1);
        for &b in &m.blocks {
            assert_eq!(pool.rc_of(b), 2, "命中即 pin");
        }
        cache.unpin(&mut pool, &[1, 2, 3], m.blocks.len()); // 还原，不污染后续断言

        // 精确 2 块
        let m = cache.match_prefix(&mut pool, &[1, 2, 3, 4]);
        assert_eq!(m.tokens, 4);
        assert_eq!(m.blocks.len(), 2);
        cache.unpin(&mut pool, &[1, 2, 3, 4], m.blocks.len());

        // 前缀不匹配 → 0（无 pin 副作用）
        let m = cache.match_prefix(&mut pool, &[9, 9, 9, 9]);
        assert_eq!(m.tokens, 0);
        assert!(m.blocks.is_empty());
    }

    /// m5b · 路径分叉：两个序列共享前缀，各自尾巴独立
    #[test]
    fn shared_prefix_forks() {
        let mut cache = RadixCache::new();
        let mut pool = bs2_pool(16);
        let a: Vec<_> = (0..2).map(|_| pool.alloc().unwrap()).collect();
        let b: Vec<_> = (0..2).map(|_| pool.alloc().unwrap()).collect();
        cache.insert(&[1, 2, 3, 4], &a);
        cache.insert(&[1, 2, 9, 9], &b);

        assert_eq!(cache.match_prefix(&mut pool, &[1, 2, 9, 9]).blocks.len(), 2);
        assert_eq!(cache.match_prefix(&mut pool, &[1, 2, 3, 4]).blocks.len(), 2);
        assert_eq!(cache.match_prefix(&mut pool, &[1, 2, 8, 8]).tokens, 2, "分叉后只共享 [1,2]");
    }

    /// m5c · 同首 token 不同块并存：子节点按整块内容索引。首 token 单键会
    /// 让后插的链把先插的覆盖掉（另一条链的块泄漏且永不命中）——chat 模板
    /// 让所有请求的首块都同首 token，这个坑在真实流量下必踩
    #[test]
    fn same_first_token_blocks_coexist() {
        let mut cache = RadixCache::new();
        let mut pool = bs2_pool(16);
        let a: Vec<_> = (0..2).map(|_| pool.alloc().unwrap()).collect();
        let b: Vec<_> = (0..2).map(|_| pool.alloc().unwrap()).collect();
        cache.insert(&[7, 1, 2, 3], &a); // 首块 [7,1]，次块 [2,3]
        cache.insert(&[7, 9, 8, 6], &b); // 首块 [7,9]——与 a 只共享首 token

        assert_eq!(cache.match_prefix(&mut pool, &[7, 1, 2, 3]).blocks.len(), 2, "先插的链仍完整命中");
        assert_eq!(cache.match_prefix(&mut pool, &[7, 9, 8, 6]).blocks.len(), 2, "后插的链独立命中");
        assert_eq!(cache.match_prefix(&mut pool, &[7, 1, 9, 9]).tokens, 2, "同首块派生 = 共享 1 块");
    }
}

mod m6_radix {
    use wein_memory::pool::BlockPool;
    use wein_memory::radix::RadixCache;

    /// m6 · LRU 驱逐：只驱逐无 pin 的叶子；命中刷新热度；
    ///      驱逐的块走 deferred（MemState 在准入路径显式 recycle）
    #[test]
    fn evict_lru_leaf_first_pinned_immune() {
        let mut cache = RadixCache::new();
        let mut pool = BlockPool::new(16, 0.1);

        let a: Vec<_> = (0..2).map(|_| pool.alloc().unwrap()).collect();
        let b: Vec<_> = (0..2).map(|_| pool.alloc().unwrap()).collect();
        cache.insert(&[1, 1, 1, 1], &a); // 先插入 = 旧
        cache.insert(&[2, 2, 2, 2], &b);

        // 触碰 A（match 后立即 unpin = 纯 LRU touch）：B 变最旧
        let m = cache.match_prefix(&mut pool, &[1, 1, 1, 1]);
        cache.unpin(&mut pool, &[1, 1, 1, 1], m.blocks.len());

        // LRU：B（最旧未 pin）被逐出，A 存活
        let freed = cache.evict(&mut pool, 2);
        assert_eq!(freed, 2, "B 被逐出");
        assert_eq!(cache.match_prefix(&mut pool, &[2, 2, 2, 2]).tokens, 0, "B 已逐出");
        let alive = cache.match_prefix(&mut pool, &[1, 1, 1, 1]);
        assert_eq!(alive.tokens, 4, "A 存活");
        cache.unpin(&mut pool, &[1, 1, 1, 1], alive.blocks.len());
        assert_eq!(pool.deferred_len(), 2, "驱逐进 deferred");

        // pin 免疫：A 被运行中请求持有时，驱逐失败也不误伤
        let pinned = cache.match_prefix(&mut pool, &[1, 1, 1, 1]);
        assert_eq!(pinned.blocks.len(), 2);
        assert_eq!(cache.evict(&mut pool, 2), 0, "唯一叶子被 pin 时驱逐失败");

        // 解 pin：rc 归 cache（1），此后可逐
        cache.unpin(&mut pool, &[1, 1, 1, 1], 2);
        for &blk in &pinned.blocks {
            assert_eq!(pool.rc_of(blk), 1, "解 pin 后仍由 cache 持有");
        }
        assert_eq!(cache.evict(&mut pool, 2), 2, "解 pin 后可驱逐");
    }
}

// ---------- m7 req_to_token ----------

mod m7_r2t {
    use wein_memory::pool::BlockPool;
    use wein_memory::r2t::{Push, ReqToToken};

    /// m7 · 块边界处的分配时机：push 跨过 ceil 边界才 alloc；remove 返还布局
    #[test]
    fn push_allocates_only_at_boundary() {
        let mut pool = BlockPool::new(8, 0.1);
        let mut r2t = ReqToToken::new(2);

        // 命中 1 块 + 私有 2 块，len=5（第 3 块半满：ceil(5/2)=3）
        let hit: Vec<_> = (0..1).map(|_| pool.alloc().unwrap()).collect();
        let priv_blocks: Vec<_> = (0..2).map(|_| pool.alloc().unwrap()).collect();
        let mut blocks = hit;
        blocks.extend(priv_blocks);
        r2t.bind(7, blocks, 1, 5);
        assert_eq!(pool.free_now(), 5);

        assert!(matches!(r2t.push(7, &mut pool), Push::Ok), "5→6：填满第 3 块，无需新块");
        assert_eq!(pool.free_now(), 5, "未跨界不分配");

        assert!(matches!(r2t.push(7, &mut pool), Push::OkAlloc(_)), "6→7：ceil(7/2)=4 → 分配第 4 块");
        assert_eq!(pool.free_now(), 4);

        let layout = r2t.remove(7).expect("已绑定");
        assert_eq!(layout.len, 7);
        assert_eq!(layout.blocks.len(), 4);
        assert_eq!(layout.hit_len_blocks, 1, "命中块标记跨生命周期保留");

        // 耗尽：push 在无块可分时返回 false（调用方触发强制抢占）
        r2t.bind(8, vec![pool.alloc().unwrap()], 0, 2); // free=3
        for i in 0..6 {
            assert!(
                matches!(r2t.push(8, &mut pool), Push::Ok | Push::OkAlloc(_)),
                "第 {i} 次仍有块可用"
            );
        }
        assert!(matches!(r2t.push(8, &mut pool), Push::Exhausted), "池耗尽必须显式报告而非 panic");
    }
}

// ---------- m8-m10 MemState ----------

mod m8_state {
    use wein_memory::state::{MemConfig, MemState};

    fn cfg(blocks: u32) -> MemConfig {
        MemConfig { block_size: 2, num_blocks: blocks, low_watermark: 0.1 }
    }

    /// m8 · 完成入缓存 → 后续同前缀准入命中：cache_hit_tokens 记账，私有块分配减少
    #[test]
    fn finish_inserts_cache_and_next_admission_hits() {
        let mut mem = MemState::new(cfg(16));
        assert!(mem.admit(1, &[1, 2, 3, 4], 0));
        assert_eq!(mem.stats().blocks_allocated, 2, "4 token / bs 2 → 2 块");

        for t in [5u32, 6, 7, 8] {
            assert!(mem.on_token(1, t));
        }
        assert_eq!(mem.stats().blocks_allocated, 4, "生成 4 token → 追加 2 块");

        mem.finish(1);
        // 完整序列 [1..8]（4 整块）入 radix；无尾块

        // 第二个请求共享 4 token 前缀：命中 2 块，只补 1 块私有
        assert!(mem.admit(2, &[1, 2, 3, 4, 9, 9], 0));
        assert_eq!(mem.stats().cache_hit_tokens, 4, "前缀命中按 token 记账");
        assert_eq!(mem.stats().blocks_allocated, 5, "总分配 4 + 1（命中省 2 块）");
    }
}

mod m9_state {
    use wein_memory::state::{MemConfig, MemState};

    fn cfg(blocks: u32) -> MemConfig {
        MemConfig { block_size: 2, num_blocks: blocks, low_watermark: 0.1 }
    }

    /// m9 · 抢占保留序列（keep_seq）：释放块但保留 token 序列；
    ///      重准入时以完整上下文（prompt+已生成）匹配 radix → 廉价重算；
    ///      取消丢弃序列
    #[test]
    fn preempt_keeps_seq_readmit_hits_full_context() {
        let mut mem = MemState::new(cfg(16));

        // 先让 A 跑完 [1,2,3,4] + 生成 [5,5,5,5]，全部入缓存
        assert!(mem.admit(1, &[1, 2, 3, 4], 0));
        for t in [5u32, 5, 5, 5] {
            assert!(mem.on_token(1, t));
        }
        mem.finish(1);

        // B 同前缀 + 同生成内容，跑到一半被抢占
        assert!(mem.admit(2, &[1, 2, 3, 4], 0));
        assert!(mem.on_token(2, 5));
        assert!(mem.on_token(2, 5));
        mem.release(2, true); // 抢占：块释放（deferred），序列保留

        assert_eq!(mem.stats().deferred_releases, 1, "仅 1 块私有块释放；2 块命中块解 pin 后仍由缓存持有");
        mem.step_end(); // 回收

        // 重准入：完整上下文 [1,2,3,4,5,5] 在 radix 全命中 → 零私有块
        let hit_before = mem.stats().cache_hit_tokens;
        assert!(mem.admit(2, &[1, 2, 3, 4], 2));
        assert_eq!(mem.stats().cache_hit_tokens - hit_before, 6, "重准入按完整上下文命中");
        let alloc_before = mem.stats().blocks_allocated;
        assert!(mem.on_token(2, 5)); // 继续生成
        assert_eq!(mem.stats().blocks_allocated - alloc_before, 1, "只需 1 块跨界块");

        // 取消：序列丢弃（引擎不会再重准入，但状态必须干净）
        mem.release(2, false);
        assert!(mem.admit(3, &[1, 2, 3, 4], 0), "取消后同前缀仍可正常准入");
    }
}

mod m10_state {
    use wein_memory::state::{MemConfig, MemState};

    /// m10 · 悲观准入：free_now + 可驱逐 ≥ 需求 + 低水位余量（deferred 不算——
    ///      跨步在途不可用；余量防准入→抢占乒乓）。可驱逐 = 无 pin 依赖子树全体
    ///      （与级联驱逐一致）；准入路径驱逐 LRU 凑块；水位判定用 free_soon（含在途）
    #[test]
    fn can_admit_pessimistic_and_evict_on_admit() {
        let mut mem = MemState::new(MemConfig {
            block_size: 1,
            num_blocks: 8,
            low_watermark: 0.25, // 水位 2
        });

        // 缓存 6 块（两次完成：[1,1,1] 与 [2,2,2]，bs=1）
        assert!(mem.admit(1, &[1, 1, 1], 0));
        mem.finish(1);
        assert!(mem.admit(2, &[2, 2, 2], 0));
        mem.finish(2);
        assert_eq!(mem.free_blocks(), 2, "6 块归缓存");

        // 需 4 块：free 2 不够，但 free + evictable(6) = 8 ≥ 4 + 水位2 → 可准入，
        // 且准入即驱逐凑块
        assert!(mem.can_admit(4));
        assert!(mem.admit(3, &[9, 9, 9, 9], 0), "驱逐 LRU 后准入成功");
        assert!(mem.stats().evicted_blocks >= 2);
        assert!(mem.stats().blocks_allocated > 0);

        // deferred 排除：在途释放的块不参与悲观判定
        mem.release(3, false); // 3 的块全部进 deferred
        assert!(mem.stats().deferred_releases > 0);
        let free = mem.free_blocks();
        let evictable = 8usize - free - mem.stats().deferred_releases as usize;
        assert!(
            !mem.can_admit(((free + evictable + 1) as usize).max(1)),
            "deferred 不算可用：需求超过 free+evictable 必须拒绝"
        );
        mem.step_end();
        assert!(mem.free_blocks() > free, "回收后在途块回到可用");
    }

    /// m10b · over_pressure 用 free_soon（含 deferred）：在途释放可立即缓解压力
    ///        （抢占循环在同一 step 内多次判定，被抢占者的块虽在 deferred 也要算数）
    #[test]
    fn over_pressure_counts_inflight() {
        let mut mem = MemState::new(MemConfig {
            block_size: 1,
            num_blocks: 6,
            low_watermark: 0.25, // 水位 1
        });
        // 占到 free=1 → 压力
        assert!(mem.admit(1, &[1, 1, 1, 1, 1], 0));
        assert_eq!(mem.free_blocks(), 1);
        assert!(mem.over_pressure());

        // 释放到 deferred（未回收）：free_soon=6 > 1 → 压力解除
        mem.release(1, false);
        assert_eq!(mem.stats().deferred_releases, 5);
        assert!(!mem.over_pressure(), "在途块算入 free_soon");
    }
}

// ---------- m11 前缀缓存部分退回与块列表驱逐（radix 快照接线底座） ----------

mod m11_radix {
    use wein_memory::pool::BlockPool;
    use wein_memory::radix::RadixCache;

    fn bs2_pool() -> BlockPool {
        BlockPool::new(16, 0.1)
    }

    /// m11 · unpin_range：match 全量 pin 后退回尾段 [from,to)——
    /// 命中截断到有快照槽祖先的支撑语义。头部 pin 保持、尾部 rc 回落、
    /// 树形不变（unpin 不是驱逐）。
    #[test]
    fn unpin_range_releases_tail_keeps_head() {
        let mut cache = RadixCache::new();
        let mut pool = bs2_pool();
        let blocks: Vec<_> = (0..3).map(|_| pool.alloc().unwrap()).collect();
        cache.insert(&[1, 2, 3, 4, 5, 6], &blocks);

        let m = cache.match_prefix(&mut pool, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(m.blocks.len(), 3);
        // 退回尾部 2 块（命中截断到块 0）
        cache.unpin_range(&mut pool, &[1, 2, 3, 4, 5, 6], 1, 3);
        assert_eq!(pool.rc_of(m.blocks[0]), 2, "头部 pin 保持（cache+请求）");
        for &b in &m.blocks[1..] {
            assert_eq!(pool.rc_of(b), 1, "尾部退回 cache 份");
        }
        // 头部 unpin 后树完整（三块都可再命中——未被驱逐）
        cache.unpin(&mut pool, &[1, 2, 3, 4, 5, 6], 1);
        let again = cache.match_prefix(&mut pool, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(again.blocks.len(), 3, "unpin_range 不动树形");
        cache.unpin(&mut pool, &[1, 2, 3, 4, 5, 6], 3);
        pool.recycle();
    }

    /// m11b · evict_blocks：返回被逐块列表（快照槽挂块的联动钩子）——
    /// 数量与 evict 一致、id 唯一、被逐块确已不在树上。
    #[test]
    fn evict_blocks_returns_list() {
        let mut cache = RadixCache::new();
        let mut pool = bs2_pool();
        let a: Vec<_> = (0..2).map(|_| pool.alloc().unwrap()).collect();
        let b: Vec<_> = (0..2).map(|_| pool.alloc().unwrap()).collect();
        cache.insert(&[1, 1, 1, 1], &a);
        cache.insert(&[2, 2, 2, 2], &b);

        let freed = cache.evict_blocks(&mut pool, 1);
        assert_eq!(freed.len(), 1, "驱逐 1 块");
        assert_eq!(cache.evict(&mut pool, 1), 1, "计数口径与列表一致");
        assert_eq!(cache.match_prefix(&mut pool, &[1, 1, 1, 1]).tokens, 0, "A 已逐出");
        let alive = cache.match_prefix(&mut pool, &[2, 2, 2, 2]);
        assert_eq!(alive.tokens, 4, "B 存活（LRU 后插）");
        cache.unpin(&mut pool, &[2, 2, 2, 2], alive.blocks.len());
    }
}
