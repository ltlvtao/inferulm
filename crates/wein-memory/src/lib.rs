//! wein-memory：KV 内存管理——布局规划 / BlockPool / radix 索引 / req_to_token / MemState。
//!
//! 02-architecture §5 内存层的 M0 落地：
//! - 布局规划器（`layout`）：KvSpec × 平台 block_size → 字节布局，纯数学
//! - `pool`：块级物理管理（free-list + 引用计数 + DeferredRelease + 水位）
//! - `radix`：token 前缀 → 块的索引（pin 免驱逐、LRU 驱逐；索引与数据分离——
//!   节点只持 BlockId，不持数据）
//! - `r2t`：req_to_token 的宿主镜像（唯一事实源在设备侧 flat u32 表，kernel 时代接入）
//! - `state`：MemState——把上面四件组装成容量模型语义（准入/增长/完成/抢占/取消）
//!
//! 零依赖：不依赖 wein-sched（同层禁止横向依赖，02 §2）——CapacityModel 的
//! 适配由 wein-engine 侧的适配器承担。M0 简化：radix 无节点分裂（整链插入）、
//! 驱逐只摘未 pin 叶子。

pub mod layout;
pub mod pool;
pub mod r2t;
pub mod radix;
pub mod state;

pub type ReqId = u64;
