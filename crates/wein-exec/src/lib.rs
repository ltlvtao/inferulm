//! wein-exec：exec 层第一个真身（02-architecture §5 第三线程的 M0 形态）。
//!
//! - [`device::DeviceSession`]：探测→画像→`OpTable::resolve`→流，一次装配
//!   全程只读（§4 启动时解析）
//! - [`toy::ToyExecutor`]：单块 toy 模型 + 真算子装配（M0 清单全走真路径，
//!   GEMM 走 cuBLASLt）+ BlockPool 分页 KV + host 贪心；实现 engine 的
//!   [`wein_engine::Executor`]
//! - [`reference`]：独立 CPU 镜像（数值诚实验收 x2 的对照端）
//!
//! M0 边界：贪心采样在 host（采样位置的开放问题 02 §13，倾向 exec 图尾）；
//! 图捕获 NONE 档起步（W2）。
pub mod ctx;
pub mod dram;
pub mod device;
pub mod qwen;
pub mod qwen_ref;
pub mod reference;
pub mod toy;

pub use device::DeviceSession;
pub use ctx::CtxConfig;
pub use qwen::{kv_block, kv_pages, MtpItem, Qwen35Executor, SpecRound};
pub use qwen_ref::QwenMirror;
pub use toy::{ToyConfig, ToyExecutor, ToyWeights};

/// 贪心 argmax：严格大于才替换（并列取最小下标）——设备/CPU 两侧同一规则
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    best as u32
}
