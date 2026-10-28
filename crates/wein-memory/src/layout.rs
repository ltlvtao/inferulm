//! KV 布局规划：KvSpec（模型侧）× 平台参数 block_size → 字节布局。
//! 纯数学，无状态——昇腾 128 / CUDA 16|32（02 §13-3 待 W2 反推）都从这里出账。
//!
//! 每 token 字节数 = 2(K 与 V 两份) × layers × kv_heads × head_dim × dtype 字宽。
//! W8A8/Fp8 路径只是 dtype 字宽减半——量化收益的账在同一公式里算清（m1 钉死）。

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    Bf16,
    Fp16,
    Fp8E4m3,
    Int8,
}

impl DType {
    pub fn bytes(self) -> u64 {
        match self {
            DType::Bf16 | DType::Fp16 => 2,
            DType::Fp8E4m3 | DType::Int8 => 1,
        }
    }
}

/// 模型侧 KV 形状（与权重一起来自 checkpoint 元数据）
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvSpec {
    pub num_layers: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub dtype: DType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvLayout {
    pub bytes_per_token: u64,
    pub block_size: u32,
    pub bytes_per_block: u64,
}

pub fn plan_kv(spec: &KvSpec, block_size: u32) -> KvLayout {
    let bytes_per_token = 2
        * spec.num_layers as u64
        * spec.kv_heads as u64
        * spec.head_dim as u64
        * spec.dtype.bytes();
    KvLayout {
        bytes_per_token,
        block_size,
        bytes_per_block: bytes_per_token * block_size as u64,
    }
}
