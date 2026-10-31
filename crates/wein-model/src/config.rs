//! Qwen3.5 文本骨干配置。字段名对齐 transformers `Qwen3_5TextConfig`
//!（config.json 嵌套在 text_config 下；视觉/MTP 不在文本路径必答题里）。

use crate::{ModelError, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// gated deltanet（线性注意力）
    Linear,
    /// GQA 全注意力
    Full,
}

#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub hidden_size: u32,
    pub num_layers: u32,
    pub layer_types: Vec<LayerType>,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub inter_size: u32,
    pub vocab_size: u32,
    pub rms_eps: f32,
    pub rope_theta: f64,
    /// partial rotary：rot_dim = head_dim × factor
    pub partial_rotary_factor: f32,
    pub tie_word_embeddings: bool,
    pub eos_token_id: u32,
    // linear attention 侧
    pub linear_num_v_heads: u32,
    pub linear_num_k_heads: u32,
    pub linear_k_dim: u32,
    pub linear_v_dim: u32,
    pub conv_kernel: u32,
    /// MTP（多 token 预测）草稿层数：`text_config.mtp_num_hidden_layers`
    /// （0 = checkpoint 无 MTP；文本骨干前向不依赖它）
    pub mtp_num_hidden_layers: u32,
    /// MTP 是否另有一套词表 embedding：`mtp_use_dedicated_embeddings`
    /// （False = 与主模型 tie，草稿层直用 embed 表）
    pub mtp_use_dedicated_embeddings: bool,
}

#[derive(Deserialize)]
struct TopConfig {
    text_config: TextConfigRaw,
}

#[derive(Deserialize)]
struct TextConfigRaw {
    hidden_size: u32,
    num_hidden_layers: u32,
    layer_types: Vec<String>,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    intermediate_size: u32,
    vocab_size: u32,
    rms_norm_eps: f32,
    #[serde(default = "d_true")]
    tie_word_embeddings: bool,
    eos_token_id: u32,
    linear_num_value_heads: u32,
    linear_num_key_heads: u32,
    linear_key_head_dim: u32,
    linear_value_head_dim: u32,
    linear_conv_kernel_dim: u32,
    #[serde(default)]
    mtp_num_hidden_layers: u32,
    /// MTP 专用 embedding 形态（True 时 mtp.embed_tokens 另有一套词表权重）
    #[serde(default)]
    mtp_use_dedicated_embeddings: bool,
    rope_parameters: RopeParams,
}

#[derive(Deserialize)]
struct RopeParams {
    rope_theta: f64,
    #[serde(default = "d_one")]
    partial_rotary_factor: f32,
}

fn d_true() -> bool {
    true
}
fn d_one() -> f32 {
    1.0
}

impl Qwen35Config {
    pub fn load(dir: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(dir.join("config.json"))?;
        let top: TopConfig = serde_json::from_str(&raw)
            .map_err(|e| ModelError::Json(format!("config.json: {e}")))?;
        let t = top.text_config;
        let layer_types = t
            .layer_types
            .iter()
            .map(|s| match s.as_str() {
                "linear_attention" => Ok(LayerType::Linear),
                "full_attention" => Ok(LayerType::Full),
                other => Err(ModelError::Json(format!("未知层型：{other}"))),
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Qwen35Config {
            hidden_size: t.hidden_size,
            num_layers: t.num_hidden_layers,
            layer_types,
            num_heads: t.num_attention_heads,
            num_kv_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            inter_size: t.intermediate_size,
            vocab_size: t.vocab_size,
            rms_eps: t.rms_norm_eps,
            rope_theta: t.rope_parameters.rope_theta,
            partial_rotary_factor: t.rope_parameters.partial_rotary_factor,
            tie_word_embeddings: t.tie_word_embeddings,
            eos_token_id: t.eos_token_id,
            linear_num_v_heads: t.linear_num_value_heads,
            linear_num_k_heads: t.linear_num_key_heads,
            linear_k_dim: t.linear_key_head_dim,
            linear_v_dim: t.linear_value_head_dim,
            conv_kernel: t.linear_conv_kernel_dim,
            mtp_num_hidden_layers: t.mtp_num_hidden_layers,
            mtp_use_dedicated_embeddings: t.mtp_use_dedicated_embeddings,
        })
    }

    pub fn layer_type(&self, i: u32) -> LayerType {
        self.layer_types[i as usize]
    }

    /// rotary 维度（rotate_half 布局，作用在每头前 rot_dim 维）
    pub fn rot_dim(&self) -> u32 {
        ((self.head_dim as f32 * self.partial_rotary_factor) as u32) & !1
    }

    /// GQA 组大小：每 kv 头对应 q 头数
    pub fn kv_group(&self) -> u32 {
        self.num_heads / self.num_kv_heads
    }

    /// linear 层 k 通道总数（in_proj_qkv 前 key_dim）
    pub fn linear_key_dim(&self) -> u32 {
        self.linear_k_dim * self.linear_num_k_heads
    }

    /// linear 层 v 通道总数
    pub fn linear_value_dim(&self) -> u32 {
        self.linear_v_dim * self.linear_num_v_heads
    }

    /// conv1d 通道数 = 2×key_dim + value_dim
    pub fn conv_dim(&self) -> u32 {
        2 * self.linear_key_dim() + self.linear_value_dim()
    }

    /// 形状自洽——配置读错字段/结构变了在这里报
    pub fn validate(&self) -> Result<()> {
        if self.layer_types.len() != self.num_layers as usize {
            return Err(ModelError::Json(format!(
                "layer_types {} ≠ num_layers {}",
                self.layer_types.len(),
                self.num_layers
            )));
        }
        if self.num_heads % self.num_kv_heads != 0 {
            return Err(ModelError::Json("q 头不是 kv 头整数倍".into()));
        }
        if self.linear_num_v_heads % self.linear_num_k_heads != 0 {
            return Err(ModelError::Json("v 头不是 k 头整数倍".into()));
        }
        if self.rot_dim() == 0 || self.rot_dim() > self.head_dim {
            return Err(ModelError::Json(format!("rot_dim {} 越界", self.rot_dim())));
        }
        if self.head_dim % 2 != 0 || self.linear_k_dim % 2 != 0 {
            return Err(ModelError::Json("头维非偶数（rope/l2norm 需要）".into()));
        }
        // MTP 草稿层：本刀只落单层（多头嵌套 + 层间 hidden 传递另刀）
        if self.mtp_num_hidden_layers > 1 {
            return Err(ModelError::Json(format!(
                "mtp>1 层暂不支持（mtp_num_hidden_layers={}）",
                self.mtp_num_hidden_layers
            )));
        }
        if self.mtp_num_hidden_layers > 0 && self.mtp_use_dedicated_embeddings {
            return Err(ModelError::Json(
                "mtp_use_dedicated_embeddings=true 暂不支持（草稿层复用 tie 的 embed 表）".into(),
            ));
        }
        Ok(())
    }
}
