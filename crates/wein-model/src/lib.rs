//! wein-model：I1 竖切的加载侧（纯 CPU）。
//!
//! - config：Qwen3.5 文本骨干配置（config.json 嵌套 text_config）
//! - weights：safetensors 手写解析（8 字节 LE 头长 + JSON 元数据 + 裸数据），
//!   多 shard 路由按官方 index；mmap 零整拷
//! - tokenizer：tokenizers crate（HF tokenizer.json）接线
//!
//! 依赖阶梯：本 crate 不依赖 workspace 内任何 crate、不依赖 GPU——
//! 设备上载是 wein-exec 的事。

mod config;
mod template;
mod tokenizer;
mod weights;

pub use config::{LayerType, Qwen35Config};
pub use template::{normalize_tool_calls, ChatMsg, ChatTemplate};
pub use tokenizer::HfTokenizer;
pub use weights::{Dtype, TensorRef, WeightStore};

pub type Result<T> = std::result::Result<T, ModelError>;

#[derive(Debug)]
pub enum ModelError {
    Io(std::io::Error),
    Json(String),
    /// 张量名不在目录里
    Missing(String),
    /// dtype 与期望不符
    Dtype { name: String, want: Dtype, got: Dtype },
    /// 形状与期望不符
    Shape { name: String, want: Vec<u32>, got: Vec<u32> },
    /// 数据未按元素对齐（无法安全转切片）
    Unaligned(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::Io(e) => write!(f, "io: {e}"),
            ModelError::Json(e) => write!(f, "json: {e}"),
            ModelError::Missing(n) => write!(f, "缺张量：{n}"),
            ModelError::Dtype { name, want, got } => write!(f, "{name} dtype {got:?} ≠ {want:?}"),
            ModelError::Shape { name, want, got } => write!(f, "{name} 形状 {got:?} ≠ {want:?}"),
            ModelError::Unaligned(n) => write!(f, "{n} 数据未按元素对齐"),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<std::io::Error> for ModelError {
    fn from(e: std::io::Error) -> Self {
        ModelError::Io(e)
    }
}

/// bf16（u16 位形）→ f32：尾数左移 16 位。与 kernel 侧同式
#[inline]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// f32 → bf16（RNE 半偶）：与 kernel 侧 f32_to_bf16 同式
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    let rounded = b + 0x7FFF + ((b >> 16) & 1);
    (rounded >> 16) as u16
}

/// 模型目录解析：`WEIN_MODEL_DIR` 优先，默认 `~/.cache/wein/Qwen3.5-4B`。
/// 目录必须存在（config.json 在场）才算 Some——调用方据此门控测试
pub fn model_dir() -> Option<std::path::PathBuf> {
    if let Some(d) = std::env::var_os("WEIN_MODEL_DIR") {
        let d = std::path::PathBuf::from(d);
        return d.join("config.json").is_file().then_some(d);
    }
    let home = std::env::var_os("HOME")?;
    let d = std::path::PathBuf::from(home).join(".cache/wein/Qwen3.5-4B");
    d.join("config.json").is_file().then_some(d)
}
