//! tokenizers crate 接线（HF tokenizer.json，Qwen BPE byte-level）。

use crate::{ModelError, Result};
use std::path::Path;

pub struct HfTokenizer {
    tok: tokenizers::Tokenizer,
    eos_id: u32,
}

impl HfTokenizer {
    pub fn open(dir: &Path) -> Result<Self> {
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| ModelError::Json(format!("tokenizer.json: {e}")))?;
        // eos 从 tokenizer_config（<|im_end|>）；拿不到再退 vocab 查名
        let eos = tok
            .token_to_id("<|im_end|>")
            .ok_or_else(|| ModelError::Json("词表里没有 <|im_end|>".into()))?;
        Ok(HfTokenizer { tok, eos_id: eos as u32 })
    }

    /// 文本 → token id（不加特殊包装：Qwen 无 BOS，chat 模板是 serve 层的事）
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let enc = self
            .tok
            .encode(text, false)
            .unwrap_or_else(|e| panic!("encode 失败：{e}"));
        enc.get_ids().to_vec()
    }

    /// id → 文本（保留特殊 token 字面量——增量 detok 在 serve 层做前缀管理）
    pub fn decode(&self, ids: &[u32]) -> String {
        self.tok
            .decode(ids, false)
            .unwrap_or_else(|e| panic!("decode 失败：{e}"))
    }

    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }
}
