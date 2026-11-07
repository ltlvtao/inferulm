//! tokenizer 边界与增量解码。
//!
//! M0 只有 MockTokenizer（可注入任意字节片段，用于构造跨 token 的 UTF-8 边界）；
//! 真实实现包 tokenizers crate（W5），实现同一 trait。
use std::borrow::Cow;
use std::sync::Arc;

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Vec<u32>;
    /// 单个 token 的原始字节（byte-level，可能是不完整 UTF-8——正是本模块要处理的核心问题）
    fn piece_bytes(&self, id: u32) -> Cow<'_, [u8]>;
    /// chat 模板钩子（W5）：`/v1/chat/completions` 的 messages → prompt 文本。
    /// None = 无模板（mock/未接模型——回退占位包装，既有行为）；
    /// Some(Err(msg)) = 模板拒绝（模型侧 raise_exception 语义 → HTTP 400）
    fn render_chat_template(&self, _messages: &[crate::openai::ChatMessage]) -> Option<Result<String, String>> {
        None
    }
    /// 同上，外加模板顶层 kwargs（`tools` / `enable_thinking` 等 vLLM 语义）。
    /// 默认忽略 kwargs 转调无参版——**只有真模板实现（cli 桥接）需要覆盖**；
    /// 覆盖者必须把 `tools` 交给模板，否则工具描述静默丢失。
    fn render_chat_template_with(
        &self,
        messages: &[crate::openai::ChatMessage],
        _extra: &[(String, serde_json::Value)],
    ) -> Option<Result<String, String>> {
        self.render_chat_template(messages)
    }
    /// eos token id（W5）：作为引擎 stop_token_ids——采样命中即停且不外吐。
    /// None = 无（行为与 W5 前全同）
    fn eos_id(&self) -> Option<u32> {
        None
    }
}

/// 测试用 tokenizer：id → 任意字节片段。
pub struct MockTokenizer {
    pub pieces: Vec<Vec<u8>>,
}

impl MockTokenizer {
    pub fn from_texts(texts: &[&str]) -> Self {
        Self { pieces: texts.iter().map(|s| s.as_bytes().to_vec()).collect() }
    }

    pub fn from_bytes(pieces: Vec<Vec<u8>>) -> Self {
        Self { pieces }
    }
}

impl Tokenizer for MockTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        // 占位实现：确定性切分即可，验收测试不依赖具体值
        text.as_bytes()
            .chunks(4)
            .filter(|c| !c.is_empty())
            .map(|c| (c[0] as u32) % self.pieces.len().max(1) as u32)
            .collect()
    }

    fn piece_bytes(&self, id: u32) -> Cow<'_, [u8]> {
        match self.pieces.get(id as usize) {
            Some(b) => Cow::Owned(b.clone()),
            None => Cow::Owned(b"?".to_vec()),
        }
    }
}

/// 增量解码器：token 流 → 文本流，保证每个输出都是完整 UTF-8
/// （跨 token 的多字节字符会被扣住直到补全——SGLang Detokenizer 的职责，tokio 侧实现）。
pub struct IncrementalDetok {
    tok: Arc<dyn Tokenizer>,
    buf: Vec<u8>,
}

impl IncrementalDetok {
    pub fn new(tok: Arc<dyn Tokenizer>) -> Self {
        Self { tok, buf: Vec::new() }
    }

    /// 喂入一个 token，返回本次可安全吐出的完整文本（可能为空串）。
    pub fn push(&mut self, id: u32) -> String {
        self.buf.extend_from_slice(&self.tok.piece_bytes(id));
        self.drain_valid()
    }

    /// 终结：丢弃不完整尾部（与主流引擎行为一致——BPE 正常终止时不应有残留）。
    pub fn finish(&mut self) {
        self.buf.clear();
    }

    fn drain_valid(&mut self) -> String {
        match std::str::from_utf8(&self.buf) {
            Ok(s) => {
                let out = s.to_string();
                self.buf.clear();
                out
            }
            Err(e) => {
                let valid = e.valid_up_to();
                let out = String::from_utf8(self.buf[..valid].to_vec()).unwrap_or_default();
                let incomplete_tail = e.error_len().is_none();
                self.buf.drain(..valid);
                if !incomplete_tail && !self.buf.is_empty() {
                    // 硬无效字节（正常不应出现）：丢一个防止缓冲毒化
                    self.buf.remove(0);
                }
                out
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_passthrough() {
        let tok: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::from_texts(&["Hel", "lo"]));
        let mut d = IncrementalDetok::new(tok);
        assert_eq!(d.push(0), "Hel");
        assert_eq!(d.push(1), "lo");
    }

    #[test]
    fn utf8_char_split_across_tokens() {
        // "你" = e4 bd a0，"好" = e5 a5 bd：片段刻意在字节中间切开
        let tok: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::from_bytes(vec![
            vec![0xe4, 0xbd],
            vec![0xa0, 0xe5, 0xa5],
            vec![0xbd],
        ]));
        let mut d = IncrementalDetok::new(tok);
        assert_eq!(d.push(0), ""); // 半个"你"，扣住
        assert_eq!(d.push(1), "你"); // "你"补全 + "好"的前两字节扣住
        assert_eq!(d.push(2), "好");
    }

    #[test]
    fn finish_drops_incomplete_tail() {
        let tok: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::from_bytes(vec![vec![0xe4, 0xbd]]));
        let mut d = IncrementalDetok::new(tok);
        assert_eq!(d.push(0), "");
        d.finish();
    }
}
