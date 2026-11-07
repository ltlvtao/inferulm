//! 思维链（`</think>`）切分：把生成流分成「推理」与「正文」两段。
//!
//! 形态来源：Qwen 系模板的生成提示以 `<think>\n` 收尾（开思考），模型先吐推理、
//! 再吐 `</think>`、再吐正文——所以**推理段没有起始标记**，唯一的边界是第一次
//! 出现的 `</think>`（标记本身属于格式，不外发）。这与 vLLM 的
//! `--reasoning-parser deepseek_r1/qwen3` 同语义（`text.split("</think>")` 前后两段）。
//!
//! **必须按请求/模型能力开启**：没有思考能力的模型永远不吐 `</think>`，
//! 若无条件切分会把全部正文误判成推理（vLLM 用服务端 `--reasoning-parser`
//! 开关表达同一约束；我们的开关在 `ServeState.reasoning`，CLI `--reasoning-parser`
//! 可 `auto` 探测模板）。
//!
//! 单测钉三件：跨 token 边界（标记被劈开）、多次标记只认第一次、
//! 没有标记时全部算推理且 `finish()` 不丢字。
use crate::marker::safe_prefix;

const END: &str = "</think>";

/// 切出的一段
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Seg {
    Reasoning(String),
    Content(String),
}

pub struct ReasoningSplitter {
    buf: String,
    /// 已见过 `</think>`（之后全是正文）
    done: bool,
}

impl Default for ReasoningSplitter {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningSplitter {
    pub fn new() -> Self {
        Self { buf: String::new(), done: false }
    }

    pub fn feed(&mut self, text: &str) -> Vec<Seg> {
        if self.done {
            return if text.is_empty() { vec![] } else { vec![Seg::Content(text.to_string())] };
        }
        self.buf.push_str(text);
        let Some(i) = self.buf.find(END) else {
            // 尾巴可能是标记的一半（如 "</thi"）——扣住，别漏进推理文本
            let keep = safe_prefix(&self.buf, END);
            if keep == 0 {
                return vec![];
            }
            let head: String = self.buf.drain(..keep).collect();
            return vec![Seg::Reasoning(head)];
        };
        let rest = self.buf[i + END.len()..].to_string();
        let head = self.buf[..i].to_string();
        self.buf.clear();
        self.done = true;
        let mut out = Vec::new();
        if !head.is_empty() {
            out.push(Seg::Reasoning(head));
        }
        if !rest.is_empty() {
            out.push(Seg::Content(rest));
        }
        out
    }

    /// 生成结束：没等到 `</think>` ⇒ 已收到的全是推理（被截断的思考），不丢字
    pub fn finish(&mut self) -> Vec<Seg> {
        if self.done {
            return vec![];
        }
        self.done = true;
        if self.buf.is_empty() {
            vec![]
        } else {
            vec![Seg::Reasoning(std::mem::take(&mut self.buf))]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(text: &str, chunk: usize) -> (String, String) {
        let mut s = ReasoningSplitter::new();
        let mut segs = Vec::new();
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            // 按 UTF-8 字符边界切（模拟 token 解码器行为）
            let mut j = (i + chunk).min(bytes.len());
            while j < bytes.len() && !text.is_char_boundary(j) {
                j += 1;
            }
            segs.extend(s.feed(&text[i..j]));
            i = j;
        }
        segs.extend(s.finish());
        let mut r = String::new();
        let mut c = String::new();
        for seg in segs {
            match seg {
                Seg::Reasoning(x) => r.push_str(&x),
                Seg::Content(x) => c.push_str(&x),
            }
        }
        (r, c)
    }

    /// r1 · 标准形态：推理 + 标记 + 正文；标记本身不外发
    #[test]
    fn r1_basic_split() {
        assert_eq!(run("想一想</think>\n\n答案", 100), ("想一想".into(), "\n\n答案".into()));
    }

    /// r2 · 任意切分粒度（含逐字节）结果一致
    #[test]
    fn r2_any_chunking() {
        let text = "推理里有中文和 </thi 这种半截</think>正文也很长";
        let want = ("推理里有中文和 </thi 这种半截".to_string(), "正文也很长".to_string());
        for chunk in [1usize, 2, 3, 5, 7, 13, 100] {
            assert_eq!(run(text, chunk), want, "chunk={chunk}");
        }
    }

    /// r3 · 没有标记（未开思考的模型 / 被截断）：全部算推理，finish 不丢字
    #[test]
    fn r3_no_marker_all_reasoning() {
        assert_eq!(run("只有推理没结束", 3), ("只有推理没结束".into(), String::new()));
    }

    /// r4 · 多个标记：只认第一次，后面的属于正文
    #[test]
    fn r4_only_first_marker() {
        assert_eq!(run("a</think>b</think>c", 4), ("a".into(), "b</think>c".into()));
    }

    /// r5 · 标记在开头：推理为空（正文从空串开始）
    #[test]
    fn r5_marker_at_start() {
        assert_eq!(run("</think>正文", 2), (String::new(), "正文".into()));
    }
}
