//! 文本标记的流式安全切分（工具调用的 `<tool_call>`、思维链的 `</think>` 共用）。
//!
//! 流式下「正文」只能放行**不再可能变成标记前缀**的部分，否则标记会被劈成两半、
//! 一半漏进正文。这里给出该前缀长度——注意候选切点必须是 UTF-8 字符边界
//! （首版按字节切在中文上 panic 过）。

/// 可安全外发的前缀字节数（末尾若是 `marker` 的前缀则扣住）
pub fn safe_prefix(s: &str, marker: &str) -> usize {
    let start = s.len().saturating_sub(marker.len() - 1);
    (start..s.len())
        .find(|&cut| s.get(cut..).is_some_and(|suf| marker.starts_with(suf)))
        .unwrap_or(s.len())
}
