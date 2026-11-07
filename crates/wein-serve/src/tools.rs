//! 生成文本 → OpenAI `tool_calls` 的**增量**解析器。
//!
//! Qwen 系模板要求模型按固定 XML 形态作答（模板原文 `<tool_call>\n<function=…>`），
//! 而 OpenAI 客户端只认 `message.tool_calls[i].function.{name, arguments}`（arguments
//! 是 JSON 字符串）。本模块把生成流实时切成「正文」与「调用」两类片，正文照旧
//! 出 `content`，调用转成 OpenAI 形状；`<tool_call>` 标记本身绝不外泄。
//!
//! 三条硬约束（均由单测钉住）：
//! 1. **跨 token 安全**：调用标记可能被任意切开，正文只能吐「不可能再变成标记」
//!    的前缀（`holdback`）。
//! 2. **不丢文本**：未闭合的 `<tool_call>`（被 max_tokens 截断）原样作为正文吐出。
//! 3. **参数按 schema 定类型**：模板对 mapping/sequence 用 `tojson`、标量用
//!    `str()`，所以参数值要按声明类型还原（string 就保持 string，别把 "007"
//!    变成数字 7）。
use serde_json::{json, Map, Value};

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

/// `WEIN_TOOL_PARSER=0` 关闭解析（回退旧行为：标记原样出现在 content 里）
pub fn parser_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("WEIN_TOOL_PARSER").map(|v| v != "0").unwrap_or(true))
}

/// 解析出的单个调用（arguments 已是 JSON 文本）
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// 解析器输出的一片
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Piece {
    Content(String),
    Call(ToolCall),
}

pub struct ToolParser {
    /// 未决策文本（可能是标记前缀、也可能是一整个未闭合调用）
    buf: String,
    /// 已进入 `<tool_call>`，正在攒 body
    in_call: bool,
    /// 调用 id 前缀（每请求唯一，取响应 id）
    id_prefix: String,
    /// 已产出调用数（id 序号 + `has_calls` 判据）
    n_calls: usize,
    /// 工具声明（`tools` 数组原文，用于参数定类型）
    schema: Value,
}

impl ToolParser {
    pub fn new(id_prefix: impl Into<String>, tools: Value) -> Self {
        Self { buf: String::new(), in_call: false, id_prefix: id_prefix.into(), n_calls: 0, schema: tools }
    }

    pub fn has_calls(&self) -> bool {
        self.n_calls > 0
    }

    /// 喂一片生成文本，返回可以立刻下发的片（顺序有意义）
    pub fn feed(&mut self, text: &str) -> Vec<Piece> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        loop {
            if !self.in_call {
                match self.buf.find(OPEN) {
                    Some(i) => {
                        if i > 0 {
                            out.push(Piece::Content(self.buf[..i].to_string()));
                        }
                        self.buf.drain(..i + OPEN.len());
                        self.in_call = true;
                    }
                    None => {
                        // 只放行「不再可能是标记前缀」的部分
                        let keep = holdback(&self.buf);
                        if keep < self.buf.len() {
                            out.push(Piece::Content(self.buf[..keep].to_string()));
                            self.buf.drain(..keep);
                        }
                        break;
                    }
                }
            } else {
                match self.buf.find(CLOSE) {
                    Some(i) => {
                        let body: String = self.buf.drain(..i + CLOSE.len()).collect();
                        let body = &body[..body.len() - CLOSE.len()];
                        self.in_call = false;
                        if let Some(c) = self.parse_body(body) {
                            out.push(Piece::Call(c));
                        }
                    }
                    None => break,
                }
            }
        }
        out
    }

    /// 生成结束：吐出所有残留（未闭合标记按正文处理——截断不该吞掉用户可见文本）
    pub fn finish(&mut self) -> Vec<Piece> {
        let mut out = Vec::new();
        if self.in_call {
            // 回填被吃掉的起始标记，客户端看到的是模型真吐的东西
            self.buf.insert_str(0, OPEN);
            self.in_call = false;
        }
        if !self.buf.is_empty() {
            out.push(Piece::Content(std::mem::take(&mut self.buf)));
        }
        out
    }

    fn parse_body(&mut self, body: &str) -> Option<ToolCall> {
        let (name, args) = parse_xml_body(body).or_else(|| parse_json_body(body))?;
        let decl = self
            .schema
            .as_array()
            .and_then(|a| a.iter().find(|t| t["function"]["name"].as_str() == Some(name.as_str())))
            .map(|t| &t["function"]["parameters"]);
        let coerced = coerce_args(&args, decl);
        let id = format!("{}_{}", self.id_prefix, self.n_calls);
        self.n_calls += 1;
        Some(ToolCall { id, name, arguments: Value::Object(coerced).to_string() })
    }
}

/// 保留的最长「可能是 `<tool_call>` 前缀」的后缀长度
fn holdback(s: &str) -> usize {
    let max = OPEN.len() - 1;
    let bytes = s.as_bytes();
    let start = bytes.len().saturating_sub(max);
    // 从最长的候选（最小切点）开始：长后缀能匹配时短后缀必然不是答案
    // （`<tool_call>` 无自重叠前缀/后缀，最多命中一个）
    (start..bytes.len())
        .find(|&cut| s.get(cut..).is_some_and(|suf| OPEN.starts_with(suf)))
        .unwrap_or(bytes.len())
}

/// `<function=NAME>\n<parameter=K>\nV\n</parameter>…</function>`
fn parse_xml_body(body: &str) -> Option<(String, Map<String, Value>)> {
    let fpos = body.find("<function=")?;
    let rest = &body[fpos + "<function=".len()..];
    let end = rest.find('>')?;
    let name = rest[..end].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut after = &rest[end + 1..];
    let mut args = Map::new();
    while let Some(p) = after.find("<parameter=") {
        let r = &after[p + "<parameter=".len()..];
        let Some(kend) = r.find('>') else { break };
        let key = r[..kend].trim().to_string();
        let r2 = &r[kend + 1..];
        let Some(vend) = r2.find("</parameter>") else { break };
        // 模板形态是 `<parameter=k>\nVALUE\n</parameter>`：剥一层换行，保留内部换行
        let raw = r2[..vend].strip_prefix('\n').unwrap_or(&r2[..vend]);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        args.insert(key, Value::String(raw.to_string()));
        after = &r2[vend + "</parameter>".len()..];
    }
    Some((name, args))
}

/// 退化形态：body 直接是 `{"name":…, "arguments":{…}}`（部分模型/版本会这样吐）
fn parse_json_body(body: &str) -> Option<(String, Map<String, Value>)> {
    let v: Value = serde_json::from_str(body.trim()).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let args = match v.get("arguments") {
        Some(Value::Object(o)) => o.clone(),
        Some(Value::String(s)) => serde_json::from_str::<Map<String, Value>>(s).ok()?,
        _ => Map::new(),
    };
    Some((name, args))
}

/// 按声明类型还原参数值：声明 string ⇒ 原样；其余（object/array/number/bool）
/// 尝试 JSON 解析，失败保持字符串（宁可比声明宽松，也不能丢值）
fn coerce_args(args: &Map<String, Value>, decl: Option<&Value>) -> Map<String, Value> {
    let props = decl.and_then(|d| d.get("properties"));
    let mut out = Map::new();
    for (k, v) in args {
        let ty = props.and_then(|p| p.get(k)).and_then(|p| p.get("type")).and_then(|t| t.as_str());
        let val = match (ty, v) {
            (Some("string"), Value::String(s)) => Value::String(s.clone()),
            (_, Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone())),
            (_, other) => other.clone(),
        };
        out.insert(k.clone(), val);
    }
    out
}

/// OpenAI 流式 `tool_calls` 增量片
pub fn call_delta(index: usize, c: &ToolCall) -> Value {
    json!({
        "tool_calls": [{
            "index": index,
            "id": c.id,
            "type": "function",
            "function": { "name": c.name, "arguments": c.arguments }
        }]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Value {
        json!([{"type": "function", "function": {"name": "get_weather", "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"type": "string"},
                           "opts": {"type": "object"}}}}},
               {"type": "function", "function": {"name": "add", "parameters": {
            "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}}}])
    }

    fn calls(ps: &[Piece]) -> Vec<ToolCall> {
        ps.iter().filter_map(|p| match p {
            Piece::Call(c) => Some(c.clone()),
            _ => None,
        })
        .collect()
    }

    fn content(ps: &[Piece]) -> String {
        ps.iter().filter_map(|p| match p {
            Piece::Content(s) => Some(s.as_str()),
            _ => None,
        })
        .collect()
    }

    const ONE: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nBeijing\n</parameter>\n\
                       <parameter=unit>\nc\n</parameter>\n</function>\n</tool_call>";

    /// t1 · 全文一次喂：调用被解出、标记不外泄、参数按声明定类型（string 保持）
    #[test]
    fn t1_whole_text() {
        let mut p = ToolParser::new("chatcmpl-x", tools());
        let ps = p.feed(&format!("Sure.{ONE}"));
        assert_eq!(content(&ps), "Sure.");
        let cs = calls(&ps);
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].name, "get_weather");
        assert_eq!(cs[0].arguments, r#"{"city":"Beijing","unit":"c"}"#);
        assert!(p.has_calls());
        assert!(p.finish().is_empty());
    }

    /// t2 · 逐字节喂：与整段喂等价（标记跨「token」切开也不漏正文）
    #[test]
    fn t2_byte_by_byte_equals_whole() {
        let text = format!("好的。{ONE}\n完成。{ONE}");
        let mut whole = ToolParser::new("r", tools());
        let mut all = whole.feed(&text);
        all.extend(whole.finish());
        let mut byb = ToolParser::new("r", tools());
        let mut split = Vec::new();
        for ch in text.chars() {
            let mut b = [0u8; 4];
            split.extend(byb.feed(ch.encode_utf8(&mut b)));
        }
        split.extend(byb.finish());
        assert_eq!(content(&split), content(&all));
        assert_eq!(calls(&split), calls(&all));
        assert_eq!(content(&all), "好的。\n完成。");
        assert_eq!(calls(&all).len(), 2);
        // id 按请求内序号递增
        assert_eq!(calls(&all)[0].id, "r_0");
        assert_eq!(calls(&all)[1].id, "r_1");
    }

    /// t3 · 悬空前缀：`<tool` 不出正文，补全后成调用
    #[test]
    fn t3_partial_marker_is_held() {
        let mut p = ToolParser::new("r", tools());
        assert_eq!(content(&p.feed("ok <tool")), "ok ");
        let ps = p.feed(
            "_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(calls(&ps).len(), 1);
        assert_eq!(calls(&ps)[0].arguments, r#"{"a":1,"b":2}"#);
        assert!(p.feed("").is_empty());
    }

    /// t4 · 未闭合（被 max_tokens 截断）：finish 把原文当正文吐出（不丢字）
    #[test]
    fn t4_unterminated_falls_back_to_content() {
        let mut p = ToolParser::new("r", tools());
        let ps = p.feed("看：<tool_call>\n<function=add>\n<parameter=a>\n1");
        assert_eq!(content(&ps), "看：");
        let tail = p.finish();
        assert_eq!(content(&tail), "<tool_call>\n<function=add>\n<parameter=a>\n1");
        assert!(calls(&tail).is_empty());
        assert!(!p.has_calls());
    }

    /// t5 · 纯正文（含 `<` 但不是标记）：逐片外发，结尾残留空前缀不吞字
    #[test]
    fn t5_plain_text_passthrough() {
        let mut p = ToolParser::new("r", tools());
        let mut out = content(&p.feed("a < b > c "));
        out += &content(&p.feed("d"));
        out += &content(&p.finish());
        assert_eq!(out, "a < b > c d");
    }

    /// t6 · 参数定类型：声明 string 的 "007" 不被转成数字；声明 integer 的 "1" 转数字；
    ///      object 参数按 JSON 还原；嵌套内部换行保留
    #[test]
    fn t6_argument_typing() {
        let mut p = ToolParser::new("r", tools());
        let ps = p.feed(
            "<tool_call>\n<function=get_weather>\n<parameter=city>\n007\n</parameter>\n\
             <parameter=opts>\n{\"a\": [1, 2], \"b\": {\"c\": true}}\n</parameter>\n</function>\n</tool_call>",
        );
        let c = calls(&ps).remove(0);
        let v: Value = serde_json::from_str(&c.arguments).unwrap();
        assert_eq!(v["city"], json!("007"));
        assert_eq!(v["opts"], json!({"a": [1, 2], "b": {"c": true}}));
    }

    /// t7 · 退化形态：body 是 JSON（`{"name":…,"arguments":"{…}"}`）也能解
    #[test]
    fn t7_json_body_fallback() {
        let mut p = ToolParser::new("r", tools());
        let ps = p.feed(
            "<tool_call>{\"name\": \"add\", \"arguments\": \"{\\\"a\\\": 3, \\\"b\\\": 4}\"}</tool_call>",
        );
        let c = calls(&ps).remove(0);
        assert_eq!(c.name, "add");
        assert_eq!(c.arguments, r#"{"a":3,"b":4}"#);
    }

    /// t8 · 多行参数值：内部换行原样保留（模板明确允许）
    #[test]
    fn t8_multiline_value() {
        let mut p = ToolParser::new("r", tools());
        let ps = p.feed(
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nline1\nline2\n</parameter>\n</function>\n</tool_call>",
        );
        let v: Value = serde_json::from_str(&calls(&ps)[0].arguments).unwrap();
        assert_eq!(v["city"], json!("line1\nline2"));
    }
}
