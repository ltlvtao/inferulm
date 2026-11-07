//! chat 模板真身（W5）：tokenizer_config.json 的 `chat_template`（Jinja）+
//! minijinja 渲染——与 HF transformers 同款环境（trim_blocks/lstrip_blocks、
//! raise_exception global），golden 对拍 Python jinja2。
//!
//! 分层：模板渲染是模型目录的属性（wein-model）；serve 层经 Tokenizer
//! trait 的 `render_chat_template` 钩子获得能力，桥接在 cli（§2 单向）。

use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use crate::{ModelError, Result};

/// serve 侧消息的最小面（文本对话 + 工具调用回放）。`content` 可空（带
/// `tool_calls` 的 assistant 消息 content 为 null）；`tool_calls` 原样透传
/// （模板读 `function.name` / `function.arguments`）。
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct ChatMsg {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

pub struct ChatTemplate {
    env: minijinja::Environment<'static>,
}

/// OpenAI wire 形态的 `tool_calls[i].function.arguments` 是 **JSON 字符串**
/// （如 `"{\"city\":\"Oslo\"}"`），而 Qwen 系模板按 mapping 消费
/// （`tool_call.arguments|items`）——直接喂字符串会渲染失败（jinja2 报
/// "Can only get item pairs from a mapping"，我们这边报 minijinja 同名错）。
/// 这里把「能解析成 JSON 值的字符串」就地换成解析结果，其余原样保留
/// （非字符串、解析失败、`{}` 等都不动）。`tool_calls` 与顶层 `arguments`
/// 两种形状（模板都认）都处理。
pub fn normalize_tool_calls(msgs: &Value) -> Value {
    let mut out = msgs.clone();
    let Some(arr) = out.as_array_mut() else { return out };
    for msg in arr.iter_mut() {
        let Some(calls) = msg.get_mut("tool_calls").and_then(|c| c.as_array_mut()) else {
            continue;
        };
        for call in calls.iter_mut() {
            // 形状 A：{"function": {"name", "arguments"}}；形状 B：{"name","arguments"}
            let holder = match call.get_mut("function") {
                Some(f) if f.is_object() => f,
                _ => call,
            };
            let Some(args) = holder.get_mut("arguments") else { continue };
            let Some(s) = args.as_str() else { continue };
            if let Ok(parsed) = serde_json::from_str::<Value>(s) {
                *args = parsed;
            }
        }
    }
    out
}

/// Python `json.dumps` 默认格式（jinja2 `tojson` 的底层）：`, `/`: ` 分隔、
/// ensure_ascii（非 ASCII 一律 `\uXXXX`，增补平面出代理对）。键序由
/// serde_json 的 BTreeMap 保证升序，与 `sort_keys=True` 同。
fn py_json(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else {
                // Python repr(float)：整数值补 ".0"（json.dumps(1.0) == "1.0"）
                let f = n.as_f64().unwrap_or(0.0);
                if f.is_finite() && f.fract() == 0.0 && f.abs() < 1e16 {
                    out.push_str(&format!("{f:.1}"));
                } else {
                    out.push_str(&format!("{f}"));
                }
            }
        }
        Value::String(s) => py_json_str(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_json(item, out);
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            for (i, (k, item)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_json_str(k, out);
                out.push_str(": ");
                py_json(item, out);
            }
            out.push('}');
        }
    }
}

fn py_json_str(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if (c as u32) < 0x7f => out.push(c),
            c => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

impl ChatTemplate {
    /// 模型目录 → 模板：`tokenizer_config.json: chat_template`（缺失即无模板）
    pub fn open(dir: &Path) -> Result<Self> {
        let cfg: Value = std::fs::read_to_string(dir.join("tokenizer_config.json"))?
            .parse()
            .map_err(|e| ModelError::Json(format!("tokenizer_config.json: {e}")))?;
        let src = match cfg.get("chat_template") {
            Some(Value::String(s)) => s.clone(),
            // 独立文件变体（transformers 同支持）：tokenizer_config 无字段时
            // 读模型目录 chat_template.jinja
            _ => std::fs::read_to_string(dir.join("chat_template.jinja"))?,
        };
        let mut env = minijinja::Environment::new();
        // HF transformers 渲染环境（_compile_jinja_template）：
        // trim_blocks/lstrip_blocks
        env.set_syntax(
            minijinja::syntax::SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()
                .map_err(|e| ModelError::Json(format!("syntax 构建：{e}")))?,
        );
        // tojson：jinja2 内建过滤器（工具描述与参数值按 JSON 落进 prompt）。
        // 语义对齐 jinja2 的 `htmlsafe_json_dumps`：
        //   1. **Python json.dumps 默认格式**——`", "`/`": "` 分隔、键排序、
        //      ensure_ascii（非 ASCII 转 `\uXXXX`）；
        //   2. `< > & '` 再转 `\u00XX`。
        // serde_json 未开 preserve_order ⇒ 对象天然按键升序，与 sort_keys 同。
        env.add_filter(
            "tojson",
            |v: minijinja::Value| -> std::result::Result<minijinja::Value, minijinja::Error> {
                let plain: serde_json::Value = serde_json::to_value(&v)
                    .map_err(|e| minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string()))?;
                let mut s = String::new();
                py_json(&plain, &mut s);
                let s = s
                    .replace('<', "\\u003c")
                    .replace('>', "\\u003e")
                    .replace('&', "\\u0026")
                    .replace('\'', "\\u0027");
                Ok(minijinja::Value::from(s))
            },
        );
        // raise_exception：模板的拒绝路径（无 user query / 非法角色等）→ 渲染 Err
        // （返回类型写全路径：文件顶部 Result 是 crate 别名 Result<T, ModelError>）
        env.add_global(
            "raise_exception",
            minijinja::Value::from_function(
                |msg: String| -> std::result::Result<minijinja::Value, minijinja::Error> {
                    Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
                },
            ),
        );
        // 3.0 的 Environment<'source> 借用模板源——set_loader 让闭包持有源串
        // （owned，按名惰性加载缓存）
        env.set_loader(move |_name| Ok(Some(src.clone())));
        // Python str 方法仿真：HF 模板按「Jinja2 + Python 运行时」书写，
        // minijinja 引擎吃语法（含 [::-1] 切片）但不带 str 方法——
        // 模板家族用到的方法在此补齐（llama.cpp minja 同思路）
        env.set_unknown_method_callback(|_state, value, method, args| {
            let Some(s) = value.as_str() else {
                return Err(minijinja::Error::new(
                    minijinja::ErrorKind::UnknownMethod,
                    format!("unknown method: {value:?} has no method named {method}"),
                ));
            };
            let arg_str = |i: usize| -> Option<String> {
                args.get(i).and_then(|a| a.as_str().map(|s| s.to_string()))
            };
            match method {
                "startswith" => Ok(minijinja::Value::from(
                    arg_str(0).map(|p| s.starts_with(&p)).unwrap_or(false),
                )),
                "endswith" => Ok(minijinja::Value::from(
                    arg_str(0).map(|p| s.ends_with(&p)).unwrap_or(false),
                )),
                "split" => {
                    let (Some(sep), true) = (arg_str(0), args.len() == 1) else {
                        return Err(minijinja::Error::new(
                            minijinja::ErrorKind::InvalidOperation,
                            "split 需恰一个字符串参数（模板语义）",
                        ));
                    };
                    let parts: Vec<minijinja::Value> =
                        s.split(&sep as &str).map(minijinja::Value::from).collect();
                    Ok(minijinja::Value::from(parts))
                }
                "rstrip" => {
                    let cut = arg_str(0);
                    let out = match &cut {
                        Some(chars) => s.trim_end_matches(|c| chars.contains(c)).to_string(),
                        None => s.trim_end().to_string(),
                    };
                    Ok(minijinja::Value::from(out))
                }
                "lstrip" => {
                    let cut = arg_str(0);
                    let out = match &cut {
                        Some(chars) => s.trim_start_matches(|c| chars.contains(c)).to_string(),
                        None => s.trim_start().to_string(),
                    };
                    Ok(minijinja::Value::from(out))
                }
                _ => Err(minijinja::Error::new(
                    minijinja::ErrorKind::UnknownMethod,
                    format!("unknown method: string has no method named {method}"),
                )),
            }
        });
        Ok(ChatTemplate { env })
    }

    /// 渲染：add_generation_prompt=true 固定语义 + 可选顶层 kwarg（如
    /// enable_thinking）。上下文整体走 serde_json::Value——kwarg 即模板
    /// 顶层变量，未传即 undefined（模板 `is defined` 分支靠这一点）
    pub fn render_with(&self, messages: &[ChatMsg], extra: &[(String, Value)]) -> Result<String> {
        // 消息先过 wire 形态归一（OpenAI 的 `arguments` 是 JSON 字符串，模板按
        // mapping 消费——见 `normalize_tool_calls`）
        let msgs = normalize_tool_calls(
            &serde_json::to_value(messages)
                .map_err(|e| ModelError::Json(format!("messages 序列化失败：{e}")))?,
        );
        let mut ctx = serde_json::json!({ "messages": msgs, "add_generation_prompt": true });
        for (k, v) in extra {
            ctx[k] = v.clone();
        }
        self.env
            .get_template("chat")
            .expect("模板注册于 open")
            .render(minijinja::Value::from(minijinja::value::Serde(&ctx)))
            .map_err(|e| ModelError::Json(format!("chat 模板渲染失败：{e}")))
    }

    pub fn render(&self, messages: &[ChatMsg]) -> Result<String> {
        self.render_with(messages, &[])
    }
}
