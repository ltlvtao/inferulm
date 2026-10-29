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

/// serve 侧消息的最小面（M0 文本对话：role + content 字符串）
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct ChatMsg {
    pub role: String,
    pub content: String,
}

pub struct ChatTemplate {
    env: minijinja::Environment<'static>,
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
        let mut ctx = serde_json::json!({ "messages": messages, "add_generation_prompt": true });
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
