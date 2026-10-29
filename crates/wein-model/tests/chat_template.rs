//! W5-0 冒烟 + W5-C 验收（ct 系列）——Qwen3.5 真 chat 模板渲染。
//!
//! golden 参考：Python jinja2 3.1.6（HF transformers 同款环境：
//! ImmutableSandboxedEnvironment(trim_blocks, lstrip_blocks) + raise_exception
//! global）——模板是模型目录 tokenizer_config.json 的 `chat_template` 字段，
//! 逐字节对拍。冒烟要回答的问题：minijinja 能否渲染真模板的
//! `messages[::-1]` 切片 / content.startswith·split·rstrip 字符串方法 /
//! tojson·items 过滤器 / namespace·macro·loop.previtem。

use wein_model::{model_dir, ChatTemplate};

fn tpl_dir() -> Option<std::path::PathBuf> {
    let d = model_dir();
    if d.is_none() {
        eprintln!("SKIP（模型目录不在场：WEIN_MODEL_DIR 或 ~/.cache/wein/Qwen3.5-4B）");
    }
    d
}

fn msgs(v: serde_json::Value) -> Vec<wein_model::ChatMsg> {
    serde_json::from_value(v).unwrap()
}

/// ct1 · 单 user 消息 + 生成提示尾（含 `<think>\n`——enable_thinking 未传
/// 即开）：最简但完整的主力路径
#[test]
fn ct1_single_user() {
    let Some(dir) = tpl_dir() else { return };
    let t = ChatTemplate::open(&dir).unwrap();
    assert_eq!(
        t.render(&msgs(serde_json::json!([{"role":"user","content":"Hello"}]))).unwrap(),
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
}

/// ct2 · system 前置：system 块插在 user 前
#[test]
fn ct2_system_user() {
    let Some(dir) = tpl_dir() else { return };
    let t = ChatTemplate::open(&dir).unwrap();
    assert_eq!(
        t.render(
            &msgs(serde_json::json!([
                {"role":"system","content":"You are a helpful assistant."},
                {"role":"user","content":"What is 2+2?"}
            ])),
        )
        .unwrap(),
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
         <|im_start|>user\nWhat is 2+2?<|im_end|>\n\
         <|im_start|>assistant\n<think>\n"
    );
}

/// ct3 · 多轮：历史 user/assistant 对逐块渲染
#[test]
fn ct3_multiturn() {
    let Some(dir) = tpl_dir() else { return };
    let t = ChatTemplate::open(&dir).unwrap();
    assert_eq!(
        t.render(
            &msgs(serde_json::json!([
                {"role":"user","content":"Hi"},
                {"role":"assistant","content":"Hello!"},
                {"role":"user","content":"How are you?"}
            ])),
        )
        .unwrap(),
        "<|im_start|>user\nHi<|im_end|>\n\
         <|im_start|>assistant\nHello!<|im_end|>\n\
         <|im_start|>user\nHow are you?<|im_end|>\n\
         <|im_start|>assistant\n<think>\n"
    );
}

/// ct4 · 历史 assistant 带 `<think>...</think>`：最后一个 user query 之前的
/// assistant 消息把 reasoning 剥掉只留正文（`</think>` split 分支）
#[test]
fn ct4_assistant_think_stripped() {
    let Some(dir) = tpl_dir() else { return };
    let t = ChatTemplate::open(&dir).unwrap();
    assert_eq!(
        t.render(
            &msgs(serde_json::json!([
                {"role":"user","content":"Solve 1+1"},
                {"role":"assistant","content":"<think>one plus one</think>The answer is 2."},
                {"role":"user","content":"Thanks"}
            ])),
        )
        .unwrap(),
        "<|im_start|>user\nSolve 1+1<|im_end|>\n\
         <|im_start|>assistant\nThe answer is 2.<|im_end|>\n\
         <|im_start|>user\nThanks<|im_end|>\n\
         <|im_start|>assistant\n<think>\n"
    );
}

/// ct5 · 非思考模式（额外 kwarg）：尾换成 `<think>\n\n</think>\n\n`
#[test]
fn ct5_enable_thinking_false() {
    let Some(dir) = tpl_dir() else { return };
    let t = ChatTemplate::open(&dir).unwrap();
    assert_eq!(
        t.render_with(
            &msgs(serde_json::json!([{"role":"user","content":"Hello"}])),
            &[("enable_thinking".to_string(), serde_json::Value::Bool(false))],
        )
        .unwrap(),
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
}

/// ct6 · 模板拒绝：无 user query → raise_exception 透传为 Err
#[test]
fn ct6_no_user_query_is_error() {
    let Some(dir) = tpl_dir() else { return };
    let t = ChatTemplate::open(&dir).unwrap();
    let e = t
        .render(&msgs(serde_json::json!([{"role":"assistant","content":"hi"}])))
        .unwrap_err();
    let msg = e.to_string();
    assert!(msg.contains("No user query found"), "got {msg:?}");
}
