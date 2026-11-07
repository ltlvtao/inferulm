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

// ── ct7-ct10 · 工具调用（tools kwarg / tool 角色 / tool_calls 历史回放）──
//
// golden 与 ct1-ct6 同源：Python jinja2 3.1.6（ImmutableSandboxedEnvironment,
// trim_blocks/lstrip_blocks/raise_exception）渲染模型目录真模板。工具描述块
// （`# Tools` … `</IMPORTANT>`）由模型模板原文提供，故抽成前缀常量，测试用
// 「前缀逐字 + 尾逐字」两段拼接等价全串。

const TOOLS_PREAMBLE: &str = "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n\
{\"function\": {\"description\": \"Get current weather for a city\", \"name\": \"get_weather\", \"parameters\": {\"properties\": {\"city\": {\"type\": \"string\"}, \"unit\": {\"enum\": [\"c\", \"f\"], \"type\": \"string\"}}, \"required\": [\"city\"], \"type\": \"object\"}}, \"type\": \"function\"}\n\
{\"function\": {\"description\": \"Add two ints\", \"name\": \"add\", \"parameters\": {\"properties\": {\"a\": {\"type\": \"integer\"}, \"b\": {\"type\": \"integer\"}}, \"required\": [\"a\", \"b\"], \"type\": \"object\"}}, \"type\": \"function\"}\n\
</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

fn tools() -> serde_json::Value {
    serde_json::json!([
        {"type": "function", "function": {"name": "get_weather",
         "description": "Get current weather for a city",
         "parameters": {"type": "object", "properties": {"city": {"type": "string"},
         "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}},
        {"type": "function", "function": {"name": "add", "description": "Add two ints",
         "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
         "required": ["a", "b"]}}},
    ])
}

fn render_tools(t: &ChatTemplate, msgs_v: serde_json::Value) -> String {
    t.render_with(&msgs(msgs_v), &[("tools".to_string(), tools())]).unwrap()
}

fn assert_prefix_tail(out: &str, tail: &str) {
    let rest = match out.strip_prefix(TOOLS_PREAMBLE) {
        Some(r) => r,
        None => {
            // 定位首个差异字节并给出上下文（模板/prompt 逐字节对拍时必需）
            let (a, b) = (out.as_bytes(), TOOLS_PREAMBLE.as_bytes());
            let i = a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
            let lo = i.saturating_sub(40);
            panic!(
                "前缀不符 @{i}：out={:?} exp={:?}",
                &out[lo..(i + 40).min(out.len())],
                &TOOLS_PREAMBLE[lo..(i + 40).min(TOOLS_PREAMBLE.len())]
            )
        }
    };
    assert_eq!(rest, tail, "尾不符");
}

/// ct7 · tools 注入：无 system 消息时工具块即 system 消息，user 紧随
#[test]
fn ct7_tools_block_without_system() {
    let Some(dir) = tpl_dir() else { return };
    let t = ChatTemplate::open(&dir).unwrap();
    let out = render_tools(&t, serde_json::json!([{"role":"user","content":"Weather in Beijing?"}]));
    assert_prefix_tail(
        &out,
        "<|im_end|>\n<|im_start|>user\nWeather in Beijing?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    );
}

/// ct8 · 工具块 + system 追加：system 文本接在 IMPORTANT 块后（两个换行）
#[test]
fn ct8_tools_block_appends_system_text() {
    let Some(dir) = tpl_dir() else { return };
    let t = ChatTemplate::open(&dir).unwrap();
    let out = render_tools(
        &t,
        serde_json::json!([
            {"role":"system","content":"You are terse."},
            {"role":"user","content":"Weather in Beijing?"}
        ]),
    );
    assert_prefix_tail(
        &out,
        "\n\nYou are terse.<|im_end|>\n\
         <|im_start|>user\nWeather in Beijing?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    );
}

/// ct9 · 历史回放：assistant(tool_calls, content=null) + tool 响应
///      —— `arguments` 按 OpenAI 合同是 JSON **字符串**，模板按 mapping 消费
///      （`arguments|items`）⇒ 渲染前必须归一（`normalize_tool_calls`）。
///      tool 消息合并进同一个 `<|im_start|>user` 块（模板的 tool_response 语义）。
#[test]
fn ct9_history_tool_calls_and_response() {
    let Some(dir) = tpl_dir() else { return };
    let t = ChatTemplate::open(&dir).unwrap();
    let out = render_tools(
        &t,
        serde_json::json!([
            {"role":"system","content":"You are terse."},
            {"role":"user","content":"Weather in Beijing?"},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"call_1","type":"function",
                 "function":{"name":"get_weather","arguments":"{\"city\":\"Beijing\",\"unit\":\"c\"}"}}]},
            {"role":"tool","content":"-3C clear","tool_call_id":"call_1"}
        ]),
    );
    assert_prefix_tail(
        &out,
        "\n\nYou are terse.<|im_end|>\n\
         <|im_start|>user\nWeather in Beijing?<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\n\
         <tool_call>\n<function=get_weather>\n<parameter=city>\nBeijing\n</parameter>\n\
         <parameter=unit>\nc\n</parameter>\n</function>\n</tool_call><|im_end|>\n\
         <|im_start|>user\n<tool_response>\n-3C clear\n</tool_response><|im_end|>\n\
         <|im_start|>assistant\n<think>\n",
    );
}

/// ct10 · 并行调用 + 前置自然语言：content 非空时 `\n\n` 分隔首个 tool_call；
///         两个 tool 响应合并进一个 user 块（无分隔）
#[test]
fn ct10_parallel_calls_and_merged_responses() {
    let Some(dir) = tpl_dir() else { return };
    let t = ChatTemplate::open(&dir).unwrap();
    let out = render_tools(
        &t,
        serde_json::json!([
            {"role":"user","content":"Weather and sum"},
            {"role":"assistant","content":"Two calls.","tool_calls":[
                {"function":{"name":"get_weather","arguments":"{\"city\":\"Oslo\",\"unit\":\"c\"}"}},
                {"function":{"name":"add","arguments":{"a":1,"b":2}}}]},
            {"role":"tool","content":"-7C snow"},
            {"role":"tool","content":"3"}
        ]),
    );
    assert_prefix_tail(
        &out,
        "<|im_end|>\n<|im_start|>user\nWeather and sum<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\nTwo calls.\n\n\
         <tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n\
         <parameter=unit>\nc\n</parameter>\n</function>\n</tool_call>\n\
         <tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n\
         <parameter=b>\n2\n</parameter>\n</function>\n</tool_call><|im_end|>\n\
         <|im_start|>user\n<tool_response>\n-7C snow\n</tool_response>\n\
         <tool_response>\n3\n</tool_response><|im_end|>\n\
         <|im_start|>assistant\n<think>\n",
    );
}

/// ct11 · `tools: []` 与未传等价（模板 falsy 分支）；normalize 幂等且不碰坏值
#[test]
fn ct11_empty_tools_and_normalize() {
    let Some(dir) = tpl_dir() else { return };
    let t = ChatTemplate::open(&dir).unwrap();
    let with_empty = t
        .render_with(
            &msgs(serde_json::json!([{"role":"user","content":"Hello"}])),
            &[("tools".to_string(), serde_json::json!([]))],
        )
        .unwrap();
    assert_eq!(
        with_empty,
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
    // 非 JSON 字符串 / 非字符串 一律原样保留（不 panic、不丢字段）
    let v = serde_json::json!([{"role":"assistant","content":null,"tool_calls":[
        {"function":{"name":"f","arguments":"not json"}},
        {"function":{"name":"g","arguments":{"a":1}}},
        {"name":"h","arguments":"[1,2]"}]}]);
    let n = wein_model::normalize_tool_calls(&v);
    assert_eq!(n[0]["tool_calls"][0]["function"]["arguments"], serde_json::json!("not json"));
    assert_eq!(n[0]["tool_calls"][1]["function"]["arguments"], serde_json::json!({"a":1}));
    assert_eq!(n[0]["tool_calls"][2]["arguments"], serde_json::json!([1,2]));
}
