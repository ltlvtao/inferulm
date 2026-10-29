//! 逐层对拍驱动：与 HF transformers 参照比逐层激活（vs vLLM 对比发现的
//! 递推偏差定位战）。WEIN_TRACE_DIR 设定时，executor 在每次 run_forward
//! 落盘 embed/每层残差流/logits（f32 LE）。
//!
//! 用法：
//!   WEIN_TRACE_DIR=<dir> cargo run -p wein-exec --example qwen_trace --release -- \
//!       <model_dir> <tok_id...> [max_tokens]
//! 输出：生成的 token 序列（供参照端 forced decode 对齐同链）
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("用法: qwen_trace <model_dir> <tok_id...> [max_tokens]");
        std::process::exit(2);
    }
    let dir = &args[1];
    let max_tokens: u32 = args.last().unwrap().parse().unwrap_or_else(|_| {
        eprintln!("末参须为 max_tokens 整数");
        std::process::exit(2);
    });
    let prompt: Vec<u32> = args[2..args.len() - 1]
        .iter()
        .map(|a| a.parse().expect("token id 须为 u32"))
        .collect();
    let mut ex = wein_exec::Qwen35Executor::open(std::path::Path::new(dir))
        .unwrap_or_else(|e| panic!("open：{e}"));
    let out = ex.generate(&prompt, max_tokens);
    println!("prompt: {prompt:?}");
    println!("tokens: {out:?}");
}
