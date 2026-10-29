//! 对拍用：与 vLLM /tokenize 比较同一组 prompt 的 token ids。
//! 用法：cargo run -p wein-model --example tokenize_dump --release -- <model_dir> < prompt文件
//! （stdin 每行一个 prompt，输出 JSON 数组每行一组 ids——与 vllm-$i.ids 同格式）
use std::io::{BufRead, Write};

fn main() {
    let dir = std::env::args().nth(1).expect("用法: tokenize_dump <model_dir>");
    let tok = wein_model::HfTokenizer::open(std::path::Path::new(&dir)).expect("open tokenizer");
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line.expect("read line");
        let ids = tok.encode(&line);
        writeln!(out, "{ids:?}").expect("write");
    }
}
