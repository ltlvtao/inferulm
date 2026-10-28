//! wein-bench：并发性能验收工具（归因阶梯的执行器，方法论见 bench-reports/BASELINE.md）。
//!
//! 两种工况：
//!   流式 token 泵  —— TTFT/ITL/聚合吞吐（默认）
//!   --flood       —— 纯路由上限（GET /v1/models，不进引擎）
//!
//! 噪声纪律：--repeat N 每项指标取 N 次运行的中位数；报告带 git rev + 时间戳，
//! 基线演化只认 battery 产出的报告（bench/battery.sh）。
//!
//! 用法：
//!   wein-bench --url http://127.0.0.1:8080 --model mock-model \
//!              --concurrency 100 --num 200 --max-tokens 32 --repeat 3 \
//!              [--json report.json] [--smoke "ttft_p99_ms<=100;tps_min>=2000"]
use std::time::{Duration, Instant};

use futures::StreamExt;
use serde_json::{json, Value};

struct Args {
    url: String,
    model: String,
    concurrency: usize,
    num: usize,
    max_tokens: u32,
    json: Option<String>,
    smoke: Option<String>,
    warmup: bool,
    repeat: usize,
}

fn arg(args: &[String], k: &str, d: &str) -> String {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| d.to_string())
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    Args {
        url: arg(&args, "--url", "http://127.0.0.1:8080"),
        model: arg(&args, "--model", "mock-model"),
        concurrency: arg(&args, "--concurrency", "32").parse().unwrap(),
        num: arg(&args, "--num", "64").parse().unwrap(),
        max_tokens: arg(&args, "--max-tokens", "32").parse().unwrap(),
        json: args.iter().position(|a| a == "--json").and_then(|i| args.get(i + 1)).cloned(),
        smoke: args.iter().position(|a| a == "--smoke").and_then(|i| args.get(i + 1)).cloned(),
        warmup: args.iter().any(|a| a == "--warmup"),
        repeat: arg(&args, "--repeat", "1").parse().unwrap(),
    }
}

fn percentile(mut v: Vec<f64>, p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((v.len() - 1) as f64 * p).round() as usize;
    v[idx]
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn meta() -> Value {
    json!({
        "ts_unix": now_unix(),
        "git_rev": std::env::var("WEIN_GIT_REV").unwrap_or_else(|_| "unknown".into()),
        "profile": std::env::var("WEIN_PROFILE").unwrap_or_else(|_| "unspecified".into()),
    })
}

/// GET /v1/models 洪泛：量 router（accept/parse/route/小 JSON 响应）的 QPS 上限。
async fn flood(args: &Args) {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .unwrap();
    let url = format!("{}/v1/models", args.url);
    for _ in 0..20 {
        let _ = client.get(&url).send().await;
    }

    let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(args.concurrency));
    let wall0 = Instant::now();
    let mut handles = Vec::new();
    for _ in 0..args.num {
        let sem = sem.clone();
        let client = client.clone();
        let url = url.clone();
        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire_owned().await.unwrap();
            let t0 = Instant::now();
            let ok = matches!(client.get(&url).send().await, Ok(r) if r.status().is_success());
            (ok, t0.elapsed().as_secs_f64() * 1e3)
        }));
    }
    let results = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|h| h.unwrap())
        .collect::<Vec<_>>();
    let wall = wall0.elapsed().as_secs_f64();

    let lat: Vec<f64> = results.iter().map(|(_, ms)| *ms).collect();
    let errors = results.iter().filter(|(ok, _)| !ok).count();
    let report = json!({
        "mode": "flood GET /v1/models",
        "meta": meta(),
        "config": {"concurrency": args.concurrency, "num": args.num},
        "result": {
            "errors": errors,
            "qps": args.num as f64 / wall,
            "lat_p50_ms": percentile(lat.clone(), 0.50),
            "lat_p99_ms": percentile(lat.clone(), 0.99),
            "wall_s": wall,
        }
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if let Some(path) = &args.json {
        std::fs::write(path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
        eprintln!("report written to {path}");
    }
    if errors > 0 {
        std::process::exit(1);
    }
}

#[derive(Default, Clone)]
struct ReqStat {
    ttft_ms: Option<f64>,
    itls_ms: Vec<f64>,
    tokens: u32,
    total_ms: f64,
    error: Option<String>,
}

async fn one_request(client: &reqwest::Client, url: &str, model: &str, max_tokens: u32) -> ReqStat {
    let mut stat = ReqStat::default();
    let t0 = Instant::now();
    let resp = match client
        .post(url)
        .json(&json!({
            "model": model,
            "messages": [{"role": "user", "content": "bench"}],
            "max_tokens": max_tokens,
            "stream": true,
        }))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            stat.error = Some(format!("send: {e}"));
            return stat;
        }
    };
    if !resp.status().is_success() {
        stat.error = Some(format!("status {}", resp.status()));
        return stat;
    }
    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    let mut last_content_at: Option<Instant> = None;
    'outer: while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                stat.error = Some(format!("stream: {e}"));
                break;
            }
        };
        buf.extend_from_slice(&chunk);
        loop {
            let Some(pos) = buf.windows(2).position(|w| w == b"\n\n") else { break };
            let frame: Vec<u8> = buf.drain(..pos + 2).collect();
            let mut data: Option<String> = None;
            for line in frame.split(|b| *b == b'\n') {
                if let Some(d) = line.strip_prefix(b"data: ") {
                    data = Some(String::from_utf8_lossy(d).into_owned());
                }
            }
            let Some(data) = data else { continue };
            if data == "[DONE]" {
                stat.total_ms = t0.elapsed().as_secs_f64() * 1e3;
                break 'outer;
            }
            let Ok(v) = serde_json::from_str::<Value>(&data) else { continue };
            let content = v["choices"][0]["delta"]["content"].as_str().unwrap_or("");
            if !content.is_empty() {
                let now = Instant::now();
                if stat.ttft_ms.is_none() {
                    stat.ttft_ms = Some(now.duration_since(t0).as_secs_f64() * 1e3);
                } else if let Some(prev) = last_content_at {
                    stat.itls_ms.push(now.duration_since(prev).as_secs_f64() * 1e3);
                }
                last_content_at = Some(now);
                stat.tokens += 1;
            }
        }
    }
    if stat.total_ms == 0.0 {
        stat.total_ms = t0.elapsed().as_secs_f64() * 1e3;
    }
    stat
}

/// 单轮流式测量，返回扁平指标（便于跨轮取中位数与 smoke 匹配）。
async fn streaming_once(client: &reqwest::Client, args: &Args) -> Value {
    let url = format!("{}/v1/chat/completions", args.url);
    let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(args.concurrency));
    let wall0 = Instant::now();
    let mut handles = Vec::new();
    for _ in 0..args.num {
        let sem = sem.clone();
        let client = client.clone();
        let url = url.clone();
        let model = args.model.clone();
        let max_tokens = args.max_tokens;
        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire_owned().await.unwrap();
            one_request(&client, &url, &model, max_tokens).await
        }));
    }
    let stats: Vec<ReqStat> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|h| h.unwrap())
        .collect();
    let wall = wall0.elapsed().as_secs_f64();

    let errors = stats.iter().filter(|s| s.error.is_some()).count();
    let ttfts: Vec<f64> = stats.iter().filter_map(|s| s.ttft_ms).collect();
    let itls: Vec<f64> = stats.iter().flat_map(|s| s.itls_ms.clone()).collect();
    let total_tokens: u32 = stats.iter().map(|s| s.tokens).sum();
    let mean = |v: &[f64]| if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 };
    json!({
        "errors": errors,
        "ttft_p50_ms": percentile(ttfts.clone(), 0.50),
        "ttft_p90_ms": percentile(ttfts.clone(), 0.90),
        "ttft_p99_ms": percentile(ttfts.clone(), 0.99),
        "itl_p50_ms": percentile(itls.clone(), 0.50),
        "itl_p99_ms": percentile(itls.clone(), 0.99),
        "itl_mean_ms": mean(&itls),
        "wall_s": wall,
        "total_tokens": total_tokens,
        "aggregate_tps": total_tokens as f64 / wall,
    })
}

#[tokio::main]
async fn main() {
    let args = parse_args();

    if std::env::args().any(|a| a == "--flood") {
        return flood(&args).await;
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()
        .unwrap();
    let url = format!("{}/v1/chat/completions", args.url);
    if args.warmup {
        for _ in 0..4 {
            let _ = one_request(&client, &url, &args.model, 2).await;
        }
    }

    let mut runs: Vec<Value> = Vec::new();
    for _ in 0..args.repeat.max(1) {
        runs.push(streaming_once(&client, &args).await);
    }
    // 每项指标取各轮中位数
    let mut med = serde_json::Map::new();
    for (k, _) in runs[0].as_object().unwrap().clone() {
        let vals: Vec<f64> = runs.iter().filter_map(|r| r[&k].as_f64()).collect();
        med.insert(k, json!(median(vals)));
    }

    let report = json!({
        "mode": "streaming chat/completions",
        "meta": meta(),
        "config": {"concurrency": args.concurrency, "num": args.num, "max_tokens": args.max_tokens,
                    "repeat": args.repeat},
        "result": med,
        "runs": runs,
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if let Some(path) = &args.json {
        std::fs::write(path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
        eprintln!("report written to {path}");
    }

    // --smoke "ttft_p99_ms<=100;tps_min>=5000"：性能验收线，违例退出码非 0
    if let Some(spec) = &args.smoke {
        let r = &report["result"];
        let mut failed = Vec::new();
        for rule in spec.split(';') {
            let (key, op, val) = match rule.split_once("<=") {
                Some((k, v)) => (k.trim(), "<=", v.trim()),
                None => match rule.split_once(">=") {
                    Some((k, v)) => (k.trim(), ">=", v.trim()),
                    None => {
                        eprintln!("bad smoke rule: {rule}");
                        continue;
                    }
                },
            };
            let actual = match key {
                "ttft_p99_ms" => r["ttft_p99_ms"].as_f64(),
                "ttft_p90_ms" => r["ttft_p90_ms"].as_f64(),
                "itl_p99_ms" => r["itl_p99_ms"].as_f64(),
                "tps_min" => r["aggregate_tps"].as_f64(),
                "errors_max" => r["errors"].as_f64(),
                _ => {
                    eprintln!("unknown smoke key: {key}");
                    continue;
                }
            };
            let val: f64 = val.parse().unwrap();
            let ok = match (op, actual) {
                ("<=", Some(a)) => a <= val,
                (">=", Some(a)) => a >= val,
                _ => false,
            };
            println!("smoke  {key} {op} {val}: {} (actual {:?})", if ok { "PASS" } else { "FAIL" }, actual);
            if !ok {
                failed.push(key);
            }
        }
        if !failed.is_empty() {
            eprintln!("SMOKE FAILED: {failed:?}");
            std::process::exit(1);
        }
        println!("SMOKE PASSED");
    }
}
