//! wein-bench：并发性能验收工具（C 组用例的执行器）。
//! 今天打 mock（测 HTTP/SSE 栈自身的开销上界），M0 接真实引擎后打真实负载——
//! 指标口径不变：TTFT / ITL(TPOT) / 聚合吞吐。
//!
//! 用法：
//!   wein-bench --url http://127.0.0.1:8080 --model mock-model \
//!              --concurrency 100 --num 200 --max-tokens 32 \
//!              [--json report.json] [--smoke "ttft_p99_ms<=100;tps_min>=5000"]
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

fn percentile(mut v: Vec<f64>, p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((v.len() - 1) as f64 * p).round() as usize;
    v[idx]
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
    let results = futures::future::join_all(handles).await.into_iter().map(|h| h.unwrap()).collect::<Vec<_>>();
    let wall = wall0.elapsed().as_secs_f64();

    let lat: Vec<f64> = results.iter().map(|(_, ms)| *ms).collect();
    let errors = results.iter().filter(|(ok, _)| !ok).count();
    let qps = args.num as f64 / wall;
    let report = json!({
        "mode": "flood GET /v1/models",
        "config": {"concurrency": args.concurrency, "num": args.num},
        "result": {
            "errors": errors,
            "qps": qps,
            "latency_ms": {"p50": percentile(lat.clone(), 0.50), "p99": percentile(lat.clone(), 0.99)},
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

#[tokio::main]
async fn main() {
    let args = parse_args();

    // --flood：纯路由吞吐上限测量（GET /v1/models，不进引擎）。
    // 回答"router 本身能跑多少 QPS"——与流式 token 泵是两种工况。
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
    let tps = total_tokens as f64 / wall;

    let report = json!({
        "config": {"url": args.url, "model": args.model, "concurrency": args.concurrency,
                    "num": args.num, "max_tokens": args.max_tokens},
        "result": {
            "errors": errors,
            "ttft_ms": {"p50": percentile(ttfts.clone(), 0.50), "p90": percentile(ttfts.clone(), 0.90), "p99": percentile(ttfts.clone(), 0.99)},
            "itl_ms": {"p50": percentile(itls.clone(), 0.50), "p99": percentile(itls.clone(), 0.99), "mean": if itls.is_empty() { f64::NAN } else { itls.iter().sum::<f64>() / itls.len() as f64 }},
            "wall_s": wall,
            "total_tokens": total_tokens,
            "aggregate_tps": tps,
        }
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
                "ttft_p99_ms" => r["ttft_ms"]["p99"].as_f64(),
                "ttft_p90_ms" => r["ttft_ms"]["p90"].as_f64(),
                "itl_p99_ms" => r["itl_ms"]["p99"].as_f64(),
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
