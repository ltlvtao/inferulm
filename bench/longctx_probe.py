#!/usr/bin/env python3
"""长上下文单请求探针：TTFT（含 prefill）与首 N 个 token 的 ITL。

与 bench/prefix_probe.py 的区别：prompt 用 tokenizer 精确数 token（不是字符/词数），
冷/热两组用同一 prompt（冷跑前调 /reset_prefix_cache，需 VLLM_SERVER_DEV_MODE=1），
贪心（temperature=0），记录机器可读 JSON + 每次请求的原始时序。

用法：
  longctx_probe.py --url http://127.0.0.1:8155 --model <dir> \
      --tokens 3000 --max-tokens 32 --repeats 3 \
      --cold --hot --out bench-reports/vs-vllm-longctx/default/long-3000.json
"""
import argparse
import http.client
import json
import statistics
import time
import urllib.parse

PARA = (
    "In the autumn of the year the survey team mapped the northern valley, "
    "recording every stream, ridge and forest boundary in a ledger that would "
    "later be copied by hand into three separate archives kept in different "
    "cities, so that no single fire could destroy the record of the land. "
    "The cartographers disagreed about the eastern marsh, and their notes "
    "remain the only account of how the drainage channels were later moved. "
)
TAIL = "\n\nSummarize the passage above in one sentence.\n"


def build_prompt(tok, target):
    """纯文本段落填充 → 用 tokenizer 数到 target 附近（±0）后截断。"""
    text = ""
    while len(tok.encode(text, add_special_tokens=False)) < target - len(
        tok.encode(TAIL, add_special_tokens=False)
    ):
        text += PARA
    body = tok.encode(text, add_special_tokens=False)
    tail = tok.encode(TAIL, add_special_tokens=False)
    body = body[: target - len(tail)]
    return tok.decode(body) + TAIL


def http_post(conn_kw, path, body, timeout=180):
    c = http.client.HTTPConnection(**conn_kw, timeout=timeout)
    c.request("POST", path, body, {"Content-Type": "application/json"})
    return c, c.getresponse()


def stream_once(conn_kw, model, prompt, max_tokens):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )
    t0 = time.perf_counter()
    c, r = http_post(conn_kw, "/v1/chat/completions", body)
    if r.status != 200:
        raise SystemExit(f"HTTP {r.status}: {r.read()[:400]!r}")
    stamps = []
    usage = None
    for raw in r:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        d = json.loads(payload)
        if d.get("usage"):
            usage = d["usage"]
        for ch in d.get("choices") or []:
            piece = (ch.get("delta") or {}).get("content") or ""
            # 每个 SSE 块一个 token（vLLM 默认）；非空文本才计到达时间
            if piece:
                stamps.append(time.perf_counter() - t0)
    c.close()
    if not stamps:
        raise SystemExit("无 token 到达")
    ttft = stamps[0] * 1000
    itls = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]
    return {
        "ttft_ms": ttft,
        "n_tokens": len(stamps),
        "itl_ms": itls,
        "itl_p50_ms": statistics.median(itls) if itls else None,
        "itl_mean_ms": statistics.fmean(itls) if itls else None,
        "wall_ms": stamps[-1] * 1000,
        "usage": usage,
    }


def scrape_metrics(conn_kw):
    try:
        c = http.client.HTTPConnection(**conn_kw, timeout=30)
        c.request("GET", "/metrics")
        r = c.getresponse()
        txt = r.read().decode("utf-8", "ignore")
        c.close()
    except Exception as e:  # 指标非测量轴，缺了不影响 TTFT
        return {"error": str(e)}
    keep = (
        "prefix_cache_hits_total",
        "prefix_cache_queries_total",
        "num_requests_waiting",
        "prompt_tokens_total",
        "generation_tokens_total",
    )
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for k in keep:
            if k in line:
                val = line.rsplit(" ", 1)[-1]
                out.setdefault(k, 0.0)
                try:
                    out[k] += float(val)
                except ValueError:
                    pass
    return out


def reset_prefix_cache(conn_kw):
    c, r = http_post(conn_kw, "/reset_prefix_cache", "{}")
    ok = json.loads(r.read() or b"{}").get("success", False)
    c.close()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None, help="默认用 --model 目录")
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--hot", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    par = urllib.parse.urlparse(a.url)
    conn_kw = {"host": par.hostname, "port": par.port or 80}

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    prompt = build_prompt(tok, a.tokens)
    n_prompt_tok_local = len(tok.encode(prompt))

    rec = {
        "ts_unix": int(time.time()),
        "config": {
            "target_tokens": a.tokens,
            "prompt_tokens_local": n_prompt_tok_local,
            "max_tokens": a.max_tokens,
            "repeats": a.repeats,
            "temperature": 0,
            "endpoint": "/v1/chat/completions (stream, include_usage)",
            "cold": a.cold,
            "hot": a.hot,
            "note": a.note,
        },
        "cold": [],
        "hot": [],
    }
    first = 0.0
    for i in range(a.repeats):
        if a.cold:
            ok = reset_prefix_cache(conn_kw)
            m0 = scrape_metrics(conn_kw)
            r = stream_once(conn_kw, a.model, prompt, a.max_tokens)
            m1 = scrape_metrics(conn_kw)
            r["reset_prefix_cache"] = ok
            r["metrics_delta"] = {
                k: round(m1.get(k, 0.0) - m0.get(k, 0.0), 3) for k in m0
            }
            rec["cold"].append(r)
            first = r["ttft_ms"]
            print(f"[cold {i+1}/{a.repeats}] ttft={r['ttft_ms']:.1f}ms "
                  f"itl_p50={r['itl_p50_ms']:.2f}ms tokens={r['n_tokens']} "
                  f"prompt_tokens={(r['usage'] or {}).get('prompt_tokens')}")
        if a.hot:
            # 紧接一次同 prompt：首块必命中（上一条刚把前缀写进 cache）
            m0 = scrape_metrics(conn_kw)
            r = stream_once(conn_kw, a.model, prompt, a.max_tokens)
            m1 = scrape_metrics(conn_kw)
            r["metrics_delta"] = {
                k: round(m1.get(k, 0.0) - m0.get(k, 0.0), 3) for k in m0
            }
            r["ref_cold_ttft_ms"] = round(first, 3)
            rec["hot"].append(r)
            print(f"[hot  {i+1}/{a.repeats}] ttft={r['ttft_ms']:.1f}ms "
                  f"itl_p50={r['itl_p50_ms']:.2f}ms tokens={r['n_tokens']} "
                  f"prompt_tokens={(r['usage'] or {}).get('prompt_tokens')}")

    def med(key):
        vals = [r[key] for r in rec["cold"] + rec["hot"] if r.get(key) is not None]
        return statistics.median(vals) if vals else None

    rec["median"] = {
        "cold_ttft_ms": statistics.median([r["ttft_ms"] for r in rec["cold"]])
        if rec["cold"] else None,
        "cold_itl_p50_ms": statistics.median(
            [r["itl_p50_ms"] for r in rec["cold"] if r["itl_p50_ms"] is not None]
        ) if rec["cold"] else None,
        "hot_ttft_ms": statistics.median([r["ttft_ms"] for r in rec["hot"]])
        if rec["hot"] else None,
        "hot_itl_p50_ms": statistics.median(
            [r["itl_p50_ms"] for r in rec["hot"] if r["itl_p50_ms"] is not None]
        ) if rec["hot"] else None,
    }
    rec["prompt_head"] = prompt[:120]
    import pathlib

    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    print(json.dumps(rec["median"], ensure_ascii=False))


if __name__ == "__main__":
    main()
