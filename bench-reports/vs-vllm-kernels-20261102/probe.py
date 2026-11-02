#!/usr/bin/env python3
"""单请求 + vLLM torch profiler 窗口（场景 B prefill / C decode）。

只测量：start_profile → 发一条 6009 token 的流式请求 → stop_profile。
delay/max iterations 让窗口只落在想要的那一段 worker step 上。
"""
import argparse
import http.client
import json
import statistics
import time
import pathlib
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
    text = ""
    while len(tok.encode(text, add_special_tokens=False)) < target - len(
        tok.encode(TAIL, add_special_tokens=False)
    ):
        text += PARA
    body = tok.encode(text, add_special_tokens=False)
    tail = tok.encode(TAIL, add_special_tokens=False)
    body = body[: target - len(tail)]
    return tok.decode(body) + TAIL


def post(conn_kw, path, body, timeout=300):
    c = http.client.HTTPConnection(**conn_kw, timeout=timeout)
    c.request("POST", path, body, {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


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
    c = http.client.HTTPConnection(**conn_kw, timeout=300)
    c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raise SystemExit(f"HTTP {r.status}: {r.read()[:400]!r}")
    stamps, usage = [], None
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
            if (ch.get("delta") or {}).get("content"):
                stamps.append(time.perf_counter() - t0)
    c.close()
    itls = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]
    return {
        "ttft_ms": stamps[0] * 1000,
        "n_tokens": len(stamps),
        "itl_ms": itls,
        "itl_p50_ms": statistics.median(itls) if itls else None,
        "itl_mean_ms": statistics.fmean(itls) if itls else None,
        "wall_ms": stamps[-1] * 1000,
        "usage": usage,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--mode", choices=["prefill", "decode"], required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--delay", type=int, default=0)
    ap.add_argument("--max-iters", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    par = urllib.parse.urlparse(a.url)
    conn_kw = {"host": par.hostname, "port": par.port or 80}

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    prompt = build_prompt(tok, a.tokens)
    n_local = len(tok.encode(prompt))

    st, _ = post(conn_kw, "/reset_prefix_cache", "{}")
    print(f"reset_prefix_cache -> {st}", flush=True)

    req = {"profile_prefix": a.prefix}
    if a.delay:
        req["delay_iterations"] = a.delay
    if a.max_iters:
        req["max_iterations"] = a.max_iters
    st, body = post(conn_kw, "/start_profile", json.dumps(req))
    print(f"start_profile {req} -> {st} {body[:200]!r}", flush=True)

    time.sleep(0.3)
    r = stream_once(conn_kw, a.model, prompt, a.max_tokens)
    print(
        f"[{a.mode}] ttft={r['ttft_ms']:.1f}ms itl_p50={(r['itl_p50_ms'] or 0):.2f}ms "
        f"tokens={r['n_tokens']} prompt_tokens={(r['usage'] or {}).get('prompt_tokens')}",
        flush=True,
    )
    # 让 profiler 自然走完 max_iterations 并写盘
    time.sleep(6)
    st, body = post(conn_kw, "/stop_profile", "{}")
    print(f"stop_profile -> {st} {body[:200]!r}", flush=True)
    time.sleep(2)

    rec = {
        "ts_unix": int(time.time()),
        "mode": a.mode,
        "profile_prefix": a.prefix,
        "delay_iterations": a.delay,
        "max_iterations": a.max_iters,
        "prompt_tokens_local": n_local,
        "result": r,
        "note": a.note,
    }
    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
