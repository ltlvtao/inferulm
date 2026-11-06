#!/usr/bin/env python3
"""c=100 稳态 decode 负载 + vLLM 自带 torch profiler 开关（只测量）。

100 个线程同时打 /v1/completions（prompt="1\\n"，stream，ignore_eos，max_tokens 大），
主线程在预热 warmup_s 后 POST /start_profile，窗口 window_s 后 /stop_profile。
每个请求记录首 token 与后续 token 到达时刻 → ITL 分布与稳态聚合吞吐。
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request

import requests


def one(idx, url, model, prompt, max_tokens, out, barrier, stop_ev):
    times = []
    try:
        r = requests.post(
            url,
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "ignore_eos": True,
                "stream": True,
            },
            stream=True,
            timeout=300,
        )
        if r.status_code != 200:
            out[idx] = {"error": f"status {r.status_code} {r.text[:200]}"}
            return
        rec = None
        barrier.wait()
        t_first = None
        for line in r.iter_lines(chunk_size=1):
            if stop_ev.is_set():
                break
            if not line or not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                break
            try:
                rec = json.loads(data)
            except Exception:
                continue
            ch = rec.get("choices") or []
            if not ch:
                continue
            txt = ch[0].get("text") or ""
            if txt:
                t = time.time()
                times.append(t)
                if t_first is None:
                    t_first = t
        out[idx] = {
            "t_first": t_first,
            "t_last": times[-1] if times else None,
            "times": times,
            "tokens": len(times),
        }
    except Exception as e:  # noqa: BLE001
        out[idx] = {"error": repr(e)}


def pctl(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def profile(path, prefix=None, iters=None):
    body = {}
    if prefix:
        body["profile_prefix"] = prefix
    if iters:
        body["max_iterations"] = iters
    req = urllib.request.Request(
        "http://127.0.0.1:%s/%s" % (PORT, path),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read()[:200].decode(errors="ignore")
    except Exception as e:  # noqa: BLE001
        return None, repr(e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--prompt", default="1\n")
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--window-s", type=float, default=4.0)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global PORT
    PORT = a.port
    url = "http://127.0.0.1:%d/v1/completions" % a.port
    model = urllib.request.urlopen(
        "http://127.0.0.1:%d/v1/models" % a.port, timeout=10
    ).read().decode()
    model = json.loads(model)["data"][0]["id"]

    out = {}
    barrier = threading.Barrier(a.n + 1)
    stop_ev = threading.Event()
    ths = [
        threading.Thread(
            target=one,
            args=(i, url, model, a.prompt, a.max_tokens, out, barrier, stop_ev),
            daemon=True,
        )
        for i in range(a.n)
    ]
    t0 = time.time()
    for t in ths:
        t.start()
    try:
        barrier.wait(timeout=120)
    except threading.BrokenBarrierError:
        print("barrier broken (a request failed early)")
    print("all %d requests in flight at t=%.2fs" % (a.n, time.time() - t0), flush=True)

    prof_txt = []
    if a.profile:
        time.sleep(a.warmup_s)
        st, body = profile("start_profile", a.prefix)
        print("start_profile %s %s at t=%.2fs" % (st, body, time.time() - t0), flush=True)
        prof_txt.append(("start", time.time()))
        time.sleep(a.window_s)
        st, body = profile("stop_profile")
        print("stop_profile %s %s at t=%.2fs" % (st, body, time.time() - t0), flush=True)
        prof_txt.append(("stop", time.time()))

    for t in ths:
        t.join()
    wall = time.time() - t0

    errs = [v["error"] for v in out.values() if "error" in v]
    ok = [v for v in out.values() if "error" not in v and v.get("tokens")]
    # 聚合 ITL（相邻 token 间隔，跨所有请求）
    itls = []
    for v in ok:
        ts = v["times"]
        itls.extend((b - a_) * 1e3 for a_, b in zip(ts, ts[1:]))
    firsts = [v["t_first"] for v in ok if v["t_first"]]
    lasts = [v["t_last"] for v in ok if v["t_last"]]
    tot_tok = sum(v["tokens"] for v in ok)
    span = max(lasts) - min(firsts) if lasts else 0.0
    ttfts = [(v["t_first"] - t0) * 1e3 for v in ok if v["t_first"]]
    res = {
        "n": a.n,
        "max_tokens": a.max_tokens,
        "requests_ok": len(ok),
        "errors": errs[:3],
        "wall_s": round(wall, 3),
        "tokens_total": tot_tok,
        "agg_tps_span": round(tot_tok / span, 1) if span else None,
        "agg_tps_wall": round(tot_tok / wall, 1) if wall else None,
        "itl_p50_ms": round(pctl(itls, 0.5), 3) if itls else None,
        "itl_p99_ms": round(pctl(itls, 0.99), 3) if itls else None,
        "itl_mean_ms": round(statistics.fmean(itls), 3) if itls else None,
        "ttft_p50_ms": round(pctl(ttfts, 0.5), 2) if ttfts else None,
        "tokens_per_req_mean": round(tot_tok / len(ok), 1) if ok else None,
        "profile": [(k, round(t - t0, 3)) for k, t in prof_txt],
    }
    print(json.dumps(res, indent=1), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
