#!/usr/bin/env python3
"""前缀缓存 TTFT 探针：同一长前缀的两次请求（A 冷、B 热），量 TTFT。

用法：python3 bench/prefix_probe.py <url> <model> [前缀词数]
输出：A/B 的 TTFT、总时长、首段文本。
"""
import http.client
import json
import sys
import time
import urllib.parse

url = sys.argv[1]
model = sys.argv[2]
n_words = int(sys.argv[3]) if len(sys.argv) > 3 else 130

par = urllib.parse.urlparse(url)
conn_kw = {"host": par.hostname, "port": par.port or 80}

# 长前缀：可复用段落（约 1.35 词/token）
para = (
    "In the autumn of the year the survey team mapped the northern valley, "
    "recording every stream, ridge and forest boundary in a ledger that would "
    "later be copied by hand into three separate archives kept in different "
    "cities, so that no single fire could destroy the record of the land. "
)
prefix = ""
while len(prefix.split()) < n_words:
    prefix += para


def stream_ttft(prompt, max_tokens=8):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
        }
    )
    c = http.client.HTTPConnection(**conn_kw, timeout=120)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse()
    ttft = None
    text = ""
    for raw in r:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            d = json.loads(payload)
        except json.JSONDecodeError:
            continue
        ch = d.get("choices") or []
        piece = ""
        if ch:
            delta = ch[0].get("delta") or {}
            piece = delta.get("content") or ""
        if piece and ttft is None:
            ttft = time.time() - t0
        text += piece
    total = time.time() - t0
    c.close()
    return ttft, total, text


qa = prefix + "\n\nQuestion A: name the three archives in one short sentence."
qb = prefix + "\n\nQuestion B: name the survey team in one short sentence."

# 预热：首请求含一次性惰性初始化（实测 ~145ms），不能算进 A
stream_ttft("warmup: say hi", max_tokens=2)

for tag, prompt in (("A_cold", qa), ("B_warm", qb)):
    ttft, total, text = stream_ttft(prompt)
    print(
        f"{tag}: ttft={ttft * 1000:.1f}ms total={total * 1000:.0f}ms text={text[:60]!r}"
    )
