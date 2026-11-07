#!/usr/bin/env python3
"""Agent 场景端到端压测：多轮对话 × 工具调用 × 长上下文增长。

为什么单独一个脚本：现有 wein-bench 量的是"单发 chat、固定 prompt"的瞬时吞吐，
而 Agent 负载的形态是——会话内多轮、每轮先调工具、把结果灌回上下文、再问，
上下文逐轮变长（前缀缓存能不能吃到决定成败）。这里就用官方 `openai` SDK
（Agent 框架的事实标准）跑真实调用序列：顺带把"标准客户端能不能用"验掉
（SDK 的 pydantic 会对响应做严格校验：finish_reason / tool_calls / delta 形状）。

用法：
  python bench/agent_bench.py --url http://127.0.0.1:8193 --model <id> \
      --sessions 4 --turns 6 --payload 512 --json out.json

口径：
  * TTFT = 首个非空 delta（正文或思维链）到达时间；ITL = 相邻此类 delta 的间隔
  * tool_ok = 该轮 `finish_reason == "tool_calls"` 且参数是合法 JSON
  * 前缀命中率 = /metrics 的 wein_prefix_hit_tokens_total 增量 / prompt token 增量
"""
import argparse
import asyncio
import json
import random
import statistics
import time
import urllib.request

from openai import AsyncOpenAI

# 一个"像样的" Agent 工具集：查库、跑命令、写文件
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the internal documentation index and return matching snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keywords"},
                    "top_k": {"type": "integer", "description": "How many snippets to return"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command in the project sandbox and return stdout.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}, "timeout_s": {"type": "integer"}},
                "required": ["cmd"],
            },
        },
    },
]

SYSTEM = (
    "You are a coding agent working inside a repository. "
    "You must call a tool before answering: use search_docs to look things up or "
    "run_shell to inspect the project. Keep prose short."
)

TASKS = [
    "Find where the KV cache page table is allocated, then summarise it.",
    "Check the release profile settings in Cargo.toml and report them.",
    "Locate the attention kernel entrypoint and describe its tiling.",
    "Find the batch scheduler's token budget logic and summarise it.",
    "Inspect the bench scripts and report which scenarios they cover.",
    "Find the prefix cache implementation and describe its eviction policy.",
]

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
         "omicron pi rho sigma tau upsilon phi chi psi omega").split()


def filler_tokens(n, seed):
    """确定性伪文本（模拟工具输出：日志/文档片段），按空白切分≈token 数"""
    rnd = random.Random(seed)
    return " ".join(rnd.choice(WORDS) + str(rnd.randrange(10000)) for _ in range(n))


async def metrics(url):
    """读 /metrics 里关心的三个累计量（拿不到就返回 None）"""
    def get():
        try:
            with urllib.request.urlopen(f"{url}/metrics", timeout=5) as r:
                txt = r.read().decode()
        except Exception:
            return None
        out = {}
        for line in txt.splitlines():
            # vLLM 侧同名语义的指标（它能给 prefix cache 命中 token 数）
            if line.startswith("vllm:prompt_tokens_total"):
                out["prompt"] = float(line.rsplit(" ", 1)[1]); continue
            if line.startswith("vllm:generation_tokens_total"):
                out["completion"] = float(line.rsplit(" ", 1)[1]); continue
            if line.startswith("vllm:prefix_cache_hits_total"):
                out["hit"] = float(line.rsplit(" ", 1)[1]); continue
            if line.startswith("wein_tokens_total"):
                kind = "prompt" if 'kind="prompt"' in line else "completion"
                out[kind] = float(line.rsplit(" ", 1)[1])
            elif line.startswith("wein_prefix_hit_tokens_total"):
                out["hit"] = float(line.rsplit(" ", 1)[1])
        return out
    return await asyncio.to_thread(get)


async def one_turn(client, model, messages, max_tokens, tools):
    """一轮：流式取完，返回 (ttft_ms, itls_ms, n_tok, finish_reason, tool_calls, reasoning_len)"""
    t0 = time.perf_counter()
    stream = await client.chat.completions.create(
        model=model, messages=messages, tools=tools, max_tokens=max_tokens,
        temperature=0, stream=True,
    )
    ttft = None
    last = None
    itls = []
    n_tok = 0
    reason = []
    text = []
    calls = {}          # index -> {"id","name","arguments"}
    finish = None
    async for ch in stream:
        if not ch.choices:
            continue
        c = ch.choices[0]
        if c.finish_reason:
            finish = c.finish_reason
        d = c.delta
        if d is None:
            continue
        # 推理片段：wein 用 `reasoning_content`（vLLM 同名字段的旧形态），
        # vLLM 0.32 的 reasoning parser 吐的是 `reasoning`——两个都认
        ex = d.model_extra or {}
        has = bool(d.content) or bool(ex.get("reasoning_content")) or bool(ex.get("reasoning"))
        rc = ex.get("reasoning_content") or ex.get("reasoning")
        if rc:
            reason.append(rc)
        if has:
            now = time.perf_counter()
            if ttft is None:
                ttft = (now - t0) * 1e3
            elif last is not None:
                itls.append((now - last) * 1e3)
            last = now
            n_tok += 1
        if d.content:
            text.append(d.content)
        for tc in (d.tool_calls or []):
            slot = calls.setdefault(tc.index, {"id": None, "name": None, "args": ""})
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                slot["args"] += tc.function.arguments
    return {
        "ttft_ms": ttft,
        "itls_ms": itls,
        "n_tok": n_tok,
        "finish": finish,
        "calls": [calls[k] for k in sorted(calls)],
        "content": "".join(text),
        "reasoning": "".join(reason),
        "wall_ms": (time.perf_counter() - t0) * 1e3,
    }


async def session(client, model, sid, args, out):
    """一个 Agent 会话：T 轮，每轮工具调用 → 结果回灌 → 下一问"""
    rnd = random.Random(1000 + sid)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": TASKS[sid % len(TASKS)]},
    ]
    for turn in range(1, args.turns + 1):
        try:
            r = await one_turn(client, model, messages, args.max_tokens, TOOLS)
        except Exception as e:
            # 上下文/服务端拒绝不该炸掉整个压测：记一行失败、该会话收工
            out.append({"session": sid, "turn": turn, "ttft_ms": None, "itl_ms": None,
                        "n_tok": 0, "finish": "error", "tool_ok": False, "n_calls": 0,
                        "content_chars": 0, "reasoning_chars": 0, "wall_ms": 0,
                        "error": str(e)[:200]})
            return
        tool_ok = bool(r["calls"]) and r["finish"] == "tool_calls"
        for c in r["calls"]:
            try:
                json.loads(c["args"] or "{}")
            except Exception:
                tool_ok = False
        out.append({
            "session": sid,
            "turn": turn,
            "ttft_ms": r["ttft_ms"],
            "itl_ms": statistics.median(r["itls_ms"]) if r["itls_ms"] else None,
            "n_tok": r["n_tok"],
            "finish": r["finish"],
            "tool_ok": tool_ok,
            "n_calls": len(r["calls"]),
            "content_chars": len(r["content"]),
            "reasoning_chars": len(r["reasoning"]),
            "wall_ms": r["wall_ms"],
        })
        # 回灌：assistant（含调用）+ 工具结果（带一段工具输出，撑长上下文）
        amsg = {"role": "assistant", "content": r["content"] or None}
        if r["reasoning"]:
            amsg["reasoning_content"] = r["reasoning"]
        if r["calls"]:
            amsg["tool_calls"] = [
                {"id": c["id"] or f"call_{sid}_{turn}_{i}", "type": "function",
                 "function": {"name": c["name"] or "search_docs", "arguments": c["args"] or "{}"}}
                for i, c in enumerate(r["calls"])
            ]
        messages.append(amsg)
        for i, c in enumerate(r["calls"]):
            messages.append({
                "role": "tool",
                "tool_call_id": c["id"] or f"call_{sid}_{turn}_{i}",
                "content": filler_tokens(args.payload, rnd.randrange(1 << 30)),
            })
        # 下一轮：追问，逼模型再调一次工具（Agent 的典型循环）
        messages.append({"role": "user",
                         "content": f"Turn {turn}: keep investigating — call another tool now."})


async def run(args):
    client = AsyncOpenAI(base_url=f"{args.url}/v1", api_key="wein", max_retries=0, timeout=600)
    before = await metrics(args.url)
    rows = []
    t0 = time.perf_counter()
    await asyncio.gather(*[session(client, args.model, i, args, rows) for i in range(args.sessions)])
    wall = time.perf_counter() - t0
    after = await metrics(args.url)

    rows.sort(key=lambda r: (r["turn"], r["session"]))
    cold = [r for r in rows if r["turn"] == 1]
    hot = [r for r in rows if r["turn"] > 1]

    def rnd(v, n=1):
        """None 安全的中位数（有会话在早期轮次就失败时 median 会是空集）"""
        return None if v is None else round(v, n)

    def agg(rs, key):
        vs = [r[key] for r in rs if r.get(key) is not None]
        return (statistics.median(vs) if vs else None,
                max(vs) if vs else None)

    def p95(vs):
        """最近秩 p95：n 小时返回最大值（旧式 int(0.95*(n-1)) 会取到最小值）"""
        vs = sorted(vs)
        if not vs:
            return None
        return vs[max(0, -(-95 * len(vs) // 100) - 1)]

    turns = len(rows)
    total_tok = sum(r["n_tok"] for r in rows)
    hit_rate = None
    tok_per_s = None
    if before and after:
        dp = after.get("prompt", 0) - before.get("prompt", 0)
        dh = after.get("hit", 0) - before.get("hit", 0)
        dc = after.get("completion", 0) - before.get("completion", 0)
        if dp > 0:
            hit_rate = dh / dp
        if dc > 0:
            tok_per_s = dc / wall

    summary = {
        "sessions": args.sessions,
        "turns_per_session": args.turns,
        "turns": turns,
        "payload_tokens": args.payload,
        "max_tokens": args.max_tokens,
        "wall_s": round(wall, 3),
        "turns_per_s": round(turns / wall, 3),
        "tok_per_s": round(tok_per_s, 1) if tok_per_s else None,
        "tool_ok_rate": round(sum(1 for r in rows if r["tool_ok"]) / max(1, turns), 4),
        "turn_errors": sum(1 for r in rows if r["finish"] == "error"),
        "ttft_cold_ms": rnd(agg(cold, "ttft_ms")[0]) if cold else None,
        "ttft_cold_p95_ms": rnd(p95([r["ttft_ms"] for r in cold if r["ttft_ms"]])) if cold else None,
        "ttft_hot_ms": rnd(agg(hot, "ttft_ms")[0]) if hot else None,
        "ttft_hot_p95_ms": rnd(p95([r["ttft_ms"] for r in hot if r["ttft_ms"]])) if hot else None,
        "itl_med_ms": rnd(agg(rows, "itl_ms")[0], 2) if rows else None,
        "itl_p95_ms": rnd(p95([r["itl_ms"] for r in rows if r["itl_ms"]]), 2) if rows else None,
        "prefix_hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
        "prompt_tokens": (after or {}).get("prompt") and round(after["prompt"] - before["prompt"]),
        "completion_tokens": (after or {}).get("completion") and round(after["completion"] - before["completion"]),
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"summary": summary, "turns": rows}, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    # 每轮（含跨会话聚合）的 TTFT：看上下文变长 + 前缀命中的效果
    print("\n轮次 | TTFT p50 / p95 (ms) | ITL p50 (ms) | 工具成功率 | 错误")
    for turn in range(1, args.turns + 1):
        rs = [r for r in rows if r["turn"] == turn]
        if not rs:
            continue
        tt = [r["ttft_ms"] for r in rs if r["ttft_ms"]]
        it = [r["itl_ms"] for r in rs if r["itl_ms"]]
        ok = sum(1 for r in rs if r["tool_ok"]) / len(rs)
        err = sum(1 for r in rs if r["finish"] == "error")
        t50 = f"{statistics.median(tt):>9.1f}" if tt else "        -"
        t95 = f"{p95(tt):>9.1f}" if tt else "        -"
        i50 = f"{statistics.median(it):>12.2f}" if it else "           -"
        print(f"{turn:>4} | {t50} / {t95} | {i50} | {ok:>6.0%} | {err}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8193")
    ap.add_argument("--model", required=True)
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--payload", type=int, default=512, help="每个工具结果的伪 token 数")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--json")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
