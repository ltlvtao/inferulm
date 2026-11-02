#!/usr/bin/env python3
"""把 vLLM torch-profiler chrome trace 拆成 kernel 归桶表 + 逐步账。

要点：用 `gpu_user_annotation`（GPU 侧时间戳）切步——CPU 侧 `user_annotation`
在 CUDA graph 下只覆盖 launch（约 1.7ms），与 GPU 上真正跑的 9.5ms 错位。
kernel 归桶按名字（任务要求的口径）。
"""
import argparse
import bisect
import collections
import gzip
import json
import pathlib
import re

BUCKETS = [
    ("attn", re.compile(r"unified_attention|flash_attn|flash_fwd|varlen_fwd|fmha|paged_attn|reshape_and_cache|concat_and_cache|mha_|_attn_|attention|merge_attn|cache_kv", re.I)),
    ("mamba/GDN", re.compile(r"qwen_gdn|gdn|gated_delta|causal_conv1d|mamba|selective_scan|chunk_fwd_kernel|chunk_state|recompute_w_u|wy_fast|solve_tril|merge_16x16|chunk_gated", re.I)),
    ("lm_head", re.compile(r"lm_head|logits|logprob|sampling_logits", re.I)),
    ("rms_norm", re.compile(r"rms_norm|layernorm|layer_norm|fused_add_rms|_norm", re.I)),
    ("rope", re.compile(r"rotary|rope", re.I)),
    ("sampling", re.compile(r"sampl|topk|top_k|top_p|multinomial|gumbel|exponential|random|penalt|argmax", re.I)),
    ("quant/dequant", re.compile(r"quant|dequant|scaled_mm|per_token_|awq|gptq|fp8", re.I)),
    ("memcpy/memset", None),
    ("gemm", re.compile(r"cutlass|cublas|gemm|gemv|nvjet|ampere_|hopper_|bgemm|matmul|deep_gemm|_group_mm|s16816|s1688|Kernel2|splitKreduce|gemvx", re.I)),
    ("elementwise/act", re.compile(r".", re.S)),
]


def bucket_of(name, cat, lmhead_kernel=False):
    if cat in ("gpu_memcpy", "gpu_memset") or "Memcpy" in name or "Memset" in name or name.startswith("memcpy") or name.startswith("memset"):
        return "memcpy/memset"
    if lmhead_kernel:
        return "lm_head"
    for b, rx in BUCKETS:
        if rx is not None and rx.search(name):
            return b
    return "其它"


STEP_RX = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")


def load_trace(path):
    p = pathlib.Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--label", default="scene")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--topk", type=int, default=20)
    a = ap.parse_args()

    ev = load_trace(a.trace)["traceEvents"]
    kerns = sorted([e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")], key=lambda x: x["ts"])
    wins = sorted([e for e in ev if e.get("cat") == "gpu_user_annotation"], key=lambda x: x["ts"])
    steps = []
    for e in wins:
        m = STEP_RX.match(e["name"])
        if not m:
            continue
        steps.append({
            "ts": e["ts"], "dur": e.get("dur", 0.0),
            "ctx_reqs": int(m.group(1)), "ctx_tokens": int(m.group(2)),
            "gen_reqs": int(m.group(3)), "gen_tokens": int(m.group(4)),
        })

    # 归属：kernel 落到「最近的前一个步边界」——CUDA graph 之外的 eager kernel
    # （lm_head GEMV 就落在窗口结束之后 ~7us）也算进那一步，这样步内账才完整。
    kts = [k["ts"] for k in kerns]
    wts = [s["ts"] for s in steps]
    owner = [None] * len(kerns)
    for j, k in enumerate(kerns):
        i = bisect.bisect_right(wts, k["ts"]) - 1
        owner[j] = i if i >= 0 else None

    # lm_head：每步一次的大 GEMM/GEMV（1 x 2560 @ 2560 x 248320 → 1.27GB → ~1.3-1.7ms）。
    # 判据：单个 kernel >1.1ms 且属 gemm 族，且这种"大 kernel"总数不超过步数的 3 倍
    #（每步至多一次；prefill 步里的 837us GEMM 因此不会被误判）。
    ns = max(1, len(steps))
    cand = [i for i, k in enumerate(kerns)
            if k.get("dur", 0.0) > 1100.0
            and re.search(r"gemv|gemm|cublas|cutlass", k["name"], re.I)
            and k["name"].startswith(("std::enable_if", "void cutlass", "ampere", "nvjet"))]
    lmh = set(cand) if 0 < len(cand) <= 3 * ns else set()

    def summarize(idx_list, title):
        tot = 0.0
        bb = collections.defaultdict(float)
        bn = collections.defaultdict(lambda: [0, 0.0])
        for i in idx_list:
            k = kerns[i]
            d = k.get("dur", 0.0)
            tot += d
            bb[bucket_of(k["name"], k["cat"], i in lmh)] += d
            s = bn[k["name"]]
            s[0] += 1
            s[1] += d
        print(f"\n### {title} — GPU 时间 {tot/1000:.2f} ms")
        for b, t in sorted(bb.items(), key=lambda x: -x[1]):
            print(f"  {b:16s} {t/1000:9.3f} ms  {100*t/tot:5.1f}%")
        print(f"\n  top-{a.topk} kernel:")
        for i, (n, (c, t)) in enumerate(sorted(bn.items(), key=lambda x: -x[1][1])[: a.topk], 1):
            print(f"   {i:2d}. {t/1000:9.3f}ms {100*t/tot:5.1f}%  n={c:6d} avg={t/c:8.1f}us  {n[:88]}")
        return tot, dict(bb)

    all_idx = list(range(len(kerns)))
    tot_all, bb_all = summarize(all_idx, f"{a.label} 全窗口")

    gen_idx = [i for i in all_idx if owner[i] is not None and steps[owner[i]]["ctx_tokens"] == 0]
    ctx_idx = [i for i in all_idx if owner[i] is not None and steps[owner[i]]["ctx_tokens"] > 0]
    tot_gen = sum(kerns[i].get("dur", 0) for i in gen_idx) / 1000
    tot_ctx = sum(kerns[i].get("dur", 0) for i in ctx_idx) / 1000
    print(f"\n# 归属：gen 步 {tot_gen:.2f}ms / ctx 步 {tot_ctx:.2f}ms / 未归属 {(tot_all-tot_gen-tot_ctx):.2f}ms")

    res = {"label": a.label, "total_ms": tot_all / 1000, "buckets_all": bb_all, "buckets_gen": {}, "buckets_ctx": {}}
    if gen_idx:
        tg, bg = summarize(gen_idx, f"{a.label} 仅 decode(gen) 步")
        res["buckets_gen"] = bg
        res["gen_ms"] = tg / 1000
    if ctx_idx:
        tc, bc = summarize(ctx_idx, f"{a.label} 仅 prefill(ctx) 步")
        res["buckets_ctx"] = bc
        res["ctx_ms"] = tc / 1000

    print(f"\n### 逐步（{len(steps)} 步）")
    rows = []
    own_sum = collections.defaultdict(float)
    own_n = collections.Counter()
    for j, k in enumerate(kerns):
        if owner[j] is not None:
            own_sum[owner[j]] += k.get("dur", 0.0)
            own_n[owner[j]] += 1
    for i, s in enumerate(steps):
        nxt = steps[i + 1]["ts"] if i + 1 < len(steps) else s["ts"] + s["dur"] + 500
        cad = (nxt - s["ts"]) / 1000.0
        ksum = own_sum[i] / 1000.0
        rows.append((cad, ksum, cad - ksum, s["ctx_tokens"], s["gen_reqs"], own_n[i]))
    for ph, pred in (("gen(纯 decode 步)", lambda r: r[3] == 0), ("ctx(prefill 步)", lambda r: r[3] > 0), ("全部", lambda r: True)):
        sel = [r for r in rows if pred(r)]
        if not sel:
            continue
        n = len(sel)
        ms = sum(r[0] for r in sel) / n
        ks = sum(r[1] for r in sel) / n
        print(f"  {ph:18s} n={n:4d} step(cadence)={ms:8.2f}ms kernel={ks:8.2f}ms 空洞={ms-ks:7.2f}ms ({100*(ms-ks)/ms:5.1f}%)  kernels/step={sum(r[5] for r in sel)/n:.0f}")
    g = [r for r in rows if r[3] == 0]
    c = [r for r in rows if r[3] > 0]
    res["steps"] = {
        "n": len(rows),
        "gen": {"n": len(g), "step_ms": sum(r[0] for r in g) / max(1, len(g)), "kernel_ms": sum(r[1] for r in g) / max(1, len(g))},
        "ctx": {"n": len(c), "step_ms": sum(r[0] for r in c) / max(1, len(c)), "kernel_ms": sum(r[1] for r in c) / max(1, len(c))},
    }
    if a.json_out:
        pathlib.Path(a.json_out).write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
