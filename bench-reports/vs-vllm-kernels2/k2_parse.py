#!/usr/bin/env python3
"""vLLM c=100 稳态 decode 的 torch-profiler 逐 kernel 账。

切步：用 GPU 侧 `gpu_user_annotation`（TSG 域时间戳，与 kernel 同域），
名字 `execute_context_0(0)_generation_100(100)` = 0 个 context token、100 路 generate
（本次 205 步全部如此，无 prefill 串扰）。kernel 归属用「最近的前一个步边界」，
因为 lm_head 的 GEMM 与 splitK 归约跑在 CUDA graph 之外，落在窗口结束后几 µs。

输出：kernels-c100.csv（name,calls_per_step,us_per_step）+ 逐桶/逐 kernel 表 + 逐步账。
"""
import argparse
import bisect
import collections
import gzip
import json
import pathlib
import re

# ---------------- 归桶 ----------------
# 顺序敏感：先具体后笼统。
GROUPS = [
    ("gemm:lm_head", re.compile(r"ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn$|internal::gemvx", re.I)),
    ("gemm:model", re.compile(r"cutlass_80_tensorop_bf16_s16816gemm|cutlass_80_wmma_tensorop_bf16|nvjet|cublas_sgemm|cublasLt::splitKreduce|cublasLt::gemm", re.I)),
    ("mamba:conv1d", re.compile(r"causal_conv1d", re.I)),
    ("mamba:ssm_core", re.compile(r"fused_recurrent_gated_delta_rule|gated_delta_rule|chunk_gated_delta|chunk_fwd_kernel|selective_scan|wy_fast|recompute_w_u|chunk_scaled_dot|solve_tril|merge_16x16", re.I)),
    ("mamba:state_mgmt", re.compile(r"mamba_align_fused|postprocess_mamba|get_aligned_state_indices", re.I)),
    ("attn:core", re.compile(r"flash_fwd_splitkv|flash_fwd|unified_attention|varlen_fwd|fmha|paged_attn", re.I)),
    ("attn:kv_cache", re.compile(r"reshape_and_cache|concat_and_cache|cache_kv", re.I)),
    ("norm:qk_rope_gate", re.compile(r"_fused_qk_rmsnorm_rope_gate", re.I)),
    ("norm:gated_rms(含silu)", re.compile(r"mean_mul_pow_rsqrt_silu", re.I)),
    ("norm:rms/fused_add_rms", re.compile(r"fused_add_rms_norm|embedding_rms_norm|rms_norm|layernorm|layer_norm", re.I)),
    ("sampling:gumbel/argmax", re.compile(r"gumbel_sample|ArgMaxOps|multinomial|top_k_renorm|_topk_topp", re.I)),
    ("sampling:spec_plumbing", re.compile(r"_get_num_sampled_and_rejected|_combine_sampled_and_draft_tokens|_scatter_num_accepted|_scatter_num_accepted_tokens", re.I)),
    ("elem:线性层 b/a 拆分拷贝", re.compile(r"triton_poi_fused_4$|triton_poi_fused_1$", re.I)),
    ("elem:线性层 out-buffer 清零", re.compile(r"triton_poi_fused_zeros_5|triton_poi_fused_zeros_2", re.I)),
    ("elem:at::native 拷贝/fill", re.compile(r"at::native::elementwise_kernel|vectorized_elementwise_kernel|at::native::vectorized_elementwise|elementwise_kernel_with_index", re.I)),
    ("elementwise:silu_mul", re.compile(r"triton_poi_fused_mul_silu_slice", re.I)),
    ("elementwise:attn_gate(sigmoid)", re.compile(r"triton_poi_fused_mul_sigmoid_view", re.I)),
    ("elementwise:triton_poi/at::native", re.compile(r"triton_poi_fused|at::native::elementwise_kernel|vectorized_elementwise_kernel|at::native::vectorized_elementwise|elementwise_kernel_with_index", re.I)),
    ("engine:meta/index", re.compile(r"slot_mappings|gather_block_tables|prepare_pos_seq_lens|_post_update_kernel|_zero_kv_blocks|index_elementwise_kernel|_scatter_gather_elementwise|vectorized_gather_kernel|memcpy32_post|_apply_write_kernel|arange_cuda", re.I)),
]

# 报表要求的 8 大功能组（从细桶折叠）
FOLD = {
    "gemm:model": "gemm/gemv",
    "gemm:lm_head": "gemm/gemv",
    "mamba:conv1d": "mamba conv1d",
    "mamba:ssm_core": "mamba ssm 递推",
    "mamba:state_mgmt": "mamba 状态搬运",
    "attn:core": "attention",
    "attn:kv_cache": "attention",
    "norm:qk_rope_gate": "norm",
    "norm:gated_rms(含silu)": "norm",
    "norm:rms/fused_add_rms": "norm",
    "elementwise:silu_mul": "elementwise",
    "elementwise:attn_gate(sigmoid)": "elementwise",
    "elementwise:triton_poi/at::native": "elementwise",
    "elem:线性层 b/a 拆分拷贝": "elementwise",
    "elem:线性层 out-buffer 清零": "elementwise",
    "elem:at::native 拷贝/fill": "elementwise",
    "sampling:gumbel/argmax": "sampling/argmax",
    "sampling:spec_plumbing": "sampling/argmax",
    "engine:meta/index": "其它(引擎元数据/索引)",
    "其它": "其它(引擎元数据/索引)",
}

STEP_RX = re.compile(r"context_(\d+)\((\d+)\).*?generation_(\d+)\((\d+)\)")


def group_of(name):
    for g, rx in GROUPS:
        if rx.search(name):
            return g
    return "其它"


def load_trace(path):
    p = pathlib.Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--md-out", default=None)
    a = ap.parse_args()

    ev = load_trace(a.trace)["traceEvents"]
    kerns = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda x: x["ts"])
    copies = sorted([e for e in ev if e.get("cat") in ("gpu_memcpy", "gpu_memset")], key=lambda x: x["ts"])
    ann = sorted([e for e in ev if e.get("cat") == "gpu_user_annotation"], key=lambda x: x["ts"])

    steps = []
    for e in ann:
        m = STEP_RX.search(e["name"])
        if not m:
            continue
        steps.append({"ts": e["ts"], "dur": e.get("dur", 0.0), "ctx": int(m.group(2)),
                      "gen_reqs": int(m.group(3)), "gen": int(m.group(4))})
    steps.sort(key=lambda s: s["ts"])
    N = len(steps)
    pure = [i for i, s in enumerate(steps) if s["ctx"] == 0]
    assert N > 0

    def owner(ts):
        i = bisect.bisect_right([s["ts"] for s in steps], ts) - 1
        return i if i >= 0 else None

    def in_scope(lst):
        out, unowned, ctxk = [], 0, 0
        for e in lst:
            o = owner(e["ts"])
            if o is None:
                unowned += 1
            elif steps[o]["ctx"] > 0:
                ctxk += 1
            else:
                out.append(e)
        return out, unowned, ctxk

    kern_in, kern_unowned, kern_ctx = in_scope(kerns)
    copy_in, copy_unowned, _ = in_scope(copies)

    # ---- 逐步账 ----
    nxt = [steps[i + 1]["ts"] if i + 1 < N else steps[i]["ts"] + steps[i]["dur"] for i in range(N)]
    per = collections.defaultdict(lambda: [0, 0.0])
    for e in kern_in:
        o = owner(e["ts"])
        per[o][0] += 1
        per[o][1] += e.get("dur", 0.0)
    per_c = collections.defaultdict(float)
    for e in copy_in:
        per_c[owner(e["ts"])] += e.get("dur", 0.0)

    cad = [nxt[i] - steps[i]["ts"] for i in pure]
    ann_dur = [steps[i]["dur"] for i in pure]
    busy = [per[i][1] for i in pure]
    cnt = [per[i][0] for i in pure]
    cp = [per_c[i] for i in pure]

    M = len(pure)
    stat = {
        "steps_total": N,
        "steps_pure_decode": M,
        "gen_reqs": collections.Counter(steps[i]["gen_reqs"] for i in pure).most_common(),
        "cadence_ms": sum(cad) / M / 1000,
        "annotation_ms": sum(ann_dur) / M / 1000,
        "kernel_busy_ms": sum(busy) / M / 1000,
        "kernel_per_step": sum(cnt) / M,
        "memcpy_memset_ms": sum(cp) / M / 1000,
        "memcpy_memset_per_step": sum(1 for _ in copy_in) / M,
        "unattributed_kernels": kern_unowned,
        "ctx_kernels": kern_ctx,
    }
    stat["gap_ms"] = stat["cadence_ms"] - stat["kernel_busy_ms"]
    stat["gap_pct"] = 100 * stat["gap_ms"] / stat["cadence_ms"]

    # ---- 逐 kernel ----
    agg = collections.defaultdict(lambda: [0, 0.0])
    for e in kern_in:
        k = agg[e["name"]]
        k[0] += 1
        k[1] += e.get("dur", 0.0)
    rows = []
    for name, (c, t) in agg.items():
        rows.append({"name": name, "group": group_of(name), "calls_per_step": c / M,
                     "us_per_step": t / M, "us_per_call": t / c, "total_ms": t / 1000})
    rows.sort(key=lambda r: -r["us_per_step"])

    # ---- CSV ----
    import csv as _csv
    with open(a.csv, "w", newline="") as f:
        w = _csv.writer(f, quoting=_csv.QUOTE_MINIMAL)
        w.writerow(["name", "calls_per_step", "us_per_step"])
        for r in rows:
            w.writerow([r["name"], "%.3f" % r["calls_per_step"], "%.3f" % r["us_per_step"]])

    # ---- 桶表 ----
    gsum = collections.defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        g = gsum[r["group"]]
        g[0] += r["us_per_step"]
        g[1] += r["calls_per_step"]
    fold = collections.defaultdict(lambda: [0.0, 0.0])
    for g, (us, c) in gsum.items():
        f_ = fold[FOLD.get(g, "其它(引擎元数据/索引)")]
        f_[0] += us
        f_[1] += c
    busy_ms = stat["kernel_busy_ms"]

    L = []
    P = L.append
    P("# %s" % pathlib.Path(a.trace).name)
    P("")
    P("## 逐步账（%d 步纯 decode，每步 %d 路 generate）" % (M, steps[pure[0]]["gen_reqs"]))
    P("")
    P("| 指标 | 值 |")
    P("|---|---|")
    P("| 步 cadence（步首到步首） | **%.3f ms** |" % stat["cadence_ms"])
    P("| gpu_user_annotation 覆盖（图形+尾随 eager） | %.3f ms |" % stat["annotation_ms"])
    P("| **kernel busy 合计** | **%.3f ms（%.2f 个 kernel）** |" % (busy_ms, stat["kernel_per_step"]))
    P("| memcpy+memset | %.3f ms（%.1f 次） |" % (stat["memcpy_memset_ms"], stat["memcpy_memset_per_step"]))
    P("| 空洞（cadence − kernel） | %.3f ms（%.1f%%） |" % (stat["gap_ms"], stat["gap_pct"]))
    P("")
    P("## 功能组（每步）")
    P("")
    P("| 组 | ms/步 | %% kernel | 发数/步 |")
    P("|---|---|---|---|")
    for g, (us, c) in sorted(fold.items(), key=lambda x: -x[1][0]):
        P("| %s | %.4f | %.1f%% | %.2f |" % (g, us / 1000, 100 * us / 1000 / busy_ms, c))
    P("| **kernel 合计** | **%.4f** | 100%% | **%.2f** |" % (busy_ms, stat["kernel_per_step"]))
    P("")
    P("## 细桶（每步）")
    P("")
    P("| 细桶 | ms/步 | %% | 发数/步 |")
    P("|---|---|---|---|")
    for g, (us, c) in sorted(gsum.items(), key=lambda x: -x[1][0]):
        P("| %s | %.4f | %.1f%% | %.2f |" % (g, us / 1000, 100 * us / 1000 / busy_ms, c))
    P("")
    P("## 逐 kernel（每步，按 us/步 降序）")
    P("")
    P("| # | us/步 | ms/步 | 发数/步 | 单发 µs | 细桶 | kernel |")
    P("|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        P("| %d | %.2f | %.4f | %.2f | %.2f | %s | `%s` |" % (
            i, r["us_per_step"], r["us_per_step"] / 1000, r["calls_per_step"],
            r["us_per_call"], r["group"], r["name"][:110]))
    P("")
    P("## memcpy/memset（每步）")
    P("")
    cagg = collections.defaultdict(lambda: [0, 0.0])
    for e in copy_in:
        k = cagg[e["name"]]
        k[0] += 1
        k[1] += e.get("dur", 0.0)
    P("| 发数/步 | 单发 µs | us/步 | 名称 |")
    P("|---|---|---|---|")
    for n, (c, t) in sorted(cagg.items(), key=lambda x: -x[1][1]):
        P("| %.1f | %.2f | %.1f | %s |" % (c / M, t / c, t / M, n))
    P("")

    out = "\n".join(L)
    print(out)
    if a.md_out:
        pathlib.Path(a.md_out).write_text(out)
    if a.json_out:
        pathlib.Path(a.json_out).write_text(json.dumps(
            {"stat": stat, "fold": fold, "groups": gsum, "rows": rows}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
