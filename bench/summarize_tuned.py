#!/usr/bin/env python3
"""把 bench-reports/vs-vllm-tuned/<档>/*.json 汇总成 markdown 表（含相对参考档的 Δ%）。

用法：
  summarize_tuned.py [--root bench-reports/vs-vllm-tuned] [--ref default] [--json out.json]

每档需要的文件（缺了填 “—”，不报错）：
  short-light-c4.json / short-full-c100.json   wein-bench 产物（result = 3 次中位）
  long-3000.json / long-6000.json              longctx_probe 产物（median + 每次 metrics_delta）
  vram-peak-c100.txt                           满载窗口显存峰值
  fail.txt / config.txt                        失败原因 / 配置
"""
import argparse
import json
import pathlib
import statistics

NUM = [
    ("light_itl50", "轻载 ITL p50", "ms"),
    ("light_itl99", "轻载 ITL p99", "ms"),
    ("light_ttft50", "轻载 TTFT p50", "ms"),
    ("light_ttft99", "轻载 TTFT p99", "ms"),
    ("light_tps", "轻载 tps", "tok/s"),
    ("full_itl50", "满载 ITL p50", "ms"),
    ("full_itl99", "满载 ITL p99", "ms"),
    ("full_ttft50", "满载 TTFT p50", "ms"),
    ("full_ttft99", "满载 TTFT p99", "ms"),
    ("full_tps", "满载 tps", "tok/s"),
    ("vram_peak", "满载峰值显存", "MiB"),
    ("c3000", "3009 冷 TTFT", "ms"),
    ("c6000", "6009 冷 TTFT", "ms"),
    ("h3000", "3009 热 TTFT", "ms"),
    ("h6000", "6009 热 TTFT", "ms"),
    ("hit3000", "3009 命中 token", "tok"),
    ("hit6000", "6009 命中 token", "tok"),
    ("cold_itl", "长上下文冷 ITL p50", "ms"),
]
# 越低越好 / 越高越好（用于 Δ% 方向染色，不做判断）
LOWER_BETTER = {
    "light_itl50", "light_itl99", "light_ttft50", "light_ttft99",
    "full_itl50", "full_itl99", "full_ttft50", "full_ttft99",
    "c3000", "c6000", "h3000", "h6000", "cold_itl",
}


def load(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def bench_metrics(path):
    d = load(path)
    if not d:
        return {}
    r = d.get("result", {})
    return {
        "itl50": r.get("itl_p50_ms"),
        "itl99": r.get("itl_p99_ms"),
        "ttft50": r.get("ttft_p50_ms"),
        "ttft99": r.get("ttft_p99_ms"),
        "tps": r.get("aggregate_tps"),
        "errors": r.get("errors"),
    }


def hits(rec):
    """热跑的 prefix_cache_hits_total 增量中位（= 命中 token 数）。"""
    vals = []
    for h in rec.get("hot", []):
        v = (h.get("metrics_delta") or {}).get("prefix_cache_hits_total")
        if v is not None:
            vals.append(v)
    return statistics.median(vals) if vals else None


def collect(tier_dir):
    row = {}
    light = bench_metrics(tier_dir / "short-light-c4.json")
    full = bench_metrics(tier_dir / "short-full-c100.json")
    row.update({f"light_{k}": v for k, v in light.items()})
    row.update({f"full_{k}": v for k, v in full.items()})
    vp = tier_dir / "vram-peak-c100.txt"
    row["vram_peak"] = int(vp.read_text().strip()) if vp.exists() and vp.read_text().strip().isdigit() else None
    for n in (3000, 6000):
        rec = load(tier_dir / f"long-{n}.json")
        if not rec:
            continue
        med = rec.get("median") or {}
        key = "3000" if n == 3000 else "6000"
        row[f"c{key}"] = med.get("cold_ttft_ms")
        row[f"h{key}"] = med.get("hot_ttft_ms")
        row[f"hit{key}"] = hits(rec)
        row["cold_itl"] = med.get("cold_itl_p50_ms")
        row[f"hold{key}"] = med.get("hot_itl_p50_ms")
    cfg = tier_dir / "config.txt"
    row["fail"] = (tier_dir / "fail.txt").read_text().splitlines()[0][:160] if (tier_dir / "fail.txt").exists() else None
    row["cfgtext"] = cfg.read_text() if cfg.exists() else ""
    return row


def delta(val, ref):
    if val is None or ref in (None, 0):
        return None
    return (val - ref) / ref * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="bench-reports/vs-vllm-tuned")
    ap.add_argument("--ref", default="default")
    ap.add_argument("--json", default=None, help="另存原始汇总 JSON")
    ap.add_argument("--emit-config-json", action="store_true",
                    help="给每档写 <档>/config.json（config.txt 的键值 + cache_config_info 几何）")
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    tiers = {}
    for d in sorted(root.iterdir()):
        if d.is_dir():
            tiers[d.name] = collect(d)
    if a.emit_config_json:
        for name, row in tiers.items():
            d = root / name
            cfg = {}
            for line in (row.get("cfgtext") or "").splitlines():
                if ": " in line:
                    k, v = line.split(": ", 1)
                    cfg[k.strip()] = v.strip()
            cc = {}
            p = d / "cache-config.txt"
            if p.exists():
                import re
                txt = p.read_text()
                for k in ("block_size", "num_gpu_blocks", "kv_cache_size_tokens",
                          "cache_dtype", "mamba_ssm_cache_dtype", "mamba_cache_mode",
                          "enable_prefix_caching", "gpu_memory_utilization", "hash_block_size"):
                    m = re.search(k + r'="([^"]*)"', txt)
                    if m:
                        cc[k] = m.group(1)
            out = {
                "tier": name,
                "config_txt": cfg,
                "cache_config_info": cc,
                "fail": row.get("fail"),
                "cmd": (d / "cmd.txt").read_text().strip() if (d / "cmd.txt").exists() else None,
                "server_flags": (d / "server-flags.txt").read_text().splitlines()
                if (d / "server-flags.txt").exists() else [],
            }
            (d / "config.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"# config.json 已写入 {len(tiers)} 档", flush=True)
    if a.ref not in tiers:
        raise SystemExit(f"参考档 {a.ref} 不存在")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(tiers, ensure_ascii=False, indent=1))

    ref = tiers[a.ref]
    order = [a.ref] + [t for t in tiers if t != a.ref]
    hdr = ["档"] + [f"{lbl}" for _, lbl, _ in NUM]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for t in order:
        r = tiers[t]
        cells = [f"**{t}**" if t == a.ref else t]
        for k, _, unit in NUM:
            v = r.get(k)
            if v is None:
                cells.append("—")
                continue
            s = f"{v:.0f}" if unit in ("tok", "MiB", "tok/s") else f"{v:.2f}"
            d = delta(v, ref.get(k))
            if t != a.ref and d is not None and abs(d) >= 0.5:
                s += f" ({d:+.1f}%)"
            cells.append(s)
        print("| " + " | ".join(cells) + " |")

    print("\n## 失败档")
    for t, r in tiers.items():
        if r.get("fail"):
            print(f"- `{t}`: {r['fail']}")
        elif all(r.get(k) is None for k, _, _ in NUM):
            print(f"- `{t}`: 无数据（起服/测量中断）")


if __name__ == "__main__":
    main()
