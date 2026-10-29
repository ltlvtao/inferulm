"""逐层对拍比较：wein trace vs HF transformers 参照。

用法（venv python，numpy）:
  python bench/trace_diff.py <wein_trace_dir> <ref_dir>

每步每层：max|Δ| / mean|Δ|（f32）；预填步附逐位置 max|Δ| 行（递推类偏差
从某位置开始增长的特征签名）。首个超阈层的即定位锚点。
"""
import sys, os, json
import numpy as np

wein_dir, ref_dir = sys.argv[1], sys.argv[2]
meta = json.load(open(f"{wein_dir}/meta.json"))
H, kinds = meta["hidden"], meta["layers"]

def load(d, name):
    p = f"{d}/{name}.f32"
    return np.fromfile(p, dtype=np.float32) if os.path.exists(p) else None

steps = sorted({int(f.split("-")[0][1:]) for f in os.listdir(wein_dir) if f.endswith(".f32")})
THRESH = 0.05  # bf16 残差流两三 ulp 的量级；超过即结构性漂移

for s in steps:
    print(f"\n== 步 s{s} ==")
    names = [f"s{s}-embed"] + [f"s{s}-layer-{i:02}" for i in range(len(kinds))] + [f"s{s}-logits"]
    first_bad = None
    for n in names:
        w, r = load(wein_dir, n), load(ref_dir, n)
        if w is None or r is None or w.shape != r.shape:
            print(f"  {n:<16} 缺失/形状不符（wein {None if w is None else w.shape} vs ref）")
            continue
        d = np.abs(w - r)
        tag = ""
        if n.startswith(f"s{s}-layer-"):
            li = int(n[-2:])
            tag = f"  [{kinds[li]}]"
        flag = ""
        if d.max() > THRESH and first_bad is None:
            first_bad = n
            flag = "  ← 首个超阈"
        print(f"  {n:<16} max|Δ|={d.max():9.5f}  mean|Δ|={d.mean():9.6f}{tag}{flag}")
    # 预填步：首超阈层做逐位置剖析（递推签名：从某位置起增长）
    if first_bad and s == 1:
        w, r = load(wein_dir, first_bad), load(ref_dir, first_bad)
        t = w.size // H
        pw = w.reshape(t, H); pr = r.reshape(t, H)
        print(f"  -- {first_bad} 逐位置 max|Δ|：")
        for p in range(t):
            d = np.abs(pw[p] - pr[p]).max()
            print(f"     pos{p}: {d:.5f}")
    if first_bad and s > 1:
        print(f"  （decode 步 t=1，无逐位置剖析）")
print(f"\n阈值 {THRESH}（bf16 残差流 ulp 量级；超阈=结构性漂移锚点）")
