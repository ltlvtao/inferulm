#!/usr/bin/env bash
# prefill lane 大块验收扫描：chunk × {3009,6009,8009} 冷 TTFT + prefill 段表
# （进程内 eager 探针；GPU 一次一进程）。产出 bench/bigchunk_sweep/
# 下的原始日志 + 汇总表 chunk_scan.txt（TTFT/段表/贪心 md5）
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=bench/bigchunk_sweep
mkdir -p "$OUT"
BIN=target/release/examples/qwen_longctx_probe
: > "$OUT/chunk_scan.txt"
for len in 3009 6009 8009; do
  for chunk in 256 512 1024 2048 4096 8192; do
    log="$OUT/len${len}_c${chunk}.log"
    WEIN_CTX_LEN=8192 WEIN_KV_PAGES=512 WEIN_RADIX=0 \
      WEIN_CHUNK="$chunk" WEIN_CHUNK2="$chunk" \
      timeout 600 "$BIN" "$len" 32 >"$log" 2>&1 || { echo "FAIL len=$len c=$chunk（$log）"; tail -5 "$log"; exit 1; }
    python3 - "$log" "$len" "$chunk" >> "$OUT/chunk_scan.txt" <<'EOF'
import re, sys, hashlib
log, ln, c = open(sys.argv[1], encoding='utf-8').read(), sys.argv[2], sys.argv[3]
ttft = float(re.search(r"\| 1 \| \d+ \| 0 \| (\d+) \| ([\d.]+) \|", log).group(2))
steps = int(re.search(r"TTFT ([\d.]+) ms（prefill (\d+) 步）", log).group(2))
seg = {m.group(1): float(m.group(2)) for m in re.finditer(r"\| (\w+) \| ([\d.]+) \|", log)}
wall = float(re.search(r"wall ([\d.]+) ms/步", log).group(1))
tok = re.search(r"① tokens\(32\): \[([^\]]*)\]", log).group(1)
md5 = hashlib.md5(tok.encode()).hexdigest()[:10]
tot = lambda k: seg.get(k, 0.0) * steps
hollow = wall * steps - (tot('gemm') + tot('attn') + tot('delta') + tot('logits')
    + tot('norm') + tot('eltw') + tot('conv') + tot('gather') + tot('rope') + tot('kv') + tot('argmax') + tot('embed'))
print(f"len={ln} chunk={c:>4} steps={steps:>2} TTFT={ttft:7.1f} | gemm {tot('gemm'):6.1f} attn {tot('attn'):6.1f} "
      f"delta {tot('delta'):6.1f} logits {tot('logits'):5.1f} norm+eltw {tot('norm')+tot('eltw'):5.1f} "
      f"conv {tot('conv'):5.1f} hollow {hollow:5.1f} | md5 {md5}")
EOF
    tail -1 "$OUT/chunk_scan.txt"
  done
done
echo "== 汇总输出：$OUT/chunk_scan.txt"
