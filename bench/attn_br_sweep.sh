#!/bin/bash
# attention prefill BR（=ntok，一块覆盖的 q token 数）真流水扫档。
# 每档一个进程（GPU 串行独占），只改 WEIN_ATTN_CFG（decode 档恒 r2）。
#   bench/attn_br_sweep.sh [n] [out.tsv]
set -u
cd "$(dirname "$0")/.."
n=${1:-6009}
out=${2:-/tmp/attn_br.tsv}
printf "cfg\tTTFT_ms\tattn_ms\ttotal_ms\ttok\n" > "$out"
for cfg in "$@"; do :; done
for cfg in \
  "n8r4" \
  "n8r4,ntok=8" \
  "nd=4,ntok=4" \
  "nd=4,rpt=2,ntok=4" \
  "nd=4,ntok=8" \
  "nd=16,ntok=8" \
  "nd=16,ntok=16" \
  "nd=16,rpt=2,ntok=8" \
  "nd=16,rpt=2,ntok=16" \
  "nd=16,ntok=32" ; do
  o=$(WEIN_CTX_LEN=8192 WEIN_KV_PAGES=512 WEIN_CHUNK=256 WEIN_ATTN_CFG="$cfg" \
      ./target/release/examples/qwen_longctx_probe "$n" 1 2>&1)
  ttft=$(echo "$o" | awk -F'|' '/^\| 1 \| 256/{gsub(/ /,"",$6); print $6}')
  attn=$(echo "$o" | awk -F'|' '/^\| attn \|/{gsub(/ /,"",$3); print $3; exit}')
  tot=$(echo "$o" | awk -F'|' '/^wall /{gsub(/ /,"",$2); print $2; exit}')
  tok=$(echo "$o" | grep -o '① tokens([0-9]*): \[[0-9]*\]' | grep -o '\[[0-9]*\]')
  printf "%s\t%s\t%s\t%s\t%s\n" "$cfg" "$ttft" "$attn" "$tot" "$tok" | tee -a "$out"
done
