#!/usr/bin/env bash
# Qwen3.5-4B 批扩展曲线：ITL(m)/步时延/tps vs 并发行数（eager vs 图档）。
# vs vLLM 对比（vs-vllm/REPORT.md §1）显示解码步差距随批扩大（×2.8@批4 →
# ×4.7@批16）——本曲线是 wein 侧的批扩展账本：每点 n=c 同时起步、
# max_tokens=32，追平后稳态即纯 decode 步（无 churn 混合步干扰）。
# 产出 bench-reports/qwen-sweep-<stamp>/。GPU 互斥：独占运行。
set -euo pipefail
cd "$(dirname "$0")/.."
# 基准必须直连：环境代理（WSL autoProxy 注入的 http_proxy）会劫持 127.0.0.1 流量
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
BIN=target/release
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=bench-reports/qwen-sweep-$STAMP
mkdir -p "$OUT"
PORT=${WEIN_SWEEP_PORT:-8191}
DIR=${WEIN_MODEL_DIR:-$HOME/.cache/wein/Qwen3.5-4B}
CONC="${WEIN_SWEEP_CONC:-4 16 32 64 100 128}"

if [ ! -d "$DIR" ]; then echo "模型目录不在场：$DIR" >&2; exit 1; fi

run_mode () {  # $1=tag，$2..=wein-cli 附加旗标
  local tag=$1; shift
  "$BIN/wein-cli" --port "$PORT" --engine core --exec qwen --model "$DIR" --token-delay-ms 0 "$@" \
      >/dev/null 2>&1 &
  local pid=$!
  trap "kill $pid 2>/dev/null || true" EXIT
  until curl -s "localhost:$PORT/v1/models" 2>/dev/null | grep -qF "$DIR"; do sleep 0.2; done
  for c in $CONC; do
    echo "-- $tag · c=$c n=$c mt=32 --"
    "$BIN/wein-bench" --url "http://127.0.0.1:$PORT" --model "$DIR" --concurrency "$c" --num "$c" \
      --max-tokens 32 --repeat 1 --warmup --json "$OUT/$tag-c$c.json" \
      | grep -E '"(ttft_p50_ms|itl_p50_ms|aggregate_tps|errors)"' | head -4
  done
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  trap - EXIT
}

run_mode eager
run_mode graph --graph

python3 - "$OUT" "$STAMP" <<'PYEOF' > "$OUT/SUMMARY.md"
import json, sys
out, stamp = sys.argv[1], sys.argv[2]
def load(p):
    try:
        with open(p) as f: return json.load(f)["result"]
    except FileNotFoundError: return None
print(f"# qwen 批扩展曲线 {stamp}（n=c 同步起步，mt=32，纯 decode 稳态）\n")
print("| c | eager ITL p50 / tps | graph ITL p50 / tps | graph−eager ITL |")
print("|---|---|---|---|")
for c in [4, 16, 32, 64, 100, 128]:
    e, g = load(f"{out}/eager-c{c}.json"), load(f"{out}/graph-c{c}.json")
    if not e and not g: continue
    def fmt(r): return ("—" if r is None else
        f"{r['itl_p50_ms']:.2f}ms / {r['aggregate_tps']:.1f}")
    delta = "—"
    if e and g:
        d = (g['itl_p50_ms'] / e['itl_p50_ms'] - 1) * 100
        delta = f"{d:+.1f}%"
    print(f"| {c} | {fmt(e)} | {fmt(g)} | {delta} |")
PYEOF
echo "== SUMMARY =="
cat "$OUT/SUMMARY.md"
