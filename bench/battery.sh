#!/usr/bin/env bash
# wein 性能基线电池——归因阶梯的执行器（方法论文档：bench-reports/BASELINE.md）
#
#   L0  路由上限       flood GET /v1/models（与引擎无关）
#   L1  serve 层开销    零延迟引擎（--token-delay-ms 0）下的 TTFT/ITL —— serve 层的"纯账"
#   L1' 节拍泵参照     1ms/token 引擎（对照基线；真实引擎接入后被 L2+ 替代）
#
# 用法：bench/battery.sh [release|debug]     （默认 release；报告落 bench-reports/battery-<时间戳>/）
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE=${1:-release}
PORT=${WEIN_BENCH_PORT:-8125}
BIN=target/$PROFILE
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=bench-reports/battery-$STAMP
export WEIN_GIT_REV=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
export WEIN_PROFILE=$PROFILE
mkdir -p "$OUT"
echo "== wein battery  $STAMP  profile=$PROFILE  rev=$WEIN_GIT_REV  → $OUT =="

run_suite () { # $1=tag；$2.. 传给 wein-cli 的延迟参数
  local tag=$1; shift
  "$BIN/wein-cli" --port "$PORT" "$@" >/dev/null 2>&1 &
  local pid=$!
  trap "kill $pid 2>/dev/null || true" EXIT
  until curl -s "localhost:$PORT/v1/models" 2>/dev/null | grep -q mock-model; do sleep 0.2; done

  echo "-- $tag · L0 router flood --"
  "$BIN/wein-bench" --url "http://127.0.0.1:$PORT" --concurrency 100 --num 20000 \
      --flood --json "$OUT/$tag-l0-router.json" | grep -E '"(qps|lat_p50_ms|lat_p99_ms|errors)"'
  echo "-- $tag · L1 streaming (repeat=3, 中位数) --"
  "$BIN/wein-bench" --url "http://127.0.0.1:$PORT" --concurrency 100 --num 500 --max-tokens 32 \
      --repeat 3 --warmup --json "$OUT/$tag-l1-stream.json" \
      | grep -E '"(ttft_p50_ms|ttft_p99_ms|itl_p50_ms|itl_p99_ms|aggregate_tps|errors)"'

  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  trap - EXIT
}

run_suite zero-delay --prefill-delay-ms 0 --token-delay-ms 0
run_suite paced-1ms --prefill-delay-ms 2 --token-delay-ms 1

python3 - "$OUT" "$STAMP" <<'PYEOF' > "$OUT/SUMMARY.md"
import json, sys
out, stamp = sys.argv[1], sys.argv[2]
def load(name):
    try:
        with open(f"{out}/{name}") as f: return json.load(f)["result"]
    except FileNotFoundError: return None
rows = []
r = load("zero-delay-l0-router.json")
if r: rows.append(("L0 路由上限", f"qps={r['qps']:.0f} p50={r['lat_p50_ms']:.2f}ms p99={r['lat_p99_ms']:.2f}ms"))
r = load("zero-delay-l1-stream.json")
if r: rows.append(("L1 serve 纯开销(零延迟引擎)", f"ttft_p50={r['ttft_p50_ms']:.2f}ms ttft_p99={r['ttft_p99_ms']:.2f}ms itl_p50={r['itl_p50_ms']:.3f}ms itl_p99={r['itl_p99_ms']:.2f}ms tps={r['aggregate_tps']:.0f}"))
r = load("paced-1ms-l1-stream.json")
if r: rows.append(("L1' 节拍泵(1ms/token) 参照", f"ttft_p50={r['ttft_p50_ms']:.2f}ms itl_p50={r['itl_p50_ms']:.2f}ms itl_p99={r['itl_p99_ms']:.2f}ms tps={r['aggregate_tps']:.0f}"))
print(f"# battery {stamp}\n")
print("| 层 | 指标（中位数, c=100, n=500, max_tokens=32） |")
print("|---|---|")
for name, m in rows: print(f"| {name} | {m} |")
PYEOF

echo "== summary =="
cat "$OUT/SUMMARY.md"
