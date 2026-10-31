#!/usr/bin/env bash
# 快照栅格 A/B：只跑 L3q 满负荷（eager 与图档两模式），逐档打 JSON。
# 用法：bash bench/snap_grid_ab.sh <tag> <WEIN_SNAP_GRID 值|off>
#   值 = token 栅格（16/32/64...）；off = WEIN_RADIX=0（无前缀缓存基线）
set -euo pipefail
cd "$(dirname "$0")/.."
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY 2>/dev/null || true
BIN=target/release
PORT=8140
TAG=$1
GRID=$2
OUT=bench-reports/snap-ab-$TAG
mkdir -p "$OUT"
DIR=${WEIN_MODEL_DIR:-$HOME/.cache/wein/Qwen3.5-4B}

run_one () { # $1=名字 $2..=cli 旗标
  local name=$1; shift
  env "${ENVV[@]}" "$BIN/wein-cli" --port "$PORT" --engine core --exec qwen --model "$DIR" \
      --token-delay-ms 0 "$@" >/dev/null 2>&1 &
  local pid=$!
  until curl -s "localhost:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; do sleep 0.2; done
  "$BIN/wein-bench" --url "http://127.0.0.1:$PORT" --model "$DIR" --concurrency 100 --num 500 \
      --max-tokens 32 --repeat 3 --warmup --json "$OUT/$name.json" \
      | grep -E '"(ttft_p50_ms|itl_p50_ms|itl_p99_ms|aggregate_tps|errors)"'
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  sleep 1
}

if [ "$GRID" = off ]; then ENVV=("WEIN_RADIX=0"); else ENVV=("WEIN_SNAP_GRID=$GRID"); fi
echo "=== $TAG (grid=$GRID) ==="
run_one eager-full
run_one graph-full --graph
