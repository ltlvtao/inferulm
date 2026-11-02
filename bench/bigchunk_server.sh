#!/usr/bin/env bash
# 服务端图档（FULL_DECODE 桶图 + eager 大块 prefill）6009 对照：
# --graph --ctx-len 8192 --kv-pages 512 --chunk-size {2048,8192} ×2 冷跑，
# 期间 0.3s 粒度采 nvidia-smi（显存峰值）。GPU 一次一进程，逐档起停。
set -euo pipefail
cd "$(dirname "$0")/.."
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
OUT=bench/bigchunk_sweep
mkdir -p "$OUT"
DIR=${WEIN_MODEL_DIR:-$HOME/.cache/wein/Qwen3.5-4B}
PY=~/venvs/vllm/bin/python
PORT=8157
for chunk in 2048 8192; do
  echo "== server graph chunk=$chunk =="
  target/release/wein-cli --port "$PORT" --engine core --exec qwen --model "$DIR" \
    --token-delay-ms 0 --graph --ctx-len 8192 --kv-pages 512 --chunk-size "$chunk" \
    > "$OUT/server-c${chunk}.log" 2>&1 &
  PID=$!
  trap "kill $PID 2>/dev/null || true" EXIT
  until curl -s "localhost:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; do
    kill -0 "$PID" 2>/dev/null || { echo "server died"; tail -5 "$OUT/server-c${chunk}.log"; exit 1; }
    sleep 0.3
  done
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.3; done ) \
    > "$OUT/vram-c${chunk}.txt" 2>/dev/null &
  VS=$!
  "$PY" bench/longctx_probe.py --url "http://127.0.0.1:$PORT" --model "$DIR" \
    --tokens 6009 --max-tokens 32 --repeats 2 --cold \
    --note "wein graph ctx8192 chunk=$chunk" --out "$OUT/server-6009-c${chunk}.json" \
    | grep -E "^\[cold"
  kill "$VS" 2>/dev/null || true
  echo "peak_vram_mib=$(sort -n "$OUT/vram-c${chunk}.txt" | tail -1)"
  kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true
  trap - EXIT
  sleep 2
done
echo "== done =="
