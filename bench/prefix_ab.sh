#!/usr/bin/env bash
# 前缀缓存长 prompt TTFT A/B：同一长前缀两次请求（A 冷填缓存/B 热命中）。
# 用法：bash bench/prefix_ab.sh <tag> <on|off>
set -euo pipefail
cd "$(dirname "$0")/.."
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY 2>/dev/null || true
BIN=target/release
PORT=8141
TAG=$1
MODE=$2
DIR=${WEIN_MODEL_DIR:-$HOME/.cache/wein/Qwen3.5-4B}
if [ "$MODE" = off ]; then ENVV=(WEIN_RADIX=0); else ENVV=(WEIN_RADIX_DEBUG=1); fi
echo "=== prefix A/B · $TAG ($MODE) ==="
# --chunk-size：让长 prompt 跨多步 → 中途落块边界检查点（MAX_POS=256 下
# 单步追平的 prompt 只会在末块留一个检查点，命中无从谈起）
env "${ENVV[@]}" "$BIN/wein-cli" --port "$PORT" --engine core --exec qwen --model "$DIR" \
    --token-delay-ms 0 ${CHUNK:+--chunk-size $CHUNK} >/tmp/prefix-$TAG.log 2>&1 &
pid=$!
until curl -s "localhost:$PORT/v1/models" 2>/dev/null | grep -qF "$DIR"; do sleep 0.2; done
python3 bench/prefix_probe.py "http://127.0.0.1:$PORT" "$DIR" "${WORDS:-130}"
kill "$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true
