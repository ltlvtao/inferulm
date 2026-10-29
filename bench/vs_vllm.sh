#!/usr/bin/env bash
# wein vs vLLM 对比基准——同一把尺子（wein-bench，OpenAI 协议）打两端。
#
# 用法：bench/vs_vllm.sh <wein|vllm> [tag]
#   wein  : 起 wein-cli qwen 执行器（eager；如需图档加 --graph 见下）
#   vllm  : 起 vllm serve（需 ~/venvs/vllm 已装；首次启动编译几分钟）
#
# 负载两档（报告落 bench-reports/vs-vllm-<tag>/）：
#   轻（L3q 同款）   c=4  n=8  max_tokens=12 —— 与 BASELINE §6 直接连线
#   中（批界内满载） c=16 n=32 max_tokens=32 —— wein MAX_ROWS=128/T_MAX=512
#        批界内的上限档（prompt 11 + 32 生成 < T_MAX/2·c 预算），vLLM 远未到
#        其容量——量的是 wein 批界内的追赶差距，不是 vLLM 上限
#
# GPU 互斥：两端各自起服→跑→停（4090 24G 装不下双份模型）
set -euo pipefail
cd "$(dirname "$0")/.."

SIDE=${1:?用法: bench/vs_vllm.sh <wein|vllm> [tag]}
TAG=${2:-$SIDE}
DIR=${WEIN_MODEL_DIR:-$HOME/.cache/wein/Qwen3.5-4B}
PORT=${WEIN_VLLM_PORT:-8155}
OUT=bench-reports/vs-vllm-$TAG
mkdir -p "$OUT"
echo "== vs-vllm [$SIDE]  → $OUT =="

start_wein () {
  ./target/release/wein-cli --port "$PORT" --engine core --exec qwen \
      --model "$DIR" --token-delay-ms 0 ${WEIN_EXTRA_FLAGS:-} >/dev/null 2>&1 &
  PID=$!
  until curl -s "localhost:$PORT/v1/models" 2>/dev/null | grep -qF "$DIR"; do sleep 0.5; done
}
start_vllm () {
  source ~/venvs/vllm/bin/activate
  # max-model-len 压到 512（对齐 wein T_MAX 账面；KV 占用小意义在于公平
  # 的调度行为）；gpu-mem-util 0.85 默认档
  vllm serve "$DIR" --port "$PORT" --max-model-len 512 \
      --gpu-memory-utilization 0.85 >/dev/null 2>&1 &
  PID=$!
  # vLLM 就绪慢（加载+编译图）；上限 10 分钟
  for _ in $(seq 600); do
    curl -s "localhost:$PORT/v1/models" 2>/dev/null | grep -qF "$DIR" && return
    sleep 1
  done
  echo "vLLM 10 分钟未就绪" >&2; kill $PID 2>/dev/null; exit 1
}

case "$SIDE" in
  wein) start_wein ;;
  vllm) start_vllm ;;
  *) echo "未知端: $SIDE" >&2; exit 1 ;;
esac
trap 'kill $PID 2>/dev/null || true' EXIT

run_load () { # $1=档名 $2=c $3=n $4=max_tokens
  echo "-- $1 · c=$2 n=$3 max_tokens=$4 --"
  "$PWD/target/release/wein-bench" --url "http://127.0.0.1:$PORT" --model "$DIR" \
      --concurrency "$2" --num "$3" --max-tokens "$4" --repeat 1 --warmup \
      --json "$OUT/$1.json" | grep -E '"(ttft_p50_ms|ttft_p99_ms|itl_p50_ms|itl_p99_ms|aggregate_tps|errors)"'
}

run_load light-c4 4 8 12
run_load mid-c16 16 32 32

echo "== [$SIDE] 完成 =="
