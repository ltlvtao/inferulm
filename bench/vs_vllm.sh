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
# 基准必须直连：环境代理（WSL autoProxy 注入的 http_proxy）会劫持 127.0.0.1 流量
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

SIDE=${1:?用法: bench/vs_vllm.sh <wein|vllm> [tag]}
TAG=${2:-$SIDE}
DIR=${WEIN_MODEL_DIR:-$HOME/.cache/wein/Qwen3.5-4B}
# vLLM 侧用文本骨干副本：本地检查点是 VL 统体（738 张量含 297 个 model.visual.*），
# wein 只加载文本骨干——副本滤掉视觉塔后两端跑同一组文本张量（make_text_copy.py 产物）
TEXT_DIR=${WEIN_VLLM_MODEL:-$HOME/.cache/wein/Qwen3.5-4B-text}
PORT=${WEIN_VLLM_PORT:-8155}
OUT=bench-reports/vs-vllm-$TAG
mkdir -p "$OUT"
echo "== vs-vllm [$SIDE]  → $OUT =="

start_wein () {
  ./target/release/wein-cli --port "$PORT" --engine core --exec qwen \
      --model "$DIR" --token-delay-ms 0 ${WEIN_EXTRA_FLAGS:-} >/dev/null 2>&1 &
  PID=$!
  # 就绪探针 = /health（引擎活即 ok）；旧探针用 /v1/models 的模型 id，
  # 引擎任务死掉后 /v1/models 仍回 200，探针会假绿
  until curl -s "localhost:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; do sleep 0.5; done
}
start_vllm () {
  source ~/venvs/vllm/bin/activate
  # WSL 陷阱 ×2：
  # ① PATH 混入 Windows 宿主 CUDA（/mnt/c/...CUDA/v12.4/bin），inductor 调 nvcc
  #   经 interop 执行 Windows PE → Permission denied——剥掉 /mnt/c 的 CUDA 段
  # ② 该 EACCES 会**盖住真正的编译错误**（dynamo 错误复现构建器先跑 nvcc --version
  #   收集环境，自己先炸）——把 venv 自带的好 nvcc（nvidia/cu13，tilelang 依赖）
  #   同时给 CUDACXX（inductor 首选）与 PATH（复现构建器用），真错误才可见
  PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/mnt/c/.*CUDA' | paste -sd:)
  local nvcc_dir
  nvcc_dir=$(dirname "$(find "$(python -c 'import site; print(site.getsitepackages()[0])')/nvidia" -name nvcc -type f 2>/dev/null | head -1)" 2>/dev/null)
  [ -x "$nvcc_dir/nvcc" ] && { export CUDACXX="$nvcc_dir/nvcc"; PATH="$nvcc_dir:$PATH"; }
  # ③ flashinfer JIT（sampling 算子）要求 nvcc 与 torch cu130 头严格同版——venv 的
  #   nvcc 是 13.4（tilelang 依赖）而 torch 是 cu130 → CCCL "compiler and toolkit
  #   headers incompatible"。贪心基准下采样非测量轴，切 vLLM 原生 sampler 绕开 JIT
  export VLLM_USE_FLASHINFER_SAMPLER=0
  # max-model-len 压到 512（对齐 wein T_MAX 账面；KV 占用小意义在于公平
  # 的调度行为）；gpu-mem-util 0.85 默认档
  # hf-overrides：本地 config 的 architectures=Qwen3_5ForConditionalGeneration
  # 被 vLLM 注册表路由到 VL 多模态类（要 image processor → 起 400/500）；
  # 纯文本权重对应 Qwen3_5ForCausalLM——只改 vLLM 视图，不动模型文件
  vllm serve "$TEXT_DIR" --port "$PORT" --max-model-len 512 \
      --gpu-memory-utilization 0.85 \
      --hf-overrides '{"architectures": ["Qwen3_5ForCausalLM"]}' >/dev/null 2>&1 &
  PID=$!
  # vLLM 就绪慢（加载+编译图）；上限 10 分钟
  # 注意：这一侧**必须**继续用 /v1/models——vLLM 没有 /health 路由（会 404），
  # 换成 /health 会让 vLLM 侧永远等不到就绪。wein 侧才用 /health。
  for _ in $(seq 600); do
    curl -s "localhost:$PORT/v1/models" 2>/dev/null | grep -qF "$TEXT_DIR" && return
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
  # --model 必须匹配当侧服务的 served-model-name（wein=原目录，vLLM=文本副本目录）
  local m=$DIR; [ "$SIDE" = vllm ] && m=$TEXT_DIR
  "$PWD/target/release/wein-bench" --url "http://127.0.0.1:$PORT" --model "$m" \
      --concurrency "$2" --num "$3" --max-tokens "$4" --repeat 1 --warmup \
      --json "$OUT/$1.json" | grep -E '"(ttft_p50_ms|ttft_p99_ms|itl_p50_ms|itl_p99_ms|aggregate_tps|errors)"'
}

run_load light-c4 4 8 12
run_load mid-c16 16 32 32

echo "== [$SIDE] 完成 =="
