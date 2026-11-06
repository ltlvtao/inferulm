#!/usr/bin/env bash
# 起 vLLM serve（端口 8199，与对比基线同档），带 vLLM 自带 torch profiler。
# 只测量：不改 wein 源码。
set -uo pipefail
cd /home/lvtao/inferE/deepin/wein
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

OUT=$PWD/bench-reports/vs-vllm-kernels2
TRACE=$OUT/traces
PORT=8199
TEXT_DIR=$HOME/.cache/wein/Qwen3.5-4B-text
PY=$HOME/venvs/vllm/bin/python
mkdir -p "$TRACE"
LOG=$OUT/run.log

say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# --- WSL 修补 ①: PATH 剥掉 /mnt/c/...CUDA ---
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/mnt/c/.*CUDA' | paste -sd:)
# --- 修补 ②: CUDACXX/PATH 指向 venv 自带 nvcc ---
nvcc_dir=$(dirname "$(find "$("$PY" -c 'import site; print(site.getsitepackages()[0])')/nvidia" -name nvcc -type f 2>/dev/null | head -1)" 2>/dev/null)
[ -x "$nvcc_dir/nvcc" ] && { export CUDACXX="$nvcc_dir/nvcc"; PATH="$nvcc_dir:$PATH"; }
# --- 修补 ③: 原生 sampler（flashinfer JIT 会因 cu130/cu134 头不兼容而炸）---
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_SERVER_DEV_MODE=1
export VLLM_TORCH_PROFILER_DIR="$TRACE"

args=(serve "$TEXT_DIR" --port "$PORT" \
  --max-model-len 512 --gpu-memory-utilization 0.85 \
  --mamba-ssm-cache-dtype bfloat16 \
  --max-num-batched-tokens 2048 --max-num-seqs 256 \
  --hf-overrides '{"architectures": ["Qwen3_5ForCausalLM"]}' \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_dir="$TRACE" \
  --profiler-config.torch_profiler_with_stack=false \
  --profiler-config.torch_profiler_use_gzip=true)

printf 'VLLM_SERVER_DEV_MODE=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_TORCH_PROFILER_DIR=%s \\\n  %s -m vllm.entrypoints.cli.main %s\n' \
    "$TRACE" "$PY" "${args[*]}" > "$OUT/cmd.txt"

setsid "$PY" -m vllm.entrypoints.cli.main "${args[@]}" > "$OUT/server.log" 2>&1 &
PID=$!
echo "$PID" > "$OUT/vllm.pid"
say "vLLM 启动中 pid=$PID port=$PORT"

for _ in $(seq 600); do
  curl -s "localhost:$PORT/v1/models" 2>/dev/null | grep -qF "$TEXT_DIR" && { say "vLLM ready"; exit 0; }
  kill -0 "$PID" 2>/dev/null || { say "启动失败"; tail -40 "$OUT/server.log" | tee -a "$LOG"; exit 1; }
  sleep 1
done
say "10 分钟未就绪"; kill -- -"$PID" 2>/dev/null; exit 1
