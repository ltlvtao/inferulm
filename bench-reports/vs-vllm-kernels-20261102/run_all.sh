#!/usr/bin/env bash
# vLLM kernel 级剖面：A 满载 decode / B 6009 冷 prefill / C 6009 decode。
# 只测量：不改 wein 生产代码，不再测已采信的段表。
set -uo pipefail
cd /home/lvtao/inferE/deepin/wein
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

OUT=$PWD/bench-reports/vs-vllm-kernels-20261102
TRACE=$OUT/traces
LOG=$OUT/run.log
PORT=8177
TEXT_DIR=$HOME/.cache/wein/Qwen3.5-4B-text
PY=$HOME/venvs/vllm/bin/python
mkdir -p "$TRACE"

say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

gpu_state () {
  say "  linux  nvidia-smi: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1)"
  say "  win    compute-apps: $(/mnt/c/Windows/System32/nvidia-smi.exe --query-compute-apps=pid,process_name 2>/dev/null | tail -n +2 | tr '\n' '|')"
}

start_vllm () {
  PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/mnt/c/.*CUDA' | paste -sd:)
  local nvcc_dir
  nvcc_dir=$(dirname "$(find "$("$PY" -c 'import site; print(site.getsitepackages()[0])')/nvidia" -name nvcc -type f 2>/dev/null | head -1)" 2>/dev/null)
  [ -x "$nvcc_dir/nvcc" ] && { export CUDACXX="$nvcc_dir/nvcc"; PATH="$nvcc_dir:$PATH"; }
  export VLLM_USE_FLASHINFER_SAMPLER=0
  export VLLM_SERVER_DEV_MODE=1
  export VLLM_TORCH_PROFILER_DIR="$TRACE"
  local args=(serve "$TEXT_DIR" --port "$PORT" --max-model-len 8192
      --gpu-memory-utilization 0.85
      --mamba-ssm-cache-dtype bfloat16
      --max-num-batched-tokens 2048 --max-num-seqs 256
      --hf-overrides '{"architectures": ["Qwen3_5ForCausalLM"]}'
      --profiler-config.profiler=torch
      --profiler-config.torch_profiler_dir="$TRACE"
      --profiler-config.torch_profiler_with_stack=false
      --profiler-config.torch_profiler_use_gzip=true)
  printf 'VLLM_SERVER_DEV_MODE=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_TORCH_PROFILER_DIR=%s \\\n  %s -m vllm.entrypoints.cli.main %s\n' \
      "$TRACE" "$PY" "${args[*]}" > "$OUT/cmd.txt"
  setsid "$PY" -m vllm.entrypoints.cli.main "${args[@]}" > "$OUT/server.log" 2>&1 &
  PID=$!
  say "vLLM 启动中 pid=$PID（约 107s）"
  for _ in $(seq 900); do
    curl -s "localhost:$PORT/v1/models" 2>/dev/null | grep -qF "$TEXT_DIR" && { say "vLLM ready"; return; }
    kill -0 "$PID" 2>/dev/null || { say "启动失败"; tail -40 "$OUT/server.log" | tee -a "$LOG"; exit 1; }
    sleep 1
  done
  say "15 分钟未就绪"; exit 1
}

stop_vllm () {
  [ -n "${PID:-}" ] || return 0
  kill -- -"$PID" 2>/dev/null || kill "$PID" 2>/dev/null || true
  for _ in $(seq 60); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
  kill -9 -- -"$PID" 2>/dev/null || true
  pkill -f "vllm.entrypoints.cli.main serve" 2>/dev/null || true
  for _ in $(seq 40); do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2000 ] && { say "显存回落 ${used}MiB"; return; }
    sleep 2
  done
  say "显存未回落"
}

trap stop_vllm EXIT

say "== 场景 A/B/C 开始 =="
gpu_state
say "== 起服前确认：显存空闲 (见上) =="
start_vllm
gpu_state
{
  date -Iseconds
  git -C . rev-parse --short HEAD
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
  "$PY" -c 'import vllm,torch;print("vllm",vllm.__version__,"torch",torch.__version__,torch.version.cuda)'
} > "$OUT/env.txt" 2>&1 || true
curl -s "localhost:$PORT/metrics" 2>/dev/null | grep -E "cache_config_info" > "$OUT/cache-config.txt" || true

# ---------------- A: 满载 decode c=100 ----------------
say "== A 满载 decode c=100 n=4000 mt=32 =="
./target/release/wein-bench --url "http://127.0.0.1:$PORT" --model "$TEXT_DIR" \
    --concurrency 100 --num 4000 --max-tokens 32 --repeat 1 --warmup \
    --json "$OUT/A-load.json" > "$OUT/A-load.log" 2>&1 &
BENCH=$!
sleep 8
say "A: 稳态已建立，start_profile (max_iterations=200 ≈5s)"
curl -s -X POST "localhost:$PORT/start_profile" -H 'Content-Type: application/json' \
    -d '{"profile_prefix":"A-hot","max_iterations":200}' -o /dev/null -w '  start_profile HTTP %{http_code}\n' | tee -a "$LOG"
date +%s.%N > "$OUT/A-prof-window.txt"
sleep 14
curl -s -X POST "localhost:$PORT/stop_profile" -o /dev/null -w '  stop_profile HTTP %{http_code}\n' | tee -a "$LOG"
date +%s.%N >> "$OUT/A-prof-window.txt"
say "A: 等负载跑完"
wait $BENCH || true
say "A load: $(grep -E '"(itl_p50_ms|aggregate_tps|wall_s|errors)"' "$OUT/A-load.log" | tr '\n' ' ')"
ls -la "$TRACE" | tee -a "$LOG"

# ---------------- B: 6009 冷 prefill ----------------
say "== B 6009 冷 prefill =="
"$PY" "$OUT/probe.py" --url "http://127.0.0.1:$PORT" --model "$TEXT_DIR" --tokens 6009 \
    --max-tokens 32 --mode prefill --prefix B-prefill --max-iters 3 \
    --out "$OUT/B-probe.json" 2>&1 | tee -a "$LOG"

# ---------------- C: 6009 decode ----------------
say "== C 6009 decode =="
"$PY" "$OUT/probe.py" --url "http://127.0.0.1:$PORT" --model "$TEXT_DIR" --tokens 6009 \
    --max-tokens 32 --mode decode --prefix C-decode --delay 4 --max-iters 30 \
    --out "$OUT/C-probe.json" 2>&1 | tee -a "$LOG"

ls -la "$TRACE" | tee -a "$LOG"
say "== 停服 =="
stop_vllm
PID=""
gpu_state
say "== 完成 =="
