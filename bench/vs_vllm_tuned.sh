#!/usr/bin/env bash
# vLLM 0.32.0 调优扫参（只测量，不改 wein）。同一把尺子：wein-bench + longctx_probe。
#
# 用法：bench/vs_vllm_tuned.sh <档名> [<档名> ...]
#   档名见下方 case 表；每个档独立起服→测→停服，产物落
#   bench-reports/vs-vllm-tuned/<档名>/{cmd.txt,config.json,*.json,server.log}
#   单档失败（起不来/OOM/请求报错）只写 fail.txt 并继续下一档，不中断整批。
#
# GPU 互斥：一次一进程；停服后复核 pgrep 为空且显存回落到 idle 水位。
# 代理：WSL autoProxy 注入的 http_proxy 会劫持 127.0.0.1 —— 全清。
set -uo pipefail
cd "$(dirname "$0")/.."
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

TEXT_DIR=${WEIN_VLLM_MODEL:-$HOME/.cache/wein/Qwen3.5-4B-text}
TOK_DIR=$TEXT_DIR
PORT=${WEIN_VLLM_PORT:-8155}
MAXLEN=${WEIN_VLLM_MAXLEN:-8192}   # 长上下文扫描固定 8192（与上一轮基线同口径）
PY=~/venvs/vllm/bin/python
ROOT=bench-reports/vs-vllm-tuned
STARTUP_TIMEOUT=${WEIN_VLLM_STARTUP_TIMEOUT:-1500}

# ---------- 档位表：EXTRA=服务参数，XENV=环境变量 ----------
cfg_args () { # $1=档名 → 设置 GMU / EXTRA[] / XENV[]
  GMU=0.9                      # vLLM 默认档（0.85 是上一轮脚本的保守值，见 REPORT §0）
  EXTRA=()
  XENV=()
  case "$1" in
    default|default-r2|default-r3) EXTRA=(--max-num-batched-tokens 2048 --max-num-seqs 256) ;;
    mbt1024)   EXTRA=(--max-num-batched-tokens 1024 --max-num-seqs 256) ;;
    mbt4096)   EXTRA=(--max-num-batched-tokens 4096 --max-num-seqs 256) ;;
    mbt8192)   EXTRA=(--max-num-batched-tokens 8192 --max-num-seqs 256) ;;
    seqs128)   EXTRA=(--max-num-batched-tokens 2048 --max-num-seqs 128) ;;
    gpu95)     GMU=0.95; EXTRA=(--max-num-batched-tokens 2048 --max-num-seqs 256) ;;
    # 复现上一轮基线口径（gpu-mem 0.85）：用于把"0.9 的差"与"今日环境漂移"分开
    mem85)     GMU=0.85; EXTRA=(--max-num-batched-tokens 2048 --max-num-seqs 256) ;;
    # 反方向：降到 0.8 换更多显存余量（0.9 已证贴墙）
    gpu80)     GMU=0.80; EXTRA=(--max-num-batched-tokens 2048 --max-num-seqs 256) ;;
    kvfp8)     EXTRA=(--kv-cache-dtype fp8 --max-num-batched-tokens 2048 --max-num-seqs 256) ;;
    # fp8 KV 的替代路径：auto 会选 FlashInfer（本机 JIT 不可用，见 REPORT），
    # TRITON_ATTN 是唯一在 supports_kv_cache_dtypes 里声明支持 fp8 的可离线后端
    kvfp8triton) EXTRA=(--kv-cache-dtype fp8 --attention-backend TRITON_ATTN --max-num-batched-tokens 2048 --max-num-seqs 256) ;;
    ssmbf16)   EXTRA=(--mamba-ssm-cache-dtype bfloat16 --max-num-batched-tokens 2048 --max-num-seqs 256) ;;
    # 独立复跑（查首跑冷跑 1.7 s 是否为真）
    ssmbf16-r2|ssmbf16-r3) EXTRA=(--mamba-ssm-cache-dtype bfloat16 --max-num-batched-tokens 2048 --max-num-seqs 256) ;;
    flashinfer) XENV=(VLLM_ATTENTION_BACKEND=FLASHINFER)
                EXTRA=(--max-num-batched-tokens 2048 --max-num-seqs 256) ;;
    # 独立复跑（vLLM 0.32.0 源码里没有 VLLM_ATTENTION_BACKEND 引用 ⇒ 疑为环境一次性的慢档）
    flashinfer-r2) XENV=(VLLM_ATTENTION_BACKEND=FLASHINFER)
                EXTRA=(--max-num-batched-tokens 2048 --max-num-seqs 256) ;;
    # 真正的后端选择走 CLI 参数（0.32.0 的 VLLM_ATTENTION_BACKEND 环境变量是 no-op：
    # 启动日志会报 "Unknown vLLM environment variable detected"）
    flashinfer-cli) EXTRA=(--attention-backend FLASHINFER --max-num-batched-tokens 2048 --max-num-seqs 256) ;;
    async)     EXTRA=(--async-scheduling --max-num-batched-tokens 2048 --max-num-seqs 256) ;;
    noasync)   EXTRA=(--no-async-scheduling --max-num-batched-tokens 2048 --max-num-seqs 256) ;;
    # 关 chunked prefill ⇒ 单请求 prefill 必须一次装下 6009 token，故 mbt 必须 ≥ 8192
    nochunk)   EXTRA=(--no-enable-chunked-prefill --max-num-batched-tokens 8192 --max-num-seqs 256) ;;
    # mamba cache mode 'align'（prefix caching 开着时的默认）断言要求 chunked prefill，故需同时关 prefix caching
    nochunk-nopc) EXTRA=(--no-enable-chunked-prefill --no-enable-prefix-caching --max-num-batched-tokens 8192 --max-num-seqs 256) ;;
    # 默认 compilation mode 已是 VLLM_COMPILE(3)（见 REPORT §1）；该档是"证实无差"控制组
    compO3)    EXTRA=(--compilation-config '{"mode":3}' --max-num-batched-tokens 2048 --max-num-seqs 256) ;;
    # 默认桶集 1,2,4,8,16,24..512 步长 8（不含 100，c=100 落到 104 桶）；此档给精确 100 桶
    cg100)     EXTRA=(--compilation-config '{"mode":3,"cudagraph_capture_sizes":[1,2,4,8,16,32,64,100,128,192,256,384,512]}' --max-num-batched-tokens 2048 --max-num-seqs 256) ;;
    fp8quant)  EXTRA=(--quantization fp8 --max-num-batched-tokens 2048 --max-num-seqs 256) ;;
    # 组合最优：单变量赢的项叠加（gpu 余量 0.85 + 大 prefill 块 8192 + fp8 W8A8 + 精简图桶含 100）
    combo)     GMU=0.85
               EXTRA=(--quantization fp8 --max-num-batched-tokens 8192 --max-num-seqs 256
                      --compilation-config '{"mode":3,"cudagraph_capture_sizes":[1,2,4,8,16,32,64,100,128,192,256,384,512]}') ;;
    combo-ssmbf16) GMU=0.85
               EXTRA=(--quantization fp8 --mamba-ssm-cache-dtype bfloat16 --max-num-batched-tokens 8192 --max-num-seqs 256
                      --compilation-config '{"mode":3,"cudagraph_capture_sizes":[1,2,4,8,16,32,64,100,128,192,256,384,512]}') ;;
    mtp)       EXTRA=(--speculative-config '{"method":"mtp","num_speculative_tokens":1}' --max-num-batched-tokens 2048 --max-num-seqs 256) ;;
    *) return 1 ;;
  esac
}

# ---------- 环境修补（沿用 bench/vs_vllm.sh 的三条 WSL 经验）----------
prep_env () {
  # ① PATH 里 Windows CUDA 会盖住真 nvcc 错误；/mnt/c 的 ninja.exe 在 interop 下 EACCES
  PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/mnt/c/' | paste -sd:)
  # ② venv/bin 自带 Linux ninja（flashinfer/triton JIT 用；不激活 venv 时 PATH 里没有）
  export PATH="$HOME/venvs/vllm/bin:$PATH"
  local nvcc_dir
  nvcc_dir=$(dirname "$(find "$("$PY" -c 'import site; print(site.getsitepackages()[0])')/nvidia" -name nvcc -type f 2>/dev/null | head -1)" 2>/dev/null)
  [ -x "$nvcc_dir/nvcc" ] && { export CUDACXX="$nvcc_dir/nvcc"; PATH="$nvcc_dir:$PATH"; }
  # ③ flashinfer sampling JIT 与 torch cu130 头不同版；贪心基准采样非测量轴
  export VLLM_USE_FLASHINFER_SAMPLER=0
  # ④ 仅为 POST /reset_prefix_cache（冷跑清前缀），不进引擎性能路径
  export VLLM_SERVER_DEV_MODE=1
}

gpu_used_mib () { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }

stop_vllm () {
  kill -- -"$PID" 2>/dev/null || kill "$PID" 2>/dev/null || true
  for _ in $(seq 60); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
  kill -9 -- -"$PID" 2>/dev/null || true
  pkill -f "vllm.entrypoints.cli.main serve" 2>/dev/null || true
  # 显存回落复核（下一档要独占 GPU）
  for _ in $(seq 40); do
    [ "$(gpu_used_mib)" -lt 3000 ] 2>/dev/null && ! pgrep -f "vllm.entrypoints.cli.main serve" >/dev/null && break
    sleep 2
  done
  sleep 2
}

run_case () {
  local CFG=$1
  local OUT=$ROOT/$CFG
  rm -rf "$OUT"; mkdir -p "$OUT"
  if ! cfg_args "$CFG"; then echo "未知档: $CFG" >&2; return 1; fi
  prep_env
  local args=(serve "$TEXT_DIR" --port "$PORT" --max-model-len "$MAXLEN"
      --gpu-memory-utilization "$GMU"
      --hf-overrides '{"architectures": ["Qwen3_5ForCausalLM"]}'
      "${EXTRA[@]}")
  printf 'VLLM_SERVER_DEV_MODE=1 VLLM_USE_FLASHINFER_SAMPLER=0 %s %s -m vllm.entrypoints.cli.main %s\n' \
      "${XENV[*]:-}" "$PY" "${args[*]}" > "$OUT/cmd.txt"
  echo "== [$CFG] 起服 → $OUT =="
  # 上一档的 socket 可能还没释放：先等端口空闲，否则新进程 bind 失败而探针会被旧 server 答成"秒就绪"
  for _ in $(seq 60); do
    curl -s --max-time 1 "localhost:$PORT/v1/models" >/dev/null 2>&1 || break
    sleep 1
  done
  local T0; T0=$(date +%s)
  env "${XENV[@]:-VLLM_NOOP=1}" setsid "$PY" -m vllm.entrypoints.cli.main "${args[@]}" > "$OUT/server.log" 2>&1 &
  PID=$!
  local ready=0
  for _ in $(seq "$STARTUP_TIMEOUT"); do
    if curl -s "localhost:$PORT/v1/models" 2>/dev/null | grep -qF "$TEXT_DIR"; then ready=1; break; fi
    if ! kill -0 "$PID" 2>/dev/null; then
      # fail.txt 首行 = 真正的异常行（不是 Traceback 头），再补最后 15 行上下文
      { grep -oE "[A-Za-z_.]*(Error|Exception)[^ ]*: .{0,300}" "$OUT/server.log" | head -3; } > "$OUT/fail.txt" || true
      [ -s "$OUT/fail.txt" ] || echo "<无异常行，见 server.log>" > "$OUT/fail.txt"
      echo "---" >> "$OUT/fail.txt"; tail -15 "$OUT/server.log" >> "$OUT/fail.txt"
      echo "[$CFG] 起服失败（$(( $(date +%s)-T0 ))s），见 $OUT/fail.txt / server.log" >&2
      stop_vllm; return 1
    fi
    sleep 1
  done
  [ "$ready" = 1 ] || { echo "[$CFG] $STARTUP_TIMEOUT s 未就绪" >&2; stop_vllm; return 1; }
  local T1; T1=$(date +%s)
  echo "[$CFG] 就绪用时 $((T1-T0))s"

  # 运行期权威口径：cache_config_info + 启动日志里的回退/桶集/后端
  curl -s "localhost:$PORT/metrics" > "$OUT/metrics-after-start.txt" || true
  grep -E "cache_config_info" "$OUT/metrics-after-start.txt" > "$OUT/cache-config.txt" || true
  grep -E "non-default args|attention backend out of|Graph capturing|GPU KV cache size|Free memory on device|torch.compile took|CUDA graph memory profiling|enable_prefix_caching|mamba_cache_mode|mamba_ssm_cache_dtype|Setting attention block size|Padding mamba page|speculative|MTP|Async scheduling|async scheduling" \
      "$OUT/server.log" | sort -u > "$OUT/server-flags.txt" || true
  {
    echo "ts: $(date -Iseconds)"
    echo "cfg: $CFG"
    echo "git: $(git -C . rev-parse --short HEAD 2>/dev/null || echo n/a)"
    echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
    echo "vllm: $("$PY" -c 'import vllm;print(vllm.__version__)')"
    echo "torch: $("$PY" -c 'import torch;print(torch.__version__, torch.version.cuda)')"
    echo "model: $TEXT_DIR"
    echo "max_model_len: $MAXLEN"
    echo "gpu_memory_utilization: $GMU"
    echo "extra flags: ${EXTRA[*]}"
    echo "extra env: ${XENV[*]:-<none>}"
    echo "startup_s: $((T1-T0))"
    echo "gpu_used_after_start_mib: $(gpu_used_mib)"
  } | tee "$OUT/config.txt"

  # 显存/时钟采样（0.2s 粒度）——峰值显存与最低 SM 时钟都只取满载窗口（WSL 不能锁频，需留证据）
  ( while :; do nvidia-smi --query-gpu=memory.used,clocks.sm,temperature.gpu,power.draw --format=csv,noheader,nounits | tr -d ' '; sleep 0.2; done ) > "$OUT/vram-samples.txt" 2>/dev/null &
  local VS=$!

  echo "-- [$CFG] 短上下文 c=4 n=8 mt=12 ×3 --"
  ./target/release/wein-bench --url "http://127.0.0.1:$PORT" --model "$TEXT_DIR" \
      --concurrency 4 --num 8 --max-tokens 12 --repeat 3 --warmup \
      --json "$OUT/short-light-c4.json" | grep -E '"(ttft_p50_ms|ttft_p99_ms|itl_p50_ms|itl_p99_ms|aggregate_tps|errors|wall_s)"'
  echo "-- [$CFG] 满载 c=100 n=500 mt=32 ×3 --"
  local V0; V0=$(wc -l < "$OUT/vram-samples.txt")
  ./target/release/wein-bench --url "http://127.0.0.1:$PORT" --model "$TEXT_DIR" \
      --concurrency 100 --num 500 --max-tokens 32 --repeat 3 --warmup \
      --json "$OUT/short-full-c100.json" | grep -E '"(ttft_p50_ms|ttft_p99_ms|itl_p50_ms|itl_p99_ms|aggregate_tps|errors|wall_s)"'
  # 满载窗口的显存峰值 + 最低 SM 时钟（从满载开始那一行算起）
  tail -n +"$((V0+1))" "$OUT/vram-samples.txt" > "$OUT/load-samples.txt" 2>/dev/null || true
  cut -d, -f1 "$OUT/load-samples.txt" | sort -n | tail -1 > "$OUT/vram-peak-c100.txt" 2>/dev/null || true
  cut -d, -f2 "$OUT/load-samples.txt" | sort -n | head -1 > "$OUT/sm-clock-min-c100.txt" 2>/dev/null || true
  echo "[$CFG] 满载窗口峰值显存: $(cat "$OUT/vram-peak-c100.txt" 2>/dev/null) MiB, 最低 SM 时钟: $(cat "$OUT/sm-clock-min-c100.txt" 2>/dev/null) MHz"
  kill "$VS" 2>/dev/null || true

  echo "-- [$CFG] 长上下文 3000 token 冷/热 ×3 --"
  "$PY" bench/longctx_probe.py --url "http://127.0.0.1:$PORT" --model "$TEXT_DIR" \
      --tokenizer "$TOK_DIR" --tokens 3000 --max-tokens 32 --repeats 3 \
      --cold --hot --note "cfg=$CFG maxlen=$MAXLEN" --out "$OUT/long-3000.json" || echo "[$CFG] long-3000 失败" >&2
  echo "-- [$CFG] 长上下文 6000 token 冷/热 ×3 --"
  "$PY" bench/longctx_probe.py --url "http://127.0.0.1:$PORT" --model "$TEXT_DIR" \
      --tokenizer "$TOK_DIR" --tokens 6000 --max-tokens 32 --repeats 3 \
      --cold --hot --note "cfg=$CFG maxlen=$MAXLEN" --out "$OUT/long-6000.json" || echo "[$CFG] long-6000 失败" >&2

  echo "== [$CFG] 停服 =="
  stop_vllm
  echo "[$CFG] 停服后显存: $(gpu_used_mib) MiB, pgrep=$(pgrep -fc 'vllm.entrypoints.cli.main serve' 2>/dev/null || echo 0)"
  return 0
}

FAILED=()
# 异常退出也不能留 vLLM 占卡（PID 未设时不动作，避免 kill -0 打到自己进程组）
cleanup () { [ -n "${PID:-}" ] && kill -- -"$PID" 2>/dev/null || true; }
trap cleanup EXIT
for CFG in "$@"; do
  run_case "$CFG" || FAILED+=("$CFG")
done
echo "=== 完成: 成功 $(( $# - ${#FAILED[@]} ))/$# ；失败: ${FAILED[*]:-<none>} ==="
