#!/usr/bin/env bash
# vLLM 长上下文基线（只测量，不改 wein）。与 bench/vs_vllm.sh 同款环境修补 + 同一把尺子。
#
# 用法：bench/vs_vllm_longctx.sh [default|bf16ssm|chunk256]
#   default  : 全量——短上下文 sanity（3 档）+ 3000/6000 冷/热，3 次取中位
#   bf16ssm  : 非默认档 --mamba-ssm-cache-dtype bfloat16（模型 config 是 float32），只测长上下文
#   chunk256 : 非默认档 --max-num-batched-tokens 256（对齐 wein chunk_size=256），只测 6000
#
# 输出：bench-reports/vs-vllm-longctx/<config>/{short-*.json,long-*.json,server.log,config.json}
# GPU 互斥：一次一进程；跑完 kill 整组（vLLM 有子进程，必须杀进程组）。
set -euo pipefail
cd "$(dirname "$0")/.."
# 基准必须直连：环境代理（WSL autoProxy 注入的 http_proxy）会劫持 127.0.0.1 流量
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

CFG=${1:-default}
TEXT_DIR=${WEIN_VLLM_MODEL:-$HOME/.cache/wein/Qwen3.5-4B-text}
TOK_DIR=$TEXT_DIR
PORT=${WEIN_VLLM_PORT:-8155}
OUT=bench-reports/vs-vllm-longctx/$CFG
mkdir -p "$OUT"
MAXLEN=${WEIN_VLLM_MAXLEN:-8192}
PY=~/venvs/vllm/bin/python

EXTRA=()
# 调度默认值：24GB 非 H100 档 + OPENAI_API_SERVER → max_num_batched_tokens=2048,
# max_num_seqs=256（arg_utils.py:2991-3003）。显式钉住 = 可记录、可复现，不改行为。
SCHED=(--max-num-batched-tokens 2048 --max-num-seqs 256)
case "$CFG" in
  default) ;;
  bf16ssm) EXTRA=(--mamba-ssm-cache-dtype bfloat16) ;;
  chunk256) SCHED=(--max-num-batched-tokens 256 --max-num-seqs 256) ;;
  maxlen512) MAXLEN=512 ;;  # 旧基线 full-c100.json 的口径（max-model-len 512），只做短上下文
  *) echo "未知配置: $CFG" >&2; exit 1 ;;
esac

start_vllm () {
  # WSL 陷阱（同 vs_vllm.sh）：PATH 里的 Windows CUDA 会盖住真 nvcc 错误
  PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/mnt/c/.*CUDA' | paste -sd:)
  local nvcc_dir
  nvcc_dir=$(dirname "$(find "$("$PY" -c 'import site; print(site.getsitepackages()[0])')/nvidia" -name nvcc -type f 2>/dev/null | head -1)" 2>/dev/null)
  [ -x "$nvcc_dir/nvcc" ] && { export CUDACXX="$nvcc_dir/nvcc"; PATH="$nvcc_dir:$PATH"; }
  export VLLM_USE_FLASHINFER_SAMPLER=0
  # dev 端点仅为 /reset_prefix_cache（冷跑前清前缀缓存）；不影响引擎性能路径
  export VLLM_SERVER_DEV_MODE=1
  local args=(serve "$TEXT_DIR" --port "$PORT" --max-model-len "$MAXLEN"
      --gpu-memory-utilization 0.85
      --hf-overrides '{"architectures": ["Qwen3_5ForCausalLM"]}'
      "${SCHED[@]}" ${EXTRA[@]+"${EXTRA[@]}"})
  printf 'VLLM_SERVER_DEV_MODE=1 %s -m vllm.entrypoints.cli.main %s\n' "$PY" "${args[*]}" > "$OUT/cmd.txt"
  # setsid：让 vLLM 自成进程组，停服时 kill -- -PID 才只杀它（否则会连脚本自己一起杀）
  setsid "$PY" -m vllm.entrypoints.cli.main "${args[@]}" > "$OUT/server.log" 2>&1 &
  PID=$!
  for _ in $(seq 900); do
    curl -s "localhost:$PORT/v1/models" 2>/dev/null | grep -qF "$TEXT_DIR" && return
    kill -0 "$PID" 2>/dev/null || { echo "vLLM 启动失败，见 $OUT/server.log" >&2; tail -30 "$OUT/server.log"; exit 1; }
    sleep 1
  done
  echo "vLLM 15 分钟未就绪" >&2; exit 1
}

stop_vllm () {
  kill -- -"$PID" 2>/dev/null || kill "$PID" 2>/dev/null || true
  for _ in $(seq 60); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
  kill -9 -- -"$PID" 2>/dev/null || true
  # 复核：不留孤儿引擎进程，等显存释放（下一档要独占 GPU）
  for _ in $(seq 30); do
    pgrep -f "vllm.entrypoints.cli.main serve" >/dev/null || break; sleep 2
  done
  pkill -f "vllm.entrypoints.cli.main serve" 2>/dev/null || true
  sleep 2
}

short_load () { # $1=档名 $2=c $3=n $4=max_tokens
  echo "-- 短上下文 $1 · c=$2 n=$3 max_tokens=$4 ×3 --"
  ./target/release/wein-bench --url "http://127.0.0.1:$PORT" --model "$TEXT_DIR" \
      --concurrency "$2" --num "$3" --max-tokens "$4" --repeat 3 --warmup \
      --json "$OUT/short-$1.json" | grep -E '"(ttft_p50_ms|ttft_p99_ms|itl_p50_ms|itl_p99_ms|aggregate_tps|errors|wall_s)"'
}

long_probe () { # $1=token 数 $2=额外参数
  echo "-- 长上下文 ${1} token 冷/热 ×3 --"
  local cold=--cold hot=--hot
  if [ "${2:-}" = "cold" ]; then hot=; fi
  "$PY" bench/longctx_probe.py --url "http://127.0.0.1:$PORT" --model "$TEXT_DIR" \
      --tokenizer "$TOK_DIR" --tokens "$1" --max-tokens 32 --repeats 3 \
      $cold $hot --note "cfg=$CFG maxlen=$MAXLEN" --out "$OUT/long-$1.json"
}

dump_config () {
  echo "== 配置 $CFG =="
  {
    echo "ts: $(date -Iseconds)"
    echo "cfg: $CFG"
    echo "git: $(git -C . rev-parse --short HEAD 2>/dev/null || echo n/a)"
    echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
    echo "vllm: $("$PY" -c 'import vllm;print(vllm.__version__)')"
    echo "torch: $("$PY" -c 'import torch;print(torch.__version__, torch.version.cuda)')"
    echo "model: $TEXT_DIR"
    echo "max_model_len: $MAXLEN"
    echo "sched flags: ${SCHED[*]}"
    echo "extra flags: ${EXTRA[*]:-<defaults>}"
  } | tee "$OUT/config.txt"
  grep -E "enable_prefix_caching|mamba_cache_mode|mamba_ssm_cache_dtype|max_num_batched_tokens|block size|Padding mamba page|KV cache size|torch.compile|Graph capturing" \
      "$OUT/server.log" | sort -u > "$OUT/server-flags.txt" || true
}

echo "== vs-vllm-longctx [$CFG] → $OUT =="
start_vllm
trap 'stop_vllm' EXIT
dump_config
# 附加：服务端自报的 config（metrics 的 cache_config_info 是唯一权威运行期口径）
curl -s "localhost:$PORT/metrics" > "$OUT/metrics-after-start.txt" || true
grep -E "cache_config_info" "$OUT/metrics-after-start.txt" > "$OUT/cache-config.txt" || true

case "$CFG" in
  default)
    short_load light-c4 4 8 12
    short_load c100-n8 100 8 12
    short_load full-c100 100 500 32
    long_probe 3000
    long_probe 6000
    ;;
  bf16ssm)
    long_probe 3000
    long_probe 6000
    ;;
  chunk256)
    long_probe 6000
    ;;
  maxlen512)
    short_load light-c4 4 8 12
    short_load full-c100 100 500 32
    ;;
esac

echo "== [$CFG] 完成，停服 =="
