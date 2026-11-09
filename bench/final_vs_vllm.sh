#!/usr/bin/env bash
# 终验：wein vs vLLM 成对交错测量（同一把尺子、同一题面、同一环境）
#
# 用法：bench/final_vs_vllm.sh [rounds]
#
# 设计要点（为什么这样测）：
# 1. **同一客户端**：短上下文用 wein-bench（OpenAI 协议，两端同一份实现），
#    长上下文用 longctx_probe.py（同一脚本、同一 tokenizer 副本）——排除
#    "客户机不同"这类系统性偏差。
# 2. **同一题面**：wein-bench 固定发 "bench" 三 token prompt；longctx_probe
#    用 $TEXT_DIR 的 tokenizer 生成确定性长 prompt。两端见到的 token id 全同。
# 3. **成对交错**：每轮 = wein 块 → vLLM 块，轮内配置顺序固定。跨轮取中位，
#    同时报**逐轮配对差**（热漂移/时钟漂移的一阶消去）。
# 4. **口径并列**：vLLM 跑两档——`bf16ssm`（`--mamba-ssm-cache-dtype bfloat16`，
#    与 wein 的 bf16 状态同口径）与 `default`（模型 config 声明的 f32，vLLM 默认）。
#    wein 无 f32 状态档，只有一个口径。
# 5. **环境纪律**：每块前后查 Windows 侧 compute-apps（WSL 看不到）、记录
#    SM/MEM 时钟与温度、代理变量一律 unset、两侧 setsid 自成进程组。
set -uo pipefail
cd "$(dirname "$0")/.."
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

ROUNDS=${1:-3}
PORT=${WEIN_VLLM_PORT:-8299}
DIR=${WEIN_MODEL_DIR:-$HOME/.cache/wein/Qwen3.5-4B}
TEXT_DIR=${WEIN_VLLM_MODEL:-$HOME/.cache/wein/Qwen3.5-4B-text}
PY=~/venvs/vllm/bin/python
OUT=bench-reports/final-vs-vllm-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
SMINT=/mnt/c/Windows/System32/nvidia-smi.exe
GWIN=bench/longctx_probe.py

say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }
gpu_watch () { # $1=标签
  {
    echo "=== $1"
    "$SMINT" --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used \
        --format=csv,noheader 2>/dev/null
    "$SMINT" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null
  } >> "$OUT/gpu.log" 2>&1
}
kill_all () {
  for p in $(pgrep -f "wein-cli --port $PORT" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  pkill -f "vllm.entrypoints.cli.main serve" 2>/dev/null
  for _ in $(seq 40); do
    pgrep -f "vllm.entrypoints.cli.main serve|wein-cli --port $PORT" >/dev/null || return
    sleep 1
  done
}

start_wein () {
  setsid ./target/release/wein-cli --port "$PORT" --engine core --exec qwen \
      --model "$DIR" --token-delay-ms 0 --graph \
      --ctx-len 8192 --kv-pages 512 --chunk-size 2048 \
      > "$OUT/wein-server.log" 2>&1 &
  WPID=$!
  for _ in $(seq 300); do
    curl -s "localhost:$PORT/health" 2>/dev/null | grep -q '"status":"ok"' && return 0
    sleep 1
  done
  say "wein 未就绪"; return 1
}
stop_wein () { kill -- -"$WPID" 2>/dev/null; sleep 2; kill_all; }

start_vllm () { # $1=extra flags
  PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '^/mnt/c/.*CUDA' | paste -sd:)
  local nvcc_dir
  nvcc_dir=$(dirname "$(find "$("$PY" -c 'import site; print(site.getsitepackages()[0])')/nvidia" -name nvcc -type f 2>/dev/null | head -1)" 2>/dev/null)
  [ -x "$nvcc_dir/nvcc" ] && { export CUDACXX="$nvcc_dir/nvcc"; PATH="$nvcc_dir:$PATH"; }
  export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_SERVER_DEV_MODE=1
  setsid "$PY" -m vllm.entrypoints.cli.main serve "$TEXT_DIR" --port "$PORT" \
      --max-model-len 8192 --gpu-memory-utilization 0.85 \
      --hf-overrides '{"architectures": ["Qwen3_5ForCausalLM"]}' \
      --max-num-batched-tokens 2048 --max-num-seqs 256 ${1:-} \
      > "$OUT/vllm-server-$2.log" 2>&1 &
  VPID=$!
  for _ in $(seq 900); do
    curl -s "localhost:$PORT/v1/models" 2>/dev/null | grep -qF "$TEXT_DIR" && return 0
    kill -0 "$VPID" 2>/dev/null || { say "vLLM 启动失败"; tail -20 "$OUT/vllm-server-$2.log" | tee -a "$OUT/run.log"; return 1; }
    sleep 1
  done
  say "vLLM 未就绪"; return 1
}
stop_vllm () {
  kill -- -"$VPID" 2>/dev/null; sleep 2
  kill_all
  sleep 3
}

short () { # $1=side $2=dir $3=tag $4=c $5=n $6=mt $7=repeat
  local m=$DIR; [ "$1" = vllm ] && m=$TEXT_DIR
  ./target/release/wein-bench --url "http://127.0.0.1:$PORT" --model "$m" \
      --concurrency "$4" --num "$5" --max-tokens "$6" --repeat "$7" --warmup \
      --json "$2/$3.json" > /dev/null 2>&1
  printf '  %-12s ' "$3"
  /usr/bin/python3 - "$2/$3.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))['runs']
def med(k):
    v=sorted(x[k] for x in r); return v[len(v)//2]
print("tps %7.1f  itl50 %6.2f  itl99 %6.2f  ttft50 %6.1f  ttft99 %7.1f  err %d"
      % (med('aggregate_tps'), med('itl_p50_ms'), med('itl_p99_ms'),
         med('ttft_p50_ms'), med('ttft_p99_ms'), sum(x.get('errors',0) for x in r)))
PY
}
longctx () { # $1=dir $2=side $3=tokens
  local m=$TEXT_DIR; [ "$2" = wein ] && m=$DIR
  "$PY" "$GWIN" --url "http://127.0.0.1:$PORT" --model "$m" \
      --tokenizer "$TEXT_DIR" --tokens "$3" --max-tokens 32 --repeats 3 --cold --hot \
      --out "$1/long-$3.json" > /dev/null 2>&1
  printf '  %-12s ' "long-$3"
  /usr/bin/python3 -c "
import json
j=json.load(open('$1/long-$3.json'))['median']
print('cold_ttft %8.1f  hot_ttft %6.1f  cold_itl50 %6.2f  hot_itl50 %6.2f' % (
  j['cold_ttft_ms'] or 0, j['hot_ttft_ms'] or 0,
  j['cold_itl_p50_ms'] or 0, j['hot_itl_p50_ms'] or 0))"
}

run_block () { # $1=side $2=dir $3=extra $4=tag $5=long
  mkdir -p "$2"
  say "块开始：$1 ($4)"
  gpu_watch "before-$4"
  if [ "$1" = wein ]; then start_wein || return 1; else start_vllm "$3" "$4" || return 1; fi
  gpu_watch "after-start-$4"
  short "$1" "$2" full-c100 100 4000 32 2
  short "$1" "$2" mid-c16   16  400 32 3
  short "$1" "$2" light-c4  4   64  32 3
  short "$1" "$2" single-c1 1   8   64 3
  if [ "${5:-}" = long ]; then
    longctx "$2" "$1" 3009
    longctx "$2" "$1" 6009
  fi
  gpu_watch "before-stop-$4"
  if [ "$1" = wein ]; then stop_wein; else stop_vllm; fi
  say "块结束：$1 ($4)"
}

kill_all
say "终验开始 → $OUT"
{
  echo "wein git: $(git rev-parse --short HEAD)"
  echo "vllm: $("$PY" -c 'import vllm;print(vllm.__version__)' 2>/dev/null)"
  echo "torch: $("$PY" -c 'import torch;print(torch.__version__, torch.version.cuda)' 2>/dev/null)"
  echo "gpu: $("$SMINT" --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader 2>/dev/null)"
  echo "wein cfg: --graph --ctx-len 8192 --kv-pages 512 --chunk-size 2048 (mix_cap 4096 + mix_budget 26ms/rows32 默认)"
  echo "vllm cfg: maxlen 8192 mem 0.85 mbt 2048 maxseqs 256"
} | tee "$OUT/env.txt"

for r in $(seq "$ROUNDS"); do
  run_block wein "$OUT/r$r-wein" "" "r$r-wein" long
  run_block vllm "$OUT/r$r-vllm-bf16" "--mamba-ssm-cache-dtype bfloat16" "r$r-vllm-bf16" long
  # 口径①（vLLM 模型声明的 f32 状态）只跑满载一项，省时
  say "块开始：vllm-f32 (r$r)"
  mkdir -p "$OUT/r$r-vllm-f32"
  if start_vllm "" "r$r-vllm-f32"; then
    short vllm "$OUT/r$r-vllm-f32" full-c100 100 4000 32 2
    stop_vllm
  fi
done

say "== 汇总 =="
/usr/bin/python3 - "$OUT" <<'PY' | tee -a "$OUT/run.log"
import json,glob,os,sys,statistics as st
out=sys.argv[1]
rows={}
for f in sorted(glob.glob(os.path.join(out,'r*-*','*.json'))):
    side=os.path.basename(os.path.dirname(f))
    # side like r1-wein / r1-vllm-bf16
    _,tag=side.split('-',1)
    name=os.path.basename(f)[:-5]
    try: j=json.load(open(f))
    except Exception: continue
    if 'runs' in j:
        runs=j['runs']
        def med(k):
            v=sorted(x[k] for x in runs); return v[len(v)//2]
        d={k:med(k) for k in ('aggregate_tps','itl_p50_ms','itl_p99_ms','ttft_p50_ms','ttft_p99_ms')}
        d['errors']=sum(x.get('errors',0) for x in runs)
    else:
        mm=j.get('median',{})
        d={'ttft_p50_ms':mm.get('cold_ttft_ms'),'hot_ttft50':mm.get('hot_ttft_ms'),
           'itl_p50_ms':mm.get('hot_itl_p50_ms'),'itl_p99_ms':None}
    rows.setdefault((name,tag),[]).append(d)
print("%-12s %-12s %8s %8s %8s %8s %8s" % ("config","side","tps","itl50","itl99","ttft50","n"))
for (name,tag),ds in sorted(rows.items()):
    def m(k):
        v=[d[k] for d in ds if d.get(k) is not None]
        return st.median(v) if v else float('nan')
    print("%-12s %-12s %8.1f %8.2f %8.2f %8.1f %8d" % (name,tag,m('aggregate_tps'),m('itl_p50_ms'),m('itl_p99_ms'),m('ttft_p50_ms'),len(ds)))
PY
say "完成 → $OUT"
