#!/usr/bin/env bash
# wein 性能基线电池——归因阶梯的执行器（方法论文档：bench-reports/BASELINE.md）
#
#   L0   路由上限        flood GET /v1/models（与引擎无关）
#   L1   serve 层开销    零延迟 mock（--token-delay-ms 0）下的 TTFT/ITL —— serve 层的"纯账"
#   L1'  节拍泵参照      1ms/token mock（对照基线；真实引擎接入后退役为 L2）
#   L2   engine core     真调度器（wein-sched）+ 脚本执行器，零步延迟 —— 调度层的"纯账"
#   L2'  节拍泵(调度器)  1ms/步，与 L1' 对照：同节拍下 mock 与真调度器的端到端差
#   L2m  内存层记账      L2 + BlockPool 水位/radix/DeferredRelease —— 内存层的"纯账"
#   L3   exec 层         真执行器（DeviceSession + toy 模型，真 GPU 步）—— exec 层的"纯账"
#   L3q  真模型竖切      Qwen3.5-4B 真权重（I1）—— 混合骨干全链路的 sanity 账
#                        （小负载 c=4/n=8：批组成受 MAX_ROWS/T_MAX 约束，
#                         量的是 TTFT（真权重追平）与 ITL（~34ms/步的账），非满负荷承诺）
#
# 用法：bench/battery.sh [release|debug]     （默认 release；报告落 bench-reports/battery-<时间戳>/）
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE=${1:-release}
PORT=${WEIN_BENCH_PORT:-8125}
BIN=target/$PROFILE
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=bench-reports/battery-$STAMP
export WEIN_GIT_REV=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
export WEIN_PROFILE=$PROFILE
mkdir -p "$OUT"
echo "== wein battery  $STAMP  profile=$PROFILE  rev=$WEIN_GIT_REV  → $OUT =="

run_suite () { # $1=tag；$2.. 传给 wein-cli 的延迟参数（模型 id = 就绪探针的 grep 目标）
  local tag=$1; shift
  local ready=${WEIN_READY:-mock-model}
  "$BIN/wein-cli" --port "$PORT" "$@" >/dev/null 2>&1 &
  local pid=$!
  trap "kill $pid 2>/dev/null || true" EXIT
  until curl -s "localhost:$PORT/v1/models" 2>/dev/null | grep -q "$ready"; do sleep 0.2; done

  echo "-- $tag · L0 router flood --"
  "$BIN/wein-bench" --url "http://127.0.0.1:$PORT" --concurrency 100 --num 20000 \
      --flood --json "$OUT/$tag-l0-router.json" | grep -E '"(qps|lat_p50_ms|lat_p99_ms|errors)"'
  echo "-- $tag · L1 streaming (repeat=3, 中位数) --"
  "$BIN/wein-bench" --url "http://127.0.0.1:$PORT" --concurrency 100 --num 500 --max-tokens 32 \
      --repeat 3 --warmup --json "$OUT/$tag-l1-stream.json" \
      | grep -E '"(ttft_p50_ms|ttft_p99_ms|itl_p50_ms|itl_p99_ms|aggregate_tps|errors)"'

  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  trap - EXIT
}

# 真模型竖切（L3q）：轻载 sanity 账（c=4/n=8/12 tok）——exec 即节拍器，
# TTFT 含真权重追平、ITL 是 ~34ms/步的账。满负荷 profile（c=100/n=500）
# 会撞 exec 的 MAX_ROWS/T_MAX 批界，不适用于本层。模型不在场则跳过（与 q 系列同门）
run_qwen_suite () {
  local dir=${WEIN_MODEL_DIR:-$HOME/.cache/wein/Qwen3.5-4B}
  if [ ! -d "$dir" ]; then
    echo "-- L3q SKIP（模型目录不在场：$dir）--"
    return
  fi
  local tag=core-qwen
  "$BIN/wein-cli" --port "$PORT" --engine core --exec qwen --model "$dir" --token-delay-ms 0 \
      >/dev/null 2>&1 &
  local pid=$!
  trap "kill $pid 2>/dev/null || true" EXIT
  until curl -s "localhost:$PORT/v1/models" 2>/dev/null | grep -qF "$dir"; do sleep 0.2; done

  echo "-- $tag · L3q streaming (轻载 c=4 n=8 max_tokens=12) --"
  "$BIN/wein-bench" --url "http://127.0.0.1:$PORT" --model "$dir" --concurrency 4 --num 8 \
      --max-tokens 12 --repeat 1 --warmup --json "$OUT/$tag-l1-stream.json" \
      | grep -E '"(ttft_p50_ms|ttft_p99_ms|itl_p50_ms|itl_p99_ms|aggregate_tps|errors)"'

  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  trap - EXIT
}

run_suite zero-delay --engine mock --prefill-delay-ms 0 --token-delay-ms 0
run_suite paced-1ms --engine mock --prefill-delay-ms 2 --token-delay-ms 1
run_suite core-zero --engine core --token-delay-ms 0
run_suite core-paced-1ms --engine core --token-delay-ms 1
run_suite core-mem-zero --engine core --token-delay-ms 0 --memory
run_suite core-exec-zero --engine core --exec toy --token-delay-ms 0
run_qwen_suite

python3 - "$OUT" "$STAMP" <<'PYEOF' > "$OUT/SUMMARY.md"
import json, sys
out, stamp = sys.argv[1], sys.argv[2]
def load(name):
    try:
        with open(f"{out}/{name}") as f: return json.load(f)["result"]
    except FileNotFoundError: return None
rows = []
r = load("zero-delay-l0-router.json")
if r: rows.append(("L0 路由上限", f"qps={r['qps']:.0f} p50={r['lat_p50_ms']:.2f}ms p99={r['lat_p99_ms']:.2f}ms"))
r = load("zero-delay-l1-stream.json")
if r: rows.append(("L1 serve 纯开销(零延迟引擎)", f"ttft_p50={r['ttft_p50_ms']:.2f}ms ttft_p99={r['ttft_p99_ms']:.2f}ms itl_p50={r['itl_p50_ms']:.3f}ms itl_p99={r['itl_p99_ms']:.2f}ms tps={r['aggregate_tps']:.0f}"))
r = load("paced-1ms-l1-stream.json")
if r: rows.append(("L1' 节拍泵(1ms/token) 参照", f"ttft_p50={r['ttft_p50_ms']:.2f}ms itl_p50={r['itl_p50_ms']:.2f}ms itl_p99={r['itl_p99_ms']:.2f}ms tps={r['aggregate_tps']:.0f}"))
r = load("core-zero-l1-stream.json")
if r: rows.append(("L2 engine core 开销(零延迟执行器)", f"ttft_p50={r['ttft_p50_ms']:.2f}ms ttft_p99={r['ttft_p99_ms']:.2f}ms itl_mean={r.get('itl_mean_ms', 0):.3f}ms tps={r['aggregate_tps']:.0f}"))
r = load("core-paced-1ms-l1-stream.json")
if r: rows.append(("L2' 节拍泵(1ms/步, 真调度器)", f"ttft_p50={r['ttft_p50_ms']:.2f}ms itl_mean={r.get('itl_mean_ms', 0):.2f}ms tps={r['aggregate_tps']:.0f}"))
r = load("core-mem-zero-l1-stream.json")
if r: rows.append(("L2m 内存层记账开销(core+BlockPool/radix, 零延迟)", f"ttft_p50={r['ttft_p50_ms']:.2f}ms ttft_p99={r['ttft_p99_ms']:.2f}ms itl_mean={r.get('itl_mean_ms', 0):.3f}ms tps={r['aggregate_tps']:.0f}"))
r = load("core-exec-zero-l1-stream.json")
if r: rows.append(("L3 exec 层开销(core+真执行器 toy, 真 GPU 步)", f"ttft_p50={r['ttft_p50_ms']:.2f}ms ttft_p99={r['ttft_p99_ms']:.2f}ms itl_mean={r.get('itl_mean_ms', 0):.2f}ms tps={r['aggregate_tps']:.0f}"))
r = load("core-qwen-l1-stream.json")
if r: rows.append(("L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4)", f"ttft_p50={r['ttft_p50_ms']:.2f}ms ttft_p99={r['ttft_p99_ms']:.2f}ms itl_p50={r['itl_p50_ms']:.2f}ms tps={r['aggregate_tps']:.1f}"))
print(f"# battery {stamp}\n")
print("| 层 | 指标（中位数; L0-L3: c=100, n=500, max_tokens=32; L3q: c=4, n=8, max_tokens=12） |")
print("|---|---|")
for name, m in rows: print(f"| {name} | {m} |")
PYEOF

echo "== summary =="
cat "$OUT/SUMMARY.md"
