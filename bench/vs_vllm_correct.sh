#!/usr/bin/env bash
# 正确性对拍：wein vs vLLM
#   轴 1：贪心生成——completions 原始 prompt（不经 chat 模板，排除模板差变量），
#         temperature=0 两端同为贪心；对拍文本公共前缀 + completion_tokens
#   轴 2：chat 模板交叉验证——同一 chat 请求打两端，比 prompt_tokens
#         （vLLM 用 HF transformers 同款 Jinja 渲染——wein W5 minijinja 的外部 oracle）
# GPU 互斥：capture 分侧执行（先起一侧服务采完停，再换侧），最后 compare 纯本地。
#
# 用法：bench/vs_vllm_correct.sh capture <wein|vllm> <base_url>
#       bench/vs_vllm_correct.sh compare
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=bench-reports/vs-vllm-correct
CMD=${1:?用法: vs_vllm_correct.sh capture <wein|vllm> <base_url> | compare}

# --- 固定 prompt 组（成对覆盖：事实续写 / 模式延续 / 中文经典 / 重复陷阱）---
PROMPTS=(
  "The capital of France is"
  "1, 2, 3, 4, 5, 6,"
  "床前明月光，"
  "The quick brown fox jumps over the lazy dog. The quick brown fox"
  "def fibonacci(n):"
)

capture () { # $1=side $2=base_url
  local side=$1 url=$2 model=${WEIN_MODEL_DIR:-$HOME/.cache/wein/Qwen3.5-4B}
  mkdir -p "$OUT/$side"
  local i=0
  for p in "${PROMPTS[@]}"; do
    i=$((i+1))
    curl -s "$url/v1/completions" -H 'content-type: application/json' -d "$(python3 - "$p" "$model" <<'PY'
import json, sys
print(json.dumps({
  "model": sys.argv[2], "prompt": sys.argv[1], "max_tokens": 48,
  "temperature": 0, "stream": False}))
PY
)" > "$OUT/$side/resp-$i.json"
  done
  # 轴 2：chat 模板 prompt_tokens 交叉验证（与 battery 同款流量）
  python3 -c "import json,sys;print(json.dumps({
    'model': sys.argv[1], 'messages': [{'role': 'user', 'content': 'bench'}],
    'max_tokens': 4, 'temperature': 0, 'stream': False}))" "$model" \
    | curl -s "$url/v1/chat/completions" -H 'content-type: application/json' -d @- > "$OUT/$side/chat.json"
  echo "[$side] captured: ${#PROMPTS[@]} completions + 1 chat → $OUT/$side/"
}

compare () {
  python3 - "$OUT" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
def load(side, name):
    p = out / side / name
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return {"__err__": f"{e}"}
n = 5
print(f"{'#':>2} {'前缀=':>7} {'wein':>5} {'vllm':>5}  文本（| 分界=首个分歧点）")
for i in range(1, n+1):
    w, v = load("wein", f"resp-{i}.json"), load("vllm", f"resp-{i}.json")
    wt = w.get("choices", [{}])[0].get("text", "ERR") if "__err__" not in w else "ERR"
    vt = v.get("choices", [{}])[0].get("text", "ERR") if "__err__" not in v else "ERR"
    k = next((j for j in range(min(len(wt), len(vt))) if wt[j] != vt[j]), min(len(wt), len(vt)))
    wc = w.get("usage", {}).get("completion_tokens", "?")
    vc = v.get("usage", {}).get("completion_tokens", "?")
    mark = "✓" if k == len(wt) == len(vt) else (" prefix" if k >= 8 else " DIVERGE")
    print(f"{i:>2} {k:>7} {wc:>5} {vc:>5}  {wt[:k]}|{wt[k:k+12]}⋯ / {vt[k:k+12]}⋯{mark}")
wc, vc = load("wein", "chat.json"), load("vllm", "chat.json")
print(f"\nchat 模板 prompt_tokens: wein={wc.get('usage',{}).get('prompt_tokens','ERR')}"
      f"  vllm={vc.get('usage',{}).get('prompt_tokens','ERR')}"
      f"  {'✓ 模板渲染一致' if wc.get('usage',{}).get('prompt_tokens')==vc.get('usage',{}).get('prompt_tokens') else '✗ 需查'}")
PY
}

case "$CMD" in
  capture) capture "$2" "$3" ;;
  compare) compare ;;
  *) echo "未知子命令: $CMD" >&2; exit 1 ;;
esac
