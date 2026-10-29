"""参照端逐层激活落盘：HF transformers（vLLM 同源数学）跑文本骨干。

与 wein 的 WEIN_TRACE_DIR 产物同构：`s{n}-embed.f32` / `s{n}-layer-{i:02}.f32`
/ `s{n}-logits.f32`（f32 LE 行主序 [t, dim]）。

用法（venv python）:
  python bench/trace_reference.py <text_model_dir> <out_dir> <tok_id,...> [max_tokens] [wein_tokens,...]

- 预填步 s1：整段 prompt 前向（t = len(prompt)，与 wein 预填批同形状）
- 解码步 s2..：**forced decode**——喂 wein 生成的 token 链（两链在分歧前
  同路，分歧后比各自的 forward 数值而非生成一致；定位用 forced 对齐）
- 同时输出 HF 自家贪心链（对照）
"""
import sys, os, json
import numpy as np
import torch
from transformers import Qwen3_5ForCausalLM

model_dir, out_dir = sys.argv[1], sys.argv[2]
ids = [int(x) for x in sys.argv[3].split(",")]
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 4
wein_toks = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else []
os.makedirs(out_dir, exist_ok=True)

m = Qwen3_5ForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).cuda().eval()

# ── layer-3（首个 full-attn 层）组件捕获：q/v 投影输出、o_proj 输入
#    （=sigmoid 门后 attn）、self_attn 输出（=o_proj 后）——与 wein
#    s1-L03-{qg,vh,agated,o} 对拍 ──
CAP = {}
def _cap(name):
    def hook(mod, args, output=None):
        t = output if output is not None else args[0]
        if isinstance(t, tuple):
            t = t[0]
        CAP.setdefault(name, []).append(t.detach().float().cpu().numpy().reshape(t.shape[0], -1))
    return hook
sa = m.model.layers[3].self_attn
sa.q_proj.register_forward_hook(_cap("qg"))
sa.v_proj.register_forward_hook(_cap("vh"))
sa.o_proj.register_forward_pre_hook(lambda mod, args: CAP.setdefault("agated", []).append(args[0].detach().float().cpu().numpy().reshape(args[0].shape[0], -1)) or None)
sa.register_forward_hook(_cap("o"))
def flush_cap(step):
    for k, v in CAP.items():
        np.asarray(v[0], dtype=np.float32).tofile(f"{out_dir}/{step}-L03-{k}.f32")
    CAP.clear()

def dump(tag, tensor):  # tensor: [t, dim] on gpu
    np.asarray(tensor.float().cpu().numpy(), dtype=np.float32).tofile(f"{out_dir}/{tag}.f32")

def run(seq):
    """整段前向，返回 hidden_states 元组 + logits（贪心位 = 末行）"""
    inp = torch.tensor([seq]).cuda()
    with torch.no_grad():
        out = m(inp, output_hidden_states=True, use_cache=False)
    return out.hidden_states, out.logits

# ── s1 预填（整段）──
hs, lg = run(ids)
flush_cap("s1")
dump("s1-embed", hs[0][0])
for i, h in enumerate(hs[1:]):
    dump(f"s1-layer-{i:02}", h[0])
dump("s1-logits", lg[0])

hf_greedy = [int(lg[0, -1].argmax())]
# ── s2.. 解码（forced：走 wein 的链；两链同段时与 HF 贪心一致）──
chain = list(ids)
for s in range(2, steps + 1):
    tok = wein_toks[s - 2] if s - 2 < len(wein_toks) else hf_greedy[-1]
    chain.append(tok)
    hs, lg = run(chain)
    # 只落末位（wein 解码步 t=1）——embed/每层/logits 均取最后一行
    dump(f"s{s}-embed", hs[0][0, -1:])
    for i, h in enumerate(hs[1:]):
        dump(f"s{s}-layer-{i:02}", h[0, -1:])
    dump(f"s{s}-logits", lg[0, -1:])
    hf_greedy.append(int(lg[0, -1].argmax()))

json.dump({"ids": ids, "hf_greedy": hf_greedy, "steps": steps},
          open(f"{out_dir}/ref-meta.json", "w"))
print(f"hf_greedy: {hf_greedy}")
