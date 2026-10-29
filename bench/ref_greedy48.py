"""HF transformers 贪心 48 token（噪声地板测量）。

产出 ref48.json：每 prompt 的 input ids、贪心链 ids、解码文本——供
① wein 链 token 级对比（qwen_trace 同 ids 输入）② vLLM 文本前缀对比
（HF vs vLLM = 跨实现 bf16 噪声地板）。
"""
import sys, os, json
import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

model_dir, out_json = sys.argv[1], sys.argv[2]
PROMPTS = [
    "The capital of France is",
    "1, 2, 3, 4, 5, 6,",
    "床前明月光，",
    "The quick brown fox jumps over the lazy dog. The quick brown fox",
    "def fibonacci(n):",
]
tok = AutoTokenizer.from_pretrained(model_dir)
m = Qwen3_5ForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).cuda().eval()
out = []
for p in PROMPTS:
    ids = tok.encode(p)
    chain = list(ids)
    generated = []
    for _ in range(48):
        with torch.no_grad():
            lg = m(torch.tensor([chain]).cuda()).logits[0, -1]
        t = int(lg.argmax())
        chain.append(t)
        generated.append(t)
        if t == tok.eos_token_id:
            break
    out.append({"prompt": p, "ids": ids, "greedy": generated,
                "text": tok.decode(generated)})
json.dump(out, open(out_json, "w"), ensure_ascii=False, indent=1)
for i, o in enumerate(out):
    print(f'#{i+1}: {o["text"][:60]!r}')
