# wein vs vLLM 首次直接对比（2026-10-29）

同机同卡（RTX 4090 24G / WSL 驱动 610.88）、同模型、同尺子（wein-bench，OpenAI 协议互打）。
两端各自起服→跑→停（GPU 互斥串行）；报告 JSON 在 `../vs-vllm-<端>-*/`，对拍采样在 `../vs-vllm-correct/`。

- **wein**：rev `746874a`，`--engine core --exec qwen`，eager 档 + `--graph` FULL_DECODE 档
- **vLLM**：0.32.0（torch 2.13.0+cu130），`--max-model-len 512 --gpu-memory-utilization 0.85`
  `VLLM_USE_FLASHINFER_SAMPLER=0`（贪心基准采样非测量轴，绕开 flashinfer JIT 的
  CCCL 13.4↔cu130 版本错配），起服排障链见 `bench/vs_vllm.sh` 注释
- **权重**：两端同一组文本张量。本地检查点是 Qwen3.5 **VL 统一体**（738 张量含 297 个
  `model.visual.*`），wein 历来只加载文本骨干（441 张量）；为 vLLM 生成滤视觉塔副本
  `Qwen3.5-4B-text`（张量字节原样，config/tokenizer 符号链接），
  `--hf-overrides architectures=[Qwen3_5ForCausalLM]`（config 原名 `ForConditionalGeneration`
  被 vLLM 路由到 VL 类）

## 1. 吞吐（wein-bench 同尺子，repeat=1 + warmup）

| 档 | 端 | TTFT p50/p99 (ms) | ITL p50 (ms) | tps | 差距 |
|---|---|---|---|---|---|
| 轻 c=4 n=8 mt=12 | wein eager | 115.4 / 128.6 | 33.9 | 97.7 | ITL ×2.99 |
| 轻 c=4 n=8 mt=12 | wein graph | 112.3 / 114.4 | 31.6 | 103.9 | ITL ×2.80 |
| 轻 c=4 n=8 mt=12 | **vLLM** | **53.9 / 58.2** | **11.3** | **272.9** | — |
| 中 c=16 n=32 mt=32 | wein eager | 244.4 / 244.9 | 60.0 | 243.1 | ITL ×4.80 |
| 中 c=16 n=32 mt=32 | wein graph | 249.0 / 284.2 | 58.4 | 247.1 | ITL ×4.67 |
| 中 c=16 n=32 mt=32 | **vLLM** | **63.7 / 65.0** | **12.5** | **1130.5** | — |

**解读**：

- **解码步差距 2.8×（批 4）→ 4.7×（批 16）——差距随批扩大**。vLLM ITL 批 4→16 只
  +11%（11.3→12.5ms，仍贴显存带宽屋顶线：8.65GB/12.5ms ≈ 692GB/s ≈ 4090 峰值 69%）；
  wein +73%（33.9→60.0ms，≈148GB/s，屋顶线下 6.7×）。批 16 下 4B 模型仍是显存受限
  （128 GFLOP/步 vs 125 TFLOPS ≈ 1ms 计算账）——wein 的批扩展账不在算力，在
  **每行 CPU 侧工作与 kernel 形状**（cuBLASLt 按 m 选 algo、每行 host argmax/上传）。
  这是满负荷优化的路线图数据，不是终局。
- **TTFT 差距 2.1×（轻）→ 3.8×（中）**：wein TTFT = 追平 + 首步，被解码步差距主导；
  vLLM 批 16 追平几乎免费（63.7ms，chunked prefill 并批）。
- **图档收益随批缩水**：轻档 ITL −6.8%（33.9→31.6），中档 −2.7%（60.0→58.4）——
  批大计算占比升、launch 债占比降，与 W2 归因一致。
- 4090 roofline 参照：vLLM 轻档 8.65GB/11.3ms ≈ 765GB/s（76% 峰值）——**这就是
  满负荷 target 的现实锚**，wein 当前 274GB/s（轻）。

## 2. 正确性（贪心对拍，temperature=0，completions 原始 prompt 不经模板）

分词**五组逐 id 一致**（含中文，`tokenize/` 落盘对账）——排除输入侧。

| # | prompt | 公共前缀 | 首分歧 | wein 走向 | vLLM 走向 |
|---|---|---|---|---|---|
| 1 | The capital of France is | 6 字符（" Paris"） | token 2 | 数字复读退化 | 连贯（"Paris.\nA. True…"） |
| 2 | 1, 2, 3, 4, 5, 6, | 1 字符（" "） | token 2 | 数字复读 | " 7, 8, 9, … 19" |
| 3 | 床前明月光， | 0 | **token 1** | " 3…" 乱码 | "疑是地上霜。A. 李白…" |
| 4 | The quick brown fox… | 1 字符（" "） | token 1 | "has long 355…" | "is a mammal…" |
| 5 | def fibonacci(n): | 2 字符（"\n "） | token 2 | 数字复读 | 缩进正确的 Python |

**这是真发现，不是 ulp 噪声**：

- **5 个 prompt 里 4 个首 token 正确**（Paris/空格×2/\n）——预填强决策对；
- 但分歧点边际不小：#3 vLLM 首 token 疑 logprob −0.61、次名 −3.30（**边际 2.7 nats**，
  bf16 ulp 翻不动）；#4 首二名真并列（is/jumps 均 −1.17）但 wein 选了第三名
  has（−3.42，低 2.25 nats）；
- wein 侧特征性症状：**一律滑向数字复读**（555/888/avav）——递推状态被洗成垃圾的
  典型签名；
- 内部证据链（q5 逐步 logits vs 独立 f64 镜像容差 81% + 贪心逐位一致）与外部矛盾 →
  **偏差在共享公式层**（镜像与 GPU 同一实现理解），内部对拍结构性看不见。

**定位假设（待 W-next 验证）**：首 token 多数正确 + 第 2 token 起系统性走偏 →
状态递推路径（gated deltanet 递推细节 / conv 状态滚动 / KV 追加）公式级偏差；
#3 的 token 1 分歧说明预填路径也有份（或该 prompt 低边际敏感）。验证路径：
logit 级 diff——wein 逐步 top-5 落盘 vs vLLM logprobs，逐层二分（先 full-attn
8 层 vs 24 linear-attn 层归因）。

## 3. chat 模板交叉验证（vLLM = W5 minijinja 的外部 oracle）

| 端 | chat prompt_tokens | |
|---|---|---|
| wein（minijinja + str 方法仿真层） | 11 | ✓ 相等 |
| vLLM（HF transformers 同款 Jinja 生产渲染） | 11 | |

W5 的模板渲染获得生产级外部确认：同一 `chat_template.jinja`、同一 "bench" 消息，
两端 prompt token 数严格相等。

## 4. 环境排障链（vLLM 起服 6 次尝试，全部钉进 bench/vs_vllm.sh 注释可复现）

1. config `architectures=Qwen3_5ForConditionalGeneration` → vLLM 路由 VL 类，要
   `preprocessor_config.json`（快照无）→ `--hf-overrides` 改 `Qwen3_5ForCausalLM`；
2. CausalLM 下加载撞 `visual.*` 张量 → 发现检查点是 VL 统一体，滤塔出文本副本；
3. `PermissionError: nvcc`（WSL interop 执行 Windows PE）→ 它还会**盖住真错误**
   （dynamo 错误复现构建器先跑 nvcc --version 自爆）→ venv 自带 nvcc 塞进
   CUDACXX+PATH 揭面具；
4. 真错误现身：flashinfer JIT 的 CCCL 与 nvcc 13.4↔torch cu130 版本错配 →
   `VLLM_USE_FLASHINFER_SAMPLER=0` 绕开（贪心负载无行为差）。
