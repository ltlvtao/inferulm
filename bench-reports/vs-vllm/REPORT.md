# wein vs vLLM 首次直接对比（2026-10-29）

同机同卡（RTX 4090 24G / WSL 驱动 610.88）、同模型、同尺子（wein-bench，OpenAI 协议互打）。
两端各自起服→跑→停（GPU 互斥串行）；报告 JSON 在 `../vs-vllm-<端>-*/`，对拍采样在 `../vs-vllm-correct/`。

- **wein**：rev `746874a`，`--engine core --exec qwen`，eager 档 + `--graph` FULL_DECODE 档
  （吞吐数据与 §2.1 的正确性发现在此 rev；修复后正确性复验见 §2.4，battery 带内见 BASELINE §6 `+fix` 行）
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

## 2. 正确性（贪心对拍，temperature=0，completions 原始 prompt 不经模板）——发现与闭环

分词**五组逐 id 一致**（含中文，`tokenize/` 落盘对账）——排除输入侧。

### 2.1 发现（修复前，rev `746874a`）

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

### 2.2 定位（logit 级 diff，同日闭环）

harness 三件套（已入库）：

- **wein 侧落盘**：`WEIN_TRACE_DIR` executor 逐层残差流 + 组件 dump（f32 LE，
  仅图未捕获时生效，不污染图档路径）+ `examples/qwen_trace.rs` 直接驱动；
- **参照**：`bench/trace_reference.py`——HF transformers forced-decode 逐层落盘，
  跟随 wein 的贪心链（同 token 序列，排除链漂移）；
- **diff**：`bench/trace_diff.py` 逐步逐层 max|Δ|/mean|Δ|。

二分链：embed 精确 → 前置线性层干净 → 首个 full-attn 层爆（max 4.16）→
pos0 单点 softmax 恒等破而 v 读路径干净 ⇒ **V 读路径** → 输出对 16 个 kv 头余弦
相似度 → **头映射**。

判别实验（无须任何外部参照）：同一 token 走整段预填 vs 5 预填 + 1 解码步——
预填 ≡ HF 而解码 ≠ 自身预填，纯 wein 内部即可把问题隔离到解码路径。

### 2.3 双 bug（同类根因：内部测试共享公式盲区）

1. **GQA 头映射**：`qh % kv_heads`（0,1,…,15,0,1,…）应为 `qh // group`
   （HF repeat_kv 语义：kv 头 k 服务 q 头 [k·group,(k+1)·group)；本模型全注意力层
   64 q 头 / 16 kv 头 / group 4）。**PTX kernel + CPU 参考 + f64 镜像三处同错**——
   o12/golden 测试形状 (2,1) 两映射恒等故永远绿；升级 (8,2) 非平凡 group +
   pos0 单点 softmax 恒等（独立真值，从 softmax 语义反推映射而非共享参考公式）钉死。
2. **conv 状态存值约定**：`conv_state_roll` 喂了 conv+silu 后激活（qkvc），而 tap
   读状态与段内同源要**原始 qkv**（HF/FLA causal_conv1d cache 同约定）。exec
   调用点 + 镜像两处同错；预填单段永不触状态故全绿，解码一触即炸（qkvc
   max|Δ|=26.6）。

免疫钉：**段等价性质测试**（kernel `tests/segment_equiv.rs`，conv + deltanet）——
任意切段 + 状态接力 ≡ 整段一次算，位对齐；"段内自洽、跨段露馅"类缺陷的
永久测试类（解码就是 count=1 的切段特例）。

### 2.4 修复后全绿（`vs-vllm-correct/` 全量重捕）

| # | 前缀 token | 文本 | 长度（字符） |
|---|---|---|---|
| 1 | 48/48 | **字节全同** | 156 |
| 2 | 48/48 | **字节全同** | 48 |
| 3 | 48/48 | **字节全同** | 126 |
| 4 | 48/48 | **字节全同** | 175 |
| 5 | 48/48 | **字节全同** | 135 |

48 token × 5 prompt：wein ≡ vLLM **逐字节全同**（HTTP completions 全链路：分词 →
预填 → 48 步贪心解码 → detok）；chat 模板 prompt_tokens 11=11 复验一致。

噪声地板刻度（`bench/ref_greedy48.py`，HF bf16 eager forced-decode）：

- **HF ≡ vLLM：48/48 × 5 全同**——跨实现 bf16 贪心链的地板是 0 分歧，"bf16 噪声
  翻贪心位"在本组 prompt 上不构成解释；
- 修复前 wein vs HF：48/48/48/23/35——正是 §2.1 表中的分歧点；
- 修复后 wein ≡ HF ≡ vLLM 三方全同；解码步偏差回落预填同带（logits mean|Δ|
  ~0.026，无累积）。

### 2.5 事故记录：陈旧二进制

conv 修复后只重编了 `--example qwen_trace`，serve 用的 wein-cli 仍是旧二进制——
服务器"复现"了已修复的偏差，引发一轮徒劳的 serve 路径排查（engine probe /
serve 栈探针 / detok / bind_thread 全干净）。教训（已成本仓库纪律）：
**任何 serve 路径测量前 `cargo build --release`（全 workspace）先行**。

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
