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

## 5. c=100 满负荷补测 + 流量账（2026-10-30，wein@`c721cbd`）——修正 ×2.9 外推

§1 只实测到 c=16，"对 vLLM ×2.9@c=100" 是按权重流外推的。本节补齐实测（同尺子
wein-bench / 同卡同权重 / GPU 互斥串行；JSON 落 `../vs-vllm-vllm/mid-c32.json`、
`full-c100.json`，c=16 复测 `mid-c16-rerun.json`），并给每步 DRAM 流量账。

### 5.1 实测（c=100/n=500/mt=32 与 battery L3q 满负荷同 profile）

| 端 | ITL p50/p99 (ms) | TTFT p50 (ms) | tps |
|---|---|---|---|
| wein eager（battery-124320） | 27.26 / 80.88 | 160.9 | 3022 |
| wein 图档（battery-124320） | 25.84 / — | 127.2 | 3183 |
| **vLLM 0.32.0** | **25.23 / 38.80** | **121.2** | **3431** |

**真实差距 ×1.08（ITL p50）**，不是 ×2.9。c=16→32→100 的 vLLM ITL 曲线
13.0→15.4→25.2ms：边际 ~145µs/行——**线性注意力状态池流量随批线性放大**，
外推假设"ITL 停在权重流 ~12ms"不成立。尾部与 TTFT 仍是 vLLM 好
（p99 38.8 vs 80.9；TTFT 121 vs 161——wein 混合步全 eager，PIECEWISE 刀的靶）。

### 5.2 每步 DRAM 流量账（c=100，短上下文 ~43 tok，4090 峰值 1008GB/s）

| 项 | wein | vLLM | 依据 |
|---|---|---|---|
| 权重 bf16（含 lm_head 1.27GB） | 8.4GB | 8.4GB | |
| 线性注意力状态 r+w | **5.15GB**（bf16 池） | **10.3GB**（**f32**） | config `mamba_ssm_dtype: float32` 被 vLLM 遵守（`vllm/model_executor/layers/mamba/mamba_utils.py:98-110`）；状态 [32,128,128]/层/序列，decode 每步全量读+原地全量写（`third_party/flash_linear_attention/ops/fused_recurrent.py:344-497`，grid (NV, B·HV)） |
| conv 状态 + align 快照拷贝 | 0.12GB | 0.5-0.8GB | vLLM mamba 前缀缓存（align 模式，block_size=16 边界快照）默认开 |
| KV/激活/logits | ~0.4GB | ~0.4GB | |
| **合计** | **~14.1GB** | **~19.7GB** | |

wein 搬的字节少 30%，但实测达成效率 521GB/s vs vLLM 773GB/s——**净平价**。
差距是 kernel 效率工程，不是物理：gemm 段 683 vs ~800GB/s（布局刀，−1.5ms）、
delta 段 398GB/s（kd=128 行驻寄存器 4 块/SM 占用率代价，−3.5~5.5ms——
**占用率刀，已提上顺位**）。

### 5.3 vLLM 未打的牌（评估"好多少"时须计入）

1. **`--mamba-ssm-cache-dtype bfloat16`**：一个 flag 状态流量减半
   （`models/config.py:944-954`）→ 他们 ~18ms/~4500 tps；数值验证他们没做过
   （wein o25 已钉漂移预算 rel2≈1 bf16 量子，这张考卷 wein 先交了）。
2. **MTP 投机解码**：config `mtp_num_hidden_layers=1`，vLLM 有现成融合 kernel
   （`fused_gdn_decode_post_conv_mtp`）——开了可能 ×1.4-1.6。wein 路线图暂无
   投机解码条目（**战略缺口，已入 open questions**；行分账状态池做 draft/verify
   回滚有天然优势：状态快照=行拷贝）。
3. **W8A8/fp8 量化 checkpoint**：vLLM 走 GEMM 层量化需序列化 checkpoint；
   wein W8A8 是 v1 硬需求开箱即用。

**终局判断（本卡/本模型/短上下文/纯 decode 吞吐轴）**：双方出完牌后平价 ±10%。
wein 的确定性优势在 W8A8 开箱、状态 bf16 已验证、单静态二进制，以及**昇腾轴**
（vllm-ascend 对 GDN 混合架构支持远未到位——M1 见真章）。

### 5.4 E2 deltanet 全合并访存落地（2026-10-30 当日续）

§5.2 的"占用率刀"归因**证伪并修正**：病灶不是 kd=128 行驻寄存器的 4 块/SM 占用率，
是**散射 u64 跨步的访存形态**（C4 高占用纯流式对照仅 280GB/s——占用率提不动带宽）。
E2 warp-per-row 全合并（u64 整载整存 + scratch 两级归约）后 kernel 效率 502→721GB/s
（o22 金丝雀 0.424→0.284ms/launch）。battery-20261030-151826 满负荷：
wein eager ITL 24.38 / tps 3421（平价 vLLM 3431）、**图档 22.68 / 3671——反超 +7%**；
图档 TTFT 109.6 亦优于 vLLM 121.2（eager 141.4 仍落后）；ITL p99 66.1 vs 38.8 仍是
主差距（③ PIECEWISE 图档的账）。单步改善 ≈ 24 层 × ~0.12ms/launch。oracle：
239/240 贪心决策逐字同 + 1 处近并列翻转（`n-1`→`n - 1`，重结合舍入撞并列，
kernel↔reference 位对齐合同全绿）。

### 5.5 权重布局杠杆证伪（2026-10-30 当日再续）

§5.2/§5.4 预留的"gemm 布局刀 −1.5ms"**证伪**：引入 gemm 调优审计仪器
（`GemmLt::tune_report` 穿透 + `WEIN_TUNE_DUMP`）后 A/B 判别——交错分配 vs
池先行 vs 权重先行 vs 全缓冲先行四变体，大形状（n=8192/9216，占 gemm 流量
~86%）best_ms 布局无关完全同值（0.0345/0.0461ms），gemm 段中位数带内同
（11.7-11.8ms）。**"655→608GB/s 地址真损失"是测量噪声**：历史 11.42（池化前，
heuristic 时代单次）对 12.30 的对比被小形状择优翻转污染——µs 级小形状
（n=32 a/b 投影 ×48 launch/步）跨进程择优翻 2.4×（0.0061↔0.0154ms），
构成 ±0.3ms/步的运行间漂移；16 发计时加稳无效（非计量噪声，进程内禀——
时钟/地址状态），3 发协议中位数更优故保持。**gemm 段无布局可挖余量，
大形状已处 cuBLASLt 该形状可达带**；后续吞吐杠杆只有 W8A8（权重流减半）。

### 5.6 混合步封顶 mix_cap：p99 尖峰的调度侧解（2026-10-30 当日三续）

§5.4 后满负荷 p99 66.1 vs vLLM 38.8 成唯一大差距项，任务清单首刀原定
PIECEWISE 图档——**证伪**：① eager 与 FULL 图档 p99 完全相同（66.09/66.11）
→ 尖峰非 launch 债；② `qwen_mixed_probe` T-谱（d=100 常驻 decode + 混合步
T 扫描，GPU 独占）：24.33ms@T100 → 64.81ms@T512，~100µs/token 线性 →
混合步计算受限。根因在调度：token_budget=512 把波次切换期 ~412 prefill
token 与 ~100 decode 行塞进同一步，重步拖长全体 decode 行 ITL → p99；
图分片救不了计算本体。

**解法（wein-sched `mix_cap`，默认 192）**：批内有 decode 态请求 → 本步
Σ(decode+prefill token) ≤ 192（decode 计入 cap）；纯 prefill 波不受限
（TTFT 优先）；cap 被 decode 占满时 prefill 保底 `PREFILL_MIN=8`（仅
slack==0 触发，保 T≤cap 严格不变量）。规划两相位：decode 全先落位
（每 token 都是 ITL 进度），prefill 后切。CLI `--mix-cap N` 可调。
cap 取舍：224 → p99 44.18 输 vLLM，192 定默认（ITL p99 主账）。

battery-20261030-164027（图档满负荷 c=100）：ITL p50 22.90 / **p99 36.33**
（旧 66.11，−45%）/ tps 3633（旧 3671，−1.0% 预算自选）/ TTFT p50 68.12
（旧 109.63）/ TTFT p99 269.46（旧 183.6，prefill 排队后移的代价）。
eager 满负荷：p50 24.74/p99 49.18/tps 3441.9/TTFT 68.50/264.77。
轻载 c=4 图档 ITL 12.36/TTFT 41.14 带内无回归。

对 vLLM c=100（§5.1）：ITL p50 22.90 vs 25.23、p99 36.33 vs 38.80、
TTFT p50 68.12 vs 121.2、tps 3633 vs 3431 **全胜**；TTFT p99 269.46 vs
228.4 落后（ITL 平稳换波次排队——预算自选，SLO 期可调）。

oracle 复验：wein 侧重捕 + compare 4/5 精确同 + 1 既知近并列翻转
（fibonacci `n-1`）+ chat 模板 11/11——调度重排不改单请求前向序，数值
合同不受影响。测试：s10_mixed_step_cap_shields_decoders（两相位/cap
不变量/切片不丢 token/下限触发）。wein-serve 39 测试全绿（**代理变量
注入环境下**验收：测试 reqwest 全部改 `common::client()` no_proxy 直连，
759b55d 事故复发面闭合）。PIECEWISE 从任务清单除名——其目标已由调度侧
一步达成。

外推边界声明：单卡 4090/WSL、Qwen3.5-4B dense、短上下文（~43 tok）、greedy、
c≤128。长上下文（KV 流量改账 + wein T_MAX=512 限制）、多卡、MoE 不在此外推内。
