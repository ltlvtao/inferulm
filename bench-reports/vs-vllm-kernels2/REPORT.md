# vLLM c=100 稳态 decode 的逐 kernel 账（torch profiler）

日期 2026-11-06｜RTX 4090 24G（driver 610.88 / torch cu130）｜vLLM 0.32.0 + torch 2.13.0+cu130（`~/venvs/vllm`）
模型 `~/.cache/wein/Qwen3.5-4B-text`（8.6528 GB，`tie_word_embeddings=true`）｜端口 8199
**只测量，未改 wein 任何源码/bench 脚本，未 commit。** 新增脚本放在 `/tmp/`（`k2_start.sh` / `k2_load.py` / `k2_parse.py`），
为可复现另存一份副本在本目录（同内容，未入库）。

---

## 0. 结论速览

| 问题 | 答案 |
|---|---|
| 用了哪条路径 | **serve 路径 + vLLM 自带 torch profiler**（`--profiler-config.profiler=torch` + `/start_profile`）。不是离线 `LLM.generate()`。这就是任务里说的"serve 路径对照"手段本身，见 §1.2 |
| 实测吞吐 | wein-bench（Rust 客户端，与基线同档 c=100 / max_tokens=32）：**4305 tps，ITL p50 19.10 / mean 20.58 ms**（基线 4091 / 20.06 → 同档，本次机器快约 5%） |
| 每步 ms | **19.987 ms/步**（步首→步首，205 步纯 decode）＝100 token/步 ⇒ 5003 tps |
| **每步 kernel 数** | **537.05 个**（纯 `kernel` cat）＋ 65.1 个 memcpy/memset |
| **每步 kernel busy** | **19.569 ms**（空洞 0.418 ms = 2.1%） |
| 最大头 | gemm/gemv 12.602 ms（64.4%）、mamba ssm 递推 4.916 ms（25.1%）、attention 1.164 ms（5.9%） |
| 每层 kernel 数 | 线性注意力层 **16** 个，full-attention 层 **16** 个（32 层 × 16 = 512，加头尾 25 = 537） |

---

## 1. 配置、路径、实测吞吐

### 1.1 起服命令（与对比基线完全同档）

```
vllm serve "$HOME/.cache/wein/Qwen3.5-4B-text" --port 8199 \
  --max-model-len 512 --gpu-memory-utilization 0.85 \
  --mamba-ssm-cache-dtype bfloat16 \
  --max-num-batched-tokens 2048 --max-num-seqs 256 \
  --hf-overrides '{"architectures": ["Qwen3_5ForCausalLM"]}' \
  # 仅为取账追加（不改任何推理行为）：
  --profiler-config.profiler=torch --profiler-config.torch_profiler_dir=<OUT>/traces \
  --profiler-config.torch_profiler_with_stack=false --profiler-config.torch_profiler_use_gzip=true
# 环境：VLLM_SERVER_DEV_MODE=1（开 /start_profile 路由）、VLLM_TORCH_PROFILER_DIR、VLLM_USE_FLASHINFER_SAMPLER=0
# WSL 修补：PATH 剥 /mnt/c/...CUDA 段；CUDACXX+PATH 指向 venv 自带 nvcc（site-packages/nvidia/**/nvcc）
```

引擎实测事实（`server.log`）：`enforce_eager=False`、`cudagraph_mode=FULL_AND_PIECEWISE(2,1)`、
`cudagraph_capture_sizes` 含 104（100 路 → 落到 FULL 图）、capture 51 piecewise + 35 FULL、
attention backend **FLASH_ATTN(FA2)**、GDN prefill kernel = Triton/FLA、`quantization=None`、
`mamba_cache_mode=align`、`block_size=272`、`num_gpu_blocks=1245`、`kv_cache_size_tokens=79680`、
`kv_cache_max_concurrency=155.6`（100 路在容量内）、`rms_norm/fused_add_rms_norm` op priority = `native`（不影响此处，见 §6①）。
日志无 `async_scheduling` 条目 ⇒ 按默认（未开 async scheduling）。

GPU 互斥：起服前 Windows 侧 `nvidia-smi --query-gpu` = 3% / 67 W / 1428 MiB；停服后 3% / 68 W / 1413 MiB，
`--query-compute-apps` 无我们进程（`gpu-check-post.log`）。全程只有本进程占卡。

### 1.2 路径声明（重要）

我用的是 **serve 路径 + vLLM 内置 torch profiler**，而非任务里写的离线 `LLM.generate()` 方案。
理由与证据：

1. profiler 必须与 kernel 同进程；vLLM 的 `--profiler-config.profiler=torch` 正好在 **EngineCore 进程内**装
   `torch.profiler`，HTTP `/start_profile` 只开窗、不碰推理逻辑 ⇒ 得到的就是**基线那条路径**（FULL cuda graph、
   FA2、FLA GDN、mamba cache align），不存在"离线路径与 serve 路径编译/图档行为不同"的担心。
2. 离线 `LLM.generate()` 会把 prefill 和 decode 混在一次调用里、且没有 serve 的图档回放结构，无法逐项对照。
3. 开销已实测可忽略：profile 窗口内 205 步 × 19.987 ms → **5003 tps**，未开 profiler 的同配置 c=100 长 decode
   是 **5048 tps**（差 0.9%）。stop_profile 后写 6.8 MB gzip trace 会阻塞引擎约 8.7 s，但**窗口已经关闭**，不计入账。

### 1.3 实测吞吐

| 口径 | 读数 |
|---|---|
| **wein-bench c=100 / max_tokens=32 / num=4000（与基线同一把尺子）** | **4305.1 tps**，ITL p50 **19.10 ms** / mean 20.58 / p99 37.83，TTFT p50 101.9 ms，0 errors |
| 自写 100 线程流式客户端，max_tokens=400，ignore_eos，未开 profiler | 5048 tps（span 口径），397 token/请求，wall 7.98 s |
| 同上，profile 窗口内（205 步） | **19.987 ms/步** ⇒ 5003 tps |
| 对比基线（调用方给出） | 4091 tps / ITL 20.06 ms |

基线 4091/20.06 与本次 4305/19.10 差 ~5%（同日机器状态/桌面进程差异），**每步 20 ms 量级完全对上**，
所以下面的逐 kernel 账可以直接与我们的 nsys 逐项对照。

---

## 2. 每步总账

取账窗口：一次 c=100 全部并发（prompt `"1\n"` ≈ 3 token，`max_tokens=400`、`ignore_eos=true`、`stream=true`），
3.2 s 预热后 `/start_profile`，收到 **205 步纯 decode**（annotation 全部是
`execute_0(0)_generation_100(100)` = 0 个 context token、100 路 generate，**窗口内零 prefill 串扰**）。

| 指标 | 值 |
|---|---|
| 步 cadence（步首→步首） | **19.987 ms** |
| `gpu_user_annotation` 覆盖（图内 + 尾随 eager） | 18.200 ms |
| **每步 kernel 发数** | **537.05** |
| **每步 kernel busy 合计** | **19.569 ms** |
| 每步 memcpy + memset | 0.058 ms（65.1 次） |
| 空洞（cadence − kernel busy） | **0.418 ms（2.1%）** |
| kernel 数守恒 | 32 层 × 16 + 头 12 + 尾 13 = **537** ✔ |

归属方法：kernel 按"最近的前一个步边界"归属（`gpu_user_annotation` 是 GPU 时间戳域，与 kernel 对齐；
CPU 侧 `user_annotation` 在 cuda graph 下只覆盖 launch ~2 ms，不可用）。lm_head 的 GEMM 与采样确实跑在
FULL 图之外（落在窗口结束后几 µs），按上述规则归到当步 —— 这也是 §4 末尾那批 eager kernel 的由来。

---

## 3. 功能组（每步，ms）

| 组 | ms/步 | 占 kernel | 发数/步 |
|---|---|---|---|
| **gemm/gemv**（含 lm_head 与 split-K 归约） | **12.6016** | 64.4% | 241.00 |
| **mamba ssm 核心递推** | **4.9155** | 25.1% | 24.00 |
| **attention**（flash splitkv + KV 写） | **1.1636** | 5.9% | 16.00 |
| **norm**（rms / fused_add_rms / gated rms / qk+rope） | **0.2796** | 1.4% | 97.00 |
| **mamba conv1d**（含 silu） | **0.1990** | 1.0% | 24.00 |
| **elementwise**（silu_mul / gate / 拷贝 / 清零） | **0.1963** | 1.0% | 117.00 |
| **mamba 状态搬运**（align 页拷贝/回写） | **0.1106** | 0.6% | 4.00 |
| **sampling / argmax** | **0.0816** | 0.4% | 5.00 |
| **其它**（引擎元数据/索引） | **0.0211** | 0.1% | 9.05 |
| **kernel 合计** | **19.5689** | 100% | **537.05** |
| memcpy/memset（非 kernel cat） | 0.0584 | — | 65.1 |

> 注意 `elementwise` 里的 0.1963 ms 只有 0.0906 ms 是真激活运算（silu_mul 0.0767 + attn 门控 sigmoid 0.0139），
> 其余 0.1057 ms 是 inductor/引擎生成的纯拷贝与清零，见 §4.6。

---

## 4. 每组的具体 kernel（名 + 发数/步 + 单发 µs）

口径：`us_per_step` = 窗口内该 kernel 总时长 ÷ 205 步；单发 µs = 总时长 ÷ 总发数。
名字为 trace 里的原样（长模板参数已截断，完整名见 `kernels-c100.csv`）。

### 4.1 gemm/gemv — 12.6016 ms/步（241 发：152 个 GEMM + 88 个 split-K 归约 + 1 个 lm_head）

| 发数/步 | 单发 µs | ms/步 | kernel |
|---|---|---|---|
| 64.00 | 119.39 | 7.641 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>` — in_proj（24）＋ MLP gate/up（24＋8＋8） |
| 32.00 | 60.84 | 1.947 | `...gemm_relu_bf16_64x128_32x6_tn_align8` — MLP down（32 层各 1） |
| 32.00 | 35.38 | 1.132 | `...gemm_relu_bf16_128x64_32x6_tn_align8` — o_proj/out_proj（32 层各 1） |
| 24.00 | 3.18 | 0.076 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_tn_align8>` — 线性层独有的窄 GEMM（输出 64 列 = b/a 两半，见 §4.6） |
| 88.00 | 1.99 | 0.175 | `void cublasLt::splitKreduce_kernel<32, 16, int, __nv_bfloat16, __nv_bfloat16, float, ...>` — **模型 GEMM 的 split-K 归约**（线性层 3/层、全注意力层 2/层） |
| **1.00** | **1630.02** | **1.630** | **`ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn` — lm_head（1.2714 GB 词表，跑在 CUDA 图外）** |

拆开看：模型权重 7.3814 GB / 10.9716 ms = **673 GB/s**；lm_head 1.2714 GB / 1.630 ms = **780 GB/s**。

### 4.2 mamba ssm 核心递推 — 4.9155 ms/步

| 发数/步 | 单发 µs | ms/步 | kernel |
|---|---|---|---|
| 24.00 | **204.81** | **4.916** | `fused_recurrent_gated_delta_rule_packed_decode_kernel`（每线性层 1 发；"packed" 指把 v-head / 变长序列打包进一个 grid，**不是**把 24 层合成 1 发） |

### 4.3 attention — 1.1636 ms/步

| 发数/步 | 单发 µs | ms/步 | kernel |
|---|---|---|---|
| 8.00 | **143.75** | 1.150 | `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<256, 64, 64, 4, ...>>`（每全注意力层 1 发；长尾明显，p50 ~137 µs、个别步 603 µs） |
| 8.00 | 1.70 | 0.014 | `void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0>`（KV 写入） |

### 4.4 mamba conv1d — 0.1990 ms/步

| 发数/步 | 单发 µs | ms/步 | kernel |
|---|---|---|---|
| 24.00 | 8.29 | 0.199 | `_causal_conv1d_update_kernel`（**conv4 + bias + silu 一个 kernel**，见 §6②） |

### 4.5 norm — 0.2796 ms/步（97 发）

| 发数/步 | 单发 µs | ms/步 | kernel |
|---|---|---|---|
| 32.00 | 2.31 | 0.0740 | `triton_red_fused__to_copy_add_fused_add_rms_norm_1` — attention 输出**加残差 + RMSNorm**（送 MLP） |
| 32.00 | 2.15 | 0.0687 | `triton_red_fused__to_copy_add_fused_add_rms_norm_3` — MLP 输出**加残差 + RMSNorm**（送下一层；最后一层这一发顺带承担 `model.norm`） |
| 24.00 | 4.53 | 0.1088 | `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_silu_0` — 线性层的 **gated RMSNorm（含 silu 门控）** |
| 8.00 | 3.20 | 0.0256 | `_fused_qk_rmsnorm_rope_gate_kernel` — 全注意力层 **QK-RMSNorm + 部分 RoPE + 输出门控** 三合一 |
| 1.00 | 2.48 | 0.0025 | `triton_red_fused__to_copy_add_embedding_rms_norm_0` — token embedding 查表 + 第 0 层输入 RMSNorm |

### 4.6 elementwise — 0.1963 ms/步（117 发，只有一半是"真" elementwise）

| 发数/步 | 单发 µs | ms/步 | kernel | 性质 |
|---|---|---|---|---|
| 32.00 | 2.40 | 0.0767 | `triton_poi_fused_mul_silu_slice_2` | MLP 的 `silu(gate)*up`（32 层各 1） |
| 8.00 | 1.74 | 0.0139 | `triton_poi_fused_mul_sigmoid_view_0` | 全注意力层输出门控 `sigmoid` 乘（attn_output_gate） |
| 24.00 | 1.27 | 0.0304 | `triton_poi_fused_zeros_{5}`（23/步）＋`_zeros_2`（1/步） | 线性层 ssm 输出 buffer 的 **zeros 清零**（纯开销） |
| 24.00 | 0.90 | 0.0216 | `triton_poi_fused_4`（23/步）＋`triton_poi_fused_1`（1/步） | 线性层 `[*,64]`→`[*,32]+[*,32]` 的 **b/a 拆分拷贝**（纯拷贝） |
| 16.00 | 2.37 | 0.0379 | `at::native::elementwise_kernel<128, 4, direct_copy_kernel>` | 全注意力层 2 发/层的 contiguous 拷贝 |
| 8.00 | 1.41 | 0.0113 | `at::native::elementwise_kernel<128, 2, direct_copy_kernel>` | 全注意力层 1 发/层的拷贝 |
| 2.00 | 0.90 | 0.0018 | `at::native::vectorized_elementwise_kernel<4, FillFunctor<int>>` | 引擎侧 int 全填 |
| 2.00 | 0.88 | 0.0018 | `at::native::vectorized_elementwise_kernel<4, FillFunctor<bool>>` | 引擎侧 bool 全填 |
| 1.00 | 0.85 | 0.0008 | `(anonymous namespace)::elementwise_kernel_with_index<...arange...>` | arange |

### 4.7 mamba 状态搬运 — 0.1106 ms/步

| 发数/步 | 单发 µs | ms/步 | kernel |
|---|---|---|---|
| 1.00 | 69.49 | 0.0695 | `precopy_mamba_align_fused_kernel` — 进模型前把 SSM/conv state 页搬进 align 布局。**均值被尖峰拉高**：200 步是 35.95 µs，5 步（新 mamba 页分配时）是 **1411 µs** |
| 1.00 | 38.35 | 0.0383 | `postprocess_mamba_fused_kernel` — 出模型后回写 state 页 |
| 1.00 | 1.53 | 0.0015 | `preprocess_mamba_align_fused_kernel` |
| 1.00 | 1.25 | 0.0012 | `get_aligned_state_indices_multi_group_kernel` |

### 4.8 sampling / argmax — 0.0816 ms/步

| 发数/步 | 单发 µs | ms/步 | kernel |
|---|---|---|---|
| 1.00 | **73.96** | 0.0740 | `_gumbel_sample_kernel`（默认 temperature=1.0 走随机采样；wein-bench 基线也不带 temperature，同一条路） |
| 1.00 | 3.81 | 0.0038 | `at::native::reduce_kernel<512, 1, ReduceOp<float, ArgMaxOps<float>, ...>>` |
| 1.00 | 1.46 | 0.0015 | `_get_num_sampled_and_rejected_kernel`（无投机解码，管线仍跑） |
| 1.00 | 1.25 | 0.0013 | `_combine_sampled_and_draft_tokens_kernel` |
| 1.00 | 1.14 | 0.0011 | `_scatter_num_accepted_kernel` |

### 4.9 其它（引擎元数据/索引）— 0.0211 ms/步

`_prepare_pos_seq_lens_kernel` 1.11 µs、`_gather_block_tables_kernel` 1.82、`_compute_slot_mappings_kernel` 2.39、
`_post_update_kernel` 2.18、`at::native::index_elementwise_kernel` ×2（2.37 / 2.29）、`_scatter_gather_elementwise_kernel` 1.87、
`at::native::vectorized_gather_kernel` 1.66、`memcpy32_post` 1.17、`at::native::vectorized_elementwise_kernel` 0.96，
各 1 发/步；另 `_zero_kv_blocks_kernel` 5 发/窗口（170 µs/发，折 0.02 发/步）与 `_apply_write_kernel` 5 发/窗口。

### 4.10 memcpy / memset（`cat` 不是 `kernel`，不计入 537）

| 发数/步 | 单发 µs | ms/步 | 名称 |
|---|---|---|---|
| 40.0 | 1.01 | 0.0403 | `Memset (Unknown)` |
| 18.1 | 0.56 | 0.0101 | `Memcpy HtoD (Pageable -> Device)` |
| 5.0 | 0.94 | 0.0047 | `Memcpy DtoD (Device -> Device)` |
| 2.0 | 1.25 | 0.0025 | `Memcpy DtoH (Device -> Pinned)` |

---

## 5. 每层 kernel 预算（205 步平均、按层类型汇总；µs/发）

### 5.1 线性注意力层（24 层）— **16 个 kernel / 层**，583.91 µs/层 ⇒ 14.014 ms/步

| # | µs/发 | kernel | 角色 |
|---|---|---|---|
| 1 | 90.94 | `cutlass ... 64x64_32x6` | in_proj（q/k/v/z/ba 合一） |
| 2 | 3.18 | `cutlass wmma ... 32x32_64x1` | ba 窄 GEMM（输出 64 列） |
| 3 | 1.50 | `cublasLt::splitKreduce_kernel` | in_proj 的 split-K 归约 |
| 4 | 0.90 | `triton_poi_fused_4` | b/a 拆分拷贝 |
| 5 | 1.27 | `triton_poi_fused_zeros_5` | ssm 输出 buffer 清零 |
| 6 | 8.29 | `_causal_conv1d_update_kernel` | conv4+bias+**silu** |
| 7 | **204.81** | `fused_recurrent_gated_delta_rule_packed_decode_kernel` | **l2norm + gate/decay + 递推 + state** |
| 8 | 4.53 | `triton_per_fused..._mean_mul_pow_rsqrt_silu_0` | gated RMSNorm（含 silu） |
| 9 | 36.99 | `cutlass ... 128x64_32x6` | out_proj |
| 10 | 2.10 | `cublasLt::splitKreduce_kernel` | out_proj 的 split-K 归约 |
| 11 | 2.32 | `triton_red_fused..._add_fused_add_rms_norm_1` | 残差 add + RMSNorm |
| 12 | 159.63 | `cutlass ... 64x64_32x6` | MLP gate/up |
| 13 | 2.41 | `triton_poi_fused_mul_silu_slice_2` | silu(gate)*up |
| 14 | 60.62 | `cutlass ... 64x128_32x6` | MLP down |
| 15 | 2.27 | `cublasLt::splitKreduce_kernel` | MLP down 的 split-K 归约 |
| 16 | 2.15 | `triton_red_fused..._add_fused_add_rms_norm_3` | 残差 add + RMSNorm |

### 5.2 full-attention 层（8 层）— **16 个 kernel / 层**，463.06 µs/层 ⇒ 3.705 ms/步

| # | µs/发 | kernel | 角色 |
|---|---|---|---|
| 1 | 76.80 | `cutlass ... 64x64_32x6` | in_proj（q/kv/output-gate） |
| 2 | 3.20 | `_fused_qk_rmsnorm_rope_gate_kernel` | QK-RMSNorm + 部分 RoPE + 门控 |
| 3 | 1.70 | `vllm::reshape_and_cache_flash_kernel` | KV 写缓存 |
| 4 | 2.25 | `at::native::elementwise_kernel<128,4,copy>` | 元数据/输出 contiguous |
| 5 | **143.75** | `flash::flash_fwd_splitkv_kernel<256,64,64,4>` | FA2 split-KV |
| 6 | 2.49 | `at::native::elementwise_kernel<128,4,copy>` | |
| 7 | 1.41 | `at::native::elementwise_kernel<128,2,copy>` | |
| 8 | 1.74 | `triton_poi_fused_mul_sigmoid_view_0` | 输出门控 sigmoid 乘 |
| 9 | 30.55 | `cutlass ... 128x64_32x6` | o_proj |
| 10 | 2.33 | `cublasLt::splitKreduce_kernel` | o_proj split-K 归约 |
| 11 | 2.28 | `triton_red_fused..._add_fused_add_rms_norm_1` | 残差 add + RMSNorm |
| 12 | 126.61 | `cutlass ... 64x64_32x6` | MLP gate/up |
| 13 | 2.36 | `triton_poi_fused_mul_silu_slice_2` | silu(gate)*up |
| 14 | 61.52 | `cutlass ... 64x128_32x6` | MLP down |
| 15 | 1.94 | `cublasLt::splitKreduce_kernel` | MLP down split-K 归约 |
| 16 | 2.13 | `triton_red_fused..._add_fused_add_rms_norm_3` | 残差 add + RMSNorm |

### 5.3 层外（头 12 + 尾 13 = 25 个 kernel / 步，1.846 ms/步）

头：4 个填/arange 拷贝（0.83–0.90）、`_prepare_pos_seq_lens` 1.11、`_combine_sampled_and_draft_tokens` 1.25、
`_gather_block_tables` 1.82、`_compute_slot_mappings` 2.39、`preprocess_mamba_align` 1.53、
`precopy_mamba_align_fused` **69.49（含 1411 µs 尖峰）**、`get_aligned_state_indices` 1.24、
`embedding_rms_norm_0` 2.46。
尾：`memcpy32_post` 1.17、`vectorized_gather` 1.66、**lm_head GEMM 1630.02**、2×`index_elementwise` 2.37/2.29、
`_gumbel_sample` 73.96、`reduce(ArgMax)` 3.81、`_scatter_gather_elementwise` 1.87、`vectorized_elementwise` 0.96、
`_get_num_sampled_and_rejected` 1.46、`_post_update` 2.18、`_scatter_num_accepted` 1.14、`postprocess_mamba_fused` 38.35。

---

## 6. 五个明确回答

**① rms_norm 是独立 kernel 还是和 residual add 融合？**
→ **全部融合，没有任何独立 rms_norm kernel。** 每层 2 发 `triton_red_fused__to_copy_add_fused_add_rms_norm_{1,3}`
（2.15–2.32 µs/发，一封做完 `残差 add + RMSNorm`）。全图只有 3 种 norm fusion 形态：
`_1`（32 发/步 = attention 后）、`_3`（32 发/步 = MLP 后，最后一层这一发兼作 `model.norm`）、
`_embedding_rms_norm_0`（1 发/步 = embedding 查表 + 第 0 层输入 norm）。
所以 32 层总共 **65 发 norm/步**、0.145 ms —— 如果我们是"每层 3 个独立 norm（pre-attn / post-attn / final）"，
光 kernel 数就差 31 发、字节上多两次全量 `[100,2560]` 读写。
另注：线性层的 **gated RMSNorm 是另一个 kernel**（`triton_per_fused ... rsqrt_silu`，24 发/步 4.53 µs，内含 silu 门控）。

**② causal_conv1d 是否顺带做了 silu 激活？**
→ **是，conv 与 silu 在同一个 kernel 里。** `_causal_conv1d_update_kernel` 8.29 µs/发、24 发/步；
源码 `qwen_gdn_linear_attn.py:1806` 把 `self.activation`（=silu）传给 `causal_conv1d_update`。
trace 序列也印证：conv kernel 后面**紧跟**就是 ssm 核心 kernel，中间没有 silu/act kernel
（对比 prefill 路径还多一个独立的 `_fused_post_conv_kernel`，decode 没有）。

**③ 线性注意力的 l2norm / gating 是否被融进"mamba 核心"大 kernel，还是独立小 kernel？**
→ **全融进核心大 kernel。** `fused_recurrent_gated_delta_rule_packed_decode_kernel`（204.81 µs/发 × 24 发/步）
一次做完：**q/k l2norm**（`use_qk_l2norm_in_kernel=True`，源码 1776 行）+ **gate/decay（g=−exp(A_log)·softplus(a+dt_bias)）**
+ **逐 head 递推** + **SSM state 读改写**。序列里没有独立 l2norm kernel、也没有独立 gate kernel。
但线性层确实还有两条**独立小 kernel**（不是 l2norm，是 inductor 生成的）：
`triton_poi_fused_4` = `[*,64]→[*,32]+[*,32]` 的 b/a 拆分**拷贝** 0.90 µs/发，
`triton_poi_fused_zeros_5` = ssm 输出 buffer **清零** 1.27 µs/发；两者 24 发/步，合计 0.052 ms/步（2.7% 的 kernel 时间）。

**④ 每层线性注意力一共几个 kernel？**
→ **16 个**（§5.1）：GEMM 5 发（in_proj / ba / out_proj / MLP up / MLP down）+ split-K 归约 3 发 +
conv1d 1 + ssm 核心 1 + gated norm 1 + b/a 拷贝 1 + buffer 清零 1 + silu_mul 1 + fused_add_rms_norm 2。
24 层 ⇒ 384 发/步、14.014 ms/步。

**⑤ 每层 full-attention 一共几个 kernel？**
→ **也是 16 个**（§5.2）：GEMM 4 发 + split-K 归约 2 发 + qk_rmsnorm_rope_gate 1 + KV 写 1 +
flash_fwd_splitkv 1 + `at::native` 拷贝 3 + 输出门控 sigmoid 1 + silu_mul 1 + fused_add_rms_norm 2。
8 层 ⇒ 128 发/步、3.705 ms/步。

---

## 7. 字节账（每步）

权重 8.6528 GB（含 tied 词表 1.2714 GB）；mamba state **bf16** 100 路 × 24 层 ×
(32 v-head × 128 × 128 × 2 B = 1.0486 MB) = **2.517 GB**；KV 32 KB/token（8 层 × 4 kv-head × 256 dim × K+V × 2 B）。

| 段 | ms/步 | 字节/步 | 有效带宽 |
|---|---|---|---|
| 模型 GEMM（7.3814 GB 权重） | 10.9716 | 7.381 GB | **673 GB/s** |
| lm_head（1.2714 GB 词表） | 1.6300 | 1.271 GB | **780 GB/s** |
| mamba ssm 递推 | 4.9155 | 2.517 读 + 2.517 写 = 5.03 GB | **1024 GB/s**（只算读则 512 GB/s） |
| attention（KV） | 1.1500 | ≈0.83 GB（按窗口内平均序列长度 ~260 估） | ≈720 GB/s（**估算**） |
| **合计** | 19.569 | ≈14.5 GB | **741 GB/s** |

对照基线：4091 tps ⇒ **24.44 ms/步**，本账 **19.99 ms/步**，差 **4.45 ms/步**（≈ 1.8k tps 的差距）。
按本表逐项看，这 4.45 ms 的候选来源是：
① ssm 递推段的形态（vLLM 一次吃下 5.03 GB @1024 GB/s = 4.92 ms，占 25%；若我们按旧报的 ~6.46 ms 则这里就 ~1.5 ms）；
② lm_head 780 GB/s vs 旧报我们 438 GB/s（1.63 vs 2.90 ms，差 ~1.27 ms）；
③ 空洞 0.418 ms（2.1%）vs 旧报我们 2.10 ms（9%），差 ~1.7 ms；
④ gemm 主体 673 GB/s vs 旧报我们 718 GB/s（这一项我们**略优**，不是输点）。
（wein 侧数字引自 `vs-vllm-kernels-20261102`，本次未重测。）

---

## 8. 与 2026-11-02 旧账（`vs-vllm-kernels-20261102`）的两处订正

1. **`cublasLt::splitKreduce_kernel` 不属于 lm_head。** 旧报把 88 发/步的 splitKreduce 记进了 lm_head 桶，
   但逐步核序列显示它紧跟在 **模型 GEMM**（in_proj / out_proj / MLP down）之后，而 lm_head 的
   `ampere_bf16_..._128x64_ldg8_f2f_tn` 之后**没有**归约 kernel（88 = 24×3 + 8×2，正好没有余量）。
   本表已改归 gemm:model。影响：lm_head 从 1.81 降到 **1.630 ms/步**（780 GB/s），gemm:model 升到 10.97。
2. **旧报的 A 场景窗口污染已消除。** 旧窗口 200 步里混了 42 个 chunked-prefill 混合步（每步 33.6 ms）；
   本次 `max_tokens=400` + 一次性 100 并发 ⇒ **205 步全部是 `context_0 / generation_100`**，无任何 prefill。

其余结论方向一致（gemm 是大头、ssm packed 每层 1 发、每步 ~20 ms、空洞 2%）。

---

## 9. 限制与偏差（如实）

1. **客户端 ITL 不可用于判步**：自写脚本用 100 个 Python 线程读 SSE，GIL 会把时间戳打散
   （未开 profiler 那次 `itl_p50=10.07` 而 `mean=19.70`）。**步时长以 trace 的 `gpu_user_annotation` 为准（19.987 ms）**；
   与 wein-bench（Rust 客户端，`itl_p50=19.10`）一致。
2. **profiler 开销**：窗口内 5003 tps vs 未开 5048 tps（0.9%）。`stop_profile` 写 6.8 MB gzip 期间引擎阻塞 8.7 s，
   已在窗口外，不入账。
3. **窗口内序列长度在变**：从 ~160 token 长到 ~360 token（跨过 block_size=272 时新分配 KV 页，
   因此 5 步里出现 `_zero_kv_blocks` 和 1411 µs 的 `precopy_mamba_align` 尖峰）。这些尖峰留在均值里
   （precopy 均值 69.49 vs 常态 35.95 µs），即下表数字是**含摊销**的真实稳态，不是理想态。
4. **attention 长尾**：`flash_fwd_splitkv` 均值 143.75 µs，多数步 134–140 µs，个别步 603 µs（未深究，可能是页命中/调度抖动）。
5. 采样是**随机采样**（`_gumbel_sample_kernel`，temperature 默认 1.0），与 wein-bench 基线同一条路；
   若要 greedy 对照，需显式加 `temperature=0`，届时 argmax 会替代 gumbel（本表不含该变体）。
6. 桌面图形进程占用（起服前 3% / 67 W）全程存在，只影响绝对吞吐 ±5%，不影响逐 kernel 的相对结构。

---

## 10. 文件

| 文件 | 内容 |
|---|---|
| **`kernels-c100.csv`** | **交付表：`name,calls_per_step,us_per_step`（46 行，按 us_per_step 降序）** |
| `REPORT.md` | 本报告 |
| `kernels-c100.md` | 机器生成的完整表（逐步账 + 功能组 + 细桶 + 全量逐 kernel + memcpy/memset） |
| `kernels-c100.json` | 同上，结构化（含 fold/groups/rows 原始数值） |
| `traces/K2-decode_rank0.*.pt.trace.json.gz` | 原始 torch profiler trace（6.8 MB，205 步）＋ `*.profiler_out.txt` |
| `load-noprof.json` / `load-prof.json` | 自写 100 并发负载读数（未开/开 profiler） |
| `wein-bench-c100.json` | wein-bench c=100 / mt=32 读数（4305 tps、ITL p50 19.10） |
| `cmd.txt` / `server.log` / `run.log` / `env.txt` / `vllm.pid` | 起服命令 / 服务端日志 / 驱动日志 / 环境 / pid |
| `gpu-check-post.log` | 停服后 GPU：3% / 68 W / 1413 MiB，无 vLLM 进程 |
| `/tmp/k2_start.sh`、`/tmp/k2_load.py`、`/tmp/k2_parse.py` | 起服 / 负载+开窗 / trace 解析（本次新增，未入库） |

---

## 附录 A：机器生成的逐 kernel 全量表

# K2-decode_rank0.1793963195074116496.pt.trace.json.gz

## 逐步账（205 步纯 decode，每步 100 路 generate）

| 指标 | 值 |
|---|---|
| 步 cadence（步首到步首） | **19.987 ms** |
| gpu_user_annotation 覆盖（图形+尾随 eager） | 18.200 ms |
| **kernel busy 合计** | **19.569 ms（537.05 个 kernel）** |
| memcpy+memset | 0.058 ms（65.1 次） |
| 空洞（cadence − kernel） | 0.418 ms（2.1%） |

## 功能组（每步）

| 组 | ms/步 | 占 kernel | 发数/步 |
|---|---|---|---|
| gemm/gemv | 12.6016 | 64.4% | 241.00 |
| mamba ssm 递推 | 4.9155 | 25.1% | 24.00 |
| attention | 1.1636 | 5.9% | 16.00 |
| norm | 0.2796 | 1.4% | 97.00 |
| mamba conv1d | 0.1990 | 1.0% | 24.00 |
| elementwise | 0.1963 | 1.0% | 117.00 |
| mamba 状态搬运 | 0.1106 | 0.6% | 4.00 |
| sampling/argmax | 0.0816 | 0.4% | 5.00 |
| 其它(引擎元数据/索引) | 0.0211 | 0.1% | 9.05 |
| **kernel 合计** | **19.5689** | 100% | **537.05** |

## 细桶（每步）

| 细桶 | ms/步 | 占 kernel | 发数/步 |
|---|---|---|---|
| gemm:model | 10.9716 | 56.1% | 240.00 |
| mamba:ssm_core | 4.9155 | 25.1% | 24.00 |
| gemm:lm_head | 1.6300 | 8.3% | 1.00 |
| attn:core | 1.1500 | 5.9% | 8.00 |
| mamba:conv1d | 0.1990 | 1.0% | 24.00 |
| norm:rms/fused_add_rms | 0.1452 | 0.7% | 65.00 |
| mamba:state_mgmt | 0.1106 | 0.6% | 4.00 |
| norm:gated_rms(含silu) | 0.1088 | 0.6% | 24.00 |
| sampling:gumbel/argmax | 0.0778 | 0.4% | 2.00 |
| elementwise:silu_mul | 0.0767 | 0.4% | 32.00 |
| elem:at::native 拷贝/fill | 0.0536 | 0.3% | 29.00 |
| elem:线性层 out-buffer 清零 | 0.0304 | 0.2% | 24.00 |
| norm:qk_rope_gate | 0.0256 | 0.1% | 8.00 |
| elem:线性层 b/a 拆分拷贝 | 0.0216 | 0.1% | 24.00 |
| engine:meta/index | 0.0211 | 0.1% | 9.05 |
| elementwise:attn_gate(sigmoid) | 0.0139 | 0.1% | 8.00 |
| attn:kv_cache | 0.0136 | 0.1% | 8.00 |
| sampling:spec_plumbing | 0.0039 | 0.0% | 3.00 |

## 逐 kernel（每步，按 us/步 降序）

| # | us/步 | ms/步 | 发数/步 | 单发 µs | 细桶 | kernel |
|---|---|---|---|---|---|---|
| 1 | 7640.93 | 7.6409 | 64.00 | 119.39 | gemm:model | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(cutlass_80_tensorop_` |
| 2 | 4915.48 | 4.9155 | 24.00 | 204.81 | mamba:ssm_core | `fused_recurrent_gated_delta_rule_packed_decode_kernel` |
| 3 | 1947.02 | 1.9470 | 32.00 | 60.84 | gemm:model | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x128_32x6_tn_align8>(cutlass_80_tensorop` |
| 4 | 1630.02 | 1.6300 | 1.00 | 1630.02 | gemm:lm_head | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn` |
| 5 | 1149.98 | 1.1500 | 8.00 | 143.75 | attn:core | `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<256, 64, 64, 4, false, false, cutlass::bfloat16_t` |
| 6 | 1132.17 | 1.1322 | 32.00 | 35.38 | gemm:model | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_32x6_tn_align8>(cutlass_80_tensorop` |
| 7 | 198.98 | 0.1990 | 24.00 | 8.29 | mamba:conv1d | `_causal_conv1d_update_kernel` |
| 8 | 175.17 | 0.1752 | 88.00 | 1.99 | gemm:model | `void cublasLt::splitKreduce_kernel<32, 16, int, __nv_bfloat16, __nv_bfloat16, float, __nv_bfloat16, false, __n` |
| 9 | 108.79 | 0.1088 | 24.00 | 4.53 | norm:gated_rms(含silu) | `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_silu_0` |
| 10 | 76.72 | 0.0767 | 32.00 | 2.40 | elementwise:silu_mul | `triton_poi_fused_mul_silu_slice_2` |
| 11 | 76.32 | 0.0763 | 24.00 | 3.18 | gemm:model | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_tn_align8>(cutlass_80_wmma_ten` |
| 12 | 74.01 | 0.0740 | 32.00 | 2.31 | norm:rms/fused_add_rms | `triton_red_fused__to_copy_add_fused_add_rms_norm_1` |
| 13 | 73.96 | 0.0740 | 1.00 | 73.96 | sampling:gumbel/argmax | `_gumbel_sample_kernel` |
| 14 | 69.49 | 0.0695 | 1.00 | 69.49 | mamba:state_mgmt | `precopy_mamba_align_fused_kernel` |
| 15 | 68.70 | 0.0687 | 32.00 | 2.15 | norm:rms/fused_add_rms | `triton_red_fused__to_copy_add_fused_add_rms_norm_3` |
| 16 | 38.35 | 0.0383 | 1.00 | 38.35 | mamba:state_mgmt | `postprocess_mamba_fused_kernel` |
| 17 | 37.89 | 0.0379 | 16.00 | 2.37 | elem:at::native 拷贝/fill | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` |
| 18 | 29.23 | 0.0292 | 23.00 | 1.27 | elem:线性层 out-buffer 清零 | `triton_poi_fused_zeros_5` |
| 19 | 25.64 | 0.0256 | 8.00 | 3.20 | norm:qk_rope_gate | `_fused_qk_rmsnorm_rope_gate_kernel` |
| 20 | 20.74 | 0.0207 | 23.00 | 0.90 | elem:线性层 b/a 拆分拷贝 | `triton_poi_fused_4` |
| 21 | 13.95 | 0.0139 | 8.00 | 1.74 | elementwise:attn_gate(sigmoid) | `triton_poi_fused_mul_sigmoid_view_0` |
| 22 | 13.61 | 0.0136 | 8.00 | 1.70 | attn:kv_cache | `void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0>(__nv_bflo` |
| 23 | 11.28 | 0.0113 | 8.00 | 1.41 | elem:at::native 拷贝/fill | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` |
| 24 | 4.15 | 0.0042 | 0.02 | 170.15 | engine:meta/index | `_zero_kv_blocks_kernel` |
| 25 | 3.81 | 0.0038 | 1.00 | 3.81 | sampling:gumbel/argmax | `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int,` |
| 26 | 2.48 | 0.0025 | 1.00 | 2.48 | norm:rms/fused_add_rms | `triton_red_fused__to_copy_add_embedding_rms_norm_0` |
| 27 | 2.39 | 0.0024 | 1.00 | 2.39 | engine:meta/index | `_compute_slot_mappings_kernel` |
| 28 | 2.37 | 0.0024 | 1.00 | 2.37 | engine:meta/index | `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<a` |
| 29 | 2.29 | 0.0023 | 1.00 | 2.29 | engine:meta/index | `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<a` |
| 30 | 2.18 | 0.0022 | 1.00 | 2.18 | engine:meta/index | `_post_update_kernel` |
| 31 | 1.87 | 0.0019 | 1.00 | 1.87 | engine:meta/index | `void at::native::_scatter_gather_elementwise_kernel<128, 8, at::native::_cuda_scatter_gather_internal_kernel<f` |
| 32 | 1.82 | 0.0018 | 1.00 | 1.82 | engine:meta/index | `_gather_block_tables_kernel` |
| 33 | 1.80 | 0.0018 | 2.00 | 0.90 | elem:at::native 拷贝/fill | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, ` |
| 34 | 1.76 | 0.0018 | 2.00 | 0.88 | elem:at::native 拷贝/fill | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<bool>, std::array<char*, 1ul> >(int,` |
| 35 | 1.66 | 0.0017 | 1.00 | 1.66 | engine:meta/index | `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool)` |
| 36 | 1.53 | 0.0015 | 1.00 | 1.53 | mamba:state_mgmt | `preprocess_mamba_align_fused_kernel` |
| 37 | 1.46 | 0.0015 | 1.00 | 1.46 | sampling:spec_plumbing | `_get_num_sampled_and_rejected_kernel` |
| 38 | 1.25 | 0.0013 | 1.00 | 1.25 | sampling:spec_plumbing | `_combine_sampled_and_draft_tokens_kernel` |
| 39 | 1.25 | 0.0012 | 1.00 | 1.25 | mamba:state_mgmt | `get_aligned_state_indices_multi_group_kernel` |
| 40 | 1.21 | 0.0012 | 1.00 | 1.21 | elem:线性层 out-buffer 清零 | `triton_poi_fused_zeros_2` |
| 41 | 1.17 | 0.0012 | 1.00 | 1.17 | engine:meta/index | `memcpy32_post` |
| 42 | 1.14 | 0.0011 | 1.00 | 1.14 | sampling:spec_plumbing | `_scatter_num_accepted_kernel` |
| 43 | 1.11 | 0.0011 | 1.00 | 1.11 | engine:meta/index | `_prepare_pos_seq_lens_kernel` |
| 44 | 0.85 | 0.0008 | 1.00 | 0.85 | elem:线性层 b/a 拆分拷贝 | `triton_poi_fused_1` |
| 45 | 0.85 | 0.0008 | 1.00 | 0.85 | elem:at::native 拷贝/fill | `void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(c10::Scalar const&,` |
| 46 | 0.04 | 0.0000 | 0.02 | 1.54 | engine:meta/index | `_apply_write_kernel` |

## memcpy/memset（每步）

| 发数/步 | 单发 µs | us/步 | 名称 |
|---|---|---|---|
| 40.0 | 1.01 | 40.3 | Memset (Unknown) |
| 18.1 | 0.56 | 10.1 | Memcpy HtoD (Pageable -> Device) |
| 5.0 | 0.94 | 4.7 | Memcpy DtoD (Device -> Device) |
| 2.0 | 1.26 | 2.5 | Memcpy DtoH (Device -> Pinned) |
