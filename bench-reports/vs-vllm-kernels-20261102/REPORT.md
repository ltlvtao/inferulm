# vLLM 时间账拆到 kernel 级 · 与 wein 段表逐段并列

日期 2026-11-02｜机器 RTX 4090 24G（driver 610.57.01 / CUDA UMD 13.3）｜单 GPU 进程串行
vLLM 0.32.0 + torch 2.13.0+cu130（`~/venvs/vllm`），模型 `~/.cache/wein/Qwen3.5-4B-text`（8,652,766,016 B = 8.65 GB，`tie_word_embeddings=true`）
**只测量，未改 wein 生产代码，未 commit。** wein 段表直接采信调用方给的数，未重测。

## 0. 方法

- 路径：vLLM 自带 torch profiler（`--profiler-config.profiler=torch` + `VLLM_TORCH_PROFILER_DIR`
  + `POST /start_profile` / `/stop_profile`，`VLLM_SERVER_DEV_MODE=1`）。nsys 未使用。
- 关键坑 1：只设 `VLLM_TORCH_PROFILER_DIR` 不会挂 `/start_profile` 路由，必须给 `--profiler-config.profiler=torch`。
- 关键坑 2：CUDA graph 下 **CPU 侧 `user_annotation` 只覆盖 launch**（A: 1.66 ms vs GPU 9.49 ms），不能用于归属。
  改用 trace 里的 **`gpu_user_annotation`**（GPU 时间戳，与 kernel 对齐，由
  `vllm/v1/worker/gpu_worker.py:1183 annotate_profile()` 发出，名字形如
  `execute_204800_context_0(...)_generation_100(...)`），归属规则为「就近归到前一个 step 边界」——
  因为 **lm_head 的 GEMV 跑在 CUDA graph 之外，落在上一个窗口结束后约 7 µs**。
- 关键坑 3：`gpu_memcpy` / `gpu_memset` 的 `cat` 不是 `kernel`，必须一起统计才能对上 step 时间。
- 端口 8177；起服 44 s；起服前 1085 MiB / 起服后 22481 MiB / 停服后 1098 MiB（`gpu-check-post.log`，无残留）。
- 场景：A = `wein-bench --concurrency 100 --num 4000 --max-tokens 32 --warmup`，profile 窗口 14 s 稳态；
  B/C = `probe.py`（6009 token prompt → `/reset_prefix_cache` → `/start_profile` → 单请求）。
- **偏差与限制（如实）**：
  1. A 的 profile 窗口含 **42/200 个 chunked-prefill 混合步**（ctx_tokens 11…264），下表 A 的每步数
     是 **158 个纯 gen 步**（并发 100×134 / 99×18 / 81×6）的子集；混合步单独列一行。
  2. B 的 prompt 实际 6018 token（不是 6009）；B 在 profiler 下 TTFT = 2203.8 ms（gzip trace 写盘开销），
     墙上时间口径一律用未 profile 的 382.8–418.0 ms。
  3. C 同窗口含 2 个 prefill 步，已剔除。
  4. wein 段表自身有 0.79 ms 口径差（列出的 8 段相加 22.39，声称段和 21.6），A 表用「口径差」行平账。

## 1. kernel 归桶表（占该场景 GPU kernel 时间 %）

| 桶 | A 满载 decode（纯 gen 步） | B 6009 冷 prefill | C 6009 decode（纯 gen 步） |
|---|---|---|---|
| GEMM（cutlass/cublas/nvjet） | **58.8%** 11.014 ms/步 | **80.0%** 268.40(总) | **79.2%** 8.492 ms/步 |
| mamba/GDN（fused_recurrent…delta_rule / causal_conv1d / qwen_gdn / postprocess_mamba） | **27.8%** 5.219 | 6.6% 22.68 | 1.2% 0.127 |
| lm_head/lm_head（gemvx / cublasLt splitK） | 8.8% 1.652 | 1.4% 4.87 | 13.2% 1.411 |
| elementwise/activation（triton poi/red/per） | 1.7% 0.326 | 5.7% 19.40 | 1.2% 0.134 |
| attention（flash_fwd_splitkv / unified_attention） | 1.1% 0.212 | 5.7% 19.59 | 3.7% 0.397 |
| rms_norm（triton add_rms_norm） | 0.8% 0.152 | 1.3% 4.48 | 1.1% 0.115 |
| sampling（gumbel_sample / argmax） | 0.4% 0.080 | 0.0% 0.02 | 0.1% 0.008 |
| memcpy/memset | 0.3% 0.059 | 0.3% 1.05 | 0.2% 0.025 |
| rope | 0.1% 0.028 | 0.3% 1.08 | 0.1% 0.013 |
| quant/dequant | **0（无 W8A8，`quantization=None`）** | 0 | 0 |
| 其它 | 0.0% | 0.0% | 0.0% |
| **kernel 合计** | 18.741 | 341.57 | 10.722 |
| **step 实测** | **19.133** | 3×116.52 = 349.6 | **10.931** |
| **空洞（step − kernel）** | **0.392（2.1%）** | 7.996（2.3%） | **0.209（1.9%）** |

### top-20 kernel（A 纯 gen 步，GPU 2961.03 ms / 158 步）

| # | ms | % | n | avg | kernel |
|---|---|---|---|---|---|
| 1 | 1227.486 | 41.5 | 10118 | 121.3 µs | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8` |
| 2 | 781.588 | 26.4 | 3792 | 206.1 µs | `fused_recurrent_gated_delta_rule_packed_decode_kernel` |
| 3 | 307.772 | 10.4 | 5056 | 60.9 µs | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x128_32x6_tn_align` |
| 4 | 250.040 | 8.4 | 152 | 1645.0 µs | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn`（lm_head） |
| 5 | 177.610 | 6.0 | 5056 | 35.1 µs | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_32x6_tn_align` |
| 6 | 31.281 | 1.1 | 1264 | 24.7 µs | `flash::flash_fwd_splitkv_kernel<256,64,64,4>` |
| 7 | 31.277 | 1.1 | 3792 | 8.2 µs | `_causal_conv1d_update_kernel` |
| 8 | 26.443 | 0.9 | 13904 | 1.9 µs | `cublasLt::splitKreduce_kernel<32,16,int,bf16,bf16,float>` |
| 9 | 17.926 | 0.6 | 3792 | 4.7 µs | `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_silu_0`（gated RMSNorm） |
| 10 | 12.650 | 0.4 | 5056 | 2.5 µs | `triton_red_fused__to_copy_add_fused_add_rms_norm_1` |
| 11 | 11.805 | 0.4 | 3792 | 3.1 µs | `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_tn_align` |
| 12 | 11.678 | 0.4 | 5056 | 2.3 µs | `triton_poi_fused_mul_silu_slice_2` |
| 13 | 11.595 | 0.4 | 158 | 73.4 µs | `_gumbel_sample_kernel` |
| 14 | 10.949 | 0.4 | 5056 | 2.2 µs | `triton_red_fused__to_copy_add_fused_add_rms_norm_3` |
| 15 | 6.468 | 0.2 | 6464 | 1.0 µs | `Memset (Unknown)` |
| 16 | 5.866 | 0.2 | 158 | 37.1 µs | `postprocess_mamba_fused_kernel` |
| 17 | 5.756 | 0.2 | 2528 | 2.3 µs | `at::native::elementwise_kernel<128,4,...>` |
| 18 | 5.631 | 0.2 | 158 | 35.6 µs | `precopy_mamba_align_fused_kernel` |
| 19 | 4.760 | 0.2 | 3634 | 1.3 µs | `triton_poi_fused_zeros_5` |
| 20 | 4.408 | 0.1 | 1264 | 3.5 µs | `_fused_qk_rmsnorm_rope_gate_kernel` |

### top-20 kernel（B 6009 冷 prefill，GPU 341.57 ms / 3 chunk × 2048）

| # | ms | % | n | avg | kernel |
|---|---|---|---|---|---|
| 1 | 241.216 | 70.6 | 288 | 837.6 µs | `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` |
| 2 | 26.116 | 7.6 | 96 | 272.0 µs | `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` |
| 3 | 19.267 | 5.6 | 24 | 802.8 µs | `flash::flash_fwd_splitkv_kernel<256,64,64,4>` |
| 4 | 10.111 | 3.0 | 96 | 105.3 µs | `triton_poi_fused_mul_silu_slice_2` |
| 5 | 7.310 | 2.1 | 72 | 101.5 µs | `chunk_fwd_kernel_o`（GDN chunked） |
| 6 | 5.271 | 1.5 | 72 | 73.2 µs | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` |
| 7 | 4.870 | 1.4 | **3** | 1623.5 µs | `internal::gemvx::kernel<...>`（lm_head） |
| 8 | 4.254 | 1.2 | 72 | 59.1 µs | `merge_16x16_to_64x64_inverse_kernel` |
| 9 | 3.798 | 1.1 | 72 | 52.7 µs | `_causal_conv1d_fwd_kernel` |
| 10 | 2.424 | 0.7 | 96 | 25.3 µs | `triton_red_fused__to_copy_add_fused_add_rms_norm_3` |
| 11 | 2.336 | 0.7 | 72 | 32.4 µs | `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_silu_0` |
| 12 | 2.102 | 0.6 | 72 | 29.2 µs | `_fused_post_conv_kernel` |
| 13 | 2.008 | 0.6 | 96 | 20.9 µs | `triton_red_fused__to_copy_add_fused_add_rms_norm_1` |
| 14 | 1.901 | 0.6 | 72 | 26.4 µs | `recompute_w_u_fwd_kernel` |
| 15 | 1.180 | 0.3 | 72 | 16.4 µs | `chunk_scaled_dot_kkt_fwd_kernel` |
| 16 | 1.077 | 0.3 | 24 | 44.9 µs | `_fused_qk_rmsnorm_rope_gate_kernel` |
| 17 | 0.955 | 0.3 | 72 | 13.3 µs | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_...` |
| 18 | 0.890 | 0.3 | 75 | 11.9 µs | `Memcpy DtoD (Device -> Device)` |
| 19 | 0.661 | 0.2 | 24 | 27.6 µs | `triton_poi_fused_mul_sigmoid_view_0` |
| 20 | 0.579 | 0.2 | 69 | 8.4 µs | `triton_poi_fused_zeros_5` |

### top-20 kernel（C 6009 decode 纯 gen 步，GPU 300.20 ms / 28 步）

| # | ms | % | n | avg | kernel |
|---|---|---|---|---|---|
| 1 | 251.955 | 83.9 | 3388 | 74.4 µs | `internal::gemvx::kernel<int,int,bf16,...>`（GEMV 主集） |
| 2 | 25.333 | 8.4 | 896 | 28.3 µs | `internal::gemvx::kernel<int,int,bf16,...>`（小 K 变体） |
| 3 | 8.459 | 2.8 | 224 | 37.8 µs | `flash::flash_fwd_splitkv_kernel<256,64,64,4>` |
| 4 | 2.355 | 0.8 | 672 | 3.5 µs | `fused_recurrent_gated_delta_rule_packed_decode_kernel` |
| 5 | 2.317 | 0.8 | 224 | 10.3 µs | `flash_fwd_splitkv_combine_kernel` |
| 6 | 1.800 | 0.6 | 896 | 2.0 µs | `triton_red_fused__to_copy_add_fused_add_rms_norm_3` |
| 7 | 1.362 | 0.5 | 896 | 1.5 µs | `triton_red_fused__to_copy_add_fused_add_rms_norm_1` |
| 8 | 1.084 | 0.4 | 672 | 1.6 µs | `_causal_conv1d_update_kernel` |
| 9 | 0.958 | 0.3 | 672 | 1.4 µs | `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_silu_0` |
| 10 | 0.917 | 0.3 | 896 | 1.0 µs | `triton_poi_fused_mul_silu_slice_2` |
| 11 | 0.547 | 0.2 | 644 | 0.8 µs | `triton_poi_fused_4` |
| 12 | 0.465 | 0.2 | 644 | 0.7 µs | `triton_poi_fused_zeros_5` |
| 13 | 0.359 | 0.1 | 224 | 1.6 µs | `_fused_qk_rmsnorm_rope_gate_kernel` |
| 14 | 0.335 | 0.1 | 224 | 1.5 µs | `vllm::reshape_and_cache_flash_kernel` |
| 15 | 0.277 | 0.1 | 504 | 0.5 µs | `Memcpy HtoD (Pageable -> Device)` |
| 16 | 0.212 | 0.1 | 252 | 0.8 µs | `memcpy32_post` |
| 17 | 0.200 | 0.1 | 224 | 0.9 µs | `triton_poi_fused_mul_sigmoid_view_0` |
| 18 | 0.140 | 0.0 | 140 | 1.0 µs | `Memcpy DtoD (Device -> Device)` |
| 19 | 0.099 | 0.0 | 28 | 3.6 µs | `at::native::index_elementwise_kernel<128,4,...>` |
| 20 | 0.093 | 0.0 | 28 | 3.3 µs | `at::native::reduce_kernel<512,1,ArgMaxOps<float>>` |

## 2. 逐段并列表

口径：conv 在两侧都并入 delta(mamba)（vLLM 的 `_causal_conv1d_*` 本来就在 mamba 桶里）；
vLLM 无 sampling/memcpy 独立段，并入「norm+rope+conv+eltw」；lm_head 从 gemm 拆出对齐 wein 的 `logits` 段。
`buckets_all` 里 B 的 lm_head 未拆（混在 gemm 273.267 内），表中已扣出 4.870。

### 表 A · 满载 decode c=100（每步，ms）

| 段 | vLLM | wein | 差 | 我的归因 |
|---|---|---|---|---|
| gemm | 11.014 | 11.70 | **−0.69** | 两边同量级；wein 8.4 GB/11.7 ms = 718 GB/s 略优于 vLLM 7.38 GB/11.01 = 670 GB/s。**gemm 不是我们的输点** |
| attn | 0.212 | 0.33 | −0.12 | vLLM 只 1264 次 flash_splitkv（24.7 µs） |
| delta(mamba)+conv | 5.219 | 6.46 | **−1.24** | vLLM 把 24 层 GDN 状态更新合成 **1 个 packed kernel**（206 µs/层）+ conv 8.2 µs；wein 的 conv 单独 0.6 ms。字节侧两边都是 state 带宽墙（~2.6 GB/步） |
| logits(lm_head) | 1.652 | 2.90 | **−1.25** | 同一张 1.2714 GB 词表：vLLM **770 GB/s** vs wein **438 GB/s**。vLLM 在 graph 外用 `128x64_ldg8` GEMM + `splitKreduce`(101 次/步) 摊 K |
| norm+rope+conv+eltw | 0.645 | 1.00 | −0.36 | vLLM 每步 5056 个 triton rms_norm 调用（2.2–2.5 µs），全图内 |
| **空洞** | **0.392** | **2.10** | **−1.71** | vLLM 600+ kernel 一次 graph 回放，非 kernel 间隙 2.1%；wein 9% |
| wein 段表口径差 | — | −0.80 | +0.80 | wein 列出 8 段相加 22.39 ≠ 声称段和 21.6，差 0.79；本表按列出值取用 |
| **合计** | **19.133** | **23.69** | **−4.56** | |

参考：`step 实测` 与 ITL p50 同档（本次 profiler 在跑时 c=100 ITL p50 **19.045 ms**，未 profile 基线 20.06）。
A 的 **42 个 chunked-prefill 混合步**（ctx 11…264 token）每步 33.59 ms、kernel 27.73、**空洞 5.85（17.4%）**
——这本身是一条独立结论：**prefill 插进 decode 时我们和图两端都会掉出稳态**。

### 表 B · 6009 冷 prefill（整段总账，ms）

vLLM 3 步 × 2048 token（`--max-num-batched-tokens 2048`）；wein 24 步 × 256 token（`chunk_size=256`）。

| 段 | vLLM(总) | wein(总) | 差 | 我的归因 |
|---|---|---|---|---|
| gemm | 268.40 | 470.88 | **−202.5** | **算法/形状**：vLLM 162 TFLOPS（44.4 TFLOP/273 ms，≈4090 bf16 稠密峰值 165）；wein 94 TFLOPS 且**是带宽墙**（24 × 8.4 GB / 471 ms = 428 GB/s，权重被翻了 23 倍） |
| attn | 19.59 | 342 | **−322.4** | 全表最大单项。vLLM 3.2 µs/query-token（2048 query 一批）；wein 55.7 µs/token。24 步重扫 prefix 占 ~8×，其余 ~2.2× 是 256-query 小 M 效率 |
| delta(mamba) | 22.68 | 333.12 | **−310.4** | **算法**：vLLM 是 chunked-GEMM 形态（`chunk_gated_delta_rule_fwd_kernel_h` / `chunk_fwd_kernel_o` / `recompute_w_u_fwd_kernel` / `merge_16x16_to_64x64_inverse`），0.154 µs/token/层；wein 是逐 token 递推链，2.26 µs/token/层（14.7×），且 wein 自测 72% 时间在 kvm/o 折叠依赖链 |
| logits(lm_head) | 4.87 | 59.28 | −54.4 | 只 3 次（每 chunk 1 次）vs 24 次 |
| norm+rope+conv+eltw | 26.03 | 40.80 | −14.8 | |
| **空洞** | **8.00** | **53.92** | **−45.9** | |
| **合计** | **349.6** | **1300** | **−950.4** | |

**差额四分解（全部实测，可加总到 1300）**：
`349.6` vLLM GPU@chunk2048 ＋ `45.2` 调度/首 token（→ 未 profile TTFT 394.8，与旧报一致）
＋ `481.5` chunk 惩罚（旧报实测 vLLM 自己压到 `--max-num-batched-tokens 256` 时 6000 token = 876.3 ms）
＋ `423.7` 同 chunk 下的 kernel 单价劣势（wein 1300 − vLLM@256 876.3）＝ **1300**。

### 表 C · 6009 decode（每步，ms）

wein 侧只有 ITL 总账，没有分段：`单请求 @6009 ITL 15.76`、`@3009 14.87`。

| 段 | vLLM | wein | 差 | 我的归因 |
|---|---|---|---|---|
| gemm | 8.492 | n/a | — | vLLM 113 次 `gemvx`/步、74.4 µs，7.38 GB/8.49 = **869 GB/s**（≈87% 峰值）；wein 按 878 GB/s 反推 ≈ 9.57 ms |
| logits(lm_head) | 1.411 | n/a | — | 1.2714 GB/1.41 = 901 GB/s |
| attn | 0.397 | n/a | — | 224 次 flash_splitkv，KV 192 MB/0.40 ms = **484 GB/s** |
| delta(mamba)+conv | 0.127 | n/a | — | 672 次（24 层 × 28 步），3.5 µs/层 |
| norm+rope+conv+eltw | 0.290 | n/a | — | |
| **空洞** | **0.209** | n/a | — | 1.9% |
| **合计** | **10.931** | **15.76** | **−4.83** | 本次实测 10.93（基线 10.67），同档 |

C 的分段差只能推：wein 3k→6k ITL 只涨 0.89 ms，而 KV 多读 96 MB ⇒ **wein 的 KV 有效带宽仅 ~108 GB/s**
（vLLM 484 GB/s）。若 wein 的 192 MB KV 也跑 484 GB/s，attn 只要 0.4 ms ⇒ 该段可省 ~1.4 ms；
余下 ~3.4 ms 落在 gemm 与 lm_head 的 GEMV 形态和空洞上。**证据强度：推断**（wein 无 C 分段剖面）。

## 3. 每场景字节账

字节口径：权重 **8.6528 GB**（含 tied 词表 1.2714 GB）；KV **32 KB/token**（8 个全注意力层 × 4 kv head × 256 dim × K+V × 2 B）；
mamba state **26.35 MB/序列**（24 层 ×〔32×128×128 bf16 = 1.0486 MB ssm ＋ 8192×3 卷积 = 49 KB〕）。
`kv_cache_memory=11.4 GiB`、`kv_cache_size_tokens=303989`、`block_size=272`、`num_gpu_blocks=1373`（`cache-config.txt`）。

- **A（c=100）**：权重 8.65 ＋ mamba state 100×26.35 MB = 2.635 ＋ KV ≈ 2.8（按 wein 14.1 GB 总账反推，
  trace 未记每步序列长度，**这一格 vLLM 侧无法独立算出，缺数据**）≈ **14.1 GB / 19.133 ms = 737 GB/s**，
  对 wein 595 GB/s（**+24%**）。拆段：gemm 670 vs 718 GB/s（**wein 略好**）、lm_head 770 vs 438 GB/s（**vLLM 1.76×**）、
  delta 5.219 ms 搬 2.635 GB = 505 GB/s（wein 同字节 ≈ 407 GB/s）。
- **B（6009 冷 prefill）**：vLLM 权重搬运 3 × 8.6528 = **25.96 GB**，44.4 TFLOP，gemm 段 **162 TFLOPS**、权重带宽仅 74 GB/s
  ⇒ **算力墙**。wein 24 × 8.4 = **201.6 GB**（同样的 4B 权重搬了 8×，一整个 prompt 搬了 23 倍）、94 TFLOPS、
  **428 GB/s ⇒ 带宽墙**。这就是 `--chunk 256` 的结构性代价：每步只有 256 token 的活，却被 8.4 GB 权重拖住。
- **C（6009 单请求）**：8.6528 ＋ KV 0.1926 ＋ state 0.0264 = **8.872 GB / 10.931 ms = 812 GB/s**
  （按 kernel 和 10.722 算 827）。wein 侧 C 无分段字节表；若采信其 878 GB/s 与 15.76 ms 反推 **13.84 GB/步**，
  比 vLLM 多 **5.0 GB** —— 这 5.0 GB 用权重/KV/state 都解释不掉，**要 wein 的分段字节表才能定案（缺数据）**。

## 4. 结论（按「可拿回的毫秒 × 确定性」排序）

1. **prefill 的 chunk 与 GDN 算法形态：可拿回 ~700–950 ms TTFT（确定性：实测）**
   *性质：kernel 形态 + 字节（权重翻 23 倍）。*
   证据：同机同 prompt，vLLM 2048-chunk 394.8 ms / 256-chunk 876.3 ms / wein 1300 ms（三个都是实测）。
   要做两件事，都不动数值：(a) `wein-sched/src/lib.rs:69 chunk_size: 256 → 2048`，配套抬
   `wein-kernel/kernels/src/lib.rs:451 const MAX_LEN=256` 与 `T_MAX=512` scratch —— 单这一项就是 −481 ms；
   (b) delta 段换成 vLLM 的 chunked-GEMM 形态（`chunk_gated_delta_rule_fwd_kernel_h` + `recompute_w_u` + `chunk_fwd_kernel_o`），
   实测 0.154 vs 2.26 µs/token/层 —— 对应 −310 ms，且 wein 自己的 B 档预估（45→19 ms/step）方向一致。
   全拿满是 1300 → ~395，成本是 prefill 内核重写 + MAX_LEN 改造，所以排第一但不排最急。
2. **满载 decode 的段间空洞：2.10 → 0.39 ms（确定性：实测双侧同口径）**
   *性质：空洞，不是 kernel。*
   证据：A 场景 vLLM 空洞 0.392 ms（2.1%）vs wein 2.10 ms（9%），机制是 vLLM 把 600+ kernel 一次 graph 回放，
   连 lm_head 的 GEMV 都露在 graph 外也只花 1.65 ms。我们的混合步空洞更差（**5.85 ms，17.4%**）。
   **这一项只需 1.7 ms/步、几乎零算法风险、且 c=100 满载是收入路径**，所以按「毫秒×确定性」它该排最急。
3. **lm_head / logits 的 GEMV 形态：−1.25 ms/步（decode）＋ 更长的影子（确定性：实测字节，推断机制）**
   *性质：字节效率。*
   证据：同一张 1.2714 GB 词表，vLLM 770 GB/s（A）、901 GB/s（C）vs wein 438 GB/s（A，2.9 ms）。
   vLLM 靠 `128x64_ldg8` GEMM + `cublasLt::splitKreduce`（101 次/步、1.9 µs）把 K 摊到多个 SM；
   wein 的 438 GB/s 说明并行度不足。这一项在满载是 1.25 ms，在单请求 decode 里量级相同（C 场景 1.41 ms），
   且和 delta 的 1.24 ms（conv 单独 0.6 ms ＋ 24 次 per-layer launch）是同一类「小 kernel 形态」问题 —— 合并看是 ~2.5 ms/步。

**没拿到/需补的**：① wein 侧 A/B/C 的分段**字节**账（现在只有时间），否则 §3 的 5.0 GB 悬空；
② wein 侧 6009 decode 的分段剖面（表 C 的 wein 列全空）；③ 一份**纯 gen 无 prefill 串扰**的 A 窗口
（本次窗口含 42/200 混合步；补法：`--max-num-seqs 100` 下先让 n 个请求跑完再开 profile，或临时
`--enable-chunked-prefill false` 取相对结构）。这三项都不影响上面三条结论的方向。

## 5. 文件

| 文件 | 内容 |
|---|---|
| `run_all.sh` / `probe.py` / `parse_trace.py` | 起服+三场景驱动 / 单请求探针 / trace 解析（桶+top-N+步归属） |
| `cmd.txt` / `env.txt` / `cache-config.txt` | 实际起服命令 / 环境 / `EngineConfig` 关键字段 |
| `server.log` / `run.log` / `A-load.log` | 服务端全量日志 / 驱动日志 / wein-bench 输出 |
| `traces/` | `A-hot_rank0.*.pt.trace.json.gz`(13.4 MB)、`B-prefill_rank0`(495 KB)、`C-decode_rank0`(1.05 MB) ＋ 各自 `*.profiler_out.txt` |
| `A-kernels.txt/.json`、`B-…`、`C-…` | 桶表 + top-20 + 逐步 cadence/kernel/空洞 |
| `A-load.json`、`B-probe.json`、`C-probe.json` | 压测与探针读数（itl/ttft/errors） |
| `gpu-check-post.log` | 停服后 `nvidia-smi`：1098 MiB，无 vLLM 进程 |
