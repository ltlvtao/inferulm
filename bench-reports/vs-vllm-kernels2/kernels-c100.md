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
