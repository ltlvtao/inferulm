# 终验：wein vs vLLM 成对交错实测（2026-11-07，HEAD = fed1fbe + 停服修复）

这是**最后一次**性能对抗验证。目标是"环境尽可能相同、测量尽可能精确"，
所以放弃了历次那种"两个不同脚本、不同轮次各测一遍"的做法。

## 一、方法（可复现：`bench/final_vs_vllm.sh 3`）

| 维度 | 处理 |
|---|---|
| 客户端 | **完全同一个客户端**：短上下文 `wein-bench`（OpenAI 协议，两端同一份二进制）、长上下文 `bench/longctx_probe.py`（同一脚本、同一 tokenizer 副本） |
| 题面 | 同一 token id 序列：短上下文固定发 `"bench"`（3 token）；长上下文由 `$TEXT_DIR` 的 tokenizer 确定性生成 3009/6009 token |
| 交错 | 3 轮，每轮 = wein 块 → vLLM(bf16 状态) 块 → vLLM(f32 状态) 块；**逐轮配对差**与跨轮中位并列给出 |
| 口径 | wein 只有 bf16 递推状态（`o25` 容差档 + oracle 背书）。vLLM 跑两档：`--mamba-ssm-cache-dtype bfloat16`（= 与 wein 同口径，主对比）与默认（模型 config 声明的 `mamba_ssm_dtype: float32`） |
| wein 配置 | 生产形态：`--graph --ctx-len 8192 --kv-pages 512 --chunk-size 2048`，`token-delay 0`，`mix_cap 192` 默认，radix 前缀缓存开 |
| vLLM 配置 | `vllm 0.32.0` / torch 2.13.0+cu130：`--max-model-len 8192 --gpu-memory-utilization 0.85 --max-num-batched-tokens 2048 --max-num-seqs 256`（24G 卡上的**实测最优工作点**，0.9 在本卡是不可用点）+ `VLLM_USE_FLASHINFER_SAMPLER=0`；模型 = 滤掉视觉塔的纯文本副本 `Qwen3.5-4B-text`（与 wein 跑同一组文本张量） |
| 环境 | 同一台 4090/WSL；每块前后记录 Windows 侧 compute-apps、SM/MEM 时钟、温度、功耗；代理变量全 unset；两侧 setsid 独占 GPU（一次只有一个服务） |
| 采样 | wein-bench `--repeat`（满载 2、其余 3）+ 3 轮跨进程 ⇒ 每格 6-9 次取中位；长上下文 `--repeats 3 --cold --hot` |
| 环境证据 | 全程 SM 2595 MHz / MEM 10501 MHz / 56-59 ℃、块间 GPU util 1-5%；两端在**同一**桌面图形背景下交替，偏差相消 |

## 二、结果（3 轮中位）

### 口径②：**双方 bf16 递推状态**（唯一公平的 apples-to-apples）

| 场景 | 指标 | wein | vLLM 0.32.0 | 差 |
|---|---|---|---|---|
| 满载 c=100<br>n=4000 mt=32 | **tps** | **4408.0** | 4468.6 | **−1.4%** |
| | ITL p50 | 21.30 | 19.11 | +11.5% |
| | **ITL p99** | **26.60** | 35.99 | **−26.1%** |
| | **TTFT p50** | **49.1** | 93.4 | **−47.4%** |
| | TTFT p99 | **107.4** | 190.0 | **−43.5%** |
| 中载 c=16 | tps | 1138.7 | 1195.2 | −4.7% |
| | ITL p50 / p99 | 13.10 / 15.64 | 11.99 / 14.13 | +9.3% / +10.7% |
| | TTFT p50 | **41.8** | 51.4 | **−18.7%** |
| 轻载 c=4 | tps | 316.2 | 315.3 | **+0.3%** |
| | ITL p50 / p99 | 11.83 / 13.98 | 11.33 / 13.37 | +4.4% / +4.6% |
| | TTFT p50 | **41.6** | 48.7 | −14.6% |
| 单流 c=1 | tps | 84.6 | 93.0 | −9.0% |
| | ITL p50 / p99 | 11.49 / 13.51 | 10.31 / 12.35 | +11.4% / +9.4% |
| | TTFT p50 | 42.8 | **26.1** | +64% |
| 3009 token<br>冷 prefill | **冷 TTFT** | 223.8 | 203.5 | +10.0% |
| | **热 TTFT**（前缀命中） | **19.8** | 30.0 | **−34%** |
| | decode ITL p50 | 11.93 | 10.41 | +14.6% |
| 6009 token<br>冷 prefill | **冷 TTFT** | 429.7 | 398.7 | +7.8% |
| | **热 TTFT** | **19.9** | 33.4 | **−40%** |
| | decode ITL p50 | 12.36 | 10.63 | +16.3% |

**逐轮配对差（满载 tps）**：r1 −1.03% / r2 −2.98% / r3 −1.36%；单流 c=1 三轮
−8.97 / −9.16 / −9.12%（极稳，是本轮最确定的一条差距）。

### 口径①：vLLM 模型声明默认（f32 递推状态，vLLM 独有的工作点）

| 指标 | vLLM f32 | wein(bf16) |
|---|---|---|
| 满载 tps | 3544.6（3533/3544/3589） | **4408.0（+24.4%）** |
| ITL p50 / p99 | 25.33 / 46.06 | 21.30 / 26.60 |
| TTFT p50 | 97.8 | 49.1 |

⇒ 若双方都按"模型作者声明的精度"跑，我们**快 24%**；把 vLLM 放到与我们同样的
bf16 状态（它有一个显式 flag 做这件事，且有精度代价），差距收敛到 **−1.4%**。

## 三、结论

1. **满载吞吐 = 平价**（−1.4%，且逐轮配对差全在 −3% 以内）。我们已经把
   上年 11 月那条"×2.9 / ×1.08 / −13.5%"的历史差距全部吃掉。
2. **延迟形状上我们占优**：ITL p99 **−26%**、TTFT p50 **−47%**、长 prompt 热
   TTFT **−34~40%**（radix 前缀缓存 + 4096-token 快照粒度）。
3. **vLLM 仍赢两处，都在"单步稳态"**：ITL p50 +11.5%（满载）/ +16%
   （长上下文 decode）、单流 c=1 tps −9%。这两处的物理来源已查清（kernel 级账）：
   我们的 gemm/lm_head 吞吐**不输**（满载 718 vs 670 GB/s、单流 957 vs 869 GB/s），
   差在*每步组织*——vLLM 把 8 层 mamba + lm_head 的固定开销压得更紧，我们的
   norm/eltw 家族仍是 1.05ms/步 vs 其 0.645ms。这是唯一还剩的量化空间（约 2%），
   需破位级合同才拿得到。
4. **冷 prefill 仍小幅落后**（3009 +10%、6009 +7.8%），但热路径反超
   （19.9 vs 33.4ms）——Agent 会话的真实形态是热路径。
5. 单流 c=1 的 −9% 我们**没有针对性优化过**（历史刀全是批量/长上下文），
   若要补：c=1 的步里 8.4GB 权重读是唯一账，vLLM 靠 splitK GEMV 把
   m=1 的 tile 占用率抬起来了；我们有 lm_head splitK 的现成空间（438→900GB/s
   那条留刀），换算约 +0.5ms/步 ⇒ 可以把 −9% 收窄到 −4% 左右。

## 四、本轮顺带修的真 bug（停服段错误）

终验把服务端 SIGTERM 停掉时，每块都看到：

```
thread 'tokio-rt-worker' panicked at crates/wein-kernel/src/cuda/mod.rs:402:
    cuMemFree: CuError { code: 4, symbol: "cuMemFree" }
bench/final_vs_vllm.sh: line 44: 294005 Segmentation fault (core dumped)
```

终止路径上 CUDA 上下文已销毁，`cuMemFree` 只能拿到 `INVALID_CONTEXT`——驱动会
随上下文回收显存，这里 `expect` 在 Drop 里 panic 就是 abort，运维看到的是
"core dumped"，像崩溃。修法：`DevBuf::drop` 只对
`DEINITIALIZED(3) / INVALID_CONTEXT(4) / CONTEXT_IS_DESTROYED(709)` 放行，
其余错误码照旧炸（泄漏检测不放水）。验证：SIGTERM → `drain 完成（无在飞请求）：
退出码 0`，无 panic、无段错误。

## 五、产出

- `bench/final_vs_vllm.sh`（新，可复现；`bash bench/final_vs_vllm.sh 3`）
- `bench-reports/final-vs-vllm-20261107-122345/`：18 份原始 JSON + 两端 server.log
  + `env.txt` + `gpu.log` + `run.log`
- `crates/wein-kernel/src/cuda/{mod,driver}.rs`：停服释放容错
