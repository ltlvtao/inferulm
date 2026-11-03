# battery 20261103-115228

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=174267 p50=0.51ms p99=1.34ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=11.05ms ttft_p99=12.95ms itl_p50=0.003ms itl_p99=9.54ms tps=79698 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.30ms itl_p50=0.01ms itl_p99=17.80ms tps=43537 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.69ms ttft_p99=2.91ms itl_mean=0.370ms tps=168037 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.04ms itl_mean=1.68ms tps=45028 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.85ms ttft_p99=2.53ms itl_mean=0.558ms tps=154617 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.39ms ttft_p99=6.50ms itl_mean=0.46ms tps=131478 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.88ms ttft_p99=57.89ms itl_p50=13.91ms tps=239.2 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=71.82ms ttft_p99=263.00ms itl_p50=25.04ms tps=3401.3 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.22ms ttft_p99=43.96ms itl_p50=13.12ms tps=269.7 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=71.27ms ttft_p99=236.21ms itl_p50=23.72ms tps=3556.6 |
