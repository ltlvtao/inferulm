# battery 20261107-232102

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=165231 p50=0.52ms p99=2.19ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.42ms ttft_p99=12.36ms itl_p50=0.002ms itl_p99=9.60ms tps=80522 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.29ms itl_p50=0.00ms itl_p99=17.78ms tps=43596 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.85ms ttft_p99=2.23ms itl_mean=0.480ms tps=161284 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.97ms itl_mean=1.67ms tps=45415 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.92ms ttft_p99=2.33ms itl_mean=0.477ms tps=155362 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.49ms ttft_p99=6.99ms itl_mean=0.47ms tps=128996 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.74ms ttft_p99=53.48ms itl_p50=12.28ms tps=269.4 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=47.78ms ttft_p99=217.86ms itl_p50=20.83ms tps=4290.5 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.91ms ttft_p99=43.69ms itl_p50=11.64ms tps=307.6 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=48.39ms ttft_p99=188.00ms itl_p50=19.89ms tps=4498.9 |
