# battery 20261031-173620

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=157950 p50=0.53ms p99=1.32ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.50ms ttft_p99=12.58ms itl_p50=0.002ms itl_p99=9.60ms tps=80491 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.44ms itl_p50=0.01ms itl_p99=17.72ms tps=43605 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.25ms ttft_p99=2.93ms itl_mean=0.059ms tps=290549 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.75ms itl_mean=1.66ms tps=45738 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.45ms ttft_p99=3.76ms itl_mean=0.074ms tps=288332 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.42ms ttft_p99=6.62ms itl_mean=0.53ms tps=108632 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.89ms ttft_p99=61.13ms itl_p50=14.35ms tps=223.5 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=75.41ms ttft_p99=270.18ms itl_p50=25.10ms tps=3412.8 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.40ms ttft_p99=44.04ms itl_p50=12.14ms tps=265.4 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=69.99ms ttft_p99=238.25ms itl_p50=24.05ms tps=3561.1 |
