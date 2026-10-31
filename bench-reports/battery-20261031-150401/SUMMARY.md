# battery 20261031-150401

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=163926 p50=0.54ms p99=1.70ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.58ms ttft_p99=12.15ms itl_p50=0.002ms itl_p99=9.62ms tps=72602 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.49ms itl_p50=0.00ms itl_p99=17.70ms tps=43375 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.22ms ttft_p99=3.64ms itl_mean=0.110ms tps=285845 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.36ms itl_mean=1.68ms tps=45656 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.16ms ttft_p99=2.98ms itl_mean=0.080ms tps=289651 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=4.85ms ttft_p99=8.65ms itl_mean=0.47ms tps=118157 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.86ms ttft_p99=63.05ms itl_p50=14.72ms tps=221.2 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=74.49ms ttft_p99=217.66ms itl_p50=24.92ms tps=3430.2 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.70ms ttft_p99=45.71ms itl_p50=12.72ms tps=264.6 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=69.19ms ttft_p99=242.79ms itl_p50=23.69ms tps=3556.5 |
