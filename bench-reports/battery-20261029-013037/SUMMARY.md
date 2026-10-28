# battery 20261029-013037

| 层 | 指标（中位数; L0-L3: c=100, n=500, max_tokens=32; L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=169691 p50=0.52ms p99=1.36ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.58ms ttft_p99=12.09ms itl_p50=0.003ms itl_p99=9.66ms tps=71633 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.37ms itl_p50=0.00ms itl_p99=17.78ms tps=43411 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.13ms ttft_p99=2.18ms itl_mean=0.027ms tps=312436 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.72ms itl_mean=1.64ms tps=28426 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.17ms ttft_p99=2.52ms itl_mean=0.037ms tps=338228 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=6.63ms ttft_p99=12.99ms itl_mean=0.74ms tps=49014 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=75.64ms ttft_p99=76.16ms itl_p50=31.28ms tps=115.4 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=72.29ms ttft_p99=73.96ms itl_p50=28.79ms tps=122.1 |
