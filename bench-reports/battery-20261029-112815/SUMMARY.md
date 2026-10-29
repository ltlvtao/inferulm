# battery 20261029-112815

| 层 | 指标（中位数; L0-L3: c=100, n=500, max_tokens=32; L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=142868 p50=0.58ms p99=1.77ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.99ms ttft_p99=12.69ms itl_p50=0.002ms itl_p99=9.49ms tps=81477 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.61ms itl_p50=0.00ms itl_p99=17.79ms tps=44088 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.21ms ttft_p99=4.12ms itl_mean=0.032ms tps=312086 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.74ms itl_mean=1.66ms tps=28211 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.14ms ttft_p99=2.38ms itl_mean=0.046ms tps=312121 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=6.39ms ttft_p99=9.63ms itl_mean=0.76ms tps=49524 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=117.79ms ttft_p99=118.52ms itl_p50=33.48ms tps=99.2 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=111.72ms ttft_p99=113.80ms itl_p50=31.24ms tps=104.2 |
