# battery 20261029-091642

| 层 | 指标（中位数; L0-L3: c=100, n=500, max_tokens=32; L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=162763 p50=0.53ms p99=1.94ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.60ms ttft_p99=11.80ms itl_p50=0.002ms itl_p99=9.48ms tps=80656 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.66ms itl_p50=0.01ms itl_p99=17.67ms tps=43376 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.10ms ttft_p99=5.47ms itl_mean=0.025ms tps=312220 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.78ms itl_mean=1.68ms tps=28423 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.05ms ttft_p99=2.80ms itl_mean=0.050ms tps=312386 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=6.87ms ttft_p99=12.75ms itl_mean=0.80ms tps=48303 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=112.25ms ttft_p99=115.84ms itl_p50=32.91ms tps=100.2 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=111.98ms ttft_p99=114.93ms itl_p50=31.94ms tps=104.3 |
