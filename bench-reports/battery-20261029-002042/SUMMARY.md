# battery 20261029-002042

| 层 | 指标（中位数; L0-L3: c=100, n=500, max_tokens=32; L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=165559 p50=0.53ms p99=1.50ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.78ms ttft_p99=12.10ms itl_p50=0.002ms itl_p99=9.49ms tps=81567 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.30ms itl_p50=0.00ms itl_p99=17.74ms tps=43554 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.05ms ttft_p99=4.59ms itl_mean=0.036ms tps=311561 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.65ms itl_mean=1.65ms tps=28526 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.22ms ttft_p99=4.19ms itl_mean=0.058ms tps=290868 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=6.76ms ttft_p99=12.39ms itl_mean=0.84ms tps=47205 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=72.35ms ttft_p99=72.47ms itl_p50=30.91ms tps=115.7 |
