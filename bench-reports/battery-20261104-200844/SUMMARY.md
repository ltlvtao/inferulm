# battery 20261104-200844

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=166014 p50=0.54ms p99=1.61ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.77ms ttft_p99=13.20ms itl_p50=0.002ms itl_p99=9.57ms tps=77499 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=21.05ms itl_p50=0.00ms itl_p99=17.57ms tps=43497 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.83ms ttft_p99=2.31ms itl_mean=0.505ms tps=161227 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.06ms itl_mean=1.67ms tps=45518 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.79ms ttft_p99=3.65ms itl_mean=0.601ms tps=118616 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.71ms ttft_p99=7.37ms itl_mean=0.45ms tps=161062 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.33ms ttft_p99=54.69ms itl_p50=13.80ms tps=247.7 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=67.67ms ttft_p99=262.90ms itl_p50=24.31ms tps=3505.3 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.48ms ttft_p99=45.36ms itl_p50=12.22ms tps=286.8 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=67.82ms ttft_p99=233.01ms itl_p50=23.17ms tps=3648.2 |
