# battery 20261031-150439

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=148962 p50=0.53ms p99=2.20ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.84ms ttft_p99=12.48ms itl_p50=0.003ms itl_p99=9.64ms tps=80286 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=21.15ms itl_p50=0.01ms itl_p99=17.75ms tps=43413 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.23ms ttft_p99=2.88ms itl_mean=0.029ms tps=290508 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.61ms itl_mean=1.67ms tps=45379 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.40ms ttft_p99=3.01ms itl_mean=0.082ms tps=289264 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.38ms ttft_p99=8.87ms itl_mean=0.44ms tps=114804 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.92ms ttft_p99=60.42ms itl_p50=13.64ms tps=230.8 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=73.67ms ttft_p99=265.96ms itl_p50=24.89ms tps=3423.8 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.19ms ttft_p99=43.71ms itl_p50=12.78ms tps=257.7 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=71.36ms ttft_p99=254.57ms itl_p50=23.66ms tps=3549.7 |
