# battery 20261031-113926

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=133608 p50=0.57ms p99=2.76ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.61ms ttft_p99=12.01ms itl_p50=0.003ms itl_p99=9.59ms tps=80554 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.87ms itl_p50=0.01ms itl_p99=17.59ms tps=43696 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.41ms ttft_p99=3.88ms itl_mean=0.028ms tps=313000 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.26ms itl_mean=1.68ms tps=45542 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.13ms ttft_p99=3.79ms itl_mean=0.050ms tps=290277 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.48ms ttft_p99=7.05ms itl_mean=0.48ms tps=126107 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=42.93ms ttft_p99=43.76ms itl_p50=13.67ms tps=225.5 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=73.60ms ttft_p99=264.06ms itl_p50=24.25ms tps=3439.6 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.91ms ttft_p99=44.68ms itl_p50=12.71ms tps=255.2 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=74.00ms ttft_p99=262.29ms itl_p50=23.20ms tps=3559.2 |
