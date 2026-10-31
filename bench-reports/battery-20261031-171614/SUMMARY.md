# battery 20261031-171614

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=150692 p50=0.57ms p99=2.21ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.96ms ttft_p99=12.41ms itl_p50=0.002ms itl_p99=9.59ms tps=71346 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.57ms itl_p50=0.00ms itl_p99=17.62ms tps=43803 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.36ms ttft_p99=2.77ms itl_mean=0.069ms tps=289698 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.14ms itl_mean=1.67ms tps=45478 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.17ms ttft_p99=2.41ms itl_mean=0.046ms tps=290232 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.20ms ttft_p99=8.59ms itl_mean=0.48ms tps=112009 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.36ms ttft_p99=60.96ms itl_p50=14.82ms tps=216.7 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=67.11ms ttft_p99=266.00ms itl_p50=24.36ms tps=3527.1 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.95ms ttft_p99=47.76ms itl_p50=12.95ms tps=260.3 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=67.29ms ttft_p99=230.85ms itl_p50=22.67ms tps=3709.0 |
