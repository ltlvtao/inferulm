# battery 20261031-150946

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=161153 p50=0.53ms p99=1.95ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.70ms ttft_p99=12.49ms itl_p50=0.002ms itl_p99=9.59ms tps=79159 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.61ms itl_p50=0.00ms itl_p99=17.71ms tps=43366 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.25ms ttft_p99=5.33ms itl_mean=0.066ms tps=289617 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.04ms itl_mean=1.66ms tps=45253 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.12ms ttft_p99=4.64ms itl_mean=0.136ms tps=290335 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.11ms ttft_p99=7.13ms itl_mean=0.45ms tps=114831 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=47.38ms ttft_p99=95.61ms itl_p50=16.12ms tps=191.4 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=69.26ms ttft_p99=264.55ms itl_p50=25.22ms tps=3448.0 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.21ms ttft_p99=43.85ms itl_p50=12.60ms tps=262.8 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=72.06ms ttft_p99=239.54ms itl_p50=24.15ms tps=3537.1 |
