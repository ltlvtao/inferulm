# battery 20261109-111456

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=164975 p50=0.54ms p99=1.49ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.62ms ttft_p99=12.95ms itl_p50=0.002ms itl_p99=9.48ms tps=81185 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.70ms itl_p50=0.00ms itl_p99=17.73ms tps=43722 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.78ms ttft_p99=2.35ms itl_mean=0.414ms tps=167682 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.30ms itl_mean=1.67ms tps=45333 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.80ms ttft_p99=2.44ms itl_mean=0.533ms tps=160677 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.45ms ttft_p99=7.35ms itl_mean=0.46ms tps=144787 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.95ms ttft_p99=52.21ms itl_p50=12.34ms tps=266.2 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=47.40ms ttft_p99=219.10ms itl_p50=20.83ms tps=4266.1 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.01ms ttft_p99=44.52ms itl_p50=11.75ms tps=300.6 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=47.46ms ttft_p99=200.12ms itl_p50=19.81ms tps=4478.7 |
