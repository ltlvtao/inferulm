# battery 20261029-205735

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=165014 p50=0.51ms p99=1.59ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.69ms ttft_p99=12.31ms itl_p50=0.002ms itl_p99=9.60ms tps=80130 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.34ms itl_p50=0.00ms itl_p99=17.70ms tps=43714 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.15ms ttft_p99=3.54ms itl_mean=0.028ms tps=310540 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.20ms itl_mean=1.65ms tps=45908 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.38ms ttft_p99=3.67ms itl_mean=0.059ms tps=289848 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=6.64ms ttft_p99=8.66ms itl_mean=0.78ms tps=91740 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=45.77ms ttft_p99=55.99ms itl_p50=20.62ms tps=168.4 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=195.24ms ttft_p99=293.19ms itl_p50=51.33ms tps=1724.7 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=46.31ms ttft_p99=46.73ms itl_p50=18.97ms tps=180.5 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=199.36ms ttft_p99=290.67ms itl_p50=49.76ms tps=1776.2 |
