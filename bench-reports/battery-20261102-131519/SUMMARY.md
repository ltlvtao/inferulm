# battery 20261102-131519

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=156316 p50=0.55ms p99=1.71ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.43ms ttft_p99=11.97ms itl_p50=0.002ms itl_p99=9.61ms tps=71512 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.57ms itl_p50=0.00ms itl_p99=17.73ms tps=43758 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.77ms ttft_p99=2.79ms itl_mean=0.376ms tps=167738 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.04ms itl_mean=1.75ms tps=40990 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.15ms ttft_p99=2.41ms itl_mean=0.568ms tps=144186 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.53ms ttft_p99=7.55ms itl_mean=0.50ms tps=139836 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=47.28ms ttft_p99=56.38ms itl_p50=14.10ms tps=240.5 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=70.32ms ttft_p99=261.26ms itl_p50=24.97ms tps=3402.4 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.73ms ttft_p99=44.08ms itl_p50=12.78ms tps=279.4 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=70.37ms ttft_p99=237.38ms itl_p50=23.75ms tps=3562.0 |
