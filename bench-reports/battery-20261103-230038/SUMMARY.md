# battery 20261103-230038

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=166739 p50=0.53ms p99=1.50ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.86ms ttft_p99=12.08ms itl_p50=0.002ms itl_p99=9.62ms tps=81011 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.61ms itl_p50=0.00ms itl_p99=17.70ms tps=43755 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.92ms ttft_p99=2.17ms itl_mean=0.551ms tps=154042 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.67ms itl_mean=1.67ms tps=45388 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.77ms ttft_p99=2.30ms itl_mean=0.411ms tps=161670 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.95ms ttft_p99=6.84ms itl_mean=0.46ms tps=143843 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.15ms ttft_p99=57.34ms itl_p50=13.83ms tps=243.3 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=71.63ms ttft_p99=267.97ms itl_p50=24.71ms tps=3421.5 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.78ms ttft_p99=44.48ms itl_p50=12.21ms tps=283.9 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=70.55ms ttft_p99=243.67ms itl_p50=23.60ms tps=3575.4 |
