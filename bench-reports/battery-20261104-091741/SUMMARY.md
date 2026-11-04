# battery 20261104-091741

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=166237 p50=0.52ms p99=1.70ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.56ms ttft_p99=12.07ms itl_p50=0.002ms itl_p99=9.58ms tps=79662 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.51ms itl_p50=0.01ms itl_p99=17.71ms tps=44005 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.71ms ttft_p99=2.24ms itl_mean=0.393ms tps=161689 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.33ms itl_mean=1.67ms tps=45234 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.87ms ttft_p99=2.19ms itl_mean=0.527ms tps=161091 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.83ms ttft_p99=7.57ms itl_mean=0.46ms tps=131577 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.26ms ttft_p99=59.20ms itl_p50=14.40ms tps=238.0 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=70.56ms ttft_p99=263.49ms itl_p50=24.46ms tps=3451.3 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.71ms ttft_p99=44.04ms itl_p50=12.62ms tps=275.2 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=68.08ms ttft_p99=233.98ms itl_p50=23.39ms tps=3633.0 |
