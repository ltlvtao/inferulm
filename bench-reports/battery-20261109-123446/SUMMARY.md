# battery 20261109-123446

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=170456 p50=0.51ms p99=1.61ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.79ms ttft_p99=12.40ms itl_p50=0.002ms itl_p99=9.53ms tps=80042 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.98ms itl_p50=0.00ms itl_p99=17.60ms tps=43585 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.83ms ttft_p99=2.16ms itl_mean=0.550ms tps=153746 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.95ms itl_mean=1.67ms tps=45367 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.81ms ttft_p99=2.20ms itl_mean=0.477ms tps=161180 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.17ms ttft_p99=6.30ms itl_mean=0.45ms tps=138047 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.63ms ttft_p99=55.39ms itl_p50=12.63ms tps=259.8 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=47.76ms ttft_p99=223.21ms itl_p50=20.68ms tps=4293.9 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.96ms ttft_p99=43.81ms itl_p50=12.22ms tps=282.7 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=50.33ms ttft_p99=199.33ms itl_p50=20.71ms tps=4363.0 |
