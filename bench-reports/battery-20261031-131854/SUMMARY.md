# battery 20261031-131854

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=166605 p50=0.52ms p99=1.80ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.64ms ttft_p99=12.40ms itl_p50=0.002ms itl_p99=9.51ms tps=79547 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.29ms itl_p50=0.00ms itl_p99=17.55ms tps=44067 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.17ms ttft_p99=4.06ms itl_mean=0.105ms tps=289093 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.59ms itl_mean=1.67ms tps=45867 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.29ms ttft_p99=3.14ms itl_mean=0.084ms tps=289303 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.34ms ttft_p99=14.02ms itl_mean=0.49ms tps=117144 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=42.35ms ttft_p99=43.96ms itl_p50=14.18ms tps=220.5 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=75.87ms ttft_p99=264.26ms itl_p50=24.70ms tps=3388.8 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.18ms ttft_p99=44.79ms itl_p50=12.24ms tps=256.4 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=79.71ms ttft_p99=263.79ms itl_p50=23.19ms tps=3504.4 |
