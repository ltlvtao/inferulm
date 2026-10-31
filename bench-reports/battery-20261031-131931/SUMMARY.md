# battery 20261031-131931

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=158943 p50=0.54ms p99=1.83ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.83ms ttft_p99=12.71ms itl_p50=0.002ms itl_p99=9.63ms tps=71729 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.26ms itl_p50=0.00ms itl_p99=17.79ms tps=43764 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.18ms ttft_p99=3.17ms itl_mean=0.043ms tps=289547 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.16ms itl_mean=1.68ms tps=45428 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.09ms ttft_p99=3.22ms itl_mean=0.065ms tps=289428 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.00ms ttft_p99=7.78ms itl_mean=0.46ms tps=117763 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=42.14ms ttft_p99=43.91ms itl_p50=14.90ms tps=210.8 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=75.19ms ttft_p99=263.47ms itl_p50=24.78ms tps=3402.0 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.04ms ttft_p99=44.45ms itl_p50=12.45ms tps=256.4 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=73.49ms ttft_p99=262.57ms itl_p50=23.13ms tps=3531.0 |
