# battery 20261102-103930

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=170770 p50=0.52ms p99=1.50ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.59ms ttft_p99=11.99ms itl_p50=0.003ms itl_p99=9.59ms tps=70251 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.82ms itl_p50=0.01ms itl_p99=17.79ms tps=43398 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.12ms ttft_p99=2.53ms itl_mean=0.054ms tps=309726 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.92ms itl_mean=1.68ms tps=45807 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.19ms ttft_p99=3.37ms itl_mean=0.125ms tps=271299 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=4.69ms ttft_p99=7.04ms itl_mean=0.46ms tps=115255 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.03ms ttft_p99=58.36ms itl_p50=14.41ms tps=225.5 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=71.25ms ttft_p99=261.39ms itl_p50=25.03ms tps=3435.5 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.03ms ttft_p99=43.77ms itl_p50=12.01ms tps=269.9 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=71.74ms ttft_p99=235.86ms itl_p50=23.66ms tps=3584.3 |
