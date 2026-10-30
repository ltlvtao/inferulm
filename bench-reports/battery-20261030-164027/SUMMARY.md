# battery 20261030-164027

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=163570 p50=0.54ms p99=1.49ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.96ms ttft_p99=12.19ms itl_p50=0.002ms itl_p99=9.55ms tps=71892 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.39ms itl_p50=0.00ms itl_p99=17.65ms tps=43950 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.22ms ttft_p99=4.91ms itl_mean=0.035ms tps=314037 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.51ms itl_mean=1.67ms tps=45838 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.29ms ttft_p99=4.47ms itl_mean=0.067ms tps=290170 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.27ms ttft_p99=6.70ms itl_mean=0.43ms tps=118258 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.06ms ttft_p99=44.32ms itl_p50=14.19ms tps=217.5 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=68.50ms ttft_p99=264.77ms itl_p50=24.74ms tps=3441.9 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.14ms ttft_p99=44.44ms itl_p50=12.36ms tps=258.2 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=68.12ms ttft_p99=269.46ms itl_p50=22.90ms tps=3633.0 |
