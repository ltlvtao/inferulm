# battery 20261108-111716

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=165722 p50=0.53ms p99=1.49ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.46ms ttft_p99=12.06ms itl_p50=0.002ms itl_p99=9.64ms tps=81214 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.61ms itl_p50=0.00ms itl_p99=17.69ms tps=43893 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.94ms ttft_p99=2.18ms itl_mean=0.514ms tps=144089 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.77ms itl_mean=1.67ms tps=45675 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.82ms ttft_p99=2.25ms itl_mean=0.506ms tps=160951 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.32ms ttft_p99=6.60ms itl_mean=0.47ms tps=141297 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.41ms ttft_p99=51.61ms itl_p50=12.53ms tps=262.1 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=47.12ms ttft_p99=219.34ms itl_p50=20.98ms tps=4267.4 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.33ms ttft_p99=43.85ms itl_p50=11.56ms tps=300.6 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=50.18ms ttft_p99=199.55ms itl_p50=20.86ms tps=4323.8 |
