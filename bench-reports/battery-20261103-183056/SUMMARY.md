# battery 20261103-183056

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=169689 p50=0.51ms p99=1.56ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.62ms ttft_p99=12.06ms itl_p50=0.002ms itl_p99=9.49ms tps=81209 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.66ms itl_p50=0.00ms itl_p99=17.59ms tps=43896 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.91ms ttft_p99=2.21ms itl_mean=0.541ms tps=155103 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.16ms itl_mean=1.68ms tps=45365 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.86ms ttft_p99=2.07ms itl_mean=0.495ms tps=161112 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.61ms ttft_p99=7.40ms itl_mean=0.44ms tps=147680 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.79ms ttft_p99=56.29ms itl_p50=13.81ms tps=242.0 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=70.06ms ttft_p99=265.72ms itl_p50=25.05ms tps=3423.4 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.64ms ttft_p99=44.35ms itl_p50=12.20ms tps=281.1 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=71.14ms ttft_p99=240.21ms itl_p50=23.99ms tps=3560.0 |
