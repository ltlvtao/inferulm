# battery 20261106-093552

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=160736 p50=0.54ms p99=1.78ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.66ms ttft_p99=12.06ms itl_p50=0.002ms itl_p99=9.61ms tps=80910 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=21.03ms itl_p50=0.01ms itl_p99=17.71ms tps=43348 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.88ms ttft_p99=2.08ms itl_mean=0.562ms tps=154669 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.74ms itl_mean=1.72ms tps=44470 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.84ms ttft_p99=2.20ms itl_mean=0.439ms tps=161107 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.61ms ttft_p99=6.56ms itl_mean=0.47ms tps=145001 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.28ms ttft_p99=56.19ms itl_p50=13.88ms tps=244.5 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=68.72ms ttft_p99=262.34ms itl_p50=24.71ms tps=3464.4 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.42ms ttft_p99=44.04ms itl_p50=13.14ms tps=272.5 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=69.18ms ttft_p99=237.74ms itl_p50=23.73ms tps=3596.2 |
