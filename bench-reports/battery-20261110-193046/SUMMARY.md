# battery 20261110-193046

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=157051 p50=0.56ms p99=1.84ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.77ms ttft_p99=12.68ms itl_p50=0.003ms itl_p99=9.52ms tps=80116 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.25ms itl_p50=0.00ms itl_p99=17.79ms tps=43560 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.90ms ttft_p99=2.63ms itl_mean=0.432ms tps=161109 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.73ms itl_mean=1.72ms tps=41299 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.87ms ttft_p99=2.58ms itl_mean=0.496ms tps=153678 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.58ms ttft_p99=8.31ms itl_mean=0.53ms tps=130414 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.60ms ttft_p99=52.88ms itl_p50=12.18ms tps=270.4 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=60.42ms ttft_p99=161.91ms itl_p50=20.85ms tps=4218.5 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.15ms ttft_p99=45.95ms itl_p50=11.57ms tps=306.7 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=60.30ms ttft_p99=141.38ms itl_p50=19.74ms tps=4472.1 |
