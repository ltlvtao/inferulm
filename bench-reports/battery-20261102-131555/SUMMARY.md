# battery 20261102-131555

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=168887 p50=0.51ms p99=1.63ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.82ms ttft_p99=12.60ms itl_p50=0.002ms itl_p99=9.56ms tps=71033 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.44ms itl_p50=0.00ms itl_p99=17.70ms tps=43624 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.78ms ttft_p99=3.00ms itl_mean=0.477ms tps=160983 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.90ms itl_mean=1.67ms tps=45299 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.91ms ttft_p99=2.32ms itl_mean=0.537ms tps=154671 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.49ms ttft_p99=6.89ms itl_mean=0.48ms tps=136375 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=42.88ms ttft_p99=79.21ms itl_p50=14.34ms tps=222.2 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=67.62ms ttft_p99=259.16ms itl_p50=25.03ms tps=3428.5 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.26ms ttft_p99=43.67ms itl_p50=12.09ms tps=283.7 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=68.49ms ttft_p99=233.31ms itl_p50=23.67ms tps=3595.6 |
