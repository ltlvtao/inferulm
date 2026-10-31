# battery 20261031-134113

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=163666 p50=0.53ms p99=1.79ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.51ms ttft_p99=11.86ms itl_p50=0.002ms itl_p99=9.63ms tps=79877 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.78ms itl_p50=0.00ms itl_p99=17.82ms tps=43537 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.17ms ttft_p99=2.70ms itl_mean=0.072ms tps=290894 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.49ms itl_mean=1.68ms tps=45198 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.07ms ttft_p99=3.27ms itl_mean=0.046ms tps=290157 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.19ms ttft_p99=12.24ms itl_mean=0.49ms tps=121362 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.84ms ttft_p99=71.31ms itl_p50=15.08ms tps=207.0 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=73.91ms ttft_p99=288.95ms itl_p50=24.65ms tps=3411.6 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.21ms ttft_p99=43.75ms itl_p50=12.72ms tps=250.2 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=73.52ms ttft_p99=263.32ms itl_p50=23.31ms tps=3542.7 |
