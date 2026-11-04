# battery 20261104-133644

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=163584 p50=0.54ms p99=1.58ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.89ms ttft_p99=12.38ms itl_p50=0.002ms itl_p99=9.63ms tps=80814 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.23ms itl_p50=0.00ms itl_p99=17.80ms tps=43175 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.82ms ttft_p99=2.01ms itl_mean=0.373ms tps=167589 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.77ms itl_mean=1.68ms tps=45599 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.80ms ttft_p99=2.16ms itl_mean=0.559ms tps=149285 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.42ms ttft_p99=7.73ms itl_mean=0.46ms tps=158643 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=42.90ms ttft_p99=79.78ms itl_p50=13.79ms tps=231.0 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=69.61ms ttft_p99=261.91ms itl_p50=24.35ms tps=3494.9 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.70ms ttft_p99=43.78ms itl_p50=13.03ms tps=275.2 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=68.22ms ttft_p99=234.50ms itl_p50=23.54ms tps=3632.4 |
