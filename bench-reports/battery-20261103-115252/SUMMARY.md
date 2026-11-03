# battery 20261103-115252

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=170494 p50=0.51ms p99=1.53ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.66ms ttft_p99=12.09ms itl_p50=0.002ms itl_p99=9.60ms tps=70393 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.84ms itl_p50=0.00ms itl_p99=17.60ms tps=43786 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.81ms ttft_p99=2.15ms itl_mean=0.468ms tps=161294 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.29ms itl_mean=1.69ms tps=45105 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.88ms ttft_p99=2.27ms itl_mean=0.432ms tps=155223 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.58ms ttft_p99=6.97ms itl_mean=0.45ms tps=158706 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.65ms ttft_p99=57.64ms itl_p50=13.72ms tps=240.9 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=69.57ms ttft_p99=276.19ms itl_p50=25.10ms tps=3393.6 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.86ms ttft_p99=44.78ms itl_p50=12.54ms tps=278.9 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=70.74ms ttft_p99=234.95ms itl_p50=23.79ms tps=3553.8 |
