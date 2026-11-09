# battery 20261109-205320

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=163720 p50=0.51ms p99=1.78ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=11.16ms ttft_p99=12.31ms itl_p50=0.003ms itl_p99=9.49ms tps=79817 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.42ms itl_p50=0.00ms itl_p99=17.69ms tps=44162 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.82ms ttft_p99=2.22ms itl_mean=0.508ms tps=161528 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.92ms itl_mean=1.70ms tps=41322 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.83ms ttft_p99=2.32ms itl_mean=0.352ms tps=168113 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.40ms ttft_p99=6.83ms itl_mean=0.49ms tps=140758 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.96ms ttft_p99=53.49ms itl_p50=12.08ms tps=273.1 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=60.19ms ttft_p99=162.38ms itl_p50=20.75ms tps=4213.3 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.60ms ttft_p99=44.42ms itl_p50=11.23ms tps=309.0 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=63.55ms ttft_p99=140.53ms itl_p50=20.65ms tps=4297.7 |
