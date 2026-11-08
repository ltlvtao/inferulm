# battery 20261108-121037

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=161342 p50=0.55ms p99=1.59ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.48ms ttft_p99=12.38ms itl_p50=0.003ms itl_p99=9.51ms tps=79740 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.65ms itl_p50=0.00ms itl_p99=17.79ms tps=43567 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.85ms ttft_p99=1.99ms itl_mean=0.502ms tps=155100 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.29ms itl_mean=1.78ms tps=44170 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.88ms ttft_p99=2.30ms itl_mean=0.354ms tps=166501 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.46ms ttft_p99=6.22ms itl_mean=0.47ms tps=159121 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.83ms ttft_p99=51.32ms itl_p50=12.11ms tps=273.7 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=47.20ms ttft_p99=216.97ms itl_p50=20.99ms tps=4255.8 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.43ms ttft_p99=43.71ms itl_p50=11.79ms tps=293.7 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=50.06ms ttft_p99=193.10ms itl_p50=20.85ms tps=4321.1 |
