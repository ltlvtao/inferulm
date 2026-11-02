# battery 20261102-120558

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=171029 p50=0.52ms p99=1.43ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.67ms ttft_p99=12.54ms itl_p50=0.002ms itl_p99=9.50ms tps=80335 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.40ms itl_p50=0.00ms itl_p99=17.73ms tps=43621 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.20ms ttft_p99=3.69ms itl_mean=0.048ms tps=290855 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.94ms itl_mean=1.67ms tps=45563 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.22ms ttft_p99=4.24ms itl_mean=0.084ms tps=291401 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.11ms ttft_p99=6.97ms itl_mean=0.53ms tps=114578 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.36ms ttft_p99=57.65ms itl_p50=13.90ms tps=231.1 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=72.72ms ttft_p99=261.48ms itl_p50=24.93ms tps=3414.3 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.04ms ttft_p99=43.95ms itl_p50=12.37ms tps=267.4 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=72.12ms ttft_p99=214.69ms itl_p50=23.88ms tps=3545.6 |
