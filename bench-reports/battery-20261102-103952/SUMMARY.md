# battery 20261102-103952

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=169573 p50=0.52ms p99=1.53ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.64ms ttft_p99=12.11ms itl_p50=0.003ms itl_p99=9.59ms tps=71149 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.32ms itl_p50=0.00ms itl_p99=17.87ms tps=43621 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.20ms ttft_p99=2.95ms itl_mean=0.037ms tps=312703 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.18ms itl_mean=1.68ms tps=45372 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.14ms ttft_p99=4.45ms itl_mean=0.054ms tps=290187 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.11ms ttft_p99=6.72ms itl_mean=0.42ms tps=135678 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.19ms ttft_p99=56.56ms itl_p50=14.05ms tps=232.3 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=68.50ms ttft_p99=262.45ms itl_p50=24.88ms tps=3458.9 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.75ms ttft_p99=43.93ms itl_p50=12.63ms tps=267.7 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=68.49ms ttft_p99=233.95ms itl_p50=23.69ms tps=3583.1 |
