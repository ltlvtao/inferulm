# battery 20261106-174854

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=162273 p50=0.52ms p99=1.51ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.79ms ttft_p99=12.37ms itl_p50=0.003ms itl_p99=9.56ms tps=80453 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.43ms itl_p50=0.00ms itl_p99=17.70ms tps=43619 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.99ms ttft_p99=3.42ms itl_mean=0.429ms tps=154864 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.42ms itl_mean=1.72ms tps=45646 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.99ms ttft_p99=2.16ms itl_mean=0.497ms tps=161049 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.66ms ttft_p99=7.61ms itl_mean=0.48ms tps=141620 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=47.80ms ttft_p99=55.41ms itl_p50=13.10ms tps=249.6 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=63.31ms ttft_p99=226.88ms itl_p50=22.33ms tps=3844.0 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.06ms ttft_p99=44.40ms itl_p50=12.25ms tps=281.8 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=61.81ms ttft_p99=205.09ms itl_p50=21.41ms tps=4025.4 |
