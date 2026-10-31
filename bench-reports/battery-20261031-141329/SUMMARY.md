# battery 20261031-141329

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=155261 p50=0.57ms p99=1.68ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.74ms ttft_p99=12.10ms itl_p50=0.003ms itl_p99=9.68ms tps=80667 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.94ms itl_p50=0.00ms itl_p99=17.88ms tps=43267 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.11ms ttft_p99=2.27ms itl_mean=0.040ms tps=289391 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.58ms itl_mean=1.71ms tps=45612 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.16ms ttft_p99=2.88ms itl_mean=0.087ms tps=271331 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.38ms ttft_p99=6.49ms itl_mean=0.42ms tps=125150 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.02ms ttft_p99=60.95ms itl_p50=13.80ms tps=225.0 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=68.79ms ttft_p99=263.54ms itl_p50=23.49ms tps=3563.9 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.80ms ttft_p99=44.15ms itl_p50=12.22ms tps=267.6 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=72.56ms ttft_p99=238.63ms itl_p50=22.54ms tps=3687.7 |
