# battery 20261109-195103

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=163504 p50=0.53ms p99=1.62ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.40ms ttft_p99=12.26ms itl_p50=0.003ms itl_p99=9.60ms tps=80963 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.60ms itl_p50=0.00ms itl_p99=17.86ms tps=43541 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.88ms ttft_p99=2.98ms itl_mean=0.510ms tps=155252 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.95ms itl_mean=1.69ms tps=45337 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.71ms ttft_p99=2.06ms itl_mean=0.558ms tps=149329 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.49ms ttft_p99=8.65ms itl_mean=0.50ms tps=131019 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.90ms ttft_p99=54.18ms itl_p50=12.73ms tps=257.2 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=77.32ms ttft_p99=148.93ms itl_p50=21.66ms tps=4167.0 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.80ms ttft_p99=44.65ms itl_p50=11.10ms tps=305.1 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=69.58ms ttft_p99=125.23ms itl_p50=20.74ms tps=4369.8 |
