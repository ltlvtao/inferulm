# battery 20261106-162247

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=162298 p50=0.51ms p99=1.63ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.69ms ttft_p99=12.14ms itl_p50=0.003ms itl_p99=9.55ms tps=79027 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.74ms itl_p50=0.00ms itl_p99=17.73ms tps=43708 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.86ms ttft_p99=2.71ms itl_mean=0.414ms tps=161135 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.04ms itl_mean=1.75ms tps=45289 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.81ms ttft_p99=2.84ms itl_mean=0.527ms tps=154693 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=6.60ms ttft_p99=8.10ms itl_mean=0.53ms tps=127825 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.97ms ttft_p99=52.79ms itl_p50=13.32ms tps=250.3 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=59.23ms ttft_p99=224.85ms itl_p50=21.66ms tps=3937.7 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.12ms ttft_p99=43.69ms itl_p50=12.23ms tps=286.2 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=61.01ms ttft_p99=194.50ms itl_p50=20.69ms tps=4085.6 |
