# battery 20261106-093622

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=164573 p50=0.54ms p99=1.64ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.64ms ttft_p99=12.78ms itl_p50=0.003ms itl_p99=9.60ms tps=80523 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.02ms itl_p50=0.00ms itl_p99=17.76ms tps=43781 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.90ms ttft_p99=2.59ms itl_mean=0.500ms tps=160000 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.53ms itl_mean=1.66ms tps=45913 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.81ms ttft_p99=2.33ms itl_mean=0.423ms tps=161685 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.45ms ttft_p99=6.56ms itl_mean=0.49ms tps=139711 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.54ms ttft_p99=57.17ms itl_p50=14.04ms tps=238.8 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=69.62ms ttft_p99=262.07ms itl_p50=24.72ms tps=3447.4 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.09ms ttft_p99=43.60ms itl_p50=13.29ms tps=270.2 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=70.09ms ttft_p99=240.76ms itl_p50=23.67ms tps=3578.4 |
