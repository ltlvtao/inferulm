# battery 20261030-174539

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=158055 p50=0.54ms p99=1.93ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.81ms ttft_p99=12.31ms itl_p50=0.003ms itl_p99=9.61ms tps=80627 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.23ms itl_p50=0.00ms itl_p99=17.68ms tps=43736 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.10ms ttft_p99=5.29ms itl_mean=0.041ms tps=309566 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.25ms itl_mean=1.68ms tps=45238 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.00ms ttft_p99=3.49ms itl_mean=0.097ms tps=289904 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.02ms ttft_p99=6.10ms itl_mean=0.42ms tps=125730 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=42.03ms ttft_p99=44.40ms itl_p50=14.47ms tps=218.6 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=69.98ms ttft_p99=268.01ms itl_p50=24.67ms tps=3418.5 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=40.96ms ttft_p99=44.64ms itl_p50=12.91ms tps=256.9 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=74.44ms ttft_p99=268.17ms itl_p50=22.82ms tps=3603.7 |
