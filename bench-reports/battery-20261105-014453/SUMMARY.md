# battery 20261105-014453

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=169949 p50=0.51ms p99=1.56ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.67ms ttft_p99=12.71ms itl_p50=0.003ms itl_p99=9.54ms tps=71104 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.04ms itl_p50=0.00ms itl_p99=17.83ms tps=43611 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.84ms ttft_p99=2.38ms itl_mean=0.370ms tps=161511 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.34ms itl_mean=1.73ms tps=45500 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.74ms ttft_p99=2.28ms itl_mean=0.527ms tps=161112 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.78ms ttft_p99=6.84ms itl_mean=0.46ms tps=144840 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.62ms ttft_p99=55.39ms itl_p50=13.63ms tps=245.8 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=67.83ms ttft_p99=262.64ms itl_p50=24.68ms tps=3474.8 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.95ms ttft_p99=43.76ms itl_p50=12.40ms tps=281.7 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=67.87ms ttft_p99=233.87ms itl_p50=23.57ms tps=3629.1 |
