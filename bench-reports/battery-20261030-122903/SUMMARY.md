# battery 20261030-122903

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=164861 p50=0.54ms p99=1.52ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.89ms ttft_p99=12.68ms itl_p50=0.002ms itl_p99=9.61ms tps=72784 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.55ms itl_p50=0.00ms itl_p99=17.91ms tps=43765 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.18ms ttft_p99=2.95ms itl_mean=0.027ms tps=312565 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.90ms itl_mean=1.76ms tps=45318 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.19ms ttft_p99=4.65ms itl_mean=0.088ms tps=289877 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=9.89ms ttft_p99=17.98ms itl_mean=0.56ms tps=92072 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=157.97ms ttft_p99=225.15ms itl_p50=15.36ms tps=107.9 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=598.60ms ttft_p99=921.78ms itl_p50=29.35ms tps=1678.3 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=146.28ms ttft_p99=226.57ms itl_p50=13.44ms tps=140.0 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=573.17ms ttft_p99=831.65ms itl_p50=25.77ms tps=1929.0 |
