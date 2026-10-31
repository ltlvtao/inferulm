# battery 20261031-172516

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=162641 p50=0.53ms p99=1.82ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.72ms ttft_p99=12.24ms itl_p50=0.002ms itl_p99=9.55ms tps=70533 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.55ms itl_p50=0.00ms itl_p99=17.73ms tps=43395 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.28ms ttft_p99=3.01ms itl_mean=0.047ms tps=290198 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.76ms itl_mean=1.72ms tps=40985 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.10ms ttft_p99=2.95ms itl_mean=0.086ms tps=290375 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.03ms ttft_p99=6.56ms itl_mean=0.50ms tps=129835 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=45.59ms ttft_p99=61.47ms itl_p50=14.01ms tps=226.6 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=72.22ms ttft_p99=294.59ms itl_p50=25.08ms tps=3381.3 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.55ms ttft_p99=44.32ms itl_p50=12.40ms tps=259.5 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=69.52ms ttft_p99=239.28ms itl_p50=23.99ms tps=3564.4 |
