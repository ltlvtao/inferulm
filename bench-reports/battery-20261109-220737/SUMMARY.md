# battery 20261109-220737

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=158698 p50=0.57ms p99=1.54ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.58ms ttft_p99=12.61ms itl_p50=0.003ms itl_p99=9.54ms tps=76388 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.29ms itl_p50=0.00ms itl_p99=17.73ms tps=43736 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.99ms ttft_p99=2.14ms itl_mean=0.551ms tps=149471 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.84ms itl_mean=1.72ms tps=45212 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.81ms ttft_p99=2.16ms itl_mean=0.526ms tps=161130 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.41ms ttft_p99=6.34ms itl_mean=0.45ms tps=134391 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.87ms ttft_p99=52.85ms itl_p50=12.14ms tps=271.7 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=62.87ms ttft_p99=168.28ms itl_p50=21.29ms tps=4134.0 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.44ms ttft_p99=43.80ms itl_p50=12.25ms tps=289.8 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=61.89ms ttft_p99=145.68ms itl_p50=20.24ms tps=4358.0 |
