# battery 20261031-213015

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=147059 p50=0.60ms p99=1.98ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=11.40ms ttft_p99=13.03ms itl_p50=0.003ms itl_p99=10.07ms tps=76640 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.91ms itl_p50=0.00ms itl_p99=18.42ms tps=42421 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.39ms ttft_p99=3.33ms itl_mean=0.086ms tps=290654 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.72ms itl_mean=1.73ms tps=43906 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.25ms ttft_p99=3.71ms itl_mean=0.077ms tps=291055 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=10.12ms ttft_p99=11.53ms itl_mean=0.90ms tps=71907 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=64.52ms ttft_p99=114.48ms itl_p50=26.96ms tps=118.5 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=141.41ms ttft_p99=596.74ms itl_p50=50.06ms tps=1701.8 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=59.19ms ttft_p99=60.55ms itl_p50=24.96ms tps=140.8 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=150.47ms ttft_p99=404.64ms itl_p50=48.97ms tps=1785.3 |
