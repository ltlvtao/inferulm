# battery 20261031-213123

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=147077 p50=0.60ms p99=1.72ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=11.01ms ttft_p99=13.04ms itl_p50=0.006ms itl_p99=10.07ms tps=66886 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.68ms itl_p50=0.00ms itl_p99=18.57ms tps=42950 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.24ms ttft_p99=4.65ms itl_mean=0.053ms tps=289499 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.15ms itl_mean=1.82ms tps=39841 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.05ms ttft_p99=2.67ms itl_mean=0.114ms tps=288861 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=11.58ms ttft_p99=14.13ms itl_mean=0.95ms tps=69485 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=64.28ms ttft_p99=116.32ms itl_p50=28.17ms tps=117.9 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=134.85ms ttft_p99=512.83ms itl_p50=48.94ms tps=1757.1 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=58.46ms ttft_p99=64.85ms itl_p50=25.50ms tps=136.3 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=165.92ms ttft_p99=483.36ms itl_p50=52.70ms tps=1635.5 |
