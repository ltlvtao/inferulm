# battery 20261030-112009

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=160600 p50=0.54ms p99=1.89ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.53ms ttft_p99=11.85ms itl_p50=0.002ms itl_p99=9.60ms tps=70343 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=21.05ms itl_p50=0.00ms itl_p99=17.76ms tps=43727 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.24ms ttft_p99=2.98ms itl_mean=0.034ms tps=307017 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.43ms itl_mean=1.68ms tps=45524 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.24ms ttft_p99=3.67ms itl_mean=0.063ms tps=289381 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.75ms ttft_p99=8.13ms itl_mean=0.58ms tps=102785 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=48.43ms ttft_p99=52.23ms itl_p50=15.71ms tps=211.1 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=148.32ms ttft_p99=232.33ms itl_p50=29.41ms tps=2827.0 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=40.96ms ttft_p99=45.22ms itl_p50=14.31ms tps=233.7 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=146.02ms ttft_p99=233.85ms itl_p50=27.48ms tps=2982.9 |
