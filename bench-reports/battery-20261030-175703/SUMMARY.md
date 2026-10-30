# battery 20261030-175703

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=162162 p50=0.54ms p99=1.68ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.76ms ttft_p99=12.19ms itl_p50=0.003ms itl_p99=9.64ms tps=80229 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.12ms itl_p50=0.00ms itl_p99=17.76ms tps=43768 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.16ms ttft_p99=2.89ms itl_mean=0.027ms tps=312037 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.44ms itl_mean=1.66ms tps=45747 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.19ms ttft_p99=2.61ms itl_mean=0.057ms tps=311289 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.06ms ttft_p99=6.58ms itl_mean=0.45ms tps=129912 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.62ms ttft_p99=45.86ms itl_p50=14.52ms tps=215.3 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=75.91ms ttft_p99=277.64ms itl_p50=26.17ms tps=3298.0 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.70ms ttft_p99=46.45ms itl_p50=12.58ms tps=256.7 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=72.95ms ttft_p99=277.74ms itl_p50=24.45ms tps=3461.0 |
