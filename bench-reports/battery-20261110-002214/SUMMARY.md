# battery 20261110-002214

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=167752 p50=0.53ms p99=1.47ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.73ms ttft_p99=12.37ms itl_p50=0.002ms itl_p99=9.55ms tps=82263 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.32ms itl_p50=0.00ms itl_p99=17.76ms tps=43793 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.94ms ttft_p99=2.21ms itl_mean=0.356ms tps=160203 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.18ms itl_mean=1.69ms tps=45704 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.83ms ttft_p99=2.61ms itl_mean=0.403ms tps=161823 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.45ms ttft_p99=6.53ms itl_mean=0.46ms tps=147190 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=47.90ms ttft_p99=54.05ms itl_p50=13.22ms tps=253.9 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=63.51ms ttft_p99=173.38ms itl_p50=21.89ms tps=4044.1 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=40.95ms ttft_p99=43.66ms itl_p50=12.11ms tps=287.5 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=63.29ms ttft_p99=147.19ms itl_p50=20.79ms tps=4255.2 |
