# battery 20261106-181754

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=168735 p50=0.51ms p99=1.46ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.33ms ttft_p99=12.04ms itl_p50=0.002ms itl_p99=9.61ms tps=70472 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=19.89ms itl_p50=0.01ms itl_p99=17.79ms tps=43723 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.80ms ttft_p99=2.20ms itl_mean=0.472ms tps=161308 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.50ms itl_mean=1.69ms tps=45272 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.79ms ttft_p99=2.78ms itl_mean=0.575ms tps=149275 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.69ms ttft_p99=6.95ms itl_mean=0.46ms tps=143131 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=48.31ms ttft_p99=55.21ms itl_p50=13.06ms tps=254.3 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=60.89ms ttft_p99=225.87ms itl_p50=22.42ms tps=3847.7 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.72ms ttft_p99=44.75ms itl_p50=11.63ms tps=295.0 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=60.90ms ttft_p99=204.03ms itl_p50=21.42ms tps=4015.8 |
