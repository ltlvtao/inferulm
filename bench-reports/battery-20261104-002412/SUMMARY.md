# battery 20261104-002412

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=161071 p50=0.53ms p99=1.89ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.53ms ttft_p99=12.38ms itl_p50=0.002ms itl_p99=9.52ms tps=71641 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.64ms itl_p50=0.01ms itl_p99=17.63ms tps=43331 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.93ms ttft_p99=2.22ms itl_mean=0.527ms tps=154978 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.15ms itl_mean=1.68ms tps=45337 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.73ms ttft_p99=2.32ms itl_mean=0.429ms tps=167927 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.38ms ttft_p99=6.26ms itl_mean=0.45ms tps=135681 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.54ms ttft_p99=80.71ms itl_p50=13.80ms tps=230.3 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=68.03ms ttft_p99=261.13ms itl_p50=24.94ms tps=3463.8 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.39ms ttft_p99=43.73ms itl_p50=12.38ms tps=281.0 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=68.25ms ttft_p99=239.07ms itl_p50=23.64ms tps=3616.4 |
