# battery 20261106-204146

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=162627 p50=0.54ms p99=1.55ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.45ms ttft_p99=11.68ms itl_p50=0.002ms itl_p99=9.59ms tps=71799 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.84ms itl_p50=0.00ms itl_p99=17.67ms tps=43498 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.77ms ttft_p99=2.20ms itl_mean=0.419ms tps=161451 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.96ms itl_mean=1.67ms tps=45446 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.76ms ttft_p99=2.37ms itl_mean=0.421ms tps=161356 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=6.29ms ttft_p99=7.79ms itl_mean=0.52ms tps=124452 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=48.26ms ttft_p99=51.36ms itl_p50=12.79ms tps=258.4 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=47.64ms ttft_p99=215.00ms itl_p50=20.95ms tps=4278.9 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=40.98ms ttft_p99=44.89ms itl_p50=11.88ms tps=297.1 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=47.86ms ttft_p99=200.13ms itl_p50=20.18ms tps=4399.2 |
