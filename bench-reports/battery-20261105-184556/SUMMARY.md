# battery 20261105-184556

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=154071 p50=0.56ms p99=1.80ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.68ms ttft_p99=12.27ms itl_p50=0.002ms itl_p99=9.67ms tps=73148 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.61ms itl_p50=0.00ms itl_p99=17.75ms tps=43750 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.77ms ttft_p99=2.37ms itl_mean=0.389ms tps=168036 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.57ms itl_mean=1.67ms tps=45128 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.86ms ttft_p99=2.32ms itl_mean=0.395ms tps=159858 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=6.04ms ttft_p99=7.19ms itl_mean=0.48ms tps=125585 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.47ms ttft_p99=55.37ms itl_p50=13.55ms tps=246.0 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=67.92ms ttft_p99=260.64ms itl_p50=24.51ms tps=3475.0 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.44ms ttft_p99=43.79ms itl_p50=12.23ms tps=282.8 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=68.75ms ttft_p99=233.68ms itl_p50=23.51ms tps=3607.0 |
