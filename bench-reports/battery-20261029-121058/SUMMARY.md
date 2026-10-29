# battery 20261029-121058

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=174069 p50=0.51ms p99=1.52ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.54ms ttft_p99=11.81ms itl_p50=0.002ms itl_p99=9.52ms tps=73095 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.62ms itl_p50=0.00ms itl_p99=17.67ms tps=43806 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.17ms ttft_p99=3.75ms itl_mean=0.035ms tps=289900 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.17ms itl_mean=1.69ms tps=45663 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.15ms ttft_p99=2.46ms itl_mean=0.085ms tps=290480 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=7.06ms ttft_p99=9.63ms itl_mean=0.79ms tps=91355 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=106.21ms ttft_p99=119.35ms itl_p50=32.98ms tps=96.4 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=804.08ms ttft_p99=1327.25ms itl_p50=141.61ms tps=574.1 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=104.35ms ttft_p99=110.41ms itl_p50=31.48ms tps=101.5 |
| L3q+ 图档满负荷(桶{1..128}, c=100) | ttft_p50=795.98ms ttft_p99=1309.73ms itl_p50=138.99ms tps=581.4 |
