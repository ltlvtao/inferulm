# battery 20261102-145144

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=167890 p50=0.52ms p99=1.61ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.54ms ttft_p99=12.73ms itl_p50=0.002ms itl_p99=9.44ms tps=71689 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.61ms itl_p50=0.00ms itl_p99=17.75ms tps=43659 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.90ms ttft_p99=2.35ms itl_mean=0.418ms tps=161373 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.23ms itl_mean=1.68ms tps=44553 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.81ms ttft_p99=2.94ms itl_mean=0.416ms tps=160871 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.63ms ttft_p99=7.18ms itl_mean=0.47ms tps=133612 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.81ms ttft_p99=57.50ms itl_p50=13.55ms tps=240.9 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=68.63ms ttft_p99=261.56ms itl_p50=24.98ms tps=3445.5 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.07ms ttft_p99=44.16ms itl_p50=12.21ms tps=281.7 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=67.98ms ttft_p99=233.83ms itl_p50=23.86ms tps=3598.4 |
