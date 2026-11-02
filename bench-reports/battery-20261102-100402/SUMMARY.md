# battery 20261102-100402

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=166620 p50=0.52ms p99=1.63ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.60ms ttft_p99=12.03ms itl_p50=0.002ms itl_p99=9.52ms tps=73018 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.61ms itl_p50=0.00ms itl_p99=17.75ms tps=44025 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.22ms ttft_p99=4.38ms itl_mean=0.061ms tps=290681 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.50ms itl_mean=1.67ms tps=45355 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.32ms ttft_p99=3.44ms itl_mean=0.068ms tps=288984 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=4.86ms ttft_p99=6.67ms itl_mean=0.55ms tps=115381 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.01ms ttft_p99=56.95ms itl_p50=14.58ms tps=226.5 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=71.13ms ttft_p99=264.54ms itl_p50=25.28ms tps=3434.8 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.22ms ttft_p99=43.84ms itl_p50=12.79ms tps=257.7 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=71.27ms ttft_p99=236.63ms itl_p50=23.98ms tps=3573.9 |
