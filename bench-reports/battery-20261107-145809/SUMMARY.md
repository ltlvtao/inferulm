# battery 20261107-145809

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=162738 p50=0.52ms p99=2.05ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.19ms ttft_p99=12.26ms itl_p50=0.003ms itl_p99=9.54ms tps=81729 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.20ms itl_p50=0.01ms itl_p99=17.56ms tps=43816 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.84ms ttft_p99=2.34ms itl_mean=0.356ms tps=167912 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.96ms itl_mean=1.67ms tps=44181 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.73ms ttft_p99=2.50ms itl_mean=0.370ms tps=159940 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.77ms ttft_p99=7.78ms itl_mean=0.48ms tps=130705 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=42.59ms ttft_p99=72.93ms itl_p50=12.18ms tps=256.3 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=46.92ms ttft_p99=213.68ms itl_p50=20.80ms tps=4287.1 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.67ms ttft_p99=43.70ms itl_p50=11.12ms tps=317.0 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=49.57ms ttft_p99=197.79ms itl_p50=21.00ms tps=4325.9 |
