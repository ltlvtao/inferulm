# battery 20261107-203205

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=159235 p50=0.54ms p99=1.73ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.13ms ttft_p99=11.86ms itl_p50=0.003ms itl_p99=9.56ms tps=82302 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.75ms itl_p50=0.00ms itl_p99=17.70ms tps=43391 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.89ms ttft_p99=2.40ms itl_mean=0.488ms tps=161127 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.14ms itl_mean=1.66ms tps=45403 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.84ms ttft_p99=2.20ms itl_mean=0.393ms tps=161182 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.46ms ttft_p99=7.16ms itl_mean=0.45ms tps=144041 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.84ms ttft_p99=52.80ms itl_p50=12.22ms tps=265.5 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=49.01ms ttft_p99=216.05ms itl_p50=21.54ms tps=4209.8 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.60ms ttft_p99=43.71ms itl_p50=11.69ms tps=297.8 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=49.03ms ttft_p99=197.53ms itl_p50=20.52ms tps=4395.4 |
