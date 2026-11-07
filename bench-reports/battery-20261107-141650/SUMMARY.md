# battery 20261107-141650

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=157011 p50=0.53ms p99=1.77ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.56ms ttft_p99=12.34ms itl_p50=0.003ms itl_p99=9.48ms tps=82296 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.50ms itl_p50=0.00ms itl_p99=17.70ms tps=43476 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.85ms ttft_p99=2.59ms itl_mean=0.455ms tps=161783 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.99ms itl_mean=1.73ms tps=44454 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.82ms ttft_p99=2.26ms itl_mean=0.455ms tps=155122 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.68ms ttft_p99=7.34ms itl_mean=0.45ms tps=160795 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.83ms ttft_p99=51.30ms itl_p50=12.23ms tps=271.5 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=49.66ms ttft_p99=214.93ms itl_p50=21.32ms tps=4196.6 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.20ms ttft_p99=43.51ms itl_p50=11.59ms tps=298.3 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=50.51ms ttft_p99=197.66ms itl_p50=21.08ms tps=4288.9 |
