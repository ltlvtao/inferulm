# battery 20261031-213438

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=143135 p50=0.61ms p99=1.80ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=11.03ms ttft_p99=13.33ms itl_p50=0.003ms itl_p99=10.38ms tps=67020 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=21.52ms itl_p50=0.01ms itl_p99=17.89ms tps=42706 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.27ms ttft_p99=2.58ms itl_mean=0.104ms tps=291601 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.15ms itl_mean=1.73ms tps=44361 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.18ms ttft_p99=5.83ms itl_mean=0.070ms tps=286035 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=11.76ms ttft_p99=15.09ms itl_mean=1.03ms tps=60389 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=71.43ms ttft_p99=129.29ms itl_p50=31.19ms tps=105.3 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=158.93ms ttft_p99=450.26ms itl_p50=52.17ms tps=1673.0 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=62.52ms ttft_p99=64.87ms itl_p50=24.89ms tps=136.4 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=150.99ms ttft_p99=425.37ms itl_p50=50.99ms tps=1712.9 |
