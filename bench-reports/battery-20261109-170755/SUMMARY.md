# battery 20261109-170755

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=172615 p50=0.51ms p99=1.48ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.57ms ttft_p99=12.31ms itl_p50=0.003ms itl_p99=9.52ms tps=79870 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.80ms itl_p50=0.01ms itl_p99=17.78ms tps=43180 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.76ms ttft_p99=2.30ms itl_mean=0.416ms tps=161358 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.70ms itl_mean=1.72ms tps=40599 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.87ms ttft_p99=2.98ms itl_mean=0.534ms tps=155780 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.72ms ttft_p99=7.41ms itl_mean=0.48ms tps=132439 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.93ms ttft_p99=54.40ms itl_p50=13.46ms tps=248.9 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=49.62ms ttft_p99=225.28ms itl_p50=22.05ms tps=4116.3 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.04ms ttft_p99=44.35ms itl_p50=11.59ms tps=297.0 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=50.12ms ttft_p99=199.41ms itl_p50=21.11ms tps=4297.5 |
