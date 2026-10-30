# battery 20261030-123906

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=158565 p50=0.55ms p99=1.75ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.84ms ttft_p99=12.23ms itl_p50=0.003ms itl_p99=9.57ms tps=79974 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.74ms itl_p50=0.00ms itl_p99=17.98ms tps=43374 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.11ms ttft_p99=3.12ms itl_mean=0.034ms tps=339392 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.72ms itl_mean=1.69ms tps=45890 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.23ms ttft_p99=3.02ms itl_mean=0.069ms tps=288428 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=8.96ms ttft_p99=13.53ms itl_mean=0.49ms tps=113824 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=143.40ms ttft_p99=202.09ms itl_p50=13.63ms tps=119.5 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=521.31ms ttft_p99=748.93ms itl_p50=26.57ms tps=1828.2 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=131.94ms ttft_p99=202.30ms itl_p50=12.62ms tps=153.1 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=521.00ms ttft_p99=877.55ms itl_p50=24.73ms tps=1893.4 |
