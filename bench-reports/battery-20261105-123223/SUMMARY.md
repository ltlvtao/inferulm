# battery 20261105-123223

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=162231 p50=0.53ms p99=1.57ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.78ms ttft_p99=12.24ms itl_p50=0.003ms itl_p99=9.61ms tps=79720 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.03ms itl_p50=0.00ms itl_p99=17.86ms tps=43346 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.79ms ttft_p99=2.46ms itl_mean=0.510ms tps=155339 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.70ms itl_mean=1.79ms tps=45000 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.82ms ttft_p99=2.31ms itl_mean=0.427ms tps=161339 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.59ms ttft_p99=7.14ms itl_mean=0.46ms tps=140703 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.47ms ttft_p99=56.75ms itl_p50=13.09ms tps=247.0 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=70.97ms ttft_p99=258.42ms itl_p50=24.69ms tps=3452.5 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.90ms ttft_p99=43.87ms itl_p50=12.24ms tps=284.9 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=70.06ms ttft_p99=233.83ms itl_p50=23.66ms tps=3593.6 |
