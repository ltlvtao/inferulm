# battery 20261106-122319

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=164235 p50=0.53ms p99=1.74ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.62ms ttft_p99=12.04ms itl_p50=0.002ms itl_p99=9.57ms tps=80934 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.50ms itl_p50=0.00ms itl_p99=17.63ms tps=44113 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.85ms ttft_p99=2.24ms itl_mean=0.429ms tps=155283 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.78ms itl_mean=1.74ms tps=40901 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.84ms ttft_p99=2.28ms itl_mean=0.382ms tps=161428 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.72ms ttft_p99=7.05ms itl_mean=0.46ms tps=142255 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.98ms ttft_p99=52.85ms itl_p50=13.97ms tps=242.3 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=71.49ms ttft_p99=264.59ms itl_p50=24.88ms tps=3452.7 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.04ms ttft_p99=43.81ms itl_p50=12.94ms tps=274.0 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=70.14ms ttft_p99=238.43ms itl_p50=23.90ms tps=3586.5 |
