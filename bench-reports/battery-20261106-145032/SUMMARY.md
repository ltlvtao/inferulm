# battery 20261106-145032

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=160049 p50=0.53ms p99=1.74ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.57ms ttft_p99=11.88ms itl_p50=0.002ms itl_p99=9.55ms tps=72664 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=21.04ms itl_p50=0.01ms itl_p99=17.67ms tps=43554 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.78ms ttft_p99=2.59ms itl_mean=0.457ms tps=161950 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.53ms itl_mean=1.70ms tps=44786 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.81ms ttft_p99=2.43ms itl_mean=0.396ms tps=167831 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.17ms ttft_p99=7.34ms itl_mean=0.46ms tps=141078 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.88ms ttft_p99=56.74ms itl_p50=14.91ms tps=232.7 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=71.92ms ttft_p99=290.52ms itl_p50=24.96ms tps=3430.1 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.36ms ttft_p99=44.02ms itl_p50=12.28ms tps=278.9 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=72.73ms ttft_p99=234.39ms itl_p50=23.73ms tps=3542.0 |
