# battery 20261102-094431

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=170215 p50=0.50ms p99=1.86ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.75ms ttft_p99=12.15ms itl_p50=0.003ms itl_p99=9.58ms tps=71288 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.85ms itl_p50=0.01ms itl_p99=17.85ms tps=43338 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.26ms ttft_p99=2.97ms itl_mean=0.027ms tps=338631 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.98ms itl_mean=1.69ms tps=45200 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.18ms ttft_p99=2.88ms itl_mean=0.073ms tps=269981 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.00ms ttft_p99=8.19ms itl_mean=0.40ms tps=138797 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.93ms ttft_p99=56.50ms itl_p50=14.08ms tps=228.7 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=71.70ms ttft_p99=263.92ms itl_p50=24.83ms tps=3427.3 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.65ms ttft_p99=43.76ms itl_p50=12.70ms tps=267.1 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=69.84ms ttft_p99=233.09ms itl_p50=23.73ms tps=3571.5 |
