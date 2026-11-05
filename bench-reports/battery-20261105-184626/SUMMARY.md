# battery 20261105-184626

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=165802 p50=0.53ms p99=1.57ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.89ms ttft_p99=12.01ms itl_p50=0.003ms itl_p99=9.58ms tps=71866 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.37ms itl_p50=0.00ms itl_p99=17.78ms tps=43686 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.71ms ttft_p99=2.18ms itl_mean=0.423ms tps=161177 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.00ms itl_mean=1.74ms tps=45645 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.90ms ttft_p99=2.32ms itl_mean=0.506ms tps=154785 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.88ms ttft_p99=7.25ms itl_mean=0.46ms tps=141203 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.32ms ttft_p99=55.75ms itl_p50=13.48ms tps=244.6 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=70.84ms ttft_p99=262.45ms itl_p50=25.08ms tps=3419.8 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.04ms ttft_p99=43.99ms itl_p50=12.67ms tps=276.5 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=71.74ms ttft_p99=239.35ms itl_p50=23.85ms tps=3577.8 |
