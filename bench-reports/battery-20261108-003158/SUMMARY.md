# battery 20261108-003158

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=166071 p50=0.53ms p99=1.58ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.57ms ttft_p99=12.39ms itl_p50=0.002ms itl_p99=9.56ms tps=80125 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.43ms itl_p50=0.00ms itl_p99=17.72ms tps=43790 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.81ms ttft_p99=1.96ms itl_mean=0.508ms tps=161103 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.08ms itl_mean=1.71ms tps=40618 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.78ms ttft_p99=2.49ms itl_mean=0.449ms tps=161029 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.58ms ttft_p99=6.83ms itl_mean=0.47ms tps=140796 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.59ms ttft_p99=55.23ms itl_p50=12.39ms tps=260.0 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=50.12ms ttft_p99=247.23ms itl_p50=21.47ms tps=4180.7 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.25ms ttft_p99=44.52ms itl_p50=11.21ms tps=308.5 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=46.71ms ttft_p99=188.85ms itl_p50=19.74ms tps=4526.9 |
