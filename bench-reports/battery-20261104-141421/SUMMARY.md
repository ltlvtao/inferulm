# battery 20261104-141421

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=166073 p50=0.54ms p99=1.53ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.44ms ttft_p99=12.45ms itl_p50=0.002ms itl_p99=9.55ms tps=72330 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.64ms itl_p50=0.00ms itl_p99=17.66ms tps=43637 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.79ms ttft_p99=2.28ms itl_mean=0.470ms tps=155103 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.33ms itl_mean=1.66ms tps=46070 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.86ms ttft_p99=2.08ms itl_mean=0.534ms tps=154932 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.41ms ttft_p99=8.36ms itl_mean=0.49ms tps=123427 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.66ms ttft_p99=56.22ms itl_p50=13.15ms tps=248.8 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=71.75ms ttft_p99=259.52ms itl_p50=24.12ms tps=3470.5 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.60ms ttft_p99=44.09ms itl_p50=12.62ms tps=275.1 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=71.58ms ttft_p99=233.25ms itl_p50=23.16ms tps=3608.4 |
