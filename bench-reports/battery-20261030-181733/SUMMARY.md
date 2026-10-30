# battery 20261030-181733

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=164739 p50=0.53ms p99=1.73ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.45ms ttft_p99=11.70ms itl_p50=0.002ms itl_p99=9.63ms tps=71908 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.87ms itl_p50=0.00ms itl_p99=17.70ms tps=43819 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.15ms ttft_p99=3.27ms itl_mean=0.024ms tps=334198 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.60ms itl_mean=1.67ms tps=45915 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.26ms ttft_p99=2.87ms itl_mean=0.035ms tps=323314 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=4.87ms ttft_p99=6.40ms itl_mean=0.44ms tps=125917 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=42.48ms ttft_p99=43.86ms itl_p50=14.03ms tps=218.7 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=67.85ms ttft_p99=266.49ms itl_p50=24.83ms tps=3447.3 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.54ms ttft_p99=44.21ms itl_p50=12.64ms tps=254.0 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=68.39ms ttft_p99=270.54ms itl_p50=23.08ms tps=3621.1 |
