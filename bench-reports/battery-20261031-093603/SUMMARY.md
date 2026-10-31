# battery 20261031-093603

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=152002 p50=0.54ms p99=1.69ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.60ms ttft_p99=12.36ms itl_p50=0.002ms itl_p99=9.60ms tps=71249 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.45ms itl_p50=0.00ms itl_p99=17.69ms tps=43380 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.19ms ttft_p99=2.58ms itl_mean=0.047ms tps=312232 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.63ms itl_mean=1.66ms tps=45640 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.14ms ttft_p99=3.46ms itl_mean=0.095ms tps=291366 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.21ms ttft_p99=8.63ms itl_mean=0.43ms tps=138344 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=42.86ms ttft_p99=47.89ms itl_p50=15.78ms tps=206.0 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=69.34ms ttft_p99=270.02ms itl_p50=24.94ms tps=3430.0 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.52ms ttft_p99=43.72ms itl_p50=13.33ms tps=247.9 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=70.36ms ttft_p99=273.17ms itl_p50=23.21ms tps=3592.5 |
