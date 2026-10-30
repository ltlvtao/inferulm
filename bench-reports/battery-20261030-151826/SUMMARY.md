# battery 20261030-151826

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=171503 p50=0.51ms p99=1.60ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.42ms ttft_p99=12.40ms itl_p50=0.002ms itl_p99=9.63ms tps=70301 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.33ms itl_p50=0.00ms itl_p99=17.66ms tps=43566 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.52ms ttft_p99=3.84ms itl_mean=0.034ms tps=289771 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.50ms itl_mean=1.67ms tps=45972 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.17ms ttft_p99=3.58ms itl_mean=0.064ms tps=311122 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.03ms ttft_p99=7.33ms itl_mean=0.47ms tps=122082 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.80ms ttft_p99=47.83ms itl_p50=14.37ms tps=219.7 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=141.41ms ttft_p99=185.02ms itl_p50=24.38ms tps=3421.5 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.79ms ttft_p99=44.72ms itl_p50=12.23ms tps=258.3 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=109.63ms ttft_p99=183.65ms itl_p50=22.68ms tps=3670.6 |
