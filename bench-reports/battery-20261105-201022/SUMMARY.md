# battery 20261105-201022

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=169609 p50=0.52ms p99=1.41ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.78ms ttft_p99=13.02ms itl_p50=0.003ms itl_p99=9.59ms tps=79741 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.28ms itl_p50=0.00ms itl_p99=17.71ms tps=43330 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.79ms ttft_p99=2.03ms itl_mean=0.437ms tps=155773 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.90ms itl_mean=1.77ms tps=40775 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.84ms ttft_p99=2.08ms itl_mean=0.491ms tps=149712 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.41ms ttft_p99=6.04ms itl_mean=0.46ms tps=161678 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.71ms ttft_p99=52.67ms itl_p50=13.77ms tps=244.2 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=70.73ms ttft_p99=258.08ms itl_p50=23.89ms tps=3525.8 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.65ms ttft_p99=44.53ms itl_p50=12.04ms tps=291.4 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=68.50ms ttft_p99=234.03ms itl_p50=22.68ms tps=3689.6 |
