# battery 20261103-183117

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=169620 p50=0.51ms p99=1.49ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.56ms ttft_p99=12.07ms itl_p50=0.003ms itl_p99=9.53ms tps=71437 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.65ms itl_p50=0.01ms itl_p99=17.69ms tps=43701 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.84ms ttft_p99=2.00ms itl_mean=0.520ms tps=155035 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.54ms itl_mean=1.69ms tps=44878 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.78ms ttft_p99=2.56ms itl_mean=0.404ms tps=167721 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.61ms ttft_p99=7.28ms itl_mean=0.46ms tps=161640 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.41ms ttft_p99=57.14ms itl_p50=13.68ms tps=241.5 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=70.22ms ttft_p99=265.07ms itl_p50=25.04ms tps=3416.7 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.90ms ttft_p99=43.43ms itl_p50=12.76ms tps=273.5 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=69.03ms ttft_p99=241.91ms itl_p50=23.89ms tps=3573.2 |
