# battery 20261102-120534

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=164216 p50=0.54ms p99=1.58ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.56ms ttft_p99=12.56ms itl_p50=0.002ms itl_p99=9.49ms tps=81194 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.55ms itl_p50=0.00ms itl_p99=17.64ms tps=43672 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.11ms ttft_p99=3.55ms itl_mean=0.097ms tps=289297 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.72ms itl_mean=1.68ms tps=45560 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.18ms ttft_p99=4.08ms itl_mean=0.086ms tps=287795 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=4.87ms ttft_p99=6.98ms itl_mean=0.41ms tps=125677 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.59ms ttft_p99=59.71ms itl_p50=13.97ms tps=230.9 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=68.78ms ttft_p99=260.38ms itl_p50=24.96ms tps=3460.8 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=40.72ms ttft_p99=43.77ms itl_p50=12.16ms tps=268.7 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=68.84ms ttft_p99=231.42ms itl_p50=23.54ms tps=3611.1 |
