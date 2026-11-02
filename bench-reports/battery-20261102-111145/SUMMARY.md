# battery 20261102-111145

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=168487 p50=0.52ms p99=1.57ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.68ms ttft_p99=12.18ms itl_p50=0.002ms itl_p99=9.51ms tps=80851 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.77ms itl_p50=0.01ms itl_p99=17.77ms tps=43446 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.29ms ttft_p99=2.65ms itl_mean=0.111ms tps=289117 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.77ms itl_mean=1.68ms tps=45040 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.20ms ttft_p99=3.66ms itl_mean=0.104ms tps=289036 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.09ms ttft_p99=7.95ms itl_mean=0.43ms tps=125191 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.52ms ttft_p99=56.96ms itl_p50=13.93ms tps=233.4 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=69.58ms ttft_p99=260.73ms itl_p50=24.99ms tps=3440.2 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.94ms ttft_p99=44.64ms itl_p50=12.27ms tps=270.0 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=70.74ms ttft_p99=235.63ms itl_p50=23.69ms tps=3587.5 |
