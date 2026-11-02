# battery 20261102-111208

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=165610 p50=0.53ms p99=1.55ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.50ms ttft_p99=12.17ms itl_p50=0.003ms itl_p99=9.58ms tps=79782 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.50ms itl_p50=0.00ms itl_p99=17.73ms tps=43655 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.42ms ttft_p99=3.48ms itl_mean=0.041ms tps=289476 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.73ms itl_mean=1.67ms tps=45619 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.23ms ttft_p99=2.78ms itl_mean=0.047ms tps=288765 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=4.98ms ttft_p99=7.07ms itl_mean=0.51ms tps=122154 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.59ms ttft_p99=58.96ms itl_p50=13.65ms tps=229.6 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=71.95ms ttft_p99=260.73ms itl_p50=25.14ms tps=3392.7 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.94ms ttft_p99=44.41ms itl_p50=12.30ms tps=266.1 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=72.18ms ttft_p99=238.23ms itl_p50=23.74ms tps=3550.8 |
