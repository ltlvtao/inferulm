# battery 20261108-092156

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=157564 p50=0.52ms p99=1.66ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=11.02ms ttft_p99=12.60ms itl_p50=0.002ms itl_p99=9.61ms tps=70697 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.32ms itl_p50=0.00ms itl_p99=17.92ms tps=43427 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.77ms ttft_p99=2.77ms itl_mean=0.427ms tps=160944 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.37ms itl_mean=1.73ms tps=44310 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.88ms ttft_p99=2.10ms itl_mean=0.413ms tps=161214 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.55ms ttft_p99=7.64ms itl_mean=0.46ms tps=157533 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=47.27ms ttft_p99=55.01ms itl_p50=12.40ms tps=259.0 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=47.38ms ttft_p99=223.88ms itl_p50=20.93ms tps=4228.6 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.65ms ttft_p99=44.68ms itl_p50=11.03ms tps=318.8 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=50.15ms ttft_p99=187.34ms itl_p50=20.21ms tps=4430.8 |
