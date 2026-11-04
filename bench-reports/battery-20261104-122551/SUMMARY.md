# battery 20261104-122551

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=161969 p50=0.55ms p99=1.67ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.79ms ttft_p99=12.10ms itl_p50=0.003ms itl_p99=9.58ms tps=72390 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.75ms itl_p50=0.00ms itl_p99=17.80ms tps=43630 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.89ms ttft_p99=2.90ms itl_mean=0.352ms tps=168118 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.35ms itl_mean=1.69ms tps=45345 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.84ms ttft_p99=2.91ms itl_mean=0.472ms tps=153895 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.45ms ttft_p99=7.32ms itl_mean=0.47ms tps=150485 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.12ms ttft_p99=79.06ms itl_p50=14.22ms tps=227.3 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=68.78ms ttft_p99=260.37ms itl_p50=24.50ms tps=3475.4 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.03ms ttft_p99=48.21ms itl_p50=12.32ms tps=280.6 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=68.57ms ttft_p99=235.85ms itl_p50=22.90ms tps=3685.6 |
