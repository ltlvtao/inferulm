# battery 20261031-212716

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=138040 p50=0.63ms p99=1.85ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=11.50ms ttft_p99=12.71ms itl_p50=0.003ms itl_p99=10.12ms tps=65912 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.97ms itl_p50=0.00ms itl_p99=18.12ms tps=42974 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.20ms ttft_p99=3.01ms itl_mean=0.128ms tps=270779 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.15ms itl_mean=1.79ms tps=44239 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.17ms ttft_p99=2.76ms itl_mean=0.057ms tps=285377 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=11.11ms ttft_p99=13.59ms itl_mean=0.93ms tps=70857 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=74.03ms ttft_p99=120.97ms itl_p50=29.86ms tps=107.8 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=123.64ms ttft_p99=452.73ms itl_p50=44.95ms tps=1933.2 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=54.31ms ttft_p99=58.35ms itl_p50=22.75ms tps=150.6 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=150.20ms ttft_p99=401.76ms itl_p50=53.44ms tps=1667.1 |
