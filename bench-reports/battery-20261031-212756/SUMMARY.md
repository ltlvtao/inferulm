# battery 20261031-212756

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=147434 p50=0.60ms p99=1.69ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=11.00ms ttft_p99=13.51ms itl_p50=0.003ms itl_p99=10.03ms tps=69252 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.55ms itl_p50=0.00ms itl_p99=18.44ms tps=42629 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.20ms ttft_p99=2.85ms itl_mean=0.094ms tps=289297 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.71ms itl_mean=1.70ms tps=44739 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.21ms ttft_p99=2.96ms itl_mean=0.096ms tps=271375 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=10.84ms ttft_p99=14.96ms itl_mean=0.97ms tps=68761 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=74.74ms ttft_p99=128.73ms itl_p50=30.82ms tps=104.9 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=133.69ms ttft_p99=574.33ms itl_p50=49.54ms tps=1742.4 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=62.86ms ttft_p99=63.86ms itl_p50=25.52ms tps=132.4 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=148.59ms ttft_p99=485.86ms itl_p50=50.04ms tps=1689.1 |
