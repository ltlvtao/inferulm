# battery 20261028-144250

| 层 | 指标（中位数, c=100, n=500, max_tokens=32） |
|---|---|
| L0 路由上限 | qps=170415 p50=0.52ms p99=1.59ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.84ms ttft_p99=12.25ms itl_p50=0.002ms itl_p99=9.56ms tps=80763 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.43ms itl_p50=0.00ms itl_p99=17.72ms tps=43762 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.11ms ttft_p99=2.68ms itl_mean=0.027ms tps=309822 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.99ms itl_mean=1.69ms tps=45312 |
