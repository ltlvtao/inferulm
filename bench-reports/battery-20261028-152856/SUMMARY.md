# battery 20261028-152856

| 层 | 指标（中位数, c=100, n=500, max_tokens=32） |
|---|---|
| L0 路由上限 | qps=168289 p50=0.53ms p99=1.61ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.56ms ttft_p99=12.21ms itl_p50=0.003ms itl_p99=9.50ms tps=82554 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.61ms itl_p50=0.00ms itl_p99=17.68ms tps=43623 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.18ms ttft_p99=4.98ms itl_mean=0.022ms tps=313140 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.72ms itl_mean=1.67ms tps=46105 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.05ms ttft_p99=2.22ms itl_mean=0.035ms tps=312638 |
