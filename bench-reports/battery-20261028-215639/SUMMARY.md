# battery 20261028-215639

| 层 | 指标（中位数, c=100, n=500, max_tokens=32） |
|---|---|
| L0 路由上限 | qps=168890 p50=0.52ms p99=1.66ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.20ms ttft_p99=12.31ms itl_p50=0.003ms itl_p99=9.52ms tps=70763 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.26ms itl_p50=0.00ms itl_p99=17.74ms tps=43907 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.15ms ttft_p99=3.08ms itl_mean=0.041ms tps=311056 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.82ms itl_mean=1.72ms tps=28397 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.06ms ttft_p99=3.44ms itl_mean=0.043ms tps=290849 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=6.76ms ttft_p99=14.27ms itl_mean=0.80ms tps=47111 |
