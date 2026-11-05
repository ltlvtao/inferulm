# battery 20261105-113633

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=163073 p50=0.54ms p99=1.53ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.75ms ttft_p99=12.94ms itl_p50=0.002ms itl_p99=9.56ms tps=81219 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.46ms itl_p50=0.01ms itl_p99=17.62ms tps=43519 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.84ms ttft_p99=2.35ms itl_mean=0.451ms tps=155305 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.48ms itl_mean=1.67ms tps=45575 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.81ms ttft_p99=2.22ms itl_mean=0.501ms tps=161137 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.55ms ttft_p99=7.27ms itl_mean=0.47ms tps=159850 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.47ms ttft_p99=56.71ms itl_p50=14.30ms tps=233.4 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=71.07ms ttft_p99=262.30ms itl_p50=24.54ms tps=3456.0 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.15ms ttft_p99=44.11ms itl_p50=13.04ms tps=273.2 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=69.46ms ttft_p99=235.14ms itl_p50=23.49ms tps=3593.5 |
