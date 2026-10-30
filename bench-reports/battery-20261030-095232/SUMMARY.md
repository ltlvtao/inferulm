# battery 20261030-095232

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=170091 p50=0.51ms p99=1.66ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.15ms ttft_p99=11.83ms itl_p50=0.002ms itl_p99=9.50ms tps=72498 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.39ms itl_p50=0.00ms itl_p99=17.71ms tps=43563 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.14ms ttft_p99=2.64ms itl_mean=0.042ms tps=312258 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.69ms itl_mean=1.69ms tps=45660 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.15ms ttft_p99=2.80ms itl_mean=0.040ms tps=312237 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=6.21ms ttft_p99=12.15ms itl_mean=0.53ms tps=111962 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.28ms ttft_p99=44.98ms itl_p50=14.32ms tps=229.4 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=163.25ms ttft_p99=268.41ms itl_p50=34.64ms tps=2394.8 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.56ms ttft_p99=43.78ms itl_p50=12.84ms tps=268.9 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=161.55ms ttft_p99=267.88ms itl_p50=33.08ms tps=2492.2 |
