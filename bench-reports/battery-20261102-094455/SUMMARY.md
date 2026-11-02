# battery 20261102-094455

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=174002 p50=0.51ms p99=1.35ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.88ms ttft_p99=12.53ms itl_p50=0.002ms itl_p99=9.55ms tps=71561 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.40ms itl_p50=0.00ms itl_p99=17.66ms tps=43525 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.14ms ttft_p99=3.19ms itl_mean=0.120ms tps=288945 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.08ms itl_mean=1.67ms tps=45506 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.15ms ttft_p99=4.38ms itl_mean=0.127ms tps=312384 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=4.94ms ttft_p99=7.33ms itl_mean=0.54ms tps=115897 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=47.92ms ttft_p99=56.80ms itl_p50=14.58ms tps=223.9 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=70.90ms ttft_p99=262.02ms itl_p50=24.78ms tps=3443.0 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.56ms ttft_p99=43.81ms itl_p50=12.42ms tps=266.6 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=70.10ms ttft_p99=236.80ms itl_p50=23.61ms tps=3580.0 |
