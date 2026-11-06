# battery 20261106-201016

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=158577 p50=0.54ms p99=1.89ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.60ms ttft_p99=12.12ms itl_p50=0.002ms itl_p99=9.56ms tps=69873 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.27ms itl_p50=0.00ms itl_p99=17.64ms tps=43819 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.84ms ttft_p99=2.24ms itl_mean=0.474ms tps=159995 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.65ms itl_mean=1.69ms tps=44576 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.81ms ttft_p99=2.36ms itl_mean=0.406ms tps=168170 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.89ms ttft_p99=7.13ms itl_mean=0.49ms tps=144282 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.91ms ttft_p99=54.72ms itl_p50=12.51ms tps=259.9 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=47.48ms ttft_p99=214.29ms itl_p50=21.04ms tps=4242.6 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.64ms ttft_p99=43.85ms itl_p50=11.41ms tps=306.8 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=49.11ms ttft_p99=197.68ms itl_p50=20.50ms tps=4351.0 |
