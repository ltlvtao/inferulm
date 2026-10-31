# battery 20261031-141300

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=164255 p50=0.53ms p99=1.67ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.63ms ttft_p99=11.93ms itl_p50=0.003ms itl_p99=9.68ms tps=79586 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.40ms itl_p50=0.01ms itl_p99=17.80ms tps=43827 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.14ms ttft_p99=3.07ms itl_mean=0.054ms tps=289978 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.71ms itl_mean=1.67ms tps=45259 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.14ms ttft_p99=2.63ms itl_mean=0.095ms tps=289447 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.88ms ttft_p99=7.31ms itl_mean=0.48ms tps=111768 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.81ms ttft_p99=62.52ms itl_p50=14.02ms tps=225.7 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=73.00ms ttft_p99=264.34ms itl_p50=23.67ms tps=3549.2 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.31ms ttft_p99=46.69ms itl_p50=12.31ms tps=261.2 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=73.08ms ttft_p99=236.56ms itl_p50=22.55ms tps=3682.4 |
