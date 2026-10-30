# battery 20261030-124320

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=164297 p50=0.53ms p99=1.56ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.76ms ttft_p99=12.43ms itl_p50=0.002ms itl_p99=9.52ms tps=71092 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.74ms itl_p50=0.00ms itl_p99=17.69ms tps=43816 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.22ms ttft_p99=3.61ms itl_mean=0.032ms tps=312279 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.57ms itl_mean=1.67ms tps=45656 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.07ms ttft_p99=3.27ms itl_mean=0.030ms tps=308011 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.70ms ttft_p99=6.98ms itl_mean=0.55ms tps=114323 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=41.60ms ttft_p99=43.68ms itl_p50=14.04ms tps=227.3 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=160.85ms ttft_p99=208.94ms itl_p50=27.26ms tps=3022.2 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.44ms ttft_p99=43.70ms itl_p50=12.61ms tps=259.3 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=127.15ms ttft_p99=219.90ms itl_p50=25.84ms tps=3182.7 |
