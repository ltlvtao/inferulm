# battery 20261031-173650

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=165056 p50=0.52ms p99=1.77ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.69ms ttft_p99=12.51ms itl_p50=0.003ms itl_p99=9.59ms tps=72217 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.48ms itl_p50=0.00ms itl_p99=17.69ms tps=43577 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.16ms ttft_p99=4.93ms itl_mean=0.049ms tps=307862 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.69ms itl_mean=1.68ms tps=45324 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.14ms ttft_p99=3.03ms itl_mean=0.081ms tps=289547 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.09ms ttft_p99=7.35ms itl_mean=0.44ms tps=139025 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.90ms ttft_p99=59.01ms itl_p50=14.69ms tps=223.2 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=68.91ms ttft_p99=267.56ms itl_p50=25.19ms tps=3443.4 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.63ms ttft_p99=44.06ms itl_p50=13.56ms tps=253.7 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=72.67ms ttft_p99=240.22ms itl_p50=24.10ms tps=3547.2 |
