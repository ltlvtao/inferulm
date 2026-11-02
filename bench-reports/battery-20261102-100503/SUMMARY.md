# battery 20261102-100503

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=159145 p50=0.54ms p99=1.73ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=11.19ms ttft_p99=12.66ms itl_p50=0.003ms itl_p99=9.59ms tps=70998 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.72ms itl_p50=0.00ms itl_p99=17.56ms tps=43773 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.14ms ttft_p99=3.41ms itl_mean=0.070ms tps=289412 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.74ms itl_mean=1.68ms tps=45596 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.43ms ttft_p99=3.84ms itl_mean=0.063ms tps=289592 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.20ms ttft_p99=7.10ms itl_mean=0.51ms tps=108900 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.90ms ttft_p99=58.18ms itl_p50=14.46ms tps=226.6 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=68.17ms ttft_p99=260.47ms itl_p50=25.37ms tps=3450.8 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.61ms ttft_p99=43.83ms itl_p50=12.77ms tps=265.8 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=69.09ms ttft_p99=233.20ms itl_p50=23.93ms tps=3595.0 |
