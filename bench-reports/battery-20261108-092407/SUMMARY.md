# battery 20261108-092407

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=169216 p50=0.52ms p99=1.42ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=11.17ms ttft_p99=12.66ms itl_p50=0.003ms itl_p99=9.62ms tps=70648 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.90ms itl_p50=0.00ms itl_p99=17.77ms tps=43018 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.90ms ttft_p99=2.73ms itl_mean=0.401ms tps=155096 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.35ms itl_mean=1.70ms tps=45019 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.82ms ttft_p99=2.16ms itl_mean=0.497ms tps=155486 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.49ms ttft_p99=6.57ms itl_mean=0.46ms tps=143682 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.55ms ttft_p99=75.68ms itl_p50=12.38ms tps=248.5 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=47.09ms ttft_p99=210.51ms itl_p50=20.73ms tps=4310.4 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.73ms ttft_p99=44.74ms itl_p50=11.19ms tps=311.7 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=49.18ms ttft_p99=193.37ms itl_p50=20.64ms tps=4379.6 |
