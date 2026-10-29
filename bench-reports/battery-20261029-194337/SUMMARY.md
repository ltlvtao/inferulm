# battery 20261029-194337

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=158378 p50=0.52ms p99=1.68ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.52ms ttft_p99=12.35ms itl_p50=0.003ms itl_p99=9.59ms tps=72699 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.14ms itl_p50=0.00ms itl_p99=17.63ms tps=43609 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.09ms ttft_p99=2.67ms itl_mean=0.040ms tps=309837 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.37ms itl_mean=1.68ms tps=45371 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.24ms ttft_p99=4.67ms itl_mean=0.042ms tps=291910 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=6.42ms ttft_p99=9.37ms itl_mean=0.75ms tps=93860 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=82.93ms ttft_p99=90.93ms itl_p50=31.00ms tps=107.3 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=335.86ms ttft_p99=551.28ms itl_p50=89.67ms tps=985.7 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=79.14ms ttft_p99=88.08ms itl_p50=29.98ms tps=111.8 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=334.63ms ttft_p99=546.10ms itl_p50=87.61ms tps=1001.8 |
