# battery 20261107-221722

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=168091 p50=0.52ms p99=1.55ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.76ms ttft_p99=12.05ms itl_p50=0.003ms itl_p99=9.53ms tps=77696 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.43ms itl_p50=0.00ms itl_p99=17.66ms tps=43970 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.78ms ttft_p99=2.58ms itl_mean=0.486ms tps=154860 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.90ms itl_mean=1.67ms tps=45638 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.79ms ttft_p99=1.94ms itl_mean=0.530ms tps=155150 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.37ms ttft_p99=6.36ms itl_mean=0.45ms tps=137286 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.81ms ttft_p99=54.02ms itl_p50=12.93ms tps=255.9 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=51.00ms ttft_p99=227.43ms itl_p50=21.91ms tps=4122.2 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=44.29ms ttft_p99=44.95ms itl_p50=11.67ms tps=300.1 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=49.82ms ttft_p99=194.57ms itl_p50=20.98ms tps=4324.2 |
