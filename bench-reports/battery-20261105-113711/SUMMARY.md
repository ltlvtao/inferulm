# battery 20261105-113711

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=166388 p50=0.52ms p99=1.63ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.82ms ttft_p99=11.93ms itl_p50=0.003ms itl_p99=9.55ms tps=80736 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.71ms itl_p50=0.00ms itl_p99=17.73ms tps=43697 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.80ms ttft_p99=2.14ms itl_mean=0.437ms tps=161212 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.09ms itl_mean=1.65ms tps=45743 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.81ms ttft_p99=2.05ms itl_mean=0.462ms tps=160876 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.35ms ttft_p99=6.58ms itl_mean=0.47ms tps=146577 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.82ms ttft_p99=54.73ms itl_p50=14.08ms tps=239.3 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=69.40ms ttft_p99=265.29ms itl_p50=24.79ms tps=3462.6 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.57ms ttft_p99=43.84ms itl_p50=12.60ms tps=272.7 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=69.09ms ttft_p99=240.69ms itl_p50=23.78ms tps=3586.9 |
