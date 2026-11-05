# battery 20261105-123258

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=173172 p50=0.50ms p99=1.49ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=11.12ms ttft_p99=12.24ms itl_p50=0.003ms itl_p99=9.57ms tps=80292 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.08ms itl_p50=0.00ms itl_p99=17.75ms tps=44030 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.91ms ttft_p99=2.09ms itl_mean=0.490ms tps=161121 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.03ms itl_mean=1.66ms tps=45496 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.81ms ttft_p99=2.35ms itl_mean=0.536ms tps=154924 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.60ms ttft_p99=7.33ms itl_mean=0.46ms tps=156683 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.83ms ttft_p99=55.90ms itl_p50=13.43ms tps=242.9 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=69.39ms ttft_p99=261.90ms itl_p50=24.63ms tps=3467.7 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.21ms ttft_p99=44.57ms itl_p50=12.01ms tps=286.2 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=68.33ms ttft_p99=236.05ms itl_p50=23.42ms tps=3622.5 |
