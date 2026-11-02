# battery 20261102-145232

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=170276 p50=0.51ms p99=1.53ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.65ms ttft_p99=12.14ms itl_p50=0.002ms itl_p99=9.60ms tps=79451 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.28ms itl_p50=0.00ms itl_p99=17.68ms tps=43549 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.90ms ttft_p99=2.30ms itl_mean=0.493ms tps=155310 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.07ms itl_mean=1.72ms tps=44610 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.83ms ttft_p99=2.55ms itl_mean=0.522ms tps=149451 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.86ms ttft_p99=7.30ms itl_mean=0.49ms tps=134665 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.81ms ttft_p99=56.71ms itl_p50=13.55ms tps=241.1 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=70.24ms ttft_p99=261.58ms itl_p50=25.06ms tps=3428.7 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.61ms ttft_p99=44.78ms itl_p50=12.35ms tps=278.8 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=70.98ms ttft_p99=234.97ms itl_p50=23.82ms tps=3565.4 |
