# battery 20261107-135006

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=168730 p50=0.53ms p99=1.54ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.80ms ttft_p99=12.07ms itl_p50=0.002ms itl_p99=9.51ms tps=81618 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.65ms itl_p50=0.00ms itl_p99=17.75ms tps=43734 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.83ms ttft_p99=2.23ms itl_mean=0.515ms tps=155325 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=19.06ms itl_mean=1.68ms tps=45471 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.84ms ttft_p99=2.20ms itl_mean=0.409ms tps=161865 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.37ms ttft_p99=6.58ms itl_mean=0.43ms tps=151240 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.02ms ttft_p99=55.30ms itl_p50=12.27ms tps=266.6 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=48.46ms ttft_p99=219.47ms itl_p50=21.32ms tps=4200.9 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.43ms ttft_p99=44.53ms itl_p50=11.39ms tps=297.9 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=51.18ms ttft_p99=198.90ms itl_p50=21.38ms tps=4244.8 |
