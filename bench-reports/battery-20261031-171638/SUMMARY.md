# battery 20261031-171638

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=164563 p50=0.51ms p99=1.64ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=11.17ms ttft_p99=12.52ms itl_p50=0.003ms itl_p99=9.47ms tps=71706 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.77ms itl_p50=0.00ms itl_p99=17.73ms tps=43430 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=1.06ms ttft_p99=3.20ms itl_mean=0.067ms tps=289999 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.26ms itl_mean=1.70ms tps=45426 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.19ms ttft_p99=4.22ms itl_mean=0.091ms tps=285948 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.33ms ttft_p99=6.65ms itl_mean=0.46ms tps=114857 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=44.78ms ttft_p99=58.64ms itl_p50=13.93ms tps=230.3 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=69.86ms ttft_p99=273.44ms itl_p50=23.91ms tps=3538.2 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=43.23ms ttft_p99=44.64ms itl_p50=12.24ms tps=267.2 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=68.76ms ttft_p99=236.93ms itl_p50=22.93ms tps=3654.1 |
