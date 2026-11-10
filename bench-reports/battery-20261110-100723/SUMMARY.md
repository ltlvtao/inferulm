# battery 20261110-100723

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=161008 p50=0.53ms p99=1.67ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.44ms ttft_p99=12.70ms itl_p50=0.003ms itl_p99=9.56ms tps=71705 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=20.65ms itl_p50=0.00ms itl_p99=17.46ms tps=43868 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.95ms ttft_p99=2.24ms itl_mean=0.441ms tps=161103 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.41ms itl_mean=1.70ms tps=44575 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=1.10ms ttft_p99=2.21ms itl_mean=0.589ms tps=143236 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.44ms ttft_p99=7.01ms itl_mean=0.53ms tps=128505 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.93ms ttft_p99=51.15ms itl_p50=12.12ms tps=275.5 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=62.14ms ttft_p99=165.88ms itl_p50=21.50ms tps=4133.9 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=42.05ms ttft_p99=43.65ms itl_p50=11.60ms tps=299.2 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=62.27ms ttft_p99=143.44ms itl_p50=20.71ms tps=4312.2 |
