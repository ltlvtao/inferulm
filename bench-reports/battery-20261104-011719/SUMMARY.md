# battery 20261104-011719

| 层 | 指标（中位数; L0-L3/满负荷: c=100, n=500, max_tokens=32; 轻载 L3q: c=4, n=8, max_tokens=12） |
|---|---|
| L0 路由上限 | qps=153127 p50=0.57ms p99=1.86ms |
| L1 serve 纯开销(零延迟引擎) | ttft_p50=10.53ms ttft_p99=12.56ms itl_p50=0.002ms itl_p99=9.54ms tps=79156 |
| L1' 节拍泵(1ms/token) 参照 | ttft_p50=19.96ms itl_p50=0.00ms itl_p99=17.83ms tps=43513 |
| L2 engine core 开销(零延迟执行器) | ttft_p50=0.74ms ttft_p99=2.49ms itl_mean=0.382ms tps=168011 |
| L2' 节拍泵(1ms/步, 真调度器) | ttft_p50=18.81ms itl_mean=1.67ms tps=45206 |
| L2m 内存层记账开销(core+BlockPool/radix, 零延迟) | ttft_p50=0.71ms ttft_p99=2.15ms itl_mean=0.467ms tps=161075 |
| L3 exec 层开销(core+真执行器 toy, 真 GPU 步) | ttft_p50=5.69ms ttft_p99=6.95ms itl_mean=0.46ms tps=146529 |
| L3q 真模型竖切(Qwen3.5-4B 真权重, 轻载 c=4) | ttft_p50=43.82ms ttft_p99=57.85ms itl_p50=13.24ms tps=246.1 |
| L3q 满负荷(同 L0-L3 profile c=100) | ttft_p50=69.19ms ttft_p99=260.88ms itl_p50=24.80ms tps=3492.8 |
| L3q+ 图档(FULL_DECODE 桶图回放, 轻载 c=4) | ttft_p50=41.19ms ttft_p99=43.91ms itl_p50=12.60ms tps=275.2 |
| L3q+ 图档满负荷(桶{1,2,4,8,16,32,64,100,128}, c=100) | ttft_p50=69.22ms ttft_p99=235.68ms itl_p50=23.55ms tps=3627.2 |
