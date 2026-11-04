# attention v5 刀账（2026-11-04，commit 7968e67 + cp.async 后续）

四步计划收官：①4 warp×64 行 CTA 共享 K/V smem ②双缓冲 ③cp.async ④32K 验证。
（步骤①②合并 commit 7968e67；③④见 HEAD 链。）

## 内核形态（v5）

- 块 = 128 线程（4 warp × 16 行，同 q 头）；grid = 16 qh × ⌈t/64⌉。
- K 平铺 staging 进 smem（wmma load.b col-major 视图 = row-major K，零转置），
  **cp.async.cg 16B ×4 块/线程**绕寄存器直达 smem；V^T 转置 staging 仍 ld+st
  （cp.async 做不了转置；改 raw+smem 转置需 +16KB smem 顶占用红线，弃）。
- K/V 双缓冲：tile t 计算与 t+1 staging 重叠，唯一 CTA 屏障在迭代头
  （wait_all + syncthreads）。
- **位级合同 v5 ≡ v4 逐位**（o39 六档钉死，含 cp.async 改造后 0 位差）。
- ptxas：254 regs / 0 spill / 34816B smem / 1 barrier → 2 CTA/SM。
- cp.async 旁路：intrinsic 死路（ptr addrspace(3)/(1) 表达不了）→ `#[no_mangle]`
  占位函数（volatile 体防 IPO）+ build.rs `patch_cp_async` 文本替换（补丁失败
  即 build panic）。**踩坑**：16B 块寻址必须按 token 跨步（he<<1/token），首版
  线性寻址读成交错 kvh 数据（位差 2M，双跑却确定——"确定性≠正确性"）。

## 读数

| 指标 | v4/旧 | v5（+cp.async） | 倍率 |
|---|---|---|---|
| 单发 6009 attn（ms/层） | 12.82 | 7.44 | 1.72× |
| 单发 8009 attn | 22.00 | 12.44 | 1.77× |
| 单发 32000 attn（8 层合计） | 520.8 | 210.8 | 2.47× |
| 单发 3009 attn | 3.91 | 2.54 | 1.54× |
| 6009 e2e attn 段（chunk2048） | 49.1 | 38.3 ms/步 | −22% |
| 6009 e2e TTFT（同进程对） | 720.9 | 673.4 ms | −6.6% |
| 32K e2e attn 段（chunk2048 冷） | 248.0 | 139.5 ms/步 | −44% |
| 32K e2e 冷 TTFT | 6380（旧基线 6444） | **4587 ms** | −28% |
| 32K 热（radix 命中 30720） | 523 | 515 ms | 持平 |
| 32K decode ITL p50 | 18.6 | 18.5 ms | —（decode 不走 v5） |

- cp.async 单项：6009 8.21→7.44（−9%）、32000 252.6→210.8（−16%）。
- 正确性：6009/32K 贪心 V5=0/1 **逐位同**（md5 相等）；32K 三轮
  ①②③全同（radix 恢复 + 分段粒度）；o39 六档 0 位差；kernel 76 测试绿。
- battery 带内：图档满载 3608.4 tps/ITL 23.16/p99 37.8（基线 3633/23.39/35.2），
  轻载 12.62 持平，eager 3470 微升——v5 不触 decode 路径，符合预期。

## 与 vLLM 对照（三方）

| 口径 | 上一轮（v4） | 本轮（v5+cp.async） | vLLM |
|---|---|---|---|
| 6009 prefill attn（8 层） | 104.7 ms | **59.5 ms** | 19.27 ms |
| 6009 冷 TTFT | 602（服务端） | 673→543（进程内对） | 383–418 |
| 32K prefill 吞吐 | ~5.0k tok/s | **7.0k tok/s** | ~19–21k |
| 满载 tps / ITL（不受本刀影响） | 3633/23.39 | 3608/23.16 | 4091/20.06 |

- attn 对 vLLM 差距 5.43× → **3.1×**。剩余差距解剖：22% tensor 利用率
  （纯延迟受限——Bc=16 tile 每屏障 mma 太少、375 tile/6009 的 per-tile 开销
  主导），**Bc=64 大瓦片（容差档，破位级合同）是追平的最后一段**，留刀。
- 32K 冷账本（chunk2048/16 步）：attn 139.5（49.3%）> gemm 93.7（33.1%）>
  delta 32.9（11.6%）——gemm 与 vLLM 同墙（162 TFLOPS 算力墙），vLLM 全速
  prefill 还靠 attention 与 lm_head 组织，非 gemm 单价。

## 交接

- 门控：`WEIN_ATTN_V5`（默认开）、`WEIN_ATTN_V5_T`（默认 2048）、
  `WEIN_ATTN_V5_CTX`（默认 1<<20 无上限）；t≥2048 且纯 prefill 批接管，
  含 decode 行禁用（b2 教训）；tokens%64==0 由 chunk 对齐保证。
- 探针：`examples/attn_v5_probe`（位级+计时）、`qwen_longctx_probe 32000 32`
  配 `WEIN_CHUNK=2048 WEIN_CTX_LEN=32768 WEIN_KV_PAGES=2048 WEIN_SNAP_STRIDE=256`。
- 留刀顺位：Bc=64 大瓦片（容差档 o40 式）> delta h 换形态 > BT 自适应 >
  W8A8（对手 +24%）> tools > 昇腾。

## 追记：大瓦片刀证伪（2026-11-04，未 commit 即删）

四步收官后开"追平 vLLM 最后一段"刀。四个变体全部落在 v5 ±5% 带内，
**结构性假说全灭**；根因定位后关刀，死代码（v6/v7/v8/v7_ablate）当场删净。

### 变体与读数（6009 ms/单层，v5=7.0-7.4）

| 变体 | 假说 | 读数 | 判决 |
|---|---|---|---|
| v6 Bc=32 单缓冲 | 屏障频率/mma 链长是墙 | 7.1-7.3 | 打平（意外保位级 ≡v5） |
| v7 Br=128（8 warp/CTA） | CTA 粒度/Q 流量 18GB/层 是墙 | 6.9-7.1 | 打平（Q 重读被 L2 吸收；位级 ≡v5） |
| v8 S 四路 k-劈链 | 单依赖链停顿（独立测 −45%） | 7.1 | 全量打平——softmax/O 在场收益蒸发 |
| v7_ablate 消融相位 0-10 | 定位 | 见下 | S 段独占 60% |

### S 相位判决性分解（@6009，消融 kernel 实测）

| 项 | ms | 占 S 段 4.32 |
|---|---|---|
| 纯载入（qa+kb 全迭代，无 mma） | **2.07** | **48%** |
| 纯 mma（frag 预载一次，无逐迭代载入） | **1.17** | 27% |
| S(A@smem 假源)（延迟隔离） | 4.03 | −7% ⇒ 载入**延迟**无关 |
| S 四链独立（ILP 隔离） | 2.56 | −45% ⇒ 载入**指令/依赖**有关 |

### 根因（nvcc 实测定案）

PTX 全 401 条 `wmma.load.*.bf16` 在 SASS 落成 **4×LD.E 32 位标量 generic 载入**
（每 fragment 4 条，LD.E 可走 generic window 触 shared），**零 LDSM**。用
nvcc -ptx/-arch=sm_89 生成 canonical shared wmma（`__align__(16)`）对照：
SASS 同样零 LDSM——**ptxas 对 wmma.load 永不生成 ldmatrix**，vLLM/FA2 的
LDSM 全部来自手写 `ldmatrix.sync` 内联 PTX。我们的 Rust→NVVM 管线没有
ldmatrix intrinsic，asm! 在 nvptx 无寄存器类——换 LDSM 只剩 build.rs
raw-PTX 注入引擎（per-lane 地址改写 + 4-reg 返回旁路）+ Q staging 32KB
smem（>48KB 静态上限需动态 smem + cuFuncSetAttribute FFI）。

### 关刀判据

纯 mma 地板 1.17 + staging 1.44 + softmax 0.29 + O 1.09 ≈ **即使 LDSM
全套落地也只到 ~5.3ms/层（仍 2.2× vLLM 2.41）**；投入（PTX 注入引擎+
动态 smem+FFI）与回报（6009 −25ms/543、32K −~70ms/4587）不成比例。
**attention 微架构深挖到此为止**：v5（59.5ms@6009/8 层）为该管线终态，
redirect 到 6009 账本更大的项——空洞 ~104ms、delta h 换形态 ~80ms、
lm_head/logits ~20ms、W8A8。

### 收尾

- 删净：paged_attn_v6/v7/v8/v7_ablate 四 kernel + V6_KS/VT/P、V7_KS/VT/P
  smem 符号 + f_v6/f_v7/f_v8/f_v7a + run_tile_v6/v7/v8/v7_ablate +
  v6/v7/v8_meta + 探针五列/消融段。lib.rs 8076→6894 行。
- 验证：wein-kernel 全测 76 过（k1 入口 64 不变——v6/v7/v8 是未提交增量）、
  qwen 23/23、探针 v5≡v4 位级 0 差、6009 7.26/32000 210.4 与基线一致。
  battery 免（生产路径 v5 一字未动）。
