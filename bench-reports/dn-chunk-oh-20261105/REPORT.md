# delta chunked o/h + prep 刀（2026-11-05）：o 尾相 cp.async 竞态修复 + vsplit 定档 4 + tpar 整体删除 + prep 前代 8 行 ILP

## 结论速览

| 项 | 改前 | 改后 | 上一轮 vLLM |
|---|---|---|---|
| delta 段（6009 单 span，24 层） | 100.4 ms | **76.5 ms**（−23.9） | 22.7 |
| 其中 o kernel | 1095 µs/层 | **643 µs/层** | — |
| 其中 h kernel | 1492 µs/层 | **1341 µs/层** | — |
| 其中 prep kernel | 1492 µs/层 | **1114 µs/层**（−25%，§6） | — |
| 6009 冷 TTFT（图档，配对 A/B 中位） | 459.6 ms | **436.5 ms**（−5.0%） | 383-418 |
| battery 图档满载 tps / ITL p50 | 3629.1 / 23.57 | 3607.0 / 23.51、3577.8 / 23.85 | 4091 / 20.06 |

（§1-§5 = o/h 段；§6 = prep 前代 ILP；§7 = 合并后复测。）

## 1. o kernel：cp.async 尾相 wait 不足（真 bug）

`dn_chunk_o` 每相提交 S₀(hv+1) 后无条件 `cp_async_wait1`。末相（`hv+1 >= vph`）
不再提交新组，未决组恰为 1 ⇒ `wait_group 1` 立即返回、不担保 δ 完成，
δ(hv) 的读/写与之竞态。改为条件分派：

```
if hv >= 1 {
    if hv + 1 < vph { cp_async_wait1(...) } else { cp_async_wait_all(...) }
    _syncthreads();
}
```

`compute-sanitizer --tool racecheck` 由
`Write at $dn_chunk_o$cp_async_cg16+0x26930 vs Read at dn_chunk_o+0x1d6d0` 转
`RACECHECK SUMMARY: 0 hazards`；memcheck 0 errors。

**方法论留档**：本类时序 bug 的 `memcheck` 全程 0 error 且测试通过——只有
`racecheck` 能定位。

## 2. o35 sp=8 档 700 的根因：ptxas 对「动态段基址 + 互斥分支」的重排

racecheck 干净后 o35 第 4 case（sp=8/bt=64/vh=4/kh=2）仍稳定 700，出错
launch 是 h（`launch_smem smem=0 grid=32 block=128`）。定位手法：临时给
`cuda::launch/_smem` 加 `WEIN_LAUNCH_SYNC`（逐 launch `stream_synchronize`）
+ 三处 launch 前 eprintln 几何，一次跑出坏 launch（诊断代码已全部删除）。

判定链：① 临时把 sp8 档动态 smem 从 0 改成 8704 → 全过（确系写动态段）；
② 编译期关 tpar → 全过（锁定 tpar 的 C 片段标量写 `DN_H2_W` 头部）；
③ 把 `DN_H2_W` 改成真尺寸 `[u64; 8704]` → 仍崩（排除 LLVM 越界 UB）；
④ PTX 侧确认 `cvta.shared.u64 ..., DN_H2_W` 仅 1 处、无预取——是 ptxas 13.4
把该 store 提到了 `tpar=0` 兄弟路径。

**决策**：整条 tpar（t-并行 wmma 备胎，probe 实测与串行路平价、默认关）
删除，而不是继续与 ptxas 斗。删除面：`DnA16`、`DN_H2_BH/BL/I`、
`DnH.tpar`、档门与恒等 B 初始化、C 片段初值 mma、`lim`/`t<n` 分支化简、
A 镜像 hi/lo 写、② 分支、kernel 形参 `tpar`、`ops` 的 `dn_tpar` 与
`WEIN_DN_TPAR`、probe 实参。h 静态 smem 22784（原 28416）+ 动态 69632 = 92416B。

**留档**：动态 `.shared-dyn` 段上再放「互斥分支专用」的槽，风险已知。

## 3. vsplit 定档 4（探针三次取稳）

`dn_chunk_probe` 连跑 3 次，档位表稳定（改前同档曾记 vs8 3950、vs4 4182）：

| BT/vs | delta 段 ms（3 次） | 分解（µs/层，vs4） |
|---|---|---|
| 32/4 | 97.1 / 99.0 / 99.0 | — |
| **64/4** | **91.3 / 90.8 / 90.4** | l2 86-96 + prep 1494-1497 + h 1337-1342 + o 642-646 = 3565-3578 |
| 64/8 | 95.3 / 92.3 / 94.9 | h 1495-1501（bdim128 不 staging 档） |
| 128/4 | 124.2 / 124.5 / 125.8 | — |

⇒ `DN_CHUNK_VSPLIT` 默认 8→**4**（bdim=256 起 w hi/lo smem 双缓冲档），并加
`WEIN_DN_VSPLIT`∈{1,2,4,8} 旋钮（OnceLock）。上一轮「vs8 与 vs4 结论不稳、
h 在 1483/2429 双峰」的现象本次三次未复现，取 vs4。

## 4. 验证

- `ops_i1` 13/13（o35 容差 4 case、o20 位级、o22 延迟门禁）、`acceptance` 11/11；
  wein-kernel 其余二进制串行全绿；wein-exec 端到端 qwen 23 / qwen_graph 4 /
  acceptance 8 / graph_toy 3 / graph_engine 1 / qwen_bigchunk 2 全绿。
- 6009 冷配对 A/B（git worktree c33853a vs 本刀，交替 4 组 ×2 发）：
  HEAD 475.5/474.6/473.4/473.8（中位 474.5）vs 本刀 461.9/462.7/461.8/458.1
  （中位 461.9）⇒ **−12.6ms**，与 delta 段 −15.1ms 同量级。
  **注意**：刀账上一版记的 521-534 是 11-04 环境读数，11-05 同 HEAD 已复现
  474——跨日不可直接比，必须配对。
- battery 两轮（battery-20261105-184556 / -184626）：图档满载
  3607.0 tps / ITL 23.51 / p99 35.54 与 3577.8 / 23.85 / 38.08，对上一轮
  3629.1 / 23.57 / 35.49 属带内（当日 Windows 桌面计算进程 20 个 vs 基线 19）。

## 5. prep 的第一次尝试（smem 档，已整体撤除）

prep 三段（rhs → solve → pack）的中间态 X 常驻动态 smem（`DN_P_X`，
bt≤64 → 64KB，`#[link_section = ".shared-dyn"]` + host `launch_smem`），
省掉 f32 三写两读往返：

- 位级正确（o20 位级、o35 容差、其余全绿），但 **prep 1103 → 1648 µs（慢
  50%）**；A/B 把同一版代码切回全局档（`smx=false`）得 1138 ⇒ 慢因是 smem
  访问形态本身（每线程独占一列、列内串行、每 v 头 2 次屏障），不是 regs、
  不是算法。
- 撤除面：`DN_P_X` 声明、`smx` 分支、host/探针的 `launch_smem`+属性+`psm`，
  build.rs 动态段清单回到 `dyn ["DN_H2_W", "DN_O_DYN"]`（h/o 各一份）。

## 6. prep 前代：8 行一组 ILP（本刀收益）

病灶：`solve` 段每线程独占一列（n=64），每元素是 n²/2 ≈ 2016 条串行 fma
依赖链（4 周期延迟 ⇒ 8064 周期纯延迟，6 载/4 fma 也喂不满），prep 254 regs
⇒ 8 warp/SM 占用上限，无处藏延迟。

改法（`dn_prep_solve_col`）：8 行一组，组内 8 个独立累加器并行吸收 p<8g
的贡献（**共享** X[p] 载入，p 按 2 展开），再按 j 升序做组内前代。
每元素的 fma 序与逐行串行版**逐字相同**（p 升序 → 组内 j 升序，正是全局
p 升序）⇒ 输出位级不变；依赖链拆成 8 条并行链。尾组（n%8 行）保留原逐行
4× 展开。行指针预存 `[usize; 8]`（避开指针数组触发的 local 栈帧）。

| 项 | 改前 | 改后 |
|---|---|---|
| prep | 1492-1497 µs/层（探针 3 次） | **1112-1121 µs/层**（−25%） |
| prep regs / spill | 254 / 0 | **247 / 0** |
| delta 段（BT=64/vs4，24 层） | 90.4-91.3 ms | **76.5-77.2 ms** |

位级证据：`o20` 位级 golden 绿、`o35` 四 case rel2 1.26e-3~1.31e-3 / 峰
3.2e-3~6.5e-3、state rel2 1.2e-3~1.6e-3 —— 与改前**同档**；`ops_i1` 13/13、
`acceptance` 11/11。

## 7. 合并后复测（当日配对）

6009 冷 TTFT，git worktree c33853a 与本刀交替 3 组 ×2 发（取第 2 发）：

| 组 | HEAD(c33853a) | 本刀 |
|---|---|---|
| 1 | 459.6 | 436.5 |
| 2 | 457.7 | 435.6 |
| 3 | 459.7 | 437.7 |
| 中位 | 459.6 | **436.5**（−5.0%） |

−23.1 ms 与探针账（prep −9.1 + o −10.8 + h −3.6 = −23.5 ms/24 层）几乎
1:1 落到 TTFT。**HEAD 今日再测 459.6，而 §4 早先记 474.5** ⇒ 同日环境也能漂
3%，配对 A/B 已成硬纪律。

## 8. 留刀

prep 1114 µs/层仍是 delta 段的 35%、最大单项：

1. u/w 走 f32 三写两读往返（rhs 写 → solve 原位读重写 → pack 重读 + hi/lo
   镜像写）≈880 MB/层，改 bf16 直存可省；代价是动 o35 精度预算（余量 ~2.3×）。
2. A Gram 段 ntasks=64 < 256 线程饥饿、跨步 256（250 µs/层）。
3. rhs+scan 段 294 µs/层。
4. h 1341 µs/层（块间 94 链串行，255 regs ⇒ 2 块/SM）、o 643 µs/层。

顺带记录（非本刀）：`WEIN_STEP_PROFILE` 开启时 SIGTERM 收尾会打两条
`cuMemFree: CUDA_ERROR_DEINITIALIZED` panic（HEAD 与本刀各 2 条，完全相同
⇒ 既有、仅影响 profile 模式退出日志，不影响读数/退出码）。
