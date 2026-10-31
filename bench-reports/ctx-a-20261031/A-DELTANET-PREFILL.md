# 长上下文刀 · 第二件：deltanet prefill A 档（位级不变）

日期 2026-10-31｜`crates/wein-kernel/kernels/src/lib.rs`（仅 `specs==0` 的 E2 分支）
所有数字均在同一台机（RTX 4090 24G）、串行、单进程、每次基准前 `cargo build --release` 测得。

## 结论

| 项 | 旧 | 新 | Δ |
|---|---|---|---|
| 250 token 单 span prefill：`delta` 段 | 94.84 / 92.36 ms | 62.65 / 64.39 ms | **−32%** |
| 250 token prefill TTFT（5 次中位） | 118.99 / 118.88 ms | 88.68 / 88.18 ms | **−26%** |
| m=100 decode `delta` 段 | 6.75 / 6.48 ms | 5.86 ms | **−13%** |
| `o22` decode 几何时延（门禁 0.36ms） | 0.286–0.299 ms | 0.262–0.264 ms | **−9%** |
| `deltanet` 寄存器 / spill | 167 / 0 | **167 / 0** | 不变（仍 3 block/SM） |
| 每 token 每层 delta | 15.6 µs | **10.6 µs** | −32% |
| 距带宽地板（12GB ÷ 800GB/s = 15.0ms） | 6.2× | **4.2×** | — |

位级：`o20/o25/o26/o27/o22/o16_deltanet_segment_equivalence` 全绿；`o25` 输出/末态漂移
与改动前逐字同（`rel2 1.261e-3 / 峰值 4.202e-3（4231/65536 位不同）`、`末态 1.635e-3 / 2.142e-3`）。
`wein-exec` qwen 23 例 + qwen_graph 4 例全绿（`qs9 accepted=3 target=[9662,85111,22857,310]`）。
`acceptance::k1` 的 `.visible .entry == 19` 是**既有陈旧门禁**：旧 PTX 与旧 PTX 逐名同集合、
都是 20 个 entry（`/tmp/old.entries` ≡ `new.entries`），与本刀无关。

## 改动（三处，均位级不变）

1. `fold32!`（lib.rs:1134-1175）—— kvm/o 两级折叠的**第二级**（跨 lane 32 项 u 升序）
   从滚动 `while u < 32` 改成下标全 literal 展开。加法序逐字同（u 升序、`acc = v + acc`），
   拿掉的只是循环回边；这正是旧版每项 LDG 都吃满一次 L1 时延的根源（本刀主项 ≈ −24ms）。
2. o 折叠去冗余（lib.rs:1235-1242）—— 旧版 32 个 lane 各把批内 4 行都折一遍、只留 `b == s`
   的一个（3/4 纯冗余）；改成 `if s < 4` 只折自己那行，同序 ⇒ 位同。
3. l2norm 的 u64 分组载入（lib.rs:1084-1119）—— **和树不动**（j 升序 128 项单链），
   只把 128 次标量 u16 载入合并成 32 次 u64 整载并把每组的 4 项展开；`sq`/`sk` 两条链
   各自的加法序仍是 j 升序且 acc 在前。实测 −5.4…−6.7ms。

barrier 数、8 批结构、scratch 槽布局、`specs==1` 投机链分支（lib.rs:946-1065）**一个字节未动**。

## 被否掉的方案（有数）

- **状态行整段驻寄存器**（A 档原第 1 项）：实现并实测过。prefill 收益 ≈ 0
  （仅驻寄存器版 95.86→95.24ms；与 fold32 合并版 63.54 vs 不驻寄存器 64.39 = 噪声内），
  代价是 32×u64=64 个寄存器 → 167 → **255 regs** → 2 block/SM（168 才是 3 block 的门槛），
  decode 几何 `o22` 0.376ms **超 0.36ms 门禁**（不可放松）、decode delta 反劣。
  ⇒ 已回滚；根因是状态行本来就在 L1/L2 里热着，真瓶颈是归约链的 LDG 依赖，不是状态往返。
- **l2norm 换归约树（scratch 两级折叠）**：位级不可行。128 个 bf16 平方和不是精确和，
  两级折叠 49152 组里 42525 组末位不同（`/tmp/l2chk`）。唯一位级安全的"去冗余"是让单线程
  独算再广播——128 深依赖链不变，还多一道屏障，严格更慢。

## 探针归因（250 token prefill delta 的构成）

| 探针（临时，已回滚） | delta ms |
|---|---|
| 现版 | 62.65 |
| 去掉 l2norm（invq=invk=1） | 57.27 ⇒ l2norm 残值 ≈ 5.4–6.7ms（9%） |
| 去掉 dn_batch 的 16 道屏障/token | 53.88 ⇒ 屏障 ≈ 8.8–10.5ms（15%） |
| 余下 = kvm/o 折叠依赖链 | ≈ 45ms（72%） |

## 外推（线性，前提：同一 32-block 单 span 几何、每 token 每层 10.6µs 不随位置变）

- 1k：delta 251ms；GPU 段和 ≈ 343ms
- 8k：delta 2.01s；GPU 段和 ≈ 2.8s
- 按调度 chunk_size=256 切块时：8k = 32 步 × ~88.7ms ≈ 2.84s（attention 段另计，见下）

## 8k 的真实阻碍（确认用户判断）

1. `kernels/src/lib.rs:451,462` `const MAX_LEN: usize = 256` —— `pos+1 > MAX_LEN` 整块静默不写。
   且 M0 是块私有 `[MAX_LEN+2] f32` 两遍扫，抬到 8k 需要 flash 版（W2）。
2. `crates/wein-exec/src/qwen.rs:65` `MAX_POS = 256`（位置上限/分母数组；`SPEC_K_MAX` 由它推）。
3. 容量：`NUM_PAGES=512 × BLOCK=16 = 8192` token **全池**（qwen.rs:101-102）——单个 8k 序列
   就吃光整池，多行并发要成倍扩页。
4. 调度 `token_budget=512 / chunk_size=256`（wein-sched/src/lib.rs:67,69）——8k 单提示是
   32 个 256-token chunk 串行，所以"250 token 段表"就是它的单位账；`T_MAX=512` scratch 够用。

## 剩余空间

- 屏障合并（16→4/token）：把 scratch 从 1024 抬到 2048 f32/block（host `ops/mod.rs` 的
  `DELTA_BLOCK_CAP` × 1024 → 2048），批 4 行 → 16 行 ×2 批。算术序按行独立 ⇒ 位级不变，
  预期再省 ~6.7ms（11%）。
- B 档 chunked（把 v 行按块切开：每块 32 v 行 ⇒ grid 32→128，每 warp 8 行 = 2 批）：
  折叠依赖链 ÷4 ⇒ delta ≈ 45/4 + 6 + 9/4 ≈ 19ms，TTFT ≈ 45ms（2.5–3.5× 于当前）。

## 文件

- `prefill-250-old-ab.txt` / `prefill-250-new.txt` / `decode-100-old-ab.txt` / `decode-100-new.txt`
- 旧二进制副本 `/tmp/wein-old-bin/`（同日 A/B 用）
- battery：`battery-20261031-134113`（上一轮）、`battery-20261031-141300`（新）、
  `battery-20261031-141329`（旧 binary 同日对照）
