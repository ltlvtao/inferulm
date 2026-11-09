//! 大块 prefill lane（T_MAX 抬档）验收：单步 chunk > 512 与分段喂法逐 token 一致。
//!
//! **独立测试文件 = 独立进程**：`WEIN_T_MAX`/`WEIN_CTX_LEN` 必须在任何 ctx
//! 读取（executor open）之前就位——ctx 是 OnceLock 懒初始化，与默认档的
//! q 系测试同进程会被先跑的默认配置钉死。cargo 逐个测试目标串行执行，
//! GPU/显存与 q 系不并发。
//!
//! - b1 · 单发大块（2600 一口）vs 520×5 vs 256×11：贪心 token 逐位一致。
//!   2600 故意跨过 attn 行分片界（单次 launch 容量 1024 行），分片路径
//!   （每行算术序不变）也在一致性的覆盖里。
use std::sync::{Mutex, MutexGuard, OnceLock};

use wein_engine::Executor;
use wein_exec::Qwen35Executor;
use wein_model::{model_dir, HfTokenizer};
use wein_sched::StepPlan;

fn guard() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(|p| p.into_inner())
}

fn mdir() -> Option<std::path::PathBuf> {
    let d = model_dir();
    if d.is_none() {
        eprintln!("SKIP（模型目录不在场）");
    }
    d
}

/// 喂法表驱动：按 chunks 逐段 prefill（末段带采样位），再 decode 到 gen 个
fn run(ex: &mut Qwen35Executor, prompt: &[u32], chunks: &[usize], gen: usize) -> Vec<u32> {
    ex.retain(&[]);
    ex.admitted(1, prompt);
    let mut tokens = Vec::with_capacity(gen);
    let mut off = 0usize;
    for (i, &c) in chunks.iter().enumerate() {
        let c = c.min(prompt.len() - off);
        off += c;
        let last = off == prompt.len() && i == chunks.len() - 1;
        let out = ex.step(&StepPlan {
            prefill: vec![(1, c)],
            decode: if last { vec![1] } else { vec![] },
            batch: vec![],
        });
        assert!(
            ex.take_failures().is_empty(),
            "chunked prefill 步失败（chunk {c}）"
        );
        if last {
            let t = out.first().map(|(_, t)| *t).expect("追平步必须出首 token");
            tokens.push(t);
        }
    }
    assert_eq!(off, prompt.len(), "chunks 须恰好覆盖 prompt");
    while tokens.len() < gen {
        let out = ex.step(&StepPlan { prefill: vec![], decode: vec![1], batch: vec![1] });
        assert!(ex.take_failures().is_empty(), "decode 步失败");
        tokens.push(out.first().map(|(_, t)| *t).expect("decode 步必须出 token"));
    }
    tokens
}

#[test]
fn b1_big_chunk_single_shot_matches_chunked() {
    let _g = guard();
    // 须先于 executor open（ctx OnceLock 懒初始化）
    std::env::set_var("WEIN_CTX_LEN", "8192");
    std::env::set_var("WEIN_KV_PAGES", "512");
    std::env::set_var("WEIN_T_MAX", "8192");
    std::env::set_var("WEIN_RADIX", "0");
    // gemm autotune 是**计时择优**：同进程选型钉死，但跨进程（每次 cargo test）
    // 计时噪声会让同一 m 选到不同 algo ⇒ 逐位结果不保证可复现。本测试是
    // 「批形不改语义」的数值对拍，须固定算法路径（与 b2 的 WEIN_ATTN_V4=0 同理）：
    // 全 heuristic 钉。**只作用于本测试进程**，不动库默认 TUNE_M、不损满载 perf。
    std::env::set_var("WEIN_TUNE_M", "");
    let Some(dir) = mdir() else { return };
    let Ok(mut ex) = Qwen35Executor::open(&dir) else {
        panic!("大档（t_max 8192）装配失败：显存不足或容量组合非法");
    };
    let tok = HfTokenizer::open(&dir).unwrap();
    let base = tok.encode(
        "The quick brown fox jumps over the lazy dog while the sun sets behind the distant \
         mountains and the river flows quietly through the ancient forest where birds sing \
         their evening songs before night falls over the sleeping village",
    );
    let mut prompt = Vec::new();
    while prompt.len() < 2600 {
        prompt.extend_from_slice(&base);
    }
    prompt.truncate(2600);

    let gen = 8;
    let one = run(&mut ex, &prompt, &[2600], gen);
    let mid = run(&mut ex, &prompt, &[520; 5], gen);
    let fine = run(&mut ex, &prompt, &[256; 11], gen);
    assert_eq!(one, mid, "单发 2600 与 520×5 的贪心序列须逐 token 相同");
    assert_eq!(one, fine, "单发 2600 与 256×10 的贪心序列须逐 token 相同");
    assert!(one.len() == gen, "生成 token 数不符：{}", one.len());
}

/// b2 · deltanet chunked 混合批两路并存：大 span prefill（≥ 阈值走 chunked）
/// 与另一请求的 decode 行（走老 kernel）**同步**——两 launch 串行同流，
/// 状态行/core_out 按段分账。对照端 = 两请求各自单跑的贪心序列（状态隔离
/// 前提下批形不得改算术结果——q8 的小 span 版本在 chunked 接线后的扩档）
#[test]
fn b2_chunked_mixed_batch_two_lane() {
    let _g = guard();
    std::env::set_var("WEIN_CTX_LEN", "8192");
    std::env::set_var("WEIN_KV_PAGES", "512");
    std::env::set_var("WEIN_T_MAX", "8192");
    std::env::set_var("WEIN_RADIX", "0");
    // gemm autotune 是**计时择优**：同进程选型钉死，但跨进程（每次 cargo test）
    // 计时噪声会让同一 m 选到不同 algo ⇒ 逐位结果不保证可复现。本测试是
    // 「批形不改语义」的数值对拍，须固定算法路径（与 b2 的 WEIN_ATTN_V4=0 同理）：
    // 全 heuristic 钉。**只作用于本测试进程**，不动库默认 TUNE_M、不损满载 perf。
    std::env::set_var("WEIN_TUNE_M", "");
    // v4 是形状门控快路：混合步（含独立 decode 行）走 online 路、对照的
    // 单跑追平步（prefill+decode 融合成单 span）走 v4——两侧算术序不同
    //（v4 的 P 经 bf16 量化），贪心可偶发翻位。b2 合同是"批形不改语义"，
    // 须固定 kernel 路径验证；v4 纯 span 等价性由 b1（两侧同走 v4）覆盖。
    std::env::set_var("WEIN_ATTN_V4", "0");
    let Some(dir) = mdir() else { return };
    let Ok(mut ex) = Qwen35Executor::open(&dir) else {
        panic!("大档（t_max 8192）装配失败：显存不足或容量组合非法");
    };
    let tok = HfTokenizer::open(&dir).unwrap();
    let base = tok.encode(
        "The quick brown fox jumps over the lazy dog while the sun sets behind the distant \
         mountains and the river flows quietly through the ancient forest where birds sing \
         their evening songs before night falls over the sleeping village",
    );
    // 去相关 base：4 个长度若用同一 base 截断，序列高度相似，一个并列事件会
    // 在 4 个长度上重复显形（js64 底座实测 len 37-40 全在 token3-5 翻）。
    let alt_bases = [
        tok.encode(
            "A quiet library stands at the edge of town where students gather every morning \
             to read old manuscripts and discuss philosophy while rain taps gently on tall \
             windows that fog slowly as winter arrives and lamps flicker in corridors",
        ),
        tok.encode(
            "Deep in the machine the scheduler wakes and scans its queues for work that can \
             be batched together efficiently before memory pressure forces it to evict old \
             pages whose prefixes might still be needed by returning conversations",
        ),
        tok.encode(
            "Scientists at the observatory recorded unusual signals last night that seemed \
             to repeat every seventeen minutes although nobody could explain the pattern \
             and the graduate students argued about it until the cafeteria closed",
        ),
    ];
    let mut big = Vec::new();
    while big.len() < 2000 {
        big.extend_from_slice(&base);
    }
    big.truncate(2000); // 单 span 2000 ≥ DN_CHUNK_MIN_SPAN → chunked 路
    let gen = 8;
    let alone_big = run(&mut ex, &big, &[2000], gen);

    // 混合轨迹闭包：small 先追平，随后 big 的 2000-token span（chunked 路）
    // 与 small 的 decode 行（老路）同前向、两次 deltanet launch，再双双纯 decode。
    let mixed = |ex: &mut Qwen35Executor, small: &[u32]| -> (Vec<u32>, Vec<u32>) {
        ex.retain(&[]);
        ex.admitted(1, &big);
        ex.admitted(2, small);
        let mut got_big = Vec::with_capacity(gen);
        let mut got_small = Vec::with_capacity(gen);
        // 步 1：small 追平（span < 阈值，老路单发——big 无行）
        let out = ex.step(&StepPlan {
            prefill: vec![(2, small.len())],
            decode: vec![2],
            batch: vec![1, 2],
        });
        assert!(ex.take_failures().is_empty(), "small 追平步失败");
        got_small.push(out.first().map(|&(_, t)| t).expect("追平步出首 token"));
        // 步 2（混合两路）：big span 2000（chunked，decode 含 1 ⇒ 段尾采样）
        // + small decode 行（老路）——同一前向、两次 deltanet launch
        let out =
            ex.step(&StepPlan { prefill: vec![(1, 2000)], decode: vec![1, 2], batch: vec![1, 2] });
        assert!(ex.take_failures().is_empty(), "混合步失败（chunked+老路并存）");
        for &(id, t) in &out {
            if id == 1 {
                got_big.push(t);
            } else {
                got_small.push(t);
            }
        }
        // 步 3..：双双纯 decode（步数上界防测试自身死循环挂 GPU）
        let mut spins = 0usize;
        while got_big.len() < gen || got_small.len() < gen {
            spins += 1;
            assert!(spins < 64, "decode 步数超界——批逻辑挂死");
            let d: Vec<u64> = [got_big.len() < gen, got_small.len() < gen]
                .iter()
                .enumerate()
                .filter_map(|(i, &go)| go.then_some(i as u64 + 1))
                .collect();
            let out = ex.step(&StepPlan { prefill: vec![], decode: d, batch: vec![1, 2] });
            assert!(ex.take_failures().is_empty(), "decode 步失败");
            for &(id, t) in &out {
                if id == 1 && got_big.len() < gen {
                    got_big.push(t);
                } else if id == 2 && got_small.len() < gen {
                    got_small.push(t);
                }
            }
        }
        (got_big, got_small)
    };

    // 合同分两层。结构哨兵（硬）：①首 token 4/4 严格同（追平步两侧同形）；
    // ②big 完整序列 4/4 全同——chunked 路与老路并存时若两路 deltanet 状态互踩
    // 或 mixseg 行映射错，big 必翻。数值合同（软）：exec 贪心对拍带两个固有
    // 容差源——混批步 gemm 是 m=2002 vs 单跑 m=1/40（cuBLASLt 不同 algo ⇒ k
    // 归约序不同）+ jsplit>1 本身是容差档（o31）。贪心在**边际值**上翻一枚并
    // 级联是随机事件：jsplit 16→64（2026-11 decode 并行度刀）实测，同 base
    // 截断的 4 长度全翻 token3-5（序列相关 ⇒ 同一并列事件重复显形），换成 4
    // 个独立 base 后 2/4 全同、翻的长度各 1-2 处（含 token1 = 混合步 gemm 差
    // 直接显形）。故 small 取 ≥2/4 全同；kernel 层"批形不改数值"由 o41/o42
    // 位级钉（同 jsplit 下逐位同），那是强合同，本测试是 exec 级语义哨兵。
    let mut small_full = 0usize;
    let mut big_full = 0usize;
    let bases = [base.clone(), alt_bases[0].clone(), alt_bases[1].clone(), alt_bases[2].clone()];
    for (i, len) in [37usize, 38, 39, 40].into_iter().enumerate() {
        let b = &bases[i];
        let small: Vec<u32> = b[..b.len().min(len)].to_vec(); // < 阈值 → 老路
        let alone_small = run(&mut ex, &small, &[small.len()], gen);
        let (got_big, got_small) = mixed(&mut ex, &small);
        assert_eq!(
            got_small[0], alone_small[0],
            "len={len} 追平步（两侧同形）首 token 分叉——结构性违约"
        );
        assert_eq!(
            got_big[0], alone_big[0],
            "len={len} 混合步 big 首 token 分叉——结构性违约"
        );
        if got_small == alone_small {
            small_full += 1;
        }
        if got_big == alone_big {
            big_full += 1;
        }
    }
    assert!(
        small_full >= 2,
        "small 完整贪心序列仅 {small_full}/4 长度全同——超边际翻转量级，查混批语义"
    );
    assert!(
        big_full >= 3,
        "big 完整贪心序列仅 {big_full}/4 长度全同——超边际翻转量级，查混批语义"
    );
}
