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
    let mut big = Vec::new();
    while big.len() < 2000 {
        big.extend_from_slice(&base);
    }
    big.truncate(2000); // 单 span 2000 ≥ DN_CHUNK_MIN_SPAN → chunked 路
    let small: Vec<u32> = base[..base.len().min(40)].to_vec(); // < 阈值 → 老路
    let gen = 8;

    // 对照：各自单跑
    let alone_big = run(&mut ex, &big, &[2000], gen);
    let alone_small = run(&mut ex, &small, &[small.len()], gen);

    // 混合：small 先追平，随后 big 的 2000-token span 与 small 的 decode 行同步
    ex.retain(&[]);
    ex.admitted(1, &big);
    ex.admitted(2, &small);
    let mut got_big = Vec::with_capacity(gen);
    let mut got_small = Vec::with_capacity(gen);
    // 步 1：small 追平（span 40，老路单发——big 无行）
    let out = ex.step(&StepPlan {
        prefill: vec![(2, small.len())],
        decode: vec![2],
        batch: vec![1, 2],
    });
    assert!(ex.take_failures().is_empty(), "small 追平步失败");
    got_small.push(out.first().map(|&(_, t)| t).expect("追平步出首 token"));
    // 步 2（混合两路）：big 的 span 2000（chunked，decode 含 1 ⇒ 段尾采样）
    // + small decode 行（老路）——同一前向、两次 deltanet launch
    let out = ex.step(&StepPlan { prefill: vec![(1, 2000)], decode: vec![1, 2], batch: vec![1, 2] });
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
    // 合同（kernel 层）：批形不改 attention 数值——同一行的 j 分段/归约序只
    // 依其自身 (seq,pos) 与页表容量 bpr×bs，与邻居行的位置/行数无关（见
    // o41/o42 位级档）。故混批与
    // 单跑的贪心 token 序列须**逐 token 同**。注意：全程 gemm 路径由测试钉成
    // heuristic（见 open 前的 WEIN_TUNE_M=""）——autotune 的跨进程计时选型
    // 差异会翻转末位，但那不是 attention 的批形依赖，不能混进这条对拍。
    let big_div = got_big.iter().zip(&alone_big).filter(|(a, b)| a != b).count();
    let small_div = got_small.iter().zip(&alone_small).filter(|(a, b)| a != b).count();
    if std::env::var("WEIN_ATTN_V3").map(|v| v != "0").unwrap_or(true) {
        assert_eq!(
            big_div, 0,
            "v3 下混合批 big 贪心序列应逐 token 同（分叉 {big_div}/{}）",
            gen
        );
        assert_eq!(
            small_div, 0,
            "v3 下混合批 small 贪心序列应逐 token 同（分叉 {small_div}/{}）",
            gen
        );
    } else {
        assert_eq!(got_big, alone_big, "混合批里 big（chunked 路）须与单跑逐 token 同");
        assert_eq!(got_small, alone_small, "混合批里 small（老路）须与单跑逐 token 同");
    }
}
