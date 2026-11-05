//! 构建 kernels/ 子 crate 到 PTX（nvptx64 目标，nightly + build-std=core），
//! 产物拷入 OUT_DIR 并经 WEIN_PTX 环境变量注入本 crate（测试与运行时都用它）。
//!
//! 这是"cuda crate 纯 Rust→PTX（无 nvcc）"路线的构建期落点：一次 cargo 调用
//! 完成全部编译，产物是 PTX 文本，host 侧由 driver JIT（k7 验证其缓存）。
//!
//! 子 cargo 必须与外层隔离：独立 target-dir（防锁冲突）+ 剥离外层注入的
//! CARGO_* 环境变量（防外层 workspace 上下文泄漏进子构建）。
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernels = manifest.join("kernels");

    println!("cargo:rerun-if-changed={}", kernels.join("src").display());
    println!("cargo:rerun-if-changed={}", kernels.join("Cargo.toml").display());
    println!(
        "cargo:rerun-if-changed={}",
        kernels.join(".cargo/config.toml").display()
    );

    let target_dir = out.join("kbuild");

    // 注意：build script 环境里的 CARGO 是**已解析的工具链内 cargo 二进制**
    // （stable），不认识 RUSTUP_TOOLCHAIN——必须走 rustup shim 或直接定位
    // nightly 工具链目录，否则内层悄悄用回 stable（nvptx64 std 不在，E0463）
    let (cargo, via_shim) = nightly_cargo();

    let mut cmd = Command::new(&cargo);
    cmd.current_dir(&kernels) // 读 kernels/.cargo/config.toml（target 等）
        .args([
            "build",
            "--release",
            "--target",
            "nvptx64-nvidia-cuda",
            "--target-dir",
        ])
        .arg(&target_dir);
    if via_shim {
        cmd.env("RUSTUP_TOOLCHAIN", "nightly"); // 只有 shim 解释这个变量
    }
    // 剥离外层 build 脚本环境里指向 stable 工具链的变量：
    // - CARGO_*（manifest 路径/target dir 等会误导子构建）
    // - RUSTC/RUSTFLAGS 等（cargo 给 build script 设的是 stable 的 rustc，
    //   内层 cargo 继承它就悄悄用回 stable——E0463 的第二个来源）
    for (k, _) in env::vars_os() {
        let name = k.to_string_lossy();
        if name.starts_with("CARGO_")
            || name == "RUSTC"
            || name.starts_with("RUSTC_")
            || name == "RUSTFLAGS"
            || name == "RUSTDOCFLAGS"
        {
            cmd.env_remove(&k);
        }
    }

    let st = cmd.status().expect("启动子 cargo（kernels → PTX）失败");
    assert!(st.success(), "kernels PTX 构建失败（nightly + rust-src 装了吗？）");

    // nvptx64 的 cdylib 产物即 PTX 文本：预编译 std 时落在 release 根，
    // build-std 后备路径下落在 build/*/out/——先查根，再递归兜底
    let rel = target_dir.join("nvptx64-nvidia-cuda/release");
    let mut ptx: Vec<PathBuf> = std::fs::read_dir(&rel)
        .unwrap_or_else(|e| panic!("找 PTX 产物：{rel:?}：{e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "ptx"))
        .collect();
    if ptx.is_empty() {
        fn walk(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
            if let Ok(rd) = std::fs::read_dir(dir) {
                for e in rd.flatten() {
                    let p = e.path();
                    if p.is_dir() {
                        walk(&p, out);
                    } else if p.extension().is_some_and(|x| x == "ptx") {
                        out.push(p);
                    }
                }
            }
        }
        walk(&rel, &mut ptx);
    }
    assert_eq!(ptx.len(), 1, "应恰有一个 .ptx 产物，实得 {ptx:?}");

    let dst = out.join("wein_kernels.ptx");
    // PTX 文本级 patch：把源里 `#[link_section=".shared"]` 标记的数组从
    // `.global` 重写为 `.shared`（nvptx 后端不识别该段名，直出 .global）。
    // 配套三处改写：① 声明 .global→.shared ② 取址 mov.b64→cvta.shared.u64
    // ③ 引用该符号的函数体内 ld/st 去 .global（泛型空间——cvta 后指针在
    // generic 空间）。符号名单由源文件扫描而来（新增 smem 数组自动进表）。
    patch_smem_symbols(&ptx[0], &dst, &kernels.join("src/lib.rs"));
    // PTX 文本级 patch 之二：cp.async 占位函数体替换。Rust 表达不了
    // addrspace(3)/(1) 指针（intrinsic 签名 mismatch），nvptx 后端又无 asm!，
    // 故 lib.rs 用 `#[no_mangle]` 占位函数（volatile 直拷体，保证不被 IPO
    // 删调用），本补丁把函数体文本替换为真 cp.async 指令序列。占位体若
    // 未被替换就上线会写坏 smem——任何一步失败都 panic，绝不静默。
    patch_cp_async(&dst);
    // PTX 文本级 patch 之三：v9 手写 attention kernel（FA2 形态）——
    // ldmatrix + mma.sync + C→A 直用 + cp.async 双缓冲 + 99KB 动态 smem。
    // nvptx 无 ldmatrix intrinsic、asm! 无寄存器类 ⇒ 只能整函数体注入。
    patch_v9(&dst);
    println!("cargo:rustc-env=WEIN_PTX={}", dst.display());
}

/// 扫描源文件取 `#[link_section = ".shared"]` 标记的静态数组名，把 PTX 里
/// 这些符号的三处形态改写为真共享内存。改写失败（找不到符号）直接 panic——
/// 不许静默产出跑不动的 PTX。
fn patch_smem_symbols(ptx_src: &PathBuf, dst: &PathBuf, lib_rs: &PathBuf) {
    let src = std::fs::read_to_string(lib_rs).expect("读 kernels src");
    // 符号名：#[link_section = ".shared"] 后（允许隔注释行）的 static mut NAME
    let mut names: Vec<String> = Vec::new();
    // 动态段符号：#[link_section = ".shared-dyn"]——声明改写为 extern
    // 动态 smem（宿主 launch_smem 供字节），其余三处改写同静态段
    let mut dyn_names: Vec<String> = Vec::new();
    let mut mark = 0u8; // 0 无 / 1 静态 / 2 动态
    for line in src.lines() {
        let t = line.trim();
        if t.contains("#[link_section = \".shared\"]") {
            mark = 1;
            continue;
        }
        if t.contains("#[link_section = \".shared-dyn\"]") {
            mark = 2;
            continue;
        }
        if mark > 0 {
            if t.starts_with("//") {
                continue;
            }
            if let Some(rest) = t.strip_prefix("static mut ") {
                let name: String =
                    rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
                if !name.is_empty() {
                    if mark == 1 {
                        names.push(name);
                    } else {
                        dyn_names.push(name);
                    }
                }
            }
            mark = 0;
        }
    }
    if names.is_empty() && dyn_names.is_empty() {
        // 无 smem 数组：原样拷贝
        std::fs::copy(ptx_src, dst).expect("拷贝 PTX 产物");
        return;
    }
    let ptx = std::fs::read_to_string(ptx_src).expect("读 PTX 产物");
    let mut in_fn = false;
    let mut patched_count = 0usize;
    // v3 entry 插 .maxntid 512（.version 7.0 无 .maxnreg）：逼 ptxas 把 regs
    // 压到 65536/512=128 → bdim≤512 均可发射且 bdim256 得 2 块/SM 占用。
    // （dn_chunk_o 曾试 .maxnreg 128 注入：瓦片 acc 组溢出净亏 60%，已废）
    let mut pending_dir: Option<&str> = None;
    let mut lines_out: Vec<String> = Vec::with_capacity(ptx.len() / 24);
    for l in ptx.lines() {
        if l.starts_with(".visible .entry ") {
            in_fn = true;
            if l.starts_with(".visible .entry paged_attn_v3_") {
                pending_dir = Some(".maxntid 512, 1, 1");
            }
        }
        let mut line = l.to_string();
        let has_dyn = dyn_names.iter().any(|n| l.contains(n.as_str()));
        let has_sym = has_dyn || names.iter().any(|n| l.contains(n.as_str()));
        if has_dyn && l.starts_with(".global ") {
            // 动态段声明整行改 extern 动态 smem（名字取 mangled 前缀不变）：
            // 声明形如 `.global .align 8 .b8 SYM[32];`
            let sym: String = l
                .split_whitespace()
                .last()
                .unwrap_or_default()
                .chars()
                .take_while(|c| *c != '[' && *c != ';')
                .collect();
            line = format!(".extern .shared .align 16 .b8 {sym}[];");
            patched_count += 1;
        } else if has_sym && l.contains(".global") {
            line = l.replace(".global", ".shared");
            patched_count += 1;
        } else if has_sym && l.contains("mov.b64") {
            line = l.replace("mov.b64", "cvta.shared.u64");
            patched_count += 1;
        } else if in_fn && (l.contains("st.global") || l.contains("ld.global")) {
            line = l.replace("st.global", "st").replace("ld.global", "ld");
        }
        // 待插指令在 '{' 行**前**落（入体内 ptxas 解析报错）
        if let Some(d) = pending_dir {
            if l.trim_start().starts_with('{') {
                lines_out.push(d.to_string());
                pending_dir = None;
            }
        }
        lines_out.push(line);
        if in_fn && l == "}" {
            in_fn = false;
        }
    }
    let out = lines_out.join("\n");
    for n in &names {
        assert!(
            out.contains(&format!(".shared .b8 ")) || patched_count > 0,
            "smem 符号 {n} 未在 PTX 中改写"
        );
    }
    for n in &dyn_names {
        assert!(
            out.contains(".extern .shared .align 16 .b8 ") && out.contains(&format!("{n}[]")),
            "动态 smem 符号 {n} 未在 PTX 中改写"
        );
    }
    std::fs::write(dst, out).expect("写改写后 PTX");
    println!(
        "cargo:warning=smem 改写：{:?}（dyn {:?}，{patched_count} 处）",
        names, dyn_names
    );
}

/// cp.async 占位函数体替换。签名形态（LLVM NVPTX 稳定产出）：
/// `.visible .func NAME(\n\t.param .b64 NAME_param_0[,...]\n) // @NAME` + `{体}`。
/// 体替换为 ld.param 取参 → cvta 空间转换 → 真 cp.async 指令。同时把
/// V5_KS 声明 `.align 4` 提升 `.align 16`（cp-size 16 要求 dst 16B 对齐，
/// global 侧 kbase=32768·page+512·kvh 恒 16B 对齐）。
fn patch_cp_async(dst: &PathBuf) {
    let ptx = std::fs::read_to_string(dst).expect("读 smem 改写后 PTX");
    let mut out = replace_cp_placeholder(&ptx, "cp_async_cg16", 2, |p| format!(
        "\t.reg .b64 \t%rd<2>;\n\n\
         \tld.param.b64 \t%rd0, [{p0}];\n\
         \tld.param.b64 \t%rd1, [{p1}];\n\
         \tcvta.to.shared.u64 \t%rd0, %rd0;\n\
         \tcvta.to.global.u64 \t%rd1, %rd1;\n\
         \tcp.async.cg.shared.global \t[%rd0], [%rd1], 16;\n\
         \tret;\n",
        p0 = p[0], p1 = p[1],
    ));
    out = replace_cp_placeholder(&out, "cp_async_commit", 1, |_| {
        "\tcp.async.commit_group;\n\tret;\n".to_string()
    });
    out = replace_cp_placeholder(&out, "cp_async_wait_all", 1, |_| {
        "\tcp.async.wait_all;\n\tret;\n".to_string()
    });
    out = replace_cp_placeholder(&out, "cp_async_wait1", 1, |_| {
        "\tcp.async.wait_group 1;\n\tret;\n".to_string()
    });
    // prefetch.global.L2：ld.param 取参 → cvta.global → 真 prefetch（纯
    // 提示指令，无落点）
    out = replace_cp_placeholder(&out, "dn_l2pf", 1, |p| format!(
        "\t.reg .b64 \t%rd<1>;\n\n\
         \tld.param.b64 \t%rd0, [{p0}];\n\
         \tcvta.to.global.u64 \t%rd0, %rd0;\n\
         \tprefetch.global.L2 \t[%rd0];\n\
         \tret;\n",
        p0 = p[0],
    ));
    // V5_KS 对齐提升（smem 补丁已把声明改写为 .shared）
    let mut bumped = 0usize;
    let lines: Vec<String> = out
        .lines()
        .map(|l| {
            if l.contains("V5_KS") && l.contains(".align 4 .b8 ") {
                bumped += 1;
                l.replace(".align 4 .b8 ", ".align 16 .b8 ")
            } else {
                l.to_string()
            }
        })
        .collect();
    assert_eq!(bumped, 1, "V5_KS 对齐提升应恰一处，实得 {bumped}");
    out = lines.join("\n");
    for needle in [
        "cp.async.cg.shared.global",
        "cp.async.commit_group",
        "cp.async.wait_all",
        "cp.async.wait_group",
        "prefetch.global.L2",
    ] {
        assert!(out.contains(needle), "cp.async 指令 {needle} 未出现在产物 PTX");
    }
    assert!(
        !out.contains("st.volatile.b64") && !out.contains("st.volatile.b32"),
        "占位 volatile 体残留——绝不允许上线"
    );
    std::fs::write(dst, out).expect("写 cp.async 改写后 PTX");
    println!("cargo:warning=cp.async 改写：四函数体 + V5_KS .align 16");
}

/// 定位 `name` 函数定义，花括号匹配替换体。`body(params) -> 指令文本`。
fn replace_cp_placeholder(
    ptx: &str,
    name: &str,
    n_params: usize,
    body: impl FnOnce(&[String]) -> String,
) -> String {
    // 占位可能是普通函数（cp.async）或 kernel 入口（v9）
    let start = match ptx.find(&format!(".visible .func {name}(")) {
        Some(i) => i,
        None => ptx
            .find(&format!(".visible .entry {name}("))
            .unwrap_or_else(|| panic!("占位函数 {name} 不在 PTX 中（.func/.entry 均未找到）")),
    };
    let after = &ptx[start..];
    let close = after
        .find("\n)")
        .unwrap_or_else(|| panic!("{name} 签名缺收尾 ) 行"));
    let params: Vec<String> = after[..close]
        .lines()
        .filter_map(|l| {
            let t = l.trim();
            t.starts_with(".param ")
                .then(|| t.rsplit(' ').next().unwrap_or("").trim_end_matches(',').to_string())
        })
        .collect();
    assert_eq!(params.len(), n_params, "{name} 参数个数 {params:?}");
    let rest = &after[close..];
    let open = rest
        .find('{')
        .unwrap_or_else(|| panic!("{name} 缺函数体 {{"));
    let mut depth = 0usize;
    let mut end = 0usize;
    for (i, c) in rest[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = open + i;
                    break;
                }
            }
            _ => {}
        }
    }
    assert!(end > 0, "{name} 函数体花括号不匹配");
    // open/end 相对 rest，换算回 ptx 绝对偏移再拼接
    let (abs_open, abs_end) = (start + close + open, start + close + end);
    format!(
        "{}{{\n{}\n}}{}",
        &ptx[..abs_open],
        body(&params),
        &ptx[abs_end + 1..]
    )
}

// ═══ patch 之四：v9 手写 PTX attention kernel（FA2 形态）注入 ═══
//
// ldmatrix（LDSM）装载 + mma.sync m16n8k16(S)/m16n8k8(PV) + P 的 C→A
// fragment 寄存器直用（省 P smem 往返）+ K/V/Q 全动态 smem（101376B=99KB，
// 宿主 cuFuncSetAttribute opt-in）+ cp.async 双缓冲。几何同 v5：CTA=128
// 线程（4 warp × 16 行），grid=16qh×⌈tokens/64⌉，Bc=32（v5=16，容差档）。
//
// smem 布局（行距 528=512+16：16 对齐满足 cp.async 16B；mod 128=16 无
// bank 整周期）：Q [64][528] @0｜K [2][32][528] @33792｜V 同 @67584。
fn patch_v9(dst: &PathBuf) {
    println!("cargo:rerun-if-env-changed=WEIN_V9_PACK_SWAP");
    println!("cargo:rerun-if-env-changed=WEIN_V9_DBG");
    let ptx = std::fs::read_to_string(dst).expect("读 cp.async 改写后 PTX");
    // 模块级动态 smem 声明（插 .address_size 行后）
    let anchor = ptx.find(".address_size").expect("PTX 缺 .address_size");
    let line_end = anchor + ptx[anchor..].find('\n').unwrap() + 1;
    let out0 = format!(
        "{}\t.extern .shared .align 16 .b8 v9_smem[];\n{}",
        &ptx[..line_end],
        &ptx[line_end..]
    );
    let swap = env::var("WEIN_V9_PACK_SWAP").is_ok();
    let out = replace_cp_placeholder(&out0, "paged_attn_v9", 20, |p| v9_body(p, swap));
    for needle in [
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16",
        // K 以 [j][d]（=B^T）驻 smem：B 片段须非 trans；x2 两矩阵 = k 低/高 8 段
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16",
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16",
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
        "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32",
        "cp.async.wait_all",
        "cvta.shared.u64 \t%rd14, v9_smem",
    ] {
        assert!(out.contains(needle), "v9 注入断言失败：缺 {needle}");
    }
    assert!(!out.contains("st.volatile.b16"), "v9 占位体残留——绝不允许上线");
    assert_eq!(
        out.matches(".extern .shared .align 16 .b8 v9_smem").count(),
        1,
        "v9_smem 声明应恰一处"
    );
    std::fs::write(dst, out).expect("写 v9 改写后 PTX");
    if swap {
        println!("cargo:warning=v9：WEIN_V9_PACK_SWAP=1 —— cvt 打包半字序对调（fragment 布局兜底）");
    }
    println!("cargo:warning=v9 注入：LDSM + mma.sync + Bc=32 + 99KB 动态 smem");
}

/// K/V tile staging：128 线程各 8 个 16B 块（c = tid + i*128；tok = c>>5；
/// cb = (c&31)<<4）。page = bt[tile*2 + tok>>4]（16-token 页；tok>>4 =
/// i>>1 常量——tid<128 时 tok = warp+4i）。K 与 V 同形（V 无需转置：
/// ldmatrix.trans 现场转），只差 cache 指针与 smem 基址。
fn v9_stage_kv(o: &mut Vec<String>, tile: &str, base: &str, cache: &str) {
    for i in 0..8usize {
        o.push("\tmov.u32 \t%r36, %r0;".into());
        o.push(format!("\tadd.u32 \t%r36, %r36, {};", i * 128));
        o.push("\tshr.u32 \t%r37, %r36, 5;".into());
        o.push("\tand.b32 \t%r38, %r36, 31;".into());
        o.push("\tshl.b32 \t%r38, %r38, 4;".into());
        // page = bt[tile*2 + i>>2]（u32 数组：字节偏移 ×4）。
        // 每迭代 stage 4 行（tid+i*128 → r=i*4..i*4+3）：行 16-31 ⇒ i≥4 ⇒ 第 2 页。
        o.push(format!("\tshl.b32 \t%r39, {tile}, 1;"));
        o.push(format!("\tadd.u32 \t%r39, %r39, {};", i / 4));
        o.push("\tshl.b32 \t%r39, %r39, 2;".into());
        o.push("\tcvt.u64.u32 \t%rd17, %r39;".into());
        o.push("\tadd.u64 \t%rd17, %rd10, %rd17;".into());
        o.push("\tld.u32 \t%r40, [%rd17];".into());
        // src = cache + page*32768 + kvh*512 + (tok&15)*2048 + cb
        o.push("\tshl.b32 \t%r41, %r40, 15;".into());
        o.push("\tshl.b32 \t%r42, %r6, 9;".into());
        o.push("\tadd.u32 \t%r41, %r41, %r42;".into());
        o.push("\tand.b32 \t%r43, %r37, 15;".into());
        o.push("\tshl.b32 \t%r43, %r43, 11;".into());
        o.push("\tadd.u32 \t%r41, %r41, %r43;".into());
        o.push("\tadd.u32 \t%r41, %r41, %r38;".into());
        o.push("\tcvt.u64.u32 \t%rd18, %r41;".into());
        o.push(format!("\tadd.u64 \t%rd18, %rd18, {cache};"));
        // dst = base + tok*528 + cb
        o.push("\tmul.lo.u32 \t%r44, %r37, 528;".into());
        o.push("\tadd.u32 \t%r44, %r44, %r38;".into());
        o.push(format!("\tadd.u32 \t%r44, %r44, {base};"));
        o.push("\tcp.async.cg.shared.global \t[%r44], [%rd18], 16;".into());
    }
    o.push("\tcp.async.commit_group;".into());
}

/// v9 kernel 函数体（纯 PTX）。p = .entry 签名的 20 个 .param 名（顺序同
/// Rust 占位签名）。寄存器规划：
///   f0..127   O 累加（32 dim 块 × 4：d0/d1 行 g、d2/d3 行 g+8）
///   f128..143 S/p（4 j 子块 × 4）——pass1 掩码后存 S、pass2 原地存 p
///   f144..147 m_g/m_g8/l_g/l_g8；f148 q_scale；f150+ 临时
///   b0..3 Q 片段、b4/5 K、b8..11 V、b12..19 PV 的 A 打包、b20/21 尾写
///   r25 = smem 共享窗口基址；rd0..6 = 七指针（cvta 后 global）
fn v9_body(p: &[String], swap: bool) -> String {
    let mut o: Vec<String> = Vec::new();
    macro_rules! ln {
        ($($t:tt)*) => { o.push(format!($($t)*)); };
    }
    // cvt.rn.bf16x2.f32 半字序：默认第一操作数→高半字（lo=第二）；
    // fragment 元素 0（偶 k/偶 dim）应在低半字。SWAP=对调兜底。
    let pk = |hi: String, lo: String| -> String {
        if swap {
            format!("{}, {}", lo, hi)
        } else {
            format!("{}, {}", hi, lo)
        }
    };
    let f4 = |b: usize| format!("{{%f{}, %f{}, %f{}, %f{}}}", b, b + 1, b + 2, b + 3);

    o.push("\t.reg .pred \t%p<16>;".into());
    o.push("\t.reg .b32 \t%r<96>;".into());
    o.push("\t.reg .b64 \t%rd<32>;".into());
    o.push("\t.reg .b32 \t%b<24>;".into());
    o.push("\t.reg .f32 \t%f<240>;".into());

    // ── 参数与身份 ──
    for i in 0..7 {
        ln!("ld.param.b64 \t%rd{i}, [{}];", p[i]);
        ln!("cvta.to.global.u64 \t%rd{i}, %rd{i};");
    }
    ln!("ld.param.f32 \t%f148, [{}];", p[9]); // q_scale
    ln!("ld.param.u32 \t%r16, [{}];", p[10]); // bpr
    ln!("ld.param.u32 \t%r17, [{}];", p[15]); // tokens
    ln!("mov.u32 \t%r0, %tid.x;");
    ln!("and.b32 \t%r1, %r0, 31;");
    ln!("shr.b32 \t%r2, %r0, 5;"); // warp
    ln!("mov.u32 \t%r3, %ctaid.x;");
    ln!("and.b32 \t%r4, %r3, 15;"); // qh
    ln!("shr.b32 \t%r5, %r3, 4;");
    ln!("shl.b32 \t%r5, %r5, 6;"); // row0
    ln!("shr.b32 \t%r6, %r4, 2;"); // kvh
    ln!("shr.b32 \t%r7, %r1, 2;"); // g = lane/4
    ln!("and.b32 \t%r8, %r1, 3;");
    ln!("shl.b32 \t%r9, %r8, 1;"); // t2 = 2*(lane%4)
    ln!("sub.u32 \t%r10, %r17, 1;"); // last
    ln!("shl.b32 \t%r12, %r2, 4;");
    ln!("add.u32 \t%r12, %r5, %r12;"); // rbase = row0 + warp*16
    ln!("add.u32 \t%r14, %r12, %r7;"); // rbase+g（未钳）
    // seq = seq_ids[row0]（u32 数组：字节偏移 = idx*4）
    ln!("shl.b32 \t%r60, %r5, 2;");
    ln!("cvt.u64.u32 \t%rd7, %r60;");
    ln!("add.u64 \t%rd7, %rd4, %rd7;");
    ln!("ld.u32 \t%r11, [%rd7];");
    // bt = block_table + seq*bpr
    ln!("cvt.u64.u32 \t%rd8, %r11;");
    ln!("cvt.u64.u32 \t%rd9, %r16;");
    ln!("mul.wide.u32 \t%rd8, %r11, %r16;");
    ln!("add.u64 \t%rd10, %rd3, %rd8;");
    // pos_g / pos_g8 / max_pos（钳 last）
    ln!("min.u32 \t%r13, %r14, %r10;");
    ln!("add.u32 \t%r15, %r14, 8;");
    ln!("min.u32 \t%r15, %r15, %r10;");
    ln!("add.u32 \t%r18, %r5, 63;");
    ln!("min.u32 \t%r18, %r18, %r10;");
    ln!("shl.b32 \t%r61, %r13, 2;");
    ln!("cvt.u64.u32 \t%rd11, %r61;");
    ln!("add.u64 \t%rd11, %rd5, %rd11;");
    ln!("ld.u32 \t%r19, [%rd11];"); // pos_g
    ln!("shl.b32 \t%r62, %r15, 2;");
    ln!("cvt.u64.u32 \t%rd12, %r62;");
    ln!("add.u64 \t%rd12, %rd5, %rd12;");
    ln!("ld.u32 \t%r20, [%rd12];"); // pos_g8
    ln!("shl.b32 \t%r63, %r18, 2;");
    ln!("cvt.u64.u32 \t%rd13, %r63;");
    ln!("add.u64 \t%rd13, %rd5, %rd13;");
    ln!("ld.u32 \t%r21, [%rd13];"); // max_pos
    ln!("shr.u32 \t%r22, %r21, 5;");
    ln!("add.u32 \t%r22, %r22, 1;"); // n_tile（Bc=32 早停）
    ln!("mov.u32 \t%r23, 0;"); // tile
    ln!("mov.u32 \t%r24, 0;"); // j0（绝对）
    // smem 基址（共享窗口 u32）
    ln!("cvta.shared.u64 \t%rd14, v9_smem;");
    ln!("cvta.to.shared.u64 \t%rd15, %rd14;");
    ln!("cvt.u32.u64 \t%r25, %rd15;");

    // ── O/m/l 清零 ──
    for i in 0..128 {
        ln!("mov.f32 \t%f{i}, 0f00000000;");
    }
    ln!("mov.f32 \t%f144, 0fFF800000;"); // m_g = −inf
    ln!("mov.f32 \t%f145, 0fFF800000;");
    ln!("mov.f32 \t%f146, 0f00000000;"); // l
    ln!("mov.f32 \t%f147, 0f00000000;");

    // ── Q staging：64 行 × 512B（512/16=32 块/行；2048 块/128 线程=16）──
    for _i in 0..16 {
        ln!("mov.u32 \t%r30, %r0;");
        ln!("add.u32 \t%r30, %r30, {};", _i * 128);
        ln!("shr.u32 \t%r31, %r30, 5;"); // r（CTA 内行 0..63）
        ln!("and.b32 \t%r32, %r30, 31;");
        ln!("shl.b32 \t%r32, %r32, 4;"); // cb
        ln!("add.u32 \t%r33, %r5, %r31;");
        ln!("shl.b32 \t%r33, %r33, 13;"); // (row0+r)*8192
        ln!("shl.b32 \t%r34, %r4, 9;"); // qh*512
        ln!("add.u32 \t%r33, %r33, %r34;");
        ln!("add.u32 \t%r33, %r33, %r32;");
        ln!("cvt.u64.u32 \t%rd16, %r33;");
        ln!("add.u64 \t%rd16, %rd16, %rd0;");
        ln!("mul.lo.u32 \t%r35, %r31, 528;");
        ln!("add.u32 \t%r35, %r35, %r32;");
        ln!("add.u32 \t%r35, %r35, %r25;");
        ln!("cp.async.cg.shared.global \t[%r35], [%rd16], 16;");
    }
    ln!("cp.async.commit_group;");
    // tile 0 的 K/V
    ln!("add.u32 \t%r28, %r25, 33792;");
    v9_stage_kv(&mut o, "%r23", "%r28", "%rd1");
    ln!("add.u32 \t%r29, %r25, 67584;");
    v9_stage_kv(&mut o, "%r23", "%r29", "%rd2");

    // ── 主循环 ──
    ln!("$Lv9_loop:");
    ln!("setp.ge.u32 \t%p0, %r23, %r22;");
    ln!("@%p0 bra $Lv9_done;");
    ln!("cp.async.wait_all;");
    ln!("bar.sync 0;");
    ln!("add.u32 \t%r26, %r23, 1;");
    ln!("setp.lt.u32 \t%p1, %r26, %r22;");
    ln!("@!%p1 bra $Lv9_nostage;");
    ln!("and.b32 \t%r27, %r26, 1;");
    ln!("mul.lo.u32 \t%r27, %r27, 16896;");
    ln!("add.u32 \t%r28, %r25, 33792;");
    ln!("add.u32 \t%r28, %r28, %r27;");
    ln!("add.u32 \t%r29, %r25, 67584;");
    ln!("add.u32 \t%r29, %r29, %r27;");
    v9_stage_kv(&mut o, "%r26", "%r28", "%rd1");
    v9_stage_kv(&mut o, "%r26", "%r29", "%rd2");
    ln!("$Lv9_nostage:");
    // 计算缓冲基址（K=%r28 / V=%r29）
    ln!("and.b32 \t%r27, %r23, 1;");
    ln!("mul.lo.u32 \t%r27, %r27, 16896;");
    ln!("add.u32 \t%r28, %r25, 33792;");
    ln!("add.u32 \t%r28, %r28, %r27;");
    ln!("add.u32 \t%r29, %r25, 67584;");
    ln!("add.u32 \t%r29, %r29, %r27;");
    // S 相位寻址不变量
    ln!("and.b32 \t%r30, %r1, 7;");
    ln!("and.b32 \t%r31, %r1, 8;");
    ln!("add.u32 \t%r30, %r30, %r31;"); // Q 行
    ln!("mul.lo.u32 \t%r30, %r30, 528;");
    ln!("mul.lo.u32 \t%r31, %r2, 8448;"); // warp 16 行偏移
    ln!("shr.b32 \t%r32, %r1, 4;");
    ln!("shl.b32 \t%r32, %r32, 4;"); // (lane>>4)*16
    ln!("add.u32 \t%r33, %r25, %r31;");
    ln!("add.u32 \t%r33, %r33, %r30;");
    ln!("add.u32 \t%r33, %r33, %r32;"); // Q 基（+k*32 即得）
    ln!("and.b32 \t%r34, %r1, 7;");
    ln!("mul.lo.u32 \t%r34, %r34, 528;"); // K/V 行项
    ln!("and.b32 \t%r35, %r1, 8;");
    ln!("shl.b32 \t%r35, %r35, 1;"); // (lane&8)*2 = K k 高半列
    ln!("shr.b32 \t%r37, %r1, 3;");
    ln!("and.b32 \t%r37, %r37, 3;");
    ln!("shl.b32 \t%r37, %r37, 4;"); // ((lane>>3)&3)*16 = V 列项
    for i in 128..144 {
        ln!("mov.f32 \t%f{i}, 0f00000000;");
    }
    // ── S = Q·K^T：16 k × 4 ni（B=K^T x2.trans）──
    for k in 0..16 {
        ln!("add.u32 \t%r38, %r33, {};", k * 32);
        ln!("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%b0, %b1, %b2, %b3}}, [%r38];");
        for ni in 0..4 {
            ln!("add.u32 \t%r39, %r28, %r34;");
            ln!("add.u32 \t%r39, %r39, {};", ni * 4224);
            ln!("add.u32 \t%r40, %r35, {};", k * 32);
            ln!("add.u32 \t%r39, %r39, %r40;");
            ln!("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{%b4, %b5}}, [%r39];");
            let base = 128 + ni * 4;
            ln!(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {}, {{%b0, %b1, %b2, %b3}}, {{%b4, %b5}}, {};",
                f4(base),
                f4(base)
            );
        }
    }
    // ── [DBG] WEIN_V9_DBG=1：tile 0 的 16 个 S 片段 + 元数据裸写 out（f32）
    //    用于 fragment 布局对拍；每 lane 16 f32 连续（tid*64B 偏移）──
    if env::var("WEIN_V9_DBG").is_ok() {
        // tile 0 全 CTA 转储：base = out + ctaid*16KB + tid*128，互不覆盖、无尾写。
        ln!("setp.eq.u32 \t%p8, %r23, 0;");
        ln!("mul.lo.u32 \t%r70, %r0, 128;");
        ln!("mad.lo.u32 \t%r70, %r3, 16384, %r70;");
        ln!("cvt.u64.u32 \t%rd25, %r70;");
        ln!("add.u64 \t%rd25, %rd6, %rd25;");
        for ni in 0..4 {
            for j in 0..4 {
                ln!("add.u64 \t%rd26, %rd25, {};", (ni * 4 + j) * 4);
                ln!("@%p8 st.b32 [%rd26], %f{};", 128 + ni * 4 + j);
            }
        }
        // [DBG-探针] 槽 16=r23(恒0) 槽 17=r24(恒0)：隔离 store 通路 vs S 值
        ln!("add.u64 \t%rd26, %rd25, 64;");
        ln!("@%p8 st.u32 [%rd26], %r23;");
        ln!("add.u64 \t%rd26, %rd25, 68;");
        ln!("@%p8 st.u32 [%rd26], %r24;");
        for (slot, reg) in [(18usize, "%r19"), (19, "%r20"), (20, "%r21"), (21, "%r22")] {
            ln!("add.u64 \t%rd26, %rd25, {};", slot * 4);
            ln!("@%p8 st.u32 [%rd26], {reg};");
        }
        ln!("@%p8 bra $Lv9_dbg_out;");
    }
    // ── softmax pass1：掩码（可见 ⇔ pos ≥ 列）×q_scale → tile 行 max ──
    ln!("mov.f32 \t%f150, 0fFF800000;");
    ln!("mov.f32 \t%f151, 0fFF800000;");
    for ni in 0..4 {
        let (s0, s1, s2, s3) = (128 + ni * 4, 129 + ni * 4, 130 + ni * 4, 131 + ni * 4);
        ln!("add.u32 \t%r41, %r24, {};", ni * 8);
        ln!("add.u32 \t%r41, %r41, %r9;"); // c0
        ln!("add.u32 \t%r42, %r41, 1;"); // c1
        ln!("mul.f32 \t%f152, %f{s0}, %f148;");
        ln!("mul.f32 \t%f153, %f{s1}, %f148;");
        ln!("mul.f32 \t%f154, %f{s2}, %f148;");
        ln!("mul.f32 \t%f155, %f{s3}, %f148;");
        ln!("setp.ge.u32 \t%p2, %r19, %r41;");
        ln!("setp.ge.u32 \t%p3, %r19, %r42;");
        ln!("setp.ge.u32 \t%p4, %r20, %r41;");
        ln!("setp.ge.u32 \t%p5, %r20, %r42;");
        ln!("selp.f32 \t%f{s0}, %f152, 0fFF800000, %p2;");
        ln!("selp.f32 \t%f{s1}, %f153, 0fFF800000, %p3;");
        ln!("selp.f32 \t%f{s2}, %f154, 0fFF800000, %p4;");
        ln!("selp.f32 \t%f{s3}, %f155, 0fFF800000, %p5;");
        ln!("max.f32 \t%f150, %f150, %f{s0};");
        ln!("max.f32 \t%f150, %f150, %f{s1};");
        ln!("max.f32 \t%f151, %f151, %f{s2};");
        ln!("max.f32 \t%f151, %f151, %f{s3};");
    }
    // 行 max 归约（行 = lane/4 组内 4 lane：bfly xor1→max→xor2→max）
    // 注：shfl 之间必须穿插 max/add——纯 shfl 链是对角线直拷，不是归约。
    for (acc, tmp) in [("%f150", "%f158"), ("%f151", "%f159")] {
        ln!("mov.b32 \t%r43, {acc};");
        ln!("shfl.sync.bfly.b32 \t%r44, %r43, 1, 31, -1;");
        ln!("mov.b32 \t{tmp}, %r44;");
        ln!("max.f32 \t{acc}, {acc}, {tmp};");
        ln!("mov.b32 \t%r43, {acc};");
        ln!("shfl.sync.bfly.b32 \t%r44, %r43, 2, 31, -1;");
        ln!("mov.b32 \t{tmp}, %r44;");
        ln!("max.f32 \t{acc}, {acc}, {tmp};");
    }
    // mn / ag（首 tile：exp(−inf−mn)=0 ✓ 行至少 1 可见列）
    // 注：全掩 quad（行 g<6 时存在 q>g/2 的列组）quadmax=−inf，与 m_old=−inf 相减得
    // NaN——mn 钳到 −1e30 地板：p=ex2(−inf−(−1e30))=0、ag 同理 0，l 仍由可见列贡献。
    ln!("max.f32 \t%f152, %f144, %f150;");
    ln!("max.f32 \t%f153, %f145, %f151;");
    ln!("max.f32 \t%f152, %f152, 0fF149F2C1;");
    ln!("max.f32 \t%f153, %f153, 0fF149F2C1;");
    ln!("sub.f32 \t%f154, %f144, %f152;");
    ln!("mul.f32 \t%f154, %f154, 0f3FB8AA3B;");
    ln!("ex2.approx.f32 \t%f154, %f154;"); // ag
    ln!("sub.f32 \t%f155, %f145, %f153;");
    ln!("mul.f32 \t%f155, %f155, 0f3FB8AA3B;");
    ln!("ex2.approx.f32 \t%f155, %f155;"); // ag8
    for di in 0..32 {
        ln!("mul.f32 \t%f{0}, %f{0}, %f154;", di * 4);
        ln!("mul.f32 \t%f{0}, %f{0}, %f154;", di * 4 + 1);
        ln!("mul.f32 \t%f{0}, %f{0}, %f155;", di * 4 + 2);
        ln!("mul.f32 \t%f{0}, %f{0}, %f155;", di * 4 + 3);
    }
    // pass2：p = ex2((s−mn)·log2e)（全掩列 = 0）+ 行和
    ln!("mov.f32 \t%f156, 0f00000000;");
    ln!("mov.f32 \t%f157, 0f00000000;");
    for ni in 0..4 {
        let (s0, s1, s2, s3) = (128 + ni * 4, 129 + ni * 4, 130 + ni * 4, 131 + ni * 4);
        ln!("sub.f32 \t%f158, %f{s0}, %f152;");
        ln!("mul.f32 \t%f158, %f158, 0f3FB8AA3B;");
        ln!("ex2.approx.f32 \t%f{s0}, %f158;");
        ln!("sub.f32 \t%f158, %f{s1}, %f152;");
        ln!("mul.f32 \t%f158, %f158, 0f3FB8AA3B;");
        ln!("ex2.approx.f32 \t%f{s1}, %f158;");
        ln!("sub.f32 \t%f158, %f{s2}, %f153;");
        ln!("mul.f32 \t%f158, %f158, 0f3FB8AA3B;");
        ln!("ex2.approx.f32 \t%f{s2}, %f158;");
        ln!("sub.f32 \t%f158, %f{s3}, %f153;");
        ln!("mul.f32 \t%f158, %f158, 0f3FB8AA3B;");
        ln!("ex2.approx.f32 \t%f{s3}, %f158;");
        ln!("add.f32 \t%f156, %f156, %f{s0};");
        ln!("add.f32 \t%f156, %f156, %f{s1};");
        ln!("add.f32 \t%f157, %f157, %f{s2};");
        ln!("add.f32 \t%f157, %f157, %f{s3};");
    }
    for (acc, tmp) in [("%f156", "%f158"), ("%f157", "%f159")] {
        ln!("mov.b32 \t%r43, {acc};");
        ln!("shfl.sync.bfly.b32 \t%r44, %r43, 1, 31, -1;");
        ln!("mov.b32 \t{tmp}, %r44;");
        ln!("add.f32 \t{acc}, {acc}, {tmp};");
        ln!("mov.b32 \t%r43, {acc};");
        ln!("shfl.sync.bfly.b32 \t%r44, %r43, 2, 31, -1;");
        ln!("mov.b32 \t{tmp}, %r44;");
        ln!("add.f32 \t{acc}, {acc}, {tmp};");
    }
    ln!("mad.rn.f32 \t%f146, %f146, %f154, %f156;");
    ln!("mad.rn.f32 \t%f147, %f147, %f155, %f157;");
    ln!("mov.f32 \t%f144, %f152;");
    ln!("mov.f32 \t%f145, %f153;");
    // ── PV：A = p 打包（C→A 直用，省 P smem 往返）× B = V x4.trans ──
    for ni in 0..4 {
        let (ba, bb) = (12 + ni * 2, 13 + ni * 2);
        let (s0, s1, s2, s3) = (128 + ni * 4, 129 + ni * 4, 130 + ni * 4, 131 + ni * 4);
        ln!(
            "cvt.rn.bf16x2.f32 %b{ba}, {};",
            pk(format!("%f{s1}"), format!("%f{s0}"))
        );
        ln!(
            "cvt.rn.bf16x2.f32 %b{bb}, {};",
            pk(format!("%f{s3}"), format!("%f{s2}"))
        );
        for dj in 0..8 {
            ln!("add.u32 \t%r47, %r29, %r34;");
            ln!("add.u32 \t%r47, %r47, {};", ni * 4224);
            ln!("add.u32 \t%r48, %r37, {};", dj * 64);
            ln!("add.u32 \t%r47, %r47, %r48;");
            ln!("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{%b8, %b9, %b10, %b11}}, [%r47];");
            for j in 0..4 {
                let di = dj * 4 + j;
                ln!(
                    "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {}, {{%b{ba}, %b{bb}}}, {{%b{}}}, {};",
                    f4(di * 4),
                    8 + j,
                    f4(di * 4)
                );
            }
        }
    }
    ln!("add.u32 \t%r23, %r23, 1;");
    ln!("add.u32 \t%r24, %r24, 32;");
    ln!("bra $Lv9_loop;");
    ln!("$Lv9_done:");

    // ── 尾写：O/l → bf16（l>0：每行至少见列 0）──
    ln!("mov.f32 \t%f158, 0f3F800000;");
    ln!("div.rn.f32 \t%f159, %f158, %f146;");
    ln!("div.rn.f32 \t%f160, %f158, %f147;");
    ln!("setp.lt.u32 \t%p6, %r14, %r17;"); // rbase+g < tokens
    ln!("add.u32 \t%r49, %r14, 8;");
    ln!("setp.lt.u32 \t%p7, %r49, %r17;");
    // out 行基址（字节）：(rbase+g)*8192 + qh*512 + t2*2
    ln!("shl.b32 \t%r50, %r14, 13;");
    ln!("shl.b32 \t%r51, %r4, 9;");
    ln!("add.u32 \t%r50, %r50, %r51;");
    ln!("shl.b32 \t%r52, %r9, 1;");
    ln!("add.u32 \t%r50, %r50, %r52;");
    ln!("cvt.u64.u32 \t%rd19, %r50;");
    ln!("add.u64 \t%rd20, %rd6, %rd19;");
    ln!("shl.b32 \t%r53, %r49, 13;");
    ln!("add.u32 \t%r53, %r53, %r51;");
    ln!("add.u32 \t%r53, %r53, %r52;");
    ln!("cvt.u64.u32 \t%rd21, %r53;");
    ln!("add.u64 \t%rd22, %rd6, %rd21;");
    for di in 0..32 {
        let (d0, d1, d2, d3) = (di * 4, di * 4 + 1, di * 4 + 2, di * 4 + 3);
        ln!("mul.f32 \t%f161, %f{d0}, %f159;");
        ln!("mul.f32 \t%f162, %f{d1}, %f159;");
        ln!("cvt.rn.bf16x2.f32 %b20, {};", pk("%f162".into(), "%f161".into()));
        ln!("add.u64 \t%rd23, %rd20, {};", di * 16);
        ln!("@%p6 st.b32 [%rd23], %b20;");
        ln!("mul.f32 \t%f161, %f{d2}, %f160;");
        ln!("mul.f32 \t%f162, %f{d3}, %f160;");
        ln!("cvt.rn.bf16x2.f32 %b21, {};", pk("%f162".into(), "%f161".into()));
        ln!("add.u64 \t%rd24, %rd22, {};", di * 16);
        ln!("@%p7 st.b32 [%rd24], %b21;");
    }
    ln!("ret;");
    if env::var("WEIN_V9_DBG").is_ok() {
        ln!("$Lv9_dbg_out:");
        ln!("ret;");
    }
    o.join("\n")
}

/// 定位能跑到 nightly 的 cargo：
/// 1. ~/.cargo/bin/cargo（rustup shim，配 RUSTUP_TOOLCHAIN=nightly）
/// 2. ~/.rustup/toolchains/nightly-*/bin/cargo（直连，无需任何 env）
/// 3. 兜底 env CARGO（若它恰是 nightly 语境）
fn nightly_cargo() -> (PathBuf, bool) {
    let home = PathBuf::from(env::var("HOME").unwrap_or_default());
    let shim = home.join(".cargo/bin/cargo");
    if shim.exists() {
        return (shim, true); // shim：经 RUSTUP_TOOLCHAIN 选 nightly
    }
    if let Ok(rd) = std::fs::read_dir(home.join(".rustup/toolchains")) {
        for e in rd.flatten() {
            let c = e.path().join("bin/cargo");
            if e.file_name().to_string_lossy().starts_with("nightly") && c.exists() {
                return (c, false); // nightly 直连，无需 env
            }
        }
    }
    (PathBuf::from(env::var("CARGO").unwrap_or_else(|_| "cargo".into())), false)
}
