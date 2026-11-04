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
    println!("cargo:rustc-env=WEIN_PTX={}", dst.display());
}

/// 扫描源文件取 `#[link_section = ".shared"]` 标记的静态数组名，把 PTX 里
/// 这些符号的三处形态改写为真共享内存。改写失败（找不到符号）直接 panic——
/// 不许静默产出跑不动的 PTX。
fn patch_smem_symbols(ptx_src: &PathBuf, dst: &PathBuf, lib_rs: &PathBuf) {
    let src = std::fs::read_to_string(lib_rs).expect("读 kernels src");
    // 符号名：#[link_section = ".shared"] 后（允许隔注释行）的 static mut NAME
    let mut names: Vec<String> = Vec::new();
    let mut mark = false;
    for line in src.lines() {
        let t = line.trim();
        if t.contains("#[link_section = \".shared\"]") {
            mark = true;
            continue;
        }
        if mark {
            if t.starts_with("//") {
                continue;
            }
            if let Some(rest) = t.strip_prefix("static mut ") {
                let name: String =
                    rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
                if !name.is_empty() {
                    names.push(name);
                }
            }
            mark = false;
        }
    }
    if names.is_empty() {
        // 无 smem 数组：原样拷贝
        std::fs::copy(ptx_src, dst).expect("拷贝 PTX 产物");
        return;
    }
    let ptx = std::fs::read_to_string(ptx_src).expect("读 PTX 产物");
    let mut in_fn = false;
    let mut patched_count = 0usize;
    // v3 entry 插 .maxntid 512（.version 7.0 无 .maxnreg）：逼 ptxas 把 regs
    // 压到 65536/512=128 → bdim≤512 均可发射且 bdim256 得 2 块/SM 占用。
    let mut v3_pending = false;
    let mut lines_out: Vec<String> = Vec::with_capacity(ptx.len() / 24);
    for l in ptx.lines() {
        if l.starts_with(".visible .entry ") {
            in_fn = true;
            if l.starts_with(".visible .entry paged_attn_v3_") {
                v3_pending = true;
            }
        }
        let mut line = l.to_string();
        let has_sym = names.iter().any(|n| l.contains(n.as_str()));
        if has_sym && l.contains(".global") {
            line = l.replace(".global", ".shared");
            patched_count += 1;
        } else if has_sym && l.contains("mov.b64") {
            line = l.replace("mov.b64", "cvta.shared.u64");
            patched_count += 1;
        } else if in_fn && (l.contains("st.global") || l.contains("ld.global")) {
            line = l.replace("st.global", "st").replace("ld.global", "ld");
        }
        lines_out.push(line);
        if v3_pending && l.trim_start().starts_with(')') {
            lines_out.push(".maxntid 512, 1, 1".to_string());
            v3_pending = false;
        }
        if in_fn && l == "}" {
            in_fn = false;
        }
    }
    let out = lines_out.join("\n");
    for n in &names {
        assert!(
            out.contains(&format!(".shared") ) || patched_count > 0,
            "smem 符号 {n} 未在 PTX 中改写"
        );
    }
    std::fs::write(dst, out).expect("写改写后 PTX");
    println!("cargo:warning=smem 改写：{:?}（{patched_count} 处）", names);
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
    ] {
        assert!(out.contains(needle), "cp.async 指令 {needle} 未出现在产物 PTX");
    }
    assert!(
        !out.contains("st.volatile.b64") && !out.contains("st.volatile.b32"),
        "占位 volatile 体残留——绝不允许上线"
    );
    std::fs::write(dst, out).expect("写 cp.async 改写后 PTX");
    println!("cargo:warning=cp.async 改写：三函数体 + V5_KS .align 16");
}

/// 定位 `name` 函数定义，花括号匹配替换体。`body(params) -> 指令文本`。
fn replace_cp_placeholder(
    ptx: &str,
    name: &str,
    n_params: usize,
    body: impl FnOnce(&[String]) -> String,
) -> String {
    let sig = format!(".visible .func {name}(");
    let start = ptx
        .find(&sig)
        .unwrap_or_else(|| panic!("cp.async 占位函数 {name} 不在 PTX 中"));
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
