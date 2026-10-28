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
    std::fs::copy(&ptx[0], &dst).expect("拷贝 PTX 产物");
    println!("cargo:rustc-env=WEIN_PTX={}", dst.display());
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
