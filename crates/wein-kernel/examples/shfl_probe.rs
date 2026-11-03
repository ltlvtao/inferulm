//! shfl 能力探针：llvm.nvvm intrinsic link_name 旁路能否穿过
//! 纯 Rust→PTX 管线（stdarch nvptx 无 shfl）。
//!
//!   cargo run --release -p wein-kernel --example shfl_probe
//!
//! 验证两件事：① `llvm.nvvm.shfl.sync.down.i32`（i32 归约链）
//! ② `llvm.nvvm.shfl.sync.idx.f32`（f32 广播）。1 block × 32 线程，
//! 全 lane 应得 Σ(1..32)=528 与 3.5。
use wein_kernel::cuda::Gpu;

fn main() {
    let gpu = Gpu::probe().expect("须有 CUDA 设备");
    let ptx = std::fs::read_to_string(wein_kernel::ptx_path()).expect("PTX 产物");
    // 证据留档：PTX 文本里应有 shfl.sync 指令
    let n = ptx.matches("shfl.sync").count();
    println!("PTX 内 shfl.sync 指令数：{n}");
    let m = gpu.load_module(&ptx);
    let f = m.func("shfl_probe");
    let stream = gpu.stream();

    let out_i = gpu.alloc_bytes(32 * 4);
    let out_f = gpu.alloc_bytes(32 * 4);
    let args = wein_kernel::kernel_args![out_i, out_f];
    gpu.launch(&f, 1, 32, &args, &stream);
    stream.sync();

    let hi = out_i.read_to_vec::<i32>();
    let hf = out_f.read_to_vec::<f32>();
    let ok_i = hi[0] == 528; // shfl.down：lane 0 拿全和
    let ok_f = hf.iter().all(|&x| x == 3.5 + 528.0); // idx 广播 + bfly 全 lane 同和
    println!("shfl.down i32 归约：{:?}（lane0 全和 528={ok_i}）", &hi[..4]);
    println!("shfl.idx 广播 + bfly 全同和：{:?}（全 531.5={ok_f}）", &hf[..4]);
    if n > 0 && ok_i && ok_f {
        println!("结论：shfl 可用——warp 级归约/广播解锁");
    } else {
        println!("结论：shfl 不可用（n={n}, ok_i={ok_i}, ok_f={ok_f}）");
    }

    // ── smem：build.rs 已把 #[link_section=".shared"] 符号改写进产物 ──
    let nsh = ptx.matches("SMEM_PROBE").count();
    let decl = ptx.lines().find(|l| l.contains("SMEM_PROBE") && l.contains(".shared"));
    println!("PTX 内 SMEM_PROBE .shared 声明：{:?}", decl.unwrap_or("<无>"));
    let m2 = gpu.load_module(&ptx);
    let fs = m2.func("smem_probe");
    let out = gpu.alloc_bytes(64 * 4);
    let args = wein_kernel::kernel_args![out];
    let st = gpu.stream();
    gpu.launch(&fs, 1, 64, &args, &st);
    st.sync();
    let v = out.read_to_vec::<u32>();
    let ok_s = (0..64u32).all(|t| v[t as usize] == (t ^ 1));
    println!("smem 探针（{nsh} 处引用）：out[4..8]={:?}（全 t^1={ok_s}）", &v[4..8]);
    if ok_s && decl.is_some() {
        println!("结论：smem 经 build.rs PTX 改写可达");
    } else {
        println!("结论：smem 改写失败");
    }
}
