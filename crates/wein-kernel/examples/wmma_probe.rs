//! wmma 能力探针：llvm.nvvm.wmma.* intrinsic 能否穿纯 Rust→PTX 管线。
//!
//!   cargo run --release -p wein-kernel --example wmma_probe
//!
//! 验证三件事：① intrinsic 解析+codegen（LLVM 签名对上）
//! ② m16n16k16 bf16×bf16→f32 数值对拍 CPU golden（容差）
//! ③ PTX 出 wmma 指令（tensor core HMMA 证据）。
use wein_kernel::cuda::Gpu;
use wein_kernel::reference::{bf16_to_f32, f32_to_bf16};

fn main() {
    let gpu = Gpu::probe().expect("须有 CUDA 设备");
    let ptx = std::fs::read_to_string(wein_kernel::ptx_path()).expect("PTX 产物");
    let n = ptx.matches("wmma").count();
    let hmma = ptx.matches("hmma").count();
    println!("PTX 内 wmma 指令数：{n}（hmma 命中 {hmma}）");
    if n == 0 {
        println!("结论：wmma intrinsic 未落 PTX——管线不通，v4 tensor core 路线需另寻入口");
        return;
    }
    let m = gpu.load_module(&ptx);
    let f = m.func("wmma_probe");
    let stream = gpu.stream();

    // A row-major 16×16，B_math 的转置摆放（col-major 视图，b[n*16+k]）
    let mut a = vec![0u16; 256];
    let mut b = vec![0u16; 256];
    let mut c_golden = vec![0f32; 256];
    for i in 0..16 {
        for j in 0..16 {
            a[i * 16 + j] = f32_to_bf16(((i * 7 + j * 3) % 13) as f32 * 0.25 - 1.0);
        }
    }
    let mut bbits = vec![0u16; 256];
    for k in 0..16 {
        for j in 0..16 {
            let v = ((k * 5 + j * 11) % 17) as f32 * 0.2 - 1.5;
            bbits[k * 16 + j] = f32_to_bf16(v);
            b[j * 16 + k] = bbits[k * 16 + j]; // 转置摆放
        }
    }
    for i in 0..16 {
        for j in 0..16 {
            let mut s = 0f32;
            for k in 0..16 {
                s += bf16_to_f32(a[i * 16 + k]) * bf16_to_f32(bbits[k * 16 + j]);
            }
            c_golden[i * 16 + j] = s;
        }
    }

    let da = gpu.alloc_from_slice(&a);
    let db = gpu.alloc_from_slice(&b);
    let dc = gpu.alloc_bytes(1024);
    let args = wein_kernel::kernel_args![da, db, dc];
    gpu.launch(&f, 1, 32, &args, &stream);
    stream.sync();

    let hc = dc.read_to_vec::<f32>();
    let mut worst = 0f32;
    let mut bad = 0usize;
    for i in 0..256 {
        let g = c_golden[i];
        let d = 1.0f32.max(g.abs());
        let err = (hc[i] - g).abs() / d;
        if err > worst {
            worst = err;
        }
        if err > 1e-2 {
            bad += 1;
        }
    }
    println!("C[0][:4] gpu={:?} golden={:?}", &hc[..4], &c_golden[..4]);
    println!("最坏相对误差 {worst:.3e}（超 1e-2 的元素 {bad}/256）");
    if bad == 0 {
        println!("结论：wmma 可用——tensor core 乘子解锁，v4 FA2 路线开工");
    } else {
        println!("结论：wmma 数值异常，需核对 fragment 布局/签名");
    }
}
