//! k7 的子进程载荷：加载 PTX 模块一次即退。
//! 单独成 bin 是因为集成测试二进制重入会再跑全部测试——探针必须是个
//! 干净的 main。父测试经 CARGO_BIN_EXE_jit-cache-probe 调起，经
//! CUDA_CACHE_PATH 控制缓存目录，用两次运行之间的耗时差证明缓存复用。
use std::time::Instant;

fn main() {
    let gpu = wein_kernel::cuda::Gpu::probe().expect("GPU 探测");
    let ptx =
        std::fs::read_to_string(std::env::var("WEIN_PTX").expect("父测试须注入 WEIN_PTX"))
            .expect("读 PTX");
    let t0 = Instant::now();
    let m = gpu.load_module(&ptx);
    let _f = m.func("vec_add_f32"); // 取函数句柄（模块真实可用的最小证明）
    println!("module-load={}ns arch={}", t0.elapsed().as_nanos(), gpu.arch());
}
