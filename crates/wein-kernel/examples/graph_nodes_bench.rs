//! 图回放宿主开销 vs 节点数（诊断探针）。
//!
//!   cargo run --release -p wein-kernel --example graph_nodes_bench
//!
//! 动机：图档服务步的 `cuGraphLaunch` 实测占宿主 ~0.5ms/步（WEIN_STEP_PROF2）。
//! 若开销与**节点数**成正比（CUDA 在 launch 时逐节点上传参数），则唯一的
//! 修法是降低每步 kernel 数（融合），而不是换 launch 姿势；若与节点数无关，
//! 说明是别的固定开销（实例化/流序），可另行处置。
//!
//! 做法：捕获 N 张 `add` 节点（同一缓冲、同参数，capture 仍各记一节点），
//! 量 `launch`（纯宿主提交）与 `launch+sync`（含执行）的均值。
use std::time::Instant;

use wein_kernel::cuda::Gpu;
use wein_kernel::ops::{AddOp, GemmOp, OpTable, RmsNormZcOp};
use wein_platform::{Device, HardwareProfile};

fn main() {
    let gpu = Gpu::probe().expect("须有 CUDA 设备");
    let profile = HardwareProfile::with_device(Device::Cuda).with_arch("sm_89");
    let t = OpTable::resolve(&gpu, &profile).expect("sm_89 画像须解析成功");
    let stream = gpu.stream();
    let n = 1024usize;
    let a = gpu.alloc_from_slice(&vec![1.0f32; n]);
    let b = gpu.alloc_from_slice(&vec![2.0f32; n]);
    let c = gpu.alloc_bytes(n * 4);
    println!("节点数 | 阶段     | launch µs | 每节点 ns");
    for nodes in [1usize, 16, 64, 200, 400, 700] {
        stream.begin_capture();
        // 每节点参数互不相同（长度微调）——避免驱动把同参数节点当成同一个
        for i in 0..nodes {
            t.add.run(&a, &b, &c, (n - (i % 8) * 8) as u32, &stream);
        }
        let exec = stream.end_capture().instantiate();
        // 单发计时：**每发前先 sync**（GPU 空闲态），否则同 exec 连续 launch
        // 会被驱动反压串起来，量到的是排队而非宿主提交开销
        let bench = |tag: &str| {
            exec.launch(&stream);
            stream.sync();
            let mut v = Vec::new();
            for _ in 0..40 {
                stream.sync();
                let t = Instant::now();
                exec.launch(&stream);
                v.push(t.elapsed().as_secs_f64() * 1e6);
            }
            stream.sync();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = v[v.len() / 2];
            println!("{nodes:6} | {tag:8} | {med:9.2} | {:8.1}", med * 1e3 / nodes as f64);
            med
        };
        bench("upload前");
        exec.upload(&stream);
        stream.sync();
        let after = bench("upload后");
        let _ = after;
        drop(exec);
        stream.sync();
    }

    // 第二段：真实 kernel（rms_norm_zc，7 参数/block-per-row）——判"每节点
    // 0.71µs"是参数体量/真实 kernel 的固有开销，还是 add 那个 4 参数特例
    let (rows, hidden) = (128u32, 2560u32);
    let x = gpu.alloc_from_slice(&vec![0x3c00u16; rows as usize * hidden as usize]);
    let wv = gpu.alloc_from_slice(&vec![0x3c00u16; hidden as usize]);
    let o = gpu.alloc_bytes(rows as usize * hidden as usize * 2);
    println!("--- rms_norm_zc（真实 kernel）");
    for nodes in [200usize, 700] {
        stream.begin_capture();
        for i in 0..nodes {
            t.rms_norm_zc.run(&x, &wv, &o, hidden - ((i % 4) as u32) * 8, rows, 1e-6, &stream);
        }
        let exec = stream.end_capture().instantiate();
        exec.launch(&stream);
        stream.sync();
        let mut v = Vec::new();
        for _ in 0..40 {
            stream.sync();
            let t0 = Instant::now();
            exec.launch(&stream);
            v.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        stream.sync();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = v[v.len() / 2];
        println!("{nodes:6} | 真实kernel | {med:9.2} | {:8.1}", med * 1e3 / nodes as f64);
        drop(exec);
        stream.sync();
    }

    // 第三段：cuBLASLt gemm 节点（真模型图里 223/703 是它）——若这类节点
    // 单价远高于自写 kernel，则"每节点 0.71µs"的真凶是 Lt，杠杆=合并投影
    let (gm, gn, gk) = (100u32, 2560u32, 2560u32);
    let ga = gpu.alloc_from_slice(&vec![0x3c00u16; (gm * gk) as usize]);
    let gb = gpu.alloc_from_slice(&vec![0x3c00u16; (gn * gk) as usize]);
    let gc = gpu.alloc_bytes((gm * gn) as usize * 2);
    // 第四段：memset / memcpy 节点（真模型图里 68 memset + 42 memcpy/步，
    // 全来自 cuBLASLt 工作区与 radix 状态快照）——图里这类节点是否比
    // kernel 节点贵一个量级？这是"700 节点 500µs"的最后一块拼图
    let mbuf = gpu.alloc_bytes(1 << 20);
    println!("--- memset / memcpy 节点");
    for (tag, nodes, is_memset) in [("memset节点", 200usize, true), ("memcpy节点", 200usize, false)] {
        let src = gpu.alloc_bytes(1 << 20);
        stream.begin_capture();
        for _ in 0..nodes {
            if is_memset {
                mbuf.zero_range_on(0, 4096, &stream);
            } else {
                mbuf.copy_range_on(0, &src, 0, 4096, &stream);
            }
        }
        let exec = stream.end_capture().instantiate();
        exec.launch(&stream);
        stream.sync();
        let mut v = Vec::new();
        for _ in 0..30 {
            stream.sync();
            let t0 = Instant::now();
            exec.launch(&stream);
            v.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        stream.sync();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = v[v.len() / 2];
        println!("{nodes:6} | {tag} | {med:9.2} | {:8.1}", med * 1e3 / nodes as f64);
        drop(exec);
        stream.sync();
    }

    println!("--- GemmLt run_wt {gm}x{gn}x{gk}");
    for nodes in [100usize, 200] {
        stream.begin_capture();
        for _ in 0..nodes {
            t.gemm.run_wt(&ga, &gb, &gc, gm, gn, gk, &stream);
        }
        let exec = stream.end_capture().instantiate();
        exec.launch(&stream);
        stream.sync();
        let mut v = Vec::new();
        for _ in 0..30 {
            stream.sync();
            let t0 = Instant::now();
            exec.launch(&stream);
            v.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        stream.sync();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = v[v.len() / 2];
        println!("{nodes:6} | gemm节点   | {med:9.2} | {:8.1}", med * 1e3 / nodes as f64);
        drop(exec);
        stream.sync();
    }
}
