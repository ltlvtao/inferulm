//! norm 归约树探针：GPU vs reference 位对齐（多 hidden/rows）+ 单块单价（µs/发）。
//!
//! 用途：`block_sum_merge` 换树（2026-11-07）后，确认 (a) kernel↔reference 仍
//! 逐位同（o16 保持位级档，无需降容差）；(b) block-per-row 单行单价变化。
use wein_kernel::cuda::Gpu;
use wein_kernel::ops::{AddNormZcOp, OpTable, RmsNormZcOp};
use wein_kernel::reference;
use wein_platform::{Device, HardwareProfile};

fn main() {
    let gpu = Gpu::probe().expect("gpu");
    let t = OpTable::resolve(
        &gpu,
        &HardwareProfile::with_device(Device::Cuda).with_arch("sm_89"),
    )
    .expect("t");
    let stream = gpu.stream();

    // ── 1. 位对齐（全 1 整数和 + 随机向量；含窄块/半 warp/整 warp 三档） ──
    let mut bad = 0;
    for &(rows, hidden) in &[
        (1usize, 16usize),
        (1, 32),
        (1, 65),
        (3, 65),
        (1, 128),
        (1, 256),
        (1, 2560),
        (5, 2560),
    ] {
        for (tag, x) in [
            ("ones", vec![reference::f32_to_bf16(1.0); rows * hidden]),
            (
                "rand",
                (0..rows * hidden)
                    .map(|i| {
                        reference::f32_to_bf16(reference::splitmix_f32(i as u64 ^ 0x1001) * 3.0)
                    })
                    .collect::<Vec<u16>>(),
            ),
        ] {
            let w: Vec<u16> = vec![reference::f32_to_bf16(0.5); hidden];
            let (dx, dw, dout) = (
                gpu.alloc_from_slice(&x),
                gpu.alloc_from_slice(&w),
                gpu.alloc_bytes(x.len() * 2),
            );
            t.rms_norm_zc
                .run(&dx, &dw, &dout, rows as u32, hidden as u32, 1e-6, &stream);
            stream.sync();
            let got: Vec<u16> = dout.read_to_vec();
            let exp = reference::rms_norm_zc(&x, &w, rows, hidden, 1e-6);
            let nd = got.iter().zip(exp.iter()).filter(|(a, b)| a != b).count();
            if nd > 0 {
                bad += 1;
            }
            println!("rms_norm_zc {rows}x{hidden} {tag}: 位差 {nd}/{}", got.len());

            // add_norm_zc（a=x, b=x）与 add→rms_norm_zc 位级同（o36 的生产前提）
            let (db, dsum, dno) = (
                gpu.alloc_from_slice(&x),
                gpu.alloc_bytes(x.len() * 2),
                gpu.alloc_bytes(x.len() * 2),
            );
            t.add_norm_zc.run(
                &dx, &db, &dw, &dsum, &dno, rows as u32, hidden as u32, 1e-6, &stream,
            );
            stream.sync();
            let s: Vec<u16> = dsum.read_to_vec();
            let n: Vec<u16> = dno.read_to_vec();
            let es = reference::add(&x, &x);
            let en = reference::rms_norm_zc(&es, &w, rows, hidden, 1e-6);
            let nds = s.iter().zip(es.iter()).filter(|(a, b)| a != b).count();
            let ndn = n.iter().zip(en.iter()).filter(|(a, b)| a != b).count();
            if nds + ndn > 0 {
                bad += 1;
            }
            println!("add_norm_zc {rows}x{hidden} {tag}: sum 位差 {nds} norm 位差 {ndn}");
        }
    }
    println!("位对齐：{} 组越界", if bad == 0 { 0 } else { bad });

    // ── 2. 单价（block-per-row，生产 hidden=2560） ──
    let (rows, hidden) = (1usize, 2560usize);
    let x = vec![reference::f32_to_bf16(1.0); rows * hidden];
    let w = vec![reference::f32_to_bf16(0.5); hidden];
    let (dx, dw, dout) = (
        gpu.alloc_from_slice(&x),
        gpu.alloc_from_slice(&w),
        gpu.alloc_bytes(x.len() * 2),
    );
    let (dsum, dno) = (gpu.alloc_bytes(x.len() * 2), gpu.alloc_bytes(x.len() * 2));
    // 注：闭包借用 t/stream，分两次调用会冲突——各自内联一份
    {
        let mut f = || {
            t.rms_norm_zc
                .run(&dx, &dw, &dout, rows as u32, hidden as u32, 1e-6, &stream)
        };
        for _ in 0..50 {
            f();
        }
        stream.sync();
        let (e0, e1) = (gpu.event(), gpu.event());
        e0.record_on(&stream);
        for _ in 0..500 {
            f();
        }
        e1.record_on(&stream);
        stream.sync();
        println!(
            "rms_norm_zc 1x2560: {:.2} µs/发",
            e1.elapsed_ms(&e0) * 1000.0 / 500.0
        );
    }
    {
        let mut f = || {
            t.add_norm_zc.run(
                &dx, &dx, &dw, &dsum, &dno, rows as u32, hidden as u32, 1e-6, &stream,
            )
        };
        for _ in 0..50 {
            f();
        }
        stream.sync();
        let (e0, e1) = (gpu.event(), gpu.event());
        e0.record_on(&stream);
        for _ in 0..500 {
            f();
        }
        e1.record_on(&stream);
        stream.sync();
        println!(
            "add_norm_zc 1x2560: {:.2} µs/发",
            e1.elapsed_ms(&e0) * 1000.0 / 500.0
        );
    }
}
