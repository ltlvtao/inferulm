//! cuBLASLt 最小 FFI（§4：标准算子走厂商库；融合算子手写的另一侧）。
//!
//! 与 driver.rs 同法：dlopen + 符号链（无 toolkit 构建依赖，库缺 = 启动期
//! Degrade，不带病运行）。数值合同：bf16 入出 + CUBLAS_COMPUTE_32F
//!（tensor core，f32 累加）——累加序由库决定，对齐标准是容差（g2 准则）
//! 而非位对齐（厂商库合同，与自控 kernel 的本质区别）。
//!
//! 形状解析纪律：每形状 (m,n,k) 首次调用做 desc/layout/heuristic 解析并
//! 缓存，其后纯 launch——"启动时解析"精神落到形状粒度（M 随 batch 变，
//! 模型投影的 n/k 固定，缓存命中即稳态）。
//!
//! 库获取（运行时依赖，非构建依赖）：pip wheel `nvidia-cublas-cu12` 提取
//! `libcublasLt.so.12` → 仓内 `vendor/`（gitignore）；候选链含
//! `WEIN_CUBLASLT` 环境变量覆盖。
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;

use libloading::Library;

use super::{DevBuf, Gpu, Stream};

// ── 常量（cublasLt 头文件原值；g2 精度验收是对这些值的经验校验） ──
const CUDA_R_32F: i32 = 0;
const CUDA_R_16BF: i32 = 14;
const CUBLAS_COMPUTE_32F: i32 = 68;
const LAYOUT_ORDER: i32 = 1; // cublasLtMatrixLayoutAttributes_t::ORDER
const ORDER_ROW: u32 = 1; // cublasLtOrder_t::ROW
const PREF_MAX_WORKSPACE: i32 = 1; // MatmulPreference::MAX_WORKSPACE_BYTES
const WS_BYTES: usize = 32 << 20;

/// 启发式结果（cublasLtMatmulHeuristicResult_t，repr(C) 对齐 ABI）
#[derive(Clone, Copy)]
#[repr(C)]
struct HeuristicResult {
    algo: [u64; 8], // cublasLtMatmulAlgoT 不透明体
    workspace_size: usize,
    state: u32,
    waves_count: f32,
    reserved: [u32; 4],
}

/// 解析产物全字段 Copy——命中路径值返回，免 RefCell 借用纠缠
#[derive(Clone, Copy)]
struct ShapePlan {
    desc: *mut c_void,
    lay_a: *mut c_void,
    lay_b: *mut c_void,
    lay_c: *mut c_void,
    algo: [u64; 8],
}

/// 符号面（裸 fn 指针——Library 钉在结构体字段上保活）
#[allow(clippy::type_complexity)]
struct LtFns {
    create: unsafe extern "C" fn(*mut *mut c_void) -> u32,
    version: unsafe extern "C" fn() -> usize,
    desc_create: unsafe extern "C" fn(*mut *mut c_void, i32, i32) -> u32,
    layout_create: unsafe extern "C" fn(*mut *mut c_void, i32, u64, u64, i64) -> u32,
    layout_set: unsafe extern "C" fn(*mut c_void, i32, *const c_void, usize) -> u32,
    pref_create: unsafe extern "C" fn(*mut *mut c_void) -> u32,
    pref_set: unsafe extern "C" fn(*mut c_void, i32, *const c_void, usize) -> u32,
    heuristic: unsafe extern "C" fn(
        *mut c_void, *mut c_void, *mut c_void, *mut c_void, *mut c_void, *mut c_void,
        *mut c_void, i32, *mut HeuristicResult, *mut i32,
    ) -> u32,
    matmul: unsafe extern "C" fn(
        *mut c_void, *mut c_void, *const c_void, *const c_void, *mut c_void,
        *const c_void, *mut c_void, *const c_void, *const c_void, *mut c_void,
        *mut c_void, *mut c_void, *const c_void, *mut c_void, usize, *mut c_void,
    ) -> u32,
}

/// bf16 GEMM：C[M,N] = A[M,K]·B[K,N]（行主序三方——layout 带 ORDER_ROW）
pub struct BlasLt<'a> {
    _lib: Library,
    gpu: &'a Gpu,
    ws: DevBuf<'a>,
    fns: LtFns,
    handle: *mut c_void,
    /// 形状 → 解析产物（desc/layouts/algo）；命中即纯 launch
    cache: RefCell<HashMap<(u32, u32, u32), ShapePlan>>,
}

macro_rules! sym {
    ($lib:expr, $name:literal, $ty:ty) => {
        *$lib.get::<$ty>($name.as_bytes()).unwrap_or_else(|e| panic!("符号 {} 缺失：{e}", $name))
    };
}

impl BlasLt<'_> {
    /// dlopen 候选链：env 覆盖 → 常规 soname → 仓内 vendor（wheel 提取）
    pub fn open(gpu: &Gpu) -> Result<BlasLt<'_>, String> {
        let candidates: Vec<String> = [
            std::env::var("WEIN_CUBLASLT").ok(),
            Some("libcublasLt.so.12".into()),
            Some("libcublasLt.so".into()),
            Some(concat!(env!("CARGO_MANIFEST_DIR"), "/../../vendor/libcublasLt.so.12").into()),
        ]
        .into_iter()
        .flatten()
        .collect();
        let mut last = String::new();
        for c in &candidates {
            match unsafe { Library::new(c) } {
                Ok(lib) => return Self::init(gpu, lib).map_err(|e| format!("{c}: {e}")),
                Err(e) => last = format!("{c}: {e}"),
            }
        }
        Err(format!("全部候选失败（最后：{last}）；可设 WEIN_CUBLASLT 指定路径"))
    }

    fn init(gpu: &Gpu, lib: Library) -> Result<BlasLt<'_>, String> {
        unsafe {
            let fns = LtFns {
                create: sym!(lib, "cublasLtCreate", unsafe extern "C" fn(*mut *mut c_void) -> u32),
                version: sym!(lib, "cublasLtGetVersion", unsafe extern "C" fn() -> usize),
                desc_create: sym!(lib, "cublasLtMatmulDescCreate", unsafe extern "C" fn(*mut *mut c_void, i32, i32) -> u32),
                layout_create: sym!(lib, "cublasLtMatrixLayoutCreate", unsafe extern "C" fn(*mut *mut c_void, i32, u64, u64, i64) -> u32),
                layout_set: sym!(lib, "cublasLtMatrixLayoutSetAttribute", unsafe extern "C" fn(*mut c_void, i32, *const c_void, usize) -> u32),
                pref_create: sym!(lib, "cublasLtMatmulPreferenceCreate", unsafe extern "C" fn(*mut *mut c_void) -> u32),
                pref_set: sym!(lib, "cublasLtMatmulPreferenceSetAttribute", unsafe extern "C" fn(*mut c_void, i32, *const c_void, usize) -> u32),
                heuristic: sym!(lib, "cublasLtMatmulAlgoGetHeuristic", unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void, *mut c_void, *mut c_void, *mut c_void, i32, *mut HeuristicResult, *mut i32) -> u32),
                matmul: sym!(lib, "cublasLtMatmul", unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void, *const c_void, *mut c_void, *const c_void, *mut c_void, *const c_void, *const c_void, *mut c_void, *mut c_void, *mut c_void, *const c_void, *mut c_void, usize, *mut c_void) -> u32),
            };
            let mut handle = std::ptr::null_mut();
            let st = (fns.create)(&mut handle);
            if st != 0 {
                return Err(format!("cublasLtCreate 失败：status {st}"));
            }
            Ok(BlasLt {
                _lib: lib,
                gpu,
                ws: gpu.alloc_bytes(WS_BYTES),
                fns,
                handle,
                cache: RefCell::new(HashMap::new()),
            })
        }
    }

    pub fn version(&self) -> u64 {
        unsafe { (self.fns.version)() as u64 }
    }

    /// C = A·B（bf16 入出，f32 累加）。首次形状走 heuristic 解析并缓存
    pub fn gemm(&self, a: &DevBuf, b: &DevBuf, c: &DevBuf, m: u32, n: u32, k: u32, stream: &Stream) {
        let plan = self.plan(m, n, k);
        let (alpha, beta) = (1.0f32, 0.0f32);
        let st = unsafe {
            (self.fns.matmul)(
                self.handle,
                plan.desc,
                &alpha as *const f32 as *const c_void,
                a.device_ptr() as *const c_void,
                plan.lay_a,
                b.device_ptr() as *const c_void,
                plan.lay_b,
                &beta as *const f32 as *const c_void,
                c.device_ptr() as *const c_void,
                plan.lay_c,
                c.device_ptr() as *mut c_void,
                plan.lay_c,
                &plan.algo as *const [u64; 8] as *const c_void,
                self.ws.device_ptr() as *mut c_void,
                WS_BYTES,
                stream.handle,
            )
        };
        assert_eq!(st, 0, "cublasLtMatmul 失败：status {st}（{m}×{n}×{k}）");
    }

    /// 形状解析：desc + 三 layout（ROW_ORDER）+ preference(工作区) +
    /// heuristic 首个可用 algo——全部一次性，缓存复用
    fn plan(&self, m: u32, n: u32, k: u32) -> ShapePlan {
        if let Some(p) = self.cache.borrow().get(&(m, n, k)) {
            return *p;
        }
        let plan = self.resolve_plan(m, n, k);
        self.cache.borrow_mut().insert((m, n, k), plan);
        plan
    }

    fn resolve_plan(&self, m: u32, n: u32, k: u32) -> ShapePlan {
        unsafe {
            let f = &self.fns;
            // desc：compute 32F + scale f32；opA/opB 默认 N（不显式设）
            let mut desc = std::ptr::null_mut();
            assert_eq!((f.desc_create)(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), 0, "desc_create");
            let mk_layout = |rows: u64, cols: u64, ld: i64| -> *mut c_void {
                let mut lay = std::ptr::null_mut();
                assert_eq!((f.layout_create)(&mut lay, CUDA_R_16BF, rows, cols, ld), 0, "layout_create");
                assert_eq!((f.layout_set)(lay, LAYOUT_ORDER, &ORDER_ROW as *const u32 as *const c_void, 4), 0, "layout ORDER=ROW");
                lay
            };
            let (lay_a, lay_b, lay_c) = (mk_layout(m as u64, k as u64, k as i64),
                mk_layout(k as u64, n as u64, n as i64), mk_layout(m as u64, n as u64, n as i64));
            let mut pref = std::ptr::null_mut();
            assert_eq!((f.pref_create)(&mut pref), 0, "pref_create");
            let ws = WS_BYTES as u64;
            assert_eq!((f.pref_set)(pref, PREF_MAX_WORKSPACE, &ws as *const u64 as *const c_void, 8), 0, "pref_set");
            let mut results = [HeuristicResult { algo: [0; 8], workspace_size: 0, state: 1, waves_count: 0.0, reserved: [0; 4] }; 8];
            let mut returned = 0i32;
            let st = (f.heuristic)(self.handle, desc, lay_a, lay_b, lay_c, lay_c, pref,
                results.len() as i32, results.as_mut_ptr(), &mut returned);
            assert_eq!(st, 0, "heuristic status {st}（{m}×{n}×{k}）");
            assert!(returned > 0 && results[0].state == 0, "无可用 algo（returned={returned}）");
            assert!(results[0].workspace_size <= WS_BYTES, "algo 工作区 {} 超预算", results[0].workspace_size);
            ShapePlan { desc, lay_a, lay_b, lay_c, algo: results[0].algo }
        }
    }
}
