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
//! algo 钉（g8）：heuristic 返回的**全部**有效候选在首 call 上用调用者的
//! 真实缓冲逐个实测（事件对在调用者流上——纯段计量），滞回择优后钉。
//! 治两账：overlap_probe 副产（heuristic 首选比最优差 13%）与分配序变化
//! 引起的 heuristic 漂移。两条选型稳定律（位级合同 qg2/重放要求双执行器
//! 选型必同）：**滞回**——近竞（<5% 且 <1µs）钉回首选，噪声翻不动真差距；
//! **进程级钉表**——同进程后开实例直接复用先钉选型，免二次调优且必同。
//! 图捕获安全由 build_graphs 的 eager 预热保证（捕获前缓存已暖）；
//! capturing() 时防御性免测直取首选。
//!
//! 库获取（运行时依赖，非构建依赖）：pip wheel `nvidia-cublas-cu12` 提取
//! `libcublasLt.so.12` → 仓内 `vendor/`（gitignore）；候选链含
//! `WEIN_CUBLASLT` 环境变量覆盖。
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use libloading::Library;

use super::{capturing, DevBuf, Gpu, Stream};

// ── 常量（cublasLt 头文件原值；g2 精度验收是对这些值的经验校验） ──
const CUDA_R_32F: i32 = 0;
const CUDA_R_16BF: i32 = 14;
const CUBLAS_COMPUTE_32F: i32 = 68;
const LAYOUT_ORDER: i32 = 1; // cublasLtMatrixLayoutAttributes_t::ORDER
const ORDER_ROW: u32 = 1; // cublasLtOrder_t::ROW
const PREF_MAX_WORKSPACE: i32 = 1; // MatmulPreference::MAX_WORKSPACE_BYTES
const MATMUL_DESC_TRANSB: i32 = 4; // cublasLtMatmulDescAttributes_t::TRANSB
const OP_T: u32 = 1; // cublasOperation_t::CUBLAS_OP_T
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

/// 每形状 algo 钉的审计记录（g8 验收面：恰调优一次/择优不劣于首选）
#[derive(Clone, Copy, Debug)]
pub struct GemmTuneRecord {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    /// weight 转置形态（gemm_wt）
    pub wt: bool,
    /// 实测过的有效候选数
    pub candidates: usize,
    /// 实测最优候选序号（滞回前）
    pub argmin: usize,
    /// 钉住的候选序号（滞回后——近竞钉回首选 0）
    pub chosen: usize,
    /// 实测最优单次耗时（事件对 5 发均值，ms）
    pub best_ms: f64,
    /// heuristic 首选同协议实测（ms）——配对比较基线
    pub first_ms: f64,
    /// 选中候选的 workspace 字节数（0 = eager 路径免 Lt 内部 memset 对）
    pub ws: usize,
}

/// 进程级 algo 钉表：同形状后开实例复用先钉选型（qg2 零漂移要求
/// eager/图档双执行器选型必同；也免每实例重复调优）
static PINS: OnceLock<Mutex<HashMap<(u32, u32, u32, bool), [u64; 8]>>> = OnceLock::new();

fn pins() -> &'static Mutex<HashMap<(u32, u32, u32, bool), [u64; 8]>> {
    PINS.get_or_init(|| Mutex::new(HashMap::new()))
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
    desc_set: unsafe extern "C" fn(*mut c_void, i32, *const c_void, usize) -> u32,
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
    /// 形状 → 解析产物（desc/layouts/algo）；命中即纯 launch。
    /// key 的 bool 位 = weight 转置形态（gemm_wt）
    cache: RefCell<HashMap<(u32, u32, u32, bool), ShapePlan>>,
    /// autotune 审计（每形状一条；g8 断言面）
    report: RefCell<Vec<GemmTuneRecord>>,
    /// 调优 m 集（升序去重）：m ∈ 集才 autotune，其余 heuristic 钉入
    /// 进程表——瞬态形状（ramp/drain 的 m 扫描）零调优债
    tune_m: Vec<u32>,
}

macro_rules! sym {
    ($lib:expr, $name:literal, $ty:ty) => {
        *$lib.get::<$ty>($name.as_bytes()).unwrap_or_else(|e| panic!("符号 {} 缺失：{e}", $name))
    };
}

impl BlasLt<'_> {
    /// dlopen 候选链：env 覆盖 → 常规 soname → 仓内 vendor（wheel 提取）。
    /// 调优集为空（全 heuristic 钉——调用方确知形状瞬态时用）
    pub fn open(gpu: &Gpu) -> Result<BlasLt<'_>, String> {
        Self::open_tuned(gpu, &[])
    }

    /// `tune_m` 内的 m 才 autotune（稳态 decode 桶集）；其余形状 heuristic
    /// 首选钉入进程表——服务进程 m 扫描（ramp/drain）的一次性形状不付
    /// 调优债（battery 实测：无门控时满负荷 tps 2827→1678，ttft ×4）
    pub fn open_tuned<'a>(gpu: &'a Gpu, tune_m: &[u32]) -> Result<BlasLt<'a>, String> {
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
                Ok(lib) => return Self::init(gpu, lib, tune_m).map_err(|e| format!("{c}: {e}")),
                Err(e) => last = format!("{c}: {e}"),
            }
        }
        Err(format!("全部候选失败（最后：{last}）；可设 WEIN_CUBLASLT 指定路径"))
    }

    fn init<'a>(gpu: &'a Gpu, lib: Library, tune_m: &[u32]) -> Result<BlasLt<'a>, String> {
        unsafe {
            let fns = LtFns {
                create: sym!(lib, "cublasLtCreate", unsafe extern "C" fn(*mut *mut c_void) -> u32),
                version: sym!(lib, "cublasLtGetVersion", unsafe extern "C" fn() -> usize),
                desc_create: sym!(lib, "cublasLtMatmulDescCreate", unsafe extern "C" fn(*mut *mut c_void, i32, i32) -> u32),
                layout_create: sym!(lib, "cublasLtMatrixLayoutCreate", unsafe extern "C" fn(*mut *mut c_void, i32, u64, u64, i64) -> u32),
                layout_set: sym!(lib, "cublasLtMatrixLayoutSetAttribute", unsafe extern "C" fn(*mut c_void, i32, *const c_void, usize) -> u32),
                desc_set: sym!(lib, "cublasLtMatmulDescSetAttribute", unsafe extern "C" fn(*mut c_void, i32, *const c_void, usize) -> u32),
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
                report: RefCell::new(Vec::new()),
                tune_m: {
                    let mut v = tune_m.to_vec();
                    v.sort_unstable();
                    v.dedup();
                    v
                },
            })
        }
    }

    pub fn version(&self) -> u64 {
        unsafe { (self.fns.version)() as u64 }
    }

    /// autotune 审计面（g8）：每形状一条，调用序即记录序
    pub fn tune_report(&self) -> Vec<GemmTuneRecord> {
        self.report.borrow().clone()
    }

    /// C = A·B（bf16 入出，f32 累加）。首次形状解析（m ∈ 调优集则
    /// autotune 择优，否则 heuristic 钉）后缓存，命中即纯 launch
    pub fn gemm(&self, a: &DevBuf, b: &DevBuf, c: &DevBuf, m: u32, n: u32, k: u32, stream: &Stream) {
        let plan = self.plan(a, b, c, m, n, k, false, stream);
        self.launch_mm(a, b, c, m, n, k, plan, stream);
    }

    /// C = A·Wᵀ：W 按存储 [n,k] 行主序直读（nn.Linear 权重布局——模型
    /// 投影的稳态形态，免转置拷贝）。desc 设 TRANSB=T，B layout 描述
    /// 存储体 n×k、op 后成 k×n
    pub fn gemm_wt(&self, a: &DevBuf, w: &DevBuf, c: &DevBuf, m: u32, n: u32, k: u32, stream: &Stream) {
        let plan = self.plan(a, w, c, m, n, k, true, stream);
        self.launch_mm(a, w, c, m, n, k, plan, stream);
    }

    fn launch_mm(
        &self,
        a: &DevBuf,
        b: &DevBuf,
        c: &DevBuf,
        m: u32,
        n: u32,
        k: u32,
        plan: ShapePlan,
        stream: &Stream,
    ) {
        let st = self.launch_algo(a, b, c, &plan, &plan.algo, stream);
        assert_eq!(st, 0, "cublasLtMatmul 失败：status {st}（{m}×{n}×{k}）");
    }

    /// 单发 launch，status 返回不 panic——候选探测要在失败候选上继续
    fn launch_algo(
        &self,
        a: &DevBuf,
        b: &DevBuf,
        c: &DevBuf,
        plan: &ShapePlan,
        algo: &[u64; 8],
        stream: &Stream,
    ) -> u32 {
        let (alpha, beta) = (1.0f32, 0.0f32);
        unsafe {
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
                algo as *const [u64; 8] as *const c_void,
                self.ws.device_ptr() as *mut c_void,
                WS_BYTES,
                stream.handle,
            )
        }
    }

    /// 形状解析：desc + 三 layout（ROW_ORDER）+ preference(工作区) +
    /// heuristic（集内 m 续 autotune 择优）——全部一次性，缓存复用。
    /// 借调用者的真实缓冲（免 logits 级 scratch；地址即稳态代表）
    fn plan(
        &self,
        a: &DevBuf,
        b: &DevBuf,
        c: &DevBuf,
        m: u32,
        n: u32,
        k: u32,
        wt: bool,
        stream: &Stream,
    ) -> ShapePlan {
        if let Some(p) = self.cache.borrow().get(&(m, n, k, wt)) {
            return *p;
        }
        let plan = self.resolve_plan(a, b, c, m, n, k, wt, stream);
        self.cache.borrow_mut().insert((m, n, k, wt), plan);
        plan
    }

    fn resolve_plan(
        &self,
        a: &DevBuf,
        b: &DevBuf,
        c: &DevBuf,
        m: u32,
        n: u32,
        k: u32,
        wt: bool,
        stream: &Stream,
    ) -> ShapePlan {
        unsafe {
            let f = &self.fns;
            // desc：compute 32F + scale f32；opA 默认 N；wt 形态 opB=T
            let mut desc = std::ptr::null_mut();
            assert_eq!((f.desc_create)(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), 0, "desc_create");
            if wt {
                assert_eq!(
                    (f.desc_set)(desc, MATMUL_DESC_TRANSB, &OP_T as *const u32 as *const c_void, 4),
                    0,
                    "desc TRANSB=T"
                );
            }
            let mk_layout = |rows: u64, cols: u64, ld: i64| -> *mut c_void {
                let mut lay = std::ptr::null_mut();
                assert_eq!((f.layout_create)(&mut lay, CUDA_R_16BF, rows, cols, ld), 0, "layout_create");
                assert_eq!((f.layout_set)(lay, LAYOUT_ORDER, &ORDER_ROW as *const u32 as *const c_void, 4), 0, "layout ORDER=ROW");
                lay
            };
            // wt：B 按存储 n×k 描述（ld=k），TRANSB 后成 k×n 参与乘法
            let lay_b = if wt {
                mk_layout(n as u64, k as u64, k as i64)
            } else {
                mk_layout(k as u64, n as u64, n as i64)
            };
            let (lay_a, lay_c) =
                (mk_layout(m as u64, k as u64, k as i64), mk_layout(m as u64, n as u64, n as i64));
            // 进程钉表命中：免 heuristic 免调优（同进程选型必同——qg2 合同）
            if let Some(algo) = pins().lock().unwrap().get(&(m, n, k, wt)) {
                return ShapePlan { desc, lay_a, lay_b, lay_c, algo: *algo };
            }
            let mut pref = std::ptr::null_mut();
            assert_eq!((f.pref_create)(&mut pref), 0, "pref_create");
            let ws = WS_BYTES as u64;
            assert_eq!((f.pref_set)(pref, PREF_MAX_WORKSPACE, &ws as *const u64 as *const c_void, 8), 0, "pref_set");
            let mut results = [HeuristicResult { algo: [0; 8], workspace_size: 0, state: 1, waves_count: 0.0, reserved: [0; 4] }; 8];
            let mut returned = 0i32;
            let st = (f.heuristic)(self.handle, desc, lay_a, lay_b, lay_c, lay_c, pref,
                results.len() as i32, results.as_mut_ptr(), &mut returned);
            assert_eq!(st, 0, "heuristic status {st}（{m}×{n}×{k}）");
            assert!(returned > 0, "heuristic 空返回（{m}×{n}×{k}）");
            let valid: Vec<HeuristicResult> = results[..returned as usize]
                .iter()
                .filter(|r| r.state == 0 && r.workspace_size <= WS_BYTES)
                .copied()
                .collect();
            assert!(!valid.is_empty(), "无可用 algo（returned={returned}，全部 state≠0 或工作区超预算）");
            // 调优门控：m ∈ 调优集才实测择优；其余（瞬态形状）heuristic
            // 首选直钉——两种路径都写进程表，同进程后开实例必同选型
            let algo = if self.tune_m.binary_search(&m).is_ok() {
                let mut probe = ShapePlan { desc, lay_a, lay_b, lay_c, algo: valid[0].algo };
                let algo = self.pick_algo(a, b, c, m, n, k, wt, &probe, &valid, stream);
                probe.algo = algo;
                probe.algo
            } else {
                valid[0].algo
            };
            pins().lock().unwrap().insert((m, n, k, wt), algo);
            ShapePlan { desc, lay_a, lay_b, lay_c, algo }
        }
    }

    /// 候选逐个实测择优：前 4 候选 ×（1 预热 + 事件对 3 计时）（调用者流
    /// 上——事件流序，前序排队工作被排除，纯段计量），**滞回**后钉——
    /// 实测最优须同时赢首选 ≥5% 且 ≥1µs 才换人，近竞钉回首选取其稳定
    /// （毫厘之得让位于跨会话选型一致）。计时写花 c 无害：真 launch 随后
    /// 以 beta=0 覆写。capturing 中不测（防御；build_graphs 的 eager 预热
    /// 保证缓存先于捕获变暖，此路径正常不达）。
    ///
    /// **ws0 偏好**（WEIN_LT_WS0=0 关）：workspace_size==0 的候选在时间
    /// 损失 ≤3% 内优先——Lt 对带工作区的 algo 每次调用内部发一对
    /// memset（实测 ~1MB+48KB，nsys 满载 c=100 每 gemm 一对、每步 ~330
    /// 次 ≈ 1.3ms CPU + ~1ms GPU）。图内 memset 是节点零 CPU，eager
    /// 混合步全裸付——省 API 比毫厘 gemm 时间值钱。偏好是形状纯函数
    /// （同候选集同规则），qg2 双执行器选型一致性不受影响。
    #[allow(clippy::too_many_arguments)]
    fn pick_algo(
        &self,
        a: &DevBuf,
        b: &DevBuf,
        c: &DevBuf,
        m: u32,
        n: u32,
        k: u32,
        wt: bool,
        probe: &ShapePlan,
        valid: &[HeuristicResult],
        stream: &Stream,
    ) -> [u64; 8] {
        let ws0_pref = std::env::var("WEIN_LT_WS0").map(|v| v != "0").unwrap_or(true);
        if capturing() {
            let rec = GemmTuneRecord { m, n, k, wt, candidates: 0, argmin: 0, chosen: 0, best_ms: 0.0, first_ms: 0.0, ws: valid[0].workspace_size };
            self.report.borrow_mut().push(rec);
            return valid[0].algo;
        }
        let n_cand = valid.len().min(4); // 滞回门 ≥5% 的真差距必在 heuristic 前 4
        let (ev0, ev1) = (self.gpu.event(), self.gpu.event());
        let mut timings: Vec<(usize, f64)> = Vec::with_capacity(n_cand);
        for (vi, r) in valid[..n_cand].iter().enumerate() {
            // 预热兼探测：status≠0 的候选（state==0 合同外的意外）弃测
            if self.launch_algo(a, b, c, probe, &r.algo, stream) != 0 {
                continue;
            }
            ev0.record_on(stream);
            for _ in 0..3 {
                self.launch_algo(a, b, c, probe, &r.algo, stream);
            }
            ev1.record_on(stream);
            stream.sync();
            timings.push((vi, ev1.elapsed_ms(&ev0) as f64 / 3.0));
        }
        // 全候选探测失败（理论不达）：退回首选用不测直取
        let Some(&(argmin, best_ms)) = timings.iter().min_by(|x, y| x.1.total_cmp(&y.1)) else {
            let rec = GemmTuneRecord { m, n, k, wt, candidates: 0, argmin: 0, chosen: 0, best_ms: 0.0, first_ms: 0.0, ws: valid[0].workspace_size };
            self.report.borrow_mut().push(rec);
            return valid[0].algo;
        };
        // 首选基线：同协议配对实测（首选未测成时无从配对，退等值免虚账）
        let first_ms = timings
            .iter()
            .find(|&&(vi, _)| vi == 0)
            .map(|&(_, t)| t)
            .unwrap_or(best_ms);
        // 滞回：近竞（gain < 5% 或 < 1µs）钉回首选取跨会话稳定
        let gain = first_ms - best_ms;
        let mut chosen = if best_ms <= first_ms * 0.95 && gain >= 0.001 { argmin } else { 0 };
        // ws0 偏好：最快候选带工作区时，找 ws==0 里最快的；时间损失 ≤3%
        // 才换（毫厘之得让位于 eager 步的 API 发射税）
        if ws0_pref && valid[chosen].workspace_size > 0 {
            if let Some(&(w0, w0_ms)) = timings
                .iter()
                .filter(|&&(vi, _)| valid[vi].workspace_size == 0)
                .min_by(|x, y| x.1.total_cmp(&y.1))
            {
                if w0_ms <= best_ms * 1.03 {
                    chosen = w0;
                }
            }
        }
        let rec = GemmTuneRecord { m, n, k, wt, candidates: timings.len(), argmin, chosen, best_ms, first_ms, ws: valid[chosen].workspace_size };
        self.report.borrow_mut().push(rec);
        valid[chosen].algo
    }
}
