//! `/metrics` 与 `/health` 的数据源。
//!
//! 两部分拼起来（任务书 §3）：
//! - **serve 侧计数器**（本模块 `Metrics`）：请求结果四分类与 token 数——只有
//!   HTTP 层知道"客户端拿到的是什么"（accepted/finished/cancelled/failed）；
//! - **引擎侧统计**（`wein_engine::EngineStats`）：steps / preempted /
//!   batch_rows / prefix_hit_tokens——只有引擎知道"跑了多少步"。
//!
//! 全部无锁原子量：每请求几条 `fetch_add`，`/metrics` 一次 Relaxed 读。
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use wein_engine::EngineStats;

/// serve 侧请求/ token 计数（Prometheus 文本见 [`Metrics::render`]）。
#[derive(Default)]
pub struct Metrics {
    /// 提交成功的请求（进入引擎）
    pub accepted: AtomicU64,
    /// 正常结束（finish_reason = stop / length）
    pub finished: AtomicU64,
    /// 取消类结束（客户端断开、stop 字符串提前收尾、引擎 Aborted）
    pub cancelled: AtomicU64,
    /// 失败结束（提交失败 503、执行器 Invalid 失败、伪造 request_id 的非法请求）
    pub failed: AtomicU64,
    pub prompt_tokens: AtomicU64,
    pub completion_tokens: AtomicU64,
}

impl Metrics {
    fn inc(c: &AtomicU64, n: u64) {
        c.fetch_add(n, Ordering::Relaxed);
    }

    pub fn add_prompt_tokens(&self, n: usize) {
        Self::inc(&self.prompt_tokens, n as u64);
    }

    pub fn add_completion_tokens(&self, n: u64) {
        Self::inc(&self.completion_tokens, n);
    }

    /// Prometheus 文本格式（`text/plain; version=0.0.4`）。
    pub fn render(&self, stats: Option<&EngineStats>) -> String {
        let (steps, preempted, batch_rows, prefix_hit) = match stats {
            Some(s) => (
                s.step_count(),
                s.preempted_count() as u64,
                s.batch_rows(),
                s.prefix_hit_tokens(),
            ),
            None => (0, 0, 0, 0),
        };
        let mut out = String::with_capacity(1024);
        out.push_str("# HELP wein_requests_total HTTP 生成请求数（按结果分类）\n");
        out.push_str("# TYPE wein_requests_total counter\n");
        out.push_str(&format!(
            "wein_requests_total{{result=\"accepted\"}} {}\n",
            self.accepted.load(Ordering::Relaxed)
        ));
        out.push_str(&format!(
            "wein_requests_total{{result=\"finished\"}} {}\n",
            self.finished.load(Ordering::Relaxed)
        ));
        out.push_str(&format!(
            "wein_requests_total{{result=\"cancelled\"}} {}\n",
            self.cancelled.load(Ordering::Relaxed)
        ));
        out.push_str(&format!(
            "wein_requests_total{{result=\"failed\"}} {}\n",
            self.failed.load(Ordering::Relaxed)
        ));
        out.push_str("# HELP wein_tokens_total token 数（prompt = 输入，completion = 已发射）\n");
        out.push_str("# TYPE wein_tokens_total counter\n");
        out.push_str(&format!(
            "wein_tokens_total{{kind=\"prompt\"}} {}\n",
            self.prompt_tokens.load(Ordering::Relaxed)
        ));
        out.push_str(&format!(
            "wein_tokens_total{{kind=\"completion\"}} {}\n",
            self.completion_tokens.load(Ordering::Relaxed)
        ));
        out.push_str("# HELP wein_steps_total 引擎前向步数\n");
        out.push_str("# TYPE wein_steps_total counter\n");
        out.push_str(&format!("wein_steps_total {steps}\n"));
        out.push_str("# HELP wein_preempted_total 抢占次数\n");
        out.push_str("# TYPE wein_preempted_total counter\n");
        out.push_str(&format!("wein_preempted_total {preempted}\n"));
        out.push_str("# HELP wein_batch_rows 上一步前向批行数\n");
        out.push_str("# TYPE wein_batch_rows gauge\n");
        out.push_str(&format!("wein_batch_rows {batch_rows}\n"));
        out.push_str("# HELP wein_prefix_hit_tokens_total 前缀缓存命中 token 累计\n");
        out.push_str("# TYPE wein_prefix_hit_tokens_total counter\n");
        out.push_str(&format!("wein_prefix_hit_tokens_total {prefix_hit}\n"));
        out
    }
}

/// 在飞请求登记簿（request_id → 客户端请求标识）。
///
/// 退出的 drain 阶段要用它回答"谁还没完"（任务书 §5：超时强制退出时日志里
/// 列出未完成的 request_id）。用 `Mutex` 而不是原子量：drain 只在退出时读一次，
/// 而登记/注销是每请求两条短临界区。
#[derive(Default)]
pub struct Inflight {
    inner: Mutex<BTreeMap<u64, String>>,
}

impl Inflight {
    pub fn entry(self: &Arc<Self>, engine_id: u64, request_id: String) -> InflightGuard {
        self.inner.lock().expect("在飞表锁中毒").insert(engine_id, request_id);
        InflightGuard { table: self.clone(), engine_id, done: false }
    }

    pub fn len(&self) -> usize {
        self.inner.lock().expect("在飞表锁中毒").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 尚未完成的（引擎 id, 客户端 request_id）快照
    pub fn snapshot(&self) -> Vec<(u64, String)> {
        self.inner
            .lock()
            .expect("在飞表锁中毒")
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect()
    }

    fn remove(&self, engine_id: u64) {
        self.inner.lock().expect("在飞表锁中毒").remove(&engine_id);
    }
}

/// 在飞登记守卫：流结束或客户端断开（Stream 被 drop）时注销。
pub struct InflightGuard {
    table: Arc<Inflight>,
    engine_id: u64,
    done: bool,
}

impl InflightGuard {
    pub fn finish(&mut self) {
        if !self.done {
            self.done = true;
            self.table.remove(self.engine_id);
        }
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        if !self.done {
            self.table.remove(self.engine_id);
        }
    }
}
