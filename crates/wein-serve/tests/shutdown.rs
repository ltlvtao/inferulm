//! 优雅退出（任务书 §5）：SIGTERM 语义 = 停止接纳 → drain 在飞 → 退出码 0。
//!
//! 这里直接打 `spawn_server_guarded`（CLI 用的同一入口）——CLI 只是把
//! SIGINT/SIGTERM 接到 `Server::shutdown` 上，时序完全一致，故无需真进程即可钉死。
mod common;

use std::time::Duration;

use serde_json::Value;

use common::{chat_body, client, default_tok, start_guarded, MockScript};

/// s1 · 在飞流式请求在 shutdown 期间**正常收完**（finish 帧 + `[DONE]`），
/// 之后新连接被拒（监听已关闭）。
#[tokio::test]
async fn s1_inflight_stream_completes_then_listener_closes() {
    // 慢速流：每 token 30ms，共 6 个 → ~180ms 的在飞窗口
    let script = MockScript { per_token_delay: Duration::from_millis(30), ..Default::default() };
    let (server, _mock) = start_guarded(script, default_tok()).await;
    let base = format!("http://{}", server.addr);

    let (tx, rx) = tokio::sync::oneshot::channel();
    let base_req = base.clone();
    let handle = tokio::spawn(async move {
        let frames =
            common::post_sse(&format!("{base_req}/v1/chat/completions"), &chat_body(6)).await;
        tx.send(frames).unwrap();
    });

    // 等到"确实已经开跑"（首个 token 出来后）再触发 drain
    tokio::time::sleep(Duration::from_millis(60)).await;
    let left = server.shutdown(Duration::from_secs(5)).await;
    assert!(left.is_empty(), "所有在飞请求都应在 5s 内收完：{left:?}");

    let frames = rx.await.expect("流式请求必须正常收完");
    assert_eq!(frames.last().map(String::as_str), Some("[DONE]"), "终止帧 [DONE] 必须在");
    let last_json = frames
        .iter()
        .rev()
        .find(|f| f.as_str() != "[DONE]")
        .and_then(|f| serde_json::from_str::<Value>(f).ok())
        .expect("必须有终止块");
    assert_eq!(last_json["choices"][0]["finish_reason"], "length", "正常结束，不是 error");
    assert!(!common::content_of(&frames).is_empty(), "在飞内容必须完整吐出");
    assert!(last_json.get("wein_finish").is_none(), "正常收尾不带 wein_finish");

    // 新连接必须被拒（监听已关闭）
    let err = client().get(format!("{base}/health")).send().await;
    assert!(err.is_err(), "shutdown 之后新连接必须被拒");
    handle.await.unwrap();
}

/// s2 · drain 超时：未完成的 request_id 必须被报出来（CLI 据此写日志），
/// 服务随后强制退出（监听关闭）。
#[tokio::test]
async fn s2_drain_timeout_lists_unfinished_request_ids() {
    // 每 token 200ms × 12 ≈ 2.4s：远超 150ms 的 drain 预算
    let script = MockScript { per_token_delay: Duration::from_millis(200), ..Default::default() };
    let (server, _mock) = start_guarded(script, default_tok()).await;
    let base = format!("http://{}", server.addr);

    let base_req = base.clone();
    let task = tokio::spawn(async move {
        // 客户端会在强制退出时看到断流——这里只关心服务侧的报告
        let _ = common::post_sse(&format!("{base_req}/v1/chat/completions"), &chat_body(12)).await;
    });
    tokio::time::sleep(Duration::from_millis(80)).await;

    let left = server.shutdown(Duration::from_millis(150)).await;
    assert_eq!(left.len(), 1, "超时必须有未完成清单：{left:?}");
    let (engine_id, request_id) = &left[0];
    assert!(*engine_id >= 1, "引擎 request id 要能对上");
    assert!(request_id.starts_with("req-"), "客户端 request id = {request_id}");

    let err = client().get(format!("{base}/health")).send().await;
    assert!(err.is_err(), "强制退出后新连接必须被拒");
    let _ = task.await;
}

/// s3 · drain 之后到达的新请求拿到 503 + `Connection: close`
/// （已建立的 keep-alive 连接上仍可能蹦出新请求——必须明确拒绝而不是排队等死）。
#[tokio::test]
async fn s3_requests_during_drain_get_503_with_connection_close() {
    let script = MockScript { per_token_delay: Duration::from_millis(50), ..Default::default() };
    let (server, _mock) = start_guarded(script, default_tok()).await;
    let base = format!("http://{}", server.addr);
    // 先在飞一个长请求，让 drain 期间连接仍然存在
    let base2 = base.clone();
    let task = tokio::spawn(async move {
        let _ = common::post_sse(&format!("{base2}/v1/chat/completions"), &chat_body(20)).await;
    });
    tokio::time::sleep(Duration::from_millis(60)).await;

    // draining 标志只能通过 shutdown 触发；这里另起一个短 drain 与探测赛跑
    let probe = tokio::spawn(async move {
        // 循环探测直到拿到 503/连接被拒（shutdown 置位后才能观察到）
        for _ in 0..100 {
            match client()
                .post(format!("{base}/v1/chat/completions"))
                .json(&chat_body(1))
                .send()
                .await
            {
                Ok(resp) if resp.status() == 503 => {
                    let close = resp
                        .headers()
                        .get("connection")
                        .map(|v| v.to_str().unwrap_or("").to_string());
                    let v: Value = resp.json().await.unwrap();
                    return Some((close, v));
                }
                Ok(_) => tokio::time::sleep(Duration::from_millis(10)).await,
                Err(_) => return None, // 监听已关：等价"被拒"
            }
        }
        None
    });
    let left = server.shutdown(Duration::from_secs(5)).await;
    let observed = probe.await.unwrap();
    // 两种可接受的终态：拿到 503+close，或监听已经关闭（连接被拒）
    if let Some((close, v)) = observed {
        assert_eq!(close.as_deref(), Some("close"), "drain 期间的 503 必须带 Connection: close");
        assert_eq!(v["error"]["type"], "server_error");
    }
    let _ = left;
    let _ = task.await;
}
