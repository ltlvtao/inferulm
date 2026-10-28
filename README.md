# wein

国产硬件上的 vLLM+SGLang：纯 Rust 核心 + PTX（NVIDIA）/ AscendC（昇腾）双平台算子层。

设计文档见 [`../design/`](../design/)（01-strategy / 02-architecture）。
开发方法：**TDD 外→内**——验收用例先于实现，从 serve 层向内核层推进。

## 当前状态（M0 第一层：serve）

- `wein-serve`：OpenAI-compatible HTTP/SSE（chat/completions + completions + models），增量 detokenizer（UTF-8 边界安全），stop 参数截断 + 引擎取消，断连自动取消
- `wein-engine`：engine↔serve 边界协议（Submit/Cancel + 事件流）+ MockEngine（可编程替身）
- `wein-cli`：`wein serve --mock` 起 mock 服务
- `wein-bench`：并发性能验收工具（TTFT/ITL/吞吐 + smoke 阈值门禁）

## 常用命令

```bash
cargo test -p wein-serve          # 验收套件（A 契约 / B 并发与 abort）
cargo test                        # 全部（含单元测试）

# 手工体验
cargo run -p wein-cli -- --mock --port 8080
curl -N localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"mock-model","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"stream":true}'

# 并发性能验收（今天打 mock，之后打真实引擎——同一把尺子）
cargo run -p wein-bench -- --url http://127.0.0.1:8080 --concurrency 100 --num 200 \
  --max-tokens 32 --repeat 3 --warmup \
  --smoke "ttft_p99_ms<=100;itl_p99_ms<=20;tps_min>=5000;errors_max<=0"

# 性能基线电池（归因阶梯 L0/L1/L1'，报告入 bench-reports/battery-*/，方法论见 bench-reports/BASELINE.md）
bench/battery.sh release
```

## TDD 推进路线（外→内，每层验收先行）

1. **serve**（当前层）：MockEngine 钉死边界协议
2. engine core：真实调度器（token 预算/FSM/BatchBuffer）——用现有 serve 验收套件直接验收，MockEngine 退役为测试专用
3. memory：paged KV / BlockPool / radix / req_to_token
4. kernel：`trait Op` + cuda/（PTX）——W1 第 1 周先做 cuda crate 冒烟实验（02-architecture §13-2）
5. 逐层替换 mock，验收套件始终不变

## 规约

- 依赖只许向下：`serve → engine → {sched, memory, exec, model} → kernel → platform`（02-architecture §2）
- 引擎边界只经 `EngineHandle::{submit, cancel}` + 事件流，禁止旁路
- 每个集成里程碑跑一次 `wein-bench` 并把 JSON 报告归档
