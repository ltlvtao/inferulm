# 进程退出期 segfault 修复：teardown 顺序化

日期：2026-11-10 · 影响面：wein-cli 全服务形态（GPU 路径）

## 1. 症状

dmesg 两次实锤（bench 会话中，12+ 发启停 1 发）：

```
wein-cli[8009]: segfault at 77d1d92bd610 ip 000077d1d92bd610 sp 00007ffe949b1f28 error 14
                （ip == fault addr：指令指针落在已卸载/无效页——error 14 = 用户态取指页错误）
```

WSL CaptureCrash 各捕获一次。退出码 139，drain 日志正常走完之后崩。

## 2. 机制

旧退出路径（`main.rs` SIGTERM 分支）：

```
server.shutdown(drain) → std::process::exit(0)
```

`process::exit` 不跑栈上 Drop（async 块里的 Server/ServeState/EngineHandle 全部存活），
直接执行 atexit 链——其中有 CUDA driver 注册的 context teardown。此刻：

1. **engine core 任务还活着**（它的退出条件是"全部 cmd_tx drop"，而 handle 还在
   async 块的作用域里）；若正处 GPU step 中，就是"一个线程在提交/等待 kernel，
   另一个线程在销毁 context"。
2. **executor 的 Drop 从未运行**（几 GB 显存、cuBLASLt 句柄、模块、4GB pinned 的
   CUDA 销毁串留给 driver teardown 兜底）。

两边并发销毁同一 context → driver 内部跳到已卸载代码页 → ip==fault addr 的形态。

## 3. 修复（顺序化）

三处改动：

1. `EngineStats` 增加 `exit_tx: watch::Sender<bool>` + `wait_exit()`/`exited()`：
   core 任务的显式退出信号（watch 保留终值，等待者不会错过）。
2. `EngineCore::run()` 末尾：**先 `drop(exec)`（跑完 CUDA 销毁串）再置位 watch**——
   等待者醒来时 GPU 资源已全部归还。
3. `main.rs` SIGTERM 路径：`server.shutdown()` 后等 `stats.wait_exit()`（10s 超时
   兜底），然后**删除 `process::exit(0)`，让 async 块自然返回**——runtime drop →
   静态析构 → atexit 全部落在干净状态上。退出码仍为 0。

### 第一次尝试的教训（已回退）

先试了"`engine.clone()` 后轮询 `is_alive()`"：**循环等待**——clone 的 handle 自己
持有 cmd_tx，core 任务挂在 `recv()` 等"全部 sender drop"，永不退出（实测 10s 超时
警告）。等待者必须不持命令通道：clone `Arc<EngineStats>`（stats 无 cmd_tx）。

## 4. 验证

- 受控复现（v1 空闲退出 40 次 + v2 满负荷在飞+drain 1ms 25 次）：**旧版均 0 崩溃**
  ——原崩溃频率低于受控循环的检出限（两轮 dmesg 实锤均出现在长 bench 会话）。
  本修复的依据是机制分析（§2），验证口径为修复版零回归：
- 修复版：idle/busy 手动两次 + 25 次满负荷循环，全部 rc=0，退出日志干净
  （无"10s 未退出"警告——wait_exit 正常等到 exec drop 完成）。
- `cargo test -p wein-engine -p wein-serve` 全绿。

## 5. 边界

- mock engine 路径无 stats（`exit_stats=None`）→ 跳过等待直接自然返回：纯 CPU 无
  CUDA，无此风险面。
- 启动失败路径的 `exit(1)`（模型加载失败等）发生在 engine spawn 之前，无竞争。
- core 任务若因长 prefill 超 10s 不退：打印警告，交给 runtime teardown 兜底
  （cancel 在 await 点 drop future，exec 依然会被 drop，只是不再保证与 atexit 顺序）。
