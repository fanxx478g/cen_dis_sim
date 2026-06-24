# long_prefill_sim

`long_prefill_sim` 是一个独立的离散事件模拟工具，专门研究长 `prefill` 请求的等待时间、排队比例和服务时延。它不依赖主仓库 `simulator/` 的配置体系，适合单独开展实验。

## 这个模块解决什么问题

它主要回答下面这类问题：

- 长上下文请求在 `centralized` 和 `distributed` 部署下，等待时间差多少？
- 当首轮请求到达更密集时，`avg_wait`、`queued_ratio`、`within_target_ratio` 怎么变化？
- `region_first_policy` 是否能缓解局部区域排队？
- 某个固定配置下，最严重的排队请求是什么样？

## 先看什么文件

推荐阅读顺序：

1. [`simulation.py`](C:/Project/cen_dis_sim/long_prefill_sim/simulation.py)
   - 核心模型、配置定义、workload 生成、事件循环、统计函数。
2. [`run_compare.py`](C:/Project/cen_dis_sim/long_prefill_sim/run_compare.py)
   - 批量实验入口，适合看“有哪些场景、扫哪些参数、写哪些结果”。
3. [`inspect_waits.py`](C:/Project/cen_dis_sim/long_prefill_sim/inspect_waits.py)
   - 单配置诊断入口，适合看“某个具体配置为什么排队严重”。

## 先怎么跑起来

### 第 1 步：跑 batch

```powershell
python .\long_prefill_sim\run_compare.py
```

你会看到：

- 场景开始/结束日志
- 每个扫描点的摘要
- `outputs/compare_results.csv`
- `outputs/wait_time_bucket_ratios.csv`

### 第 2 步：看输出文件

重点先看：

- `outputs/compare_results.csv`
  - 每个场景、每个负载点的总体统计
- `outputs/wait_time_bucket_ratios.csv`
  - 等待时间分桶累计占比

### 第 3 步：跑单配置诊断

```powershell
python .\long_prefill_sim\inspect_waits.py --help
```

然后选一个具体配置做诊断。

例如：

```powershell
python .\long_prefill_sim\inspect_waits.py --mode distributed --region-policy weight_region_policy --total-users 180 --first-arrival-interval-ms 12000
```

### 第 4 步：看最坏请求

`inspect_waits.py` 会打印：

- `overall` 汇总
- `region_stats`
- 等待时间最长的前 `N` 条请求

如果你口头上把这个单配置诊断入口叫做 `run_compare_instance`，当前仓库里对应的实际脚本就是 `inspect_waits.py`。

## 核心机制

## 1. 部署模式

当前 `DeploymentMode`：

- `centralized`
- `distributed`

含义：

- `centralized`
  - 所有请求进入一个全局等待队列，共享一组实例
- `distributed`
  - 各 region 有自己的等待队列和实例集合

## 2. 区域路由策略

当前 `UserRegionPolicy` 支持：

- `weight_region_policy`
- `queue_len_policy`
- `d_queue_len_policy`
- `region_first_policy`

含义概览：

- `weight_region_policy`
  - 用户按 region 权重固定绑定
- `queue_len_policy`
  - 到达时按当前最短队列选 region
- `d_queue_len_policy`
  - 到达时按秒级快照队列长度选 region
- `region_first_policy`
  - 优先本地，达到阈值后回退 central region

## 3. workload 生成方式

请求生成是两阶段的：

1. 所有用户的 first request 形成一个全局到达流
2. 同一用户的 follow-up 请求，到达时间 = 上一轮真实完成时间 + `think time`

相关配置在：

- `SimulationConfig`
- `RequestCountConfig`
- `ThinkTimeConfig`

## 4. 长 prefill token 采样

`PrefillConfig` 里和长请求长度最相关的字段：

- `lognormal_mu`
- `lognormal_sigma`
- `truncated_min_tokens`
- `truncated_max_tokens`
- `long_prefill_min_tokens`

当前实现采用截断 lognormal 采样，且只保留不小于 `long_prefill_min_tokens` 的样本。

## 5. 服务时延模型

当前模型里：

- `prefill_time = prefill_tokens * prefill_ms_per_token`
- `service_duration = wait_time + prefill_time`

这点需要在 README 里明确，因为它决定了输出指标的解释方式。

## 如何使用 `run_compare.py`

运行方式：

```powershell
python .\long_prefill_sim\run_compare.py
```

注意：它不是 argparse CLI。当前正确使用方式是“修改脚本内配置后运行”。

## `run_compare.py` 里最重要的配置点

### `FIRST_REQUEST_ARRIVAL_INTERVAL_MS_GRID`

控制什么：

- 扫描的 first request 平均到达间隔

什么时候该改：

- 想调负载强弱
- 想看系统从轻载到重载的变化

改哪里：

- `run_compare.py` 文件顶部常量

### `WAIT_TIME_BUCKETS_MS`

控制什么：

- `wait_time <= threshold` 的累计占比统计区间

什么时候该改：

- 想更细看 50ms / 100ms / 200ms 以内占比
- 想按你自己的 SLA 切桶

### `Scenario`

控制什么：

- 单个实验场景的部署方式、实例拓扑、用户区域分布、zero-prefill 概率、路由策略

关键字段：

- `name`
- `mode`
- `instances`
- `region_names`
- `user_weights`
- `users_by_region_override`
- `zero_prob_by_region`
- `region_policy`
- `region_first_central_region_index`
- `region_first_outstanding_threshold`

### `FixedConfig`

控制什么：

- 所有场景共享的 workload 常量

关键字段：

- `total_users`
- `request_count`
- `think_time`
- `queue_wait_threshold_ms`
- `service_duration_target_ms`
- `seed_base`

什么时候该改：

- 想统一抬高或降低所有实验的压力
- 想让每个用户多轮请求更多

### `SweepConfig`

控制什么：

- 扫描网格

什么时候该改：

- 想扩大扫描区间
- 想减少扫描点提升运行速度

### `build_default_scenarios()`

控制什么：

- 当前批量实验默认会跑哪些场景

什么时候该改：

- 想加新部署方案
- 想比较新的 region policy

## `run_compare.py` 输出什么

当前重点输出：

- `outputs/compare_results.csv`
- `outputs/wait_time_bucket_ratios.csv`

仓库里如果存在 `.xlsx` 或其他分析文件，也属于实验产物，不是源码的一部分。

## 如何使用 `inspect_waits.py`

帮助：

```powershell
python .\long_prefill_sim\inspect_waits.py --help
```

## 参数列表

- `--mode`
  - `centralized` / `distributed`
- `--region-policy`
  - `weight_region_policy` / `queue_len_policy` / `d_queue_len_policy` / `region_first_policy`
- `--region-first-central-region-index`
  - central fallback region 的下标
- `--region-first-outstanding-threshold`
  - 本地域 backlog 到多少时回退 central
- `--total-users`
- `--first-arrival-interval-ms`
- `--request-count-min`
- `--request-count-max`
- `--think-time-mean-ms`
- `--think-time-std-ms`
- `--think-time-min-ms`
- `--think-time-max-ms`
- `--queue-threshold-ms`
- `--seed`
- `--top-n`

## 示例命令

普通 distributed：

```powershell
python .\long_prefill_sim\inspect_waits.py --mode distributed --region-policy weight_region_policy --total-users 180 --first-arrival-interval-ms 12000
```

`region_first_policy`：

```powershell
python .\long_prefill_sim\inspect_waits.py --mode distributed --region-policy region_first_policy --region-first-central-region-index 4 --region-first-outstanding-threshold 1 --total-users 180
```

centralized：

```powershell
python .\long_prefill_sim\inspect_waits.py --mode centralized --total-users 180 --first-arrival-interval-ms 12000
```

## 结果怎么看

`inspect_waits.py` 输出三层信息：

- `overall`
  - 当前整体统计
- `region_stats`
  - 各 region 的统计
- 最长等待请求列表
  - 用来看最差请求的 `arrival`、`wait`、`prefill`、`total`

### `--top-n`

控制打印多少条最差请求。想诊断更细就调大。

### `--queue-threshold-ms`

决定从多大等待时间开始算“queued request”。

### `--region-first-central-region-index`

只在 `region_first_policy` 下使用，表示哪个 region 是 central fallback。

### `--region-first-outstanding-threshold`

只在 `region_first_policy` 下使用，表示本地域积压到多大时转发到 central。

## 参数改动指南

如果你想做下面这些事，优先改这些参数。

### 想提高负载

- 改 `first_request_arrival_interval_ms`
  - 越小越重载
- 改 `total_users`
  - 越大整体请求量越高

### 想改请求长度分布

- 改 `PrefillConfig.lognormal_mu`
- 改 `PrefillConfig.lognormal_sigma`
- 改 `PrefillConfig.long_prefill_min_tokens`

### 想改区域用户分布

- 改 `users_by_region_override`
- 或改 `user_weights`

### 想改部署拓扑

- 改 `instances`
- 改 `region_names`

### 想改 fallback 策略

- 改 `region_policy`
- 改 `region_first_central_region_index`
- 改 `region_first_outstanding_threshold`

## 怎样新增一个场景

推荐步骤：

1. 在 `run_compare.py` 里新增一个 `Scenario(...)`
2. 接到 `build_default_scenarios()`
3. 确认：
   - `instances` 和 `region_names` 对齐
   - `zero_prob_by_region` 和 `region_names` 对齐
   - `users_by_region_override` 总和与 `FixedConfig.total_users` 对齐

## 怎样新增一个 batch family

推荐步骤：

1. 复制 [`run_compare.py`](C:/Project/cen_dis_sim/long_prefill_sim/run_compare.py)
2. 保留 [`simulation.py`](C:/Project/cen_dis_sim/long_prefill_sim/simulation.py) 不动
3. 只改：
   - 扫描网格
   - `Scenario`
   - 输出 CSV 列
   - 控制台摘要

## 输出解读

- `StatsSummary`
  - 单个请求集合的聚合指标：`avg_wait`、`max_wait`、`queued_ratio`、`within_target_ratio`
- `WaitBucketSummary`
  - `wait_time <= threshold` 的累计占比
- `requests_to_rows()`
  - 把 request 记录展开成行
- `summarize_wait_time_buckets()`
  - 生成等待时间分桶统计
- `expected_prefill_ms()`
  - 理论期望 `prefill` 时长
- `format_stats_table()`
  - 打印多组 `StatsSummary`

## 常见任务对照表

| 我想做什么 | 先改哪里 |
| --- | --- |
| 调高负载 | `first_request_arrival_interval_ms`、`total_users` |
| 改请求长度 | `PrefillConfig.lognormal_mu/sigma`、`long_prefill_min_tokens` |
| 改用户区域分布 | `users_by_region_override` 或 `user_weights` |
| 改部署方式 | `Scenario.instances`、`DeploymentConfig` |
| 看最差请求 | `inspect_waits.py` + `--top-n` |
| 加新场景 | `build_default_scenarios()` |
| 导出更多 CSV 字段 | `write_csv()`、`requests_to_rows()` |

## 限制说明

- 这是独立工具，不和主模拟器 `SimulationConfig` 互通
- `region_first_policy` 只在 long 版存在
- `outputs/` 是产物目录，不纳入版本控制
