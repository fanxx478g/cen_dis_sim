# short_prefill_sim

`short_prefill_sim` 是一个独立的离散事件模拟工具，专门研究短 `prefill` 请求在不同部署方式和区域路由策略下的等待时间、排队比例和服务时延。

它和 `long_prefill_sim` 结构相似，但研究对象、参数约束和支持的路由策略不同，不能直接共用同一份说明。

## 这个模块解决什么问题

它主要回答下面这类问题：

- 短上下文请求在 `centralized` 和 `distributed` 部署下等待时间差异多大？
- `weight_region_policy`、`queue_len_policy`、`d_queue_len_policy` 会把请求分配成什么样？
- 首轮到达更密集时，`queued_ratio` 和 `within_target_ratio` 如何变化？
- 固定配置下，最严重的排队请求是什么样？

## 先看什么文件

推荐阅读顺序：

1. [`simulation.py`](C:/Project/cen_dis_sim/short_prefill_sim/simulation.py)
2. [`run_compare.py`](C:/Project/cen_dis_sim/short_prefill_sim/run_compare.py)
3. [`inspect_waits.py`](C:/Project/cen_dis_sim/short_prefill_sim/inspect_waits.py)

## 先怎么跑起来

### 第 1 步：跑 batch

```powershell
python .\short_prefill_sim\run_compare.py
```

你会看到：

- 场景执行日志
- 每个扫描点的摘要
- `outputs/compare_results.csv`
- `outputs/wait_time_bucket_ratios.csv`

### 第 2 步：看输出文件

优先看：

- `outputs/compare_results.csv`
- `outputs/wait_time_bucket_ratios.csv`

### 第 3 步：跑单配置诊断

```powershell
python .\short_prefill_sim\inspect_waits.py --help
```

再选一个具体配置做诊断。

例如：

```powershell
python .\short_prefill_sim\inspect_waits.py --mode distributed --region-policy weight_region_policy --total-users 180 --first-arrival-interval-ms 12000
```

如果你口头上说 `run_compare_instance`，当前仓库里对应的实际脚本仍然是 `inspect_waits.py`。

## 核心机制

## 1. 部署模式

支持：

- `centralized`
- `distributed`

## 2. 区域路由策略

当前 `UserRegionPolicy` 支持：

- `weight_region_policy`
- `queue_len_policy`
- `d_queue_len_policy`

这里明确不支持 `region_first_policy`。

## 3. workload 生成方式

同样是两段式：

1. 所有用户的 first request 构成全局到达流
2. 同一用户的 follow-up 请求，到达时间 = 上一轮真实完成时间 + `think time`

## 4. 短 prefill token 采样

关键参数在 `PrefillConfig`：

- `lognormal_mu`
- `lognormal_sigma`
- `truncated_min_tokens`
- `truncated_max_tokens`
- `short_prefill_max_tokens`

当前实现只保留不大于 `short_prefill_max_tokens` 的样本。

## 5. 服务时延模型

当前模型里：

- `prefill_time = prefill_tokens * prefill_ms_per_token`
- `service_duration = wait_time + prefill_time`

## 如何使用 `run_compare.py`

运行方式：

```powershell
python .\short_prefill_sim\run_compare.py
```

注意：它不是 argparse CLI，而是“编辑脚本内配置后运行”。

## `run_compare.py` 里最重要的配置点

### `FIRST_REQUEST_ARRIVAL_INTERVAL_MS_GRID`

控制扫描的首轮请求平均到达间隔。

适合在想比较轻载/重载行为时修改。

### `WAIT_TIME_BUCKETS_MS`

控制等待时间累计占比的分桶边界。

### `Scenario`

控制单个实验场景：

- `name`
- `mode`
- `instances`
- `region_names`
- `user_weights`
- `zero_prob_by_region`
- `region_policy`

### `FixedConfig`

控制所有场景共享的 workload 常量：

- `total_users`
- `request_count`
- `think_time`
- `queue_wait_threshold_ms`
- `service_duration_target_ms`
- `seed_base`

### `SweepConfig`

控制 batch 扫描网格。

### `build_default_scenarios()`

控制默认场景集合。

当前默认 scenarios 是：

- `centralized-5`
- `weight-distributed-1x5`
- `queue_delay_distributed-1x5`

## `run_compare.py` 输出什么

当前重点输出：

- `outputs/compare_results.csv`
- `outputs/wait_time_bucket_ratios.csv`

## 如何使用 `inspect_waits.py`

帮助：

```powershell
python .\short_prefill_sim\inspect_waits.py --help
```

## 参数列表

- `--mode`
  - `centralized` / `distributed`
- `--region-policy`
  - `weight_region_policy` / `queue_len_policy` / `d_queue_len_policy`
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
python .\short_prefill_sim\inspect_waits.py --mode distributed --region-policy weight_region_policy --total-users 180 --first-arrival-interval-ms 12000
```

按队列长度动态分配：

```powershell
python .\short_prefill_sim\inspect_waits.py --mode distributed --region-policy queue_len_policy --total-users 180 --first-arrival-interval-ms 12000
```

centralized：

```powershell
python .\short_prefill_sim\inspect_waits.py --mode centralized --total-users 180 --first-arrival-interval-ms 12000
```

## 结果怎么看

输出结构和 long 版一致：

- `overall`
- `region_stats`
- 等待时间最长的前 `N` 条请求

`--top-n` 用于控制打印多少条最差请求。

## 参数改动指南

### 想提高负载

- 改 `first_request_arrival_interval_ms`
- 改 `total_users`

### 想改请求长度分布

- 改 `PrefillConfig.lognormal_mu/sigma`
- 改 `PrefillConfig.short_prefill_max_tokens`

### 想改用户区域分布

- 改 `user_weights`

### 想改部署拓扑

- 改 `instances`
- 改 `region_names`

### 想改路由行为

- 改 `region_policy`

## 怎样扩实验

最常见的扩法：

1. 在 [`run_compare.py`](C:/Project/cen_dis_sim/short_prefill_sim/run_compare.py) 里新增 `Scenario`
2. 修改 `build_default_scenarios()`
3. 修改 `FixedConfig.total_users`
4. 修改 `FIRST_REQUEST_ARRIVAL_INTERVAL_MS_GRID`
5. 如需不同 SLA 分桶，改 `WAIT_TIME_BUCKETS_MS`

如果只是看某个具体配置的异常请求，不需要改 `run_compare.py`，直接运行 `inspect_waits.py`。

## 输出解读

- `StatsSummary`
  - 单个请求集合的聚合指标
- `WaitBucketSummary`
  - `wait_time <= threshold` 的累计占比
- `requests_to_rows()`
  - 把 request 记录展开成表格行
- `summarize_wait_time_buckets()`
  - 生成等待时间分桶统计
- `expected_prefill_ms()`
  - 理论期望 `prefill` 时长
- `format_stats_table()`
  - 格式化打印统计表

## 与 `long_prefill_sim` 的区别

- 本模块研究短 `prefill` 请求
- 关键长度约束参数是 `short_prefill_max_tokens`
- 不支持 `region_first_policy`
- 场景集合和参数含义不能直接照搬 long 版

## 常见任务对照表

| 我想做什么 | 先改哪里 |
| --- | --- |
| 调高负载 | `first_request_arrival_interval_ms`、`total_users` |
| 改请求长度 | `PrefillConfig.lognormal_mu/sigma`、`short_prefill_max_tokens` |
| 改用户区域分布 | `user_weights` |
| 改部署方式 | `Scenario.instances` |
| 看最差请求 | `inspect_waits.py` + `--top-n` |
| 加新场景 | `build_default_scenarios()` |
| 导出更多 CSV 字段 | `write_csv()`、`requests_to_rows()` |

## 限制说明

- 这是独立工具，不和主模拟器 `SimulationConfig` 互通
- `outputs/` 是运行产物目录，不进入版本控制
