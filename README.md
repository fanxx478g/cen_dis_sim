# cen_dis_sim

`cen_dis_sim` 是一个面向大模型推理场景的离散事件模拟项目。当前仓库包含三条相互关联、但职责不同的功能线：

- `simulator/` + `cases/`
  - 主模拟器。建模 `prefill -> decode` 生命周期、`cluster -> pool -> instance` 资源层级、batching、routing policy、跨集群 decode 和指标统计。
- `long_prefill_sim/`
  - 独立的长 `prefill` 等待时间分析器。适合研究长上下文请求在不同部署方式和区域路由策略下的等待时间与排队行为。
- `short_prefill_sim/`
  - 独立的短 `prefill` 等待时间分析器。适合研究短上下文请求的等待时间、区域分配和批量对比实验。

这三部分不是同一套代码的不同入口，而是“主框架 + 两个独立研究工具”。

## 仓库地图

- [`simulator/`](C:/Project/cen_dis_sim/simulator)
  - 主模拟器核心实现：配置模型、请求生成、调度器、引擎、metrics、routing policy。
- [`cases/`](C:/Project/cen_dis_sim/cases)
  - 主模拟器的可运行场景、批量对比脚本、日志汇总工具。
- [`test/`](C:/Project/cen_dis_sim/test)
  - 主模拟器回归测试，覆盖 routing、batching、request generation、metrics。
- [`long_prefill_sim/`](C:/Project/cen_dis_sim/long_prefill_sim)
  - 长 `prefill` 独立实验工具。
- [`short_prefill_sim/`](C:/Project/cen_dis_sim/short_prefill_sim)
  - 短 `prefill` 独立实验工具。

## 5 分钟上手

如果你第一次接触这个仓库，按下面顺序做。

### 第 1 步：跑主模拟器最小 demo

```powershell
python .\cases\full_config_demo.py
```

你会看到：

- `=== Summary ===`
- `=== Pool Breakdown ===`
- `=== Request Breakdown ===`

这一脚本的作用是让你快速看到主模拟器的输入配置、总体指标、pool 级统计和单请求明细是怎么组织的。

### 第 2 步：跑测试确认环境和主行为

```powershell
python -m pytest -q
```

你会看到测试通过数量。主模拟器、routing policy、metrics 语义等关键行为都由这些测试兜底。

### 第 3 步：跑一个固定 case

```powershell
python .\cases\case_distributed_2_clusters_global_summary.py
```

你会看到：

- `=== Global Summary ===`
- 一组全局指标字典
- 仿真运行时间

这一类 `case_*_global_summary.py` 是最推荐的“改配置后直接运行”的入口。

### 第 4 步：跑一个 batch 工具

```powershell
python .\cases\batch_compare_distributed_vs_centralized_2_clusters.py
```

你会看到：

- 多个场景/参数组合的执行日志
- 生成到 `cases/log/` 下的 `.log` 文件
- 汇总 CSV

这一类脚本适合做多组配置对比，而不是只看单个 case。

### 第 5 步：跑独立 prefill 分析器

长 `prefill`：

```powershell
python .\long_prefill_sim\run_compare.py
python .\long_prefill_sim\inspect_waits.py --help
```

短 `prefill`：

```powershell
python .\short_prefill_sim\run_compare.py
python .\short_prefill_sim\inspect_waits.py --help
```

这两个目录是独立研究工具，不复用主模拟器的 `SimulationConfig`。

## 主模拟器核心机制

## 1. 请求生成机制

主模拟器用 `RequestGenerationConfig` 生成多用户、多轮对话请求。

核心机制：

- 新用户首轮请求按 Poisson 过程到达
  - 对应参数：`new_user_arrival_mean_seconds`
- 同一用户的 follow-up 请求，相对上一轮真实完成时间到达
  - 对应参数：`followup_arrival_mean_seconds`、`followup_arrival_std_seconds`
- 每个 session 的轮数在 `min_turns_per_user` 到 `max_turns_per_user` 之间均匀采样
- 首轮请求会在 short/long prompt bucket 中采样
  - 对应参数：`short_context_probability`
  - `prompt_len_threshold` 决定后续由哪个 prefill pool 处理

### `accumulate_context_across_turns` 的影响

- `False`
  - 每一轮都回到首轮 prompt 尺度
- `True`
  - 后续轮次会累加历史上下文和输出 token

这决定了“多轮对话是否会逐步长上下文化”。

## 2. `prefill` / `decode` 分工

主模拟器把请求拆成两个阶段：

- `prefill`
  - 负责处理 prompt
- `decode`
  - 负责处理输出 token 生成

`Scheduler._prefill_kind_for_request()` 用 `prompt_len_threshold` 判断：

- `prompt_tokens > prompt_len_threshold`
  - 走 `LONG_PREFILL`
- 否则
  - 走 `SHORT_PREFILL`

decode 选池有两层约束：

- 首次 decode
  - 受 `allow_first_decode_cross_cluster` 控制
- 后续 decode
  - 受 `allow_following_decode_cross_cluster` 控制
  - 如果关闭，后续 decode 会尽量固定在首次绑定的 decode pool

## 3. 资源层级

主模拟器资源层级是：

```text
cluster -> pool -> instance
```

- `cluster`
  - 部署单元，例如 central / edge-a / edge-b
- `pool`
  - 同类资源池，例如 `SHORT_PREFILL` / `LONG_PREFILL` / `DECODE`
- `instance`
  - pool 内的并行服务实例

`max_batch_size` 控制一个 pool 一次最多从等待队列里取多少请求形成 batch。

## 4. routing policy 机制

当前主模拟器至少有两个内置策略：

- `p_default`
  - 从每个 cluster 里选出当前“最优” pool，再按 cluster 权重做选择
- `dis_first`
  - 对 `SHORT_PREFILL` 优先尝试 distributed cluster；如果 distributed pool 不够轻，再考虑 central pool

策略入口在：

- [`simulator/policies/default_policy.py`](C:/Project/cen_dis_sim/simulator/policies/default_policy.py)
- [`simulator/policies/dis_first_policy.py`](C:/Project/cen_dis_sim/simulator/policies/dis_first_policy.py)

## 5. 指标机制

主模拟器有两层汇总：

- `metrics.summary()`
  - 请求级和批级指标的完整汇总
- `metrics.global_summary()`
  - 更适合全局报告和 batch 日志汇总的精简指标

当前常见指标含义：

- `prefill_first_token_latency_*`
  - 当前模拟器里近似表示“prefill 完成到首 token”的历史语义
- `request_tpot_*`
  - request 级 TPOT
- `system_tpot_avg_ms`
  - 系统平均 TPOT
- `prefill_queue_*`
  - prefill 等待时间
- `decode_queue_*`
  - decode 等待时间
- `request_throughput_rps`
- `output_token_throughput_tps`
- `decode_token_throughput_tps`
- `prefill_token_throughput_tps`

## 主模拟器参数怎么改

这部分按“参数组 + 改法 + 常见用途”来写。

## 1. `RequestGenerationConfig`

常用字段：

- `user_count`
- `min_turns_per_user` / `max_turns_per_user`
- `new_user_arrival_mean_seconds`
- `followup_arrival_mean_seconds` / `followup_arrival_std_seconds`
- `accumulate_context_across_turns`
- `followup_prompt_tokens`
- `short_context_probability`
- `initial_prompt_variation_ratio`
- `short_context_prompt_tokens` / `long_context_prompt_tokens`
- `short_context_output_tokens_min/max`
- `long_context_output_tokens_min/max`

控制的行为：

- 总负载强度
- 多轮对话深度
- 首轮流量密度
- follow-up 密度
- prompt 长度分布
- 输出长度分布

常见修改场景：

- 想提高整体压力
  - 增大 `user_count`
  - 减小 `new_user_arrival_mean_seconds`
- 想模拟更多多轮对话
  - 增大 `max_turns_per_user`
  - 打开 `accumulate_context_across_turns`
- 想让 long context 比例更高
  - 减小 `short_context_probability`

通常在哪改：

- 直接在某个 `cases/*.py` 中的 `RequestGenerationConfig(...)` 改

## 2. `SchedulerConfig`

常用字段：

- `policy_name`
- `policy_config`
- `prompt_len_threshold`
- `allow_first_decode_cross_cluster`
- `allow_following_decode_cross_cluster`

控制的行为：

- 选哪种 routing policy
- policy 的权重/阈值配置
- long/short prefill 的分界
- decode 是否允许跨 cluster / 跨 pool

常见修改场景：

- 想切策略
  - 改 `policy_name`
- 想调整 `dis_first` 的行为
  - 改 `policy_config`
- 想让更多请求进入 `LONG_PREFILL`
  - 调低 `prompt_len_threshold`

通常在哪改：

- 各类 `case_*_global_summary.py`
- `cases/full_config_demo.py`

## 3. `ScenarioConfig`

核心内容：

- cluster 数量
- 每个 cluster 内有哪些 pool
- 每个 pool 的 `kind`
- 每个 pool 的 `instance_count`
- 每个 pool 的 `max_batch_size`
- `central_cluster_id`

控制的行为：

- 部署拓扑
- 各阶段资源容量
- centralized / distributed 结构差异

常见修改场景：

- 想从 2 集群改成 3 集群
  - 新增一个 `cluster(...)`
- 想加 central long prefill 池
  - 在 central cluster 里新增 `LONG_PREFILL` pool
- 想增加 decode 并发能力
  - 增大 `DECODE` pool 的 `instance_count`

通常在哪改：

- 各个 `cases/case_*_global_summary.py`

## 4. `PerformanceConfig`

字段：

- `short_prefill_ms_per_token`
- `long_prefill_ms_per_token`
- `prefill_batch_size_penalty`
- `decode_base_step_ms`
- `decode_batch_size_slope_ms`

控制的行为：

- prefill 阶段时延模型
- decode 阶段时延模型

常见修改场景：

- 想模拟更快的 prefill
  - 降低对应 `*_prefill_ms_per_token`
- 想模拟 decode batch 更敏感
  - 调整 `decode_batch_size_slope_ms`

通常在哪改：

- `SimulationConfig` 的默认值
- 或单个 case 内的 `performance=...`

## 5. `LoggingConfig`

字段：

- `level`
- `log_to_console`
- `log_to_file`
- `log_file_path`
- `logger_name`

控制的行为：

- 只打印 summary，还是输出详细事件日志
- 是否把日志落到文件

常见修改场景：

- 想做 batch 日志分析
  - 打开 `log_to_file`
- 想避免控制台过多输出
  - 提高级别或关闭 console

## 怎样新增或修改一个主模拟器 case

推荐工作流：

1. 复制 [`cases/full_config_demo.py`](C:/Project/cen_dis_sim/cases/full_config_demo.py) 或最接近的 `case_*_global_summary.py`
2. 修改 `RequestGenerationConfig`
3. 修改 `SchedulerConfig`
4. 修改 `ScenarioConfig`
5. 运行脚本看 `metrics.summary()`
6. 如需记录日志或汇总 CSV，接入 [`cases/case_batch_utils.py`](C:/Project/cen_dis_sim/cases/case_batch_utils.py)

## 最小改 case 示例

### 示例 1：把 2 集群场景改成 3 集群

可以从 [`cases/case_distributed_2_clusters_global_summary.py`](C:/Project/cen_dis_sim/cases/case_distributed_2_clusters_global_summary.py) 复制：

- 新增一个 `cluster("cluster-3", pools=[...])`
- 把它加入 `ScenarioConfig.clusters`
- 保持其余配置不变，先验证拓扑变化的影响

### 示例 2：把 `policy_name` 从 `p_default` 换成 `dis_first`

在 `SchedulerConfig(...)` 里：

- `policy_name="dis_first"`
- 再根据需要补 `policy_config`
  - 例如 `distributed_cluster_routing_weights`
  - 例如 `short_prefill_queue_threshold`

## 怎样扩 routing policy / metrics / batch 工具

### 想加新 routing policy

最短路径：

1. 在 [`simulator/policies/`](C:/Project/cen_dis_sim/simulator/policies) 新增 policy 文件
2. 在 [`simulator/policies/__init__.py`](C:/Project/cen_dis_sim/simulator/policies/__init__.py) 注册
3. 如有公共逻辑，放进 `common.py`
4. 在 [`test/test_routing_policy.py`](C:/Project/cen_dis_sim/test/test_routing_policy.py) 补测试

### 想改 metrics

入口：

- [`simulator/metrics.py`](C:/Project/cen_dis_sim/simulator/metrics.py)

重点区分：

- `summary()`
  - 详细指标
- `global_summary()`
  - 全局报告 / batch 汇总友好指标

### 想加批量实验日志和 CSV 汇总

入口：

- [`cases/case_batch_utils.py`](C:/Project/cen_dis_sim/cases/case_batch_utils.py)
- [`cases/summarize_batch_logs.py`](C:/Project/cen_dis_sim/cases/summarize_batch_logs.py)

常用函数：

- `apply_overrides()`
- `prepare_logging_config()`
- `run_config_with_file_logging()`
- `collect_batch_log_rows()`
- `export_batch_log_summary_csv()`

## 常见任务对照表

| 我想做什么 | 先看哪里 |
| --- | --- |
| 跑主模拟器 demo | `cases/full_config_demo.py` |
| 改一个现有 case | `cases/case_*_global_summary.py` |
| 新增 routing policy | `simulator/policies/` |
| 改 metrics | `simulator/metrics.py` |
| 加 batch 日志/汇总 | `cases/case_batch_utils.py` |
| 跑独立 prefill 分析 | `long_prefill_sim/` 或 `short_prefill_sim/` |

## 下一步看哪里

- 想做长 `prefill` 排队分析：
  - [long_prefill_sim/README.md](C:/Project/cen_dis_sim/long_prefill_sim/README.md)
- 想做短 `prefill` 排队分析：
  - [short_prefill_sim/README.md](C:/Project/cen_dis_sim/short_prefill_sim/README.md)
