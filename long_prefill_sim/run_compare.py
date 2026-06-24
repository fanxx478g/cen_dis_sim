from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    from .simulation import (
        DeploymentConfig,
        PrefillConfig,
        RequestCountConfig,
        SimulationConfig,
        StatsSummary,
        ThinkTimeConfig,
        WaitBucketSummary,
        Workload,
        allocate_users,
        expected_prefill_ms,
        format_stats_table,
        generate_workload,
        prepare_simulation_config,
        simulate_workload,
        summarize_wait_time_buckets,
    )
except ImportError:
    from simulation import (
        DeploymentConfig,
        PrefillConfig,
        RequestCountConfig,
        SimulationConfig,
        StatsSummary,
        ThinkTimeConfig,
        WaitBucketSummary,
        Workload,
        allocate_users,
        expected_prefill_ms,
        format_stats_table,
        generate_workload,
        prepare_simulation_config,
        simulate_workload,
        summarize_wait_time_buckets,
    )


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# 批量扫描的首请求平均到达间隔，单位 ms。
# 这是“全体用户首请求组成的全局泊松流”的平均到达间隔。
# 值越小，首请求到达越密集，系统负载越高。
FIRST_REQUEST_ARRIVAL_INTERVAL_MS_GRID = [i for i in range(300, 2000, 100)]

# 统计 wait_time <= threshold 的累计占比，单位 ms。
# 例如 50 表示“等待时间不超过 50ms 的请求占比”。
WAIT_TIME_BUCKETS_MS = [10.0, 20.0, 30.0, 50.0, 100.0, 200.0, 300.0, 500.0, 1000.0, 2000.0]


@dataclass(frozen=True)
class Scenario:
    """一个场景只描述一种部署方式和路由策略。

    这里的字段可以分成四类：
    1. 集群拓扑
       - mode: centralized 或 distributed
       - instances: centralized 时是总实例数；distributed 时是各地域实例数
       - region_names: 地域名称，顺序必须与 users_by_region / instances / zero_prob_by_region 对齐
    2. home 用户分布
       - user_weights: 未显式指定 users_by_region_override 时，用它按权重切分 total_users
       - users_by_region_override: 直接指定每个地域的 home 用户数；设置后会覆盖 user_weights
    3. 请求属性
       - zero_prob_by_region: 每个地域命中 zero-prefill 的概率
    4. 路由策略
       - region_policy: 用户首请求如何选择执行地域
       - region_first_*: 仅 region_first_policy 使用
    """

    name: str
    mode: str
    instances: int | tuple[int, ...]
    region_names: tuple[str, ...]
    user_weights: tuple[float, ...]
    users_by_region_override: tuple[int, ...] | None = None
    zero_prob_by_region: tuple[float, ...]
    region_policy: str = "weight_region_policy"
    # 仅 region_first_policy 使用。
    # 0-based 下标，指向 region_names 中哪个地域是 central fallback cluster。
    region_first_central_region_index: int | None = None
    # 仅 region_first_policy 使用。
    # 当本地地域的 “等待队列 + 服务中” 达到该阈值时，请求回退到 central。
    region_first_outstanding_threshold: int = 1


@dataclass(frozen=True)
class FixedConfig:
    """所有实验共享的固定参数。

    这些参数在一次 run_batch() 中对所有场景都相同。
    如果你想做“只换部署方式，不换输入请求”的对比，应优先把公共输入放在这里。
    """

    # 默认按“单实例每秒约 1 请求 * 5 实例 * 60 秒 * 60 分钟”设置总用户数。
    total_users: int = 1 * 5 * 60 * 60
    # 每个用户的总请求轮数。
    # 例如 [1, 1] 表示每用户恰好 1 个请求；[1, 10] 表示每用户随机 1~10 个请求。
    request_count: RequestCountConfig = RequestCountConfig(min_requests=1, max_requests=1)
    # 同一用户相邻两轮请求之间的 think time 分布。
    # 下一轮真实到达时间 = 上一轮真实完成时间 + think time。
    think_time: ThinkTimeConfig = ThinkTimeConfig(
        mean_ms=60 * 1000.0,
        std_ms=15 * 1000.0,
        min_ms=5 * 1000.0,
        max_ms=5 * 60 * 1000.0,
    )
    # 只有 wait_time > 该阈值，才计入 queued_count / queued_ratio。
    queue_wait_threshold_ms: float = 0.0
    # service_duration = wait_time + prefill_time。
    # within_target_* 统计以此目标时延为门槛。
    service_duration_target_ms: float = 5_000.0
    # 所有场景共用的随机种子基线。
    # 在保持其余 workload 参数相同的情况下，可保证不同场景比较时输入请求一致。
    seed_base: int = 42


@dataclass(frozen=True)
class SweepConfig:
    """批量扫描参数。

    这些参数定义“场景 x 负载”的扫描网格。
    """

    # 首请求泊松流的平均到达间隔扫描列表。
    first_request_arrival_interval_ms_grid: tuple[float, ...] = tuple(FIRST_REQUEST_ARRIVAL_INTERVAL_MS_GRID)
    # 生成等待时间分桶 CSV 时使用的阈值列表。
    wait_time_buckets_ms: tuple[float, ...] = tuple(WAIT_TIME_BUCKETS_MS)


def build_default_scenarios() -> list[Scenario]:
    """默认场景。

    后续新增场景时，优先只改这里。
    """

    region_names = ("r1", "r2", "r3", "r4", "r5")
    user_weights = (1.0, 1.0, 1.0, 1.0, 1.0)
    zero_prob_by_region = (0.0, 0.0, 0.0, 0.0, 0.0)
    region_first_region_names = ("local_a", "local_b", "local_c", "central")
    return [
        Scenario(
            name="centralized-5",
            mode="centralized",
            instances=5,
            region_names=region_names,
            user_weights=user_weights,
            zero_prob_by_region=zero_prob_by_region,
            region_policy="weight_region_policy",
        ),
        Scenario(
            name="weight-distributed-1x5",
            mode="distributed",
            instances=(1, 1, 1, 1, 1),
            region_names=region_names,
            user_weights=user_weights,
            zero_prob_by_region=zero_prob_by_region,
            region_policy="weight_region_policy",
        ),
        Scenario(
            name="queue_delay_distributed-1x5",
            mode="distributed",
            instances=(1, 1, 1, 1, 1),
            region_names=region_names,
            user_weights=user_weights,
            zero_prob_by_region=zero_prob_by_region,
            region_policy="d_queue_len_policy",
        ),
        Scenario(
            name="region-first-local3-plus-central",
            mode="distributed",
            # 3 个本地地域 + 1 个 central fallback 集群。
            instances=(1, 1, 1, 2),
            region_names=region_first_region_names,
            user_weights=(1.0, 1.0, 1.0, 1.0),
            # central 只做兜底执行，不承载 home 用户，因此最后一个地域为 0。
            users_by_region_override=(6000, 6000, 6000, 0),
            zero_prob_by_region=(0.0, 0.0, 0.0, 0.0),
            region_policy="region_first_policy",
            region_first_central_region_index=3,
            region_first_outstanding_threshold=1,
        ),
    ]


def run_batch(
    fixed_config: FixedConfig | None = None,
    sweep_config: SweepConfig | None = None,
) -> list[dict[str, Any]]:
    """按“场景 x 首请求负载”批量运行仿真。

    输入分层：
    - FixedConfig: 所有场景共用的 workload 输入
    - SweepConfig: 要扫描的负载点
    - Scenario: 每个部署场景自己的拓扑和路由策略
    """

    fixed = fixed_config or FixedConfig()
    sweep = sweep_config or SweepConfig()
    scenarios = build_default_scenarios()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("长 Prefill 离散事件仿真")
    print(f"结果目录: {OUTPUT_DIR}")
    print(f"固定总用户数: {fixed.total_users}")
    print(
        "每用户请求数区间: "
        f"[{fixed.request_count.min_requests}, {fixed.request_count.max_requests}]"
    )
    print(
        "后续请求 think time: "
        f"mean={fixed.think_time.mean_ms:.0f}ms, "
        f"std={fixed.think_time.std_ms:.0f}ms, "
        f"range=[{fixed.think_time.min_ms:.0f}, {fixed.think_time.max_ms:.0f}]ms"
    )
    print(f"随机种子基线: {fixed.seed_base}")
    print(f"扫描的首请求平均到达间隔: {list(sweep.first_request_arrival_interval_ms_grid)}")
    print(f"等待阈值分桶: {list(sweep.wait_time_buckets_ms)}")

    total_steps = len(scenarios) * len(sweep.first_request_arrival_interval_ms_grid)
    completed_steps = 0
    summary_rows: list[dict[str, Any]] = []
    wait_bucket_rows: list[dict[str, Any]] = []
    workload_cache: dict[tuple[Any, ...], Workload] = {}

    for scenario in scenarios:
        # 先确定这个场景下每个地域的 home 用户数。
        # 如果配置了 override，则直接使用；否则按 user_weights 切分 total_users。
        users_by_region = (
            list(scenario.users_by_region_override)
            if scenario.users_by_region_override is not None
            else allocate_users(fixed.total_users, scenario.user_weights)
        )
        validate_scenario_inputs(scenario, fixed, users_by_region)
        print()
        print(f"场景: {scenario.name}")
        print(
            "模式: "
            f"{scenario.mode} | "
            f"实例: {format_instances(scenario.instances)} | "
            f"地域绑定策略: {scenario.region_policy} | "
            f"地域: {', '.join(scenario.region_names)} | "
            f"users_by_region={format_user_region_display(scenario, users_by_region)} | "
            f"KV cache 命中概率: {list(scenario.zero_prob_by_region)}"
        )

        stats_by_interval: dict[float, StatsSummary] = {}
        for interval_ms in sweep.first_request_arrival_interval_ms_grid:
            # 所有场景都用同一个 seed_base 构造 workload。
            # 这样当 workload 参数相同时，不同场景看到的是同一批输入请求。
            simulation_config = prepare_simulation_config(
                build_simulation_config(
                    scenario=scenario,
                    users_by_region=users_by_region,
                    fixed_config=fixed,
                    first_request_arrival_interval_ms=float(interval_ms),
                    seed=fixed.seed_base,
                )
            )
            workload = get_or_create_workload(
                cache=workload_cache,
                cache_key=build_workload_cache_key(
                    scenario=scenario,
                    users_by_region=users_by_region,
                    fixed_config=fixed,
                    first_request_arrival_interval_ms=float(interval_ms),
                ),
                simulation_config=simulation_config,
            )
            result = simulate_workload(
                workload,
                simulation_config,
                DeploymentConfig(mode=scenario.mode, instances=scenario.instances),
            )

            stats_by_interval[float(interval_ms)] = result.overall_stats
            summary_rows.append(
                build_summary_row(
                    scenario=scenario,
                    result=result,
                    simulation_config=simulation_config,
                    workload=workload,
                    users_by_region=users_by_region,
                )
            )
            wait_bucket_rows.extend(
                build_wait_bucket_rows(
                    scenario=scenario,
                    result=result,
                    simulation_config=simulation_config,
                    users_by_region=users_by_region,
                    first_request_arrival_interval_ms=simulation_config.first_request_arrival_interval_ms,
                    thresholds_ms=sweep.wait_time_buckets_ms,
                )
            )

            completed_steps += 1
            print(
                build_step_progress_summary(
                    completed_steps=completed_steps,
                    total_steps=total_steps,
                    scenario=scenario,
                    interval_ms=float(interval_ms),
                    workload_total_requests=workload.total_requests,
                    stats=result.overall_stats,
                )
            )

        highest_load_interval = min(stats_by_interval)
        highest_load_stats = stats_by_interval[highest_load_interval]
        print(format_stats_table([(scenario.name, highest_load_stats)]))
        print(f"上表对应首请求平均到达间隔: {highest_load_interval:.0f}ms")
        print(build_single_scenario_summary(highest_load_stats))

    write_csv(OUTPUT_DIR / "compare_results.csv", summary_rows)
    write_csv(OUTPUT_DIR / "wait_time_bucket_ratios.csv", wait_bucket_rows)
    print()
    print(f"CSV 已写出: {OUTPUT_DIR / 'compare_results.csv'}")
    print(f"等待阈值占比 CSV 已写出: {OUTPUT_DIR / 'wait_time_bucket_ratios.csv'}")
    print_overall_takeaways(summary_rows)
    return summary_rows


def build_simulation_config(
    scenario: Scenario,
    users_by_region: list[int],
    fixed_config: FixedConfig,
    first_request_arrival_interval_ms: float,
    seed: int,
) -> SimulationConfig:
    """把 run_compare 的三层配置组装成 simulation.py 可直接消费的配置对象。"""

    return SimulationConfig(
        region_names=scenario.region_names,
        users_by_region=users_by_region,
        first_request_arrival_interval_ms=first_request_arrival_interval_ms,
        region_policy=scenario.region_policy,
        request_count=fixed_config.request_count,
        think_time=fixed_config.think_time,
        queue_wait_threshold_ms=fixed_config.queue_wait_threshold_ms,
        service_duration_target_ms=fixed_config.service_duration_target_ms,
        region_first_central_region_index=scenario.region_first_central_region_index,
        region_first_outstanding_threshold=scenario.region_first_outstanding_threshold,
        seed=seed,
        prefill=PrefillConfig(zero_prob_by_region=scenario.zero_prob_by_region),
    )


def build_workload_cache_key(
    scenario: Scenario,
    users_by_region: list[int],
    fixed_config: FixedConfig,
    first_request_arrival_interval_ms: float,
) -> tuple[Any, ...]:
    """只包含影响 workload 生成的参数。

    注意：
    - 不包含部署模式和实例数
    - 不包含 region_policy，因为 policy 影响的是仿真阶段的地域绑定，不影响输入请求生成
    - 只要这个 key 相同，不同场景就会复用同一份 workload
    """

    return (
        tuple(scenario.region_names),
        tuple(users_by_region),
        tuple(float(value) for value in scenario.zero_prob_by_region),
        float(first_request_arrival_interval_ms),
        fixed_config.request_count.min_requests,
        fixed_config.request_count.max_requests,
        fixed_config.think_time.mean_ms,
        fixed_config.think_time.std_ms,
        fixed_config.think_time.min_ms,
        fixed_config.think_time.max_ms,
        fixed_config.seed_base,
    )


def get_or_create_workload(
    cache: dict[tuple[Any, ...], Workload],
    cache_key: tuple[Any, ...],
    simulation_config: SimulationConfig,
) -> Workload:
    """按 workload key 复用输入请求，避免同一组输入被重复生成。"""

    workload = cache.get(cache_key)
    if workload is not None:
        return workload

    workload = generate_workload(simulation_config)
    cache[cache_key] = workload
    return workload


def build_summary_row(
    scenario: Scenario,
    result: Any,
    simulation_config: SimulationConfig,
    workload: Workload,
    users_by_region: list[int],
) -> dict[str, Any]:
    """生成 compare_results.csv 的一行。"""

    stats = result.overall_stats
    return {
        "scenario": scenario.name,
        "mode": result.mode,
        "instances": format_instances(scenario.instances),
        "region_policy": scenario.region_policy,
        "seed": simulation_config.seed,
        "total_users": sum(users_by_region),
        "users_by_region": format_user_region_display(scenario, users_by_region),
        "region_names": "|".join(scenario.region_names),
        "request_count_min": simulation_config.request_count.min_requests,
        "request_count_max": simulation_config.request_count.max_requests,
        "workload_total_requests": workload.total_requests,
        "first_request_arrival_interval_ms": simulation_config.first_request_arrival_interval_ms,
        "think_time_mean_ms": simulation_config.think_time.mean_ms,
        "think_time_std_ms": simulation_config.think_time.std_ms,
        "think_time_min_ms": simulation_config.think_time.min_ms,
        "think_time_max_ms": simulation_config.think_time.max_ms,
        "request_count": stats.count,
        "queued_count": stats.queued_count,
        "queued_ratio": stats.queued_ratio,
        "avg_duration": stats.avg_duration,
        "max_duration": stats.max_duration,
        "avg_wait": stats.avg_wait,
        "max_wait": stats.max_wait,
        "within_target_count": stats.within_target_count,
        "within_target_ratio": stats.within_target_ratio,
        "region_first_central_region_index": simulation_config.region_first_central_region_index,
        "region_first_outstanding_threshold": simulation_config.region_first_outstanding_threshold,
        "expected_nonzero_prefill_ms": expected_prefill_ms(simulation_config),
    }


def build_wait_bucket_rows(
    scenario: Scenario,
    result: Any,
    simulation_config: SimulationConfig,
    users_by_region: list[int],
    first_request_arrival_interval_ms: float,
    thresholds_ms: Sequence[float],
) -> list[dict[str, Any]]:
    """生成 wait_time_bucket_ratios.csv 的多行。"""

    rows: list[dict[str, Any]] = []
    bucket_stats: list[WaitBucketSummary] = summarize_wait_time_buckets(result.requests, thresholds_ms)
    request_count = len(result.requests)

    for bucket in bucket_stats:
        rows.append(
            {
                "scenario": scenario.name,
                "mode": result.mode,
                "instances": format_instances(scenario.instances),
                "region_policy": scenario.region_policy,
                "total_users": sum(users_by_region),
                "users_by_region": format_user_region_display(scenario, users_by_region),
                "region_names": "|".join(scenario.region_names),
                "first_request_arrival_interval_ms": first_request_arrival_interval_ms,
                "request_count": request_count,
                "wait_threshold_ms": bucket.threshold_ms,
                "wait_count": bucket.wait_count,
                "wait_ratio": bucket.wait_ratio,
                "region_first_central_region_index": simulation_config.region_first_central_region_index,
                "region_first_outstanding_threshold": simulation_config.region_first_outstanding_threshold,
            }
        )
    return rows


def build_step_progress_summary(
    completed_steps: int,
    total_steps: int,
    scenario: Scenario,
    interval_ms: float,
    workload_total_requests: int,
    stats: StatsSummary,
) -> str:
    return (
        f"[{completed_steps}/{total_steps}] {scenario.name} | "
        f"mode={scenario.mode} | "
        f"first_request_arrival_interval_ms={interval_ms:.0f} | "
        f"requests={workload_total_requests}\n"
        f"  avg_wait={stats.avg_wait:.1f}ms, "
        f"max_wait={stats.max_wait:.1f}ms, "
        f"queued_ratio={stats.queued_ratio:.2%}, "
        f"within_target_ratio={stats.within_target_ratio:.2%}"
    )


def build_single_scenario_summary(stats: StatsSummary) -> str:
    return (
        "摘要: "
        f"avg_wait={stats.avg_wait:.1f}ms, "
        f"max_wait={stats.max_wait:.1f}ms, "
        f"queued_ratio={stats.queued_ratio:.2%}, "
        f"within_target_ratio={stats.within_target_ratio:.2%}"
    )


def format_instances(instances: int | tuple[int, ...]) -> str:
    if isinstance(instances, int):
        return str(instances)
    return "|".join(str(item) for item in instances)


def format_user_region_display(scenario: Scenario, users_by_region: list[int]) -> str:
    # queue_len / d_queue_len 是到达时动态选地域，没有固定 users_by_region 展示意义。
    if scenario.mode == "distributed" and scenario.region_policy == "queue_len_policy":
        return "dynamic_by_queue_len"
    if scenario.mode == "distributed" and scenario.region_policy == "d_queue_len_policy":
        return "dynamic_by_queue_snapshot"
    return "|".join(str(item) for item in users_by_region)


def validate_scenario_inputs(
    scenario: Scenario,
    fixed_config: FixedConfig,
    users_by_region: list[int],
) -> None:
    """在 run_compare 层做一层轻量校验，尽早发现配置错误。"""

    if len(users_by_region) != len(scenario.region_names):
        raise ValueError(
            f"{scenario.name}: users_by_region 长度必须与 region_names 一致。"
        )
    if len(scenario.zero_prob_by_region) != len(scenario.region_names):
        raise ValueError(
            f"{scenario.name}: zero_prob_by_region 长度必须与 region_names 一致。"
        )
    if scenario.mode == "distributed" and isinstance(scenario.instances, tuple):
        if len(scenario.instances) != len(scenario.region_names):
            raise ValueError(
                f"{scenario.name}: distributed 场景的 instances 长度必须与 region_names 一致。"
            )
    if sum(users_by_region) != fixed_config.total_users:
        raise ValueError(
            f"{scenario.name}: users_by_region 总和必须等于 fixed.total_users。"
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_overall_takeaways(rows: list[dict[str, Any]]) -> None:
    print()
    print("控制台摘要")
    for scenario_name in sorted({str(row["scenario"]) for row in rows}):
        scenario_rows = [row for row in rows if row["scenario"] == scenario_name]
        highest_load_row = min(
            scenario_rows,
            key=lambda row: float(row["first_request_arrival_interval_ms"]),
        )
        print(
            f"{scenario_name}: 在首请求平均到达间隔 "
            f"{float(highest_load_row['first_request_arrival_interval_ms']):.0f}ms 下，"
            f"avg_wait={float(highest_load_row['avg_wait']):.1f}ms / "
            f"queued_ratio={float(highest_load_row['queued_ratio']):.2%} / "
            f"within_target_ratio={float(highest_load_row['within_target_ratio']):.2%}"
        )


if __name__ == "__main__":
    run_batch()
