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
FIRST_REQUEST_ARRIVAL_INTERVAL_MS_GRID = [i for i in range(300, 2000, 100)]

# 统计 wait_time <= threshold 的累计占比，单位 ms。
WAIT_TIME_BUCKETS_MS = [10.0, 20.0, 30.0, 50.0, 100.0, 200.0, 300.0, 500.0, 1000.0, 2000.0]


@dataclass(frozen=True)
class Scenario:
    """一个场景只描述一种部署方式。"""

    name: str
    mode: str
    instances: int | tuple[int, ...]
    region_names: tuple[str, ...]
    user_weights: tuple[float, ...]
    zero_prob_by_region: tuple[float, ...]
    region_policy: str = "weight_region_policy"


@dataclass(frozen=True)
class FixedConfig:
    """所有实验共享的固定参数。"""

    # 默认按“单实例每秒约 1 请求 * 5 实例 * 60 秒 * 60 分钟”设置总用户数。
    total_users: int = 1 * 5 * 60 * 60
    request_count: RequestCountConfig = RequestCountConfig(min_requests=1, max_requests=1)
    think_time: ThinkTimeConfig = ThinkTimeConfig(
        mean_ms=60 * 1000.0,
        std_ms=15 * 1000.0,
        min_ms=5 * 1000.0,
        max_ms=5 * 60 * 1000.0,
    )
    queue_wait_threshold_ms: float = 0.0
    service_duration_target_ms: float = 5_000.0
    seed_base: int = 42


@dataclass(frozen=True)
class SweepConfig:
    """批量扫描参数。"""

    first_request_arrival_interval_ms_grid: tuple[float, ...] = tuple(FIRST_REQUEST_ARRIVAL_INTERVAL_MS_GRID)
    wait_time_buckets_ms: tuple[float, ...] = tuple(WAIT_TIME_BUCKETS_MS)


def build_default_scenarios() -> list[Scenario]:
    """默认场景。

    后续新增场景时，优先只改这里。
    """

    region_names = ("r1", "r2", "r3", "r4", "r5")
    user_weights = (1.0, 1.0, 1.0, 1.0, 1.0)
    zero_prob_by_region = (0.0, 0.0, 0.0, 0.0, 0.0)
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
    ]


def run_batch(
    fixed_config: FixedConfig | None = None,
    sweep_config: SweepConfig | None = None,
) -> list[dict[str, Any]]:
    fixed = fixed_config or FixedConfig()
    sweep = sweep_config or SweepConfig()
    scenarios = build_default_scenarios()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("短 Prefill 离散事件仿真")
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
        users_by_region = allocate_users(fixed.total_users, scenario.user_weights)
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
    return SimulationConfig(
        region_names=scenario.region_names,
        users_by_region=users_by_region,
        first_request_arrival_interval_ms=first_request_arrival_interval_ms,
        region_policy=scenario.region_policy,
        request_count=fixed_config.request_count,
        think_time=fixed_config.think_time,
        queue_wait_threshold_ms=fixed_config.queue_wait_threshold_ms,
        service_duration_target_ms=fixed_config.service_duration_target_ms,
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
        "expected_nonzero_prefill_ms": expected_prefill_ms(simulation_config),
    }


def build_wait_bucket_rows(
    scenario: Scenario,
    result: Any,
    users_by_region: list[int],
    first_request_arrival_interval_ms: float,
    thresholds_ms: Sequence[float],
) -> list[dict[str, Any]]:
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
    if scenario.mode == "distributed" and scenario.region_policy == "queue_len_policy":
        return "dynamic_by_queue_len"
    return "|".join(str(item) for item in users_by_region)


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
