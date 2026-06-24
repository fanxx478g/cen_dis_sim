from __future__ import annotations

import argparse
from dataclasses import dataclass

try:
    from .simulation import (
        DeploymentConfig,
        PrefillConfig,
        RequestCountConfig,
        SimulationConfig,
        ThinkTimeConfig,
        allocate_users,
        format_stats_table,
        simulate,
    )
except ImportError:
    from simulation import (
        DeploymentConfig,
        PrefillConfig,
        RequestCountConfig,
        SimulationConfig,
        ThinkTimeConfig,
        allocate_users,
        format_stats_table,
        simulate,
    )


DEFAULT_REGION_NAMES = ("r1", "r2", "r3", "r4", "r5")
DEFAULT_REGION_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0)
DEFAULT_CENTRALIZED_INSTANCES = 5
DEFAULT_REGION_INSTANCES = (1, 1, 1, 1, 1)
DEFAULT_ZERO_PROB_BY_REGION = (0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class InspectionLayout:
    region_names: tuple[str, ...] = DEFAULT_REGION_NAMES
    region_weights: tuple[float, ...] = DEFAULT_REGION_WEIGHTS
    centralized_instances: int = DEFAULT_CENTRALIZED_INSTANCES
    distributed_instances: tuple[int, ...] = DEFAULT_REGION_INSTANCES
    zero_prob_by_region: tuple[float, ...] = DEFAULT_ZERO_PROB_BY_REGION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="诊断单个配置下的排队行为。")
    parser.add_argument("--mode", choices=["centralized", "distributed"], default="distributed")
    parser.add_argument(
        "--region-policy",
        choices=["weight_region_policy", "queue_len_policy", "d_queue_len_policy"],
        default="weight_region_policy",
    )
    parser.add_argument("--total-users", type=int, default=180)
    parser.add_argument("--first-arrival-interval-ms", type=float, default=12_000.0)
    parser.add_argument("--request-count-min", type=int, default=1)
    parser.add_argument("--request-count-max", type=int, default=10)
    parser.add_argument("--think-time-mean-ms", type=float, default=60 * 1000.0)
    parser.add_argument("--think-time-std-ms", type=float, default=15 * 1000.0)
    parser.add_argument("--think-time-min-ms", type=float, default=5 * 1000.0)
    parser.add_argument("--think-time-max-ms", type=float, default=5 * 60 * 1000.0)
    parser.add_argument("--queue-threshold-ms", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = InspectionLayout()
    simulation_config = build_inspection_config(args, layout)
    deployment_config = build_deployment_config(args.mode, layout)

    result = simulate(simulation_config, deployment_config)
    queued_requests = sorted(
        (
            request
            for request in result.requests
            if request.wait_time > simulation_config.queue_wait_threshold_ms
        ),
        key=lambda request: request.wait_time,
        reverse=True,
    )

    print("单配置排队诊断")
    print(
        f"部署模式: {args.mode} | "
        f"region_policy={simulation_config.region_policy} | "
        f"users_by_region={list(simulation_config.users_by_region)} | "
        f"首请求平均到达间隔={args.first_arrival_interval_ms:.0f}ms | "
        f"seed={args.seed}"
    )
    print(
        "每用户请求数区间: "
        f"[{simulation_config.request_count.min_requests}, "
        f"{simulation_config.request_count.max_requests}]"
    )
    print()
    print(
        format_stats_table(
            [("overall", result.overall_stats)]
            + [(region, stats) for region, stats in result.region_stats.items()]
        )
    )
    print()
    print(f"排队阈值: {simulation_config.queue_wait_threshold_ms:.1f}ms")
    print(f"超过阈值的请求数: {len(queued_requests)}")

    if not queued_requests:
        print("当前配置下没有超过阈值的排队请求。")
        return

    print()
    print("最长等待请求明细")
    for request in queued_requests[: args.top_n]:
        print(
            f"id={request.id:5d} "
            f"user={request.user_id:4d} "
            f"region={request.region:>2s} "
            f"req_idx={request.request_index:2d} "
            f"arrival={request.arrival_time:10.1f} "
            f"wait={request.wait_time:10.1f} "
            f"prefill={request.prefill_time:10.1f} "
            f"total={request.service_duration:10.1f}"
        )


def build_inspection_config(
    args: argparse.Namespace,
    layout: InspectionLayout,
) -> SimulationConfig:
    if args.total_users < len(layout.region_names):
        raise ValueError(f"total-users 至少应为 {len(layout.region_names)}。")

    users_by_region = allocate_users(args.total_users, layout.region_weights)
    return SimulationConfig(
        region_names=layout.region_names,
        users_by_region=users_by_region,
        region_policy=args.region_policy,
        first_request_arrival_interval_ms=args.first_arrival_interval_ms,
        request_count=RequestCountConfig(
            min_requests=args.request_count_min,
            max_requests=args.request_count_max,
        ),
        think_time=ThinkTimeConfig(
            mean_ms=args.think_time_mean_ms,
            std_ms=args.think_time_std_ms,
            min_ms=args.think_time_min_ms,
            max_ms=args.think_time_max_ms,
        ),
        queue_wait_threshold_ms=args.queue_threshold_ms,
        service_duration_target_ms=5_000.0,
        seed=args.seed,
        prefill=PrefillConfig(zero_prob_by_region=layout.zero_prob_by_region),
    )


def build_deployment_config(mode: str, layout: InspectionLayout) -> DeploymentConfig:
    if mode == "centralized":
        return DeploymentConfig(mode="centralized", instances=layout.centralized_instances)
    return DeploymentConfig(mode="distributed", instances=layout.distributed_instances)


if __name__ == "__main__":
    main()
