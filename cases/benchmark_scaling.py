from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulator import (
    LoggingConfig,
    RequestGenerationConfig,
    ResourceKind,
    ScenarioConfig,
    SchedulerConfig,
    SimulationConfig,
    SimulationEngine,
    cluster,
    pool,
)


def build_benchmark_config(
    request_count: int,
    edge_cluster_count: int,
    central_prefill_instances: int,
    edge_decode_instances: int,
    decode_batch_size: int,
) -> SimulationConfig:
    clusters = [
        cluster(
            "central",
            pools=[
                pool(
                    "central-short-prefill",
                    ResourceKind.SHORT_PREFILL,
                    central_prefill_instances,
                    1,
                )
            ],
        )
    ]

    for index in range(edge_cluster_count):
        clusters.append(
            cluster(
                f"edge-{index + 1}",
                pools=[
                    pool(
                        f"edge-{index + 1}-decode",
                        ResourceKind.DECODE,
                        edge_decode_instances,
                        decode_batch_size,
                    )
                ],
            )
        )

    return SimulationConfig(
        request_generation=RequestGenerationConfig(
            user_count=request_count,
            min_turns_per_user=1,
            max_turns_per_user=1,
            new_user_arrival_mean_seconds=0.005,
            short_context_probability=1.0,
            initial_prompt_variation_ratio=0.0,
            short_context_prompt_tokens=4096,
            short_context_output_tokens_min=256,
            short_context_output_tokens_max=256,
        ),
        scheduler=SchedulerConfig(
            allow_first_decode_cross_cluster=True,
            allow_following_decode_cross_cluster=False,
            prompt_len_threshold=16 * 1024,
        ),
        scenario=ScenarioConfig(
            name=(
                f"benchmark-r{request_count}-c{edge_cluster_count}"
                f"-pi{central_prefill_instances}-di{edge_decode_instances}"
            ),
            clusters=clusters,
        ),
        logging=LoggingConfig(
            level=logging.CRITICAL,
            log_to_console=False,
            log_to_file=False,
            log_file_path=None,
            logger_name="cen_dis_sim.benchmark",
        ),
    )


def run_once(config: SimulationConfig, seed: int) -> dict[str, float | int | None]:
    started = time.perf_counter()
    engine = SimulationEngine(config=config, seed=seed)
    metrics = engine.run()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    summary = metrics.summary()
    return {
        "simulator_runtime_ms": elapsed_ms,
        "requests_total": summary["requests_total"],
        "request_throughput_rps": summary["request_throughput_rps"],
        "output_token_throughput_tps": summary["output_token_throughput_tps"],
        "request_tpot_avg_ms": summary["system_tpot_avg_ms"],
        "prefill_first_token_latency_p95_ms": summary[
            "prefill_first_token_latency_p95_ms"
        ],
        "decode_queue_p95_ms": summary["decode_queue_p95_ms"],
    }


def summarize_results(
    label: str,
    config: SimulationConfig,
    repeat: int,
    seed_base: int,
) -> dict[str, float | int | str | None]:
    runs = [run_once(config, seed_base + index) for index in range(repeat)]
    return {
        "case": label,
        "requests": runs[0]["requests_total"],
        "clusters": len(config.scenario.clusters),
        "sim_runtime_ms_avg": round(mean(run["simulator_runtime_ms"] for run in runs), 3),
        "sim_runtime_ms_min": round(min(run["simulator_runtime_ms"] for run in runs), 3),
        "request_tput_rps": round(runs[0]["request_throughput_rps"] or 0.0, 3),
        "output_tput_tps": round(runs[0]["output_token_throughput_tps"] or 0.0, 3),
        "request_tpot_avg_ms": round(runs[0]["request_tpot_avg_ms"] or 0.0, 3),
        "prefill_first_token_p95_ms": round(
            runs[0]["prefill_first_token_latency_p95_ms"] or 0.0, 3
        ),
        "decode_q95_ms": round(runs[0]["decode_queue_p95_ms"] or 0.0, 3),
    }


def benchmark_request_scale(repeat: int) -> list[dict[str, float | int | str | None]]:
    results = []
    for request_count in (100, 500, 1000, 2000):
        config = build_benchmark_config(
            request_count=request_count,
            edge_cluster_count=2,
            central_prefill_instances=4,
            edge_decode_instances=2,
            decode_batch_size=8,
        )
        results.append(
            summarize_results(
                label=f"request-scale-{request_count}",
                config=config,
                repeat=repeat,
                seed_base=1000 + request_count,
            )
        )
    return results


def benchmark_instance_scale(repeat: int) -> list[dict[str, float | int | str | None]]:
    results = []
    for instance_count in (1, 2, 4, 8):
        config = build_benchmark_config(
            request_count=1000,
            edge_cluster_count=2,
            central_prefill_instances=instance_count,
            edge_decode_instances=instance_count,
            decode_batch_size=8,
        )
        results.append(
            summarize_results(
                label=f"instance-scale-{instance_count}",
                config=config,
                repeat=repeat,
                seed_base=2000 + instance_count,
            )
        )
    return results


def benchmark_cluster_scale(repeat: int) -> list[dict[str, float | int | str | None]]:
    results = []
    for edge_cluster_count in (1, 2, 4, 8):
        config = build_benchmark_config(
            request_count=1000,
            edge_cluster_count=edge_cluster_count,
            central_prefill_instances=4,
            edge_decode_instances=2,
            decode_batch_size=8,
        )
        results.append(
            summarize_results(
                label=f"cluster-scale-{edge_cluster_count}",
                config=config,
                repeat=repeat,
                seed_base=3000 + edge_cluster_count,
            )
        )
    return results


def print_table(title: str, rows: list[dict[str, float | int | str | None]]) -> None:
    print(f"\n=== {title} ===")
    print(
        "case".ljust(22),
        "requests".rjust(8),
        "clusters".rjust(8),
        "sim_avg_ms".rjust(12),
        "sim_min_ms".rjust(12),
        "req_tput".rjust(12),
        "out_tput".rjust(12),
        "tpot_avg".rjust(12),
        "ttft_p95".rjust(12),
        "decode_q95_ms".rjust(14),
    )
    for row in rows:
        print(
            str(row["case"]).ljust(22),
            str(row["requests"]).rjust(8),
            str(row["clusters"]).rjust(8),
            str(row["sim_runtime_ms_avg"]).rjust(12),
            str(row["sim_runtime_ms_min"]).rjust(12),
            str(row["request_tput_rps"]).rjust(12),
            str(row["output_tput_tps"]).rjust(12),
            str(row["request_tpot_avg_ms"]).rjust(12),
            str(row["prefill_first_token_p95_ms"]).rjust(12),
            str(row["decode_q95_ms"]).rjust(14),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scaling benchmarks for the cen_dis_sim simulator."
    )
    parser.add_argument(
        "--suite",
        choices=("requests", "instances", "clusters", "all"),
        default="all",
        help="Which benchmark suite to run.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="How many times to repeat each benchmark point.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.suite in ("requests", "all"):
        print_table("Request Scale", benchmark_request_scale(args.repeat))
    if args.suite in ("instances", "all"):
        print_table("Instance Scale", benchmark_instance_scale(args.repeat))
    if args.suite in ("clusters", "all"):
        print_table("Cluster Scale", benchmark_cluster_scale(args.repeat))


if __name__ == "__main__":
    main()
