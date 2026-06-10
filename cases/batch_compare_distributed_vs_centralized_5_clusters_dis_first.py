from __future__ import annotations

import logging
import os
import sys
from itertools import product
from pprint import pformat

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cases.case_batch_utils import (
    build_batch_log_file_path,
    collect_batch_log_rows,
    ensure_case_log_dir,
    export_batch_log_summary_csv,
    prepare_logging_config,
    run_config_with_file_logging,
)
from cases.case_centralized_short_prefill_5_clusters_dis_first_global_summary import (
    build_centralized_short_prefill_5_cluster_dis_first_config,
)
from cases.case_distributed_5_clusters_dis_first_global_summary import (
    build_distributed_5_cluster_dis_first_config,
)


SEED = 42

# Edit these lists to control the batch sweep.
USER_COUNTS = [
    100 * 60 * 5,
]

ARRIVAL_RATES_RPS = [
    0.0 + i for i in range(96, 103, 2)
]

LONG_CONTEXT_RATIOS = [
    0.0,
]

LONG_CONTEXT_PROMPT_TOKENS = [
    128 * 1024,
]


def iter_parameter_matrix():
    for (
        user_count,
        arrival_rate_rps,
        long_context_ratio,
        long_context_prompt_tokens,
    ) in product(
        USER_COUNTS,
        ARRIVAL_RATES_RPS,
        LONG_CONTEXT_RATIOS,
        LONG_CONTEXT_PROMPT_TOKENS,
    ):
        yield {
            "user_count": user_count,
            "arrival_rate_rps": arrival_rate_rps,
            "long_context_ratio": long_context_ratio,
            "long_context_prompt_tokens": long_context_prompt_tokens,
        }


def build_request_generation_overrides(params: dict[str, object]) -> dict[str, object]:
    arrival_rate_rps = float(params["arrival_rate_rps"])
    if arrival_rate_rps <= 0:
        raise ValueError("arrival_rate_rps must be positive.")

    long_context_ratio = float(params["long_context_ratio"])
    if not 0.0 <= long_context_ratio <= 1.0:
        raise ValueError("long_context_ratio must be between 0 and 1.")

    return {
        "user_count": int(params["user_count"]),
        "new_user_arrival_mean_seconds": 1.0 / arrival_rate_rps,
        # This batch family is defined as all-short traffic.
        "short_context_probability": 1.0,
        "long_context_prompt_tokens": int(params["long_context_prompt_tokens"]),
    }


def scenario_builders():
    return {
        "distributed_5_clusters_dis_first": (
            build_distributed_5_cluster_dis_first_config
        ),
        "centralized_short_prefill_5_clusters_dis_first": (
            build_centralized_short_prefill_5_cluster_dis_first_config
        ),
    }


def build_summary_csv_path() -> str:
    return os.path.join(ensure_case_log_dir(), "log_summary.csv")


def main() -> None:
    ensure_case_log_dir()
    batch_results: list[dict[str, object]] = []
    print("=== Batch Run Started ===", flush=True)

    for params in iter_parameter_matrix():
        request_generation_overrides = build_request_generation_overrides(params)

        for scenario_label, builder in scenario_builders().items():
            log_file_path = build_batch_log_file_path(
                scenario_label,
                user_count=int(params["user_count"]),
                arrival_rate_rps=float(params["arrival_rate_rps"]),
                long_context_ratio=float(params["long_context_ratio"]),
                long_context_prompt_tokens=int(params["long_context_prompt_tokens"]),
                seed=SEED,
            )
            logger_name = f"cen_dis_sim.batch.{scenario_label}.{os.path.basename(log_file_path)}"
            config = builder(
                request_generation_overrides=request_generation_overrides,
                logging_overrides={
                    "level": logging.CRITICAL,
                    "log_to_console": True,
                },
            )
            prepare_logging_config(
                config.logging,
                log_file_path=log_file_path,
                logger_name=logger_name,
            )

            run_label = f"{scenario_label}:{config.scenario.name}"
            metadata = {
                "scenario": scenario_label,
                "user_count": int(params["user_count"]),
                "arrival_rate_rps": float(params["arrival_rate_rps"]),
                # Record effective workload metadata, not just the requested sweep tag.
                "long_context_ratio": 0.0,
                "long_context_prompt_tokens": None,
                "short_context_probability": 1.0,
                "all_short_workload": True,
                # Preserve the original sweep inputs under explicit names so offline
                # analysis can still recover the requested parameter grid if needed.
                "requested_long_context_ratio": float(params["long_context_ratio"]),
                "requested_long_context_prompt_tokens": int(
                    params["long_context_prompt_tokens"]
                ),
                "seed": SEED,
                "scenario_name": config.scenario.name,
                "log_file_path": log_file_path,
            }
            print(
                f"[start] {scenario_label} | users={params['user_count']} "
                f"| rate={params['arrival_rate_rps']} rps "
                f"| long_ratio={params['long_context_ratio']} "
                f"| long_prompt={params['long_context_prompt_tokens']} "
                f"| log={log_file_path}",
                flush=True,
            )
            result = run_config_with_file_logging(
                config,
                log_file_path=log_file_path,
                run_label=run_label,
                metadata=metadata,
                seed=SEED,
            )
            batch_results.append(
                {
                    "scenario": scenario_label,
                    **params,
                    "log_file_path": log_file_path,
                    "summary": result["summary"],
                    "elapsed_seconds": result["elapsed_seconds"],
                }
            )
            print(
                f"[done] {scenario_label} | users={params['user_count']} "
                f"| rate={params['arrival_rate_rps']} rps "
                f"| long_ratio={params['long_context_ratio']} "
                f"| long_prompt={params['long_context_prompt_tokens']} "
                f"| log={log_file_path}",
                flush=True,
            )

    print("\n=== Batch Run Finished ===", flush=True)
    print(f"total_runs: {len(batch_results)}", flush=True)
    print(f"log_dir: {ensure_case_log_dir()}", flush=True)
    summary_rows = collect_batch_log_rows(ensure_case_log_dir())
    if summary_rows:
        summary_csv_path = export_batch_log_summary_csv(
            summary_rows,
            output_file_path=build_summary_csv_path(),
        )
        print(f"summary_csv: {summary_csv_path}", flush=True)
    if batch_results:
        print("\n=== Example Result ===", flush=True)
        print(
            pformat(
                {
                    "scenario": batch_results[0]["scenario"],
                    "user_count": batch_results[0]["user_count"],
                    "arrival_rate_rps": batch_results[0]["arrival_rate_rps"],
                    "long_context_ratio": batch_results[0]["long_context_ratio"],
                    "long_context_prompt_tokens": batch_results[0][
                        "long_context_prompt_tokens"
                    ],
                    "elapsed_seconds": batch_results[0]["elapsed_seconds"],
                    "log_file_path": batch_results[0]["log_file_path"],
                    "summary": batch_results[0]["summary"],
                },
                sort_dicts=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
