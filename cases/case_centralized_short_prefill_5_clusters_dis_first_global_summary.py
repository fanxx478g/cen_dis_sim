from __future__ import annotations

import logging
import os
import sys
import time
from pprint import pprint

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cases.case_batch_utils import apply_overrides
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


def build_centralized_short_prefill_5_cluster_dis_first_config(
    request_generation_overrides: dict[str, object] | None = None,
    logging_overrides: dict[str, object] | None = None,
) -> SimulationConfig:
    distributed_clusters = []

    for index in range(5):
        cluster_id = f"cluster-{index + 1}"
        distributed_clusters.append(
            cluster(
                cluster_id,
                pools=[
                    pool(
                        f"{cluster_id}-short-prefill",
                        ResourceKind.SHORT_PREFILL,
                        3,
                        1,
                    ),
                    pool(
                        f"{cluster_id}-decode",
                        ResourceKind.DECODE,
                        1,
                        2048,
                    ),
                ],
            )
        )

    central_cluster = cluster(
        "central",
        pools=[
            pool(
                "central-short-prefill",
                ResourceKind.SHORT_PREFILL,
                5,
                1,
            )
        ],
    )

    request_generation = RequestGenerationConfig(
        user_count=40 * 60 * 10,
        min_turns_per_user=1,
        max_turns_per_user=1,
        new_user_arrival_mean_seconds=0.025,
        followup_arrival_mean_seconds=30,
        followup_arrival_std_seconds=5,
        accumulate_context_across_turns=False,
        followup_prompt_tokens=0,
        # This topology only has short-prefill pools, so the standalone case must
        # generate all-short traffic to remain self-consistent.
        short_context_probability=1.0,
        initial_prompt_variation_ratio=0.125,
        short_context_prompt_tokens=4 * 1024,
        long_context_prompt_tokens=128 * 1024,
        short_context_output_tokens_min=1024,
        short_context_output_tokens_max=1024,
        long_context_output_tokens_min=1024,
        long_context_output_tokens_max=1024,
    )
    apply_overrides(request_generation, request_generation_overrides)

    logging_config = LoggingConfig(
        level=logging.CRITICAL,
        log_to_console=True,
        log_to_file=False,
        log_file_path=None,
        logger_name="cen_dis_sim.case.centralized_short_prefill_5_clusters_dis_first",
    )
    apply_overrides(logging_config, logging_overrides)

    return SimulationConfig(
        request_generation=request_generation,
        scheduler=SchedulerConfig(
            policy_name="dis_first",
            policy_config={
                "distributed_cluster_routing_weights": {
                    f"cluster-{index + 1}": {"short_prefill": 1.0}
                    for index in range(5)
                },
                "short_prefill_queue_threshold": 0,
            },
            prompt_len_threshold=16 * 1024,
            allow_first_decode_cross_cluster=False,
            allow_following_decode_cross_cluster=False,
        ),
        scenario=ScenarioConfig(
            name="centralized-short-prefill-5-clusters-dis-first-global-summary",
            clusters=[central_cluster, *distributed_clusters],
            central_cluster_id="central",
        ),
        logging=logging_config,
    )


def main() -> None:
    start_time = time.time()

    config = build_centralized_short_prefill_5_cluster_dis_first_config()
    engine = SimulationEngine(config=config, seed=42)
    metrics = engine.run()

    print("=== Global Summary ===")
    pprint(metrics.summary())

    end_time = time.time()
    print(f"===仿真时间=== {(end_time - start_time)}s")


if __name__ == "__main__":
    main()
