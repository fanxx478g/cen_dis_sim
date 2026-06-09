from __future__ import annotations

import logging
import os
import sys
import time
from pprint import pprint

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


def build_distributed_2_cluster_config() -> SimulationConfig:
    clusters = []
    for index in range(2):
        cluster_id = f"cluster-{index + 1}"
        clusters.append(
            cluster(
                cluster_id,
                pools=[
                    pool(
                        f"{cluster_id}-long-prefill",
                        ResourceKind.LONG_PREFILL,
                        1,
                        1,
                    ),
                    pool(
                        f"{cluster_id}-short-prefill",
                        ResourceKind.SHORT_PREFILL,
                        4,
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

    return SimulationConfig(
        request_generation=RequestGenerationConfig(
            user_count=40 * 60 * 2,
            min_turns_per_user=1,
            max_turns_per_user=1,
            new_user_arrival_mean_seconds=0.025,
            followup_arrival_mean_seconds=30,
            followup_arrival_std_seconds=5,
            accumulate_context_across_turns=False,
            followup_prompt_tokens=0,
            short_context_probability=0.99,
            initial_prompt_variation_ratio=0.125,
            short_context_prompt_tokens=4 * 1024,
            long_context_prompt_tokens=128 * 1024,
            short_context_output_tokens_min=1024,
            short_context_output_tokens_max=1024,
            long_context_output_tokens_min=1024,
            long_context_output_tokens_max=1024,
        ),
        scheduler=SchedulerConfig(
            policy_name="p_default",
            policy_config={},
            prompt_len_threshold=32 * 1024,
            allow_first_decode_cross_cluster=True,
            allow_following_decode_cross_cluster=False,
        ),
        scenario=ScenarioConfig(
            name="distributed-2-clusters-global-summary",
            clusters=clusters,
            central_cluster_id=None,
        ),
        logging=LoggingConfig(
            level=logging.CRITICAL,
            log_to_console=True,
            log_to_file=False,
            log_file_path=None,
            logger_name="cen_dis_sim.case.distributed_2_clusters",
        ),
    )


def main() -> None:
    start_time = time.time()

    config = build_distributed_2_cluster_config()
    engine = SimulationEngine(config=config, seed=42)
    metrics = engine.run()

    print("=== Global Summary ===")
    pprint(metrics.summary())

    end_time = time.time()
    print(f"===仿真时间=== {(end_time - start_time)}s")


if __name__ == "__main__":
    main()
