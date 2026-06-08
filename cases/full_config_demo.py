from __future__ import annotations

import logging
import os
import sys
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


def build_full_config_demo() -> SimulationConfig:
    return SimulationConfig(
        request_generation=RequestGenerationConfig(
            # User/session population
            user_count=4,
            min_turns_per_user=2,
            max_turns_per_user=5,
            # First-turn arrivals across users follow a Poisson process.
            new_user_arrival_mean_seconds=0.02,
            # Follow-up turn arrivals are sampled relative to the previous finish time.
            followup_arrival_mean_seconds=12.0,
            followup_arrival_std_seconds=3.0,
            # Enable realistic multi-turn context growth. Set to False to keep every
            # turn at the same prompt length as the first turn.
            accumulate_context_across_turns=False,
            # Later turns currently append no fresh user input, so prompt growth comes
            # entirely from accumulated history plus prior outputs.
            followup_prompt_tokens=0,
            # First turn: sample 4k-class vs 128k-class by probability, then jitter
            # around the chosen bucket center by +/- initial_prompt_variation_ratio.
            short_context_probability=1,
            initial_prompt_variation_ratio=0.125,
            short_context_prompt_tokens=4 * 1024,
            long_context_prompt_tokens=128 * 1024,
            # Output token ranges are configured independently for the two buckets.
            short_context_output_tokens_min=512,
            short_context_output_tokens_max=1024,
            long_context_output_tokens_min=128,
            long_context_output_tokens_max=256,
        ),
        scheduler=SchedulerConfig(
            policy_name="p_default",
            policy_config={},
            prompt_len_threshold=16 * 1024,
            allow_first_decode_cross_cluster=True,
            allow_following_decode_cross_cluster=False,
        ),
        scenario=ScenarioConfig(
            name="full-config-demo",
            clusters=[
                cluster(
                    "central",
                    pools=[
                        pool("central-short-prefill", ResourceKind.SHORT_PREFILL, 2, 2),
                        pool("central-long-prefill", ResourceKind.LONG_PREFILL, 1, 1),
                    ],
                ),
                cluster(
                    "edge-a",
                    pools=[pool("edge-a-decode", ResourceKind.DECODE, 1, 4)],
                ),
                cluster(
                    "edge-b",
                    pools=[pool("edge-b-decode", ResourceKind.DECODE, 1, 4)],
                ),
            ],
            central_cluster_id="central",
        ),
        logging=LoggingConfig(
            level=logging.CRITICAL,
            log_to_console=True,
            log_to_file=False,
            log_file_path=None,
            logger_name="cen_dis_sim.full_config_demo",
        ),
    )


def main() -> None:
    config = build_full_config_demo()
    engine = SimulationEngine(config=config, seed=9)
    metrics = engine.run()

    print("=== Summary ===")
    pprint(metrics.summary())

    print("\n=== Pool Breakdown ===")
    pprint(engine.pool_breakdown())

    print("\n=== Request Breakdown ===")
    for request in engine.request_breakdown():
        pprint(request)


if __name__ == "__main__":
    main()
