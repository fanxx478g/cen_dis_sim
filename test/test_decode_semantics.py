from __future__ import annotations

import logging
import unittest

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


def build_single_request_config() -> SimulationConfig:
    return SimulationConfig(
        request_generation=RequestGenerationConfig(
            user_count=1,
            min_turns_per_user=1,
            max_turns_per_user=1,
            new_user_arrival_mean_seconds=0.01,
            short_context_probability=1.0,
            initial_prompt_variation_ratio=0.0,
            short_context_prompt_tokens=4096,
            short_context_output_tokens_min=1024,
            short_context_output_tokens_max=1024,
        ),
        scheduler=SchedulerConfig(
            allow_first_decode_cross_cluster=True,
            allow_following_decode_cross_cluster=False,
            prompt_len_threshold=16 * 1024,
        ),
        scenario=ScenarioConfig(
            name="single-request-decode-semantics",
            clusters=[
                cluster(
                    "central",
                    pools=[pool("central-short-prefill", ResourceKind.SHORT_PREFILL, 1, 1)],
                ),
                cluster(
                    "edge",
                    pools=[pool("edge-decode", ResourceKind.DECODE, 1, 1)],
                ),
            ],
        ),
        logging=LoggingConfig(
            level=logging.CRITICAL,
            log_to_console=False,
            log_to_file=False,
            log_file_path=None,
            logger_name="cen_dis_sim.test.single_request",
        ),
    )


class DecodeSemanticsTest(unittest.TestCase):
    """Pin down the prefill-first-token and single-request decode semantics."""

    def test_prefill_produces_first_token(self) -> None:
        engine = SimulationEngine(config=build_single_request_config(), seed=6)
        metrics = engine.run()
        summary = metrics.summary()
        pool_stats = engine.pool_breakdown()

        self.assertEqual(summary["requests_completed"], 1)
        self.assertEqual(summary["decode_batch_executions"], 1023)
        self.assertEqual(summary["decode_batch_size_max"], 1)
        self.assertEqual(pool_stats["edge-decode"]["total_enqueued_requests"], 1)
        self.assertEqual(pool_stats["edge-decode"]["resident_count_final"], 0)


if __name__ == "__main__":
    unittest.main()
