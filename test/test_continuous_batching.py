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


def build_dual_request_config() -> SimulationConfig:
    return SimulationConfig(
        request_generation=RequestGenerationConfig(
            user_count=2,
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
            name="dual-request-continuous-batching",
            clusters=[
                cluster(
                    "central",
                    pools=[pool("central-short-prefill", ResourceKind.SHORT_PREFILL, 2, 1)],
                ),
                cluster(
                    "edge",
                    pools=[pool("edge-decode", ResourceKind.DECODE, 1, 2)],
                ),
            ],
        ),
        logging=LoggingConfig(
            level=logging.CRITICAL,
            log_to_console=False,
            log_to_file=False,
            log_file_path=None,
            logger_name="cen_dis_sim.test.dual_request",
        ),
    )


class ContinuousBatchingTest(unittest.TestCase):
    """Ensure decode batching keeps both requests resident and produces non-zero waiting time."""

    def test_dual_request_decode_batches(self) -> None:
        engine = SimulationEngine(config=build_dual_request_config(), seed=6)
        metrics = engine.run()
        summary = metrics.summary()

        self.assertEqual(summary["requests_completed"], 2)
        self.assertEqual(summary["decode_batch_size_max"], 2)
        self.assertGreater(summary["decode_batch_size_avg"], 1.9)
        self.assertGreater(summary["decode_queue_p50_ms"], 0.0)
        self.assertGreater(summary["decode_queue_p95_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
