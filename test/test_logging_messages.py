from __future__ import annotations

import io
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
from simulator.logging_utils import SimulationFormatter


def build_logging_test_config() -> SimulationConfig:
    return SimulationConfig(
        request_generation=RequestGenerationConfig(
            user_count=2,
            min_turns_per_user=1,
            max_turns_per_user=1,
            new_user_arrival_mean_seconds=0.01,
            short_context_probability=1.0,
            initial_prompt_variation_ratio=0.0,
            short_context_prompt_tokens=4096,
            short_context_output_tokens_min=1,
            short_context_output_tokens_max=1,
        ),
        scheduler=SchedulerConfig(
            allow_first_decode_cross_cluster=True,
            allow_following_decode_cross_cluster=False,
            prompt_len_threshold=16 * 1024,
        ),
        scenario=ScenarioConfig(
            name="logging-message-test",
            clusters=[
                cluster(
                    "central",
                    pools=[pool("central-short-prefill", ResourceKind.SHORT_PREFILL, 1, 1)],
                ),
            ],
        ),
        logging=LoggingConfig(
            level=logging.CRITICAL,
            log_to_console=False,
            log_to_file=False,
            log_file_path=None,
            logger_name="cen_dis_sim.test.logging_messages",
        ),
    )


class LoggingMessageTest(unittest.TestCase):
    def test_prefill_enqueue_log_contains_request_id_and_progress(self) -> None:
        engine = SimulationEngine(config=build_logging_test_config(), seed=3)
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.CRITICAL)
        handler.setFormatter(
            SimulationFormatter(
                fmt="[%(asctime)s][%(sim_time_ms)sms][%(levelname)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        engine.logger.addHandler(handler)

        try:
            engine.run()
        finally:
            engine.logger.removeHandler(handler)

        output = stream.getvalue()
        self.assertIn("请求#1（1/2, 50.0%） 已到达 short_prefill 资源池。", output)


if __name__ == "__main__":
    unittest.main()
