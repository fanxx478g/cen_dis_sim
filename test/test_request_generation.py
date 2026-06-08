from __future__ import annotations

import logging
import random
import unittest

from simulator import (
    LoggingConfig,
    RequestGenerationConfig,
    RequestGenerator,
    ResourceKind,
    ScenarioConfig,
    SchedulerConfig,
    SimulationConfig,
    SimulationEngine,
    cluster,
    pool,
)


def build_request_generation_config(
    request_generation: RequestGenerationConfig,
) -> SimulationConfig:
    return SimulationConfig(
        request_generation=request_generation,
        scheduler=SchedulerConfig(
            allow_first_decode_cross_cluster=True,
            allow_following_decode_cross_cluster=False,
            prompt_len_threshold=16 * 1024,
        ),
        scenario=ScenarioConfig(
            name="request-generation-test",
            clusters=[
                cluster(
                    "central",
                    pools=[
                        pool("central-short-prefill", ResourceKind.SHORT_PREFILL, 1, 1),
                        pool("central-long-prefill", ResourceKind.LONG_PREFILL, 1, 1),
                        pool("central-decode", ResourceKind.DECODE, 1, 4),
                    ],
                )
            ],
        ),
        logging=LoggingConfig(
            level=logging.CRITICAL,
            log_to_console=False,
            log_to_file=False,
            log_file_path=None,
            logger_name="cen_dis_sim.test.request_generation",
        ),
    )


class RequestGenerationTest(unittest.TestCase):
    def test_multi_turn_sessions_accumulate_history_and_outputs(self) -> None:
        engine = SimulationEngine(
            config=build_request_generation_config(
                RequestGenerationConfig(
                    user_count=2,
                    min_turns_per_user=3,
                    max_turns_per_user=3,
                    new_user_arrival_mean_seconds=0.01,
                    followup_arrival_mean_seconds=0.01,
                    followup_arrival_std_seconds=0.0,
                    accumulate_context_across_turns=True,
                    followup_prompt_tokens=0,
                    short_context_probability=1.0,
                    initial_prompt_variation_ratio=0.0,
                    short_context_prompt_tokens=4096,
                    short_context_output_tokens_min=4,
                    short_context_output_tokens_max=4,
                )
            ),
            seed=11,
        )

        metrics = engine.run()
        summary = metrics.summary()
        requests = sorted(
            metrics.request_records,
            key=lambda request: (request.session_id or 0, request.turn_index),
        )

        self.assertEqual(summary["sessions_total"], 2)
        self.assertEqual(summary["requests_total"], 6)

        prompts_by_session: dict[int, list[int]] = {}
        for request in requests:
            prompts_by_session.setdefault(request.session_id or 0, []).append(
                request.prompt_tokens
            )

        self.assertEqual(prompts_by_session[1], [4096, 4100, 4104])
        self.assertEqual(prompts_by_session[2], [4096, 4100, 4104])
        self.assertTrue(
            all(
                request.new_prompt_tokens == 0
                for request in requests
                if request.turn_index > 1
            )
        )

    def test_turn_growth_can_be_disabled_to_keep_first_turn_prompt_size(self) -> None:
        engine = SimulationEngine(
            config=build_request_generation_config(
                RequestGenerationConfig(
                    user_count=1,
                    min_turns_per_user=3,
                    max_turns_per_user=3,
                    new_user_arrival_mean_seconds=0.01,
                    followup_arrival_mean_seconds=0.01,
                    followup_arrival_std_seconds=0.0,
                    accumulate_context_across_turns=False,
                    followup_prompt_tokens=999,
                    short_context_probability=1.0,
                    initial_prompt_variation_ratio=0.0,
                    short_context_prompt_tokens=4096,
                    short_context_output_tokens_min=4,
                    short_context_output_tokens_max=4,
                )
            ),
            seed=5,
        )

        requests = sorted(
            engine.run().request_records,
            key=lambda request: request.turn_index,
        )

        self.assertEqual(
            [(request.history_tokens, request.new_prompt_tokens, request.prompt_tokens) for request in requests],
            [(0, 4096, 4096), (0, 4096, 4096), (0, 4096, 4096)],
        )

    def test_followup_arrivals_are_offset_from_previous_finish_time(self) -> None:
        engine = SimulationEngine(
            config=build_request_generation_config(
                RequestGenerationConfig(
                    user_count=1,
                    min_turns_per_user=3,
                    max_turns_per_user=3,
                    new_user_arrival_mean_seconds=0.01,
                    followup_arrival_mean_seconds=60.0,
                    followup_arrival_std_seconds=0.0,
                    short_context_probability=1.0,
                    initial_prompt_variation_ratio=0.0,
                    short_context_prompt_tokens=4096,
                    short_context_output_tokens_min=1,
                    short_context_output_tokens_max=1,
                )
            ),
            seed=7,
        )

        metrics = engine.run()
        requests = sorted(metrics.request_records, key=lambda request: request.turn_index)

        self.assertEqual(len(requests), 3)
        self.assertAlmostEqual(
            requests[1].arrival_time_ms - (requests[0].finish_time_ms or 0.0),
            60000.0,
        )
        self.assertAlmostEqual(
            requests[2].arrival_time_ms - (requests[1].finish_time_ms or 0.0),
            60000.0,
        )

    def test_followup_prompt_tokens_are_added_on_top_of_history(self) -> None:
        engine = SimulationEngine(
            config=build_request_generation_config(
                RequestGenerationConfig(
                    user_count=1,
                    min_turns_per_user=3,
                    max_turns_per_user=3,
                    new_user_arrival_mean_seconds=0.01,
                    followup_arrival_mean_seconds=0.01,
                    followup_arrival_std_seconds=0.0,
                    accumulate_context_across_turns=True,
                    followup_prompt_tokens=10,
                    short_context_probability=1.0,
                    initial_prompt_variation_ratio=0.0,
                    short_context_prompt_tokens=100,
                    short_context_output_tokens_min=2,
                    short_context_output_tokens_max=2,
                )
            ),
            seed=3,
        )

        requests = sorted(
            engine.run().request_records,
            key=lambda request: request.turn_index,
        )

        self.assertEqual(
            [(request.history_tokens, request.new_prompt_tokens, request.prompt_tokens) for request in requests],
            [(0, 100, 100), (102, 10, 112), (114, 10, 124)],
        )

    def test_long_context_profile_uses_128k_window_and_output_range(self) -> None:
        engine = SimulationEngine(
            config=build_request_generation_config(
                RequestGenerationConfig(
                    user_count=1,
                    min_turns_per_user=1,
                    max_turns_per_user=1,
                    new_user_arrival_mean_seconds=0.01,
                    short_context_probability=0.0,
                    initial_prompt_variation_ratio=0.125,
                    long_context_prompt_tokens=128 * 1024,
                    long_context_output_tokens_min=777,
                    long_context_output_tokens_max=777,
                )
            ),
            seed=5,
        )

        metrics = engine.run()
        request = metrics.request_records[0]

        self.assertEqual(request.initial_prompt_bucket, "long")
        self.assertGreaterEqual(request.prompt_tokens, 114688)
        self.assertLessEqual(request.prompt_tokens, 147456)
        self.assertEqual(request.output_tokens, 777)

    def test_first_turn_arrivals_follow_exponential_sampling(self) -> None:
        request_generation = RequestGenerationConfig(
            user_count=3,
            min_turns_per_user=1,
            max_turns_per_user=1,
            new_user_arrival_mean_seconds=100.0,
            short_context_probability=1.0,
            initial_prompt_variation_ratio=0.0,
            short_context_prompt_tokens=4096,
            short_context_output_tokens_min=4,
            short_context_output_tokens_max=4,
        )
        generator = RequestGenerator(request_generation, random.Random(13))
        session_states = generator._build_session_states()

        rng = random.Random(13)
        current_time_ms = 0.0
        expected_arrivals: list[float] = []
        for _ in range(3):
            current_time_ms += rng.expovariate(1.0 / 100.0) * 1000.0
            expected_arrivals.append(current_time_ms)
            rng.random()
            rng.randint(1, 1)
            rng.randint(4096, 4096)

        actual_arrivals = [session.first_arrival_time_ms for session in session_states]
        for actual, expected in zip(actual_arrivals, expected_arrivals):
            self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
