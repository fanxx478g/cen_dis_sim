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


def build_metrics_config() -> SimulationConfig:
    return SimulationConfig(
        request_generation=RequestGenerationConfig(
            user_count=2,
            min_turns_per_user=1,
            max_turns_per_user=1,
            new_user_arrival_mean_seconds=0.01,
            short_context_probability=0.5,
            initial_prompt_variation_ratio=0.0,
            short_context_prompt_tokens=4096,
            long_context_prompt_tokens=128 * 1024,
            short_context_output_tokens_min=4,
            short_context_output_tokens_max=4,
            long_context_output_tokens_min=4,
            long_context_output_tokens_max=4,
        ),
        scheduler=SchedulerConfig(
            allow_first_decode_cross_cluster=True,
            allow_following_decode_cross_cluster=False,
            prompt_len_threshold=16 * 1024,
        ),
        scenario=ScenarioConfig(
            name="metrics-summary-test",
            clusters=[
                cluster(
                    "central",
                    pools=[
                        pool("central-short-prefill", ResourceKind.SHORT_PREFILL, 1, 1),
                        pool("central-long-prefill", ResourceKind.LONG_PREFILL, 1, 1),
                        pool("central-decode", ResourceKind.DECODE, 1, 2),
                    ],
                )
            ],
        ),
        logging=LoggingConfig(
            level=logging.CRITICAL,
            log_to_console=False,
            log_to_file=False,
            log_file_path=None,
            logger_name="cen_dis_sim.test.metrics_summary",
        ),
    )


class MetricsSummaryTest(unittest.TestCase):
    def test_summary_reports_backward_compatible_and_new_ttft_tpot_fields(self) -> None:
        engine = SimulationEngine(config=build_metrics_config(), seed=4)

        metrics = engine.run()
        summary = metrics.summary()
        completed = [request for request in metrics.request_records if request.finish_time_ms]

        prefill_first_token_values = [
            request.prefill_first_token_latency_ms for request in completed
        ]
        request_tpot_values = [request.request_tpot_ms for request in completed]
        active_window_ms = (
            max(request.finish_time_ms or 0.0 for request in completed)
            - min(request.arrival_time_ms for request in completed)
        )
        active_window_s = active_window_ms / 1000.0
        total_output_tokens = sum(request.output_tokens for request in completed)
        total_decode_tokens = sum(max(request.output_tokens - 1, 0) for request in completed)
        total_prompt_tokens = sum(request.prompt_tokens for request in completed)

        self.assertIn("prefill_first_token_latency_p50_ms", summary)
        self.assertIn("system_tpot_avg_ms", summary)
        self.assertIn("ttft_p50_ms", summary)
        self.assertIn("ttft_p95_ms", summary)
        self.assertIn("tpot_ms", summary)
        self.assertEqual(summary["ttft_p50_ms"], summary["prefill_first_token_latency_p50_ms"])
        self.assertEqual(summary["ttft_p95_ms"], summary["prefill_first_token_latency_p95_ms"])
        self.assertEqual(summary["tpot_ms"], summary["system_tpot_avg_ms"])

        self.assertAlmostEqual(
            summary["system_tpot_avg_ms"] or 0.0,
            sum(request_tpot_values) / len(request_tpot_values),
        )
        self.assertAlmostEqual(summary["active_window_ms"] or 0.0, active_window_ms)
        self.assertAlmostEqual(
            summary["request_throughput_rps"] or 0.0,
            len(completed) / active_window_s,
        )
        self.assertAlmostEqual(
            summary["output_token_throughput_tps"] or 0.0,
            total_output_tokens / active_window_s,
        )
        self.assertAlmostEqual(
            summary["decode_token_throughput_tps"] or 0.0,
            total_decode_tokens / active_window_s,
        )
        self.assertAlmostEqual(
            summary["prefill_token_throughput_tps"] or 0.0,
            total_prompt_tokens / active_window_s,
        )
        self.assertEqual(
            summary["requests_with_tpot_le_50ms"],
            sum(1 for value in request_tpot_values if value is not None and value <= 50.0),
        )
        self.assertEqual(
            summary["requests_with_prefill_first_token_latency_le_2s"],
            sum(
                1
                for value in prefill_first_token_values
                if value is not None and value <= 2000.0
            ),
        )
        self.assertAlmostEqual(
            summary["requests_with_tpot_le_50ms_ratio"] or 0.0,
            sum(1 for value in request_tpot_values if value is not None and value <= 50.0)
            / len(request_tpot_values),
        )
        self.assertAlmostEqual(
            summary["requests_with_prefill_first_token_latency_le_2s_ratio"] or 0.0,
            sum(
                1
                for value in prefill_first_token_values
                if value is not None and value <= 2000.0
            )
            / len(prefill_first_token_values),
        )

    def test_request_breakdown_exposes_renamed_latency_and_request_tpot(self) -> None:
        engine = SimulationEngine(config=build_metrics_config(), seed=4)

        engine.run()
        request_rows = engine.request_breakdown()

        self.assertTrue(request_rows)
        self.assertIn("ttft_ms", request_rows[0])
        self.assertIn("prefill_first_token_latency_ms", request_rows[0])
        self.assertIn("request_tpot_ms", request_rows[0])
        self.assertEqual(
            request_rows[0]["ttft_ms"],
            request_rows[0]["prefill_first_token_latency_ms"],
        )

    def test_global_summary_contains_only_core_tpot_ttft_and_throughput_metrics(self) -> None:
        engine = SimulationEngine(config=build_metrics_config(), seed=4)

        metrics = engine.run()
        global_summary = metrics.global_summary()

        self.assertEqual(
            set(global_summary.keys()),
            {
                "request_tpot_avg_ms",
                "system_tpot_avg_ms",
                "prefill_first_token_latency_avg_ms",
                "prefill_first_token_latency_max_ms",
                "prefill_first_token_latency_p50_ms",
                "prefill_first_token_latency_p95_ms",
                "request_throughput_rps",
                "output_token_throughput_tps",
                "decode_token_throughput_tps",
                "prefill_token_throughput_tps",
            },
        )
        self.assertEqual(
            global_summary["request_tpot_avg_ms"],
            metrics.summary()["request_tpot_avg_ms"],
        )


if __name__ == "__main__":
    unittest.main()
