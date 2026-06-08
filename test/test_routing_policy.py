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
    register_policy,
)
from simulator.policies.base import PolicyContext


def build_single_request_config(
    *,
    prompt_tokens: int,
    output_tokens: int = 4,
    scheduler: SchedulerConfig | None = None,
    clusters: list | None = None,
) -> SimulationConfig:
    return SimulationConfig(
        request_generation=RequestGenerationConfig(
            user_count=1,
            min_turns_per_user=1,
            max_turns_per_user=1,
            new_user_arrival_mean_seconds=0.01,
            short_context_probability=1.0 if prompt_tokens <= 16 * 1024 else 0.0,
            initial_prompt_variation_ratio=0.0,
            short_context_prompt_tokens=prompt_tokens,
            long_context_prompt_tokens=prompt_tokens,
            short_context_output_tokens_min=output_tokens,
            short_context_output_tokens_max=output_tokens,
            long_context_output_tokens_min=output_tokens,
            long_context_output_tokens_max=output_tokens,
        ),
        scheduler=scheduler
        or SchedulerConfig(
            allow_first_decode_cross_cluster=True,
            allow_following_decode_cross_cluster=False,
            prompt_len_threshold=16 * 1024,
        ),
        scenario=ScenarioConfig(
            name="routing-policy-test",
            clusters=clusters
            or [
                cluster(
                    "central",
                    pools=[
                        pool("central-short-prefill", ResourceKind.SHORT_PREFILL, 1, 1),
                        pool("central-long-prefill", ResourceKind.LONG_PREFILL, 1, 1),
                        pool("central-decode", ResourceKind.DECODE, 1, 1),
                    ],
                )
            ],
        ),
        logging=LoggingConfig(
            level=logging.CRITICAL,
            log_to_console=False,
            log_to_file=False,
            log_file_path=None,
            logger_name="cen_dis_sim.test.routing_policy",
        ),
    )


class RoutingPolicyTest(unittest.TestCase):
    def test_prefill_routes_only_to_matching_resource_kind(self) -> None:
        short_engine = SimulationEngine(
            config=build_single_request_config(prompt_tokens=4096),
            seed=3,
        )
        short_request = short_engine.run().request_records[0]

        long_engine = SimulationEngine(
            config=build_single_request_config(prompt_tokens=32 * 1024),
            seed=3,
        )
        long_request = long_engine.run().request_records[0]

        self.assertEqual(short_request.target_prefill_pool_id, "central-short-prefill")
        self.assertEqual(long_request.target_prefill_pool_id, "central-long-prefill")

    def test_default_policy_uses_decode_cluster_weights(self) -> None:
        config = build_single_request_config(
            prompt_tokens=4096,
            scheduler=SchedulerConfig(
                allow_first_decode_cross_cluster=True,
                allow_following_decode_cross_cluster=False,
                prompt_len_threshold=16 * 1024,
                policy_config={
                    "cluster_routing_weights": {
                        "edge-a": {"decode": 0.0},
                        "edge-b": {"decode": 1.0},
                    }
                },
            ),
            clusters=[
                cluster(
                    "central",
                    pools=[pool("central-short-prefill", ResourceKind.SHORT_PREFILL, 1, 1)],
                ),
                cluster(
                    "edge-a",
                    pools=[pool("edge-a-decode", ResourceKind.DECODE, 1, 1)],
                ),
                cluster(
                    "edge-b",
                    pools=[pool("edge-b-decode", ResourceKind.DECODE, 1, 1)],
                ),
            ],
        )

        request = SimulationEngine(config=config, seed=9).run().request_records[0]

        self.assertEqual(request.target_decode_pool_id, "edge-b-decode")

    def test_first_decode_cross_cluster_flag_prefers_same_cluster_decode(self) -> None:
        config = build_single_request_config(
            prompt_tokens=4096,
            scheduler=SchedulerConfig(
                allow_first_decode_cross_cluster=False,
                allow_following_decode_cross_cluster=False,
                prompt_len_threshold=16 * 1024,
                policy_config={"cluster_routing_weights": {"edge": {"decode": 10.0}}},
            ),
            clusters=[
                cluster(
                    "central",
                    pools=[
                        pool("central-short-prefill", ResourceKind.SHORT_PREFILL, 1, 1),
                        pool("central-decode", ResourceKind.DECODE, 1, 1),
                    ],
                ),
                cluster(
                    "edge",
                    pools=[pool("edge-decode", ResourceKind.DECODE, 1, 1)],
                ),
            ],
        )

        request = SimulationEngine(config=config, seed=5).run().request_records[0]

        self.assertEqual(request.target_decode_pool_id, "central-decode")

    def test_first_decode_cross_cluster_flag_is_ignored_without_local_decode(self) -> None:
        config = build_single_request_config(
            prompt_tokens=4096,
            scheduler=SchedulerConfig(
                allow_first_decode_cross_cluster=False,
                allow_following_decode_cross_cluster=False,
                prompt_len_threshold=16 * 1024,
            ),
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
        )

        request = SimulationEngine(config=config, seed=5).run().request_records[0]

        self.assertEqual(request.target_decode_pool_id, "edge-decode")

    def test_following_decode_can_reroute_when_enabled(self) -> None:
        config = build_single_request_config(
            prompt_tokens=4096,
            scheduler=SchedulerConfig(
                allow_first_decode_cross_cluster=False,
                allow_following_decode_cross_cluster=True,
                prompt_len_threshold=16 * 1024,
                policy_config={
                    "cluster_routing_weights": {
                        "central": {"decode": 0.0},
                        "edge": {"decode": 1.0},
                    }
                },
            ),
            clusters=[
                cluster(
                    "central",
                    pools=[
                        pool("central-short-prefill", ResourceKind.SHORT_PREFILL, 1, 1),
                        pool("central-decode", ResourceKind.DECODE, 1, 1),
                    ],
                ),
                cluster(
                    "edge",
                    pools=[pool("edge-decode", ResourceKind.DECODE, 1, 1)],
                ),
            ],
        )

        engine = SimulationEngine(config=config, seed=4)
        request = engine.run().request_records[0]
        pool_stats = engine.pool_breakdown()

        self.assertEqual(request.decode_cluster_id, "edge")
        self.assertEqual(request.target_decode_pool_id, "edge-decode")
        self.assertEqual(pool_stats["central-decode"]["total_enqueued_requests"], 1)
        self.assertGreater(pool_stats["edge-decode"]["total_enqueued_requests"], 0)

    def test_following_decode_sticks_to_first_resource_by_default(self) -> None:
        config = build_single_request_config(
            prompt_tokens=4096,
            scheduler=SchedulerConfig(
                allow_first_decode_cross_cluster=False,
                allow_following_decode_cross_cluster=False,
                prompt_len_threshold=16 * 1024,
                policy_config={
                    "cluster_routing_weights": {
                        "central": {"decode": 0.0},
                        "edge": {"decode": 1.0},
                    }
                },
            ),
            clusters=[
                cluster(
                    "central",
                    pools=[
                        pool("central-short-prefill", ResourceKind.SHORT_PREFILL, 1, 1),
                        pool("central-decode", ResourceKind.DECODE, 1, 1),
                    ],
                ),
                cluster(
                    "edge",
                    pools=[pool("edge-decode", ResourceKind.DECODE, 1, 1)],
                ),
            ],
        )

        engine = SimulationEngine(config=config, seed=4)
        request = engine.run().request_records[0]
        pool_stats = engine.pool_breakdown()

        self.assertEqual(request.decode_cluster_id, "central")
        self.assertEqual(request.target_decode_pool_id, "central-decode")
        self.assertEqual(pool_stats["central-decode"]["total_enqueued_requests"], 1)
        self.assertEqual(pool_stats["edge-decode"]["total_enqueued_requests"], 0)

    def test_custom_policy_can_be_added_via_registry_without_scheduler_changes(self) -> None:
        class PreferLastPoolPolicy:
            name = "p_test_prefer_last"

            def select_pool(self, context: PolicyContext):
                return sorted(context.candidate_pools, key=lambda pool: pool.pool_id)[-1]

        register_policy(PreferLastPoolPolicy())

        config = build_single_request_config(
            prompt_tokens=4096,
            scheduler=SchedulerConfig(
                policy_name="p_test_prefer_last",
                allow_first_decode_cross_cluster=True,
                allow_following_decode_cross_cluster=False,
                prompt_len_threshold=16 * 1024,
            ),
            clusters=[
                cluster(
                    "central",
                    pools=[pool("central-short-prefill", ResourceKind.SHORT_PREFILL, 1, 1)],
                ),
                cluster(
                    "edge-a",
                    pools=[pool("edge-a-decode", ResourceKind.DECODE, 1, 1)],
                ),
                cluster(
                    "edge-b",
                    pools=[pool("edge-b-decode", ResourceKind.DECODE, 1, 1)],
                ),
            ],
        )

        request = SimulationEngine(config=config, seed=2).run().request_records[0]

        self.assertEqual(request.target_decode_pool_id, "edge-b-decode")


if __name__ == "__main__":
    unittest.main()
