from __future__ import annotations

from .config import (
    ResourceKind,
    ScenarioConfig,
    SchedulerConfig,
    SimulationConfig,
    cluster,
    pool,
)


def build_default_config() -> SimulationConfig:
    clusters = [
        cluster(
            "central",
            pools=[
                pool("central-long-prefill", ResourceKind.LONG_PREFILL, 4, 8),
                pool("central-short-prefill", ResourceKind.SHORT_PREFILL, 2, 8),
                pool("central-decode", ResourceKind.DECODE, 2, 16),
            ],
        ),
        cluster(
            "edge-a",
            pools=[
                pool("edge-a-short-prefill", ResourceKind.SHORT_PREFILL, 2, 8),
                pool("edge-a-decode", ResourceKind.DECODE, 4, 16),
            ],
        ),
        cluster(
            "edge-b",
            pools=[
                pool("edge-b-short-prefill", ResourceKind.SHORT_PREFILL, 2, 8),
                pool("edge-b-decode", ResourceKind.DECODE, 4, 16),
            ],
        ),
    ]
    config = SimulationConfig(
        scenario=ScenarioConfig(
            name="default",
            clusters=clusters,
            central_cluster_id="central",
        )
    )
    config.scheduler = SchedulerConfig(
        allow_first_decode_cross_cluster=True,
        allow_following_decode_cross_cluster=False,
        prompt_len_threshold=16 * 1024,
    )
    return config


def build_central_prefill_cross_decode_config() -> SimulationConfig:
    config = build_default_config()
    config.scenario.name = "central-prefill-cross-decode"
    config.scheduler.allow_first_decode_cross_cluster = True
    return config
