from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging


class ResourceKind(str, Enum):
    LONG_PREFILL = "long_prefill"
    SHORT_PREFILL = "short_prefill"
    DECODE = "decode"


@dataclass
class RequestGenerationConfig:
    # Number of independent users/sessions to simulate.
    user_count: int = 100
    # Each session samples its turn count uniformly within this inclusive range.
    min_turns_per_user: int = 1
    max_turns_per_user: int = 1
    # Mean inter-arrival time for brand-new users. This corresponds to a Poisson process.
    new_user_arrival_mean_seconds: float = 100.0
    # Delay between one turn finishing and the next turn arriving for the same user.
    followup_arrival_mean_seconds: float = 60.0
    followup_arrival_std_seconds: float = 15.0
    # When enabled, later turns reuse accumulated conversation history and outputs.
    # When disabled, every turn falls back to the first-turn prompt length.
    accumulate_context_across_turns: bool = False
    # New user-input tokens appended on every follow-up turn before sending the next request.
    # This only affects prompt growth when accumulate_context_across_turns is enabled.
    followup_prompt_tokens: int = 0
    # Probability that the first turn is sampled from the "4k-class" bucket.
    short_context_probability: float = 0.5
    # First-turn prompt length varies within +/- this ratio around the bucket center.
    initial_prompt_variation_ratio: float = 0.125
    short_context_prompt_tokens: int = 4 * 1024
    long_context_prompt_tokens: int = 128 * 1024
    short_context_output_tokens_min: int = 1024
    short_context_output_tokens_max: int = 1024
    long_context_output_tokens_min: int = 1024
    long_context_output_tokens_max: int = 1024


@dataclass
class PoolConfig:
    pool_id: str
    kind: ResourceKind
    instance_count: int
    max_batch_size: int


@dataclass
class ClusterConfig:
    cluster_id: str
    pools: list[PoolConfig]


def pool(
    pool_id: str,
    kind: ResourceKind,
    instance_count: int,
    max_batch_size: int,
) -> PoolConfig:
    return PoolConfig(
        pool_id=pool_id,
        kind=kind,
        instance_count=instance_count,
        max_batch_size=max_batch_size,
    )


def cluster(cluster_id: str, pools: list[PoolConfig]) -> ClusterConfig:
    return ClusterConfig(cluster_id=cluster_id, pools=pools)


@dataclass
class SchedulerConfig:
    policy_name: str = "p_default"
    policy_config: dict[str, object] = field(default_factory=dict)
    prompt_len_threshold: int = 16 * 1024
    allow_first_decode_cross_cluster: bool = False
    allow_following_decode_cross_cluster: bool = False


@dataclass
class PerformanceConfig:
    short_prefill_ms_per_token: float = 0.05
    long_prefill_ms_per_token: float = 0.025
    prefill_batch_size_penalty: float = 0.0
    decode_base_step_ms: float = 20.0
    decode_batch_size_slope_ms: float = 0.0122


@dataclass
class ScenarioConfig:
    name: str
    clusters: list[ClusterConfig]
    central_cluster_id: str | None = None


@dataclass
class SimulationConfig:
    request_generation: RequestGenerationConfig = field(
        default_factory=RequestGenerationConfig
    )
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: "LoggingConfig" = field(default_factory=lambda: LoggingConfig())
    scenario: ScenarioConfig = field(
        default_factory=lambda: ScenarioConfig(name="default", clusters=[])
    )


@dataclass
class LoggingConfig:
    level: int = logging.WARNING
    log_to_console: bool = True
    log_to_file: bool = False
    log_file_path: str | None = None
    logger_name: str = "cen_dis_sim"
