from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque, Iterable, Literal, Sequence

import numpy as np


DeploymentMode = Literal["centralized", "distributed"]
UserRegionPolicy = Literal["weight_region_policy", "queue_len_policy", "d_queue_len_policy"]


@dataclass(frozen=True)
class PrefillConfig:
    """短 Prefill 时长建模参数。"""

    lognormal_mu: float = 9.90
    lognormal_sigma: float = 1.00
    truncated_min_tokens: int = 128
    truncated_max_tokens: int = 128 * 1024
    short_prefill_max_tokens: int = 32 * 1024
    prefill_ms_per_token: float = 0.025
    zero_prob_by_region: float | Sequence[float] = 0.0


@dataclass(frozen=True)
class ThinkTimeConfig:
    """同一用户后续请求的 think time 配置。"""

    mean_ms: float = 60 * 1000.0
    std_ms: float = 15 * 1000.0
    min_ms: float = 5 * 1000.0
    max_ms: float = 5 * 60 * 1000.0


@dataclass(frozen=True)
class RequestCountConfig:
    """每个用户总请求数的整数采样区间。"""

    min_requests: int = 1
    max_requests: int = 1


@dataclass(frozen=True)
class SimulationConfig:
    """单次仿真的输入参数。

    请求生成分两段：
    1. 所有用户的首请求组成一个全局泊松到达流
    2. 某个用户的下一个请求，到达时间 = 上一个请求真实完成时间 + think time
    """

    region_names: Sequence[str]
    users_by_region: Sequence[int]
    first_request_arrival_interval_ms: float
    region_policy: UserRegionPolicy = "weight_region_policy"
    request_count: RequestCountConfig = RequestCountConfig()
    think_time: ThinkTimeConfig = ThinkTimeConfig()
    queue_wait_threshold_ms: float = 0.0
    service_duration_target_ms: float = 800.0
    seed: int = 42
    prefill: PrefillConfig = PrefillConfig()


@dataclass(frozen=True)
class DeploymentConfig:
    """部署方式配置。"""

    mode: DeploymentMode
    instances: int | Sequence[int]


@dataclass(frozen=True)
class RequestRecord:
    """单条请求的完整生命周期记录。"""

    id: int
    user_id: int
    request_index: int
    arrival_time: float
    region: str
    prefill_tokens: int
    prefill_time: float
    start_service_time: float
    end_service_time: float
    service_duration: float
    wait_time: float


@dataclass(frozen=True)
class StatsSummary:
    """请求集合的聚合统计。"""

    count: int
    queued_count: int
    queued_ratio: float
    avg_duration: float
    max_duration: float
    avg_wait: float
    max_wait: float
    within_target_count: int
    within_target_ratio: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class WaitBucketSummary:
    """wait_time <= threshold 的累计分布统计。"""

    threshold_ms: float
    wait_count: int
    wait_ratio: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class SimulationResult:
    mode: DeploymentMode
    requests: list[RequestRecord]
    overall_stats: StatsSummary
    region_stats: dict[str, StatsSummary]


@dataclass(frozen=True)
class PreparedSimulation:
    region_names: tuple[str, ...]
    users_by_region: tuple[int, ...]
    first_request_arrival_interval_ms: float
    region_policy: UserRegionPolicy
    request_count: RequestCountConfig
    think_time: ThinkTimeConfig
    queue_wait_threshold_ms: float
    service_duration_target_ms: float
    seed: int
    prefill: PrefillConfig
    zero_prob_by_region: tuple[float, ...]


@dataclass(frozen=True)
class PreparedDeployment:
    mode: DeploymentMode
    instance_counts: tuple[int, ...]


@dataclass(frozen=True)
class Workload:
    """与部署方式解耦的请求输入工作负载。

    这里不提前给用户绑定地域，而是只保存与部署无关的随机底稿：
    - 每个用户的总请求数
    - 每个用户的首请求到达时间
    - 每个请求在“非零 prefill”时的 token 数
    - 每个请求的 zero-prefill 判定随机数
    - 每个请求之间的 think time
    """

    region_names: tuple[str, ...]
    user_request_counts: np.ndarray
    user_first_arrival_times: np.ndarray
    user_request_offsets: np.ndarray
    user_think_time_offsets: np.ndarray
    request_nonzero_prefill_tokens: np.ndarray
    request_zero_draws: np.ndarray
    request_think_times_ms: np.ndarray

    @property
    def total_users(self) -> int:
        return int(self.user_request_counts.size)

    @property
    def total_requests(self) -> int:
        return int(self.request_nonzero_prefill_tokens.size)

    def user_request_count(self, user_index: int) -> int:
        return int(self.user_request_counts[user_index])

    def user_first_arrival_time(self, user_index: int) -> float:
        return float(self.user_first_arrival_times[user_index])

    def request_id_at(self, user_index: int, request_index: int) -> int:
        return int(self.user_request_offsets[user_index]) + request_index + 1

    def request_prefill_tokens_at(
        self,
        user_index: int,
        request_index: int,
        region_index: int,
        zero_prob_by_region: Sequence[float],
    ) -> int:
        request_offset = int(self.user_request_offsets[user_index]) + request_index
        if float(self.request_zero_draws[request_offset]) < float(zero_prob_by_region[region_index]):
            return 0
        return int(self.request_nonzero_prefill_tokens[request_offset])

    def think_time_after_request(self, user_index: int, request_index: int) -> float:
        think_offset = int(self.user_think_time_offsets[user_index]) + request_index
        return float(self.request_think_times_ms[think_offset])


@dataclass(frozen=True)
class PendingRequest:
    request_id: int
    user_id: int
    request_index: int
    region_index: int
    arrival_time: float
    prefill_tokens: int


@dataclass(frozen=True)
class RoutingSnapshot:
    """用于延迟路由策略的秒级系统状态快照。"""

    recorded_time_ms: float
    queue_lengths: np.ndarray
    in_service_counts: np.ndarray


def prepare_simulation_config(config: SimulationConfig) -> PreparedSimulation:
    region_names = tuple(config.region_names)
    users_by_region = tuple(int(value) for value in config.users_by_region)

    if not region_names:
        raise ValueError("region_names 不能为空。")
    if len(region_names) != len(users_by_region):
        raise ValueError("region_names 和 users_by_region 的长度必须一致。")
    if config.region_policy not in (
        "weight_region_policy",
        "queue_len_policy",
        "d_queue_len_policy",
    ):
        raise ValueError(
            "region_policy 必须是 weight_region_policy、queue_len_policy 或 d_queue_len_policy。"
        )
    if any(user_count < 0 for user_count in users_by_region):
        raise ValueError("users_by_region 不能包含负数。")
    if config.first_request_arrival_interval_ms <= 0:
        raise ValueError("first_request_arrival_interval_ms 必须大于 0。")
    if config.queue_wait_threshold_ms < 0:
        raise ValueError("queue_wait_threshold_ms 不能小于 0。")
    if config.service_duration_target_ms <= 0:
        raise ValueError("service_duration_target_ms 必须大于 0。")
    if config.request_count.min_requests <= 0:
        raise ValueError("request_count.min_requests 必须大于 0。")
    if config.request_count.max_requests < config.request_count.min_requests:
        raise ValueError("request_count.max_requests 不能小于 min_requests。")
    if config.think_time.std_ms <= 0:
        raise ValueError("think_time.std_ms 必须大于 0。")
    if config.think_time.min_ms < 0:
        raise ValueError("think_time.min_ms 不能小于 0。")
    if config.think_time.max_ms <= config.think_time.min_ms:
        raise ValueError("think_time.max_ms 必须大于 think_time.min_ms。")

    prefill = config.prefill
    if prefill.truncated_min_tokens <= 0:
        raise ValueError("prefill.truncated_min_tokens 必须大于 0。")
    if prefill.truncated_max_tokens <= prefill.truncated_min_tokens:
        raise ValueError("prefill.truncated_max_tokens 必须大于 truncated_min_tokens。")
    if prefill.short_prefill_max_tokens < prefill.truncated_min_tokens:
        raise ValueError("prefill.short_prefill_max_tokens 不能小于 truncated_min_tokens。")
    if prefill.short_prefill_max_tokens > prefill.truncated_max_tokens:
        raise ValueError("prefill.short_prefill_max_tokens 不能大于 truncated_max_tokens。")
    if prefill.prefill_ms_per_token <= 0:
        raise ValueError("prefill.prefill_ms_per_token 必须大于 0。")

    zero_prob_by_region = tuple(
        _resolve_region_values(
            prefill.zero_prob_by_region,
            region_names,
            "prefill.zero_prob_by_region",
        )
    )
    if any(prob < 0.0 or prob > 1.0 for prob in zero_prob_by_region):
        raise ValueError("prefill.zero_prob_by_region 的概率必须在 [0, 1]。")

    return PreparedSimulation(
        region_names=region_names,
        users_by_region=users_by_region,
        first_request_arrival_interval_ms=float(config.first_request_arrival_interval_ms),
        region_policy=config.region_policy,
        request_count=config.request_count,
        think_time=config.think_time,
        queue_wait_threshold_ms=float(config.queue_wait_threshold_ms),
        service_duration_target_ms=float(config.service_duration_target_ms),
        seed=int(config.seed),
        prefill=prefill,
        zero_prob_by_region=zero_prob_by_region,
    )


def prepare_deployment_config(
    simulation: PreparedSimulation,
    deployment_config: DeploymentConfig,
) -> PreparedDeployment:
    if deployment_config.mode not in ("centralized", "distributed"):
        raise ValueError("deployment_config.mode 必须是 centralized 或 distributed。")

    if deployment_config.mode == "centralized":
        if not isinstance(deployment_config.instances, int):
            raise ValueError("集中式部署的 instances 必须是整数。")
        if deployment_config.instances <= 0:
            raise ValueError("集中式部署的 instances 必须大于 0。")
        return PreparedDeployment(mode="centralized", instance_counts=(int(deployment_config.instances),))

    if isinstance(deployment_config.instances, int):
        raise ValueError("分布式部署的 instances 必须按地域分别配置。")

    instance_counts = tuple(int(value) for value in deployment_config.instances)
    if len(instance_counts) != len(simulation.region_names):
        raise ValueError("分布式部署的 instances 长度必须与 region_names 一致。")
    if any(instance_count <= 0 for instance_count in instance_counts):
        raise ValueError("分布式部署的每个地域实例数都必须大于 0。")
    return PreparedDeployment(mode="distributed", instance_counts=instance_counts)


def allocate_users(total_users: int, weights: Sequence[float]) -> list[int]:
    """按权重把总用户数分配到多个地域。"""

    if total_users < 0:
        raise ValueError("total_users 不能小于 0。")
    if not weights:
        raise ValueError("weights 不能为空。")

    weights_array = np.asarray(weights, dtype=float)
    if np.any(weights_array < 0):
        raise ValueError("weights 不能包含负数。")
    weight_sum = float(weights_array.sum())
    if weight_sum <= 0:
        raise ValueError("weights 之和必须大于 0。")

    normalized = weights_array / weight_sum
    raw = normalized * total_users
    allocated = np.floor(raw).astype(int)
    remainder = total_users - int(allocated.sum())
    if remainder > 0:
        fractional = raw - allocated
        order = np.argsort(-fractional)
        allocated[order[:remainder]] += 1
    return allocated.tolist()


def generate_workload(config: SimulationConfig | PreparedSimulation) -> Workload:
    """生成与部署方式解耦的 workload。"""

    prepared = config if isinstance(config, PreparedSimulation) else prepare_simulation_config(config)
    rng = np.random.default_rng(prepared.seed)

    total_users = int(sum(prepared.users_by_region))
    if total_users == 0:
        empty_int = np.asarray([], dtype=np.int32)
        empty_float = np.asarray([], dtype=np.float64)
        return Workload(
            region_names=prepared.region_names,
            user_request_counts=empty_int,
            user_first_arrival_times=empty_float,
            user_request_offsets=np.asarray([0], dtype=np.int32),
            user_think_time_offsets=np.asarray([0], dtype=np.int32),
            request_nonzero_prefill_tokens=empty_int,
            request_zero_draws=empty_float,
            request_think_times_ms=empty_float,
        )

    request_counts = rng.integers(
        low=prepared.request_count.min_requests,
        high=prepared.request_count.max_requests + 1,
        size=total_users,
        dtype=np.int32,
    )
    first_arrival_times = np.cumsum(
        rng.exponential(scale=prepared.first_request_arrival_interval_ms, size=total_users)
    ).astype(np.float64, copy=False)

    total_requests = int(request_counts.sum())
    request_nonzero_prefill_tokens = _sample_short_prefill_tokens(
        request_count=total_requests,
        simulation=prepared,
        rng=rng,
    )
    request_zero_draws = rng.random(total_requests).astype(np.float32, copy=False)

    user_request_offsets = np.empty(total_users + 1, dtype=np.int32)
    user_request_offsets[0] = 0
    np.cumsum(request_counts, dtype=np.int32, out=user_request_offsets[1:])

    think_counts = np.maximum(request_counts - 1, 0).astype(np.int32, copy=False)
    user_think_time_offsets = np.empty(total_users + 1, dtype=np.int32)
    user_think_time_offsets[0] = 0
    np.cumsum(think_counts, dtype=np.int32, out=user_think_time_offsets[1:])

    total_think_times = int(user_think_time_offsets[-1])
    think_time_config = prepared.think_time
    request_think_times_ms = _sample_truncated_normal(
        rng=rng,
        size=total_think_times,
        mean=think_time_config.mean_ms,
        std=think_time_config.std_ms,
        lower=think_time_config.min_ms,
        upper=think_time_config.max_ms,
    )

    return Workload(
        region_names=prepared.region_names,
        user_request_counts=request_counts,
        user_first_arrival_times=first_arrival_times,
        user_request_offsets=user_request_offsets,
        user_think_time_offsets=user_think_time_offsets,
        request_nonzero_prefill_tokens=request_nonzero_prefill_tokens,
        request_zero_draws=request_zero_draws,
        request_think_times_ms=request_think_times_ms,
    )


def simulate(
    simulation_config: SimulationConfig,
    deployment_config: DeploymentConfig,
) -> SimulationResult:
    prepared_simulation = prepare_simulation_config(simulation_config)
    workload = generate_workload(prepared_simulation)
    return simulate_workload(workload, prepared_simulation, deployment_config)


def simulate_workload(
    workload: Workload,
    simulation_config: SimulationConfig | PreparedSimulation,
    deployment_config: DeploymentConfig,
) -> SimulationResult:
    prepared_simulation = (
        simulation_config
        if isinstance(simulation_config, PreparedSimulation)
        else prepare_simulation_config(simulation_config)
    )
    prepared_deployment = prepare_deployment_config(prepared_simulation, deployment_config)

    requests = _run_event_loop(workload, prepared_simulation, prepared_deployment)
    return SimulationResult(
        mode=prepared_deployment.mode,
        requests=requests,
        overall_stats=summarize_requests(
            requests,
            queue_wait_threshold_ms=prepared_simulation.queue_wait_threshold_ms,
            service_duration_target_ms=prepared_simulation.service_duration_target_ms,
        ),
        region_stats=summarize_by_region(
            requests,
            queue_wait_threshold_ms=prepared_simulation.queue_wait_threshold_ms,
            service_duration_target_ms=prepared_simulation.service_duration_target_ms,
        ),
    )


def summarize_requests(
    requests: Sequence[RequestRecord],
    queue_wait_threshold_ms: float,
    service_duration_target_ms: float,
) -> StatsSummary:
    if not requests:
        return StatsSummary(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)

    wait_times = np.fromiter((request.wait_time for request in requests), dtype=float, count=len(requests))
    service_durations = np.fromiter(
        (request.service_duration for request in requests),
        dtype=float,
        count=len(requests),
    )
    count = int(wait_times.size)
    queued_count = int(np.count_nonzero(wait_times > queue_wait_threshold_ms))
    within_target_count = int(np.count_nonzero(service_durations <= service_duration_target_ms))
    return StatsSummary(
        count=count,
        queued_count=queued_count,
        queued_ratio=queued_count / count,
        avg_duration=float(service_durations.mean()),
        max_duration=float(service_durations.max()),
        avg_wait=float(wait_times.mean()),
        max_wait=float(wait_times.max()),
        within_target_count=within_target_count,
        within_target_ratio=within_target_count / count,
    )


def summarize_by_region(
    requests: Sequence[RequestRecord],
    queue_wait_threshold_ms: float,
    service_duration_target_ms: float,
) -> dict[str, StatsSummary]:
    aggregates: dict[str, dict[str, float | int]] = {}
    for request in requests:
        aggregate = aggregates.setdefault(
            request.region,
            {
                "count": 0,
                "queued_count": 0,
                "within_target_count": 0,
                "wait_sum": 0.0,
                "wait_max": 0.0,
                "duration_sum": 0.0,
                "duration_max": 0.0,
            },
        )
        aggregate["count"] += 1
        aggregate["wait_sum"] += request.wait_time
        aggregate["duration_sum"] += request.service_duration
        aggregate["wait_max"] = max(float(aggregate["wait_max"]), request.wait_time)
        aggregate["duration_max"] = max(float(aggregate["duration_max"]), request.service_duration)
        if request.wait_time > queue_wait_threshold_ms:
            aggregate["queued_count"] += 1
        if request.service_duration <= service_duration_target_ms:
            aggregate["within_target_count"] += 1

    summaries: dict[str, StatsSummary] = {}
    for region, aggregate in aggregates.items():
        count = int(aggregate["count"])
        queued_count = int(aggregate["queued_count"])
        within_target_count = int(aggregate["within_target_count"])
        summaries[region] = StatsSummary(
            count=count,
            queued_count=queued_count,
            queued_ratio=queued_count / count,
            avg_duration=float(aggregate["duration_sum"]) / count,
            max_duration=float(aggregate["duration_max"]),
            avg_wait=float(aggregate["wait_sum"]) / count,
            max_wait=float(aggregate["wait_max"]),
            within_target_count=within_target_count,
            within_target_ratio=within_target_count / count,
        )
    return summaries


def summarize_wait_time_buckets(
    requests: Sequence[RequestRecord],
    thresholds_ms: Sequence[float],
) -> list[WaitBucketSummary]:
    """统计 wait_time <= threshold 的累计分布。"""

    normalized_thresholds = _normalize_wait_time_thresholds(thresholds_ms)
    if not requests:
        return [
            WaitBucketSummary(threshold_ms=threshold, wait_count=0, wait_ratio=0.0)
            for threshold in normalized_thresholds
        ]

    wait_times = np.sort(
        np.fromiter((request.wait_time for request in requests), dtype=float, count=len(requests))
    )
    counts = np.searchsorted(wait_times, normalized_thresholds, side="right")
    total_count = int(wait_times.size)
    return [
        WaitBucketSummary(
            threshold_ms=float(threshold),
            wait_count=int(count),
            wait_ratio=float(count / total_count),
        )
        for threshold, count in zip(normalized_thresholds, counts)
    ]


def requests_to_rows(requests: Iterable[RequestRecord]) -> list[dict[str, float | int | str]]:
    return [
        {
            "id": request.id,
            "user_id": request.user_id,
            "request_index": request.request_index,
            "arrival_time": request.arrival_time,
            "region": request.region,
            "prefill_tokens": request.prefill_tokens,
            "prefill_time": request.prefill_time,
            "start_service_time": request.start_service_time,
            "end_service_time": request.end_service_time,
            "service_duration": request.service_duration,
            "wait_time": request.wait_time,
        }
        for request in requests
    ]


def format_stats_table(stats_by_label: Sequence[tuple[str, StatsSummary]]) -> str:
    headers = [
        "label",
        "request_count",
        "queued_ratio",
        "avg_wait",
        "max_wait",
        "avg_duration",
        "within_target_ratio",
    ]
    rows = [headers]
    for label, stats in stats_by_label:
        rows.append(
            [
                label,
                str(stats.count),
                f"{stats.queued_ratio:.2%}",
                f"{stats.avg_wait:.1f}",
                f"{stats.max_wait:.1f}",
                f"{stats.avg_duration:.1f}",
                f"{stats.within_target_ratio:.2%}",
            ]
        )

    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    return "\n".join(
        "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        for row in rows
    )


def expected_prefill_ms(config: SimulationConfig | PreparedSimulation) -> float:
    prepared = config if isinstance(config, PreparedSimulation) else prepare_simulation_config(config)
    prefill = prepared.prefill
    mu = prefill.lognormal_mu
    sigma = prefill.lognormal_sigma
    lower = math.log(prefill.truncated_min_tokens)
    upper = math.log(prefill.short_prefill_max_tokens)

    def std_norm_cdf(value: float) -> float:
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    numerator = std_norm_cdf((upper - mu - sigma**2) / sigma) - std_norm_cdf(
        (lower - mu - sigma**2) / sigma
    )
    denominator = std_norm_cdf((upper - mu) / sigma) - std_norm_cdf((lower - mu) / sigma)
    conditional_tokens = math.exp(mu + sigma**2 / 2.0) * (numerator / denominator)
    return conditional_tokens * prefill.prefill_ms_per_token


def _initialize_user_region_bindings(
    total_users: int,
    simulation: PreparedSimulation,
    deployment: PreparedDeployment,
    rng: np.random.Generator,
) -> np.ndarray:
    """初始化用户地域绑定。

    - centralized: 两种策略行为应一致，这里统一走权重绑定
    - distributed + weight_region_policy: 旧策略，用户启动时按权重固定绑定
    - distributed + queue_len_policy: 先不绑定，首请求到达时再动态选择
    """

    if total_users == 0:
        return np.asarray([], dtype=np.int16)

    if deployment.mode == "distributed" and simulation.region_policy in (
        "queue_len_policy",
        "d_queue_len_policy",
    ):
        return np.full(total_users, -1, dtype=np.int16)

    expanded = _expand_user_regions(simulation.users_by_region)
    return expanded[rng.permutation(total_users)]


def _choose_shortest_queue_region(
    queue_lengths: np.ndarray,
    in_service_counts: np.ndarray,
    rng: np.random.Generator,
) -> int:
    outstanding_counts = queue_lengths + in_service_counts
    min_outstanding = int(outstanding_counts.min())
    candidate_regions = np.flatnonzero(outstanding_counts == min_outstanding)
    if candidate_regions.size == 1:
        return int(candidate_regions[0])
    return int(candidate_regions[rng.integers(0, candidate_regions.size)])


def _maybe_update_routing_snapshot(
    snapshot: RoutingSnapshot | None,
    event_time_ms: float,
    queue_lengths: np.ndarray,
    in_service_counts: np.ndarray,
) -> RoutingSnapshot:
    """按秒级刷新一次路由快照。

    这里严格按需求实现：
    - 第一次看到事件时保存当前状态
    - 如果当前事件与上次保存处于同一秒，则丢弃
    - 如果秒级时间已变化，则用当前状态覆盖快照
    """

    event_second = int(event_time_ms // 1000.0)
    if snapshot is None:
        return RoutingSnapshot(
            recorded_time_ms=event_time_ms,
            queue_lengths=queue_lengths.copy(),
            in_service_counts=in_service_counts.copy(),
        )

    snapshot_second = int(snapshot.recorded_time_ms // 1000.0)
    if event_second == snapshot_second:
        return snapshot

    return RoutingSnapshot(
        recorded_time_ms=event_time_ms,
        queue_lengths=queue_lengths.copy(),
        in_service_counts=in_service_counts.copy(),
    )


def _run_event_loop(
    workload: Workload,
    simulation: PreparedSimulation,
    deployment: PreparedDeployment,
) -> list[RequestRecord]:
    """按事件时间推进仿真。

    事件只分两类：
    1. arrival: 请求真正进入系统的时刻
    2. completion: 某个实例完成 prefill 的时刻
    """

    event_heap: list[tuple[float, int, str, int, int, int]] = []
    event_counter = 0
    route_rng = np.random.default_rng(simulation.seed + 1)
    user_region_bindings = _initialize_user_region_bindings(
        total_users=workload.total_users,
        simulation=simulation,
        deployment=deployment,
        rng=route_rng,
    )

    for user_index in range(workload.total_users):
        user_id = user_index + 1
        heapq.heappush(
            event_heap,
            (workload.user_first_arrival_time(user_index), event_counter, "arrival", user_id, 0, -1),
        )
        event_counter += 1

    completed_requests: list[RequestRecord | None] = [None] * workload.total_requests

    if deployment.mode == "centralized":
        idle_instances = list(range(deployment.instance_counts[0]))
        wait_queue: Deque[PendingRequest] = deque()
        busy_requests: list[PendingRequest | None] = [None] * deployment.instance_counts[0]
        region_index = 0

        while event_heap:
            event_time, _, event_type, user_id, request_index, instance_id = heapq.heappop(event_heap)
            user_index = user_id - 1

            if event_type == "arrival":
                pending = PendingRequest(
                    request_id=workload.request_id_at(user_index, request_index),
                    user_id=user_id,
                    request_index=request_index,
                    region_index=region_index,
                    arrival_time=event_time,
                    prefill_tokens=workload.request_prefill_tokens_at(
                        user_index=user_index,
                        request_index=request_index,
                        region_index=region_index,
                        zero_prob_by_region=simulation.zero_prob_by_region,
                    ),
                )
                wait_queue.append(pending)
                event_counter = _dispatch_centralized_queue(
                    event_time=event_time,
                    wait_queue=wait_queue,
                    idle_instances=idle_instances,
                    busy_requests=busy_requests,
                    event_heap=event_heap,
                    event_counter=event_counter,
                    simulation=simulation,
                )
                continue

            pending = busy_requests[instance_id]
            if pending is None:
                raise RuntimeError("集中式完成事件缺少对应的运行中请求。")
            busy_requests[instance_id] = None
            completed_requests[pending.request_id - 1] = _build_request_record(
                pending=pending,
                start_service_time=event_time - pending.prefill_tokens * simulation.prefill.prefill_ms_per_token,
                end_service_time=event_time,
                region_name=workload.region_names[pending.region_index],
                prefill_ms_per_token=simulation.prefill.prefill_ms_per_token,
            )
            idle_instances.append(instance_id)

            next_request_index = request_index + 1
            if next_request_index < workload.user_request_count(user_index):
                heapq.heappush(
                    event_heap,
                    (
                        event_time + workload.think_time_after_request(user_index, request_index),
                        event_counter,
                        "arrival",
                        user_id,
                        next_request_index,
                        -1,
                    ),
                )
                event_counter += 1

            event_counter = _dispatch_centralized_queue(
                event_time=event_time,
                wait_queue=wait_queue,
                idle_instances=idle_instances,
                busy_requests=busy_requests,
                event_heap=event_heap,
                event_counter=event_counter,
                simulation=simulation,
            )
    else:
        idle_instances_by_region = [
            list(range(instance_count))
            for instance_count in deployment.instance_counts
        ]
        wait_queues = [deque() for _ in deployment.instance_counts]
        busy_requests: list[list[PendingRequest | None]] = [
            [None] * instance_count for instance_count in deployment.instance_counts
        ]
        queue_lengths = np.zeros(len(deployment.instance_counts), dtype=np.int32)
        in_service_counts = np.zeros(len(deployment.instance_counts), dtype=np.int32)
        routing_snapshot: RoutingSnapshot | None = None

        while event_heap:
            event_time, _, event_type, user_id, request_index, instance_id = heapq.heappop(event_heap)
            user_index = user_id - 1
            region_index = int(user_region_bindings[user_index])
            routing_snapshot = _maybe_update_routing_snapshot(
                snapshot=routing_snapshot,
                event_time_ms=event_time,
                queue_lengths=queue_lengths,
                in_service_counts=in_service_counts,
            )

            if event_type == "arrival":
                if region_index < 0:
                    route_queue_lengths = queue_lengths
                    route_in_service_counts = in_service_counts
                    if simulation.region_policy == "d_queue_len_policy":
                        if routing_snapshot is None:
                            raise RuntimeError("延迟路由快照尚未初始化。")
                        route_queue_lengths = routing_snapshot.queue_lengths
                        route_in_service_counts = routing_snapshot.in_service_counts
                    region_index = _choose_shortest_queue_region(
                        queue_lengths=route_queue_lengths,
                        in_service_counts=route_in_service_counts,
                        rng=route_rng,
                    )
                    user_region_bindings[user_index] = region_index

                pending = PendingRequest(
                    request_id=workload.request_id_at(user_index, request_index),
                    user_id=user_id,
                    request_index=request_index,
                    region_index=region_index,
                    arrival_time=event_time,
                    prefill_tokens=workload.request_prefill_tokens_at(
                        user_index=user_index,
                        request_index=request_index,
                        region_index=region_index,
                        zero_prob_by_region=simulation.zero_prob_by_region,
                    ),
                )
                wait_queues[region_index].append(pending)
                queue_lengths[region_index] += 1
                event_counter = _dispatch_distributed_queue(
                    event_time=event_time,
                    region_index=region_index,
                    wait_queue=wait_queues[region_index],
                    idle_instances=idle_instances_by_region[region_index],
                    busy_requests=busy_requests,
                    event_heap=event_heap,
                    event_counter=event_counter,
                    simulation=simulation,
                    queue_lengths=queue_lengths,
                    in_service_counts=in_service_counts,
                )
                continue

            if region_index < 0:
                raise RuntimeError("分布式请求完成前缺少用户地域绑定。")
            pending = busy_requests[region_index][instance_id]
            if pending is None:
                raise RuntimeError("分布式完成事件缺少对应的运行中请求。")
            busy_requests[region_index][instance_id] = None
            in_service_counts[region_index] -= 1
            completed_requests[pending.request_id - 1] = _build_request_record(
                pending=pending,
                start_service_time=event_time - pending.prefill_tokens * simulation.prefill.prefill_ms_per_token,
                end_service_time=event_time,
                region_name=workload.region_names[pending.region_index],
                prefill_ms_per_token=simulation.prefill.prefill_ms_per_token,
            )
            idle_instances_by_region[region_index].append(instance_id)

            next_request_index = request_index + 1
            if next_request_index < workload.user_request_count(user_index):
                heapq.heappush(
                    event_heap,
                    (
                        event_time + workload.think_time_after_request(user_index, request_index),
                        event_counter,
                        "arrival",
                        user_id,
                        next_request_index,
                        -1,
                    ),
                )
                event_counter += 1

            event_counter = _dispatch_distributed_queue(
                event_time=event_time,
                region_index=region_index,
                wait_queue=wait_queues[region_index],
                idle_instances=idle_instances_by_region[region_index],
                busy_requests=busy_requests,
                event_heap=event_heap,
                event_counter=event_counter,
                simulation=simulation,
                queue_lengths=queue_lengths,
                in_service_counts=in_service_counts,
            )

    return [request for request in completed_requests if request is not None]


def _dispatch_centralized_queue(
    event_time: float,
    wait_queue: Deque[PendingRequest],
    idle_instances: list[int],
    busy_requests: list[PendingRequest | None],
    event_heap: list[tuple[float, int, str, int, int, int]],
    event_counter: int,
    simulation: PreparedSimulation,
) -> int:
    """集中式只有一个全局队列，空闲实例统一从这里取请求。"""

    while wait_queue and idle_instances:
        pending = wait_queue.popleft()
        instance_id = idle_instances.pop()
        completion_time = event_time + pending.prefill_tokens * simulation.prefill.prefill_ms_per_token
        busy_requests[instance_id] = pending
        heapq.heappush(
            event_heap,
            (
                completion_time,
                event_counter,
                "completion",
                pending.user_id,
                pending.request_index,
                instance_id,
            ),
        )
        event_counter += 1
    return event_counter


def _dispatch_distributed_queue(
    event_time: float,
    region_index: int,
    wait_queue: Deque[PendingRequest],
    idle_instances: list[int],
    busy_requests: list[list[PendingRequest | None]],
    event_heap: list[tuple[float, int, str, int, int, int]],
    event_counter: int,
    simulation: PreparedSimulation,
    queue_lengths: np.ndarray,
    in_service_counts: np.ndarray,
) -> int:
    while wait_queue and idle_instances:
        pending = wait_queue.popleft()
        queue_lengths[region_index] -= 1
        instance_id = idle_instances.pop()
        completion_time = event_time + pending.prefill_tokens * simulation.prefill.prefill_ms_per_token
        busy_requests[region_index][instance_id] = pending
        in_service_counts[region_index] += 1
        heapq.heappush(
            event_heap,
            (
                completion_time,
                event_counter,
                "completion",
                pending.user_id,
                pending.request_index,
                instance_id,
            ),
        )
        event_counter += 1
    return event_counter


def _build_request_record(
    pending: PendingRequest,
    start_service_time: float,
    end_service_time: float,
    region_name: str,
    prefill_ms_per_token: float,
) -> RequestRecord:
    prefill_time = pending.prefill_tokens * prefill_ms_per_token
    wait_time = start_service_time - pending.arrival_time
    service_duration = end_service_time - pending.arrival_time
    return RequestRecord(
        id=pending.request_id,
        user_id=pending.user_id,
        request_index=pending.request_index,
        arrival_time=pending.arrival_time,
        region=region_name,
        prefill_tokens=pending.prefill_tokens,
        prefill_time=prefill_time,
        start_service_time=start_service_time,
        end_service_time=end_service_time,
        service_duration=service_duration,
        wait_time=wait_time,
    )


def _expand_user_regions(users_by_region: Sequence[int]) -> np.ndarray:
    """按地域用户数展开为逐用户地域索引数组。"""

    region_indices = np.arange(len(users_by_region), dtype=np.int16)
    return np.repeat(region_indices, np.asarray(users_by_region, dtype=np.int32))


def _sample_short_prefill_tokens(
    request_count: int,
    simulation: PreparedSimulation,
    rng: np.random.Generator,
) -> np.ndarray:
    """采样短 Prefill token 数；超过上限的样本直接丢弃并重采样。"""

    if request_count == 0:
        return np.asarray([], dtype=np.int32)

    prefill = simulation.prefill
    tokens = np.empty(request_count, dtype=np.int32)
    filled = 0
    while filled < request_count:
        needed = request_count - filled
        batch_size = max(needed * 2, 512)
        sampled = np.rint(
            rng.lognormal(mean=prefill.lognormal_mu, sigma=prefill.lognormal_sigma, size=batch_size)
        ).astype(np.int32)
        valid = sampled[
            (sampled >= prefill.truncated_min_tokens)
            & (sampled <= prefill.short_prefill_max_tokens)
        ]
        if valid.size == 0:
            continue
        take = min(needed, int(valid.size))
        tokens[filled : filled + take] = valid[:take]
        filled += take
    return tokens


def _sample_truncated_normal(
    rng: np.random.Generator,
    size: int,
    mean: float,
    std: float,
    lower: float,
    upper: float,
) -> np.ndarray:
    """简单拒绝采样版截断正态。"""

    if size <= 0:
        return np.asarray([], dtype=float)

    samples = np.empty(size, dtype=float)
    filled = 0
    while filled < size:
        needed = size - filled
        batch_size = max(needed * 2, 64)
        batch = rng.normal(loc=mean, scale=std, size=batch_size)
        valid = batch[(batch >= lower) & (batch <= upper)]
        if valid.size == 0:
            continue
        take = min(needed, int(valid.size))
        samples[filled : filled + take] = valid[:take]
        filled += take
    return samples


def _resolve_region_values(
    value: float | Sequence[float],
    region_names: Sequence[str],
    field_name: str,
) -> list[float]:
    if isinstance(value, (int, float)):
        resolved = [float(value)] * len(region_names)
    else:
        resolved = [float(item) for item in value]
        if len(resolved) != len(region_names):
            raise ValueError(f"{field_name} 的长度必须与 region_names 一致。")
    return resolved


def _normalize_wait_time_thresholds(thresholds_ms: Sequence[float]) -> list[float]:
    if not thresholds_ms:
        raise ValueError("thresholds_ms 不能为空。")
    normalized = sorted(float(value) for value in thresholds_ms)
    if any(value < 0 for value in normalized):
        raise ValueError("thresholds_ms 不能包含负数。")
    return normalized
