from __future__ import annotations

from dataclasses import dataclass, field

from .config import ResourceKind
from .models import Request, ResourcePool


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _percentiles(
    values: list[float], percentiles: tuple[float, ...]
) -> dict[float, float | None]:
    if not values:
        return {percentile: None for percentile in percentiles}

    ordered = sorted(values)
    if len(ordered) == 1:
        only_value = ordered[0]
        return {percentile: only_value for percentile in percentiles}

    results: dict[float, float | None] = {}
    last_index = len(ordered) - 1
    for percentile in percentiles:
        index = last_index * percentile
        lower = int(index)
        upper = min(lower + 1, last_index)
        fraction = index - lower
        results[percentile] = ordered[lower] + (
            ordered[upper] - ordered[lower]
        ) * fraction
    return results


@dataclass
class SimulationMetrics:
    request_records: list[Request] = field(default_factory=list)
    total_prefill_batches: int = 0
    total_decode_batches: int = 0
    total_prefill_batch_requests: int = 0
    total_decode_batch_requests: int = 0
    max_prefill_batch_size: int = 0
    max_decode_batch_size: int = 0
    cross_cluster_decode_transfers: int = 0

    def summary(self) -> dict[str, float | int | None]:
        completed: list[Request] = []
        session_ids: set[int | None] = set()
        prefill_first_token_values: list[float] = []
        e2e_values: list[float] = []
        request_tpot_values: list[float] = []
        prefill_queue_values: list[float] = []
        decode_queue_values: list[float] = []
        total_output_tokens = 0
        total_decode_tokens = 0
        total_prompt_tokens = 0
        short_prefill_total_requests = 0
        long_prefill_total_requests = 0
        short_prefill_queued_count = 0
        long_prefill_queued_count = 0
        min_arrival_time_ms: float | None = None
        max_finish_time_ms = 0.0

        for request in self.request_records:
            session_ids.add(request.session_id)
            if request.target_prefill_kind == ResourceKind.SHORT_PREFILL:
                short_prefill_total_requests += 1
                if request.queue_time_prefill_ms > 0.0:
                    short_prefill_queued_count += 1
            elif request.target_prefill_kind == ResourceKind.LONG_PREFILL:
                long_prefill_total_requests += 1
                if request.queue_time_prefill_ms > 0.0:
                    long_prefill_queued_count += 1
            finish_time_ms = request.finish_time_ms
            if not finish_time_ms:
                continue

            completed.append(request)
            if min_arrival_time_ms is None or request.arrival_time_ms < min_arrival_time_ms:
                min_arrival_time_ms = request.arrival_time_ms
            if finish_time_ms > max_finish_time_ms:
                max_finish_time_ms = finish_time_ms

            prefill_first_token_latency_ms = request.prefill_first_token_latency_ms
            if prefill_first_token_latency_ms is not None:
                prefill_first_token_values.append(prefill_first_token_latency_ms)

            end_to_end_ms = request.end_to_end_ms
            if end_to_end_ms is not None:
                e2e_values.append(end_to_end_ms)

            request_tpot_ms = request.request_tpot_ms
            if request_tpot_ms is not None:
                request_tpot_values.append(request_tpot_ms)

            prefill_queue_values.append(request.queue_time_prefill_ms)
            decode_queue_values.append(request.queue_time_decode_ms)
            total_output_tokens += request.output_tokens
            total_decode_tokens += max(request.output_tokens - 1, 0)
            total_prompt_tokens += request.prompt_tokens

        prefill_first_token_percentiles = _percentiles(prefill_first_token_values, (0.5, 0.95))
        e2e_percentiles = _percentiles(e2e_values, (0.5, 0.95))
        prefill_queue_percentiles = _percentiles(prefill_queue_values, (0.5, 0.95))
        decode_queue_percentiles = _percentiles(decode_queue_values, (0.5, 0.95))
        request_tpot_percentiles = _percentiles(request_tpot_values, (0.5, 0.95))
        active_window_ms = (
            max_finish_time_ms - min_arrival_time_ms
            if completed and min_arrival_time_ms is not None
            else 0.0
        )
        active_window_s = active_window_ms / 1000.0 if active_window_ms > 0 else None
        system_tpot_avg_ms = (
            sum(request_tpot_values) / len(request_tpot_values)
            if request_tpot_values
            else None
        )
        request_tpot_avg_ms = system_tpot_avg_ms
        prefill_first_token_latency_avg_ms = (
            sum(prefill_first_token_values) / len(prefill_first_token_values)
            if prefill_first_token_values
            else None
        )
        prefill_first_token_latency_max_ms = (
            max(prefill_first_token_values) if prefill_first_token_values else None
        )
        prefill_queue_avg_ms = (
            sum(prefill_queue_values) / len(prefill_queue_values)
            if prefill_queue_values
            else None
        )
        prefill_queue_max_ms = (
            max(prefill_queue_values) if prefill_queue_values else None
        )
        prefill_queued_count = sum(1 for value in prefill_queue_values if value > 0.0)
        tpot_le_50ms_count = sum(1 for value in request_tpot_values if value <= 50.0)
        ttft_le_2s_count = sum(
            1 for value in prefill_first_token_values if value <= 2000.0
        )
        avg_prefill_batch_size = (
            self.total_prefill_batch_requests / self.total_prefill_batches
            if self.total_prefill_batches
            else None
        )
        avg_decode_batch_size = (
            self.total_decode_batch_requests / self.total_decode_batches
            if self.total_decode_batches
            else None
        )
        return {
            "sessions_total": len(session_ids),
            "requests_total": len(self.request_records),
            "requests_completed": len(completed),
            "prefill_batch_executions": self.total_prefill_batches,
            "decode_batch_executions": self.total_decode_batches,
            "prefill_batch_size_avg": avg_prefill_batch_size,
            "prefill_batch_size_max": self.max_prefill_batch_size,
            "decode_batch_size_avg": avg_decode_batch_size,
            "decode_batch_size_max": self.max_decode_batch_size,
            "cross_cluster_decode_transfers": self.cross_cluster_decode_transfers,
            # Backward-compatible aliases for the historical TTFT naming.
            "ttft_p50_ms": prefill_first_token_percentiles[0.5],
            "ttft_p95_ms": prefill_first_token_percentiles[0.95],
            "prefill_first_token_latency_p50_ms": prefill_first_token_percentiles[0.5],
            "prefill_first_token_latency_p95_ms": prefill_first_token_percentiles[0.95],
            "e2e_p50_ms": e2e_percentiles[0.5],
            "e2e_p95_ms": e2e_percentiles[0.95],
            "prefill_queue_p50_ms": prefill_queue_percentiles[0.5],
            "prefill_queue_p95_ms": prefill_queue_percentiles[0.95],
            "decode_queue_p50_ms": decode_queue_percentiles[0.5],
            "decode_queue_p95_ms": decode_queue_percentiles[0.95],
            # Backward-compatible alias for callers that still expect a single TPOT key.
            "tpot_ms": system_tpot_avg_ms,
            "request_tpot_avg_ms": request_tpot_avg_ms,
            "request_tpot_p50_ms": request_tpot_percentiles[0.5],
            "request_tpot_p95_ms": request_tpot_percentiles[0.95],
            "system_tpot_avg_ms": system_tpot_avg_ms,
            "prefill_first_token_latency_avg_ms": prefill_first_token_latency_avg_ms,
            "prefill_first_token_latency_max_ms": prefill_first_token_latency_max_ms,
            "prefill_queue_avg_ms": prefill_queue_avg_ms,
            "prefill_queue_max_ms": prefill_queue_max_ms,
            "prefill_queued_count": prefill_queued_count,
            "prefill_queued_ratio": (
                prefill_queued_count / len(self.request_records)
                if self.request_records
                else None
            ),
            "short_prefill_queued_count": short_prefill_queued_count,
            "short_prefill_queued_ratio": (
                short_prefill_queued_count / short_prefill_total_requests
                if short_prefill_total_requests
                else None
            ),
            "long_prefill_queued_count": long_prefill_queued_count,
            "long_prefill_queued_ratio": (
                long_prefill_queued_count / long_prefill_total_requests
                if long_prefill_total_requests
                else None
            ),
            # Backward-compatible aliases for older generic prefill queue names.
            "req_queued_count": prefill_queued_count,
            "req_queued_ratio": (
                prefill_queued_count / len(self.request_records)
                if self.request_records
                else None
            ),
            "simulation_total_time_ms": max_finish_time_ms if completed else None,
            "active_window_ms": active_window_ms if completed else None,
            "request_throughput_rps": (
                len(completed) / active_window_s if active_window_s else None
            ),
            "output_token_throughput_tps": (
                total_output_tokens / active_window_s if active_window_s else None
            ),
            "decode_token_throughput_tps": (
                total_decode_tokens / active_window_s if active_window_s else None
            ),
            "prefill_token_throughput_tps": (
                total_prompt_tokens / active_window_s if active_window_s else None
            ),
            "requests_with_tpot_le_50ms": tpot_le_50ms_count,
            "requests_with_tpot_le_50ms_ratio": (
                tpot_le_50ms_count / len(request_tpot_values)
                if request_tpot_values
                else None
            ),
            "requests_with_prefill_first_token_latency_le_2s": ttft_le_2s_count,
            "requests_with_prefill_first_token_latency_le_2s_ratio": (
                ttft_le_2s_count / len(prefill_first_token_values)
                if prefill_first_token_values
                else None
            ),
        }

    def global_summary(
        self,
        pools: dict[str, ResourcePool] | None = None,
        total_time_ms: float | None = None,
    ) -> dict[str, float | int | None]:
        summary = self.summary()
        short_prefill_utilization = None
        long_prefill_utilization = None
        decode_utilization = None
        if pools is not None and total_time_ms and total_time_ms > 0:
            short_prefill_utilization = self._resource_kind_utilization(
                pools, ResourceKind.SHORT_PREFILL, total_time_ms
            )
            long_prefill_utilization = self._resource_kind_utilization(
                pools, ResourceKind.LONG_PREFILL, total_time_ms
            )
            decode_utilization = self._resource_kind_utilization(
                pools, ResourceKind.DECODE, total_time_ms
            )
        return {
            "request_tpot_avg_ms": summary["request_tpot_avg_ms"],
            "system_tpot_avg_ms": summary["system_tpot_avg_ms"],
            "prefill_first_token_latency_avg_ms": summary[
                "prefill_first_token_latency_avg_ms"
            ],
            "prefill_first_token_latency_max_ms": summary[
                "prefill_first_token_latency_max_ms"
            ],
            "prefill_queue_avg_ms": summary["prefill_queue_avg_ms"],
            "prefill_queue_max_ms": summary["prefill_queue_max_ms"],
            "prefill_queued_count": summary["prefill_queued_count"],
            "prefill_queued_ratio": summary["prefill_queued_ratio"],
            "short_prefill_queued_count": summary["short_prefill_queued_count"],
            "short_prefill_queued_ratio": summary["short_prefill_queued_ratio"],
            "long_prefill_queued_count": summary["long_prefill_queued_count"],
            "long_prefill_queued_ratio": summary["long_prefill_queued_ratio"],
            "short_prefill_utilization": short_prefill_utilization,
            "long_prefill_utilization": long_prefill_utilization,
            "decode_utilization": decode_utilization,
            "simulation_total_time_ms": summary["simulation_total_time_ms"],
            "request_throughput_rps": summary["request_throughput_rps"],
            "output_token_throughput_tps": summary["output_token_throughput_tps"],
            "decode_token_throughput_tps": summary["decode_token_throughput_tps"],
            "prefill_token_throughput_tps": summary["prefill_token_throughput_tps"],
        }

    def _resource_kind_utilization(
        self,
        pools: dict[str, ResourcePool],
        kind: ResourceKind,
        total_time_ms: float,
    ) -> float | None:
        instances = [
            instance
            for pool in pools.values()
            if pool.kind == kind
            for instance in pool.instances
        ]
        if not instances:
            return None
        total_busy_time_ms = sum(instance.total_busy_time_ms for instance in instances)
        return total_busy_time_ms / (len(instances) * total_time_ms)

    def request_breakdown(self, limit: int | None = None) -> list[dict[str, object]]:
        records = self.request_records if limit is None else self.request_records[:limit]
        return [
            {
                "request_id": request.request_id,
                "session_id": request.session_id,
                "turn_index": request.turn_index,
                "total_turns": request.total_turns,
                "initial_prompt_bucket": request.initial_prompt_bucket,
                "history_tokens": request.history_tokens,
                "new_prompt_tokens": request.new_prompt_tokens,
                "arrival_time_ms": request.arrival_time_ms,
                "prompt_tokens": request.prompt_tokens,
                "output_tokens": request.output_tokens,
                "prefill_pool_id": request.target_prefill_pool_id,
                "prefill_kind": (
                    request.target_prefill_kind.value
                    if request.target_prefill_kind is not None
                    else None
                ),
                "decode_pool_id": request.target_decode_pool_id,
                "prefill_cluster_id": request.prefill_cluster_id,
                "decode_cluster_id": request.decode_cluster_id,
                "first_token_time_ms": request.first_token_time_ms,
                "prefill_queue_time_ms": request.queue_time_prefill_ms,
                "decode_queue_time_ms": request.queue_time_decode_ms,
                # Backward-compatible alias for historical request breakdown consumers.
                "ttft_ms": request.prefill_first_token_latency_ms,
                "prefill_first_token_latency_ms": request.prefill_first_token_latency_ms,
                "end_to_end_ms": request.end_to_end_ms,
                "request_tpot_ms": request.request_tpot_ms,
                "decode_step_count": request.decode_step_count,
                "crossed_cluster_for_decode": request.crossed_cluster_for_decode,
                "stage": request.stage.value,
            }
            for request in records
        ]

    def pool_breakdown(
        self, pools: dict[str, ResourcePool], total_time_ms: float
    ) -> dict[str, dict[str, float | int | str]]:
        breakdown: dict[str, dict[str, float | int | str]] = {}
        for pool_id, pool in pools.items():
            avg_queue_depth = (pool.queue_depth_area_ms / total_time_ms) if total_time_ms else 0.0
            breakdown[pool_id] = {
                "cluster_id": pool.cluster_id,
                "kind": pool.kind.value,
                "instance_count": len(pool.instances),
                "max_batch_size": pool.max_batch_size,
                "resident_count_final": pool.resident_count(),
                "decode_capacity": pool.decode_capacity()
                if pool.kind.value == "decode"
                else 0,
                "max_queue_length_seen": pool.max_queue_length_seen,
                "avg_queue_depth": avg_queue_depth,
                "total_enqueued_requests": pool.total_enqueued_requests,
                "total_dispatched_requests": pool.total_dispatched_requests,
            }
        return breakdown
