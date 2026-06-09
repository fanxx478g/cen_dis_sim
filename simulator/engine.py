from __future__ import annotations

import heapq
import logging
import random

from .config import ResourceKind, SimulationConfig
from .logging_utils import build_logger
from .metrics import SimulationMetrics
from .models import Request, RequestStage, ResourceInstance, ResourcePool, TaskBatch
from .request_generation import RequestGenerator
from .scheduler import Scheduler


class SimulationEngine:
    def __init__(self, config: SimulationConfig, seed: int = 7) -> None:
        self.config = config
        self.random = random.Random(seed)
        self.scheduler = Scheduler(config, rng=self.random)
        self.request_generator = RequestGenerator(config.request_generation, self.random)
        self.logger = build_logger(config.logging)
        self.current_time_ms = 0.0
        self.next_batch_id = 1
        self.requests: dict[int, Request] = {}
        self.batches: dict[int, TaskBatch] = {}
        self.batch_finish_heap: list[tuple[float, int]] = []
        self.batch_instances: dict[int, ResourceInstance] = {}
        self.metrics = SimulationMetrics()
        self.pools = self._build_pools()
        self.pool_list = list(self.pools.values())
        self.prefill_pools_by_kind = self._build_prefill_pool_index()
        self.decode_pools = [
            pool for pool in self.pool_list if pool.kind == ResourceKind.DECODE
        ]
        self.dispatchable_pool_ids: dict[str, None] = {}
        self.debug_logging_enabled = self.logger.isEnabledFor(logging.DEBUG)
        self.info_logging_enabled = self.logger.isEnabledFor(logging.INFO)
        self.critical_logging_enabled = self.logger.isEnabledFor(logging.CRITICAL)

    def run(self) -> SimulationMetrics:
        self.request_generator.initialize()
        if self.critical_logging_enabled:
            self._log(
                logging.CRITICAL,
                "仿真开始。",
                details={
                    "scenario": self.config.scenario.name,
                    "session_count": self.request_generator.session_count,
                    "cluster_count": len(self.config.scenario.clusters),
                    "topology": self._topology_summary(),
                },
            )
        while (
            self.request_generator.has_pending_arrivals()
            or self.batches
            or self._has_pending_pool_work()
        ):
            next_arrival_time_ms = self.request_generator.next_arrival_time_ms()
            next_finish_time_ms = self._next_batch_finish_time_before(next_arrival_time_ms)
            if next_finish_time_ms is not None:
                self.current_time_ms = next_finish_time_ms
                finished_request_ids = self._finalize_batches_at_current_time()
                for request_id in finished_request_ids:
                    self.request_generator.on_request_finished(self.requests[request_id])
                self._dispatch_all_idle_capacity()
                continue
            if next_arrival_time_ms is None:
                break
            self._advance_until(next_arrival_time_ms)
            self.current_time_ms = next_arrival_time_ms
            self._accept_ready_requests()

        self._drain_system()
        self._finalize_pool_queue_stats()
        self.metrics.request_records = list(self.requests.values())
        global_summary = self.metrics.global_summary(
            pools=self.pools,
            total_time_ms=self.current_time_ms,
        )
        if self.critical_logging_enabled:
            self._log(
                logging.CRITICAL,
                "仿真完成事件：所有推理请求均已完成。",
                details={
                    "requests_completed": len(self.metrics.request_records),
                    "session_count": self.request_generator.session_count,
                    "decode_batch_executions": self.metrics.total_decode_batches,
                    "decode_batch_size_max": self.metrics.max_decode_batch_size,
                },
            )
            self._log(
                logging.CRITICAL,
                "global_summary",
                details=global_summary,
            )
        return self.metrics

    def _build_pools(self) -> dict[str, ResourcePool]:
        pools: dict[str, ResourcePool] = {}
        for cluster in self.config.scenario.clusters:
            for pool_config in cluster.pools:
                instances = [
                    ResourceInstance(
                        instance_id=f"{pool_config.pool_id}-inst-{index}",
                        cluster_id=cluster.cluster_id,
                        pool_id=pool_config.pool_id,
                        kind=pool_config.kind,
                    )
                    for index in range(pool_config.instance_count)
                ]
                pools[pool_config.pool_id] = ResourcePool(
                    pool_id=pool_config.pool_id,
                    cluster_id=cluster.cluster_id,
                    kind=pool_config.kind,
                    max_batch_size=pool_config.max_batch_size,
                    instances=instances,
                )
        return pools

    def _build_prefill_pool_index(self) -> dict[ResourceKind, list[ResourcePool]]:
        index = {
            ResourceKind.LONG_PREFILL: [],
            ResourceKind.SHORT_PREFILL: [],
        }
        for pool in self.pool_list:
            if pool.kind in index:
                index[pool.kind].append(pool)
        return index

    def _accept_ready_requests(self) -> None:
        for request in self.request_generator.take_ready_requests(self.current_time_ms):
            self.requests[request.request_id] = request
            self._accept_new_request(request)

    def _advance_until(self, target_time_ms: float) -> None:
        while True:
            next_finish_time = self._next_batch_finish_time_before(target_time_ms)
            if next_finish_time is None:
                break
            self.current_time_ms = next_finish_time
            finished_request_ids = self._finalize_batches_at_current_time()
            for request_id in finished_request_ids:
                self.request_generator.on_request_finished(self.requests[request_id])
            self._dispatch_all_idle_capacity()
        self.current_time_ms = target_time_ms
        self._dispatch_all_idle_capacity()

    def _next_batch_finish_time_before(self, target_time_ms: float | None) -> float | None:
        while self.batch_finish_heap:
            finish_time_ms, batch_id = self.batch_finish_heap[0]
            if batch_id not in self.batches:
                heapq.heappop(self.batch_finish_heap)
                continue
            if target_time_ms is not None and finish_time_ms > target_time_ms:
                return None
            return finish_time_ms
        return None

    def _finalize_batches_at_current_time(self) -> list[int]:
        finished_request_ids: list[int] = []
        while self.batch_finish_heap:
            finish_time_ms, batch_id = self.batch_finish_heap[0]
            if batch_id not in self.batches:
                heapq.heappop(self.batch_finish_heap)
                continue
            if finish_time_ms > self.current_time_ms:
                break

            heapq.heappop(self.batch_finish_heap)
            batch = self.batches.pop(batch_id)
            pool = self.pools[batch.pool_id]
            instance = self.batch_instances.pop(batch.batch_id)
            instance.current_batch_id = None
            instance.total_busy_time_ms += batch.finish_time_ms - batch.start_time_ms
            pool.release_instance(instance.instance_id)
            if batch.kind in (ResourceKind.LONG_PREFILL, ResourceKind.SHORT_PREFILL):
                finished_request_ids.extend(self._finish_prefill_batch(batch))
                self.metrics.total_prefill_batches += 1
                self.metrics.total_prefill_batch_requests += len(batch.request_ids)
                self.metrics.max_prefill_batch_size = max(
                    self.metrics.max_prefill_batch_size, len(batch.request_ids)
                )
            else:
                finished_request_ids.extend(self._finish_decode_batch(batch))
                self.metrics.total_decode_batches += 1
                self.metrics.total_decode_batch_requests += len(batch.request_ids)
                self.metrics.max_decode_batch_size = max(
                    self.metrics.max_decode_batch_size, len(batch.request_ids)
                )
            self._refresh_dispatchable_pool(pool)
        return finished_request_ids

    def _finish_prefill_batch(self, batch: TaskBatch) -> list[int]:
        routed_by_pool: dict[str, list[int]] = {}
        cross_cluster_by_pool: dict[str, int] = {}
        finished_request_ids: list[int] = []

        for request_id in batch.request_ids:
            request = self.requests[request_id]
            request.prefill_end_time_ms = batch.finish_time_ms
            request.prefill_cluster_id = batch.cluster_id
            request.last_enqueue_time_ms = self.current_time_ms

            if request.output_tokens > 0:
                request.generated_tokens = 1
                request.first_token_time_ms = batch.finish_time_ms
            else:
                request.first_token_time_ms = batch.finish_time_ms

            if request.generated_tokens >= request.output_tokens:
                request.stage = RequestStage.FINISHED
                request.finish_time_ms = batch.finish_time_ms
                finished_request_ids.append(request_id)
                continue

            request.stage = RequestStage.DECODE_WAITING
            decode_pool = self.scheduler.select_decode_pool(
                request,
                self.decode_pools,
                is_first_decode=True,
            )
            request.target_decode_pool_id = decode_pool.pool_id
            request.decode_cluster_id = decode_pool.cluster_id
            request.crossed_cluster_for_decode = (
                request.prefill_cluster_id is not None
                and request.decode_cluster_id is not None
                and request.prefill_cluster_id != request.decode_cluster_id
            )
            if request.crossed_cluster_for_decode:
                self.metrics.cross_cluster_decode_transfers += 1
                cross_cluster_by_pool[decode_pool.pool_id] = (
                    cross_cluster_by_pool.get(decode_pool.pool_id, 0) + 1
                )
            routed_by_pool.setdefault(decode_pool.pool_id, []).append(request.request_id)

        if self.info_logging_enabled:
            self._log(
                logging.INFO,
                "资源池完成Prefill批次。",
                batch_id=batch.batch_id,
                cluster_id=batch.cluster_id,
                pool_id=batch.pool_id,
                details={
                    "resource_kind": batch.kind.value,
                    "batch_size": len(batch.request_ids),
                },
            )

        if finished_request_ids and self.info_logging_enabled:
            self._log(
                logging.INFO,
                "资源池完成请求。",
                cluster_id=batch.cluster_id,
                pool_id=batch.pool_id,
                details={"completed_requests": len(finished_request_ids)},
            )

        for pool_id, request_ids in routed_by_pool.items():
            decode_pool = self.pools[pool_id]
            self._enqueue_requests(
                decode_pool,
                request_ids,
                event_type="decode_pool_received_requests",
                details={
                    "source_cluster_id": batch.cluster_id,
                    "request_count": len(request_ids),
                    "cross_cluster_count": cross_cluster_by_pool.get(pool_id, 0),
                },
            )
        return finished_request_ids

    def _finish_decode_batch(self, batch: TaskBatch) -> list[int]:
        pool = self.pools[batch.pool_id]
        completed_request_ids: set[int] = set()
        continuing_request_ids: set[int] = set()
        rerouted_by_pool: dict[str, list[int]] = {}
        cross_cluster_by_pool: dict[str, int] = {}
        pool.clear_inflight_requests(batch.request_ids)

        for request_id in batch.request_ids:
            request = self.requests[request_id]
            request.generated_tokens += 1
            request.decode_step_count += 1
            if request.generated_tokens >= request.output_tokens:
                request.stage = RequestStage.FINISHED
                request.finish_time_ms = batch.finish_time_ms
                completed_request_ids.add(request_id)
            else:
                continuing_request_ids.add(request_id)
                if self.config.scheduler.allow_following_decode_cross_cluster:
                    # When enabled, every decode step can hand the live sequence back to
                    # the scheduler and migrate it to another decode pool.
                    # TODO: `cross_cluster_decode_transfers` currently counts only the
                    # first decode handoff after prefill. If we later want this metric
                    # to represent all cross-cluster decode migrations, increment it for
                    # these follow-up reroutes as well.
                    request.stage = RequestStage.DECODE_WAITING
                    request.last_enqueue_time_ms = batch.finish_time_ms
                    next_decode_pool = self.scheduler.select_decode_pool(
                        request,
                        self.decode_pools,
                        is_first_decode=False,
                    )
                    request.target_decode_pool_id = next_decode_pool.pool_id
                    request.decode_cluster_id = next_decode_pool.cluster_id
                    if next_decode_pool.cluster_id != batch.cluster_id:
                        cross_cluster_by_pool[next_decode_pool.pool_id] = (
                            cross_cluster_by_pool.get(next_decode_pool.pool_id, 0) + 1
                        )
                    rerouted_by_pool.setdefault(next_decode_pool.pool_id, []).append(
                        request_id
                    )
                else:
                    request.stage = RequestStage.DECODE_ACTIVE

        released_request_ids = completed_request_ids | (
            continuing_request_ids
            if self.config.scheduler.allow_following_decode_cross_cluster
            else set()
        )
        self._remove_from_resident(pool, released_request_ids)

        for pool_id, request_ids in rerouted_by_pool.items():
            decode_pool = self.pools[pool_id]
            self._enqueue_requests(
                decode_pool,
                request_ids,
                event_type="decode_pool_received_requests",
                details={
                    "source_cluster_id": batch.cluster_id,
                    "request_count": len(request_ids),
                    "cross_cluster_count": cross_cluster_by_pool.get(pool_id, 0),
                },
            )

        if self.debug_logging_enabled:
            self._log(
                logging.DEBUG,
                "资源池完成Decode批次。",
                batch_id=batch.batch_id,
                cluster_id=batch.cluster_id,
                pool_id=batch.pool_id,
                details={
                    "batch_size": len(batch.request_ids),
                    "resident_count": pool.resident_count(),
                    "queue_length_after_dispatch": pool.waiting_queue_length(),
                },
            )

        if completed_request_ids and self.info_logging_enabled:
            self._log(
                logging.INFO,
                "资源池完成请求。",
                cluster_id=batch.cluster_id,
                pool_id=batch.pool_id,
                details={
                    "completed_requests": len(completed_request_ids),
                    "resident_count": pool.resident_count(),
                    "queue_length": pool.waiting_queue_length(),
                    "batch_id": batch.batch_id,
                },
            )
        return list(completed_request_ids)

    def _accept_new_request(self, request: Request) -> None:
        prefill_kind = self.scheduler._prefill_kind_for_request(request)
        prefill_pool = self.scheduler.select_prefill_pool(
            request, self.prefill_pools_by_kind[prefill_kind]
        )
        request.target_prefill_pool_id = prefill_pool.pool_id
        request.target_prefill_kind = prefill_kind
        request.stage = RequestStage.PREFILL_WAITING
        request.last_enqueue_time_ms = self.current_time_ms
        self._enqueue_requests(
            prefill_pool,
            [request.request_id],
            event_type="prefill_pool_received_requests",
            details={"request_count": 1},
            log_level=logging.CRITICAL,
        )
        self._dispatch_all_idle_capacity()

    def _dispatch_all_idle_capacity(self) -> None:
        dispatched = True
        while dispatched:
            dispatched = False
            for pool_id in list(self.dispatchable_pool_ids):
                pool = self.pools[pool_id]
                while pool.idle_instance_count() > 0:
                    if pool.kind == ResourceKind.DECODE:
                        admitted = self._admit_decode_waiting_requests(
                            pool, slots=pool.max_batch_size
                        )
                        if admitted:
                            dispatched = True
                    instance = pool.peek_idle_instance()
                    if instance is None:
                        break
                    batch = self._start_batch(pool, instance)
                    if batch is None:
                        break
                    acquired_instance = pool.acquire_idle_instance()
                    if acquired_instance is None:
                        break
                    self.batches[batch.batch_id] = batch
                    self.batch_instances[batch.batch_id] = acquired_instance
                    heapq.heappush(
                        self.batch_finish_heap,
                        (batch.finish_time_ms, batch.batch_id),
                    )
                    acquired_instance.current_batch_id = batch.batch_id
                    acquired_instance.busy_until_ms = batch.finish_time_ms
                    if pool.kind == ResourceKind.DECODE:
                        pool.mark_inflight_requests(batch.request_ids)
                    dispatched = True
                self._refresh_dispatchable_pool(pool)

    def _admit_decode_waiting_requests(
        self, pool: ResourcePool, slots: int
    ) -> list[int]:
        admitted: list[int] = []
        slots = min(slots, pool.available_decode_slots())
        if slots <= 0 or not pool.waiting_request_ids:
            return admitted

        self._update_pool_queue_depth_area(pool)
        admitted = pool.pop_waiting_request_ids(slots)

        for request_id in admitted:
            request = self.requests[request_id]
            wait_time_ms = self.current_time_ms - (
                request.last_enqueue_time_ms or self.current_time_ms
            )
            request.queue_time_decode_ms += wait_time_ms
            if request.decode_start_time_ms is None:
                request.decode_start_time_ms = self.current_time_ms
            request.stage = RequestStage.DECODE_ACTIVE
            pool.add_resident_request(request_id)

        return admitted

    def _start_batch(
        self, pool: ResourcePool, instance: ResourceInstance
    ) -> TaskBatch | None:
        if pool.kind == ResourceKind.DECODE:
            batch_request_ids = self._select_decode_batch_request_ids(pool)
        else:
            if not pool.waiting_request_ids:
                return None
            self._update_pool_queue_depth_area(pool)
            batch_request_ids = pool.pop_waiting_request_ids(pool.max_batch_size)

            for request_id in batch_request_ids:
                request = self.requests[request_id]
                wait_time_ms = self.current_time_ms - (
                    request.last_enqueue_time_ms or self.current_time_ms
                )
                request.queue_time_prefill_ms += wait_time_ms
                if request.prefill_start_time_ms is None:
                    request.prefill_start_time_ms = self.current_time_ms
                request.stage = RequestStage.PREFILL_RUNNING

        if not batch_request_ids:
            return None

        pool.total_dispatched_requests += len(batch_request_ids)
        finish_time_ms = self.current_time_ms + self._estimate_batch_duration_ms(
            pool.kind, batch_request_ids
        )
        batch = TaskBatch(
            batch_id=self.next_batch_id,
            kind=pool.kind,
            request_ids=batch_request_ids,
            start_time_ms=self.current_time_ms,
            finish_time_ms=finish_time_ms,
            cluster_id=pool.cluster_id,
            pool_id=pool.pool_id,
            token_step_count=1 if pool.kind == ResourceKind.DECODE else 0,
        )
        if pool.kind == ResourceKind.DECODE:
            if self.debug_logging_enabled:
                self._log(
                    logging.DEBUG,
                    "资源池开始执行批次。",
                    batch_id=batch.batch_id,
                    cluster_id=pool.cluster_id,
                    pool_id=pool.pool_id,
                    instance_id=instance.instance_id,
                    details={
                        "resource_kind": pool.kind.value,
                        "batch_size": len(batch_request_ids),
                        "resident_count": pool.resident_count(),
                        "queue_length_after_dispatch": pool.waiting_queue_length(),
                        "finish_time_ms": finish_time_ms,
                    },
                )
        elif self.info_logging_enabled:
            self._log(
                logging.INFO,
                "资源池开始执行批次。",
                batch_id=batch.batch_id,
                cluster_id=pool.cluster_id,
                pool_id=pool.pool_id,
                instance_id=instance.instance_id,
                details={
                    "resource_kind": pool.kind.value,
                    "batch_size": len(batch_request_ids),
                    "resident_count": pool.resident_count(),
                    "queue_length_after_dispatch": pool.waiting_queue_length(),
                    "finish_time_ms": finish_time_ms,
                },
            )
        self.next_batch_id += 1
        return batch

    def _select_decode_batch_request_ids(self, pool: ResourcePool) -> list[int]:
        if pool.runnable_decode_request_count() <= 0:
            return []
        return pool.pop_runnable_decode_request_ids(pool.max_batch_size)

    def _remove_from_resident(self, pool: ResourcePool, completed_ids: set[int]) -> None:
        if not completed_ids:
            return
        pool.remove_resident_requests(completed_ids)
        self._refresh_dispatchable_pool(pool)

    def _estimate_batch_duration_ms(
        self, kind: ResourceKind, request_ids: list[int]
    ) -> float:
        perf = self.config.performance
        batch_size = len(request_ids)
        if kind == ResourceKind.DECODE:
            # TODO: replace this linear placeholder with real profiling data.
            # The final model should consider context length, active sequences,
            # KV cache pressure, and cross-cluster communication cost.
            return perf.decode_base_step_ms + perf.decode_batch_size_slope_ms * batch_size

        total_prompt_tokens = sum(
            self.requests[request_id].prompt_tokens for request_id in request_ids
        )
        coeff = (
            perf.long_prefill_ms_per_token
            if kind == ResourceKind.LONG_PREFILL
            else perf.short_prefill_ms_per_token
        )
        return total_prompt_tokens * coeff

    def _drain_system(self) -> None:
        while (
            self.request_generator.has_pending_arrivals()
            or self.batches
            or self._has_pending_pool_work()
        ):
            next_arrival_time_ms = self.request_generator.next_arrival_time_ms()
            next_finish_time = self._next_batch_finish_time_before(next_arrival_time_ms)
            if next_finish_time is None and next_arrival_time_ms is not None:
                self._advance_until(next_arrival_time_ms)
                self.current_time_ms = next_arrival_time_ms
                self._accept_ready_requests()
                continue
            next_finish_time = self._next_batch_finish_time_before(float("inf"))
            if next_finish_time is None:
                self._dispatch_all_idle_capacity()
                next_finish_time = self._next_batch_finish_time_before(float("inf"))
                if next_finish_time is None:
                    break
            self.current_time_ms = next_finish_time
            finished_request_ids = self._finalize_batches_at_current_time()
            for request_id in finished_request_ids:
                self.request_generator.on_request_finished(self.requests[request_id])
            self._dispatch_all_idle_capacity()

    def _has_pending_pool_work(self) -> bool:
        return any(
            pool.waiting_request_ids or pool.resident_request_ids
            for pool in self.pools.values()
        )

    def resource_utilization(self) -> dict[str, float]:
        if not self.current_time_ms:
            return {}
        utilization: dict[str, float] = {}
        for pool in self.pools.values():
            for instance in pool.instances:
                utilization[instance.instance_id] = (
                    instance.total_busy_time_ms / self.current_time_ms
                )
        return utilization

    def queue_depths(self) -> dict[str, int]:
        return {pool.pool_id: pool.waiting_queue_length() for pool in self.pools.values()}

    def pool_breakdown(self) -> dict[str, dict[str, float | int | str]]:
        return self.metrics.pool_breakdown(self.pools, self.current_time_ms)

    def request_breakdown(self, limit: int | None = None) -> list[dict[str, object]]:
        return self.metrics.request_breakdown(limit=limit)

    def _enqueue_requests(
        self,
        pool: ResourcePool,
        request_ids: list[int],
        event_type: str,
        details: dict[str, object],
        log_level: int = logging.INFO,
    ) -> None:
        if not request_ids:
            return
        self._update_pool_queue_depth_area(pool)
        pool.extend_waiting_request_ids(request_ids)
        pool.total_enqueued_requests += len(request_ids)
        pool.max_queue_length_seen = max(
            pool.max_queue_length_seen, pool.waiting_queue_length()
        )
        self._refresh_dispatchable_pool(pool)
        if self.logger.isEnabledFor(log_level):
            self._log(
                log_level,
                self._enqueue_log_message(event_type, request_ids, pool.kind),
                cluster_id=pool.cluster_id,
                pool_id=pool.pool_id,
                details={
                    "event_type": event_type,
                    **details,
                    "queue_length": pool.waiting_queue_length(),
                },
            )

    def _enqueue_log_message(
        self,
        event_type: str,
        request_ids: list[int],
        pool_kind: ResourceKind,
    ) -> str:
        if event_type == "prefill_pool_received_requests":
            request_id = request_ids[0]
            return (
                f"请求#{request_id}"
                f"（{self._request_progress_text(request_id)}）"
                f" 已到达 {pool_kind.value} 资源池。"
            )
        if event_type == "decode_pool_received_requests":
            if len(request_ids) == 1:
                request_id = request_ids[0]
                return (
                    f"请求#{request_id}"
                    f"（{self._request_progress_text(request_id)}）"
                    " 已转发到 decode 资源池。"
                )
            progress_text = self._request_progress_text(max(request_ids))
            return (
                f"请求批次{request_ids}"
                f"（截至 {progress_text}）"
                " 已转发到 decode 资源池。"
            )
        return event_type

    def _request_progress_text(self, request_id: int) -> str:
        total = self.request_generator.total_request_count
        if total <= 0:
            return f"{request_id}/?"
        percentage = request_id / total * 100.0
        return f"{request_id}/{total}, {percentage:.1f}%"

    def _refresh_dispatchable_pool(self, pool: ResourcePool) -> None:
        if pool.idle_instance_count() <= 0:
            self.dispatchable_pool_ids.pop(pool.pool_id, None)
            return

        if pool.kind == ResourceKind.DECODE:
            has_pending_work = (
                bool(pool.waiting_request_ids) and pool.available_decode_slots() > 0
            ) or pool.runnable_decode_request_count() > 0
        else:
            has_pending_work = bool(pool.waiting_request_ids)

        if has_pending_work:
            self.dispatchable_pool_ids[pool.pool_id] = None
        else:
            self.dispatchable_pool_ids.pop(pool.pool_id, None)

    def _update_pool_queue_depth_area(self, pool: ResourcePool) -> None:
        elapsed = self.current_time_ms - pool.last_queue_depth_update_time_ms
        if elapsed > 0:
            pool.queue_depth_area_ms += pool.waiting_queue_length() * elapsed
            pool.last_queue_depth_update_time_ms = self.current_time_ms

    def _finalize_pool_queue_stats(self) -> None:
        for pool in self.pools.values():
            self._update_pool_queue_depth_area(pool)

    def _log(
        self,
        level: int,
        message: str,
        request_id: int | None = None,
        batch_id: int | None = None,
        cluster_id: str | None = None,
        pool_id: str | None = None,
        instance_id: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        if not self.logger.isEnabledFor(level):
            return
        suffix_parts: list[str] = []
        if request_id is not None:
            suffix_parts.append(f"request_id={request_id}")
        if batch_id is not None:
            suffix_parts.append(f"batch_id={batch_id}")
        if cluster_id is not None:
            suffix_parts.append(f"cluster_id={cluster_id}")
        if pool_id is not None:
            suffix_parts.append(f"pool_id={pool_id}")
        if instance_id is not None:
            suffix_parts.append(f"instance_id={instance_id}")
        if details:
            suffix_parts.append(", ".join(f"{key}={value}" for key, value in details.items()))
        final_message = message
        if suffix_parts:
            final_message = f"{message} {' | '.join(suffix_parts)}"
        self.logger.log(level, final_message, extra={"sim_time_ms": f"{self.current_time_ms:.3f}"})

    def _topology_summary(self) -> str:
        cluster_parts: list[str] = []
        for cluster in self.config.scenario.clusters:
            pool_parts = []
            for pool in cluster.pools:
                pool_parts.append(
                    f"{pool.pool_id}(kind={pool.kind.value},instances={pool.instance_count},max_batch={pool.max_batch_size})"
                )
            cluster_parts.append(f"{cluster.cluster_id}:[{';'.join(pool_parts)}]")
        return " | ".join(cluster_parts)
