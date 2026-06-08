from __future__ import annotations

from collections import deque
import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque

from .config import ResourceKind


class RequestStage(str, Enum):
    NEW = "new"
    PREFILL_WAITING = "prefill_waiting"
    PREFILL_RUNNING = "prefill_running"
    DECODE_WAITING = "decode_waiting"
    DECODE_ACTIVE = "decode_active"
    FINISHED = "finished"


@dataclass
class Request:
    request_id: int
    arrival_time_ms: float
    prompt_tokens: int
    output_tokens: int
    session_id: int | None = None
    turn_index: int = 1
    total_turns: int = 1
    initial_prompt_bucket: str | None = None
    history_tokens: int = 0
    new_prompt_tokens: int = 0
    stage: RequestStage = RequestStage.NEW
    target_prefill_pool_id: str | None = None
    target_decode_pool_id: str | None = None
    prefill_cluster_id: str | None = None
    decode_cluster_id: str | None = None
    generated_tokens: int = 0
    prefill_start_time_ms: float | None = None
    prefill_end_time_ms: float | None = None
    first_token_time_ms: float | None = None
    decode_start_time_ms: float | None = None
    finish_time_ms: float | None = None
    queue_time_prefill_ms: float = 0.0
    queue_time_decode_ms: float = 0.0
    last_enqueue_time_ms: float | None = None
    decode_step_count: int = 0
    crossed_cluster_for_decode: bool = False

    @property
    def prefill_first_token_latency_ms(self) -> float | None:
        if self.first_token_time_ms is None:
            return None
        return self.first_token_time_ms - self.arrival_time_ms

    @property
    def request_tpot_ms(self) -> float | None:
        if self.first_token_time_ms is None or self.finish_time_ms is None:
            return None
        return (self.finish_time_ms - self.first_token_time_ms) / max(
            self.output_tokens - 1, 1
        )

    @property
    def end_to_end_ms(self) -> float | None:
        if self.finish_time_ms is None:
            return None
        return self.finish_time_ms - self.arrival_time_ms


@dataclass
class TaskBatch:
    batch_id: int
    kind: ResourceKind
    request_ids: list[int]
    start_time_ms: float
    finish_time_ms: float
    cluster_id: str
    pool_id: str
    token_step_count: int = 0


@dataclass
class ResourceInstance:
    instance_id: str
    cluster_id: str
    pool_id: str
    kind: ResourceKind
    busy_until_ms: float = 0.0
    current_batch_id: int | None = None
    total_busy_time_ms: float = 0.0

    @property
    def is_idle(self) -> bool:
        return self.current_batch_id is None


@dataclass
class ResourcePool:
    pool_id: str
    cluster_id: str
    kind: ResourceKind
    max_batch_size: int
    instances: list[ResourceInstance]
    waiting_request_ids: Deque[int] = field(default_factory=deque)
    resident_request_ids: dict[int, None] = field(default_factory=dict)
    inflight_request_ids: set[int] = field(default_factory=set)
    max_queue_length_seen: int = 0
    queue_depth_area_ms: float = 0.0
    last_queue_depth_update_time_ms: float = 0.0
    total_enqueued_requests: int = 0
    total_dispatched_requests: int = 0
    _instance_by_id: dict[str, ResourceInstance] = field(init=False, repr=False)
    _idle_instance_ids: Deque[str] = field(init=False, repr=False)
    _idle_instance_id_set: set[str] = field(init=False, repr=False)
    _decode_capacity_value: int = field(init=False, repr=False)
    _waiting_request_count: int = field(init=False, repr=False, default=0)
    _resident_request_count: int = field(init=False, repr=False, default=0)
    _idle_instance_count_value: int = field(init=False, repr=False, default=0)
    _resident_sequence_by_request_id: dict[int, int] = field(
        init=False, repr=False, default_factory=dict
    )
    _next_resident_sequence: int = field(init=False, repr=False, default=0)
    _runnable_decode_heap: list[tuple[int, int]] = field(
        init=False, repr=False, default_factory=list
    )
    _runnable_decode_request_ids: set[int] = field(
        init=False, repr=False, default_factory=set
    )

    def __post_init__(self) -> None:
        self._instance_by_id = {
            instance.instance_id: instance for instance in self.instances
        }
        self._idle_instance_ids = deque(
            instance.instance_id for instance in self.instances
        )
        self._idle_instance_id_set = set(self._idle_instance_ids)
        self._decode_capacity_value = len(self.instances) * self.max_batch_size
        self._waiting_request_count = len(self.waiting_request_ids)
        self._resident_request_count = len(self.resident_request_ids)
        self._idle_instance_count_value = len(self._idle_instance_ids)

    def idle_instances(self) -> list[ResourceInstance]:
        return [
            self._instance_by_id[instance_id] for instance_id in self._idle_instance_ids
        ]

    def idle_instance_count(self) -> int:
        return self._idle_instance_count_value

    def peek_idle_instance(self) -> ResourceInstance | None:
        if not self._idle_instance_ids:
            return None
        return self._instance_by_id[self._idle_instance_ids[0]]

    def acquire_idle_instance(self) -> ResourceInstance | None:
        if not self._idle_instance_ids:
            return None
        instance_id = self._idle_instance_ids.popleft()
        self._idle_instance_id_set.remove(instance_id)
        self._idle_instance_count_value -= 1
        return self._instance_by_id[instance_id]

    def release_instance(self, instance_id: str) -> None:
        if instance_id not in self._instance_by_id:
            return
        if instance_id in self._idle_instance_id_set:
            return
        self._idle_instance_ids.append(instance_id)
        self._idle_instance_id_set.add(instance_id)
        self._idle_instance_count_value += 1

    def queue_length(self) -> int:
        return self._waiting_request_count

    def waiting_queue_length(self) -> int:
        return self._waiting_request_count

    def resident_count(self) -> int:
        return self._resident_request_count

    def add_resident_request(self, request_id: int) -> None:
        if request_id in self.resident_request_ids:
            return
        self.resident_request_ids[request_id] = None
        self._resident_request_count += 1
        resident_sequence = self._next_resident_sequence
        self._next_resident_sequence += 1
        self._resident_sequence_by_request_id[request_id] = resident_sequence
        if request_id not in self.inflight_request_ids:
            self._add_runnable_decode_request(request_id)

    def remove_resident_requests(self, request_ids: set[int]) -> None:
        for request_id in request_ids:
            if request_id not in self.resident_request_ids:
                continue
            self.resident_request_ids.pop(request_id, None)
            self._resident_sequence_by_request_id.pop(request_id, None)
            self._resident_request_count -= 1
            self.inflight_request_ids.discard(request_id)
            self._runnable_decode_request_ids.discard(request_id)

    def resident_request_iter(self):
        return self.resident_request_ids.keys()

    def decode_capacity(self) -> int:
        return self._decode_capacity_value

    def available_decode_slots(self) -> int:
        return max(0, self._decode_capacity_value - self._resident_request_count)

    def runnable_decode_request_count(self) -> int:
        return len(self._runnable_decode_request_ids)

    def mark_inflight_requests(self, request_ids: list[int]) -> None:
        for request_id in request_ids:
            if request_id in self.inflight_request_ids:
                continue
            self.inflight_request_ids.add(request_id)
            self._runnable_decode_request_ids.discard(request_id)

    def clear_inflight_requests(self, request_ids: list[int]) -> None:
        for request_id in request_ids:
            if request_id not in self.inflight_request_ids:
                continue
            self.inflight_request_ids.remove(request_id)
            if request_id in self.resident_request_ids:
                self._add_runnable_decode_request(request_id)

    def extend_waiting_request_ids(self, request_ids: list[int]) -> None:
        if not request_ids:
            return
        self.waiting_request_ids.extend(request_ids)
        self._waiting_request_count += len(request_ids)

    def pop_waiting_request_ids(self, count: int) -> list[int]:
        request_ids: list[int] = []
        for _ in range(count):
            if not self.waiting_request_ids:
                break
            request_ids.append(self.waiting_request_ids.popleft())
        self._waiting_request_count -= len(request_ids)
        return request_ids

    def pop_runnable_decode_request_ids(self, count: int) -> list[int]:
        request_ids: list[int] = []
        while self._runnable_decode_heap and len(request_ids) < count:
            _, request_id = heapq.heappop(self._runnable_decode_heap)
            if request_id not in self._runnable_decode_request_ids:
                continue
            if request_id not in self.resident_request_ids:
                self._runnable_decode_request_ids.discard(request_id)
                continue
            if request_id in self.inflight_request_ids:
                self._runnable_decode_request_ids.discard(request_id)
                continue
            self._runnable_decode_request_ids.remove(request_id)
            request_ids.append(request_id)
        return request_ids

    def _add_runnable_decode_request(self, request_id: int) -> None:
        if request_id in self._runnable_decode_request_ids:
            return
        resident_sequence = self._resident_sequence_by_request_id.get(request_id)
        if resident_sequence is None:
            return
        self._runnable_decode_request_ids.add(request_id)
        heapq.heappush(self._runnable_decode_heap, (resident_sequence, request_id))
