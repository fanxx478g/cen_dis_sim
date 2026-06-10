from __future__ import annotations

from dataclasses import dataclass

from ..config import ResourceKind
from ..models import ResourcePool
from .base import PolicyContext
from .common import best_pool_by_cluster, choose_cluster_by_weight
from .default_policy import DefaultRoutingPolicy


@dataclass(frozen=True)
class DistributedFirstRoutingPolicy:
    name: str = "dis_first"

    def select_pool(self, context: PolicyContext) -> ResourcePool:
        if context.resource_kind != ResourceKind.SHORT_PREFILL:
            return DefaultRoutingPolicy().select_pool(context)

        central_cluster_id = context.config.scenario.central_cluster_id
        best_pool_map = best_pool_by_cluster(context.candidate_pools)
        if not central_cluster_id or central_cluster_id not in best_pool_map:
            central_pool = None
            distributed_cluster_ids = tuple(best_pool_map)
        else:
            central_pool = best_pool_map.get(central_cluster_id)
            distributed_cluster_ids = tuple(
                cluster_id
                for cluster_id in best_pool_map
                if cluster_id != central_cluster_id
            )

        if not distributed_cluster_ids:
            if central_pool is None:
                raise ValueError("No distributed or central short prefill pool is available.")
            return central_pool

        chosen_distributed_cluster_id = choose_cluster_by_weight(
            context,
            distributed_cluster_ids,
            weight_config_key="distributed_cluster_routing_weights",
        )
        distributed_pool = best_pool_map[chosen_distributed_cluster_id]

        if self._is_queue_light(context, distributed_pool):
            return distributed_pool
        if central_pool is None:
            return distributed_pool
        if self._is_queue_light(context, central_pool):
            return central_pool
        return distributed_pool

    def _is_queue_light(self, context: PolicyContext, pool: ResourcePool) -> bool:
        threshold = self._queue_threshold(context)
        return (
            pool.idle_instance_count() > 0
            or pool.waiting_queue_length() <= threshold
        )

    def _queue_threshold(self, context: PolicyContext) -> int:
        raw_threshold = context.config.scheduler.policy_config.get(
            "short_prefill_queue_threshold",
            0,
        )
        threshold = int(raw_threshold)
        if threshold < 0:
            raise ValueError("policy_config.short_prefill_queue_threshold must be >= 0.")
        return threshold
