from __future__ import annotations

from dataclasses import dataclass

from .base import PolicyContext
from .common import best_pool_by_cluster, choose_cluster_by_weight
from ..models import ResourcePool


@dataclass(frozen=True)
class DefaultRoutingPolicy:
    name: str = "p_default"

    def select_pool(self, context: PolicyContext) -> ResourcePool:
        if len(context.candidate_pools) == 1:
            return context.candidate_pools[0]

        best_pool_map = best_pool_by_cluster(context.candidate_pools)

        if len(best_pool_map) == 1:
            return next(iter(best_pool_map.values()))

        chosen_cluster_id = choose_cluster_by_weight(
            context,
            tuple(best_pool_map),
        )
        return best_pool_map[chosen_cluster_id]
