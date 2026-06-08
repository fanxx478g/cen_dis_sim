from __future__ import annotations

from dataclasses import dataclass

from .base import PolicyContext
from ..models import ResourcePool


@dataclass(frozen=True)
class DefaultRoutingPolicy:
    name: str = "p_default"

    def select_pool(self, context: PolicyContext) -> ResourcePool:
        if len(context.candidate_pools) == 1:
            return context.candidate_pools[0]

        best_pool_by_cluster: dict[str, ResourcePool] = {}
        best_score_by_cluster: dict[str, tuple[int, bool, str]] = {}
        for pool in context.candidate_pools:
            score = (
                pool.queue_length(),
                pool.idle_instance_count() == 0,
                pool.pool_id,
            )
            cluster_id = pool.cluster_id
            previous_score = best_score_by_cluster.get(cluster_id)
            if previous_score is None or score < previous_score:
                best_score_by_cluster[cluster_id] = score
                best_pool_by_cluster[cluster_id] = pool

        if len(best_pool_by_cluster) == 1:
            return next(iter(best_pool_by_cluster.values()))

        chosen_cluster_id = self._choose_cluster_by_weight(
            context,
            tuple(best_pool_by_cluster),
        )
        return best_pool_by_cluster[chosen_cluster_id]

    def _choose_cluster_by_weight(
        self,
        context: PolicyContext,
        cluster_ids: tuple[str, ...],
    ) -> str:
        raw_policy_config = context.config.scheduler.policy_config
        raw_weights = raw_policy_config.get("cluster_routing_weights", {})
        if not isinstance(raw_weights, dict):
            raise ValueError("policy_config.cluster_routing_weights must be a dict.")

        if len(cluster_ids) == 1:
            return cluster_ids[0]

        weighted_clusters: list[tuple[str, float]] = []
        for cluster_id in cluster_ids:
            cluster_weights = raw_weights.get(cluster_id, {})
            if not isinstance(cluster_weights, dict):
                raise ValueError(
                    f"policy_config.cluster_routing_weights['{cluster_id}'] must be a dict."
                )
            weight = float(cluster_weights.get(context.resource_kind.value, 1.0))
            if weight < 0:
                raise ValueError(
                    "Cluster weight must be non-negative: "
                    f"cluster={cluster_id}, kind={context.resource_kind.value}"
                )
            if weight > 0:
                weighted_clusters.append((cluster_id, weight))

        if not weighted_clusters:
            raise ValueError(
                "No positive routing weight configured for resource kind "
                f"{context.resource_kind.value}."
            )

        total_weight = sum(weight for _, weight in weighted_clusters)
        threshold = context.rng.random() * total_weight
        cumulative = 0.0
        for cluster_id, weight in weighted_clusters:
            cumulative += weight
            if threshold <= cumulative:
                return cluster_id
        return weighted_clusters[-1][0]
