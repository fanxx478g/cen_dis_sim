from __future__ import annotations

from ..models import ResourcePool
from .base import PolicyContext


def best_pool_by_cluster(
    candidate_pools: list[ResourcePool],
) -> dict[str, ResourcePool]:
    best_pool_map: dict[str, ResourcePool] = {}
    best_score_map: dict[str, tuple[int, bool, str]] = {}
    for pool in candidate_pools:
        score = (
            pool.queue_length(),
            pool.idle_instance_count() == 0,
            pool.pool_id,
        )
        cluster_id = pool.cluster_id
        previous_score = best_score_map.get(cluster_id)
        if previous_score is None or score < previous_score:
            best_score_map[cluster_id] = score
            best_pool_map[cluster_id] = pool
    return best_pool_map


def choose_cluster_by_weight(
    context: PolicyContext,
    cluster_ids: tuple[str, ...],
    *,
    weight_config_key: str = "cluster_routing_weights",
) -> str:
    raw_policy_config = context.config.scheduler.policy_config
    raw_weights = raw_policy_config.get(weight_config_key, {})
    if not isinstance(raw_weights, dict):
        raise ValueError(f"policy_config.{weight_config_key} must be a dict.")

    if len(cluster_ids) == 1:
        return cluster_ids[0]

    weighted_clusters: list[tuple[str, float]] = []
    for cluster_id in cluster_ids:
        cluster_weights = raw_weights.get(cluster_id, {})
        if not isinstance(cluster_weights, dict):
            raise ValueError(
                f"policy_config.{weight_config_key}['{cluster_id}'] must be a dict."
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
