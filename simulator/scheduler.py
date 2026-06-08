from __future__ import annotations

from dataclasses import dataclass, field
import random

from .config import ResourceKind, SimulationConfig
from .models import Request, ResourcePool
from .policies import get_policy
from .policies.base import PolicyContext, RoutingPolicy


@dataclass
class Scheduler:
    config: SimulationConfig
    rng: random.Random = field(default_factory=random.Random)
    policy: RoutingPolicy = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.policy = get_policy(self.config.scheduler.policy_name)

    def select_prefill_pool(
        self, request: Request, candidate_pools: list[ResourcePool]
    ) -> ResourcePool:
        kind = self._prefill_kind_for_request(request)
        eligible = [pool for pool in candidate_pools if pool.kind == kind]
        return self._select_pool(
            kind,
            eligible,
            request,
        )

    def select_decode_pool(
        self,
        request: Request,
        candidate_pools: list[ResourcePool],
        *,
        is_first_decode: bool,
    ) -> ResourcePool:
        eligible = candidate_pools
        eligible = self._eligible_decode_pools(
            request,
            eligible,
            is_first_decode=is_first_decode,
        )
        return self._select_pool(
            ResourceKind.DECODE,
            eligible,
            request,
        )

    def _eligible_decode_pools(
        self,
        request: Request,
        pools: list[ResourcePool],
        *,
        is_first_decode: bool,
    ) -> list[ResourcePool]:
        if not pools:
            return pools

        if is_first_decode:
            if request.prefill_cluster_id is None:
                return pools
            if self.config.scheduler.allow_first_decode_cross_cluster:
                return pools
            same_cluster = [
                pool
                for pool in pools
                if pool.cluster_id == request.prefill_cluster_id
            ]
            return same_cluster or pools

        if not self.config.scheduler.allow_following_decode_cross_cluster:
            if request.target_decode_pool_id is None:
                return pools
            same_pool = [
                pool
                for pool in pools
                if pool.pool_id == request.target_decode_pool_id
            ]
            return same_pool or pools
        return pools

    def _prefill_kind_for_request(self, request: Request) -> ResourceKind:
        if request.prompt_tokens > self.config.scheduler.prompt_len_threshold:
            return ResourceKind.LONG_PREFILL
        return ResourceKind.SHORT_PREFILL

    def _select_pool(
        self,
        kind: ResourceKind,
        pools: list[ResourcePool],
        request: Request,
    ) -> ResourcePool:
        if not pools:
            raise ValueError("No eligible resource pool available for the request.")
        if len(pools) == 1:
            return pools[0]
        context = PolicyContext(
            config=self.config,
            resource_kind=kind,
            request=request,
            candidate_pools=pools,
            rng=self.rng,
        )
        return self.policy.select_pool(context)
