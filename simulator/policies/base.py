from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Protocol

from ..config import ResourceKind, SimulationConfig
from ..models import Request, ResourcePool


@dataclass(frozen=True)
class PolicyContext:
    config: SimulationConfig
    resource_kind: ResourceKind
    request: Request
    candidate_pools: list[ResourcePool]
    rng: random.Random


class RoutingPolicy(Protocol):
    name: str

    def select_pool(self, context: PolicyContext) -> ResourcePool:
        ...
