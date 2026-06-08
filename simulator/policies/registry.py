from __future__ import annotations

from .base import RoutingPolicy


_POLICY_REGISTRY: dict[str, RoutingPolicy] = {}


def register_policy(policy: RoutingPolicy) -> None:
    _POLICY_REGISTRY[policy.name] = policy


def get_policy(policy_name: str) -> RoutingPolicy:
    try:
        return _POLICY_REGISTRY[policy_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported scheduler policy: {policy_name}") from exc


def registered_policy_names() -> tuple[str, ...]:
    return tuple(sorted(_POLICY_REGISTRY))
