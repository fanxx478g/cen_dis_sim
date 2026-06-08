from .default_policy import DefaultRoutingPolicy
from .registry import get_policy, register_policy, registered_policy_names

register_policy(DefaultRoutingPolicy())

__all__ = [
    "DefaultRoutingPolicy",
    "get_policy",
    "register_policy",
    "registered_policy_names",
]
