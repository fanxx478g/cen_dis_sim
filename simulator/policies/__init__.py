from .dis_first_policy import DistributedFirstRoutingPolicy
from .default_policy import DefaultRoutingPolicy
from .registry import get_policy, register_policy, registered_policy_names

register_policy(DefaultRoutingPolicy())
register_policy(DistributedFirstRoutingPolicy())

__all__ = [
    "DistributedFirstRoutingPolicy",
    "DefaultRoutingPolicy",
    "get_policy",
    "register_policy",
    "registered_policy_names",
]
