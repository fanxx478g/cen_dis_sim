from __future__ import annotations

import unittest

from simulator import ResourceKind, cluster, pool


class BasicConfigTest(unittest.TestCase):
    """Verify the lightweight topology helpers stay ergonomic and predictable."""

    def test_pool_helper_builds_pool_config(self) -> None:
        config = pool("edge-decode", ResourceKind.DECODE, 3, 16)

        self.assertEqual(config.pool_id, "edge-decode")
        self.assertEqual(config.kind, ResourceKind.DECODE)
        self.assertEqual(config.instance_count, 3)
        self.assertEqual(config.max_batch_size, 16)

    def test_cluster_helper_accepts_pool_list(self) -> None:
        pools = [
            pool("central-long-prefill", ResourceKind.LONG_PREFILL, 2, 4),
            pool("central-short-prefill", ResourceKind.SHORT_PREFILL, 4, 8),
        ]

        config = cluster("central", pools=pools)

        self.assertEqual(config.cluster_id, "central")
        self.assertEqual(len(config.pools), 2)
        self.assertEqual(config.pools[0].pool_id, "central-long-prefill")
        self.assertEqual(config.pools[1].pool_id, "central-short-prefill")


if __name__ == "__main__":
    unittest.main()
