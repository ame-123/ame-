from __future__ import annotations

import sys
import unittest
from pathlib import Path

RAG_ROOT = Path(__file__).resolve().parents[1]
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))

from kb.index_cache import (
    build_knowledge_index,
    cache_key_for,
    cache_policy_for,
    get_cached_hits,
    remember_hits,
    reset_index_and_cache,
)
from kb.planning import is_realtime_business_query, pre_retrieval_plan


class RealtimePolicyTest(unittest.TestCase):
    def test_order_tracking_is_realtime(self) -> None:
        self.assertTrue(is_realtime_business_query("我的订单到哪了"))
        self.assertTrue(is_realtime_business_query("库存还有吗"))

    def test_stable_policy_is_not_realtime(self) -> None:
        self.assertFalse(is_realtime_business_query("签收后7天能无理由退货吗"))


class IndexCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_index_and_cache()

    def tearDown(self) -> None:
        reset_index_and_cache()

    def test_stable_scene_is_cacheable(self) -> None:
        plan = pre_retrieval_plan("签收后7天能无理由退货吗")
        policy = cache_policy_for(plan)
        self.assertTrue(policy["cacheable"])
        self.assertEqual(policy["scope"], "retrieval_hits_only")

    def test_realtime_query_is_not_cacheable(self) -> None:
        plan = pre_retrieval_plan("我的订单到哪了")
        policy = cache_policy_for(plan)
        self.assertFalse(policy["cacheable"])

    def test_cache_key_includes_index_version(self) -> None:
        plan = pre_retrieval_plan("签收后7天能无理由退货吗")
        first = build_knowledge_index(
            [{"source": "after_sale_policy.md", "chunk_id": 0, "title": "七天", "text": "无理由退货"}]
        )
        second = build_knowledge_index(
            [{"source": "after_sale_policy.md", "chunk_id": 0, "title": "七天", "text": "规则已更新"}]
        )
        self.assertNotEqual(first.version, second.version)
        self.assertNotEqual(cache_key_for(plan, first), cache_key_for(plan, second))

    def test_remembered_hits_are_returned_until_rebuild(self) -> None:
        plan = pre_retrieval_plan("签收后7天能无理由退货吗")
        index = build_knowledge_index(
            [{"source": "after_sale_policy.md", "chunk_id": 0, "title": "七天", "text": "无理由退货"}]
        )
        hits = [{"distance": 0.9, "entity": {"title": "七天", "source": "after_sale_policy.md"}}]
        debug = {"vector_titles": []}
        remember_hits(plan, index, hits, debug, collection="xiaozhe_docs")
        cached = get_cached_hits(plan, index, collection="xiaozhe_docs")
        self.assertIsNotNone(cached)
        self.assertEqual(cached[0][0]["entity"]["title"], "七天")

        reset_index_and_cache("xiaozhe_docs")
        self.assertIsNone(get_cached_hits(plan, index, collection="xiaozhe_docs"))


if __name__ == "__main__":
    unittest.main()
