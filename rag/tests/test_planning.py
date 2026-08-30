from __future__ import annotations

import sys
import unittest
from pathlib import Path

RAG_ROOT = Path(__file__).resolve().parents[1]
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))

from kb.hybrid import merge_hybrid_hits
from kb.planning import detect_scene, pre_retrieval_plan, topic_of_source


class PlanningTest(unittest.TestCase):
    def test_damaged_box_routes_to_after_sale(self) -> None:
        plan = pre_retrieval_plan("盒子被压了，赠品少一根")
        self.assertEqual(plan.scene, "after_sale")
        self.assertEqual(plan.allowed_topics, ["after_sale"])
        self.assertIn("包装盒", plan.rewritten_query)
        self.assertIn("配件", plan.keyword_terms)
        self.assertIn("赠品", plan.keyword_terms)

    def test_headset_promotion_routes_to_promotion(self) -> None:
        plan = pre_retrieval_plan("那个耳麦活动能叠券吗")
        self.assertEqual(plan.scene, "promotion")
        self.assertEqual(plan.allowed_topics, ["promotion"])
        self.assertIn("耳机", plan.rewritten_query)
        self.assertIn("春季音频节", plan.added_terms)

    def test_unknown_keeps_all_topics(self) -> None:
        plan = pre_retrieval_plan("知识库支持哪些文件")
        self.assertEqual(plan.scene, "unknown")
        self.assertEqual(set(plan.allowed_topics), {"promotion", "product", "after_sale", "shipping"})

    def test_topic_comes_from_source_filename(self) -> None:
        self.assertEqual(topic_of_source("after_sale_policy.md"), "after_sale")
        self.assertEqual(topic_of_source("promotion_policy.md"), "promotion")
        self.assertIsNone(topic_of_source("print.pdf"))

    def test_detect_scene_shipping(self) -> None:
        self.assertEqual(detect_scene("快递到哪了"), "shipping")


class HybridMergeTest(unittest.TestCase):
    def test_merge_keeps_both_sources_and_prefers_stronger_score(self) -> None:
        vector = [
            {
                "id": 1,
                "distance": 0.42,
                "vector_score": 0.42,
                "keyword_score": 0.0,
                "hit_sources": ["vector"],
                "matched_keywords": [],
                "entity": {"source": "after_sale_policy.md", "title": "七天", "text": "无理由退货", "chunk_id": 0},
            }
        ]
        keyword = [
            {
                "id": 1,
                "distance": 0.5,
                "vector_score": 0.0,
                "keyword_score": 0.5,
                "hit_sources": ["keyword"],
                "matched_keywords": ["无理由"],
                "entity": {"source": "after_sale_policy.md", "title": "七天", "text": "无理由退货", "chunk_id": 0},
            }
        ]
        merged = merge_hybrid_hits(vector, keyword, scene="after_sale")
        self.assertEqual(len(merged), 1)
        hit = merged[0]
        self.assertEqual(set(hit["hit_sources"]), {"vector", "keyword"})
        self.assertIn("无理由", hit["matched_keywords"])
        self.assertGreaterEqual(hit["distance"], 0.5)


if __name__ == "__main__":
    unittest.main()
