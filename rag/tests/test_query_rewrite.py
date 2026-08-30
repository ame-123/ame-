from __future__ import annotations

import sys
import unittest
from pathlib import Path

RAG_ROOT = Path(__file__).resolve().parents[1]
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))

from kb.query_rewrite import classify_intent, rewrite_retrieval_query


class QueryRewriteTest(unittest.TestCase):
    def test_keeps_original_query_and_rewrites_headset_promotion(self) -> None:
        original = "那个耳麦活动能叠券吗"
        result = rewrite_retrieval_query(original)

        self.assertEqual(result.original_query, original)
        self.assertEqual(result.intent, "promotion_consult")
        self.assertTrue(result.applied)
        self.assertIn("耳机", result.rewritten_query)
        self.assertNotIn("那个", result.rewritten_query)
        self.assertIn("2026", result.rewritten_query)
        self.assertIn("春季音频节", result.rewritten_query)
        self.assertIn("会员价", result.rewritten_query)
        self.assertIn("叠加", result.rewritten_query)

    def test_gold_member_adds_level_term(self) -> None:
        result = rewrite_retrieval_query("耳机活动怎么算会员价", member_level="gold")
        self.assertIn("金卡", result.added_terms)

    def test_refund_intent_adds_after_sale_terms(self) -> None:
        original = "签收后7天能无理由退货吗"
        result = rewrite_retrieval_query(original)

        self.assertEqual(result.original_query, original)
        self.assertEqual(result.intent, "refund_request")
        self.assertTrue(result.applied)
        for term in ["售后规则", "签收时间", "退货条件", "凭证"]:
            self.assertIn(term, result.added_terms)

    def test_unknown_query_stays_close_to_original(self) -> None:
        original = "知识库支持哪些文件"
        result = rewrite_retrieval_query(original)
        self.assertEqual(result.intent, "unknown")
        self.assertEqual(result.rewritten_query, original.lower())
        self.assertFalse(result.added_terms)

    def test_classify_intent_uses_normalized_text(self) -> None:
        self.assertEqual(classify_intent("耳麦促销"), "promotion_consult")


if __name__ == "__main__":
    unittest.main()
