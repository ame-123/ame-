"""预订草稿合并、指代绑定和二次确认。"""

from __future__ import annotations

import os
import unittest
from uuid import uuid4

os.environ.setdefault("AGENT_COURSE_DISABLE_LLM", "1")

from api.schemas import ChatRequest
from context.builder import SESSION_MEMORIES, current_memory
from workflows.fields import (
    bind_package_from_tool_name,
    extract_booking_date,
    extract_booking_slots,
    is_package_anaphora,
    merge_booking_draft,
)


class BookingDraftTests(unittest.TestCase):
    def setUp(self) -> None:
        SESSION_MEMORIES.clear()

    def test_dotted_date_parses(self) -> None:
        self.assertEqual(extract_booking_date("日期是2026.6-10号"), "2026-06-10")
        self.assertIsNone(extract_booking_date("日期明天"))

    def test_merge_keeps_tokyo_then_date(self) -> None:
        first = extract_booking_slots("目的地东京，日期明天")
        draft = merge_booking_draft(None, first, last_product_name=None, user_message="目的地东京，日期明天")
        self.assertEqual(draft["destination"], "东京")
        self.assertIsNone(draft["date"])
        second = extract_booking_slots("日期是2026.6-10号")
        draft = merge_booking_draft(draft, second, last_product_name=None, user_message="日期是2026.6-10号")
        self.assertEqual(draft["destination"], "东京")
        self.assertEqual(draft["date"], "2026-06-10")
        self.assertEqual(draft["missing"], [])

    def test_anaphora_binds_only_last_product_name(self) -> None:
        self.assertEqual(bind_package_from_tool_name("东京五日机票酒店"), "东京五日机票酒店")
        self.assertIsNone(bind_package_from_tool_name("随便一个商品"))
        self.assertTrue(is_package_anaphora("就这个套餐，进入预定流程"))
        draft = merge_booking_draft(
            {"destination": "东京", "date": "2026-06-10", "sources": {}, "status": "collecting", "pax": 1},
            extract_booking_slots("就这个套餐，进入预定流程"),
            last_product_name="东京五日机票酒店",
            user_message="就这个套餐，进入预定流程",
        )
        self.assertEqual(draft["package_name"], "东京五日机票酒店")
        self.assertEqual(draft["sources"]["package_name"], "tool_fact")
        unbound = merge_booking_draft(
            {"destination": "东京", "date": "2026-06-10", "sources": {}, "status": "collecting", "pax": 1},
            extract_booking_slots("就这个套餐，进入预定流程"),
            last_product_name=None,
            user_message="就这个套餐，进入预定流程",
        )
        self.assertIsNone(unbound.get("package_name"))

    def test_agent_multi_turn_confirm_then_graph(self) -> None:
        from agents.customer_service_agent import Lesson41Agent

        agent = Lesson41Agent()
        session_id = f"test-{uuid4().hex[:8]}"
        runtime = {
            "current_order_id": "SO20260602103000009-a1000009",
            "currentUserOrders": [],
        }

        def chat(message: str):
            return agent.chat(
                ChatRequest(
                    session_id=session_id,
                    runtime_user_id="U1001",
                    runtime_nickname="测试",
                    runtime_member_level="gold",
                    runtime_risk_level="low",
                    user_message=message,
                    runtime_context=runtime,
                    reasoning_view="off",
                )
            )

        first = chat("目的地东京，日期明天")
        self.assertIn("出行日期", first.answer)
        self.assertIsNone((first.session_state.get("workflow") or {}).get("pending_action"))
        draft = current_memory(session_id, "U1001")["booking_draft"]
        self.assertEqual(draft["destination"], "东京")

        second = chat("日期是2026.6-10号")
        self.assertIn("请确认预订信息", second.answer)
        self.assertIn("2026-06-10", second.answer)
        self.assertEqual(second.session_state["memory"]["booking_draft"]["status"], "awaiting_confirm")

        third = chat("东京五日机票酒店还有货吗")
        self.assertTrue(
            any(call.tool_name == "search_products" for call in third.tool_calls) or "东京五日" in third.answer
        )

        fourth = chat("就这个套餐，进入预定流程")
        self.assertIn("请确认预订信息", fourth.answer)
        self.assertIn("东京五日", fourth.answer)
        self.assertNotEqual((fourth.session_state.get("workflow") or {}).get("pending_action"), "require_approval")

        fifth = chat("确认")
        workflow = fifth.session_state.get("workflow") or {}
        self.assertEqual(workflow.get("pending_action"), "require_approval")
        self.assertTrue(fifth.session_state.get("needs_human_approval"))

    def test_date_not_available_lists_options_then_accepts_new_date(self) -> None:
        os.environ["AGENT_COURSE_OFFLINE_FACTS"] = "1"
        from agents.customer_service_agent import Lesson41Agent

        agent = Lesson41Agent()
        session_id = f"test-{uuid4().hex[:8]}"
        runtime = {
            "current_order_id": "SO20260602103000009-a1000009",
            "currentUserOrders": [],
        }

        def chat(message: str):
            return agent.chat(
                ChatRequest(
                    session_id=session_id,
                    runtime_user_id="U1001",
                    runtime_nickname="测试",
                    runtime_member_level="gold",
                    runtime_risk_level="low",
                    user_message=message,
                    runtime_context=runtime,
                    reasoning_view="off",
                )
            )

        blocked = chat("帮我预订 2026-08-16 出发的东京五日机票酒店")
        self.assertIn("没有可订名额", blocked.answer)
        self.assertIn("2026-06-10", blocked.answer)
        self.assertIn("12", blocked.answer)
        self.assertIn("换一个出行日期", blocked.answer)
        self.assertNotEqual((blocked.session_state.get("workflow") or {}).get("pending_action"), "require_approval")
        draft = current_memory(session_id, "U1001")["booking_draft"]
        self.assertEqual(draft["destination"], "东京")
        self.assertIsNone(draft["date"])
        self.assertEqual(draft["status"], "collecting")

        follow = chat("6月10日")
        self.assertEqual(follow.session_state["memory"]["booking_draft"]["date"], "2026-06-10")
        self.assertIn("请确认预订信息", follow.answer)
        self.assertIn("2026-06-10", follow.answer)


if __name__ == "__main__":
    unittest.main()
