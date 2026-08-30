# 旅行客服 Agent 全盘 Review

口径：这是 **旅行客服 Agent**（`travel-cs-agent`），覆盖行程查询、套餐事实、政策 RAG、未出行退票 HITL、预订槽位与二次确认。不再按电商课 / 第 41 课叙事。

HTTP 字段名（`orderNo`、`fulfillmentStatus`）和公开 intent 枚举（`refund_request`、`order_query`）刻意未改，避免拆 mock 契约和评测。

---

## 1. 项目定位

一次 `/chat` 不是「模型随便调工具」，而是受控编排：

1. 规则粗分意图 + 分类模型 + 硬边界覆盖  
2. Context Builder：选行程号、压缩历史、标可信度、裁冲突；顺带合并 `booking_draft`  
3. RoutePlan：工具 / RAG / 工作流白名单  
4. 执行分流：身份 / 安全 / 降级短路；FAQ 与促销走 RAG；查单、退票进度、搜套餐走只读工具；未出行退票与预订走 LangGraph + HITL  
5. 工具与 RAG 进最终模型前做 `source_guard` 污染检查  
6. 最终模型只写话术；澄清、低置信、缓存、降级、身份不调最终模型  
7. `update_memory` → Hooks → 成本摘要 → 公开 Trace  

高风险退票 / 预订不用 FAQ 缓存冒充结果。

---

## 2. 一次请求怎么走

```
ChatRequest
  → 意图（规则兜底 + 模型路由 + 硬边界）
  → Context Builder（行程号信任序 + 历史压缩 + booking_draft）
  → RoutePlan（白名单）
  → 短路：身份 / 注入 / 降级 / 低置信
  → 只读工具 或 RAG 或 LangGraph(+HITL)
  → source_guard
  → 最终话术模型（可选）
  → 记忆 / Hooks / Cost / 公开 Trace
```

**行程号信任序**：用户本句 explicit 单号 > 页面 Runtime > Session Memory。VIP 自称不能覆盖 `runtime_member_level`。没有长期记忆，只有 session 短期记忆。

**预订**：多轮槽位走 `booking_draft`；槽位齐但本句不是一次说全时，先二次确认再进图。一句说全仍直接进图，兼容评测。

---

## 3. 答辩时真正能讲的优点

| 点 | 为什么站得住 |
| --- | --- |
| 路由与硬边界 | 模型 intent 可被「退票 / 预订 / 注入 / 超时」覆盖，避免高风险走错 FAQ |
| 工具白名单 | RoutePlan 决定能调什么；LangChain `create_agent` 只能在白名单里选只读工具 |
| HITL | 未出行退票三节点直线图；预订图可绕行但不能无人批下单 |
| 二次确认 | 话术记住不等于后端槽位齐；确认不能替代可订名额和主管审批 |
| source_guard | 工具摘要和 RAG 片段都当外部数据，注入指令进不了系统规则 |
| 分层事实 | 套餐实时价 / 名额走 Tool；叠加规则走 RAG；结算页为准，不把通用满减套到当前套餐 |
| 公开 Trace | 只记执行摘要，不暴露 hidden CoT、提示词和密钥 |
| 评测 | `cases.yml` 卡回答信号、工具、引用、Trace、session_state、禁语 |

这些比「我用了 LangChain / LangGraph」更像工程判断。

---

## 4. 这次注解改写

面向用户的话术、Prompt、模块 docstring、评测期望，已从电商课口径改成旅行客服：

- 行程 / 值机 / 退票 / 套餐 / 可订名额 / 套餐页与结算页  
- Prompt：`route_contract`、`prompt_security`、`tool_usage`、`rag_answering`、角色提示  
- MCP 目录 URI 改为 `resource://travel/...`  
- Trace `agent_mode` 改为 `travel-cs-agent`  
- 评测对齐：`以套餐页和结算页为准`、`rag.rerank_mode=travel_lightweight`、`可订名额`  

未改：`backend/chat_logs/`（历史会话）、HTTP JSON 字段名、intent 枚举字符串。

---

## 5. 刻意保留的口径债

面试时如果被问「为什么还叫 order」，直接说：**内部契约未改，对外话术已改。**

| 保留项 | 说明 |
| --- | --- |
| 类名 `Lesson41Agent` | 历史标识；对外 `agent_version` 已是 `travel-cs-agent` |
| intent：`order_query` / `refund_request` / `return_request` / `product_query` | 改枚举要动 RoutePlan、cases、Trace |
| 工具名 `get_order_logistics` / `search_products` | 与 mock 路由和评测 `expected_tools` 绑定 |
| HTTP：`orderNo`、`fulfillmentStatus`、`logisticsStatus` | mock 契约；进图前映射成未出行 / 值机 / 出行中 / 已结束 |
| 模块名 `ecommerce_client`、环境变量 `ECOMMERCE_BASE_URL` | 接入层文件名，不是业务叙事 |
| policy_id `promotion_618_stack_rule` | 评测 citation 标识；正文已是出行早鸟规则 |
| 耳机 SKU | 离线种子回退，主路径是东京 / 京都 / 大阪套餐 |
| 评测用户话里的「退货」 | 映射到已出行退改，不进未出行退票图 |

---

## 6. 结构与风险

**编排层偏长。** `customer_service_agent.py` 把路由、工具、RAG、两张图、确认、安全、Trace 写在一个 `chat()` 里，读得懂但难单测单条分支。优点是主链路一眼能指；短板是后续加意图会继续膨胀。

**MCP 是目录不是协议。** `mcp_catalog` 只标注能力来源。答辩时不要说「接了 MCP 服务器」，说「用 MCP 风格目录约束工具来源，执行仍走 Tool / Hooks / Workflow」。

**不少文件有复制残留 import。** `schemas.py`、`planning.py`、`evals/runner.py`、`cost/governance.py` 等仍带着 FastAPI / httpx / yaml。不影响运行，显得不干净。

**`INTENT_FROM_COURSE` 未使用。** 只是内部别名表，主链路仍走公开 intent。

**规则路由仍认「订单 / 物流 / 耳机」。** 这是为了兜住旧说法和评测输入，不是业务主叙事。

**无长期记忆。** 跨 session 不记得偏好；预订草稿只活在当前 session。

**确定性话术 vs 模型话术。** `AGENT_COURSE_DISABLE_LLM=1` 时评测吃的是后端拼装句；开模型后措辞会漂，所以 cases 用短信号而不是整句比对。这是对的，但演示时要说明。

---

## 7. 建议的下一刀（按收益）

1. **类名** `Lesson41Agent` → `TravelCustomerServiceAgent`（纯标识，契约不变）  
2. **清无用 import**，文件头看起来像项目而不是课设拷贝  
3. **耳机 SKU**：若确认评测不再搜耳机，可以从 mock 拿掉  
4. **不要为了好看去改 `orderNo`**，除非同时改 mock、client、冻结字段和全部 cases  
5. **路由小模型**：本地 Qwen2.5-3B LoRA 已训完（测试集 97.58%），现网仍打硅基流动 Qwen3.5-4B。接入见 `docs/路由小模型微调与接入.md`

当前已经能讲一条完整故事：受控路由、分层事实、高风险 HITL、多轮预订确认、可评测。
