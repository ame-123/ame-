# 知识库助手

三个解耦目录：检索入库、旅行客服 Agent、路由小模型微调。Agent 用 RAG 拿政策出处，用只读工具拿行程事实，高风险走 LangGraph + HITL；分类小模型只输出受控 JSON。

```text
用户一句
  → 规则护栏（安全 / 降级）
  → 路由小模型（本句 + 草稿 + 近轮）
  → RoutePlan 白名单（工具 / RAG / workflow）
  → 缺槽、库存失败、HITL：不调最终模型
  → 有事实时大模型只写话术
```

| 目录 | 做什么 |
|---|---|
| `rag/` | 文档切分、Embedding、Milvus Hybrid、低置信拒答 |
| `travel_agent/` | FastAPI Agent、预订/退票状态机、24 条回归 |
| `intent-finetune/` | 13 类意图造数、LoRA SFT + DPO、独立测试集 |

演示不执行真实出票或退票。密钥只放在本地 `.env` / `course.env`，仓库里只有 example。

## 知识库 RAG

入库链路：**上传 → 结构切分 → 可追溯问答**。切分按标题和版面，不按固定字数硬切。旅行客服用独立库 `travel_kb.travel_docs`。

```powershell
cd rag
copy .env.example .env
pip install -r requirements.txt
python apps/ingest.py knowledge/uploads/sample.md
python apps/ingest_travel.py
```

阈值：`SCORE_THRESHOLD=0.35`（入场），`LOW_CONFIDENCE_THRESHOLD=0.50`（低于此线不把弱命中交给模型）。细节见 `rag/README.md`。

## 旅行客服 Agent

政策问答走 Hybrid RAG 并带 citations；行程与库存走只读 mock；预订和退票走显式状态机，风险暂停等人批。本机端口：`8081` mock，`8000` 分类，`8002` Agent HTTP。

```powershell
cd travel_agent
copy course.env.example course.env
pip install -r requirements.txt
# 终端 1
python mock_backend/main.py
# 终端 2（可选，本地 Qwen 分类）
# 见 intent-finetune README
# 终端 3
cd backend
python chat.py
```

演进记录：`travel_agent/docs/发现问题到解决-完整路径.md`。

## 路由小模型微调

基座 `Qwen2.5-3B-Instruct`，LoRA，不要全参。金标来自场景模板，不用规则回标。SFT 后再对「只看当前句就会错」的样本做 DPO。

独立测试集 258 条：SFT 后意图准确率 85.7%，DPO 后 93.8%（+8.1pp），多轮上下文 93.2%（+15.9pp）。权重不入库，需自行训练。步骤见 `intent-finetune/README.md`。

## 不包含

- API Key、`.env`、`course.env`
- LoRA / 基座权重、`outputs/`
- 聊天日志、评测 sqlite
