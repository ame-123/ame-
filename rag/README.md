# 知识库 RAG

入库链路：**上传 → 结构切分 → 可追溯问答**。

切分不是按固定字数硬切：Word 先抽出标题结构写成 Markdown；PDF 用 PyMuPDF 做版面解析（分栏、字号标题），表格先定位再裁图，交给视觉模型按 `ExtractedTable` schema 抽取；文字层不足的扫描页走 RapidOCR。统一成 Markdown 后，用 LangChain 按标题切开。

旅行客服走独立 Milvus 库 `travel_kb.travel_docs`，不覆盖通用库 `knowledge_kb.docs`。

## 流程

```text
文件（pdf/docx/md/txt）
  → PDF：版面解析正文；表格裁图走视觉模型 schema
  → 扫描页：OCR 正文，表格仍裁图走视觉模型
  → 解析成带 # 标题和 Markdown 表的文本
  → MarkdownHeaderTextSplitter 按标题切开
  → RecursiveCharacterTextSplitter 把超长段切到 chunk_size
  → Embedding 写入 Milvus（同名文件覆盖更新）
  → retrieve / answer（先过入场分，最高分低于低置信线则兜底拒答）
```

## 准备

1. 本机已有 Milvus（`localhost:19530`）。
2. 复制 `.env.example` 为 `.env`，填入自己的 API Key。不要提交 `.env`。

```powershell
conda activate langchain12
cd rag
pip install -r requirements.txt
copy .env.example .env
```

## 用法

```powershell
# 切分预览（不入库）
python apps/split.py knowledge/uploads/sample.md

# 上传并入库（同名文件会先删旧切片再写入）
python apps/ingest.py knowledge/uploads/sample.md

# 旅行客服政策入库（写入 travel_kb.travel_docs）
python apps/ingest_travel.py

# 本地界面（拖拽上传 + 对话）
streamlit run apps/ui.py
```

## 公开接口（给 Agent 用）

- `kb.retriever.retrieve(question) -> list[dict]`
- `kb.hybrid.retrieve_knowledge(...)`：向量 + 关键词 Hybrid
- `kb.chain.answer(question) -> dict`（含 `answer`、`sources`、`confidence`）

两条阈值（可写进 `.env`）：

- `SCORE_THRESHOLD`：候选入场线，默认 `0.35`
- `LOW_CONFIDENCE_THRESHOLD`：回答线，默认 `0.50`。最高分低于此值时不调用模型，`sources` 为空。换模型和语料后需要校准。
