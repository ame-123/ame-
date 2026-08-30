"""全局配置。密钥优先读本目录 .env，没有再回落到课程 env。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

RAG_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RAG_ROOT.parent
KNOWLEDGE_DIR = RAG_ROOT / "knowledge"
UPLOAD_DIR = KNOWLEDGE_DIR / "uploads"
CHUNK_DIR = KNOWLEDGE_DIR / "chunks"

MILVUS_URI = "http://localhost:19530"
DB_NAME = "knowledge_kb"
COLLECTION_NAME = "docs"
TRAVEL_DB_NAME = "travel_kb"
TRAVEL_COLLECTION_NAME = "travel_docs"
EMBED_DIM = 1024
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
CHUNK_LIMIT = 1000
CHUNK_OVERLAP = 80
BATCH_SIZE = 32
# 候选入场线：低于此分的 chunk 不进入后续判断。
SCORE_THRESHOLD = 0.35
# 回答线：最高分低于此值视为低置信，不把弱命中交给模型（课程第 12 课默认 0.68，换库后请校准）。
LOW_CONFIDENCE_THRESHOLD = 0.50
PDF_OCR_MIN_CHARS = 80
PDF_OCR_SCALE = 2.0
PDF_VISION_TABLES = True
QUERY_REWRITE_ENABLED = True
HYBRID_ENABLED = True
VECTOR_TOP_K = 4
KEYWORD_TOP_K = 4
RETRIEVAL_CACHE_ENABLED = True


def load_env() -> None:
    load_dotenv(RAG_ROOT / ".env", override=True)
    course_env = PROJECT_ROOT.parent / "03-代码" / "langchain1.2_tutorial" / ".env"
    if course_env.exists():
        load_dotenv(course_env, override=False)


load_env()
MILVUS_URI = os.getenv("MILVUS_URI", MILVUS_URI)
DB_NAME = os.getenv("DB_NAME", DB_NAME)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", COLLECTION_NAME)
TRAVEL_DB_NAME = os.getenv("TRAVEL_DB_NAME", TRAVEL_DB_NAME)
TRAVEL_COLLECTION_NAME = os.getenv("TRAVEL_COLLECTION_NAME", TRAVEL_COLLECTION_NAME)
EMBED_MODEL = os.getenv("EMBED_MODEL", EMBED_MODEL)
CHUNK_LIMIT = int(os.getenv("CHUNK_LIMIT", str(CHUNK_LIMIT)))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", str(CHUNK_OVERLAP)))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", str(SCORE_THRESHOLD)))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", str(LOW_CONFIDENCE_THRESHOLD)))
PDF_OCR_MIN_CHARS = int(os.getenv("PDF_OCR_MIN_CHARS", str(PDF_OCR_MIN_CHARS)))
PDF_OCR_SCALE = float(os.getenv("PDF_OCR_SCALE", str(PDF_OCR_SCALE)))
PDF_VISION_TABLES = os.getenv("PDF_VISION_TABLES", "1").strip().lower() not in {"0", "false", "no", "off"}
QUERY_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
HYBRID_ENABLED = os.getenv("HYBRID_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", str(VECTOR_TOP_K)))
KEYWORD_TOP_K = int(os.getenv("KEYWORD_TOP_K", str(KEYWORD_TOP_K)))
RETRIEVAL_CACHE_ENABLED = os.getenv("RETRIEVAL_CACHE_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
