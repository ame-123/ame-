"""旅行客服可离线运行的 LangChain Hybrid RAG、Query Rewrite 与索引缓存。"""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from api.schemas import Citation, Intent
from config.settings import api_key_is_missing, embedding_model_name, load_course_env, openai_base_url
from rag.documents import load_knowledge_citation


KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"
KNOWLEDGE_FILES = (
    "after_sale_policy.md",
    "order_service_policy.md",
    "complaint_escalation_policy.md",
    "payment_invoice_policy.md",
    "promotion_policy.md",
    "member_coupon_policy.md",
    "received_return_policy.md",
)
RAG_RETRIEVAL_CACHE: dict[str, "HybridRetrievalResult"] = {}
_KNOWLEDGE_INDEX: "KnowledgeIndex | None" = None


@dataclass
class KnowledgeIndex:
    version: str
    vector_store: InMemoryVectorStore
    documents: list[Document]
    embedding_mode: str


@dataclass
class HybridRetrievalResult:
    citations: list[Citation]
    debug: dict[str, Any]


class LocalTokenEmbeddings(Embeddings):
    """确定性字符/双字向量，只用于显式离线评测，不冒充商用语义 embedding。"""

    dimensions = 256

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        normalized = normalize_query(text)
        tokens = [*normalized, *[normalized[index : index + 2] for index in range(max(0, len(normalized) - 1))]]
        vector = [0.0] * self.dimensions
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:2], "big") % self.dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def build_course_embeddings() -> tuple[Embeddings, str]:
    """在线默认使用真实语义 Embedding；离线替身必须由评测显式开启。"""
    load_course_env()
    if os.getenv("AGENT_COURSE_OFFLINE_RAG") == "1":
        return LocalTokenEmbeddings(), "local_token_embedding_for_explicit_offline_eval"

    api_key = os.getenv("AGENT_OPENAI_API_KEY")
    if api_key_is_missing(api_key):
        raise RuntimeError(
            "RAG 需要 AGENT_OPENAI_API_KEY；仅离线测试可显式设置 AGENT_COURSE_OFFLINE_RAG=1"
        )
    from langchain_openai import OpenAIEmbeddings

    model_name = embedding_model_name()
    return (
        OpenAIEmbeddings(
            model=model_name,
            api_key=api_key,
            base_url=openai_base_url(),
            request_timeout=30,
            max_retries=0,
            check_embedding_ctx_length=False,
        ),
        f"openai_compatible_embedding:{model_name}",
    )


def normalize_query(text: str) -> str:
    """把常见口语别名改写成知识文件里的稳定业务词。"""
    return (
        re.sub(r"\s+", "", text.lower())
        .replace("退钱", "退票")
        .replace("退款", "退票")
        .replace("叠券", "叠加出行券")
        .replace("开票", "行程电子发票")
        .replace("那个", "")
        .replace("这个", "")
    )


def build_retrieval_plan(query: str, intent: Intent) -> dict[str, Any]:
    rewritten = normalize_query(query)
    domains = {
        "promotion_query": ["promotion", "member_coupon"],
        "faq_query": ["invoice"],
        "refund_request": ["after_sale"],
        "booking_request": ["promotion", "member_coupon"],
    }.get(intent, [])
    return {
        "original_query": query,
        "rewritten_query": rewritten,
        "intent": intent,
        "knowledge_domains": domains,
        "reason": "口语归一后按 RoutePlan 知识域执行向量与关键词双路召回。",
    }


def get_knowledge_index() -> tuple[KnowledgeIndex, bool]:
    """按真实 Markdown 内容构建一次切块和向量索引，后续请求复用。"""
    global _KNOWLEDGE_INDEX
    if _KNOWLEDGE_INDEX is not None:
        return _KNOWLEDGE_INDEX, True

    source_documents: list[Document] = []
    fingerprint_parts: list[str] = []
    for filename in KNOWLEDGE_FILES:
        path = KNOWLEDGE_DIR / filename
        text = path.read_text(encoding="utf-8")
        citation = load_knowledge_citation(filename)
        fingerprint_parts.append(f"{filename}:{text}")
        source_documents.append(
            Document(
                page_content=text.split("---", 2)[-1].strip(),
                metadata={
                    "filename": filename,
                    "source": citation.source,
                    "title": citation.title,
                    "policy_id": (citation.metadata or {}).get("policy_id"),
                    "scene_key": (citation.metadata or {}).get("scene_key"),
                    "base_score": citation.score,
                },
            )
        )
    splitter = RecursiveCharacterTextSplitter(chunk_size=220, chunk_overlap=30)
    chunks = splitter.split_documents(source_documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['policy_id']}-chunk-{index + 1}"
    embeddings, embedding_mode = build_course_embeddings()
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(chunks)
    fingerprint_parts.append(f"embedding:{embedding_mode}")
    fingerprint = hashlib.sha256("\n".join(fingerprint_parts).encode("utf-8")).hexdigest()[:12]
    _KNOWLEDGE_INDEX = KnowledgeIndex(
        version=f"travel-idx-{fingerprint}",
        vector_store=vector_store,
        documents=chunks,
        embedding_mode=embedding_mode,
    )
    return _KNOWLEDGE_INDEX, False


def retrieve_knowledge(query: str, intent: Intent, *, top_k: int = 4) -> HybridRetrievalResult:
    """执行 Query Rewrite、向量召回、关键词召回、合并重排和检索缓存。"""
    plan = build_retrieval_plan(query, intent)
    index, index_cache_hit = get_knowledge_index()
    cache_key = hashlib.sha256(f"{index.version}|{intent}|{plan['rewritten_query']}".encode("utf-8")).hexdigest()[:16]
    if cache_key in RAG_RETRIEVAL_CACHE:
        cached = RAG_RETRIEVAL_CACHE[cache_key]
        return HybridRetrievalResult(
            citations=list(cached.citations),
            debug={**cached.debug, "retrieval_cache_hit": True, "index_cache_hit": True},
        )

    domains = set(plan["knowledge_domains"])
    # InMemoryVectorStore 返回的就是余弦相似度；当前 LangChain 版本没有额外的
    # relevance-score 转换器，因此直接使用它支持的 score 接口。
    vector_pairs = index.vector_store.similarity_search_with_score(plan["rewritten_query"], k=top_k)
    vector_scores = {
        str(document.metadata["policy_id"]): float(score)
        for document, score in vector_pairs
        if _domain_allowed(str(document.metadata.get("scene_key")), domains)
    }
    keyword_scores: dict[str, float] = {}
    for document in index.documents:
        if not _domain_allowed(str(document.metadata.get("scene_key")), domains):
            continue
        score = _keyword_score(plan["rewritten_query"], document.page_content)
        if score > 0:
            policy_id = str(document.metadata["policy_id"])
            keyword_scores[policy_id] = max(keyword_scores.get(policy_id, 0.0), score)

    merged_ids = set(vector_scores) | set(keyword_scores)
    ranked: list[tuple[str, float, list[str]]] = []
    for policy_id in merged_ids:
        vector_score = vector_scores.get(policy_id, 0.0)
        keyword_score = keyword_scores.get(policy_id, 0.0)
        score = max(vector_score, keyword_score) + (0.08 if vector_score and keyword_score else 0.0)
        reasons = []
        if vector_score:
            reasons.append("vector召回")
        if keyword_score:
            reasons.append("keyword召回")
        ranked.append((policy_id, round(min(1.0, score), 3), reasons))
    ranked.sort(key=lambda item: item[1], reverse=True)

    filename_by_policy = {
        str(document.metadata["policy_id"]): str(document.metadata["filename"])
        for document in index.documents
    }
    citations: list[Citation] = []
    source_scores: dict[str, Any] = {}
    for policy_id, score, reasons in ranked[:top_k]:
        citation = load_knowledge_citation(filename_by_policy[policy_id])
        citations.append(citation.model_copy(update={"score": score}))
        source_scores[policy_id] = {
            "vector": round(vector_scores.get(policy_id, 0.0), 3),
            "keyword": round(keyword_scores.get(policy_id, 0.0), 3),
            "final": score,
            "sources": reasons,
        }
    result = HybridRetrievalResult(
        citations=citations,
        debug={
            "mode": "langchain_inmemory_hybrid",
            "embedding": index.embedding_mode,
            "plan": plan,
            "index_version": index.version,
            "index_chunk_count": len(index.documents),
            "index_cache_hit": index_cache_hit,
            "retrieval_cache_hit": False,
            "cache_key": cache_key,
            "vector_policy_ids": list(vector_scores),
            "keyword_policy_ids": list(keyword_scores),
            "source_scores": source_scores,
        },
    )
    RAG_RETRIEVAL_CACHE[cache_key] = result
    return result


def reset_hybrid_index_and_cache() -> None:
    global _KNOWLEDGE_INDEX
    _KNOWLEDGE_INDEX = None
    RAG_RETRIEVAL_CACHE.clear()


def _domain_allowed(scene_key: str, domains: set[str]) -> bool:
    if not domains:
        return True
    mapping = {
        "promotion": "promotion",
        "member_coupon": "member_coupon",
        "invoice": "invoice",
        "payment_invoice": "invoice",
        "after_sale": "after_sale",
    }
    return mapping.get(scene_key, scene_key) in domains


def _keyword_score(query: str, text: str) -> float:
    normalized_text = normalize_query(text)
    business_terms = {
        term
        for term in (
            "早鸟",
            "满减",
            "6000减400",
            "出行券",
            "金卡",
            "叠加",
            "电子发票",
            "发票",
            "24小时",
            "退票",
            "未出行",
        )
        if term in query
    }
    chinese_bigrams = {
        query[index : index + 2]
        for index in range(max(0, len(query) - 1))
        if re.fullmatch(r"[\u4e00-\u9fff]{2}", query[index : index + 2])
    }
    terms = business_terms | chinese_bigrams | set(re.findall(r"[a-z0-9]+", query))
    matched = [term for term in terms if term in normalized_text]
    return round(min(1.0, len(matched) / max(1, len(terms)) + (0.15 if matched else 0.0)), 3)
