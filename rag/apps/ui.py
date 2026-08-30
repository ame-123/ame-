from __future__ import annotations

import sys
from pathlib import Path

RAG_ROOT = Path(__file__).resolve().parents[1]
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))

import streamlit as st

from apps.ingest import ingest
from kb.chain import answer
from kb.config import UPLOAD_DIR

st.set_page_config(page_title="知识库", page_icon="📚", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested" not in st.session_state:
    st.session_state.ingested = []
if "last_upload_key" not in st.session_state:
    st.session_state.last_upload_key = None


def _save_upload(uploaded) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / Path(uploaded.name).name
    dest.write_bytes(uploaded.getvalue())
    return dest


left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("上传文档")
    files = st.file_uploader(
        "拖到这里，或点击选择",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    upload_key = tuple((item.name, item.size) for item in files) if files else None
    if files and upload_key != st.session_state.last_upload_key:
        results = []
        errors = []
        with st.spinner("正在切分并写入知识库…"):
            for item in files:
                try:
                    path = _save_upload(item)
                    results.append(ingest(str(path)))
                except Exception as exc:
                    errors.append(f"{item.name}: {exc}")
        st.session_state.last_upload_key = upload_key
        st.session_state.ingested = results
        if errors:
            st.session_state.ingest_errors = errors
        else:
            st.session_state.ingest_errors = []
        st.rerun()

    if st.session_state.ingested:
        st.success("已写入知识库")
        for item in st.session_state.ingested:
            st.caption(f"{item['name']}  ·  {item['chunks']} 段")
    for err in st.session_state.get("ingest_errors") or []:
        st.error(err)

    if st.button("清空对话"):
        st.session_state.messages = []
        st.rerun()

with right:
    st.subheader("对话")
    chat_box = st.container(height=560)
    with chat_box:
        if not st.session_state.messages:
            st.caption("先在左侧拖入文档，再在这里提问。回答会带上来源。")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                confidence = message.get("confidence") or {}
                if confidence.get("low_confidence"):
                    st.caption(
                        f"低置信兜底 · 最高分 {confidence.get('top_score', 0):.3f} "
                        f"< {confidence.get('low_confidence_threshold', 0):.2f}，未把检索结果交给模型。"
                    )
                rewrite = message.get("rewrite") or {}
                if rewrite:
                    with st.expander("检索改写"):
                        st.markdown(f"- 原始问题：{rewrite.get('original_query', '')}")
                        st.markdown(f"- 检索改写：{rewrite.get('rewritten_query', '')}")
                        st.markdown(f"- 意图：`{rewrite.get('intent', '')}`")
                        added = "、".join(rewrite.get("added_terms") or []) or "无"
                        st.markdown(f"- 补充词：{added}")
                        st.markdown(f"- 原因：{rewrite.get('reason', '')}")
                        raw_top = rewrite.get("raw_top") or {}
                        rewritten_top = rewrite.get("rewritten_top") or {}
                        if raw_top:
                            st.caption(
                                f"原始 Top：{raw_top.get('source', '')} / {raw_top.get('title', '')} "
                                f"· {float(raw_top.get('score') or 0):.3f}"
                            )
                        if rewritten_top:
                            st.caption(
                                f"改写后 Top：{rewritten_top.get('source', '')} / {rewritten_top.get('title', '')} "
                                f"· {float(rewritten_top.get('score') or 0):.3f}"
                            )
                plan = message.get("plan") or {}
                if plan:
                    with st.expander("检索计划"):
                        st.markdown(f"- 场景：`{plan.get('scene', '')}`")
                        topics = "、".join(plan.get("allowed_topics") or []) or "无"
                        st.markdown(f"- 允许主题：{topics}")
                        terms = "、".join(plan.get("keyword_terms") or []) or "无"
                        st.markdown(f"- 关键词：{terms}")
                        st.markdown(f"- 原因：{plan.get('reason', '')}")
                        vector_titles = plan.get("vector_titles") or []
                        keyword_titles = plan.get("keyword_titles") or []
                        if vector_titles:
                            st.caption("向量命中：" + "；".join(f"{src}/{title}" for src, title in vector_titles[:4]))
                        if keyword_titles:
                            st.caption("关键词命中：" + "；".join(f"{src}/{title}" for src, title in keyword_titles[:4]))
                        cache = plan.get("cache") or {}
                        if cache or plan.get("index_version"):
                            hit_text = "是" if cache.get("cache_hit") else "否"
                            cacheable = "是" if cache.get("cacheable") else "否"
                            st.caption(
                                f"索引版本：{plan.get('index_version') or cache.get('index_version') or '-'}  ·  "
                                f"缓存命中：{hit_text}  ·  可缓存：{cacheable}  ·  "
                                f"范围：{cache.get('scope') or 'retrieval_hits_only'}"
                            )
                            if cache.get("reason"):
                                st.caption(str(cache["reason"]))
                sources = message.get("sources") or []
                if sources:
                    with st.expander("来源"):
                        for src in sources:
                            score = src.get("score")
                            score_text = f"  ·  {score:.3f}" if isinstance(score, (int, float)) else ""
                            routes = "、".join(src.get("hit_sources") or [])
                            route_text = f"  ·  {routes}" if routes else ""
                            keys = "、".join(src.get("matched_keywords") or [])
                            key_text = f"  ·  关键词 {keys}" if keys else ""
                            st.markdown(
                                f"- `{src.get('source', '')}` / {src.get('title', '')}{score_text}{route_text}{key_text}"
                            )

prompt = st.chat_input("问知识库…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages[:-1]
    ]
    try:
        with st.spinner("检索并生成回答…"):
            result = answer(prompt, history=history)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources") or [],
                "confidence": result.get("confidence") or {},
                "rewrite": result.get("rewrite") or {},
                "plan": result.get("plan") or {},
            }
        )
    except Exception as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"出错了：{exc}",
                "sources": [],
            }
        )
    st.rerun()
