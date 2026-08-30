"""第 15 课 pre-retrieval：检索前先定知识场景和主题范围。

主题不入库，只按现有 source 文件名推断。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from kb.query_rewrite import classify_intent, normalize_query as base_normalize


SCENE_NORMALIZATION_RULES: list[tuple[str, str]] = [
    ("少一根", "少了 配件"),
    ("线少了", "配件 少了"),
    ("盒子", "包装盒"),
    ("压了", "压坏"),
]

TOPIC_BY_SOURCE: dict[str, str] = {
    "after_sale_policy.md": "after_sale",
    "complaint_escalation_policy.md": "after_sale",
    "received_return_policy.md": "after_sale",
    "payment_invoice_policy.md": "after_sale",
    "promotion_policy.md": "promotion",
    "member_coupon_policy.md": "promotion",
    "product_guide.md": "product",
    "shipping_faq.md": "shipping",
    "order_service_policy.md": "shipping",
}

ALLOWED_TOPICS_BY_SCENE: dict[str, list[str]] = {
    "promotion": ["promotion"],
    "after_sale": ["after_sale"],
    "shipping": ["shipping"],
    "product": ["product", "promotion"],
    "unknown": ["promotion", "product", "after_sale", "shipping"],
}


@dataclass
class RetrievalPlan:
    original_query: str
    rewritten_query: str
    scene: str
    allowed_topics: list[str]
    keyword_terms: list[str]
    added_terms: list[str]
    reason: str
    intent: str = "unknown"

    def as_dict(self) -> dict:
        return asdict(self)


def normalize_query(text: str) -> str:
    """在第 13 课口语归一化之上，补第 15 课的售后长尾说法。"""

    normalized = base_normalize(text)
    for source, target in SCENE_NORMALIZATION_RULES:
        normalized = normalized.replace(source, target)
    return normalized


def topic_of_source(source: str) -> str | None:
    """用文件名推断主题；论文等未登记来源返回 None。"""

    name = str(source or "").replace("\\", "/").split("/")[-1].lower()
    return TOPIC_BY_SOURCE.get(name)


def sources_for_topics(topics: list[str]) -> list[str]:
    """把允许主题转换成已有 source 文件名，不改入库。"""

    allowed = set(topics)
    return [source for source, topic in TOPIC_BY_SOURCE.items() if topic in allowed]


def detect_scene(query: str, intent: str | None = None) -> str:
    """根据问题和意图判断检索场景。"""

    text = normalize_query(query)
    intent = intent or classify_intent(query)
    if any(term in text for term in ["配件", "赠品", "包装盒", "退货", "退款", "退票", "退改", "值机", "行李", "签证", "无理由", "质量问题", "售后"]):
        return "after_sale"
    if any(term in text for term in ["物流", "快递", "发货", "运单", "行程", "航班", "值机进度"]):
        return "shipping"
    if any(term in text for term in ["优惠", "活动", "会员价", "优惠券", "满减", "叠加", "早鸟", "出行券"]):
        return "promotion"
    if intent == "product_consult" or any(term in text for term in ["耳机", "机票", "酒店", "套餐", "东京"]):
        return "product"
    return "unknown"


def is_realtime_business_query(query: str) -> bool:
    """识别必须走业务系统、不能当成稳定知识缓存的问题。"""

    text = normalize_query(query)
    realtime_terms = [
        "我的订单",
        "订单到哪",
        "快递到哪",
        "物流到哪",
        "退款进度",
        "库存还有",
        "有没有库存",
        "发货了吗",
    ]
    return any(term in text for term in realtime_terms)


def keyword_terms_for_scene(query: str, scene: str) -> list[str]:
    """根据场景补齐关键词召回使用的术语。"""

    text = normalize_query(query)
    terms: list[str] = []
    for term in ["当前", "早鸟", "出行券", "满减", "6000", "优惠券", "叠加", "结算页"]:
        if term in text or scene == "promotion":
            terms.append(term)
    for term in ["未出行", "退票", "退改", "值机", "无理由", "售后", "人工审批"]:
        if term in text or scene == "after_sale":
            terms.append(term)
    for term in ["行程", "航班", "物流", "值机"]:
        if term in text or scene == "shipping":
            terms.append(term)
    return list(dict.fromkeys(terms))


def build_plan_reason(scene: str, allowed_topics: list[str], added_terms: list[str]) -> str:
    """生成检索计划的可读原因。"""

    topic_text = "、".join(allowed_topics)
    scene_labels = {
        "promotion": "促销",
        "after_sale": "售后",
        "shipping": "物流",
        "product": "商品",
        "unknown": "未知",
    }
    if scene == "unknown":
        return f"未识别到稳定知识场景，保留全主题候选（{topic_text}），不额外补词。"

    reason = f"识别为{scene_labels.get(scene, scene)}知识场景，限制到 {topic_text} 主题"
    if added_terms:
        reason += f"，并补齐{'、'.join(added_terms)}关键词。"
    else:
        reason += "，不额外补词，只用归一化后的用户问题生成关键词召回项。"
    return reason


def pre_retrieval_plan(query: str, intent: str | None = None) -> RetrievalPlan:
    """在真正检索前决定主题范围和关键词补充。"""

    intent = intent or classify_intent(query)
    scene = detect_scene(query, intent)
    allowed_topics = list(ALLOWED_TOPICS_BY_SCENE[scene])
    normalized = normalize_query(query)
    added: list[str] = []
    if scene == "promotion":
        added = ["当前", "出行早鸟", "满6000减400", "出行券", "叠加", "结算页"]
    elif scene == "after_sale":
        added = ["售后规则", "未出行", "退票", "人工审批"]
    rewritten = " ".join(part for part in [normalized, *added] if part).strip() or query
    return RetrievalPlan(
        original_query=query,
        rewritten_query=rewritten,
        scene=scene,
        allowed_topics=allowed_topics,
        keyword_terms=keyword_terms_for_scene(rewritten, scene),
        added_terms=added,
        reason=build_plan_reason(scene, allowed_topics, added),
        intent=intent,
    )
