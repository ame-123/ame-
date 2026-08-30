"""第 13 课查询改写：用户原话留给回答，rewritten_query 只用于检索。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


NORMALIZATION_RULES: list[tuple[str, str, str]] = [
    ("那个", "", "去掉缺少上下文的指代词“那个”"),
    ("这个", "", "去掉缺少上下文的指代词“这个”"),
    ("那款", "", "去掉缺少上下文的指代词“那款”"),
    ("这款", "", "去掉缺少上下文的指代词“这款”"),
    ("耳麦", "耳机", "把用户口语“耳麦”对齐为知识库常用词“耳机”"),
    ("叠券", "叠加 优惠券", "把用户口语“叠券”展开为“叠加 优惠券”"),
    ("会员券", "优惠券", "把“会员券”归一到知识库里的“优惠券”"),
    ("能叠吗", "能否 叠加 优惠券", "把省略问法“能叠吗”展开为优惠叠加问题"),
    ("促销", "活动", "把“促销”归一到活动规则用词"),
]


@dataclass
class QueryRewrite:
    original_query: str
    rewritten_query: str
    applied: bool
    added_terms: list[str]
    reason: str
    intent: str = "unknown"

    def as_dict(self) -> dict:
        return asdict(self)


def normalize_query(text: str) -> str:
    """把用户口语归一到知识库更常见的检索词。"""

    normalized = text.lower()
    for source, target, _reason in NORMALIZATION_RULES:
        normalized = normalized.replace(source, target)
    return normalized


def describe_normalization(original_query: str) -> list[str]:
    """返回本轮归一化命中的原因说明。"""

    query = original_query.lower()
    return [reason for source, _target, reason in NORMALIZATION_RULES if source in query]


def add_rewrite_terms(target: list[str], terms: list[str]) -> None:
    """向改写词列表中追加不重复的检索补词。"""

    for term in terms:
        if term not in target:
            target.append(term)


def build_rewrite_reason(normalization_reasons: list[str], rewrite_reasons: list[str]) -> str:
    """把归一化和补词原因合并成调试可读说明。"""

    reasons = [*normalization_reasons, *rewrite_reasons]
    if reasons:
        return "；".join(reasons) + "。"
    return "未命中归一化或补词规则，保留原问题直接检索。"


def classify_intent(user_message: str) -> str:
    """用归一化后的文本识别粗意图。"""

    message = normalize_query(user_message)
    if any(word in message for word in ["投诉", "举报", "赔偿", "曝光", "315"]):
        return "complaint"
    if any(word in message for word in ["退款", "退货", "退票", "取消订单", "坏了", "无法开机", "质量问题", "无理由"]):
        return "refund_request"
    if any(word in message for word in ["物流", "快递", "发货", "到哪", "运单", "行程", "航班"]):
        return "order_query"
    if any(word in message for word in ["优惠", "活动", "会员价", "券", "满减", "折扣", "早鸟", "出行券"]):
        return "promotion_consult"
    if any(word in message for word in ["耳机", "充电器", "音箱", "推荐", "哪个好", "机票", "酒店", "套餐", "东京"]):
        return "product_consult"
    return "unknown"


def rewrite_retrieval_query(
    query: str,
    intent: str | None = None,
    member_level: str | None = None,
) -> QueryRewrite:
    """根据粗意图和口语表达生成检索问题，不改用户原话。"""

    intent = intent or classify_intent(query)
    normalized = normalize_query(query)
    added_terms: list[str] = []
    rewrite_reasons: list[str] = []

    if intent == "promotion_consult" and "耳机" in normalized:
        add_rewrite_terms(added_terms, ["当前", "2026", "春季音频节"])
        rewrite_reasons.append("促销咨询提到耳机，补齐当前活动时间和活动名")
        add_rewrite_terms(added_terms, ["降噪耳机"])
        rewrite_reasons.append("补齐知识库里的具体商品类目“降噪耳机”")
        add_rewrite_terms(added_terms, ["会员价", "优惠券", "叠加", "结算页"])
        rewrite_reasons.append("补齐会员价、优惠券叠加和结算页边界，避免旧活动规则抢到第一名")
        if member_level == "gold":
            add_rewrite_terms(added_terms, ["金卡"])
            rewrite_reasons.append("运行时上下文显示当前用户是金卡会员，补充会员等级用于检索")
    elif intent == "refund_request":
        add_rewrite_terms(added_terms, ["售后规则", "签收时间", "退货条件", "凭证"])
        rewrite_reasons.append("售后意图需要补齐签收时间、退货条件和凭证要求这些规则检索词")

    rewritten_query = " ".join(part for part in [normalized, *added_terms] if part).strip() or query
    return QueryRewrite(
        original_query=query,
        rewritten_query=rewritten_query,
        applied=rewritten_query != query,
        added_terms=added_terms,
        reason=build_rewrite_reason(describe_normalization(query), rewrite_reasons),
        intent=intent,
    )
