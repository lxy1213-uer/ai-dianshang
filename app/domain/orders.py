"""领域模型：订单、物流与退货资格判断（第 1 周内容，纯函数、可单测）。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

# 状态中文映射，统一在这里维护，前端不再硬编码
STATUS_LABEL = {
    "paid": "待发货",
    "shipped": "已发货",
    "delivered": "已签收",
    "refunded": "已退款",
    "closed": "已关闭",
}

# 不适用七天无理由的类目（R-03）
CATEGORY_NO_REASON_EXCLUDED = {"内衣", "定制", "生鲜", "虚拟"}

REASON_NO_REASON = "no_reason"
REASON_QUALITY = "quality"

REASON_LABEL = {
    REASON_NO_REASON: "七天无理由退货",
    REASON_QUALITY: "质量问题退货",
}

# 资格判定结果码
CODE_ELIGIBLE_NO_REASON = "ELIGIBLE_NO_REASON"
CODE_ELIGIBLE_QUALITY = "ELIGIBLE_QUALITY"
CODE_OUT_OF_WINDOW = "OUT_OF_WINDOW"
CODE_CATEGORY_EXCLUDED = "CATEGORY_EXCLUDED"
CODE_NOT_SIGNED = "NOT_SIGNED"
CODE_NOT_SHIPPED = "NOT_SHIPPED"
CODE_ORDER_CLOSED = "ORDER_CLOSED"
CODE_DATA_MISSING = "DATA_MISSING"


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("/", "-")
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def mask_address(address: str) -> str:
    """地址脱敏：只保留到区/街道，门牌号打码。"""
    if not address:
        return ""
    parts = address.split()
    if len(parts) >= 2:
        return parts[0] + " ****"
    return address[:6] + "****"


@dataclass
class Order:
    order_id: str
    user_id: str
    title: str
    category: str
    amount: float
    status: str
    signed_at: datetime | None = None
    address: str = ""
    sku: str = ""
    channel: str = "APP"

    @property
    def status_label(self) -> str:
        return STATUS_LABEL.get(self.status, self.status)

    @property
    def signed_text(self) -> str:
        return self.signed_at.strftime("%Y-%m-%d %H:%M") if self.signed_at else "未签收"

    @property
    def signed_label(self) -> str:
        if self.signed_at:
            return self.signed_at.strftime("%Y-%m-%d %H:%M")
        # 已签收但没有签收时间，属于数据异常，不能含糊写成"未签收"
        return "缺失（数据异常）" if self.status == "delivered" else "未签收"

    def to_public(self) -> dict:
        """给前端/工具返回用的安全视图（地址脱敏、字段收敛）。"""
        return {
            "order_id": self.order_id,
            "title": self.title,
            "sku": self.sku,
            "category": self.category,
            "amount": round(self.amount, 2),
            "status": self.status,
            "status_label": self.status_label,
            "signed_at": self.signed_at.strftime("%Y-%m-%d %H:%M") if self.signed_at else None,
            "signed_label": self.signed_label,
            "address": mask_address(self.address),
            "channel": self.channel,
        }


@dataclass
class Eligibility:
    eligible: bool
    code: str
    title: str
    detail: str
    reason: str = REASON_NO_REASON
    window_days: int = 7
    days_since_sign: int | None = None
    remaining_days: int | None = None
    deadline: date | None = None
    requires_human: bool = False
    suggestion: str = ""
    citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "eligible": self.eligible,
            "code": self.code,
            "title": self.title,
            "detail": self.detail,
            "reason": self.reason,
            "window_days": self.window_days,
            "days_since_sign": self.days_since_sign,
            "remaining_days": self.remaining_days,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "requires_human": self.requires_human,
            "suggestion": self.suggestion,
            "citations": list(self.citations),
        }


def load_orders(path: Path, today: date) -> dict[str, Order]:
    """读取模拟订单。支持 signed_days_ago，保证任何一天演示都是"新鲜"的边界数据。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    orders: dict[str, Order] = {}
    for item in raw:
        signed_at = parse_dt(item.get("signed_at"))
        if item.get("signed_days_ago") is not None:
            # 固定 10:30 签收，避免时分秒带来的边界抖动
            signed_at = datetime.combine(
                today - timedelta(days=int(item["signed_days_ago"])),
                datetime.min.time(),
            ).replace(hour=10, minute=30)
        order = Order(
            order_id=item["order_id"],
            user_id=item["user_id"],
            title=item["title"],
            category=item.get("category", ""),
            amount=float(item.get("amount", 0)),
            status=item.get("status", "paid"),
            signed_at=signed_at,
            address=item.get("address", ""),
            sku=item.get("sku", ""),
            channel=item.get("channel", "APP"),
        )
        orders[order.order_id] = order
    return orders


def load_logistics(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_return(
    order: Order,
    today: date,
    reason: str = REASON_NO_REASON,
    *,
    no_reason_days: int = 7,
    quality_days: int = 15,
) -> Eligibility:
    """退货资格判定：先看订单状态，再看数据完整性，最后算期限。

    期限算法与 R-01 对齐：签收当日不计入，第 N 天 = 签收日 + N 天，当天 24:00 前有效。
    """
    reason = reason if reason in (REASON_NO_REASON, REASON_QUALITY) else REASON_NO_REASON
    window = quality_days if reason == REASON_QUALITY else no_reason_days
    window_label = "15 天" if reason == REASON_QUALITY else "7 天"

    # 1) 订单状态类
    if order.status == "refunded":
        return Eligibility(
            eligible=False,
            code=CODE_ORDER_CLOSED,
            title="该订单已退款",
            detail="系统显示这笔订单已完成退款，不能重复申请退货。如退款未到账，可查询退款进度或转人工。",
            reason=reason,
            citations=["R-06", "R-08"],
            requires_human=True,
            suggestion="查询退款到账进度",
        )
    if order.status == "closed":
        return Eligibility(
            eligible=False,
            code=CODE_ORDER_CLOSED,
            title="该订单已关闭",
            detail="订单已关闭，无法发起售后。如仍需处理，请转人工核实。",
            reason=reason,
            citations=["R-08"],
            requires_human=True,
        )
    if order.status == "paid":
        return Eligibility(
            eligible=False,
            code=CODE_NOT_SHIPPED,
            title="订单还未发货",
            detail="这笔订单还在待发货状态，不需要走退货流程，可直接申请取消订单并全额退款。",
            reason=reason,
            citations=["R-04"],
            requires_human=False,
            suggestion="申请取消订单",
        )
    if order.status == "shipped" and order.signed_at is None:
        return Eligibility(
            eligible=False,
            code=CODE_NOT_SIGNED,
            title="订单尚未签收",
            detail="包裹还在运输途中，暂不能按签收时间判断退货期限。可以拒收，或签收后再申请退货。",
            reason=reason,
            citations=["R-04"],
            requires_human=False,
            suggestion="查看物流进度",
        )

    # 2) 数据完整性类（签收时间缺失属于异常，只能转人工）
    if order.signed_at is None:
        return Eligibility(
            eligible=False,
            code=CODE_DATA_MISSING,
            title="订单签收信息缺失",
            detail="这笔订单缺少签收时间，系统无法自动判定退货期限，已标记为待人工处理。",
            reason=reason,
            citations=["R-08"],
            requires_human=True,
        )
    if order.signed_at.date() > today:
        return Eligibility(
            eligible=False,
            code=CODE_DATA_MISSING,
            title="签收时间异常",
            detail="系统记录的签收时间晚于当前时间，数据异常，已转人工核实，不会自动拒绝。",
            reason=reason,
            citations=["R-08"],
            requires_human=True,
        )

    days_since = (today - order.signed_at.date()).days
    deadline = order.signed_at.date() + timedelta(days=window)
    remaining = (deadline - today).days

    # 3) 类目限制类（仅无理由退货受限）
    if reason == REASON_NO_REASON and order.category in CATEGORY_NO_REASON_EXCLUDED:
        return Eligibility(
            eligible=False,
            code=CODE_CATEGORY_EXCLUDED,
            title=f"「{order.category}」类目不支持七天无理由退货",
            detail=(
                f"该商品属于{order.category}类目，不适用七天无理由退货。"
                f"如果商品存在破损、故障等质量问题，可上传凭证，按质量问题在签收后 15 天内申请。"
            ),
            reason=reason,
            window_days=window,
            days_since_sign=days_since,
            remaining_days=remaining,
            deadline=deadline,
            citations=["R-03", "R-02"],
            requires_human=True,
            suggestion="商品有质量问题，按质量问题申请",
        )

    # 4) 期限类
    if days_since > window:
        return Eligibility(
            eligible=False,
            code=CODE_OUT_OF_WINDOW,
            title=f"已超过{window_label}退货期限",
            detail=(
                f"这笔订单签收于 {order.signed_at:%Y-%m-%d}，到今天已 {days_since} 天，"
                f"超过 {window_label}期限（截止 {deadline:%Y-%m-%d}）。如属于质量问题，可提供凭证转人工核实。"
            ),
            reason=reason,
            window_days=window,
            days_since_sign=days_since,
            remaining_days=remaining,
            deadline=deadline,
            citations=["R-01", "R-08"],
            requires_human=True,
            suggestion="对判定有异议，转人工",
        )

    if reason == REASON_QUALITY:
        return Eligibility(
            eligible=True,
            code=CODE_ELIGIBLE_QUALITY,
            title="符合质量问题退货条件",
            detail=(
                f"签收于 {order.signed_at:%Y-%m-%d}，今天是第 {days_since} 天，"
                f"在 15 天质量问题期限内（截止 {deadline:%Y-%m-%d}）。退货运费由商家承担，需上传问题凭证。"
            ),
            reason=reason,
            window_days=window,
            days_since_sign=days_since,
            remaining_days=remaining,
            deadline=deadline,
            citations=["R-02", "R-05"],
            suggestion="上传凭证并提交退货申请",
        )

    return Eligibility(
        eligible=True,
        code=CODE_ELIGIBLE_NO_REASON,
        title="符合七天无理由退货条件",
        detail=(
            f"签收于 {order.signed_at:%Y-%m-%d}，今天是第 {days_since} 天，"
            f"还剩 {remaining} 天（截止 {deadline:%Y-%m-%d}）。请保持吊牌与包装完整，运费需自理。"
        ),
        reason=reason,
        window_days=window,
        days_since_sign=days_since,
        remaining_days=remaining,
        deadline=deadline,
        citations=["R-01", "R-05"],
        suggestion="确认后创建售后工单",
    )
