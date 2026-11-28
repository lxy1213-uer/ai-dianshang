"""工单状态枚举与迁移白名单（事项 A/B 共用的公共协议）。

工单状态与会话状态（flow.State）生命周期不同，必须独立定义：
- 会话状态管的是"对话进行到哪一步"；
- 工单状态管的是"售后单在履约/退款流程里走到哪一步"。
"""

from __future__ import annotations

from enum import Enum


class TicketStatus(str, Enum):
    CREATED = "CREATED"        # 已建单，待审核
    REVIEWING = "REVIEWING"    # 审核中
    APPROVED = "APPROVED"      # 审核通过，待收货
    RECEIVED = "RECEIVED"      # 已收到退货
    REFUNDED = "REFUNDED"      # 已退款（终态）
    REJECTED = "REJECTED"      # 已驳回（终态）
    CLOSED = "CLOSED"          # 已关闭（终态）

    @property
    def label(self) -> str:
        return _LABELS[self]


_LABELS = {
    TicketStatus.CREATED: "已建单",
    TicketStatus.REVIEWING: "审核中",
    TicketStatus.APPROVED: "审核通过",
    TicketStatus.RECEIVED: "已收到退货",
    TicketStatus.REFUNDED: "已退款",
    TicketStatus.REJECTED: "已驳回",
    TicketStatus.CLOSED: "已关闭",
}

# 迁移白名单：只有白名单内的流转才允许，其余一律 409 且不落库。
ALLOWED: dict[TicketStatus, set[TicketStatus]] = {
    TicketStatus.CREATED: {TicketStatus.REVIEWING, TicketStatus.REJECTED, TicketStatus.CLOSED},
    TicketStatus.REVIEWING: {TicketStatus.APPROVED, TicketStatus.REJECTED, TicketStatus.CLOSED},
    TicketStatus.APPROVED: {TicketStatus.RECEIVED, TicketStatus.CLOSED},
    TicketStatus.RECEIVED: {TicketStatus.REFUNDED, TicketStatus.CLOSED},
    TicketStatus.REFUNDED: set(),
    TicketStatus.REJECTED: set(),
    TicketStatus.CLOSED: set(),
}

TERMINAL = {
    TicketStatus.REFUNDED,
    TicketStatus.REJECTED,
    TicketStatus.CLOSED,
}

# 每个状态的下一步说明（如实回答进度，不编造日期/金额）。
NEXT_STEP = {
    TicketStatus.CREATED: "售后专员会尽快审核你的申请，请留意进度更新。",
    TicketStatus.REVIEWING: "你的申请正在审核中，审核结果会更新到这里。",
    TicketStatus.APPROVED: "审核已通过，请按退货指引寄回商品并填写物流单号。",
    TicketStatus.RECEIVED: "商家已收到退货，正在核验商品，核验通过后安排退款。",
    TicketStatus.REFUNDED: "退款已完成，款项将按原支付方式退回，具体到账时间以支付渠道为准。",
    TicketStatus.REJECTED: "申请未通过审核，如有疑问可联系人工客服。",
    TicketStatus.CLOSED: "该售后工单已关闭，如需继续处理请重新发起申请。",
}


def parse(value: str) -> TicketStatus | None:
    """把库里的字符串转成枚举；兼容旧数据的小写 'created'。"""
    try:
        return TicketStatus(str(value).upper())
    except (ValueError, AttributeError):
        return None


def can_transition(current: TicketStatus, target: TicketStatus) -> bool:
    return target in ALLOWED.get(current, set())
