"""统一事件字典：事项 A/B/C 共用的埋点协议（事项 C 先定 schema）。

设计纪律：
- 事件名集中定义，A（状态变更）、B（人工队列）、C（漏斗）不得各自造字符串；
- 漏斗 6 个关键节点覆盖成功与失败/转人工分支；
- 每条事件带稳定 event_id，同一逻辑事件重试不重复计数；
- 埋点负载禁止出现手机号、详细地址等 PII（见 sanitize）。
"""

from __future__ import annotations

import re
from enum import Enum


class EventName(str, Enum):
    # 漏斗 6 个关键节点（成功分支）
    SESSION_START = "session_start"              # 会话开始
    INTENT_RECOGNIZED = "intent_recognized"      # 意图识别
    ORDER_QUERIED = "order_queried"              # 查单
    ELIGIBILITY_CHECKED = "eligibility_checked"  # 资格判定
    CONFIRMED = "confirmed"                      # 确认
    TICKET_CREATED = "ticket_created"            # 建单

    # 失败 / 转人工分支
    ORDER_QUERY_FAILED = "order_query_failed"
    INELIGIBLE = "ineligible"
    CANCELLED = "cancelled"
    HANDED_OFF = "handed_off"

    # 事项 A / B 的事件
    TICKET_STATUS_CHANGE = "ticket_status_change"
    HUMAN_HANDOFF = "human_handoff"
    AGENT_TAKEOVER = "agent_takeover"
    HUMAN_REPLY = "human_reply"
    RESOLVED = "resolved"
    SLA_BREACH = "sla_breach"        # 转人工待办超时（含通知尝试）
    AGENT_REASSIGN = "agent_reassign"  # 坐席转派
    REFUND_EXECUTED = "refund_executed"            # 网关退款成功
    REFUND_STATUS_FAILED = "refund_status_failed"  # 已打款但状态推进失败，需人工对账


# 漏斗节点顺序（成功路径），聚合接口按此顺序输出。
FUNNEL_STEPS: tuple[str, ...] = (
    EventName.SESSION_START.value,
    EventName.INTENT_RECOGNIZED.value,
    EventName.ORDER_QUERIED.value,
    EventName.ELIGIBILITY_CHECKED.value,
    EventName.CONFIRMED.value,
    EventName.TICKET_CREATED.value,
)

FUNNEL_LABELS = {
    EventName.SESSION_START.value: "会话开始",
    EventName.INTENT_RECOGNIZED.value: "意图识别",
    EventName.ORDER_QUERIED.value: "查单",
    EventName.ELIGIBILITY_CHECKED.value: "资格判定",
    EventName.CONFIRMED.value: "确认",
    EventName.TICKET_CREATED.value: "建单",
}

# 需要脱敏的字段（出现在 payload 里时会被替换）。
_PII_KEYS = {"phone", "mobile", "address", "detail_address", "tracking_no", "receiver"}
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_ID_RE = re.compile(r"(?<!\d)\d{15,18}(?!\d)")


def sanitize(payload: dict) -> dict:
    """脱敏：PII 键整体移除，文本里的手机号/身份证号打码。

    返回新字典，不修改入参。埋点写入失败可降级，但脱敏不能跳过。
    """
    cleaned: dict = {}
    for key, value in payload.items():
        if key.lower() in _PII_KEYS:
            continue
        if isinstance(value, str):
            text = _PHONE_RE.sub(lambda m: m.group(0)[:3] + "****" + m.group(0)[-4:], value)
            text = _ID_RE.sub(lambda m: m.group(0)[:4] + "**********" + m.group(0)[-4:], text)
            cleaned[key] = text
        elif isinstance(value, dict):
            cleaned[key] = sanitize(value)
        elif isinstance(value, list):
            cleaned[key] = [
                sanitize(v) if isinstance(v, dict)
                else (_PHONE_RE.sub("***", _ID_RE.sub("***", v)) if isinstance(v, str) else v)
                for v in value
            ]
        else:
            cleaned[key] = value
    return cleaned
