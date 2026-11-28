"""决策审计驾驶舱：把散落的事件串成可审计的决策链。

项目已经有完整的事件留痕（诊断、执行、建单、状态变更、退款、护栏拦截、SLA），
但它们是按会话/工单分散存储的。本服务把一次"决策"作为一等公民：

- 锚点是 `aftersale_diagnosis` 事件，request_id 即 decision_id；
- 同一 decision_id 下的 `aftersale_execution` 是这次决策的执行动作；
- 执行建出的工单、工单状态时间线、退款流水都挂在同一条链上；
- 会话前缀 `diag_<decision_id>_` 把护栏拦截 / 资格变化等中间事件也收拢进来。

设计纪律：
1. **只读**：不建单、不退款、不改任何状态，审计层只聚合与脱敏；
2. **不臆造证据**：没有数据库记录时明确返回空链，不用推断补齐事实；
3. **读时脱敏兜底**：log_event 写入未统一 sanitize，这里读出来再脱敏一次，
   手机号/身份证/详细地址等 PII 绝不进驾驶舱；
4. **聚合口径可解释**：执行率、退款异常数都给出分子/分母，便于运营核对。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ..db import Database
from .events import sanitize

# 决策链事件
DIAGNOSIS_KIND = "aftersale_diagnosis"
EXECUTION_KIND = "aftersale_execution"

# verdict → 中文标签（与 diagnosis.py 的 verdict.code 对齐）
VERDICT_LABELS = {
    "ELIGIBLE": "可申请售后",
    "HUMAN_REVIEW": "需人工复核",
    "INELIGIBLE": "暂不可自助办理",
    "NEED_ORDER": "需要订单号",
    "NEED_ORDER_CONFIRM": "需确认订单",
    "ORDER_NOT_FOUND": "未查到订单",
    "DEGRADED": "核单降级",
}

# 事件类型 → 中文标签与严重度（风险清单用）
EVENT_META: dict[str, dict[str, str]] = {
    "aftersale_diagnosis": {"label": "售后诊断", "group": "决策"},
    "aftersale_execution": {"label": "决策执行", "group": "决策"},
    "ticket_status_change": {"label": "工单状态变更", "group": "履约"},
    "refund_executed": {"label": "退款成功", "group": "资金"},
    "refund_status_failed": {"label": "退款状态推进失败", "group": "资金"},
    "refund_callback_received": {"label": "网关退款回调", "group": "资金"},
    "grounding_rejected": {"label": "事实校验拦截", "group": "护栏"},
    "security_violation": {"label": "安全越权拦截", "group": "护栏"},
    "confirm_invalidated": {"label": "确认摘要失效", "group": "护栏"},
    "confirm_expired": {"label": "确认超时", "group": "护栏"},
    "eligibility_changed": {"label": "资格复核变化", "group": "护栏"},
    "session_user_switched": {"label": "会话账号切换", "group": "护栏"},
    "state_recovered": {"label": "状态自愈", "group": "护栏"},
    "handoff": {"label": "转人工", "group": "履约"},
    "human_handoff": {"label": "人工队列入队", "group": "履约"},
    "agent_takeover": {"label": "坐席认领", "group": "履约"},
    "agent_reassign": {"label": "坐席转派", "group": "履约"},
    "human_reply": {"label": "坐席回复", "group": "履约"},
    "resolved": {"label": "人工结案", "group": "履约"},
    "sla_breach": {"label": "SLA 超时", "group": "履约"},
    "desk_copilot_view": {"label": "Copilot 建议查看", "group": "辅助"},
}

# 风险事件：需要人工关注但不属于正常成功路径
RISK_KINDS = (
    "refund_status_failed",
    "grounding_rejected",
    "security_violation",
    "confirm_invalidated",
    "confirm_expired",
    "eligibility_changed",
    "session_user_switched",
    "state_recovered",
    "sla_breach",
)

_DECISION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class AuditService:
    """只读决策审计服务；持有 Database 引用，所有读都在同一连接锁内。"""

    def __init__(self, db: Database):
        self.db = db

    # ---------- 总览 ----------
    def overview(
        self, date_from: str | None = None, date_to: str | None = None
    ) -> dict:
        """驾驶舱 KPI：决策量、执行率、资金动作、护栏拦截与风险待办。"""
        events = self.db.query_events(
            date_from=date_from, date_to=date_to, limit=10000
        )
        diag = [e for e in events if e["kind"] == DIAGNOSIS_KIND]
        execs = [e for e in events if e["kind"] == EXECUTION_KIND]

        verdict_counter = Counter(
            (e["payload"] or {}).get("verdict") or "UNKNOWN" for e in diag
        )
        verdicts = [
            {
                "code": code,
                "label": VERDICT_LABELS.get(code, code),
                "count": count,
            }
            for code, count in verdict_counter.most_common()
        ]

        create_execs = [
            e for e in execs if (e["payload"] or {}).get("action") == "create_ticket"
        ]
        handoff_execs = [
            e for e in execs if (e["payload"] or {}).get("action") == "handoff"
        ]
        kind_counter = Counter(e["kind"] for e in events)
        risk_items = [e for e in events if e["kind"] in RISK_KINDS]
        risk_by_kind = Counter(e["kind"] for e in risk_items)

        return {
            "from": date_from,
            "to": date_to,
            "total_decisions": len(diag),
            "verdicts": verdicts,
            "execution": {
                "executed": len(execs),
                "create_ticket": len(create_execs),
                "handoff": len(handoff_execs),
                # 分母用决策总数：一条决策可能有多个动作，执行数 >= 决策数时比率可 > 1，
                # 因此这里用"有执行记录的决策数"作分子，口径更保守可解释。
                "distinct_decisions_executed": len({e["request_id"] for e in execs}),
                "rate": self._rate(len({e["request_id"] for e in execs}), len(diag)),
            },
            "funds": {
                "refund_executed": kind_counter.get("refund_executed", 0),
                "refund_failed": kind_counter.get("refund_status_failed", 0),
                "callback_received": kind_counter.get("refund_callback_received", 0),
            },
            "guardrails": {
                "grounding_rejected": kind_counter.get("grounding_rejected", 0),
                "security_violation": kind_counter.get("security_violation", 0),
                "confirm_invalidated": kind_counter.get("confirm_invalidated", 0),
                "confirm_expired": kind_counter.get("confirm_expired", 0),
                "eligibility_changed": kind_counter.get("eligibility_changed", 0),
            },
            "risks": {
                "total": len(risk_items),
                "by_kind": [
                    {
                        "kind": kind,
                        "label": EVENT_META.get(kind, {}).get("label", kind),
                        "count": count,
                    }
                    for kind, count in risk_by_kind.most_common()
                ],
            },
            "events_total": len(events),
        }

    # ---------- 决策列表 ----------
    def decisions(
        self,
        verdict: str | None = None,
        executed: str | None = None,
        user_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
    ) -> dict:
        """按诊断事件列出决策，支持 verdict / 是否已执行 / 用户 / 时间筛选。

        executed: "yes" / "no" / None；user_id 取诊断 payload 里的归属字段（若有）。
        """
        diag_events = self.db.query_events(
            kinds=[DIAGNOSIS_KIND], date_from=date_from, date_to=date_to,
            order="desc", limit=10000,
        )
        exec_by_request: dict[str, list[dict]] = {}
        for e in self.db.query_events(kinds=[EXECUTION_KIND], limit=10000):
            exec_by_request.setdefault(e["request_id"] or "", []).append(e)

        items: list[dict] = []
        for e in diag_events:
            payload = sanitize(e["payload"] or {})
            decision_id = e["request_id"] or ""
            row = self._decision_row(e, payload, exec_by_request.get(decision_id, []))
            if verdict and row["verdict"] != verdict:
                continue
            if executed == "yes" and not row["has_execution"]:
                continue
            if executed == "no" and row["has_execution"]:
                continue
            if user_id and row["user_id"] != user_id:
                continue
            items.append(row)

        return {
            "total": len(items),
            "items": items[: max(1, min(int(limit or 50), 200))],
            "filters": {
                "verdict": verdict, "executed": executed,
                "user_id": user_id, "from": date_from, "to": date_to,
            },
        }

    # ---------- 决策详情（完整链） ----------
    def detail(self, decision_id: str) -> dict:
        """按 decision_id 串出诊断 → 执行 → 工单 → 状态 → 退款 → 护栏的完整时间线。"""
        did = (decision_id or "").strip()
        if not _DECISION_ID_RE.fullmatch(did):
            return {"decision_id": did, "found": False, "chain": [], "ticket": None}

        # 诊断事件是决策锚点；会话前缀收拢 diag_<id> 下的中间事件。
        diag_events = self.db.query_events(
            kinds=[DIAGNOSIS_KIND], request_id=did, limit=10
        )
        prefixed = self.db.query_events(
            session_prefix=f"diag_{did}", limit=500
        )
        chain_events = list(prefixed)
        for e in diag_events:
            if e["id"] not in {x["id"] for x in chain_events}:
                chain_events.append(e)
        # 工单会话名固定为 diag_<id>_ticket，状态变更事件按 ticket_id 落库，
        # 用前缀通常已覆盖；这里再按 request_id 兜底一次，避免漏掉跨会话落库。
        for e in self.db.query_events(request_id=did, limit=500):
            if e["id"] not in {x["id"] for x in chain_events}:
                chain_events.append(e)

        chain_events.sort(key=lambda x: (x["created_at"], x["id"]))

        diag = diag_events[0] if diag_events else None
        diag_payload = sanitize(diag["payload"] or {}) if diag else {}
        executions = [
            e for e in chain_events if e["kind"] == EXECUTION_KIND
        ]
        actions = [
            {
                "action": (e["payload"] or {}).get("action"),
                "ok": (e["payload"] or {}).get("ok"),
                "state": (e["payload"] or {}).get("state"),
                "created_at": e["created_at"],
            }
            for e in executions
        ]

        ticket = self.db.get_ticket_by_session(f"diag_{did}_ticket")
        ticket_view = None
        refund_view = None
        if ticket:
            ticket_view = {
                "ticket_id": ticket["ticket_id"],
                "status": ticket.get("status"),
                "order_id": ticket.get("order_id"),
                "reason": ticket.get("reason"),
                "created_at": ticket.get("created_at"),
                "updated_at": ticket.get("updated_at") or ticket.get("created_at"),
                "timeline": self.db.ticket_timeline(ticket["ticket_id"]),
            }
            refund = self.db.get_refund(ticket["ticket_id"])
            if refund:
                refund_view = sanitize(
                    {
                        "status": refund.get("status"),
                        "refund_id": refund.get("refund_id"),
                        "amount": refund.get("amount"),
                        "gateway": refund.get("gateway"),
                        "operator": refund.get("operator"),
                        "error": refund.get("error"),
                    }
                )

        chain = []
        for e in chain_events:
            meta = EVENT_META.get(e["kind"], {})
            chain.append(
                {
                    "kind": e["kind"],
                    "label": meta.get("label", e["kind"]),
                    "group": meta.get("group", "其他"),
                    "created_at": e["created_at"],
                    "session_id": e["session_id"],
                    "ticket_id": e["ticket_id"],
                    "payload": sanitize(e["payload"] or {}),
                }
            )

        return {
            "decision_id": did,
            "found": diag is not None,
            "diagnosis": {
                "verdict": diag_payload.get("verdict"),
                "verdict_label": VERDICT_LABELS.get(
                    diag_payload.get("verdict"), diag_payload.get("verdict")
                ),
                "reason": diag_payload.get("reason"),
                "order_id": diag_payload.get("order_id"),
                "user_id": diag_payload.get("user_id"),
                "citations": diag_payload.get("citations") or [],
                "executable": diag_payload.get("executable") or [],
                "created_at": diag["created_at"] if diag else None,
            },
            "actions": actions,
            "ticket": ticket_view,
            "refund": refund_view,
            "chain": chain,
        }

    # ---------- 全局事件流 ----------
    def events(
        self,
        kinds: list[str] | None = None,
        group: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 100,
    ) -> dict:
        """跨会话事件流，按类型/分组/时间过滤；payload 一律脱敏。"""
        rows = self.db.query_events(
            kinds=kinds, date_from=date_from, date_to=date_to,
            order="desc", limit=5000,
        )
        items = []
        for e in rows:
            meta = EVENT_META.get(e["kind"], {})
            if group and meta.get("group") != group:
                continue
            items.append(
                {
                    "kind": e["kind"],
                    "label": meta.get("label", e["kind"]),
                    "group": meta.get("group", "其他"),
                    "created_at": e["created_at"],
                    "session_id": e["session_id"],
                    "request_id": e["request_id"],
                    "ticket_id": e["ticket_id"],
                    "payload": sanitize(e["payload"] or {}),
                }
            )
        return {
            "total": len(items),
            "items": items[: max(1, min(int(limit or 100), 500))],
        }

    # ---------- 风险清单 ----------
    def risks(self, limit: int = 100) -> dict:
        """只列仍需人工关注的关键事件，按严重度分组，附去重后的决策号。"""
        rows = self.db.query_events(kinds=list(RISK_KINDS), order="desc", limit=5000)
        items = []
        for e in rows:
            meta = EVENT_META.get(e["kind"], {})
            payload = sanitize(e["payload"] or {})
            items.append(
                {
                    "kind": e["kind"],
                    "label": meta.get("label", e["kind"]),
                    "group": meta.get("group", "其他"),
                    "severity": self._severity(e["kind"]),
                    "created_at": e["created_at"],
                    "session_id": e["session_id"],
                    "decision_id": e["request_id"],
                    "ticket_id": e["ticket_id"] or payload.get("ticket_id"),
                    "detail": self._risk_detail(e["kind"], payload),
                }
            )
        return {
            "total": len(items),
            "by_severity": dict(Counter(i["severity"] for i in items)),
            "items": items[: max(1, min(int(limit or 100), 500))],
        }

    # ---------- 内部工具 ----------
    def _decision_row(
        self, event: dict, payload: dict, executions: list[dict]
    ) -> dict:
        verdict = payload.get("verdict")
        return {
            "decision_id": event["request_id"] or "",
            "created_at": event["created_at"],
            "verdict": verdict,
            "verdict_label": VERDICT_LABELS.get(verdict, verdict),
            "reason": payload.get("reason"),
            "order_id": payload.get("order_id"),
            "user_id": payload.get("user_id"),
            "citations": payload.get("citations") or [],
            "executable": payload.get("executable") or [],
            "has_execution": bool(executions),
            "actions": [
                {
                    "action": (e["payload"] or {}).get("action"),
                    "ok": (e["payload"] or {}).get("ok"),
                    "state": (e["payload"] or {}).get("state"),
                    "created_at": e["created_at"],
                }
                for e in executions
            ],
        }

    @staticmethod
    def _rate(part: int, whole: int) -> float:
        return round(part / whole, 4) if whole else 0.0

    @staticmethod
    def _severity(kind: str) -> str:
        if kind in ("refund_status_failed", "security_violation", "session_user_switched"):
            return "high"
        if kind in ("sla_breach", "eligibility_changed", "confirm_invalidated", "grounding_rejected"):
            return "medium"
        return "low"

    @staticmethod
    def _risk_detail(kind: str, payload: dict) -> str:
        """用事件负载拼一句可执行的中文说明；缺失字段不臆造。"""
        if kind == "refund_status_failed":
            return (
                f"工单 {payload.get('ticket_id') or '未知'} 已打款但状态推进失败，"
                f"refund_id={payload.get('refund_id') or '未知'}，需人工对账。"
            )
        if kind == "security_violation":
            return f"拦截一次 {payload.get('type') or '越权'} 尝试，订单 {payload.get('order_id') or '未知'}。"
        if kind == "grounding_rejected":
            violations = payload.get("violations") or []
            return f"模型话术触发事实校验（{', '.join(map(str, violations[:3])) or '未知类型'}），已回退到可信模板。"
        if kind == "eligibility_changed":
            return f"确认阶段复核资格变化（{payload.get('code') or '未知'}），未自动建单。"
        if kind == "confirm_invalidated":
            return (
                f"确认摘要失效：{payload.get('order_id') or '未知'} → "
                f"{payload.get('new_order_id') or '未知'}，需重新确认。"
            )
        if kind == "confirm_expired":
            return f"订单 {payload.get('order_id') or '未知'} 的确认超时，未自动建单。"
        if kind == "sla_breach":
            return f"待办 {payload.get('queue_id') or '未知'} 超时，需坐席跟进。"
        if kind == "session_user_switched":
            return f"会话账号切换：{payload.get('from') or '未知'} → {payload.get('to') or '未知'}。"
        if kind == "state_recovered":
            return f"进程自愈：{payload.get('from') or '未知'} → {payload.get('to') or '未知'}，共 {payload.get('count') or 0} 条。"
        return "需人工关注。"
