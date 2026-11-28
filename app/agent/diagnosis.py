"""售后诊断层：有证据、有边界、能执行的决策产品。

这一层不重新发明判定逻辑，而是把项目已有的能力收口成一份可执行的诊断：

- 事实来自 `ToolRegistry.check_return_eligibility`（订单归属 + 资格判定）；
- 规则依据来自 `PolicyIndex` 的真实条款与 `eligibility.citations`；
- 诉求分类来自 `llm.detect_reason` + `slots.validate_reason`（缺陷词有证据才走质量通道）；
- 执行走 `ReturnFlow` 的 AWAIT_CONFIRM → CREATING → TICKETED 状态机，
  重复执行返回原工单，转人工走同一套队列与留痕。

边界很硬：
1. 诊断只读，不退款、不改订单；
2. 执行必须携带服务端下发的 decision_id 和精确确认语；
3. 执行前重新核单，资格变化即拒绝；
4. 描述里匹配出的订单只能用于阅读诊断，执行必须显式带订单号；
5. 越权订单直接拒绝，不把「不属于你」包装成普通诊断结果。
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..domain.order_sources import OrderSourceError
from ..domain.orders import REASON_LABEL
from ..llm import Understanding, detect_reason
from ..slots import normalize_order_id, validate_reason
from .advice import build_advice, evidence_of, match_order, read_situation
from .flow import State
from .tools import ToolError

CONFIRM_CREATE = "确认创建工单"
CONFIRM_HANDOFF = "确认转人工"

_DECISION_TTL_SECONDS = 1800  # 30 分钟：超时后必须重新核单，避免拿旧结论执行


class DiagnosisError(Exception):
    """诊断/执行层对外的稳定错误码，HTTP 层按 status 映射。"""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass
class _Decision:
    decision_id: str
    user_id: str
    question: str
    order_id: str | None
    reason: str
    payload: dict
    created_at: float
    executed: dict[str, dict] = field(default_factory=dict)


class _DecisionStore:
    """进程内短期决策表；重启或超时后要求重新诊断，不拿旧结论执行。"""

    def __init__(self) -> None:
        self._items: dict[str, _Decision] = {}
        self._lock = threading.Lock()

    def put(
        self, decision_id: str, user_id: str, question: str,
        order_id: str | None, reason: str, payload: dict,
    ) -> None:
        with self._lock:
            self._items[decision_id] = _Decision(
                decision_id=decision_id, user_id=user_id, question=question,
                order_id=order_id, reason=reason, payload=payload,
                created_at=time.time(),
            )

    def get(self, decision_id: str, user_id: str, *, require_fresh: bool = True) -> _Decision:
        with self._lock:
            item = self._items.get(decision_id)
        if item is None:
            raise DiagnosisError("decision_not_found", "诊断记录不存在或已过期，请重新诊断", 404)
        if item.user_id != user_id:
            raise DiagnosisError("forbidden", "无权查看或执行其他账号的售后诊断", 403)
        if require_fresh and time.time() - item.created_at > _DECISION_TTL_SECONDS:
            raise DiagnosisError("decision_expired", "诊断已超过 30 分钟，请重新核单后再执行", 409)
        return item

    def mark_executed(self, decision_id: str, action: str, result: dict) -> None:
        with self._lock:
            item = self._items.get(decision_id)
            if item is not None:
                item.executed[action] = result


class AftersaleDiagnosisService:
    """把订单事实、条款证据和可执行动作组装成一份诊断报告。"""

    def __init__(self, coordinator: Any):
        self.coordinator = coordinator
        self.decisions = _DecisionStore()

    # ---------- 对外入口 ----------
    def diagnose(self, question: str, user_id: str, order_id: str | None = None) -> dict:
        text = (question or "").strip()
        if not text:
            raise DiagnosisError("empty_question", "请描述你要咨询的售后问题", 400)
        try:
            self.coordinator.ensure_orders_available()
        except OrderSourceError as exc:
            raise DiagnosisError("order_source_unavailable", str(exc), 503) from exc

        payload = self._build(text, user_id, order_id)
        decision_id = "DG" + uuid.uuid4().hex[:12].upper()
        payload["decision_id"] = decision_id
        payload["expires_in_seconds"] = _DECISION_TTL_SECONDS

        self.decisions.put(
            decision_id, user_id, text, payload.get("order_id"),
            payload["reason"], payload,
        )
        self.coordinator.db.log_event(
            f"diag_{decision_id}", decision_id, "aftersale_diagnosis",
            {
                "verdict": payload["verdict"]["code"],
                "reason": payload["reason"],
                "order_id": payload.get("order_id"),
                "user_id": user_id,
                "citations": [c["clause_id"] for c in payload["evidence"]],
                "executable": [a["id"] for a in payload["actions"] if a["enabled"]],
            },
        )
        return payload

    def for_export(self, decision_id: str, user_id: str) -> dict:
        """只读导出用：复用同一身份校验，但不受 30 分钟执行有效期限制。

        执行仍走 `execute()` 的新鲜度校验——过期诊断可以查阅/导出历史结论，
        但不能拿旧结论直接建单或转人工。
        """
        return self.decisions.get(decision_id, user_id, require_fresh=False).payload

    def execute(self, decision_id: str, user_id: str, action: str, confirm_text: str) -> dict:
        decision = self.decisions.get(decision_id, user_id)
        if action not in ("create_ticket", "handoff"):
            raise DiagnosisError("unknown_action", "不支持执行该动作", 400)

        expected = CONFIRM_CREATE if action == "create_ticket" else CONFIRM_HANDOFF
        if (confirm_text or "").strip() != expected:
            raise DiagnosisError(
                "confirmation_required", f"请准确回复「{expected}」后再执行", 400
            )

        if action in decision.executed:
            return {"ok": True, "duplicate": True, **decision.executed[action]}

        # 执行前重新核单：诊断到执行之间订单状态、期限都可能变化
        fresh = self._build(decision.question, decision.user_id, decision.order_id)
        old_order = (decision.payload.get("order") or {}).get("order_id")
        new_order = (fresh.get("order") or {}).get("order_id")
        if fresh["reason"] != decision.reason or new_order != old_order:
            raise DiagnosisError(
                "stale_decision", "订单或诉求已经变化，请重新诊断后再执行", 409
            )

        if action == "create_ticket":
            result = self._create_ticket(decision, fresh)
        else:
            result = self._handoff(decision, fresh)

        self.decisions.mark_executed(decision_id, action, result)
        self.coordinator.db.log_event(
            result.get("session_id") or f"diag_{decision_id}", decision_id,
            "aftersale_execution",
            {"action": action, "ok": True, "state": result.get("state")},
        )
        return result

    # ---------- 诊断组装 ----------
    def _build(self, question: str, user_id: str, order_id: str | None) -> dict:
        text = (question or "").strip()
        situation = read_situation(text)
        reason_verdict = validate_reason(detect_reason(text), text)
        reason = reason_verdict.reason
        advice = build_advice(text, reason=reason)
        token = evidence_of(text, situation)

        facts: list[dict] = []
        if token:
            facts.append(self._fact("你描述的情形", token, "用户原话"))
        facts.append(self._fact("处理方式", REASON_LABEL.get(reason, reason), "缺陷证据校验"))

        normalized = normalize_order_id(order_id)
        candidate = ""
        explicit_order = bool(normalized)
        if not normalized:
            match = match_order(self.coordinator.orders.values(), user_id, text)
            if match is not None:
                normalized = match.order.order_id
                candidate = match.token

        boundaries = [
            "本诊断只依据当前订单记录与已检索到的政策条款，不替代平台最终审核。",
            "不会自动退款、取消订单或修改订单状态；资金动作一律人工确认。",
            "商品实际状态（是否洗过、吊牌包装是否完整）由你确认，可能影响审核结果。",
            "身份或订单归属校验失败时直接拒绝，不做猜测。",
        ]
        execution = {
            "mode": "server_confirmed",
            "endpoint": "/api/aftersale/execute",
            "confirm_phrases": {
                "create_ticket": CONFIRM_CREATE,
                "handoff": CONFIRM_HANDOFF,
            },
            "note": "执行时会重新核单；资格、期限或订单归属变化都会被拒绝。",
        }

        if not normalized:
            clauses = self._clause_evidence(list(advice.clauses), text)
            gaps = ["尚未提供订单号，无法计算签收期限或确认订单归属。"]
            if candidate:
                gaps.append(
                    f"从描述里匹配到商品「{candidate}」，但匹配只用于阅读诊断；"
                    "执行前请显式带上订单号重新诊断。"
                )
            return self._result(
                question=text, user_id=user_id, order=None, order_id=None,
                matched_token=candidate, situation=situation, reason=reason,
                reason_verdict=reason_verdict.to_dict(),
                verdict={
                    "code": "NEED_ORDER_CONFIRM" if candidate else "NEED_ORDER",
                    "label": "需确认订单" if candidate else "需要订单号",
                    "headline": advice.headline,
                    "summary": advice.body,
                },
                facts=facts, evidence=clauses, gaps=gaps,
                next_steps=[
                    "在「我的订单」复制订单号后重新诊断",
                    "确实找不到订单时可转人工协助",
                ],
                actions=[
                    self._action("handoff", "转人工接手", True, "可把当前问题与匹配到的情形交给人工客服"),
                ],
                boundaries=boundaries, execution=execution,
            )

        ctx = {
            "user_id": user_id,
            "session_id": f"diag_{uuid.uuid4().hex[:12]}",
            "session": {"state": State.IDLE.value, "slots": {}},
            "confirmed": False,
            "request_id": None,
        }
        try:
            result = self.coordinator.tools.call(
                "check_return_eligibility",
                {"order_id": normalized, "reason": reason}, ctx,
            )
        except ToolError as exc:
            if exc.code == "forbidden":
                raise DiagnosisError(
                    "forbidden", "这个订单不在当前登录账号下，诊断与执行均已拒绝", 403
                ) from exc
            if exc.code == "not_found":
                verdict = {
                    "code": "ORDER_NOT_FOUND", "label": "未查到订单",
                    "headline": f"没有查到订单 {normalized}",
                    "summary": "请核对订单号是否完整（一般以 O 开头）。系统不会猜测订单，也不会拿相似订单代替。",
                }
                facts.append(self._fact("订单号", normalized, "用户提供的订单号"))
                return self._result(
                    question=text, user_id=user_id, order=None, order_id=normalized,
                    matched_token=candidate, situation=situation, reason=reason,
                    reason_verdict=reason_verdict.to_dict(), verdict=verdict,
                    facts=facts, evidence=self._clause_evidence(list(advice.clauses), text),
                    gaps=["订单号未命中数据库中的记录。"],
                    next_steps=["重新复制完整订单号", "仍查不到时转人工"],
                    actions=[self._action("handoff", "转人工接手", True, "人工可核对订单来源")],
                    boundaries=boundaries, execution=execution,
                )
            return self._result(
                question=text, user_id=user_id, order=None, order_id=normalized,
                matched_token=candidate, situation=situation, reason=reason,
                reason_verdict=reason_verdict.to_dict(),
                verdict={
                    "code": "DEGRADED", "label": "核单服务暂不可用",
                    "headline": "这次没有拿到订单数据，不给出资格结论",
                    "summary": f"后台核单返回：{exc.message}。为避免误判，本次不生成可执行结论，可稍后重试或转人工。",
                },
                facts=facts, evidence=[], gaps=[f"核单失败：{exc.code}"],
                next_steps=["稍后重新诊断", "转人工处理"],
                actions=[self._action("handoff", "转人工接手", True, "核单服务失败时转人工")],
                boundaries=boundaries, execution=execution,
            )

        order = result["order"]
        eligibility = result["eligibility"]
        eligible = bool(eligibility["eligible"])
        if eligible:
            verdict_code, verdict_label = "ELIGIBLE", "可申请售后"
        elif eligibility.get("requires_human"):
            verdict_code, verdict_label = "HUMAN_REVIEW", "需人工复核"
        else:
            verdict_code, verdict_label = "INELIGIBLE", "暂不可自助办理"

        headline = eligibility["title"]
        if token:
            headline = f"{token}：{headline}"
        summary = f"{advice.headline}。{eligibility['detail']}"
        if eligibility.get("suggestion"):
            summary += f" 建议：{eligibility['suggestion']}。"

        facts.extend([
            self._fact("订单号", order["order_id"], "订单记录"),
            self._fact("商品", order["title"], "订单记录"),
            self._fact("类目", order["category"], "订单记录"),
            self._fact("实付金额", f"¥{order['amount']:.2f}", "订单记录"),
            self._fact("订单状态", order["status_label"], "订单记录"),
            self._fact("签收时间", order["signed_label"], "订单记录"),
            self._fact("订单来源", order.get("channel") or "APP", "订单记录"),
        ])

        citations = list(dict.fromkeys(
            list(eligibility.get("citations") or []) + list(advice.clauses)
        ))
        gaps: list[str] = []
        if candidate and not explicit_order:
            gaps.append(
                f"订单由描述中的「{candidate}」匹配而来；执行前请显式提供订单号 {order['order_id']} 重新诊断。"
            )
        if eligibility.get("requires_human"):
            gaps.append("当前结论需要人工复核，不能直接创建工单。")
        if reason == "no_reason" and situation in ("used", "custom_only", "custom_defect"):
            gaps.append("商品状态或定制属性可能影响无理由退货条件，需按实际状态复核。")

        if eligible:
            next_steps = [
                "核对诊断中的订单事实与条款依据",
                f"确认无误后回复「{CONFIRM_CREATE}」创建售后工单",
                "有疑问可转人工接手",
            ]
        else:
            next_steps = [
                eligibility.get("suggestion") or "按页面提示补充信息后重新诊断",
                "对结论有异议可转人工复核",
            ]

        create_enabled = eligible and explicit_order
        create_reason = (
            "当前订单可申请售后" if create_enabled
            else "诊断阶段不满足自助建单条件" if not eligible
            else "订单由描述匹配，需显式提供订单号后重新诊断"
        )
        actions = [
            self._action("create_ticket", "创建售后工单", create_enabled, create_reason),
            self._action("handoff", "转人工接手", True, "保留诊断上下文交由人工处理"),
        ]

        return self._result(
            question=text, user_id=user_id, order=order, order_id=order["order_id"],
            matched_token=candidate, situation=situation, reason=reason,
            reason_verdict=reason_verdict.to_dict(),
            verdict={
                "code": verdict_code, "label": verdict_label,
                "headline": headline, "summary": summary,
                "eligibility_code": eligibility["code"],
                "remaining_days": eligibility.get("remaining_days"),
                "deadline": eligibility.get("deadline"),
                "requires_human": eligibility.get("requires_human", False),
            },
            facts=facts, evidence=self._clause_evidence(citations, text),
            gaps=gaps, next_steps=next_steps, actions=actions,
            boundaries=boundaries, execution=execution,
        )

    def _result(
        self, *, question: str, user_id: str, order: dict | None, order_id: str | None,
        matched_token: str, situation: str, reason: str, reason_verdict: dict,
        verdict: dict, facts: list[dict], evidence: list[dict], gaps: list[str],
        next_steps: list[str], actions: list[dict], boundaries: list[str],
        execution: dict,
    ) -> dict:
        return {
            "question": question,
            "user_id": user_id,
            "order": order,
            "order_id": order_id,
            "matched_token": matched_token,
            "situation": situation,
            "reason": reason,
            "reason_evidence": reason_verdict,
            "verdict": verdict,
            "facts": facts,
            "evidence": evidence,
            "gaps": gaps,
            "next_steps": next_steps,
            "boundaries": boundaries,
            "actions": actions,
            "execution": execution,
        }

    @staticmethod
    def _fact(label: str, value: str, source: str) -> dict:
        return {"label": label, "value": value, "source": source}

    @staticmethod
    def _action(action_id: str, label: str, enabled: bool, why: str) -> dict:
        return {"id": action_id, "label": label, "enabled": enabled, "why": why}

    def _clause_evidence(self, clause_ids: list[str], question: str) -> list[dict]:
        wanted = list(dict.fromkeys(clause_ids))
        by_id = {c.clause_id: c for c in self.coordinator.index.clauses}
        result: list[dict] = []
        for clause_id in wanted:
            clause = by_id.get(clause_id)
            if clause is None:
                continue
            result.append({
                "clause_id": clause.clause_id,
                "title": clause.title,
                "snippet": clause.salient(question, 110),
                "source": "政策条款检索",
            })
        return result

    # ---------- 执行：复用状态机，不绕过人工确认 ----------
    def _create_ticket(self, decision: _Decision, fresh: dict) -> dict:
        action = next(a for a in fresh["actions"] if a["id"] == "create_ticket")
        if not action["enabled"]:
            raise DiagnosisError("not_executable", action["why"], 409)

        session_id = f"diag_{decision.decision_id}_ticket"
        token = uuid.uuid4().hex
        issued_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        slots = {
            "order_id": fresh["order_id"],
            "reason": fresh["reason"],
            "eligibility": fresh["verdict"],
            "order": fresh["order"],
            "confirm_token": token,
            "confirm_issued_at": issued_at,
        }
        self.coordinator.db.save_session(
            session_id, decision.user_id, State.AWAIT_CONFIRM.value, slots
        )
        session = self.coordinator.db.get_session(session_id)
        output = self.coordinator.flow.handle(
            session, CONFIRM_CREATE,
            Understanding(
                intent="confirm", order_id=fresh["order_id"],
                reason=fresh["reason"], source="diagnosis",
            ),
            request_id=decision.decision_id,
        )
        if output.ticket is None:
            raise DiagnosisError("execution_failed", output.reply, 409)
        return {
            "ok": True,
            "ticket": output.ticket,
            "reply": output.reply,
            "state": output.state.value,
            "session_id": session_id,
        }

    def _handoff(self, decision: _Decision, fresh: dict) -> dict:
        session_id = f"diag_{decision.decision_id}_handoff"
        slots = {
            "order_id": fresh.get("order_id"),
            "reason": fresh["reason"],
            "last_advice": {"key": fresh["situation"], "evidence": fresh.get("matched_token", "")},
            "diagnosis_verdict": fresh["verdict"]["code"],
        }
        self.coordinator.db.save_session(
            session_id, decision.user_id, State.IDLE.value, slots
        )
        session = self.coordinator.db.get_session(session_id)
        output = self.coordinator.flow.handle(
            session, "诊断页转人工",
            Understanding(intent="human", reason=fresh["reason"], source="diagnosis"),
            request_id=decision.decision_id,
        )
        return {
            "ok": True,
            "reply": output.reply,
            "state": output.state.value,
            "session_id": session_id,
        }
