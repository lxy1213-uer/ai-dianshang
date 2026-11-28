"""售后工单服务：幂等建单的唯一写入口（第 6 周）。"""

from __future__ import annotations

import uuid

from ..config import Settings
from ..db import Database
from .events import EventName, sanitize
from .status import NEXT_STEP, TERMINAL, TicketStatus, can_transition, parse


class TicketError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class TicketService:
    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings

    def create(
        self,
        *,
        session_id: str,
        user_id: str,
        order_id: str,
        confirm_token: str,
        reason: str,
        request_id: str | None = None,
        confirmed: bool = False,
        attempt_no: int = 1,
    ) -> tuple[dict, bool]:
        """幂等键 = 会话 + 订单 + 确认令牌。

        - 用户确认由服务端状态机判定（confirmed 由流程层传入，不接受客户端伪造）；
        - 同一幂等键重复请求返回原工单，不会重复建单；
        - attempt_no == 1 且开启故障演练时，落库后抛超时，模拟"响应丢失"，用于验证重试。
        """
        if not confirmed:
            raise TicketError("not_confirmed", "未检测到有效的用户确认，服务端拒绝建单")
        if not confirm_token:
            raise TicketError("missing_confirm_token", "缺少确认令牌，请重新核单后再确认")

        idempotency_key = f"{session_id}:{order_id}:{confirm_token}"
        request_id = request_id or ("req_" + uuid.uuid4().hex[:12])
        ticket, created = self.db.insert_ticket(
            idempotency_key=idempotency_key,
            session_id=session_id,
            user_id=user_id,
            order_id=order_id,
            reason=reason,
            request_id=request_id,
        )
        self.db.log_event(
            session_id,
            request_id,
            "ticket_create_attempt",
            {
                "order_id": order_id,
                "idempotency_key": idempotency_key,
                "created": created,
                "ticket_id": ticket["ticket_id"],
                "attempt_no": attempt_no,
            },
        )
        if created and self.settings.simulate_response_loss and attempt_no == 1:
            # 工单已经写入数据库，但响应在网络上丢了：客户端会重试
            raise TicketError("response_lost", "建单响应丢失（故障演练）")
        return ticket, created

    # ---------- 事项 A：工单状态回写 ----------
    def update_status(
        self,
        ticket_id: str,
        target: str,
        *,
        operator: str = "agent",
        idempotency_key: str | None = None,
        expected_version: int | None = None,
    ) -> dict:
        """回写工单状态，返回带 version/time_line 的详情视图。

        规则：
        - target 必须是合法状态；非法跃迁抛 TicketError("invalid_transition")；
        - idempotency_key 相同即视为同一回写，重复提交不重复写审计；
        - 终态不可再迁移。
        """
        target_status = parse(target)
        if target_status is None:
            raise TicketError("invalid_status", f"未知的工单状态：{target}")

        ticket = self.db.get_ticket(ticket_id)
        if not ticket:
            raise TicketError("not_found", "工单不存在")

        current = parse(ticket["status"]) or TicketStatus.CREATED
        # 同状态重复回写：幂等返回，不写审计
        if current is target_status:
            return self.detail(ticket_id)

        if current in TERMINAL or not can_transition(current, target_status):
            raise TicketError(
                "invalid_transition",
                f"不允许从「{current.label}」变更到「{target_status.label}」",
            )

        key = idempotency_key or f"{ticket_id}:{current.value}->{target_status.value}:{operator}"
        snapshot_version = int(ticket.get("version") or 1)
        # 乐观锁：调用方传入的 expected_version 与快照不一致，说明期间已被他人更新。
        if expected_version is not None and expected_version != snapshot_version:
            raise TicketError("version_conflict", "工单已被其他人更新，请刷新后重试")
        updated, result = self.db.update_ticket_status(
            ticket_id,
            target_status.value,
            operator,
            NEXT_STEP[target_status],
            idempotency_key=key,
            expected_version=snapshot_version,
        )
        if result == "conflict":
            raise TicketError("version_conflict", "工单已被其他人更新，请刷新后重试")
        if result == "not_found":
            raise TicketError("not_found", "工单不存在")

        # 状态变更也写一条会话级 trace 事件（操作者、前后状态、时间）
        self.db.log_event(
            ticket["session_id"],
            ticket.get("request_id"),
            EventName.TICKET_STATUS_CHANGE.value,
            sanitize({
                "ticket_id": ticket_id,
                "from": current.value,
                "to": target_status.value,
                "operator": operator,
            }),
        )
        return self.detail(ticket_id)

    def detail(self, ticket_id: str) -> dict:
        """工单详情：当前状态 + 时间线 + 更新时间 + 下一步说明。"""
        ticket = self.db.get_ticket(ticket_id)
        if not ticket:
            raise TicketError("not_found", "工单不存在")
        status = parse(ticket["status"]) or TicketStatus.CREATED
        return {
            **ticket,
            "status": status.value,
            "status_label": status.label,
            "is_terminal": status in TERMINAL,
            "next_step": ticket.get("next_step") or NEXT_STEP[status],
            "timeline": self.db.ticket_timeline(ticket_id),
        }
