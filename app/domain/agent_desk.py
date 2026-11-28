"""事项 B：人工坐席工作台服务。

职责：转人工入队（打包会话上下文）、坐席列表/筛选、互斥认领、回复回写、
SLA 超时标记、结束归档。所有写操作都落 human_queue 表并写审计事件。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from ..db import Database
from .events import EventName, sanitize


class DeskError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class AgentDesk:
    SLA_MINUTES = 5  # 建议值：5 分钟未认领自动打超时标记

    def __init__(self, db: Database, notifier=None):
        self.db = db
        # 可选通知器：SLA 超时时推送企微；未配置时为 None，通知彻底 no-op。
        self.notifier = notifier
        self.last_notified: list[str] = []   # 最近一次通知的 queue_id，便于观测与测试

    def enqueue(
        self,
        session: dict,
        slots: dict,
        reason: str,
        recent_turns: list[dict] | None = None,
        ticket_id: str | None = None,
    ) -> dict:
        """转人工入队：把对话上下文、槽位、政策引用、工单号打包。"""
        session_id = session["session_id"]
        context = sanitize({
            "slots": {k: v for k, v in slots.items() if k != "confirm_token"},
            "state": session.get("state"),
            "recent_turns": (recent_turns or [])[-6:],
            "ticket_id": ticket_id or slots.get("ticket_id"),
        })
        deadline = (datetime.now() + timedelta(minutes=self.SLA_MINUTES)).strftime("%Y-%m-%d %H:%M:%S")
        record, created = self.db.enqueue_human(
            session_id=session_id,
            user_id=session["user_id"],
            reason=reason[:200],
            context=context,
            sla_deadline=deadline,
            ticket_id=ticket_id or slots.get("ticket_id"),
        )
        if created:
            self.db.log_event(
                session_id, None, EventName.HUMAN_HANDOFF.value,
                {"queue_id": record["queue_id"], "reason": reason[:80], "ticket_id": record.get("ticket_id")},
            )
        return record

    def check_sla(self) -> list[dict]:
        """扫描新超时的待办（幂等：同一条只返回一次），并推送通知。

        由 list_queue 在拉队列时调用；也供定时探测单独调用。通知失败被通知器吞掉，
        不影响队列读取。返回本次新超时的记录，便于测试与可观测。
        """
        records = self.db.mark_overdue_records()
        if not records:
            self.last_notified = []
            return []
        self.last_notified = [r["queue_id"] for r in records]
        if self.notifier is not None:
            self.notifier.notify_sla_overdue(records)
        for r in records:
            self.db.log_event(
                r["session_id"], None, EventName.SLA_BREACH.value,
                {"queue_id": r["queue_id"], "user_id": r["user_id"], "reason": r.get("reason", "")[:80]},
            )
        return records

    def list_queue(
        self, state: str | None = None, assignee: str | None = None,
        only_overdue: bool = False, limit: int = 50,
    ) -> list[dict]:
        self.check_sla()
        return self.db.list_human_queue(state=state, assignee=assignee, only_overdue=only_overdue, limit=limit)

    def detail(self, queue_id: str) -> dict:
        record = self.db.get_human_queue(queue_id)
        if not record:
            raise DeskError("not_found", "转人工记录不存在")
        return record

    def claim(self, queue_id: str, assignee: str) -> dict:
        record, result = self.db.claim_human(queue_id, assignee)
        if result == "not_found":
            raise DeskError("not_found", "转人工记录不存在")
        if result == "conflict":
            raise DeskError("already_claimed", "该会话已被其他坐席认领")
        if result in ("claimed", "idempotent"):
            self.db.log_event(
                record["session_id"], None, EventName.AGENT_TAKEOVER.value,
                {"queue_id": queue_id, "assignee": assignee},
            )
        return record

    def reassign(self, queue_id: str, assignee: str, operator: str | None = None) -> dict:
        """转派：把待办改派给另一名坐席，写审计事件（含原坐席）。"""
        before = self.db.get_human_queue(queue_id)
        record, result = self.db.reassign_human(queue_id, assignee)
        if result == "not_found":
            raise DeskError("not_found", "转人工记录不存在")
        if result == "resolved":
            raise DeskError("resolved", "该会话已归档，无法转派")
        if result == "reassigned":
            self.db.log_event(
                record["session_id"], None, EventName.AGENT_REASSIGN.value,
                {
                    "queue_id": queue_id,
                    "from": (before or {}).get("assignee"),
                    "to": assignee,
                    "operator": operator or assignee,
                },
            )
        return record

    def reply(self, queue_id: str, assignee: str, text: str) -> dict:
        record = self.db.get_human_queue(queue_id)
        if not record:
            raise DeskError("not_found", "转人工记录不存在")
        if record["state"] == "PENDING":
            raise DeskError("not_claimed", "请先认领该会话再回复")
        if record["state"] == "RESOLVED":
            raise DeskError("resolved", "该会话已归档，无法继续回复")
        if assignee and record["assignee"] != assignee:
            raise DeskError("forbidden", "只能回复自己认领的会话")
        self.db.append_human_reply(queue_id, assignee, text)
        self.db.log_event(
            record["session_id"], None, EventName.HUMAN_REPLY.value,
            {"queue_id": queue_id, "assignee": assignee, "text": text[:120]},
        )
        return {"session_id": record["session_id"], "text": text}

    def resolve(self, queue_id: str) -> dict:
        record = self.db.resolve_human(queue_id)
        if not record:
            raise DeskError("not_found", "转人工记录不存在")
        self.db.log_event(
            record["session_id"], None, EventName.RESOLVED.value,
            {"queue_id": queue_id, "assignee": record.get("assignee")},
        )
        return record

    def user_messages(self, session_id: str) -> list[dict]:
        """用户在原会话能看到的人工回复（供会话视图合并展示）。"""
        record = self.db.get_human_queue_by_session(session_id)
        if not record:
            return []
        return [
            {"role": "agent_human", "text": r["text"], "at": r["at"], "assignee": r["assignee"]}
            for r in record["replies"]
        ]
