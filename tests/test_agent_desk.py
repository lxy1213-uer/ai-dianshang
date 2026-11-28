"""事项 B：人工坐席工作台的入队、认领互斥、回复、SLA、归档测试。"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.domain.agent_desk import AgentDesk, DeskError


def _session(sid="s-desk", uid="u_1001"):
    return {"session_id": sid, "user_id": uid, "state": "HUMAN"}


def _enqueue(desk, sid="s-desk", ticket_id=None, reason="用户要求人工协助"):
    return desk.enqueue(
        _session(sid), {"order_id": "O20261115001", "ticket_id": ticket_id},
        reason, recent_turns=[{"kind": "turn", "payload": {"text": "我要退货"}}],
        ticket_id=ticket_id,
    )


def test_enqueue_packages_context(make_coordinator):
    c = make_coordinator()
    rec = _enqueue(c.desk, ticket_id="RT001")
    assert rec["state"] == "PENDING"
    assert rec["context"]["slots"]["order_id"] == "O20261115001"
    assert rec["context"]["ticket_id"] == "RT001"
    assert len(rec["context"]["recent_turns"]) == 1
    assert rec["sla_deadline"]


def test_enqueue_is_idempotent_per_session(make_coordinator):
    c = make_coordinator()
    first = _enqueue(c.desk)
    second = _enqueue(c.desk)
    assert first["queue_id"] == second["queue_id"]
    assert len(c.desk.list_queue()) == 1


def test_claim_is_mutually_exclusive(make_coordinator):
    c = make_coordinator()
    rec = _enqueue(c.desk)
    qid = rec["queue_id"]
    claimed = c.desk.claim(qid, "agent-1")
    assert claimed["state"] == "CLAIMED"
    assert claimed["assignee"] == "agent-1"
    with pytest.raises(DeskError) as e:
        c.desk.claim(qid, "agent-2")
    assert e.value.code == "already_claimed"


def test_claim_same_agent_is_idempotent(make_coordinator):
    c = make_coordinator()
    rec = _enqueue(c.desk)
    c.desk.claim(rec["queue_id"], "agent-1")
    again = c.desk.claim(rec["queue_id"], "agent-1")
    assert again["assignee"] == "agent-1"


def test_reply_requires_claim_and_owner(make_coordinator):
    c = make_coordinator()
    rec = _enqueue(c.desk)
    qid = rec["queue_id"]
    with pytest.raises(DeskError) as e:
        c.desk.reply(qid, "agent-1", "你好")
    assert e.value.code == "not_claimed"
    c.desk.claim(qid, "agent-1")
    with pytest.raises(DeskError) as e:
        c.desk.reply(qid, "agent-2", "你好")
    assert e.value.code == "forbidden"
    c.desk.reply(qid, "agent-1", "我已收到你的退货申请")
    replies = c.desk.user_messages("s-desk")
    assert len(replies) == 1
    assert replies[0]["role"] == "agent_human"
    assert replies[0]["text"] == "我已收到你的退货申请"


def test_reply_visible_in_session_view(make_coordinator):
    c = make_coordinator()
    rec = _enqueue(c.desk)
    c.desk.claim(rec["queue_id"], "agent-1")
    c.desk.reply(rec["queue_id"], "agent-1", "请寄回商品")
    view = c.session_view("s-desk", "u_1001")
    assert any(m["text"] == "请寄回商品" for m in view["human_replies"])


def test_sla_overdue_marker(make_coordinator):
    c = make_coordinator()
    rec = _enqueue(c.desk)
    qid = rec["queue_id"]
    # 手动把 deadline 改到过去，再触发标记
    past = (datetime.now() - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
    with c.db._lock:
        c.db._conn.execute(
            "UPDATE human_queue SET sla_deadline = ? WHERE queue_id = ?", (past, qid)
        )
        c.db._conn.commit()
    marked = c.db.mark_overdue()
    assert marked >= 1
    detail = c.desk.detail(qid)
    assert detail["overdue"] == 1
    assert len(c.desk.list_queue(only_overdue=True)) >= 1


def test_resolve_archives(make_coordinator):
    c = make_coordinator()
    rec = _enqueue(c.desk)
    qid = rec["queue_id"]
    c.desk.claim(qid, "agent-1")
    c.desk.reply(qid, "agent-1", "已处理")
    archived = c.desk.resolve(qid)
    assert archived["state"] == "RESOLVED"
    assert archived["resolved_at"]
    with pytest.raises(DeskError) as e:
        c.desk.reply(qid, "agent-1", "再补充")
    assert e.value.code == "resolved"


def test_queue_filters(make_coordinator):
    c = make_coordinator()
    r1 = _enqueue(c.desk, sid="s-a")
    _enqueue(c.desk, sid="s-b")
    c.desk.claim(r1["queue_id"], "agent-1")
    assert len(c.desk.list_queue(state="CLAIMED")) == 1
    assert len(c.desk.list_queue(state="PENDING")) == 1
    assert len(c.desk.list_queue(assignee="agent-1")) == 1
