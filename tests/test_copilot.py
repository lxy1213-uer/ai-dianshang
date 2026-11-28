"""坐席 Copilot（只读建议层）测试。

覆盖：核单结论、越权/缺单/查无、已退款复议、质量证据不足、超期与类目风险、
SLA 与已有工单提示、一键采纳不越权、只读无副作用、外部订单源故障、审计留痕。
"""

from __future__ import annotations

from datetime import date

import pytest

from app.agent.copilot import CopilotError
from app.domain.agent_desk import DeskError


def _queue(c, sid="s-cp", uid="u_1001", *, order_id="O20261115001",
           turns=None, reason="用户要求人工协助", ticket_id=None, reason_slot=None):
    """构造一条转人工队列记录，带可复用的交接上下文。"""
    slots = {"order_id": order_id}
    if reason_slot:
        slots["reason"] = reason_slot
    if ticket_id:
        slots["ticket_id"] = ticket_id
    recent = turns if turns is not None else [{"kind": "turn", "payload": {"message": "我要退货"}}]
    rec = c.desk.enqueue(
        {"session_id": sid, "user_id": uid, "state": "HUMAN"},
        slots, reason, recent_turns=recent, ticket_id=ticket_id,
    )
    return rec["queue_id"]


# ---------- 核单结论与证据 ----------
def test_copilot_eligible_order_has_facts_clauses_and_readonly_actions(make_coordinator):
    c = make_coordinator()
    qid = _queue(c)
    out = c.copilot.suggest(qid)

    assert out["summary"]["v_code"] == "ELIGIBLE_NO_REASON"
    assert out["summary"]["grounded"] is True
    facts = {f["label"]: f["value"] for f in out["facts"]}
    assert facts["订单"] == "O20261115001"
    assert facts["商品"] == "无线蓝牙耳机 Pro"
    assert facts["实付金额"] == "¥299.00"
    assert any(x["clause_id"] == "R-01" for x in out["clauses"])
    assert "七天无理由" in out["recommended_reply"]
    # 所有动作都需坐席显式执行，没有一个自动
    assert out["actions"] and all(a["auto"] is False for a in out["actions"])
    assert any(a["id"] == "create_ticket" for a in out["actions"])


def test_copilot_reconstructs_situation_from_earlier_turns(make_coordinator):
    """只有最近 6 条事件被打包；情形词在前几轮时要能从拼接的多轮里恢复。"""
    c = make_coordinator()
    qid = _queue(c, order_id="O20261118004", turns=[
        {"kind": "turn", "payload": {"message": "保温杯刻字了我不想要了"}},
        {"kind": "turn", "payload": {"message": "O20261118004"}},
    ])
    out = c.copilot.suggest(qid)
    assert "刻字" in out["question"]
    assert out["summary"]["situation"] == "custom_only"
    # 定制类被排除七天无理由，风险与条款都要对上
    assert any(r["code"] == "category_excluded" for r in out["risks"])
    assert any(x["clause_id"] == "R-03" for x in out["clauses"])
    assert out["summary"]["v_code"] == "CATEGORY_EXCLUDED"


# ---------- 风险提示 ----------
def test_copilot_out_of_window_never_promises_return(make_coordinator):
    c = make_coordinator()
    qid = _queue(c, order_id="O20261115002",
                 turns=[{"kind": "turn", "payload": {"message": "圆领 T 恤超期了还能退吗"}}])
    out = c.copilot.suggest(qid)
    assert out["summary"]["v_code"] == "OUT_OF_WINDOW"
    codes = [r["code"] for r in out["risks"]]
    assert "out_of_window" in codes
    assert not any(a["id"] == "create_ticket" for a in out["actions"])


def test_copilot_refunded_reapply_flags_r05(make_coordinator):
    c = make_coordinator()
    qid = _queue(c, order_id="O20261114011",
                 turns=[{"kind": "turn", "payload": {"message": "这笔钱已经到账了还能再退一次吗"}}])
    out = c.copilot.suggest(qid)
    codes = [r["code"] for r in out["risks"]]
    assert "refunded" in codes
    assert "refund_reapply" in codes
    assert any(x["clause_id"] == "R-05" for x in out["clauses"])


def test_copilot_quality_unproven_warns(make_coordinator):
    """交接槽位里模型声称 quality，但原话只有「刻字」无缺陷证据 → 提示复核。"""
    c = make_coordinator()
    qid = _queue(
        c, order_id="O20261118004", reason_slot="quality",
        turns=[{"kind": "turn", "payload": {"message": "保温杯刻字了能退吗"}}],
    )
    out = c.copilot.suggest(qid)
    assert out["summary"]["reason"] == "no_reason"  # 被证据校验降级
    assert "quality_unproven" in [r["code"] for r in out["risks"]]


def test_copilot_sla_and_existing_ticket_hints(make_coordinator):
    c = make_coordinator()
    qid = _queue(c, ticket_id="RT001")
    rec = c.desk.detail(qid)
    rec["overdue"] = True
    c.db._conn.execute("UPDATE human_queue SET overdue = 1 WHERE queue_id = ?", (qid,))
    c.db._conn.commit()
    out = c.copilot.suggest(qid)
    codes = [r["code"] for r in out["risks"]]
    assert "sla_overdue" in codes
    assert "ticket_exists" in codes


# ---------- 边界与降级 ----------
def test_copilot_ownership_mismatch_shields_order(make_coordinator):
    """队列里的订单不属于会话用户：屏蔽事实、不给资格结论。"""
    c = make_coordinator()
    qid = _queue(c, uid="u_1003", order_id="O20261115001")  # 该单属于 u_1001
    out = c.copilot.suggest(qid)
    assert "ownership_mismatch" in [r["code"] for r in out["risks"]]
    assert out["facts"] == []
    assert out["eligibility"] is None
    assert out["summary"]["grounded"] is False
    assert "核实" in out["recommended_reply"]


def test_copilot_no_order_and_unknown_order(make_coordinator):
    c = make_coordinator()
    q1 = _queue(c, sid="s1", order_id=None)
    o1 = c.copilot.suggest(q1)
    assert "no_order" in [r["code"] for r in o1["risks"]]

    q2 = _queue(c, sid="s2", order_id="O99999999999")
    o2 = c.copilot.suggest(q2)
    assert "order_not_found" in [r["code"] for r in o2["risks"]]
    assert o2["eligibility"] is None


def test_copilot_order_source_unavailable(make_coordinator):
    c = make_coordinator()
    qid = _queue(c)
    c.order_source.error = "订单服务不可用"
    with pytest.raises(CopilotError) as e:
        c.copilot.suggest(qid)
    assert e.value.code == "order_source_unavailable"
    assert e.value.status == 503


def test_copilot_missing_queue_raises(make_coordinator):
    c = make_coordinator()
    with pytest.raises(DeskError) as e:
        c.copilot.suggest("HQ_NOT_EXIST")
    assert e.value.code == "not_found"


# ---------- 只读与留痕 ----------
def test_copilot_is_readonly_no_state_or_ticket_change(make_coordinator):
    c = make_coordinator()
    qid = _queue(c)
    before_tickets = c.db.count_tickets()
    before_state = c.desk.detail(qid)["state"]
    out = c.copilot.suggest(qid)
    assert out["actions"]  # 给了建议动作
    assert c.db.count_tickets() == before_tickets  # 但没建单
    assert c.desk.detail(qid)["state"] == before_state  # 队列状态没变


def test_copilot_logs_view_event(make_coordinator):
    c = make_coordinator()
    qid = _queue(c)
    c.copilot.suggest(qid)
    events = c.db.list_events("s-cp", limit=20)
    assert any(e["kind"] == "desk_copilot_view" for e in events)


# ---------- API 层 ----------
def test_copilot_api_demo_mode_returns_suggestion(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    monkeypatch.setenv("DB_PATH", str(tmp_path / "cp_api.db"))
    monkeypatch.setenv("AGENT_TODAY", "2026-11-22")
    monkeypatch.setenv("AGENT_LLM_MODE", "rules")
    monkeypatch.setenv("AGENT_RBAC_ENABLED", "0")
    monkeypatch.setenv("AGENT_RATE_LIMIT_PER_MIN", "0")
    monkeypatch.setenv("AGENT_STAFF_ACCOUNTS", "")
    with TestClient(app) as tc:
        tc.post("/api/chat", json={
            "session_id": "s_cp_api", "user_id": "u_1001", "message": "我要退货",
        })
        tc.post("/api/chat", json={
            "session_id": "s_cp_api", "user_id": "u_1001", "message": "O20261115001",
        })
        tc.post("/api/chat", json={
            "session_id": "s_cp_api", "user_id": "u_1001", "message": "转人工",
        })
        qid = tc.get("/api/desk/queue").json()["items"][0]["queue_id"]
        r = tc.get(f"/api/desk/queue/{qid}/copilot")
        assert r.status_code == 200
        data = r.json()
        assert data["summary"]["v_code"] == "ELIGIBLE_NO_REASON"
        assert data["recommended_reply"] and data["clauses"]
        assert all(a["auto"] is False for a in data["actions"])


def test_copilot_api_strict_requires_staff_permission(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.auth import issue_staff_token, issue_token
    from app.config import Settings

    monkeypatch.setenv("DB_PATH", str(tmp_path / "cp_strict.db"))
    monkeypatch.setenv("AGENT_TODAY", "2026-11-22")
    monkeypatch.setenv("AGENT_LLM_MODE", "rules")
    monkeypatch.setenv("AGENT_RBAC_ENABLED", "1")
    monkeypatch.setenv("AGENT_AUTH_SECRET", "test-secret")
    monkeypatch.setenv("AGENT_STAFF_ACCOUNTS", "sup:pw:supervisor;ag:pw:agent")
    monkeypatch.setenv("AGENT_RATE_LIMIT_PER_MIN", "0")
    s = Settings.load(db_path=tmp_path / "cp_strict.db")
    agent_token = issue_staff_token("ag", "agent", s.auth_secret)
    user_token = issue_token("u_1001", s.auth_secret)
    with TestClient(app) as tc:
        tc.post("/api/chat", json={
            "session_id": "s_cp_strict", "user_id": "u_1001", "message": "转人工",
        }, headers={"Authorization": "Bearer " + user_token})
        qid = tc.get(
            "/api/desk/queue", headers={"X-Staff-Token": agent_token}
        ).json()["items"][0]["queue_id"]
        # 无 token 被拒
        assert tc.get(f"/api/desk/queue/{qid}/copilot").status_code == 401
        # 有坐席权限（DESK_READ）即可读建议
        r = tc.get(f"/api/desk/queue/{qid}/copilot", headers={"X-Staff-Token": agent_token})
        assert r.status_code == 200
        assert r.json()["queue_id"] == qid
