"""坐席增强验收：快捷回复模板、转派（审计/幂等/权限/归档拒绝）。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# ---------- 领域层：转派 ----------

def _enqueue(c):
    return c.desk.enqueue(
        {"session_id": "s_reassign", "user_id": "u_1001", "state": "HUMAN"},
        {}, "需要转派", [],
    )


def test_reassign_changes_assignee_and_logs(make_coordinator):
    c = make_coordinator()
    rec = _enqueue(c)
    qid = rec["queue_id"]
    c.desk.claim(qid, "坐席-01")

    out = c.desk.reassign(qid, "坐席-02", operator="主管-01")
    assert out["assignee"] == "坐席-02"
    assert out["state"] == "CLAIMED"

    evs = [e for e in c.db.list_events("s_reassign") if e["kind"] == "agent_reassign"]
    assert len(evs) == 1
    p = evs[0]["payload"]
    assert p["from"] == "坐席-01" and p["to"] == "坐席-02" and p["operator"] == "主管-01"


def test_reassign_from_pending(make_coordinator):
    c = make_coordinator()
    rec = _enqueue(c)
    out = c.desk.reassign(rec["queue_id"], "坐席-05")
    assert out["assignee"] == "坐席-05"
    assert out["state"] == "CLAIMED"


def test_reassign_idempotent(make_coordinator):
    c = make_coordinator()
    rec = _enqueue(c)
    c.desk.reassign(rec["queue_id"], "坐席-02")
    again = c.desk.reassign(rec["queue_id"], "坐席-02")
    assert again["assignee"] == "坐席-02"
    evs = [e for e in c.db.list_events("s_reassign") if e["kind"] == "agent_reassign"]
    assert len(evs) == 1  # 不重复写事件


def test_reassign_rejected_after_resolve(make_coordinator):
    from app.domain.agent_desk import DeskError

    c = make_coordinator()
    rec = _enqueue(c)
    c.desk.claim(rec["queue_id"], "坐席-01")
    c.desk.resolve(rec["queue_id"])
    with pytest.raises(DeskError) as exc:
        c.desk.reassign(rec["queue_id"], "坐席-02")
    assert exc.value.code == "resolved"


def test_reassign_not_found(make_coordinator):
    from app.domain.agent_desk import DeskError

    c = make_coordinator()
    with pytest.raises(DeskError) as exc:
        c.desk.reassign("no-such-qid", "坐席-02")
    assert exc.value.code == "not_found"


# ---------- API 层：快捷回复、转派、权限 ----------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "enh.db"))
    monkeypatch.setenv("AGENT_TODAY", "2026-11-22")
    monkeypatch.setenv("AGENT_LLM_MODE", "rules")
    monkeypatch.setenv("AGENT_RBAC_ENABLED", "0")
    monkeypatch.setenv("AGENT_RATE_LIMIT_PER_MIN", "0")
    monkeypatch.setenv("AGENT_STAFF_ACCOUNTS", "")
    from app.main import app
    with TestClient(app) as tc:
        yield tc


def test_quick_replies_endpoint(client):
    d = client.get("/api/desk/quick-replies").json()
    assert len(d["items"]) >= 5
    assert all("id" in q and "label" in q and "text" in q for q in d["items"])
    assert d["assignees"]  # demo 模式给演示工号


def test_reassign_endpoint_demo(client):
    # 通过 chat 触发转人工：显式说「转人工」
    client.post("/api/chat", json={"session_id": "s_api_r", "user_id": "u_1001", "message": "转人工"})
    q = client.get("/api/desk/queue").json()["items"]
    assert q
    qid = q[0]["queue_id"]
    r = client.post(
        f"/api/desk/queue/{qid}/reassign", json={"assignee": "坐席-09"}
    )
    assert r.status_code == 200
    assert r.json()["assignee"] == "坐席-09"


def test_reassign_forbidden_for_agent_role(tmp_path, monkeypatch):
    """strict 模式下 agent 角色不能转派（需 supervisor）。"""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "strict.db"))
    monkeypatch.setenv("AGENT_TODAY", "2026-11-22")
    monkeypatch.setenv("AGENT_LLM_MODE", "rules")
    monkeypatch.setenv("AGENT_RBAC_ENABLED", "1")
    monkeypatch.setenv("AGENT_AUTH_SECRET", "test-secret")
    monkeypatch.setenv("AGENT_STAFF_ACCOUNTS", "sup:pw:supervisor;ag:pw:agent;adm:pw:admin")
    monkeypatch.setenv("AGENT_RATE_LIMIT_PER_MIN", "0")
    from app.main import app
    from app.auth import issue_staff_token, issue_token
    from app.config import Settings
    s = Settings.load()
    with TestClient(app) as tc:
        agent_token = issue_staff_token("ag", "agent", s.auth_secret)
        admin_token = issue_staff_token("adm", "admin", s.auth_secret)
        user_token = issue_token("u_1001", s.auth_secret)
        tc.post(
            "/api/chat",
            json={"session_id": "s_strict", "user_id": "u_1001", "message": "转人工"},
            headers={"Authorization": "Bearer " + user_token},
        )
        q = tc.get("/api/desk/queue", headers={"X-Staff-Token": agent_token}).json()["items"]
        qid = q[0]["queue_id"]
        r = tc.post(
            f"/api/desk/queue/{qid}/reassign", json={"assignee": "别人"},
            headers={"X-Staff-Token": agent_token},
        )
        assert r.status_code == 403
        r2 = tc.post(
            f"/api/desk/queue/{qid}/reassign", json={"assignee": "sup"},
            headers={"X-Staff-Token": admin_token},
        )
        assert r2.status_code == 200
