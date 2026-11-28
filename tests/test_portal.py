"""自助进度页验收：按用户查全部工单、不问也能查、归属隔离、页面挂载。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "portal.db"))
    monkeypatch.setenv("AGENT_TODAY", "2026-11-22")
    monkeypatch.setenv("AGENT_LLM_MODE", "rules")
    monkeypatch.setenv("AGENT_RBAC_ENABLED", "0")
    monkeypatch.setenv("AGENT_AUTH_SECRET", "")
    monkeypatch.setenv("AGENT_RATE_LIMIT_PER_MIN", "0")
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def _create_ticket(client, session_id: str, user_id: str, order_id: str = "O20261115001") -> str:
    # 直接落库：自助进度查询只关心"工单是否在库里且归属正确"，不必走完整 chat 建单
    # （订单有 user_id 归属，非 u_1001 用户无法通过 chat 核单）。
    c = client.app.state.coordinator
    ticket = c.db.insert_ticket(
        f"k-{session_id}", session_id, user_id, order_id, "no_reason", f"req-{session_id}"
    )[0]
    return ticket["ticket_id"]


def test_portal_lists_tickets_by_user_without_session(client):
    """用户不必提供 session_id，凭身份就能查到自己名下所有工单。"""
    tid = _create_ticket(client, "s_portal_1", "u_1001")
    body = client.get("/api/portal/tickets?user_id=u_1001").json()
    assert body["user_id"] == "u_1001"
    assert any(t["ticket_id"] == tid for t in body["tickets"])
    t = next(t for t in body["tickets"] if t["ticket_id"] == tid)
    assert t["order_id"] == "O20261115001"
    assert t["status"] == "CREATED"
    assert t["created_at"] and t["updated_at"]


def test_portal_isolated_by_user(client):
    """不同 user_id 看到的工单互相隔离（demo 模式显式传参也必须隔离）。"""
    tid_a = _create_ticket(client, "s_portal_a", "u_A")
    tid_b = _create_ticket(client, "s_portal_b", "u_B", order_id="O20261115002")
    a = client.get("/api/portal/tickets?user_id=u_A").json()
    b = client.get("/api/portal/tickets?user_id=u_B").json()
    assert {t["ticket_id"] for t in a["tickets"]} == {tid_a}
    assert {t["ticket_id"] for t in b["tickets"]} == {tid_b}


def test_portal_requires_identity(client):
    """缺少身份标识应 401。"""
    r = client.get("/api/portal/tickets")
    assert r.status_code == 401


def test_portal_empty_user(client):
    body = client.get("/api/portal/tickets?user_id=nobody").json()
    assert body["tickets"] == []


def test_ticket_detail_has_status_timeline_and_next_step(client):
    """详情接口给自助页提供状态、时间线与下一步说明。"""
    tid = _create_ticket(client, "s_portal_2", "u_1001")
    client.post(f"/api/tickets/{tid}/status", json={"status": "REVIEWING", "operator": "agent1"})
    d = client.get(f"/api/tickets/{tid}?user_id=u_1001").json()
    assert d["status"] == "REVIEWING"
    assert d["status_label"] == "审核中"
    assert d["next_step"]
    assert any(e["payload"].get("to") == "REVIEWING" for e in d["timeline"])


def test_ticket_detail_forbidden_cross_user(client):
    """按工单号查详情时，跨用户读取应 403。"""
    tid = _create_ticket(client, "s_portal_3", "u_owner")
    r = client.get(f"/api/tickets/{tid}?user_id=u_other")
    assert r.status_code == 403


def test_portal_page_served(client):
    r = client.get("/portal")
    assert r.status_code == 200
    assert "自助进度" in r.text
    assert "no-store" in r.headers.get("cache-control", "")


def test_index_has_portal_tab(client):
    r = client.get("/")
    assert 'data-view="portal"' in r.text
    assert "viewPortal" in r.text
