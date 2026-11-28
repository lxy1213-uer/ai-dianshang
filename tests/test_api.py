"""第 2 / 8 周验收：HTTP 接口、参数校验、健康检查。"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("AGENT_TODAY", "2026-11-22")
    monkeypatch.setenv("AGENT_MODE", "multi")
    monkeypatch.setenv("AGENT_LLM_MODE", "rules")
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["orders"] == 13
    assert body["policy_clauses"] == 8
    assert body["llm_mode"] == "rules"


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "售后客服" in response.text


def test_chat_flow(client):
    sid = "s_api"
    first = client.post("/api/chat", json={"session_id": sid, "user_id": "u_1001", "message": "我要退货"}).json()
    assert first["state"] == "AWAIT_ORDER"

    second = client.post("/api/chat", json={"session_id": sid, "user_id": "u_1001", "message": "O20261115001"}).json()
    assert second["state"] == "AWAIT_CONFIRM"
    assert second["citations"]

    third = client.post("/api/chat", json={"session_id": sid, "user_id": "u_1001", "message": "确认"}).json()
    assert third["state"] == "TICKETED"
    assert third["ticket"]["ticket_id"].startswith("RT")
    assert third["agent_mode"] == "multi"
    assert third["trace"]


def test_blank_message_rejected(client):
    response = client.post("/api/chat", json={"session_id": "s1", "user_id": "u_1001", "message": "   "})
    assert response.status_code == 422


def test_too_long_message_rejected(client):
    response = client.post("/api/chat", json={"session_id": "s1", "user_id": "u_1001", "message": "退" * 600})
    assert response.status_code == 422


def test_session_endpoint(client):
    client.post("/api/chat", json={"session_id": "s_sess", "user_id": "u_1001", "message": "我要退货"})
    body = client.get("/api/sessions/s_sess?user_id=u_1001").json()
    assert body["state"] == "AWAIT_ORDER"
    assert body["state_label"] == "等待订单号"


def test_reset_endpoint(client):
    client.post("/api/chat", json={"session_id": "s_reset", "user_id": "u_1001", "message": "我要退货"})
    body = client.post("/api/sessions/s_reset/reset?user_id=u_1001").json()
    assert body["state"] == "IDLE"
    assert client.get("/api/sessions/s_reset?user_id=u_1001").json()["slots"] == {}


def test_tickets_and_trace_endpoints(client):
    sid = "s_list"
    for message in ("我要退货", "O20261115001", "确认"):
        client.post("/api/chat", json={"session_id": sid, "user_id": "u_1001", "message": message})
    tickets = client.get(f"/api/tickets?session_id={sid}&user_id=u_1001").json()["tickets"]
    assert len(tickets) == 1
    events = client.get(f"/api/trace/{sid}?user_id=u_1001").json()["events"]
    assert any(event["kind"] == "turn" for event in events)
    assert any(event["kind"] == "ticket_create_attempt" for event in events)


def test_oidc_style_long_password_accepted(client):
    """OIDC id_token 是长 JWT，登录请求体不能在 128 字符处被 422 拦掉。"""
    token = "eyJhbGciOiJSUzI1NiJ9." + "a" * 800 + ".signature"
    response = client.post("/api/auth/login", json={"username": "agent01", "password": token})
    assert response.status_code == 200


def test_uuid_style_user_id_accepted(client):
    """真实 IdP 的 sub 可能是 UUID，放宽 user_id 长度后仍走正常归属校验。"""
    uid = "u_" + "a" * 34
    response = client.post(
        "/api/chat", json={"session_id": "s_uuid", "user_id": uid, "message": "你好"}
    )
    assert response.status_code == 200
