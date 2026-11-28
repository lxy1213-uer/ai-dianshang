"""P0 缺陷回归：否定误建单 / 令牌时效与资格复核 / 查询归属 / 并发竞态 / CREATING 自愈。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.agent.coordinator import Coordinator
from app.config import Settings
from tests.conftest import chat

NEGATIVE_CONFIRMS = ["不确认", "我不确定", "还没确认好", "先别确认", "等等，我再想想"]


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "p0_api.db"))
    monkeypatch.setenv("AGENT_TODAY", "2026-11-22")
    monkeypatch.delenv("AGENT_AUTH_SECRET", raising=False)
    from app.main import app

    with TestClient(app) as client:
        yield client


def seed_chat(client, sid: str = "s_acl", user: str = "u_1001"):
    for message in ("我要退货", "O20261115001", "确认"):
        client.post("/api/chat", json={"session_id": sid, "user_id": user, "message": message})


# ---------- 1) 否定 / 犹豫表达不能再误建单 ----------
@pytest.mark.parametrize("idx,message", list(enumerate(NEGATIVE_CONFIRMS)))
def test_negative_confirm_does_not_create_ticket(make_coordinator, idx, message):
    c = make_coordinator(f"neg_{idx}.db")
    chat(c, "s_neg", "我要退货", "O20261115001")
    result = c.chat("s_neg", "u_1001", message)
    assert result.state == "AWAIT_CONFIRM"
    assert result.ticket is None
    assert c.db.count_tickets() == 0
    assert "先不提交" in result.reply


def test_plain_confirm_still_works(make_coordinator):
    c = make_coordinator("pos.db")
    result = chat(c, "s_pos", "我要退货", "O20261115001", "确认")
    assert result.state == "TICKETED"
    assert result.ticket["ticket_id"].startswith("RT")
    assert c.db.count_tickets() == 1


# ---------- 2) 确认令牌 TTL 与建单前复核资格 ----------
def test_expired_confirm_is_rejected(make_coordinator):
    c = make_coordinator("ttl.db")
    chat(c, "s_ttl", "我要退货", "O20261115001")
    session = c.db.get_session("s_ttl")
    slots = dict(session["slots"])
    slots["confirm_issued_at"] = (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    c.db.save_session("s_ttl", "u_1001", session["state"], slots)

    result = c.chat("s_ttl", "u_1001", "确认")
    assert result.state == "AWAIT_ORDER"
    assert c.db.count_tickets() == 0
    assert any(e["kind"] == "confirm_expired" for e in c.db.list_events("s_ttl"))


def test_eligibility_is_rechecked_before_creating(make_coordinator):
    c = make_coordinator("recheck.db")
    chat(c, "s_re", "我要退货", "O20261115001")
    # 核单后把业务日期推到超期：确认时必须重新判资格，不能照旧建单
    c.settings.frozen_today = date(2026, 12, 1)
    result = c.chat("s_re", "u_1001", "确认")
    assert result.state in ("HUMAN", "AWAIT_ORDER")
    assert c.db.count_tickets() == 0
    assert any(e["kind"] == "eligibility_changed" for e in c.db.list_events("s_re"))


# ---------- 3) 查询接口归属校验 ----------
def test_query_endpoints_require_identity(api_client):
    seed_chat(api_client)
    assert api_client.get("/api/tickets?session_id=s_acl").status_code == 401
    assert api_client.get("/api/trace/s_acl").status_code == 401
    assert api_client.get("/api/sessions/s_acl").status_code == 401
    assert api_client.post("/api/sessions/s_acl/reset").status_code == 401


def test_cross_user_read_is_forbidden(api_client):
    seed_chat(api_client)
    assert api_client.get("/api/tickets?session_id=s_acl&user_id=u_1002").status_code == 403
    assert api_client.get("/api/trace/s_acl?user_id=u_1002").status_code == 403
    assert api_client.get("/api/sessions/s_acl?user_id=u_1002").status_code == 403
    assert api_client.post("/api/sessions/s_acl/reset?user_id=u_1002").status_code == 403
    ok = api_client.get("/api/tickets?session_id=s_acl&user_id=u_1001")
    assert ok.status_code == 200
    assert len(ok.json()["tickets"]) == 1


def test_signed_mode_only_trusts_token(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "signed.db"))
    monkeypatch.setenv("AGENT_TODAY", "2026-11-22")
    monkeypatch.setenv("AGENT_AUTH_SECRET", "test-secret")
    from app.main import app

    with TestClient(app) as client:
        token = client.post("/api/auth/token", json={"user_id": "u_1001"}).json()["token"]
        assert token
        assert client.get("/api/health").json()["auth_mode"] == "signed"
        headers = {"Authorization": "Bearer " + token}

        # body 里伪造的 user_id 必须被忽略
        res = client.post(
            "/api/chat",
            json={"session_id": "s_signed", "user_id": "u_9999", "message": "我要退货"},
            headers=headers,
        )
        assert res.status_code == 200
        view = client.get("/api/sessions/s_signed?user_id=u_9999", headers=headers).json()
        assert view["user_id"] == "u_1001"

        # 无 token / 被篡改的 token 一律拒绝
        assert client.post(
            "/api/chat", json={"session_id": "s_signed", "user_id": "u_1001", "message": "你好"}
        ).status_code == 401
        assert client.get(
            "/api/sessions/s_signed", headers={"Authorization": "Bearer forged-token"}
        ).status_code == 401


def test_token_signature_and_expiry():
    from app.auth import issue_token, verify_token

    token = issue_token("u_1001", "secret", 60)
    assert verify_token(token, "secret") == "u_1001"
    assert verify_token(token, "other-secret") is None
    assert verify_token(token[:-2] + "xx", "secret") is None
    assert verify_token(issue_token("u_1001", "secret", -1), "secret") is None


# ---------- 4) 会话锁：并发确认必须串行、结果一致 ----------
def test_concurrent_confirm_is_serialized(make_coordinator):
    c = make_coordinator("conc.db")
    chat(c, "s_conc", "我要退货", "O20261115001")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(c.chat, "s_conc", "u_1001", "确认") for _ in range(2)]
        results = [f.result() for f in futures]

    assert all(r.state == "TICKETED" for r in results)
    assert len({r.ticket["ticket_id"] for r in results if r.ticket}) == 1
    assert len(c.db.list_tickets("s_conc")) == 1
    # 第二个请求看到了已建单状态，说明「读—改—写」被串行化了
    assert any(r.meta.get("duplicate_confirm") for r in results)


def test_concurrent_confirm_with_response_loss_stays_consistent(make_coordinator):
    c = make_coordinator("conc_loss.db", simulate_response_loss=True)
    chat(c, "s_cl", "我要退货", "O20261115001")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(c.chat, "s_cl", "u_1001", "确认") for _ in range(2)]
        results = [f.result() for f in futures]

    assert len(c.db.list_tickets("s_cl")) == 1
    assert all(r.state in ("AWAIT_CONFIRM", "TICKETED") for r in results)
    ticket_ids = {r.ticket["ticket_id"] for r in results if r.ticket}
    assert len(ticket_ids) <= 1


# ---------- 5) CREATING 自愈 ----------
def test_creating_session_recovers_on_restart(make_coordinator):
    first = make_coordinator("crash.db")
    chat(first, "s_crash", "我要退货", "O20261115001")
    session = first.db.get_session("s_crash")
    first.db.save_session("s_crash", "u_1001", "CREATING", session["slots"])
    assert first.db.get_session("s_crash")["state"] == "CREATING"

    second = make_coordinator("crash.db")  # 模拟进程重启，指向同一数据库
    assert second.db.get_session("s_crash")["state"] == "AWAIT_CONFIRM"
    result = chat(second, "s_crash", "确认")
    assert result.state == "TICKETED"
    assert len(second.db.list_tickets("s_crash")) == 1


def test_creating_runtime_does_not_deadlock(make_coordinator):
    c = make_coordinator("crash_rt.db")
    chat(c, "s_rt", "我要退货", "O20261115001")
    session = c.db.get_session("s_rt")
    c.db.save_session("s_rt", "u_1001", "CREATING", session["slots"])

    result = c.chat("s_rt", "u_1001", "确认")
    assert result.state == "TICKETED"
    assert c.db.count_tickets() == 1
