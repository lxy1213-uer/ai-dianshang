"""架构进阶 · 真实交易退款对接验收（沙箱默认 / 幂等 / 权限 / 失败降级）。"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import date
from threading import Thread
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.agent.coordinator import Coordinator
from app.config import Settings
from app.refund import (
    HttpRefundGateway,
    RefundError,
    RefundService,
    SandboxRefundGateway,
    build_refund_gateway,
)


def _prepare(c: Coordinator, sid: str = "s_refund") -> str:
    """建单并沿状态机推进到 RECEIVED（退款唯一合法前置状态）。"""
    for message in ["我要退货", "O20261115001", "确认"]:
        result = c.chat(sid, "u_1001", message)
    ticket_id = result.ticket["ticket_id"]
    for status in ["REVIEWING", "APPROVED", "RECEIVED"]:
        c.tickets.update_status(ticket_id, status, operator="验收")
    return ticket_id


@pytest.fixture()
def coord(tmp_path):
    c = Coordinator(Settings.load(
        llm_mode="rules", db_path=tmp_path / "r.db", frozen_today=date(2026, 11, 22),
    ))
    yield c
    c.close()


# ---------- 1. 网关层 ----------

def test_default_gateway_is_sandbox_and_deterministic():
    g = build_refund_gateway(Settings.load())
    assert isinstance(g, SandboxRefundGateway)
    payload = {"idempotency_key": "refund:T1", "ticket_id": "T1", "order_id": "O1"}
    a = g.refund(payload)
    b = build_refund_gateway(Settings.load()).refund(payload)
    assert a["ok"] and a["gateway"] == "sandbox"
    assert a["refund_id"] == b["refund_id"]  # 确定性，可复现
    assert len(g.calls) == 1


def test_http_gateway_selected_only_when_url_configured():
    g = build_refund_gateway(Settings.load(
        refund_api_url="https://gateway.example/refund",
        refund_api_secret="sk-test", refund_timeout=2.0,
    ))
    assert isinstance(g, HttpRefundGateway)
    assert g.url == "https://gateway.example/refund"
    assert g.secret == "sk-test"


class _FakeResp:
    status = 200

    def __init__(self, body: bytes = b'{"refund_id": "GATEWAY-1"}'):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_http_gateway_sends_hmac_signature_and_idempotency_key():
    payload = {
        "ticket_id": "RT1", "order_id": "O1", "amount": 199.0,
        "currency": "CNY", "idempotency_key": "refund:RT1", "operator": "验收",
    }
    g = HttpRefundGateway("https://gateway.example/refund", "sk-test", 2.0)
    with patch("app.refund.urllib.request.urlopen", return_value=_FakeResp()) as mocked:
        out = g.refund(payload)
    assert out["ok"] and out["refund_id"] == "GATEWAY-1"
    req = mocked.call_args[0][0]
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    expected = hmac.new(b"sk-test", canonical.encode(), hashlib.sha256).hexdigest()
    assert req.get_header("X-refund-signature") == expected
    assert req.get_header("Idempotency-key") == "refund:RT1"


def test_http_gateway_failure_degrades_without_raising():
    g = HttpRefundGateway("https://gateway.example/refund", "sk", 0.1)
    with patch("app.refund.urllib.request.urlopen", side_effect=TimeoutError("timeout")):
        out = g.refund({"idempotency_key": "refund:T1", "ticket_id": "T1"})
    assert out["ok"] is False and out["refund_id"] is None
    assert "TimeoutError" in out["error"]


def test_http_gateway_explicit_failure_response():
    g = HttpRefundGateway("https://gateway.example/refund", "", 1.0)
    body = '{"ok": false, "error": "余额不足"}'.encode("utf-8")
    with patch("app.refund.urllib.request.urlopen", return_value=_FakeResp(body)):
        out = g.refund({"idempotency_key": "refund:T1", "ticket_id": "T1"})
    assert out["ok"] is False and "余额不足" in out["error"]


# ---------- 2. 领域层：绝不重复打款 ----------

def test_refund_succeeds_and_advances_status(coord):
    tid = _prepare(coord)
    out = coord.refunds.execute(tid, operator="验收", amount=199.0)
    assert out["status"] == "SUCCESS"
    assert out["refund_id"].startswith("RFD-")
    assert out["ticket"]["status"] == "REFUNDED"
    assert len(coord.refunds.gateway.calls) == 1
    evs = [e["kind"] for e in coord.db.list_events("s_refund")]
    assert evs.count("refund_executed") == 1


def test_repeated_refund_never_calls_gateway_twice(coord):
    tid = _prepare(coord)
    first = coord.refunds.execute(tid)
    again = coord.refunds.execute(tid, operator="另一个人", amount=999.0)
    assert again["idempotent"] is True
    assert again["refund_id"] == first["refund_id"]
    assert len(coord.refunds.gateway.calls) == 1
    # 第二次调用不得覆盖首次流水（operator / 金额都不变）
    assert coord.db.get_refund(tid)["operator"] == "system"


def test_gateway_failure_does_not_advance_status_and_allows_retry(coord):
    class Failing:
        gateway = "failing"

        def __init__(self):
            self.calls = []

        def refund(self, payload):
            self.calls.append(payload)
            return {"ok": False, "refund_id": None, "error": "网关超时", "gateway": "failing"}

    failing = Failing()
    coord.refunds.gateway = failing
    tid = _prepare(coord)
    with pytest.raises(RefundError) as exc:
        coord.refunds.execute(tid)
    assert exc.value.code == "gateway_failed"
    assert coord.db.get_ticket(tid)["status"] == "RECEIVED"  # 不推进状态
    assert coord.db.get_refund(tid)["status"] == "FAILED"

    coord.refunds.gateway = SandboxRefundGateway()
    out = coord.refunds.execute(tid)  # FAILED 可被重试抢占
    assert out["status"] == "SUCCESS"
    assert coord.db.get_ticket(tid)["status"] == "REFUNDED"


def test_invalid_state_is_rejected(coord):
    result = coord.chat("s_new", "u_1001", "我要退货")
    coord.chat("s_new", "u_1001", "O20261115001")
    last = coord.chat("s_new", "u_1001", "确认")
    tid = last.ticket["ticket_id"]
    with pytest.raises(RefundError) as exc:
        coord.refunds.execute(tid)
    assert exc.value.code == "invalid_state"
    assert coord.refunds.status(tid) is None  # 连占位都不做


def test_refunded_without_success_record_is_refused(coord):
    """状态与流水不一致时宁可人工介入，绝不再打一次款。"""
    tid = _prepare(coord)
    coord.tickets.update_status(tid, "REFUNDED", operator="冒名")
    with pytest.raises(RefundError) as exc:
        coord.refunds.execute(tid)
    assert exc.value.code == "inconsistent"
    assert len(coord.refunds.gateway.calls) == 0


def test_concurrent_refunds_call_gateway_only_once(coord):
    tid = _prepare(coord)
    outcomes: list[dict] = []
    errors: list[str] = []

    def worker():
        try:
            outcomes.append(coord.refunds.execute(tid))
        except RefundError as exc:
            errors.append(exc.code)

    t1, t2 = Thread(target=worker), Thread(target=worker)
    t1.start(); t2.start(); t1.join(); t2.join()
    assert len(coord.refunds.gateway.calls) == 1, "并发下发生了重复打款"
    assert len(outcomes) >= 1
    assert all(o["refund_id"] == outcomes[0]["refund_id"] for o in outcomes)


# ---------- 3. API 层 ----------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("AGENT_TODAY", "2026-11-22")
    monkeypatch.setenv("AGENT_LLM_MODE", "rules")
    monkeypatch.setenv("AGENT_RBAC_ENABLED", "0")
    monkeypatch.setenv("AGENT_RATE_LIMIT_PER_MIN", "0")
    monkeypatch.setenv("AGENT_STAFF_ACCOUNTS", "")
    from app.main import app
    with TestClient(app) as tc:
        yield tc


def _api_prepare(client, sid="s_api_refund"):
    for message in ["我要退货", "O20261115001", "确认"]:
        result = client.post("/api/chat", json={"session_id": sid, "user_id": "u_1001", "message": message})
    tid = result.json()["ticket"]["ticket_id"]
    for status in ["REVIEWING", "APPROVED", "RECEIVED"]:
        client.post(f"/api/tickets/{tid}/status", json={"status": status, "operator": "验收"})
    return tid


def test_refund_endpoint_demo_success_and_idempotent(client):
    tid = _api_prepare(client)
    r = client.post(f"/api/tickets/{tid}/refund", json={"amount": 199.0, "operator": "验收"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "SUCCESS" and body["gateway"] == "sandbox"
    assert body["ticket"]["status"] == "REFUNDED"
    r2 = client.post(f"/api/tickets/{tid}/refund", json={"amount": 199.0})
    assert r2.json()["idempotent"] is True
    st = client.get(f"/api/tickets/{tid}/refund").json()
    assert st["refund"]["status"] == "SUCCESS"


def test_refund_endpoint_rejects_wrong_state(client):
    for message in ["我要退货", "O20261115001", "确认"]:
        result = client.post("/api/chat", json={"session_id": "s_bad", "user_id": "u_1001", "message": message})
    tid = result.json()["ticket"]["ticket_id"]
    r = client.post(f"/api/tickets/{tid}/refund", json={})
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "invalid_state"


def test_refund_endpoint_not_found(client):
    r = client.post("/api/tickets/NO-SUCH-TICKET/refund", json={})
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "not_found"


def test_refund_requires_supervisor_in_strict_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "strict.db"))
    monkeypatch.setenv("AGENT_TODAY", "2026-11-22")
    monkeypatch.setenv("AGENT_LLM_MODE", "rules")
    monkeypatch.setenv("AGENT_RBAC_ENABLED", "1")
    monkeypatch.setenv("AGENT_AUTH_SECRET", "test-secret")
    monkeypatch.setenv("AGENT_STAFF_ACCOUNTS", "ag:pw:agent;adm:pw:admin")
    monkeypatch.setenv("AGENT_RATE_LIMIT_PER_MIN", "0")
    from app.auth import issue_staff_token, issue_token
    from app.config import Settings
    from app.main import app
    s = Settings.load()
    agent_token = issue_staff_token("ag", "agent", s.auth_secret)
    admin_token = issue_staff_token("adm", "admin", s.auth_secret)
    user_token = issue_token("u_1001", s.auth_secret)
    with TestClient(app) as tc:
        headers = {"Authorization": "Bearer " + user_token}
        for message in ["我要退货", "O20261115001", "确认"]:
            result = tc.post("/api/chat", json={"session_id": "s_strict_refund", "user_id": "u_1001", "message": message}, headers=headers)
        tid = result.json()["ticket"]["ticket_id"]
        for status in ["REVIEWING", "APPROVED", "RECEIVED"]:
            tc.post(f"/api/tickets/{tid}/status", json={"status": status},
                    headers={"X-Staff-Token": agent_token})
        r = tc.post(f"/api/tickets/{tid}/refund", json={},
                    headers={"X-Staff-Token": agent_token})
        assert r.status_code == 403, r.text
        r2 = tc.post(f"/api/tickets/{tid}/refund", json={},
                     headers={"X-Staff-Token": admin_token})
        assert r2.status_code == 200, r2.text
        assert r2.json()["ticket"]["status"] == "REFUNDED"


def test_describe_exposes_refund_gateway_without_secret():
    d = Settings.load(refund_api_url="https://gateway.example/refund", refund_api_secret="sk-secret").describe()
    assert d["refund_gateway"] == "http"
    assert "sk-secret" not in json.dumps(d, ensure_ascii=False)
