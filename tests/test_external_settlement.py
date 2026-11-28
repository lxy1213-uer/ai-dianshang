"""退款回调验签 + 对账：纯标准库，零外部依赖。"""

import hashlib
import hmac
import json
import time

import pytest

from app.settlement import (
    CallbackError,
    RefundCallbackVerifier,
    differences_to_dict,
    parse_ledger,
    reconcile,
)

SECRET = "test-callback-secret"


def _signed(payload: dict, *, secret=SECRET, ts=None):
    ts = int(time.time()) if ts is None else ts
    body = json.dumps(payload, separators=(",", ":")).encode()
    sig = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return body, sig, str(ts)


def _verifier(**kwargs):
    return RefundCallbackVerifier(SECRET, now_fn=lambda: 1000000, **kwargs)


def test_verify_accepts_valid_callback():
    v = _verifier()
    body, sig, ts = _signed({
        "refund_id": "RFD-1", "ticket_id": "T-1", "status": "SUCCESS", "amount": 99.5,
    }, ts=1000000)
    payload = v.verify(body, sig, ts)
    assert payload["refund_id"] == "RFD-1"


def test_verify_rejects_tampered_body_and_signature():
    v = _verifier()
    body, sig, ts = _signed({"refund_id": "RFD-1", "ticket_id": "T-1", "status": "SUCCESS"}, ts=1000000)
    with pytest.raises(CallbackError, match="签名"):
        v.verify(body.replace(b"RFD-1", b"RFD-2"), sig, ts)
    with pytest.raises(CallbackError, match="签名"):
        v.verify(body, "bad-signature", ts)


def test_verify_rejects_replay_and_expired_timestamp():
    v = _verifier(tolerance_seconds=60)
    payload = {"refund_id": "RFD-1", "ticket_id": "T-1", "status": "SUCCESS"}
    body, sig, _ = _signed(payload, ts=1000000)
    v.verify(body, sig, "1000000")
    with pytest.raises(CallbackError, match="重复回调"):
        v.verify(body, sig, "1000000")
    late_body, late_sig, _ = _signed({"refund_id": "RFD-2", "ticket_id": "T-2", "status": "SUCCESS"}, ts=999000)
    with pytest.raises(CallbackError, match="重放"):
        v.verify(late_body, late_sig, "999000")


def test_verify_rejects_missing_secret_and_bad_payload():
    v = RefundCallbackVerifier("", now_fn=lambda: 1000000)
    with pytest.raises(CallbackError, match="未配置"):
        v.verify(b"{}", "x", "1000000")
    v = _verifier()
    body, sig, ts = _signed({"refund_id": "", "ticket_id": "T", "status": "SUCCESS"}, ts=1000000)
    with pytest.raises(CallbackError, match="缺少"):
        v.verify(body, sig, ts)


def test_parse_ledger_csv_and_json():
    csv_text = "refund_id,ticket_id,status,amount\nRFD-1,T-1,SUCCESS,99.5\nRFD-2,T-2,FAILED,20\n"
    rows = parse_ledger(csv_text)
    assert len(rows) == 2 and rows[0].amount == 99.5
    json_text = json.dumps([
        {"refund_id": "RFD-1", "ticket_id": "T-1", "status": "success", "amount": 99.5}
    ])
    rows = parse_ledger(json_text)
    assert rows[0].status == "SUCCESS"
    assert parse_ledger("") == []
    with pytest.raises(ValueError):
        parse_ledger("a,b,c\n1,2,3")


def test_reconcile_reports_all_difference_kinds():
    local = [
        {"refund_id": "RFD-1", "ticket_id": "T-1", "status": "SUCCESS", "amount": 100.0},
        {"refund_id": "RFD-2", "ticket_id": "T-2", "status": "SUCCESS", "amount": 50.0},
        {"refund_id": "RFD-3", "ticket_id": "T-3", "status": "SUCCESS", "amount": 30.0},
        {"refund_id": "RFD-4", "ticket_id": "T-4", "status": "SUCCESS", "amount": 20.0},
    ]
    gateway = parse_ledger(json.dumps([
        {"refund_id": "RFD-2", "ticket_id": "T-2", "status": "SUCCESS", "amount": 50.0},
        {"refund_id": "RFD-3", "ticket_id": "T-3", "status": "FAILED", "amount": 30.0},
        {"refund_id": "RFD-4", "ticket_id": "T-4", "status": "SUCCESS", "amount": 21.0},
        {"refund_id": "RFD-5", "ticket_id": "T-5", "status": "SUCCESS", "amount": 10.0},
    ]))
    kinds = {d.kind: d for d in reconcile(local, gateway)}
    assert set(kinds) == {"local_only", "status_mismatch", "amount_mismatch", "gateway_only"}
    assert "禁止自动重打款" in kinds["local_only"].suggested_action
    assert differences_to_dict([kinds["gateway_only"]])[0]["kind"] == "gateway_only"


def test_reconcile_ignores_local_failed_rows():
    local = [{"refund_id": "RFD-1", "ticket_id": "T", "status": "FAILED"}]
    assert reconcile(local, []) == []


# ---------- FastAPI 端点 ----------
def test_callback_endpoint_rejects_unsigned_request():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.post("/api/refund/callback", content=b'{"refund_id":"R"}')
    assert resp.status_code == 401


def test_reconcile_endpoint_outputs_differences(make_coordinator):
    from fastapi.testclient import TestClient
    from app.config import Settings
    import app.main as main

    s = Settings.load(auth_secret="secret", rbac_enabled=True, refund_callback_secret=SECRET)
    c = make_coordinator()
    with TestClient(main.app) as client:
        main.app.state.coordinator = c
        main.app.state.settings = s
        # 用坐席 token 通过 RBAC
        from app.auth import issue_staff_token
        token = issue_staff_token("sup", "supervisor", "secret", 3600)
        body = "refund_id,ticket_id,status,amount\nRFD-X,T-X,SUCCESS,10\n"
        resp = client.post(
            "/api/refund/reconcile", content=body,
            headers={"X-Staff-Token": token, "Content-Type": "text/csv"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["differences"][0]["kind"] == "gateway_only"
