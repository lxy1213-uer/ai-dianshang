"""应用级联调：真实 FastAPI 服务 + 本地替身，走真实 HTTP 往返。

与 test_external_integration.py 的区别：那一项验证「客户端代码 + 协议」，
这一项验证「应用端点」在真实网络下能否与替身协作：

1. OIDC 登录：/api/auth/login 用真实 id_token（替身签发）换出会话 token；
2. 支付回调：支付替身主动 POST 签名回调到真实 /api/refund/callback，
   - 合法签名 → 200 + requires_manual_review（只记录，绝不自动打款/改状态）；
   - 篡改签名 → 401；
   - 重放 → 401；
3. 对账：真实 /api/refund/reconcile 上传替身 ledger.json，输出差异且不改状态。

用 uvicorn 在线程内起真实服务（TestClient 不经过真实 socket，故不用）。
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

STUBS = Path(__file__).resolve().parents[1] / "scripts" / "local_stubs"
sys.path.insert(0, str(STUBS))

import integration_stubs as st  # noqa: E402

from app.config import Settings  # noqa: E402


def _free_port() -> int:
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


def _http(url: str, payload: dict | None = None, headers: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    final_headers = {"Content-Type": "application/json", **(headers or {})} if data else (headers or {})
    req = urllib.request.Request(url, data=data, headers=final_headers,
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


@pytest.fixture()
def live_app(tmp_path, monkeypatch):
    """在线程里起真实 uvicorn 服务，应用配置指向本地替身。"""
    import uvicorn
    from app.main import app

    stub, stub_port, _ = st.start_http_stub()
    base = f"http://127.0.0.1:{stub_port}"
    port = _free_port()

    monkeypatch.setenv("DB_PATH", str(tmp_path / "app_integration.db"))
    monkeypatch.setenv("AGENT_TODAY", "2026-11-22")
    monkeypatch.setenv("AGENT_LLM_MODE", "rules")
    # strict 模式：OIDC 身份源才会真正参与登录校验（demo 模式直接放行，不碰 IdP）
    monkeypatch.setenv("AGENT_RBAC_ENABLED", "1")
    monkeypatch.setenv("AGENT_RATE_LIMIT_PER_MIN", "0")
    monkeypatch.setenv("AGENT_AUTH_SECRET", "integration-secret")
    monkeypatch.setenv("AGENT_STAFF_ACCOUNTS", "")
    monkeypatch.setenv("AGENT_IDP_PROVIDER", "oidc")
    monkeypatch.setenv("AGENT_OIDC_ISSUER", st.ISSUER)
    monkeypatch.setenv("AGENT_OIDC_AUDIENCE", st.AUDIENCE)
    monkeypatch.setenv("AGENT_OIDC_JWKS_URL", base + "/jwks")
    monkeypatch.setenv("AGENT_REFUND_CALLBACK_SECRET", st.CALLBACK_SECRET)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}", base
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        stub.shutdown(); stub.server_close()


def test_live_app_oidc_login_with_real_id_token(live_app):
    app_url, stub_url = live_app
    body = json.dumps({"sub": "u_live", "role": "supervisor"}).encode()
    req = urllib.request.Request(
        stub_url + "/token", data=body, headers={"Content-Type": "application/json"}
    )
    token = json.load(urllib.request.urlopen(req))["access_token"]
    # 真实应用端点：id_token 经真实 JWKS HTTP 拉取 + RS256 验签后签发会话 token
    # username 由 OIDC provider 忽略（身份取自 token），但请求体约束要求非空，用占位值
    status, data = _http(app_url + "/api/auth/login", {"username": "oidc", "password": token})
    assert status == 200
    assert data["auth_mode"] == "strict"
    assert data["role"] == "supervisor"
    assert data["token"]


def test_live_app_rejects_tampered_id_token(live_app):
    app_url, _ = live_app
    token = st.issue_id_token()
    parts = token.split(".")
    tampered = parts[0] + "." + parts[1] + "." + ("0" * len(parts[2]))
    status, _ = _http(app_url + "/api/auth/login", {"username": "oidc", "password": tampered})
    assert status == 401


def test_live_app_payment_callback_verification_and_no_auto_state_change(live_app):
    app_url, _ = live_app
    status, _ = st.send_signed_callback(
        app_url + "/api/refund/callback",
        {"refund_id": "RFD-LIVE", "ticket_id": "T-LIVE", "status": "SUCCESS", "amount": 5.0},
    )
    assert status == 200

    # 篡改签名 → 401
    status, _ = st.send_signed_callback(
        app_url + "/api/refund/callback",
        {"refund_id": "RFD-BAD", "ticket_id": "T-BAD", "status": "SUCCESS", "amount": 5.0},
        tamper="signature",
    )
    assert status == 401

    # 同一 refund_id 重放 → 401（幂等去重）
    status, _ = st.send_signed_callback(
        app_url + "/api/refund/callback",
        {"refund_id": "RFD-LIVE", "ticket_id": "T-LIVE", "status": "SUCCESS", "amount": 5.0},
    )
    assert status == 401

    # 回调验证通过只写事件，没有真实工单/退款状态被自动推进
    status, health = _http(app_url + "/api/health")
    assert status == 200 and health["tickets"] == 0


def test_live_app_reconcile_from_gateway_ledger(live_app):
    """真实 /api/refund/reconcile 消费替身 ledger，输出差异不改状态。"""
    app_url, stub_url = live_app
    # strict 模式对账需主管令牌：先用真实 OIDC 登录换 staff_token
    body = json.dumps({"sub": "u_recon", "role": "supervisor"}).encode()
    req = urllib.request.Request(
        stub_url + "/token", data=body, headers={"Content-Type": "application/json"}
    )
    id_token = json.load(urllib.request.urlopen(req))["access_token"]
    _, logged = _http(app_url + "/api/auth/login", {"username": "oidc", "password": id_token})
    staff_token = logged["staff_token"]
    assert staff_token

    # 先在替身产生一笔退款流水
    from app.refund import HttpRefundGateway
    gw = HttpRefundGateway(stub_url + "/refunds", secret=st.API_SECRET)
    paid = gw.refund({"ticket_id": "T-REC", "amount": 7.5,
                      "idempotency_key": "refund:T-REC"})
    ledger = urllib.request.urlopen(stub_url + "/ledger.json").read()

    req = urllib.request.Request(
        app_url + "/api/refund/reconcile?format=json", data=ledger, method="POST",
        headers={"Content-Type": "application/json", "X-Staff-Token": staff_token},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode())
    assert data["gateway_rows"] >= 1
    # 本地没有这笔流水 → 网关单边差异，且只给建议
    kinds = [d["kind"] for d in data["differences"]]
    assert "gateway_only" in kinds
    assert all("人工" in d["suggested_action"] for d in data["differences"])
