"""安全底座验收：坐席 RBAC、登录签发、token 隔离、接口限流、demo 兼容。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.auth import issue_staff_token, issue_token, verify_staff_token, verify_token
from app.authz import AuthzError, Permission, ROLE_RANK, Role, parse_accounts, parse_role
from app.config import Settings

SECRET = "test-secret-for-rbac"
ACCOUNTS = (
    "viewer01:pwd1:viewer:u_1001;"
    "agent01:pwd2:agent:u_1001;"
    "boss01:pwd3:supervisor:u_1001;"
    "admin01:pwd4:admin:u_1001"
)


def _clean_env(monkeypatch):
    for key in (
        "AGENT_AUTH_SECRET", "AGENT_RBAC_ENABLED", "AGENT_STAFF_ACCOUNTS",
        "AGENT_RATE_LIMIT_PER_MIN", "AGENT_RATE_LIMIT_CHAT", "AGENT_RATE_LIMIT_AUTH",
        "AGENT_RATE_LIMIT_SWITCH", "STAFF_TTL_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture()
def demo_client(tmp_path, monkeypatch):
    """默认 demo 模式：不配 secret / 不开 RBAC / 不限流。"""
    _clean_env(monkeypatch)
    monkeypatch.setenv("DB_PATH", str(tmp_path / "demo.db"))
    monkeypatch.setenv("AGENT_TODAY", "2026-11-22")
    from app.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture()
def strict_client(tmp_path, monkeypatch):
    """strict 模式：配置密钥 + 账号表 + 开启 RBAC。"""
    _clean_env(monkeypatch)
    monkeypatch.setenv("DB_PATH", str(tmp_path / "strict.db"))
    monkeypatch.setenv("AGENT_TODAY", "2026-11-22")
    monkeypatch.setenv("AGENT_AUTH_SECRET", SECRET)
    monkeypatch.setenv("AGENT_RBAC_ENABLED", "1")
    monkeypatch.setenv("AGENT_STAFF_ACCOUNTS", ACCOUNTS)
    from app.main import app

    with TestClient(app) as client:
        yield client


def _staff(name: str, role: str) -> dict:
    return {"X-Staff-Token": issue_staff_token(name, role, SECRET)}


# ---------- 纯函数：角色与账号解析 ----------
def test_role_rank_order():
    assert ROLE_RANK[Role.VIEWER] < ROLE_RANK[Role.AGENT] < ROLE_RANK[Role.SUPERVISOR] < ROLE_RANK[Role.ADMIN]


def test_parse_role_is_case_insensitive_and_rejects_unknown():
    assert parse_role(" Agent ") == Role.AGENT
    assert parse_role("superuser") is None
    assert parse_role(None) is None


def test_parse_accounts_skips_invalid_entries():
    accounts = parse_accounts("a:p:agent;bad-entry;;b:p:unknown_role;c:p:supervisor:u_9")
    assert set(accounts) == {"a", "c"}
    assert accounts["c"]["user_id"] == "u_9"
    assert accounts["c"]["role"] == "supervisor"


# ---------- token 隔离 ----------
def test_user_and_staff_tokens_cannot_impersonate_each_other():
    user = issue_token("u_1001", SECRET)
    staff = issue_staff_token("agent01", "agent", SECRET)
    assert verify_token(user, SECRET) == "u_1001"
    assert verify_token(staff, SECRET) is None
    assert verify_staff_token(staff, SECRET) == {"sub": "agent01", "role": "agent"}
    assert verify_staff_token(user, SECRET) is None


def test_tampered_token_is_rejected():
    staff = issue_staff_token("agent01", "agent", SECRET)
    assert verify_staff_token(staff + "x", SECRET) is None
    assert verify_staff_token(staff, "wrong-secret") is None


# ---------- demo 兼容 ----------
def test_demo_mode_allows_desk_actions_without_token(demo_client):
    assert demo_client.get("/api/desk/queue").status_code == 200
    assert demo_client.get("/api/metrics/funnel").status_code == 200
    # 工单详情既有归属校验：demo 模式仍需显式 user_id（这是原有行为，不是 RBAC 拦截）
    assert demo_client.get("/api/tickets/T-not-exist", params={"user_id": "u_1001"}).status_code == 404


def test_demo_login_needs_no_password(demo_client):
    r = demo_client.post("/api/auth/login", json={"username": "anyone", "password": "x"})
    assert r.status_code == 200
    assert r.json()["auth_mode"] == "demo"
    assert r.json()["role"] == "admin"


# ---------- strict 模式：登录与鉴权 ----------
def test_strict_login_and_role_returned(strict_client):
    bad = strict_client.post("/api/auth/login", json={"username": "agent01", "password": "wrong"})
    assert bad.status_code == 401
    ok = strict_client.post("/api/auth/login", json={"username": "boss01", "password": "pwd3"})
    assert ok.status_code == 200
    assert ok.json()["role"] == "supervisor"
    assert ok.json()["staff_token"]


def test_strict_mode_rejects_missing_staff_token(strict_client):
    assert strict_client.get("/api/desk/queue").status_code == 401
    assert strict_client.post("/api/models/switch", json={"role": "main", "model_id": "x"}).status_code == 401


def test_viewer_can_read_but_not_mutate(strict_client):
    h = _staff("viewer01", "viewer")
    assert strict_client.get("/api/desk/queue", headers=h).status_code == 200
    r = strict_client.post("/api/desk/queue/no-such/claim", json={"assignee": "viewer01"}, headers=h)
    assert r.status_code == 403
    assert r.json()["code"] == "forbidden"
    assert strict_client.post(
        "/api/models/switch", json={"role": "main", "model_id": "x"}, headers=h
    ).status_code == 403


def test_agent_can_claim_reply_status_but_not_resolve_or_switch(strict_client):
    h = _staff("agent01", "agent")
    assert strict_client.post("/api/desk/queue/no-such/claim", json={"assignee": "fake"}, headers=h).status_code in (404, 409)
    assert strict_client.post("/api/desk/queue/no-such/reply", json={"assignee": "fake", "text": "x"}, headers=h).status_code in (404, 409)
    assert strict_client.post(
        "/api/tickets/T-none/status", json={"status": "REVIEWING", "operator": "fake"}, headers=h
    ).status_code == 404
    assert strict_client.post("/api/desk/queue/no-such/resolve", headers=h).status_code == 403
    assert strict_client.post(
        "/api/models/switch", json={"role": "main", "model_id": "x"}, headers=h
    ).status_code == 403


def test_supervisor_can_resolve_but_not_switch(strict_client):
    h = _staff("boss01", "supervisor")
    assert strict_client.post("/api/desk/queue/no-such/resolve", headers=h).status_code == 404  # 有权限，资源不存在
    assert strict_client.post(
        "/api/models/switch", json={"role": "main", "model_id": "x"}, headers=h
    ).status_code == 403


def test_admin_can_switch_model(strict_client):
    h = _staff("admin01", "admin")
    r = strict_client.post("/api/models/switch", json={"role": "main", "model_id": "no-such"}, headers=h)
    assert r.status_code == 400  # 过了权限校验，因模型不存在才 400


def test_forged_header_and_body_cannot_escalate(strict_client):
    # 请求头/请求体里伪造坐席名与角色都无效：strict 模式只信 token
    h = _staff("viewer01", "viewer")
    h["X-Agent-Name"] = "admin01"
    r = strict_client.post("/api/models/switch", json={"role": "main", "model_id": "x"}, headers=h)
    assert r.status_code == 403


def test_user_token_cannot_be_used_as_staff_token(strict_client):
    user = issue_token("u_1001", SECRET)
    r = strict_client.get("/api/desk/queue", headers={"X-Staff-Token": user})
    assert r.status_code == 401


def test_health_exposes_strict_state(strict_client):
    assert strict_client.get("/api/health").json()["rbac_enabled"] is True


def test_health_exposes_demo_state(demo_client):
    assert demo_client.get("/api/health").json()["rbac_enabled"] is False


# ---------- 限流 ----------
def test_rate_limit_returns_429_with_retry_after(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("DB_PATH", str(tmp_path / "rl.db"))
    monkeypatch.setenv("AGENT_TODAY", "2026-11-22")
    monkeypatch.setenv("AGENT_RATE_LIMIT_PER_MIN", "2")
    from app.main import app

    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200  # health 不限流
        assert client.get("/api/desk/queue").status_code == 200
        assert client.get("/api/desk/queue").status_code == 200
        third = client.get("/api/desk/queue")
        assert third.status_code == 429
        assert int(third.headers["Retry-After"]) >= 1


def test_rate_limit_zero_disables(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("DB_PATH", str(tmp_path / "rl0.db"))
    monkeypatch.setenv("AGENT_RATE_LIMIT_PER_MIN", "0")
    from app.main import app

    with TestClient(app) as client:
        for _ in range(8):
            assert client.get("/api/desk/queue").status_code == 200


def test_endpoint_specific_limit_overrides_global(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("DB_PATH", str(tmp_path / "rl1.db"))
    monkeypatch.setenv("AGENT_RATE_LIMIT_PER_MIN", "100")
    monkeypatch.setenv("AGENT_RATE_LIMIT_CHAT", "1")
    from app.main import app

    with TestClient(app) as client:
        body = {"session_id": "rl-1", "user_id": "u_1001", "message": "你好"}
        assert client.post("/api/chat", json=body).status_code == 200
        assert client.post("/api/chat", json=body).status_code == 429
        # 其它接口不受 chat 专属上限影响
        assert client.get("/api/desk/queue").status_code == 200


# ---------- 配置校验 ----------
def test_security_settings_parse_from_env(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("AGENT_RBAC_ENABLED", "yes")
    monkeypatch.setenv("AGENT_STAFF_ACCOUNTS", "a:p:agent")
    monkeypatch.setenv("AGENT_RATE_LIMIT_PER_MIN", "7")
    monkeypatch.setenv("AGENT_AUTH_SECRET", SECRET)
    s = Settings.load()
    assert s.rbac_enabled is True
    assert s.rate_limit_per_min == 7
    assert s.signed_auth_enabled is True
