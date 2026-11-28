"""订单列表接口与工具层验收：身份隔离、状态过滤、订单源故障不伪装成查无此单。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agent.tools import ToolError, ToolValidationError


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "orders_api.db"))
    monkeypatch.setenv("AGENT_TODAY", "2026-11-22")
    monkeypatch.setenv("AGENT_LLM_MODE", "rules")
    monkeypatch.setenv("AGENT_MODE", "multi")
    monkeypatch.setenv("AGENT_ORDER_SOURCE", "json")
    monkeypatch.setenv("AGENT_ORDER_SOURCE_PATH", "")
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_orders_api_lists_only_current_user(client):
    response = client.get("/api/orders", params={"user_id": "u_1001"})
    assert response.status_code == 200, response.text
    body = response.json()
    ids = [order["order_id"] for order in body["orders"]]
    assert "O20261115001" in ids
    assert "O20261115003" not in ids  # 属于 u_1002
    assert "O20261112008" not in ids  # 属于 u_1003
    assert all(order["address"].endswith("****") for order in body["orders"])
    assert body["source"]["mode"] == "json"


def test_orders_api_filters_by_status(client):
    response = client.get(
        "/api/orders",
        params={"user_id": "u_1001", "status": "refunded"},
    )
    assert response.status_code == 200
    body = response.json()
    assert [order["order_id"] for order in body["orders"]] == ["O20261114011"]
    assert body["orders"][0]["status_label"] == "已退款"


def test_orders_api_rejects_invalid_status(client):
    response = client.get("/api/orders", params={"user_id": "u_1001", "status": "退款中"})
    assert response.status_code == 400


def test_orders_api_requires_identity_in_demo_mode(client):
    response = client.get("/api/orders")
    assert response.status_code == 401


def test_orders_api_signed_mode_ignores_claimed_user(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "signed.db"))
    monkeypatch.setenv("AGENT_TODAY", "2026-11-22")
    monkeypatch.setenv("AGENT_ORDER_SOURCE", "json")
    monkeypatch.setenv("AGENT_ORDER_SOURCE_PATH", "")
    monkeypatch.setenv("AGENT_AUTH_SECRET", "order-api-secret")
    from app.main import app

    with TestClient(app) as signed_client:
        token_response = signed_client.post("/api/auth/token", json={"user_id": "u_1001"})
        assert token_response.status_code == 200
        token = token_response.json()["token"]
        assert token
        response = signed_client.get(
            "/api/orders",
            params={"user_id": "u_1002"},
            headers={"Authorization": "Bearer " + token},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["user_id"] == "u_1001"
        ids = [order["order_id"] for order in body["orders"]]
        assert "O20261115001" in ids
        assert "O20261115003" not in ids


def test_diagnosis_page_has_order_picker(client):
    response = client.get("/diagnose")
    assert response.status_code == 200
    assert 'id="orderPicker"' in response.text
    assert "/api/orders" in response.text


def test_health_exposes_order_source_detail(client):
    body = client.get("/api/health").json()
    assert body["order_source"] == "json"
    assert body["order_source_detail"]["mode"] == "json"
    assert body["order_source_detail"]["loaded"] > 0


def test_list_my_orders_tool_rejects_caller_supplied_user_id(make_coordinator):
    coordinator = make_coordinator()
    with pytest.raises(ToolValidationError):
        coordinator.tools.call(
            "list_my_orders",
            {"user_id": "u_1002"},
            {"user_id": "u_1001", "session_id": "s", "request_id": "r"},
        )


def test_list_my_orders_tool_filters_by_owner(make_coordinator):
    coordinator = make_coordinator()
    result = coordinator.tools.call(
        "list_my_orders",
        {"limit": 3, "status": "delivered"},
        {"user_id": "u_1001", "session_id": "s", "request_id": "r"},
    )
    assert len(result["orders"]) == 3
    assert all(order["status"] == "delivered" for order in result["orders"])


def test_tool_reports_source_error_instead_of_not_found(make_coordinator, tmp_path):
    coordinator = make_coordinator(
        "broken.db",
        order_source="csv",
        order_source_path=str(tmp_path / "missing.csv"),
    )
    with pytest.raises(ToolError) as exc:
        coordinator.tools.call(
            "query_order",
            {"order_id": "O20261115001"},
            {"user_id": "u_1001", "session_id": "s", "request_id": "r"},
        )
    assert exc.value.code == "order_source_unavailable"
    assert "订单源" in exc.value.message


def test_diagnosis_reports_source_error_as_503(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "broken_diag.db"))
    monkeypatch.setenv("AGENT_TODAY", "2026-11-22")
    monkeypatch.setenv("AGENT_LLM_MODE", "rules")
    monkeypatch.setenv("AGENT_ORDER_SOURCE", "csv")
    monkeypatch.setenv("AGENT_ORDER_SOURCE_PATH", str(tmp_path / "missing.csv"))
    from app.main import app

    with TestClient(app) as broken_client:
        response = broken_client.post(
            "/api/aftersale/diagnose",
            json={"question": "这个耳机能退吗？", "user_id": "u_1001", "order_id": "O20261115001"},
        )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "order_source_unavailable"
