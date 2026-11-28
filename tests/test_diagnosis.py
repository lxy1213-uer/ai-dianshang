"""售后诊断产品层验收：证据、边界、执行与幂等。"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "diagnosis.db"))
    monkeypatch.setenv("AGENT_TODAY", "2026-11-22")
    monkeypatch.setenv("AGENT_MODE", "multi")
    monkeypatch.setenv("AGENT_LLM_MODE", "rules")
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def _diagnose(client, question, user_id="u_1001", order_id=None):
    body = {"question": question, "user_id": user_id}
    if order_id:
        body["order_id"] = order_id
    response = client.post("/api/aftersale/diagnose", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_diagnosis_page(client):
    response = client.get("/diagnose")
    assert response.status_code == 200
    assert "售后诊断" in response.text
    assert "/api/aftersale/diagnose" in response.text


def test_eligible_diagnosis_has_facts_evidence_and_boundaries(client):
    data = _diagnose(client, "这个耳机签收 7 天了，还能退吗？", order_id="O20261115001")
    assert data["decision_id"].startswith("DG")
    assert data["verdict"]["code"] == "ELIGIBLE"
    assert data["order"]["order_id"] == "O20261115001"
    assert any(f["source"] == "订单记录" for f in data["facts"])
    assert any(c["clause_id"] == "R-01" for c in data["evidence"])
    assert all(c["snippet"] and c["source"] for c in data["evidence"])
    assert len(data["boundaries"]) >= 3
    assert data["next_steps"]
    assert "confirm_token" not in str(data)
    create = next(a for a in data["actions"] if a["id"] == "create_ticket")
    assert create["enabled"] is True


def test_quality_reason_requires_defect_evidence(client):
    data = _diagnose(client, "刻歪了，能退吗？", order_id="O20261118004")
    assert data["reason"] == "quality"
    assert "刻歪" in data["reason_evidence"]["evidence"]
    assert any(c["clause_id"] == "R-02" for c in data["evidence"])
    assert data["verdict"]["code"] == "ELIGIBLE"


def test_matched_order_is_read_only_until_explicit_order_id(client):
    data = _diagnose(client, "定制刻字保温杯不想要了，能退吗？")
    assert data["verdict"]["code"] == "HUMAN_REVIEW"
    assert data["matched_token"]
    assert data["order"]["order_id"] == "O20261118004"
    create = next(a for a in data["actions"] if a["id"] == "create_ticket")
    assert create["enabled"] is False
    assert any("显式" in gap for gap in data["gaps"])


def test_no_order_and_no_match_asks_for_order(client):
    data = _diagnose(client, "能退吗？")
    assert data["verdict"]["code"] == "NEED_ORDER"
    assert data["order"] is None
    assert any("订单号" in gap for gap in data["gaps"])


def test_refunded_order_is_human_review_not_executable(client):
    data = _diagnose(client, "已经退款了，怎么又让我退货？", order_id="O20261114011")
    assert data["verdict"]["code"] == "HUMAN_REVIEW"
    assert data["verdict"]["requires_human"] is True
    create = next(a for a in data["actions"] if a["id"] == "create_ticket")
    assert create["enabled"] is False


def test_foreign_order_is_denied(client):
    response = client.post(
        "/api/aftersale/diagnose",
        json={
            "question": "这个内衣能退吗？",
            "user_id": "u_1001",
            "order_id": "O20261115003",
        },
    )
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "forbidden"


def test_execute_requires_exact_confirmation_and_is_idempotent(client):
    data = _diagnose(client, "这个耳机签收 7 天了，还能退吗？", order_id="O20261115001")
    decision_id = data["decision_id"]

    bad = client.post(
        "/api/aftersale/execute",
        json={
            "decision_id": decision_id,
            "user_id": "u_1001",
            "action": "create_ticket",
            "confirm_text": "嗯",
        },
    )
    assert bad.status_code == 400
    assert bad.json()["detail"]["code"] == "confirmation_required"

    good = client.post(
        "/api/aftersale/execute",
        json={
            "decision_id": decision_id,
            "user_id": "u_1001",
            "action": "create_ticket",
            "confirm_text": "确认创建工单",
        },
    )
    assert good.status_code == 200, good.text
    body = good.json()
    assert body["ticket"]["ticket_id"].startswith("RT")
    assert body["state"] == "TICKETED"

    again = client.post(
        "/api/aftersale/execute",
        json={
            "decision_id": decision_id,
            "user_id": "u_1001",
            "action": "create_ticket",
            "confirm_text": "确认创建工单",
        },
    )
    assert again.status_code == 200
    assert again.json()["duplicate"] is True
    assert again.json()["ticket"]["ticket_id"] == body["ticket"]["ticket_id"]


def test_other_user_cannot_execute_decision(client):
    data = _diagnose(client, "这个耳机签收 7 天了，还能退吗？", order_id="O20261115001")
    response = client.post(
        "/api/aftersale/execute",
        json={
            "decision_id": data["decision_id"],
            "user_id": "u_1002",
            "action": "create_ticket",
            "confirm_text": "确认创建工单",
        },
    )
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "forbidden"


def test_handoff_action_is_executable(client):
    data = _diagnose(client, "定制刻字保温杯不想要了，能退吗？", order_id="O20261118004")
    response = client.post(
        "/api/aftersale/execute",
        json={
            "decision_id": data["decision_id"],
            "user_id": "u_1001",
            "action": "handoff",
            "confirm_text": "确认转人工",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["state"] == "HUMAN"
    assert "转人工" in body["reply"]


def test_order_not_found_does_not_guess(client):
    data = _diagnose(client, "这个能退吗？", order_id="O20269999999")
    assert data["verdict"]["code"] == "ORDER_NOT_FOUND"
    assert data["order"] is None
