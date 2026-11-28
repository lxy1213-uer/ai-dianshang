"""决策审计驾驶舱测试。

覆盖：总览 KPI / verdict 分布 / 执行率口径、决策列表筛选、
决策详情完整链（诊断→执行→工单→退款→护栏）、全局事件流脱敏、
风险清单分级、鉴权（strict 需主管权限）、PII 读时脱敏兜底。
"""

from __future__ import annotations

from datetime import date

from app.domain.audit import AuditService


def _diagnose(c, question="要退货", user_id="u_1001", order_id="O20261115001") -> dict:
    return c.diagnosis.diagnose(question, user_id, order_id)


def _execute(c, decision: dict, user_id="u_1001", action="create_ticket") -> dict:
    from app.agent.diagnosis import CONFIRM_CREATE, CONFIRM_HANDOFF
    confirm = CONFIRM_CREATE if action == "create_ticket" else CONFIRM_HANDOFF
    return c.diagnosis.execute(decision["decision_id"], user_id, action, confirm)


# ---------- 总览 ----------
def test_overview_counts_decisions_verdicts_and_execution_rate(coordinator):
    a = AuditService(coordinator.db)
    d1 = _diagnose(coordinator)
    _diagnose(coordinator, question="我没有订单号也想退货", order_id=None)
    _execute(coordinator, d1)

    ov = a.overview()
    assert ov["total_decisions"] == 2
    codes = {v["code"] for v in ov["verdicts"]}
    assert "ELIGIBLE" in codes and "NEED_ORDER" in codes
    assert ov["execution"]["executed"] == 1
    assert ov["execution"]["create_ticket"] == 1
    assert ov["execution"]["distinct_decisions_executed"] == 1
    # 2 条决策，1 条有执行 → 执行率 0.5
    assert ov["execution"]["rate"] == 0.5
    assert ov["events_total"] >= 4


def test_overview_empty_database_is_zero_not_error(coordinator):
    ov = AuditService(coordinator.db).overview()
    assert ov["total_decisions"] == 0
    assert ov["execution"]["rate"] == 0.0
    assert ov["risks"]["total"] == 0
    assert ov["verdicts"] == []


# ---------- 决策列表筛选 ----------
def test_decisions_lists_and_filters_by_verdict_executed_and_user(coordinator):
    a = AuditService(coordinator.db)
    eligible = _diagnose(coordinator)  # ELIGIBLE
    _diagnose(coordinator, question="没有订单号想退货", order_id=None)  # NEED_ORDER
    _execute(coordinator, eligible)

    all_items = a.decisions()["items"]
    assert len(all_items) == 2
    assert all_items[0]["decision_id"]  # 有决策号

    only_exec = a.decisions(executed="yes")["items"]
    assert len(only_exec) == 1
    assert only_exec[0]["decision_id"] == eligible["decision_id"]
    assert only_exec[0]["has_execution"] is True

    only_none = a.decisions(executed="no")["items"]
    assert len(only_none) == 1
    assert only_none[0]["has_execution"] is False

    by_verdict = a.decisions(verdict="NEED_ORDER")["items"]
    assert len(by_verdict) == 1 and by_verdict[0]["verdict"] == "NEED_ORDER"

    by_user = a.decisions(user_id="u_9999")["items"]
    assert by_user == []


def test_decisions_user_id_filter_uses_audited_field(coordinator):
    a = AuditService(coordinator.db)
    # 不带订单号：诊断无需归属校验即可落审计事件，便于验证 user_id 筛选口径
    _diagnose(coordinator, user_id="u_alice", order_id=None)
    assert len(a.decisions(user_id="u_alice")["items"]) == 1
    assert len(a.decisions(user_id="u_bob")["items"]) == 0


# ---------- 决策详情完整链 ----------
def test_detail_links_diagnosis_execution_ticket_and_status(coordinator):
    a = AuditService(coordinator.db)
    d = _diagnose(coordinator)
    _execute(coordinator, d)

    detail = a.detail(d["decision_id"])
    assert detail["found"] is True
    assert detail["diagnosis"]["verdict"] == "ELIGIBLE"
    assert detail["diagnosis"]["citations"], "诊断应带条款依据"
    assert detail["diagnosis"]["user_id"] == "u_1001"
    assert any(x["action"] == "create_ticket" for x in detail["actions"])
    assert detail["ticket"] is not None, "建单后应能反查工单"
    assert detail["ticket"]["ticket_id"]
    kinds = {x["kind"] for x in detail["chain"]}
    assert "aftersale_diagnosis" in kinds
    assert "aftersale_execution" in kinds
    assert "ticket_status_change" in kinds or "ticket_create_attempt" in kinds


def test_detail_without_execution_has_no_ticket(coordinator):
    a = AuditService(coordinator.db)
    d = _diagnose(coordinator)
    detail = a.detail(d["decision_id"])
    assert detail["found"] is True
    assert detail["ticket"] is None
    assert detail["refund"] is None
    assert detail["actions"] == []


def test_detail_unknown_and_malformed_ids_are_safe(coordinator):
    a = AuditService(coordinator.db)
    missing = a.detail("DG_NOT_EXIST")
    assert missing["found"] is False and missing["chain"] == []
    bad = a.detail("'; DROP TABLE events; --")
    assert bad["found"] is False
    # 恶意输入未破坏表
    assert AuditService(coordinator.db).overview()["total_decisions"] == 0


def test_detail_includes_refund_when_executed(coordinator):
    a = AuditService(coordinator.db)
    d = _diagnose(coordinator)
    result = _execute(coordinator, d)
    ticket_id = result["ticket"]["ticket_id"]
    # 直接落退款流水与资金事件：审计层只负责串联，不依赖退款状态机的前置状态
    _, row = coordinator.db.reserve_refund(
        ticket_id, f"refund:{ticket_id}", "claim-audit", "agent", "sandbox"
    )
    coordinator.db.finish_refund(
        ticket_id, "claim-audit", "SUCCESS", refund_id="RF-AUDIT", amount=299.0
    )
    coordinator.db.log_event(
        f"diag_{d['decision_id']}_ticket", d["decision_id"], "refund_executed",
        {"ticket_id": ticket_id, "refund_id": "RF-AUDIT", "operator": "agent"},
    )

    detail = a.detail(d["decision_id"])
    assert detail["refund"] is not None
    assert detail["refund"]["status"] == "SUCCESS"
    assert detail["refund"]["amount"] == 299.0
    assert detail["refund"]["refund_id"] == "RF-AUDIT"
    kinds = {x["kind"] for x in detail["chain"]}
    assert "refund_executed" in kinds


def test_detail_includes_guardrail_event_in_chain(coordinator):
    """同一会话前缀下的护栏事件也应收拢进决策链。"""
    a = AuditService(coordinator.db)
    d = _diagnose(coordinator)
    did = d["decision_id"]
    coordinator.db.log_event(
        f"diag_{did}_ticket", did, "eligibility_changed",
        {"order_id": "O20261115001", "code": "OUT_OF_WINDOW"},
    )
    detail = a.detail(did)
    assert "eligibility_changed" in {x["kind"] for x in detail["chain"]}


# ---------- 全局事件流 ----------
def test_events_filter_by_group_and_sanitize_payload(coordinator):
    a = AuditService(coordinator.db)
    d = _diagnose(coordinator)
    # 故意写入带手机号/身份证/地址的审计事件，验证读时脱敏
    coordinator.db.log_event(
        f"diag_{d['decision_id']}", d["decision_id"], "security_violation",
        {"phone": "13812345678", "note": "联系 13812345678，身份证 110101199001011234",
         "address": "某某路 1 号", "type": "order_ownership"},
    )
    stream = a.events(group="护栏")
    assert stream["total"] == 1
    payload = stream["items"][0]["payload"]
    assert "phone" not in payload and "address" not in payload
    assert "13812345678" not in str(payload)
    assert "110101199001011234" not in str(payload)
    assert "138****5678" in payload["note"]

    # 按类型过滤
    assert a.events(kinds=["aftersale_diagnosis"])["total"] == 1
    assert a.events(kinds=["does_not_exist"])["total"] == 0


# ---------- 风险清单 ----------
def test_risks_are_grouped_by_severity_with_actionable_detail(coordinator):
    a = AuditService(coordinator.db)
    coordinator.db.log_event(
        "s-risk", "DG_RISK1", "refund_status_failed",
        {"ticket_id": "RT1", "refund_id": "RF1", "error": "Timeout"},
    )
    coordinator.db.log_event(
        "s-risk", None, "sla_breach", {"queue_id": "HQ1"}
    )
    coordinator.db.log_event(
        "s-risk", None, "turn", {"message": "正常对话"}
    )
    risks = a.risks()
    assert risks["total"] == 2
    assert risks["by_severity"]["high"] == 1
    assert risks["by_severity"]["medium"] == 1
    by_kind = {i["kind"]: i for i in risks["items"]}
    assert "RT1" in by_kind["refund_status_failed"]["detail"]
    assert by_kind["refund_status_failed"]["decision_id"] == "DG_RISK1"
    assert "HQ1" in by_kind["sla_breach"]["detail"]


def test_risks_empty_is_safe(coordinator):
    r = AuditService(coordinator.db).risks()
    assert r["total"] == 0 and r["items"] == []


# ---------- API 层 ----------
def test_audit_api_demo_mode_all_endpoints(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    monkeypatch.setenv("DB_PATH", str(tmp_path / "audit_api.db"))
    monkeypatch.setenv("AGENT_TODAY", "2026-11-22")
    monkeypatch.setenv("AGENT_LLM_MODE", "rules")
    monkeypatch.setenv("AGENT_RBAC_ENABLED", "0")
    monkeypatch.setenv("AGENT_RATE_LIMIT_PER_MIN", "0")
    with TestClient(app) as tc:
        r = tc.post("/api/aftersale/diagnose", json={
            "question": "我要退货", "user_id": "u_1001", "order_id": "O20261115001",
        })
        assert r.status_code == 200
        did = r.json()["decision_id"]
        tc.post("/api/aftersale/execute", json={
            "decision_id": did, "user_id": "u_1001",
            "action": "create_ticket", "confirm_text": "确认创建工单",
        })

        assert tc.get("/api/audit/overview").json()["total_decisions"] == 1
        dec = tc.get("/api/audit/decisions").json()
        assert dec["total"] == 1
        detail = tc.get(f"/api/audit/decisions/{did}").json()
        assert detail["found"] is True and detail["ticket"] is not None
        assert tc.get("/api/audit/events").json()["total"] >= 1
        assert tc.get("/api/audit/risks").json()["total"] >= 0
        page = tc.get("/audit")
        assert page.status_code == 200 and "决策审计驾驶舱" in page.text


def test_audit_api_strict_requires_supervisor(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.auth import issue_staff_token

    monkeypatch.setenv("DB_PATH", str(tmp_path / "audit_strict.db"))
    monkeypatch.setenv("AGENT_TODAY", "2026-11-22")
    monkeypatch.setenv("AGENT_LLM_MODE", "rules")
    monkeypatch.setenv("AGENT_RBAC_ENABLED", "1")
    monkeypatch.setenv("AGENT_AUTH_SECRET", "test-secret")
    monkeypatch.setenv("AGENT_STAFF_ACCOUNTS", "sup:pw:supervisor;ag:pw:agent;vw:pw:viewer")
    monkeypatch.setenv("AGENT_RATE_LIMIT_PER_MIN", "0")
    from app.config import Settings
    s = Settings.load(db_path=tmp_path / "audit_strict.db")
    agent_token = issue_staff_token("ag", "agent", s.auth_secret)
    viewer_token = issue_staff_token("vw", "viewer", s.auth_secret)
    sup_token = issue_staff_token("sup", "supervisor", s.auth_secret)

    with TestClient(app) as tc:
        assert tc.get("/api/audit/overview").status_code == 401
        assert tc.get("/api/audit/overview", headers={"X-Staff-Token": viewer_token}).status_code == 403
        assert tc.get("/api/audit/overview", headers={"X-Staff-Token": agent_token}).status_code == 403
        r = tc.get("/api/audit/overview", headers={"X-Staff-Token": sup_token})
        assert r.status_code == 200 and "total_decisions" in r.json()
        assert tc.get("/api/audit/decisions", headers={"X-Staff-Token": sup_token}).status_code == 200
        assert tc.get("/api/audit/risks", headers={"X-Staff-Token": sup_token}).status_code == 200
