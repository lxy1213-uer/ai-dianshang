"""诊断报告导出：只读、脱敏、服务端取数、零依赖 xlsx 与鉴权。"""

from __future__ import annotations

import io
import zipfile
from xml.etree import ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from app.agent.diagnosis import CONFIRM_CREATE, DiagnosisError
from app.domain.export import ExportError, ReportExporter

NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _diagnose(c, question="签收7天了能退吗", user_id="u_1001", order_id="O20261115001"):
    return c.diagnosis.diagnose(question, user_id, order_id)


# ---------- 报告组装 ----------
def test_diagnosis_report_contains_all_sections(coordinator):
    d = _diagnose(coordinator)
    report = ReportExporter(coordinator.diagnosis, coordinator.audit).build_diagnosis(
        d["decision_id"], "u_1001"
    )
    assert report["scope"] == "diagnosis"
    assert d["decision_id"] in [m["value"] for m in report["meta"]]
    titles = [s["title"] for s in report["sections"]]
    assert any("诊断结论" in t for t in titles)
    assert any("条款证据" in t for t in titles)
    assert any("边界" in t for t in titles)
    evidence = next(s for s in report["sections"] if "条款证据" in s["title"])
    assert any(row[0].startswith("R-") for row in evidence["rows"])


def test_diagnosis_report_requires_own_identity(coordinator):
    d = _diagnose(coordinator)
    exporter = ReportExporter(coordinator.diagnosis, coordinator.audit)
    with pytest.raises(ExportError) as exc:
        exporter.build_diagnosis(d["decision_id"], "u_other")
    assert exc.value.code == "forbidden"


def test_diagnosis_report_allows_expired_decision_for_readonly_export(coordinator, monkeypatch):
    """执行有 30 分钟 TTL；查阅/导出历史结论不受限，但不能拿旧结论执行。"""
    import app.agent.diagnosis as diag_module

    d = _diagnose(coordinator)
    real_time = diag_module.time.time
    monkeypatch.setattr(diag_module.time, "time", lambda: real_time() + 3601)
    exporter = ReportExporter(coordinator.diagnosis, coordinator.audit)
    report = exporter.build_diagnosis(d["decision_id"], "u_1001")
    assert report["meta"][0]["value"] == d["decision_id"]
    with pytest.raises(DiagnosisError) as exc:
        coordinator.diagnosis.execute(d["decision_id"], "u_1001", "handoff", "确认转人工")
    assert exc.value.code == "decision_expired"


def test_audit_report_includes_execution_and_chain(coordinator):
    d = _diagnose(coordinator)
    coordinator.diagnosis.execute(d["decision_id"], "u_1001", "create_ticket", CONFIRM_CREATE)
    report = ReportExporter(coordinator.diagnosis, coordinator.audit).build_audit(
        d["decision_id"]
    )
    assert report["scope"] == "audit"
    assert d["decision_id"] in [m["value"] for m in report["meta"]]
    timeline = next(s for s in report["sections"] if "时间线" in s["title"])
    kinds = {row[2] for row in timeline["rows"]}
    assert "售后诊断" in kinds and "决策执行" in kinds


def test_audit_report_missing_decision_raises(coordinator):
    with pytest.raises(ExportError) as exc:
        ReportExporter(coordinator.diagnosis, coordinator.audit).build_audit("DGNOTFOUND")
    assert exc.value.status == 404


# ---------- 渲染 ----------
def test_html_is_printable_escaped_and_masked(coordinator):
    d = _diagnose(coordinator, question="我的电话 13812345678 签收7天能退吗")
    report = ReportExporter(coordinator.diagnosis, coordinator.audit).build_diagnosis(
        d["decision_id"], "u_1001"
    )
    html = ReportExporter.render_html(report)
    assert "@media print" in html and "window.print()" in html
    assert "138****5678" in html
    assert "13812345678" not in html
    assert "只读导出" in html


def test_xlsx_is_valid_zip_with_multiple_worksheets(coordinator):
    d = _diagnose(coordinator)
    report = ReportExporter(coordinator.diagnosis, coordinator.audit).build_diagnosis(
        d["decision_id"], "u_1001"
    )
    raw = ReportExporter.render_xlsx(report)
    assert raw[:2] == b"PK"
    zf = zipfile.ZipFile(io.BytesIO(raw))
    assert "[Content_Types].xml" in zf.namelist()
    assert "_rels/.rels" in zf.namelist()
    assert "xl/styles.xml" in zf.namelist()
    for name in zf.namelist():
        ET.fromstring(zf.read(name))  # 每个部件都必须是合法 XML
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    sheets = wb.findall(".//m:sheet", NS)
    assert len(sheets) >= 5
    assert sheets[0].get("name") == "报告概览"


def test_xlsx_masks_pii(coordinator):
    d = _diagnose(coordinator, question="电话 13812345678 想退货")
    report = ReportExporter(coordinator.diagnosis, coordinator.audit).build_diagnosis(
        d["decision_id"], "u_1001"
    )
    zf = zipfile.ZipFile(io.BytesIO(ReportExporter.render_xlsx(report)))
    payload = "".join(zf.read(n).decode("utf-8") for n in zf.namelist() if n.endswith(".xml"))
    assert "138****5678" in payload
    assert "13812345678" not in payload


# ---------- API ----------
def test_api_exports_both_formats(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "export_api.db"))
    from app.main import app

    with TestClient(app) as tc:
        d = tc.post(
            "/api/aftersale/diagnose",
            json={"question": "签收7天了能退吗", "user_id": "u_1001", "order_id": "O20261115001"},
        ).json()
        did = d["decision_id"]
        for fmt in ("html", "xlsx"):
            r = tc.get(
                f"/api/reports/diagnosis/{did}",
                params={"user_id": "u_1001", "format": fmt},
            )
            assert r.status_code == 200
            assert r.content
            assert "filename*" in r.headers["content-disposition"]
            if fmt == "xlsx":
                assert zipfile.ZipFile(io.BytesIO(r.content)).testzip() is None


def test_api_audit_export_requires_supervisor(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "export_strict.db"))
    monkeypatch.setenv("AGENT_RBAC_ENABLED", "1")
    monkeypatch.setenv("AGENT_AUTH_SECRET", "export-secret")
    monkeypatch.setenv("AGENT_STAFF_ACCOUNTS", "sup:pw:supervisor;vw:pw:viewer")
    monkeypatch.setenv("AGENT_RATE_LIMIT_PER_MIN", "0")
    from app.auth import issue_staff_token, issue_token
    from app.config import Settings
    from app.main import app

    secret = Settings.load().auth_secret
    viewer = issue_staff_token("vw", "viewer", secret)
    sup = issue_staff_token("sup", "supervisor", secret)
    user_token = issue_token("u_1001", secret)
    with TestClient(app) as tc:
        d = tc.post(
            "/api/aftersale/diagnose",
            json={"question": "签收7天了能退吗", "user_id": "u_1001", "order_id": "O20261115001"},
            headers={"Authorization": "Bearer " + user_token},
        ).json()
        url = f"/api/reports/audit/{d['decision_id']}"
        assert tc.get(url, headers={"X-Staff-Token": viewer}).status_code == 403
        assert tc.get(url, headers={"X-Staff-Token": sup}).status_code == 200


def test_api_rejects_cross_user_and_unknown(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "export_reject.db"))
    from app.main import app

    with TestClient(app) as tc:
        d = tc.post(
            "/api/aftersale/diagnose",
            json={"question": "能退吗", "user_id": "u_1001", "order_id": "O20261115001"},
        ).json()
        did = d["decision_id"]
        assert tc.get(
            f"/api/reports/diagnosis/{did}",
            params={"user_id": "u_9999", "format": "html"},
        ).status_code == 403
        assert tc.get(
            "/api/reports/diagnosis/DGNOPE",
            params={"user_id": "u_1001", "format": "html"},
        ).status_code == 404

