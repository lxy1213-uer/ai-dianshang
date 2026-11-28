"""诊断报告导出：把一次决策渲染成可打印 HTML / Excel（零第三方依赖）。

设计纪律
--------
1. **只读**：不建单、不退款、不改任何状态，只把已有数据排版输出；
2. **服务端取数**：报告内容来自决策表/审计链的权威记录，不信前端传来的结论；
3. **读时脱敏**：所有文本复用 `events.sanitize` 的同一套手机号/身份证打码规则；
4. **双视角**：
   - `diagnosis`：用户导出自己的诊断（不限制 30 分钟执行有效期，但严格按 user_id 隔离）；
   - `audit`：主管导出完整决策链（含工单/退款/护栏），走 AUDIT_READ 权限。
5. **格式降级**：本机没有 reportlab/openpyxl，HTML 自带打印样式（浏览器另存 PDF），
   xlsx 用 zipfile + XML 手写最小工作簿，不用装任何库。
"""

from __future__ import annotations

import re
import zipfile
from datetime import datetime
from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape as xml_escape

from ..agent.diagnosis import AftersaleDiagnosisService, DiagnosisError
from .audit import AuditService, VERDICT_LABELS
from .events import sanitize


class ExportError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


_SHEET_BAD = re.compile(r"[\\/*?:\[\]]")


def _clean(value: Any) -> str:
    """复用事件脱敏规则；None/容器统一转成可读文本。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        return str(value)
    if not isinstance(value, str):
        value = str(value)
    # sanitize 是项目唯一的脱敏实现，包一层 dict 复用，避免规则漂移
    return str(sanitize({"v": value})["v"])


def _verdict_label(code: str | None) -> str:
    return VERDICT_LABELS.get(code or "", code or "—")


class ReportExporter:
    def __init__(self, diagnosis: AftersaleDiagnosisService, audit: AuditService):
        self.diagnosis = diagnosis
        self.audit = audit

    # ---------- 报告组装 ----------
    def build_diagnosis(self, decision_id: str, user_id: str) -> dict:
        try:
            payload = self.diagnosis.for_export(decision_id, user_id)
        except DiagnosisError as exc:
            raise ExportError(exc.code, exc.message, exc.status) from exc
        safe = sanitize(payload)
        verdict = safe.get("verdict") or {}
        reason_evidence = safe.get("reason_evidence") or {}

        meta = [
            {"label": "决策编号", "value": _clean(safe.get("decision_id"))},
            {"label": "诊断时间", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "（导出时间）"},
            {"label": "结论", "value": _verdict_label(verdict.get("code"))},
            {"label": "诉求分类", "value": _clean(safe.get("reason"))},
            {"label": "订单号", "value": _clean(safe.get("order_id")) or "未提供"},
        ]
        sections: list[dict] = [
            {
                "title": "一、诊断结论",
                "kind": "kv",
                "rows": [
                    {"label": "结论标题", "value": _clean(verdict.get("headline")), "source": "诊断结论"},
                    {"label": "结论说明", "value": _clean(verdict.get("summary")), "source": "诊断结论"},
                    {"label": "资格代码", "value": _clean(verdict.get("eligibility_code")), "source": "核单工具"},
                    {"label": "剩余天数", "value": _clean(verdict.get("remaining_days")), "source": "核单工具"},
                    {"label": "截止日期", "value": _clean(verdict.get("deadline")), "source": "核单工具"},
                    {"label": "需人工复核", "value": _clean(verdict.get("requires_human", False)), "source": "核单工具"},
                    {"label": "用户原话", "value": _clean(safe.get("question")), "source": "用户输入"},
                ],
            },
            {
                "title": "二、订单与诉求事实",
                "kind": "table",
                "headers": ["事实项", "取值", "来源"],
                "rows": [
                    [_clean(f.get("label")), _clean(f.get("value")), _clean(f.get("source"))]
                    for f in (safe.get("facts") or [])
                ],
            },
            {
                "title": "三、条款证据",
                "kind": "table",
                "headers": ["条款编号", "标题", "命中原文", "来源"],
                "rows": [
                    [
                        _clean(c.get("clause_id")), _clean(c.get("title")),
                        _clean(c.get("snippet")), _clean(c.get("source")),
                    ]
                    for c in (safe.get("evidence") or [])
                ],
            },
            {
                "title": "四、诉求证据规则",
                "kind": "kv",
                "rows": [
                    {"label": "判定规则", "value": _clean(reason_evidence.get("rule")), "source": "缺陷证据校验"},
                    {"label": "命中缺陷证据", "value": _clean("、".join(reason_evidence.get("evidence") or [])), "source": "用户原话"},
                    {"label": "属性描述（不算缺陷）", "value": _clean("、".join(reason_evidence.get("attributes") or [])), "source": "用户原话"},
                ],
            },
            {
                "title": "五、缺口、下一步与边界",
                "kind": "bullets",
                "groups": [
                    {"label": "还缺什么", "items": [_clean(x) for x in (safe.get("gaps") or [])]},
                    {"label": "建议下一步", "items": [_clean(x) for x in (safe.get("next_steps") or [])]},
                    {"label": "边界声明", "items": [_clean(x) for x in (safe.get("boundaries") or [])]},
                ],
            },
            {
                "title": "六、可执行动作（导出不代表已执行）",
                "kind": "table",
                "headers": ["动作", "是否可执行", "原因"],
                "rows": [
                    [_clean(a.get("label")), _clean(a.get("enabled", False)), _clean(a.get("why"))]
                    for a in (safe.get("actions") or [])
                ],
            },
        ]
        return {
            "title": "售后决策诊断报告",
            "subtitle": "用户视角 · 有证据、有边界、能执行",
            "scope": "diagnosis",
            "meta": meta,
            "sections": sections,
            "footer": "本报告只读导出，不代表任何工单已创建或资金已划转；最终结论以平台审核为准。",
        }

    def build_audit(self, decision_id: str) -> dict:
        detail = sanitize(self.audit.detail(decision_id))
        if not detail.get("found"):
            raise ExportError("decision_not_found", "该决策编号没有可导出的诊断记录", 404)
        diag = detail.get("diagnosis") or {}
        ticket = detail.get("ticket") or {}
        refund = detail.get("refund") or {}
        chain = detail.get("chain") or []

        meta = [
            {"label": "决策编号", "value": _clean(detail.get("decision_id"))},
            {"label": "导出时间", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"label": "诊断结论", "value": _verdict_label(diag.get("verdict"))},
            {"label": "诉求分类", "value": _clean(diag.get("reason"))},
            {"label": "订单号", "value": _clean(diag.get("order_id")) or "未提供"},
            {"label": "用户", "value": _clean(diag.get("user_id")) or "—"},
        ]
        sections: list[dict] = [
            {
                "title": "一、诊断摘要",
                "kind": "kv",
                "rows": [
                    {"label": "结论", "value": _verdict_label(diag.get("verdict")), "source": "审计事件"},
                    {"label": "引用条款", "value": _clean("、".join(diag.get("citations") or [])), "source": "审计事件"},
                    {"label": "可执行动作", "value": _clean("、".join(diag.get("executable") or [])), "source": "审计事件"},
                    {"label": "诊断时间", "value": _clean(diag.get("created_at")), "source": "事件留痕"},
                ],
            },
            {
                "title": "二、执行与工单",
                "kind": "table",
                "headers": ["项目", "取值"],
                "rows": [
                    ["执行动作", _clean("、".join(str(a.get("action")) for a in (detail.get("actions") or [])) or "无")],
                    ["工单编号", _clean(ticket.get("ticket_id")) or "无"],
                    ["工单状态", _clean(ticket.get("status")) or "无"],
                    ["工单创建时间", _clean(ticket.get("created_at"))],
                ],
            },
            {
                "title": "三、退款流水",
                "kind": "kv",
                "rows": (
                    [
                        {"label": "退款状态", "value": _clean(refund.get("status")), "source": "网关流水"},
                        {"label": "退款单号", "value": _clean(refund.get("refund_id")), "source": "网关流水"},
                        {"label": "金额", "value": _clean(refund.get("amount")), "source": "网关流水"},
                        {"label": "操作人", "value": _clean(refund.get("operator")), "source": "网关流水"},
                    ] if refund else [
                        {"label": "退款流水", "value": "本次决策没有退款记录", "source": "审计链"},
                    ]
                ),
            },
            {
                "title": "四、完整决策时间线",
                "kind": "table",
                "headers": ["时间", "分组", "事件", "关联工单", "事件内容"],
                "rows": [
                    [
                        _clean(e.get("created_at")), _clean(e.get("group")), _clean(e.get("label")),
                        _clean(e.get("ticket_id")),
                        _clean("；".join(f"{k}={v}" for k, v in (e.get("payload") or {}).items())[:300]),
                    ]
                    for e in chain
                ],
            },
        ]
        return {
            "title": "售后决策审计报告",
            "subtitle": "主管视角 · 诊断 → 执行 → 工单 → 退款 → 护栏完整链",
            "scope": "audit",
            "meta": meta,
            "sections": sections,
            "footer": "本报告只读导出，事件内容已做脱敏；资金与身份结论请以原始流水及平台审核为准。",
        }

    # ---------- 渲染 ----------
    @staticmethod
    def render_html(report: dict) -> str:
        meta_rows = "".join(
            f"<tr><th>{xml_escape(_clean(m['label']))}</th>"
            f"<td>{xml_escape(_clean(m['value']))}</td></tr>"
            for m in report["meta"]
        )
        blocks: list[str] = []
        for section in report["sections"]:
            parts = [f"<h2>{xml_escape(section['title'])}</h2>"]
            kind = section["kind"]
            if kind == "kv":
                rows = "".join(
                    f"<tr><th>{xml_escape(_clean(r.get('label')))}</th>"
                    f"<td>{xml_escape(_clean(r.get('value')))}</td>"
                    f"<td class='src'>{xml_escape(_clean(r.get('source')))}</td></tr>"
                    for r in section.get("rows", [])
                )
                parts.append(
                    "<table><thead><tr><th style='width:22%'>项目</th><th>取值</th>"
                    f"<th style='width:18%'>来源</th></tr></thead><tbody>{rows}</tbody></table>"
                )
            elif kind == "table":
                head = "".join(f"<th>{xml_escape(h)}</th>" for h in section.get("headers", []))
                body = "".join(
                    "<tr>" + "".join(f"<td>{xml_escape(_clean(c))}</td>" for c in row) + "</tr>"
                    for row in section.get("rows", [])
                )
                parts.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
            elif kind == "bullets":
                for group in section.get("groups", []):
                    parts.append(f"<h3>{xml_escape(_clean(group.get('label')))}</h3>")
                    parts.append("<ul>" + "".join(
                        f"<li>{xml_escape(item)}</li>" for item in group.get("items", [])
                    ) + "</ul>")
            blocks.append("".join(parts))

        return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8" />
<title>{xml_escape(report['title'])} · {xml_escape(report['scope'].upper())}</title>
<style>
@page {{ size: A4; margin: 18mm 16mm; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; color: #16161a; background: #fff; font: 12px/1.65 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; }}
.doc {{ max-width: 940px; margin: 0 auto; padding: 28px 32px 48px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
.sub {{ color: #46443d; font-size: 12px; margin-bottom: 18px; }}
.stamp {{ float: right; padding: 4px 10px; border: 1px solid #b23b30; color: #b23b30;
  border-radius: 6px; font-size: 11px; }}
table {{ width: 100%; border-collapse: collapse; margin: 8px 0 16px; table-layout: fixed; }}
th, td {{ border: 1px solid #ddd7c9; padding: 7px 9px; text-align: left; vertical-align: top;
  word-break: break-word; white-space: pre-wrap; }}
thead th {{ background: #f2efe6; font-weight: 600; }}
td.src, tbody th {{ color: #6b6659; font-weight: 400; }}
h2 {{ font-size: 14px; margin: 22px 0 8px; padding-bottom: 6px; border-bottom: 2px solid #2a47c6; }}
h3 {{ font-size: 12px; margin: 14px 0 6px; color: #2a47c6; }}
ul {{ margin: 6px 0 12px; padding-left: 22px; }}
li {{ margin: 3px 0; }}
.foot {{ border-top: 1px solid #ddd7c9; margin-top: 28px; padding-top: 10px;
  color: #6b6659; font-size: 11px; }}
@media print {{ .noprint {{ display: none; }} body {{ background: #fff; }} }}
.noprint {{ position: fixed; right: 18px; bottom: 18px; }}
.noprint button {{ padding: 9px 16px; border: 0; border-radius: 8px; background: #2a47c6;
  color: #fff; font-size: 12px; cursor: pointer; }}
</style></head><body><div class="doc">
<span class="stamp">只读导出 · 未执行任何动作</span>
<h1>{xml_escape(report['title'])}</h1>
<div class="sub">{xml_escape(report['subtitle'])}</div>
<table><tbody>{meta_rows}</tbody></table>
{''.join(blocks)}
<div class="foot">{xml_escape(report['footer'])}</div>
</div>
<div class="noprint"><button onclick="window.print()">打印 / 另存为 PDF</button></div>
</body></html>"""

    @staticmethod
    def render_xlsx(report: dict) -> bytes:
        """最小 XLSX：inlineStr + 多工作表，无 sharedStrings / openpyxl 依赖。"""
        def col_letter(index: int) -> str:
            name = ""
            while index > 0:
                index, rem = divmod(index - 1, 26)
                name = chr(65 + rem) + name
            return name

        def _rows_xml(rows: list[list[str]], letter) -> list[str]:
            xml_rows = []
            for row_idx, row in enumerate(rows, 1):
                cells = []
                for col, cell in enumerate(row, 1):
                    style = ' s="1"' if row_idx == 1 else ""
                    cells.append(
                        f'<c r="{letter(col)}{row_idx}"{style} t="inlineStr">'
                        f'<is><t xml:space="preserve">{xml_escape(cell)}</t></is></c>'
                    )
                xml_rows.append(f'<row r="{row_idx}">' + "".join(cells) + "</row>")
            return xml_rows

        # 概览按两列表格排布
        overview: list[list[str]] = [["项目", "取值"]]
        overview += [[_clean(m["label"]), _clean(m["value"])] for m in report["meta"]]
        overview += [["报告说明", _clean(report["footer"])]]
        overview += [["导出范围", report["scope"]], ["导出时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]]

        flat_sheets: list[tuple[str, list[list[str]]]] = [
            ("报告概览", overview),
        ]
        for section in report["sections"]:
            rows: list[list[str]] = []
            if section["kind"] == "kv":
                rows.append(["项目", "取值", "来源"])
                rows += [
                    [_clean(r.get("label")), _clean(r.get("value")), _clean(r.get("source"))]
                    for r in section.get("rows", [])
                ]
            elif section["kind"] == "table":
                rows.append(list(section.get("headers", [])))
                rows += [[_clean(c) for c in row] for row in section.get("rows", [])]
            elif section["kind"] == "bullets":
                rows.append(["分组", "条目"])
                for group in section.get("groups", []):
                    for item in group.get("items", []):
                        rows.append([_clean(group.get("label")), _clean(item)])
            name = _SHEET_BAD.sub("_", section["title"])[:31] or "章节"
            flat_sheets.append((name, rows))

        sheet_entries: list[str] = []
        rel_entries: list[str] = []
        override_entries: list[str] = []
        sheet_xmls: list[str] = []
        for idx, (name, _rows) in enumerate(flat_sheets, 1):
            path = f"xl/worksheets/sheet{idx}.xml"
            safe_name = xml_escape(name)
            sheet_entries.append(
                f'<sheet name="{safe_name}" sheetId="{idx}" r:id="rId{idx}" />'
            )
            rel_entries.append(
                f'<Relationship Id="rId{idx}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                f'Target="worksheets/sheet{idx}.xml" />'
            )
            override_entries.append(
                f'<Override PartName="/{path}" ContentType="application/vnd.openxmlformats-'
                'officedocument.spreadsheetml.worksheet+xml" />'
            )
            sheet_xmls.append(
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                '<sheetData>' + "".join(_rows_xml(_rows, col_letter)) + '</sheetData></worksheet>'
            )

        content_types = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />'
            '<Default Extension="xml" ContentType="application/xml" />'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-'
            'officedocument.spreadsheetml.sheet.main+xml" />'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-'
            'officedocument.spreadsheetml.styles+xml" />'
            + "".join(override_entries) + "</Types>"
        )
        root_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml" /></Relationships>'
        )
        workbook = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets>' + "".join(sheet_entries) + '</sheets></workbook>'
        )
        workbook_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(rel_entries)
            + f'<Relationship Id="rId{len(flat_sheets) + 1}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml" /></Relationships>'
        )
        styles = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
            '<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
            '<borders count="1"><border/></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
            '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>'
            '</styleSheet>'
        )
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types)
            zf.writestr("_rels/.rels", root_rels)
            zf.writestr("xl/workbook.xml", workbook)
            zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
            zf.writestr("xl/styles.xml", styles)
            for idx, xml in enumerate(sheet_xmls, 1):
                zf.writestr(f"xl/worksheets/sheet{idx}.xml", xml)
        return buf.getvalue()
