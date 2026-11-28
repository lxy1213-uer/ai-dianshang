"""表达层后置事实校验（Grounding Guard）回归测试。

分两层：
1. 纯函数：verify() 对"删事实 / 造事实 / 越权承诺 / 语义反转 / 结构异常"必须回退，
   对"只改措辞、事实等价"的合法改写必须放行；
2. 端到端：注入一个会篡改话术的假大模型，验证主流程真的回退到模板，
   并且 trace / meta / 事件日志都留下可追溯的痕迹。
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from app.grounding import (
    _critical_numbers,
    _identifiers,
    _normalize_numbers,
    _numbers,
    verify,
)
from app.llm import RuleLLM

TICKET_DRAFT = (
    "售后工单已创建：RT20261122A1B2C3\n"
    "订单 O20261115001 ｜ 七天无理由退货\n\n"
    "接下来快递员会在 1 个工作日内联系取件；仓库签收质检后 1-3 个工作日退款，按原路退回。"
)

SUMMARY_DRAFT = (
    "查到订单 O20261115001 · 蓝牙耳机\n"
    "金额 ¥299.00 ｜ 已签收（2026-11-15 10:30）\n"
    "判定：符合七天无理由退货，还剩 3 天（截止 2026-11-22）\n"
    "取件：上门取件 9:00-20:00，运费由商家承担\n"
    "依据：R-01、R-02\n\n"
    "确认无误请回复「确认」，我会创建售后工单；回复「取消」放弃本次申请。"
)

REJECT_DRAFT = (
    "这笔订单暂时不能直接办理退货：\n\n"
    "【已超退货期限】签收后今天是第 8 天，超过 7 天无理由退货期限。没有创建工单。"
)


class MutatingLLM(RuleLLM):
    """离线假模型：classify 走规则引擎，polish 按注入的函数篡改话术。"""

    def __init__(self, transform):
        self.transform = transform

    def polish(self, draft: str, context: dict | None = None):
        return self.transform(draft, context), 321


# ---------------------------------------------------------------------------
# 1. 事实抽取本身
# ---------------------------------------------------------------------------

def test_identifier_extraction_handles_chinese_boundaries():
    ids = _identifiers("订单O20261115001 · 工单 RT20261122A1B2C3 · 条款【R-01】")
    assert ids == {"O20261115001", "RT20261122A1B2C3", "R-01"}


def test_identifier_extraction_allows_space_inside_id():
    assert _identifiers("订单 O 20261115001") == {"O20261115001"}


def test_numbers_strip_identifier_digits():
    """RT...A1B2C3 里的零散 1/2/3 不能进入允许数字集合，否则偷改时效会漏检。"""
    assert 20261122 not in _numbers(TICKET_DRAFT)
    assert 20261115001 not in _numbers(TICKET_DRAFT)
    assert _critical_numbers(TICKET_DRAFT) == {1, 3, 7}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("一到三个工作日", "1到3个工作日"),
        ("还剩三天", "还剩3天"),
        ("一个工作日", "1个工作日"),
        ("一律不办", "一律不办"),
    ],
)
def test_chinese_number_normalization(text, expected):
    assert _normalize_numbers(text) == expected


# ---------------------------------------------------------------------------
# 2. 短路 / 开关
# ---------------------------------------------------------------------------

def test_unchanged_draft_short_circuits():
    result = verify(TICKET_DRAFT, TICKET_DRAFT)
    assert result.ok and not result.used_polished
    assert result.reply == TICKET_DRAFT
    assert result.detail["checked"] is False


def test_disabled_guard_passes_through():
    tampered = TICKET_DRAFT.replace("RT20261122A1B2C3", "")
    result = verify(TICKET_DRAFT, tampered, enabled=False)
    assert result.ok and result.reply == tampered
    assert result.detail["disabled"] is True


def test_facts_allow_known_clause_id():
    draft = "按平台规则：签收次日起 7 天内可申请无理由退货。"
    polished = "按平台规则【R-01】：签收次日起 7 天内可申请无理由退货。"
    assert not verify(draft, polished, facts={}).ok       # 无依据的条款号 = 编造
    assert verify(draft, polished, facts={"clause_ids": ["R-01"]}).ok


# ---------------------------------------------------------------------------
# 3. 违规：必须整体回退到模板
# ---------------------------------------------------------------------------

VIOLATIONS = [
    ("删工单号", TICKET_DRAFT, TICKET_DRAFT.replace("RT20261122A1B2C3", ""), "missing_fact"),
    ("改工单号", TICKET_DRAFT, TICKET_DRAFT.replace("RT20261122A1B2C3", "RT20261122ZZZZZZ"), "fabricated_id"),
    ("编造订单号", TICKET_DRAFT, TICKET_DRAFT + " 另见 O20261119999", "fabricated_id"),
    ("删订单号", SUMMARY_DRAFT, SUMMARY_DRAFT.replace("O20261115001", "这笔订单"), "missing_fact"),
    ("删条款号", SUMMARY_DRAFT, SUMMARY_DRAFT.replace("R-01、R-02", "平台售后政策"), "missing_fact"),
    ("编造条款号", SUMMARY_DRAFT, SUMMARY_DRAFT.replace("R-01、R-02", "R-01、R-99"), "fabricated_id"),
    ("编造金额", TICKET_DRAFT, TICKET_DRAFT + " 退款 ¥999.00", "fabricated_number"),
    ("偷改时效", TICKET_DRAFT, TICKET_DRAFT.replace("1-3 个工作日", "1-2 个工作日"), "fabricated_number"),
    ("删金额", SUMMARY_DRAFT, SUMMARY_DRAFT.replace("金额 ¥299.00 ｜ ", ""), "missing_fact"),
    ("删截止日期", SUMMARY_DRAFT, SUMMARY_DRAFT.replace("（截止 2026-11-22）", ""), "missing_fact"),
    ("政策外承诺", TICKET_DRAFT, "亲，保证退款，无条件包退～ " + TICKET_DRAFT, "ungrounded_promise"),
    ("空输出", TICKET_DRAFT, "", "empty_output"),
    ("复述提示词", TICKET_DRAFT, "原始话术：抱歉我不能完成，仅供参考", "prompt_leak"),
    ("输出超长", TICKET_DRAFT, TICKET_DRAFT * 8, "overlong"),
    ("拒绝反转为已受理", REJECT_DRAFT, "好的，可以直接办理退货，工单已创建～", "negation_flip"),
    (
        "保留数字但反转结论",
        REJECT_DRAFT,
        "这笔订单暂时不能直接办理退货：今天是第 8 天，超过 7 天期限。工单已创建。",
        "negation_flip",
    ),
]


@pytest.mark.parametrize("name,draft,polished,expected", VIOLATIONS, ids=[v[0] for v in VIOLATIONS])
def test_violations_fall_back_to_draft(name, draft, polished, expected):
    result = verify(draft, polished)
    assert not result.ok
    assert expected in result.violations
    assert result.reply == draft                 # 整体回退，不做部分修补
    assert result.used_polished is False
    assert result.detail["checked"] is True


# ---------------------------------------------------------------------------
# 4. 合法改写：事实等价，应当放行
# ---------------------------------------------------------------------------

LEGAL_REWRITES = [
    ("加礼貌语", TICKET_DRAFT, "亲，" + TICKET_DRAFT),
    ("中文天数", TICKET_DRAFT, TICKET_DRAFT.replace("1 个工作日", "一个工作日")),
    ("中文范围", TICKET_DRAFT, TICKET_DRAFT.replace("1-3 个工作日", "一到三个工作日")),
    ("金额换写法", SUMMARY_DRAFT, SUMMARY_DRAFT.replace("¥299.00", "299元")),
    ("日期换写法", SUMMARY_DRAFT, SUMMARY_DRAFT.replace("2026-11-22", "2026年11月22日")),
    ("单号带空格", TICKET_DRAFT, TICKET_DRAFT.replace("O20261115001", "O 20261115001")),
    ("时间窗口换写法", SUMMARY_DRAFT, SUMMARY_DRAFT.replace("9:00-20:00", "9 点到 20 点")),
]


@pytest.mark.parametrize("name,draft,polished", LEGAL_REWRITES, ids=[v[0] for v in LEGAL_REWRITES])
def test_equivalent_rewrites_pass(name, draft, polished):
    result = verify(draft, polished)
    assert result.ok
    assert result.used_polished is True
    assert result.reply == polished
    assert result.violations == []


# ---------------------------------------------------------------------------
# 5. 端到端：注入会篡改话术的模型
# ---------------------------------------------------------------------------

def _run_main_flow(coordinator, sid="s_grounding"):
    result = None
    for message in ("我要退货", "O20261115001", "确认"):
        result = coordinator.chat(sid, "u_1001", message)
    return result


def test_coordinator_falls_back_when_polish_drops_ticket(make_coordinator):
    c = make_coordinator("grounding_fallback.db")
    c.llm = MutatingLLM(lambda draft, ctx: re.sub(r"RT\d{8}[A-Za-z0-9]{6}", "", draft))

    result = _run_main_flow(c)

    assert result.state == "TICKETED"
    assert result.ticket and result.ticket["ticket_id"].startswith("RT")
    assert result.ticket["ticket_id"] in result.reply      # 模板事实回来了
    assert result.meta["grounding"]["fallback"] is True
    assert "missing_fact" in result.meta["grounding"]["violations"]
    assert result.trace[-1]["action"] == "事实校验回退"
    assert result.trace[-1]["detail"]["grounding_ok"] is False
    kinds = [event["kind"] for event in c.db.list_events("s_grounding")]
    assert "grounding_rejected" in kinds


def test_coordinator_keeps_safe_polish(make_coordinator):
    c = make_coordinator("grounding_safe.db")
    c.llm = MutatingLLM(lambda draft, ctx: "亲，" + draft)

    result = _run_main_flow(c)

    assert result.reply.startswith("亲，")
    assert result.ticket["ticket_id"] in result.reply
    assert result.meta["grounding"]["ok"] is True
    assert result.meta["grounding"]["used_polished"] is True
    assert result.trace[-1]["action"] == "话术润色"
    kinds = [event["kind"] for event in c.db.list_events("s_grounding")]
    assert "grounding_rejected" not in kinds


def test_grounding_disabled_lets_tampering_through(make_coordinator):
    """对照演示：关掉护栏，删掉工单号的话术就真的漏出去了。"""
    c = make_coordinator("grounding_off.db", grounding_enabled=False)
    c.llm = MutatingLLM(lambda draft, ctx: re.sub(r"RT\d{8}[A-Za-z0-9]{6}", "", draft))

    result = _run_main_flow(c)

    assert result.state == "TICKETED"
    assert "RT" not in result.reply
    assert "grounding" not in result.meta


def test_rule_engine_never_triggers_guard(coordinator):
    """规则模式不改写话术，护栏应恒短路，不产生任何回退或审计噪音。"""
    for message in ("退货政策是几天？", "我要退货", "O20261115001", "确认"):
        result = coordinator.chat("s_rules", "u_1001", message)
        assert result.trace[-1]["action"] == "话术模板"
        assert "grounding" not in result.meta
    kinds = [event["kind"] for event in coordinator.db.list_events("s_rules")]
    assert "grounding_rejected" not in kinds


def test_polish_receives_structured_facts(make_coordinator):
    captured = {}

    def capture(draft, ctx):
        captured.update(ctx or {})
        return draft  # MutatingLLM.polish 负责补上 token 数，返回 (text, tokens)

    c = make_coordinator("grounding_facts.db")
    c.llm = MutatingLLM(capture)
    c.chat("s_facts", "u_1001", "退货政策是几天？")

    assert captured["state"] == "IDLE"
    assert "clause_ids" in captured
    assert "order_id" in captured and "ticket_id" in captured


# ---------------------------------------------------------------------------
# 6. HTTP 层：meta 要透出给前端
# ---------------------------------------------------------------------------

@pytest.fixture()
def grounding_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "grounding_api.db"))
    monkeypatch.setenv("AGENT_TODAY", "2026-11-22")
    monkeypatch.setenv("AGENT_MODE", "multi")
    monkeypatch.setenv("AGENT_LLM_MODE", "rules")
    monkeypatch.setenv("AGENT_GROUNDING", "1")
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_grounding(grounding_client):
    assert grounding_client.get("/api/health").json()["grounding_enabled"] is True


def test_api_exposes_grounding_meta(grounding_client):
    from app.main import app

    app.state.coordinator.llm = MutatingLLM(
        lambda draft, ctx: re.sub(r"RT\d{8}[A-Za-z0-9]{6}", "", draft)
    )
    sid = "s_http_grounding"
    body = None
    for message in ("我要退货", "O20261115001", "确认"):
        body = grounding_client.post(
            "/api/chat",
            json={"session_id": sid, "user_id": "u_1001", "message": message},
        ).json()

    assert body["state"] == "TICKETED"
    assert body["ticket"]["ticket_id"] in body["reply"]
    assert body["meta"]["grounding"]["fallback"] is True
    assert body["trace"][-1]["action"] == "事实校验回退"


# ---------------------------------------------------------------------------
# 6. 结构性事实短语（B2 真实模型复测 t08/t09 暴露的失败模式）
#    模型润色把「交接信息」「运费由商家承担」这类结论性整句压缩掉——
#    事实没编造，但信息完整性受损，必须拦下并回退模板。
# ---------------------------------------------------------------------------

HUMAN_DRAFT = (
    "已经转人工客服，预计 1-2 分钟内接入。\n"
    "交接信息：订单 O20261115001 ｜ 问题描述「不想要了」，你不用再重复说一遍。"
)

B2_T08_COMPRESSED = "已为您转接人工客服，请稍候，马上会有客服联系您。"
B2_T09_COMPRESSED = SUMMARY_DRAFT.replace("取件：上门取件 9:00-20:00，运费由商家承担",
                                          "取件：上门取件 9:00-20:00，商家会安排")


def test_missing_handoff_phrases_rejected_b2_t08():
    """B2 t08：润色把「转人工」「交接信息」整段压缩掉 → 回退模板。"""
    result = verify(HUMAN_DRAFT, B2_T08_COMPRESSED)
    assert not result.ok
    assert "missing_phrase" in result.violations
    assert set(result.detail["missing_phrases"]) == {"转人工", "交接信息"}
    assert result.reply == HUMAN_DRAFT


def test_missing_fee_responsibility_rejected_b2_t09():
    """B2 t09：润色把「运费由商家承担」改没 → 回退模板。"""
    result = verify(SUMMARY_DRAFT, B2_T09_COMPRESSED)
    assert not result.ok
    assert "missing_phrase" in result.violations
    assert result.detail["missing_phrases"] == ["运费由商家承担"]
    assert result.reply == SUMMARY_DRAFT


def test_semantic_equivalent_fee_wording_still_rejected():
    """口径说明：模型把「运费由商家承担」换成「运费商家出」也拦。
    保守策略——宁可少一次润色，不让结论性短语被等价改写绕过校验；
    判定它是否「等价」需要语义模型，会把幻觉风险带回表达层。"""
    result = verify(SUMMARY_DRAFT, SUMMARY_DRAFT.replace("运费由商家承担", "运费商家出"))
    assert not result.ok
    assert "missing_phrase" in result.violations


def test_phrase_check_tolerates_spacing_variations():
    """compact 匹配：短语带排版空格不算丢。"""
    polished = HUMAN_DRAFT.replace("交接信息", "交接 信息")
    result = verify(HUMAN_DRAFT, polished)
    assert result.ok and result.used_polished


def test_phrases_absent_from_draft_are_not_required():
    """草稿里本来没有的短语不构成要求：普通确认流程的润色照常通过。"""
    draft = "查到订单 O20261115001，符合退货条件，回复「确认」即可创建工单。"
    polished = "亲，查到订单 O20261115001 啦，符合退货条件，回复「确认」我就帮您创建工单～"
    result = verify(draft, polished)
    assert result.ok and result.used_polished


def test_fee_phrase_variants_all_covered():
    """三种运费承担方表述都在保护清单里：谁出钱是关键结论。"""
    for phrase in ("运费由商家承担", "运费由消费者承担", "运费需自理"):
        draft = f"按政策，{phrase}。"
        result = verify(draft, "好的，已为您处理。")
        assert not result.ok
        assert result.detail["missing_phrases"] == [phrase], phrase


def test_coordinator_falls_back_when_polish_drops_handoff_info(make_coordinator):
    """端到端：转人工轮的润色丢了交接信息，回复必须回到模板。"""
    c = make_coordinator("grounding_phrase.db")
    c.llm = MutatingLLM(lambda draft, ctx: "已为您转接人工客服，请稍候。")

    result = c.chat("s_phrase", "u_1001", "转人工")

    assert result.state == "HUMAN"
    assert "交接信息" in result.reply and "转人工" in result.reply
    assert result.meta["grounding"]["fallback"] is True
    assert "missing_phrase" in result.meta["grounding"]["violations"]
