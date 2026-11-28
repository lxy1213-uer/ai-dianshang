"""B12 状态型问句证据校验 + B13 引用完整性收口的回归测试。

B12：holdout4 p07「这笔 O… 的钱还能退一次吗」被真实模型判成 policy，
     答成通用退款时效、既没核单也没说「不能重复申请」。修复用两处：
     - `app/slots.py::status_question_evidence()`：状态事实 + 再次申请语素组合；
     - 理解层护栏：模型判 policy/unknown 且命中状态问句 → 规则引擎重判；
     - 规则路径同源：同样先过状态问句判定，两种分类器口径一致。
B13：holdout4 p04 答复写了「运费由商家承担」（R-05 的结论）却只引 R-02。
     修复用 `app/domain/rag.py::complete_citations()`，只加不减。

本文件全部用假模型 / 纯函数，不访问外网。
"""

from __future__ import annotations

from datetime import date

import pytest

from app.agent.coordinator import Coordinator
from app.config import Settings
from app.domain.rag import complete_citations
from app.llm import (
    OpenAICompatibleLLM,
    RuleLLM,
    needs_status_question_correction,
)
from app.slots import status_question_evidence
from app.usage import measured_usage


# --- B12 纯函数口径 --------------------------------------------------------
def test_status_question_requires_state_and_reapply_signals():
    # 状态事实 + 再次申请 → 命中
    v = status_question_evidence("已经退过款了还能再申请一次吗")
    assert v.is_status_question and v.rule == "status_reapply"
    assert v.state_hits and v.reapply_hits

    # 订单指向 + 单组语素 → 命中（p07 的真实说法）
    assert status_question_evidence("这笔 O20261114011 的钱还能退一次吗").is_status_question
    assert status_question_evidence("这个订单之前退过款").is_status_question

    # 两组都没有 → 不命中（真政策问句）
    for text in ("退款多久到账", "退货政策是几天", "运费谁承担", "我要退货"):
        assert not status_question_evidence(text).is_status_question, text


def test_status_question_negation_is_not_misread_as_reapply():
    # "还没退款" 是“未发生”，不是 "已退款" 状态事实；单看 "退款" 不算
    assert not status_question_evidence("我的退款还没到账").is_status_question


def test_needs_correction_only_touches_policy_and_unknown():
    text = "这笔订单退过款了，还能再退一次吗"
    assert needs_status_question_correction("policy", text)
    assert needs_status_question_correction("unknown", text)
    # return 已经是正确路径，不纠
    assert not needs_status_question_correction("return", text)
    assert not needs_status_question_correction("confirm", text)


def test_rule_engine_and_model_share_status_question_evidence():
    # 规则路径与模型路径走同一份证据：这句在两条路上都判 return
    text = "这笔 O20261114011 的钱还能退一次吗"
    u = RuleLLM().classify(text)
    assert u.intent == "return"
    assert u.slots["status_question_evidence"]["is_status_question"]


# --- 假模型：真实模型路径的硬护栏 -----------------------------------------
@pytest.fixture()
def make_coordinator(tmp_path):
    created: list[Coordinator] = []

    def make(**overrides):
        base = dict(
            db_path=tmp_path / f"b12_{len(created)}.db",
            frozen_today=date(2026, 11, 22),
            llm_mode="rules",
            agent_mode="multi",
            grounding_enabled=True,
            auth_secret="",
        )
        base.update(overrides)
        c = Coordinator(Settings.load(**base))
        created.append(c)
        return c

    yield make
    for c in created:
        c.close()


def _fake_policy_llm(c: Coordinator, intent_json: str):
    """理解层恒定返回指定意图（模拟模型误判），表达层回原文。"""
    c.llm = OpenAICompatibleLLM("https://example.invalid/v1", "key", "test-model")

    def fake_chat(messages, temperature=0.2, **kwargs):
        system = messages[0]["content"] if messages else ""
        if "意图识别器" in system:
            return intent_json, measured_usage(50, 10, 60, 1.0, "test-model")
        lines = (messages[-1]["content"] if messages else "").split("\n")
        return (lines[1] if len(lines) > 1 else ""), measured_usage(50, 10, 60, 1.0, "test-model")

    c.llm._chat = fake_chat
    return c.llm


def test_model_policy_misclassification_is_corrected_to_return(make_coordinator):
    """p07 复刻：模型说 policy，护栏应降级为规则重判并走核单。"""
    c = make_coordinator()
    _fake_policy_llm(c, '{"intent":"policy","order_id":"O20261114011","reason":"no_reason","confidence":0.8}')
    result = c.chat("b12-p07", "u_1001", "这笔 O20261114011 的钱还能退一次吗")
    assert "不能重复申请" in result.reply
    assert result.state == "HUMAN"
    assert any(c["clause_id"] == "R-06" for c in result.citations)
    # 护栏留痕：从 policy 纠到 return
    trace_intent = [
        step for step in result.trace
        if step.get("detail", {}).get("intent_correction")
    ]
    assert trace_intent
    correction = trace_intent[0]["detail"]["intent_correction"]
    assert correction["from"] == "policy" and correction["to"] == "return"
    assert correction["rule"] == "status_question_guard"


def test_true_policy_question_is_not_corrected(make_coordinator):
    """真政策问句（无状态事实+再申请语素）不该被护栏拉去核单。"""
    c = make_coordinator()
    _fake_policy_llm(c, '{"intent":"policy","order_id":null,"reason":"no_reason","confidence":0.8}')
    result = c.chat("b12-policy", "u_1001", "退款多久才能到账")
    assert result.state == "IDLE"
    assert "到账" in result.reply
    assert not any(
        step.get("detail", {}).get("intent_correction", {}).get("rule") == "status_question_guard"
        for step in result.trace
    )


# --- B13 引用完整性 ---------------------------------------------------------
def test_complete_citations_adds_r05_for_freight_conclusion():
    # p04 复刻：只引 R-02，但答复写了 R-05 的结论
    assert complete_citations("退货运费由商家承担", ["R-02"]) == ["R-02", "R-05"]
    # 已有则不重复
    assert complete_citations("运费由商家承担", ["R-02", "R-05"]) == ["R-02", "R-05"]
    # dict 形式的引用对象也能解析
    cites = [{"clause_id": "R-02"}, {"clause_id": "R-05"}]
    assert complete_citations("运费需自理", cites) == ["R-02", "R-05"]


def test_complete_citations_does_not_touch_generic_words():
    # 通用词（全额退款 / 转人工 / 签收）不触发补引，避免给撤单答复补出多余条款
    assert complete_citations("未发货可全额退款", ["R-04"]) == ["R-04"]
    assert complete_citations("转人工处理", []) == []
    assert complete_citations("七天无理由", ["R-01"]) == ["R-01"]


def test_citation_completion_runs_before_grounding(make_coordinator):
    """答复写了运费结论时，最终引用的 clause_ids 含 R-05。"""
    c = make_coordinator()
    result = c.chat("b13-freight", "u_1001", "运费到底是商家出还是我自己出")
    cited = {x["clause_id"] for x in result.citations}
    assert "R-05" in cited
    assert "运费由商家承担" in result.reply


# --- 样本纪律 ---------------------------------------------------------------
def test_holdout5_zero_overlap_and_not_in_full_set():
    from app.eval.runner import (
        HOLDOUT5_PATH,
        load_all_cases,
        load_cases,
        load_dev2,
        load_holdout2,
        load_holdout3,
        load_holdout4,
        load_holdout5,
    )

    cases = load_holdout5()
    assert len(cases) == 16 and len({c["id"] for c in cases}) == 16
    assert all(c["split"] == "holdout5" for c in cases)
    assert HOLDOUT5_PATH.exists()
    assert len(load_all_cases()) == 48  # 历史口径不变

    def seqs(items):
        return {"||".join(c["messages"]) for c in items}

    new = seqs(cases)
    for loader in (load_cases, load_dev2, load_holdout2, load_holdout3, load_holdout4):
        assert not new & seqs(loader()), loader.__name__

    sources = "\n".join(
        open(p, encoding="utf-8").read()
        for p in ("app/llm.py", "app/slots.py", "app/domain/rag.py", "app/agent/advice.py")
    )
    for case in cases:
        for message in case["messages"]:
            if len(message) < 8 or not any("\u4e00" <= ch <= "\u9fff" for ch in message):
                continue
            assert message not in sources, message
