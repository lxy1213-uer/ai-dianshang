"""B11 / B12 / B13：单 Agent 口语召回、撤单意图、商品名完整性、缺陷词表同源。

来源按 README「下一步候选」①②③：holdout3 首次基线暴露的口语召回缺口、
B6 x05 润色丢商品名、「蓝莓发霉了」在规则侧被判查单。修复只基于这几条
**已公开登记**的根因；用于验收的 holdout4 另行新建，未参与修复。
"""

from __future__ import annotations

import pytest

from app.agent.coordinator import Coordinator
from app.eval.runner import (
    HOLDOUT4_PATH,
    load_all_cases,
    load_cases,
    load_dev2,
    load_holdout2,
    load_holdout3,
    load_holdout4,
    load_llm_sample,
    load_llm_sample2,
    load_llm_sample3,
    load_llm_sample4,
    load_llm_sample5,
    load_llm_sample6,
    load_llm_sample7,
)
from app.grounding import verify
from app.llm import RuleLLM
from app.slots import defect_evidence

LLM = RuleLLM()

# ---------------------------------------------------------------------------
# B11 · 口语诉求：两种分类器共用词表，不再人为制造召回差
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "这双鞋穿两次就不想要了",
        "耳机用不上了，帮我走一下售后流程",
        "这条围巾闲置着也是吃灰，能走售后吗",
        "当时拍错了，多买了一个",
        "这笔钱还能再申请一次吗",
    ],
)
def test_colloquial_return_is_recognized_without_the_character_return(text):
    for focused in (True, False):
        assert LLM.classify(text, "IDLE", focused).intent == "return", (text, focused)


@pytest.mark.parametrize("text", ["键盘 O20261121007 还没发出去，能不能撤掉", "先别发了，帮我撤掉这单 O20261121007"])
def test_unshipped_cancel_is_recognized(text):
    for focused in (True, False):
        assert LLM.classify(text, "IDLE", focused).intent == "cancel", (text, focused)


@pytest.mark.parametrize(
    "text",
    ["你们门店周末开门吗", "今天天气怎么样", "退货运费谁承担"],
)
def test_widened_words_do_not_swallow_unrelated_or_policy_questions(text):
    for focused in (True, False):
        intent = LLM.classify(text, "IDLE", focused).intent
        assert intent in ("unknown", "policy"), (text, focused, intent)


def test_policy_question_with_defect_is_still_policy():
    assert LLM.classify("东西坏了寄回去，路费谁出", "IDLE", False).intent == "policy"


def test_defect_statement_without_order_id_is_an_aftersale_request():
    """「蓝莓有一盒发霉了」既无「退」字也无单号，仍应进入售后流程。"""
    for focused in (True, False):
        u = LLM.classify("蓝莓有一盒发霉了", "IDLE", focused)
        assert u.intent == "return"
        assert u.reason == "quality"


def test_cancel_with_unshipped_order_explains_r04_without_making_ticket(make_coordinator):
    c = make_coordinator()
    result = c.chat("b11-cancel", "u_1001", "键盘 O20261121007 还没发出去，能不能撤掉")
    assert result.state == "IDLE"
    assert "取消订单" in result.reply
    assert [x["clause_id"] for x in result.citations] == ["R-04"]
    assert result.ticket is None
    assert result.meta["cancel_guide"] == "unshipped"


def test_cancel_without_order_keeps_the_original_meaning(make_coordinator):
    c = make_coordinator()
    result = c.chat("b11-cancel-none", "u_1001", "算了")
    assert "已取消本次操作" in result.reply
    assert result.citations == []


def test_cancel_on_shipped_order_does_not_promise_a_direct_cancellation(make_coordinator):
    c = make_coordinator()
    result = c.chat("b11-cancel-shipped", "u_1001", "O20261120005 帮我撤掉")
    assert "不能按未发货直接取消订单" in result.reply


def test_colloquial_opening_answers_the_policy_before_asking_for_order(make_coordinator):
    c = make_coordinator()
    result = c.chat("b11-open", "u_1001", "耳机用不上了，帮我走一下售后流程")
    assert result.meta["advice"] == "no_reason"
    assert result.meta["advice_basis"] == "situation"
    assert "七天无理由" in result.reply
    assert "订单号" in result.reply


def test_full_colloquial_journey_still_creates_one_ticket(make_coordinator):
    c = make_coordinator()
    c.chat("b11-journey", "u_1001", "这副耳机用不上了，帮我走一下售后流程")
    c.chat("b11-journey", "u_1001", "O20261115001")
    result = c.chat("b11-journey", "u_1001", "没问题，提交吧")
    assert result.state == "TICKETED"
    assert result.ticket and result.ticket["ticket_id"].startswith("RT")


# ---------------------------------------------------------------------------
# B13 · 意图触发与 reason 证据同源（缺陷表只有一个事实来源）
# ---------------------------------------------------------------------------
def test_defect_trigger_shares_the_reason_evidence_table():
    for text in ("蓝莓有一盒发霉了", "刻歪的杯子", "洗一次就掉色了"):
        assert defect_evidence(text), text
        assert LLM.classify(text, "IDLE", False).intent == "return", text


def test_negated_defect_does_not_trigger_the_quality_channel():
    u = LLM.classify("没有破损，就是不喜欢", "IDLE", False)
    assert u.reason == "no_reason"


# ---------------------------------------------------------------------------
# B12 · 商品名是结构性事实，润色删除即整体回退
# ---------------------------------------------------------------------------
_DRAFT = "查到订单 O20261115001 · 无线蓝牙耳机 Pro\n金额 ¥299.00\n判定：符合七天无理由退货条件"


def test_verify_rejects_polish_that_drops_the_product_title():
    polished = _DRAFT.replace("无线蓝牙耳机 Pro", "")
    result = verify(_DRAFT, polished, {"order_title": "无线蓝牙耳机 Pro"})
    assert not result.ok
    assert "missing_phrase" in result.violations
    assert "无线蓝牙耳机 Pro" in result.detail["missing_phrases"]
    assert result.reply == _DRAFT


def test_verify_accepts_polish_that_keeps_the_product_title():
    polished = _DRAFT.replace("判定：", "结论：")
    result = verify(_DRAFT, polished, {"order_title": "无线蓝牙耳机 Pro"})
    assert result.ok
    assert result.used_polished


def test_verify_does_not_invent_a_title_when_draft_has_none():
    polished = "查到订单 O20261115001，金额 ¥299.00"
    result = verify(
        "查到订单 O20261115001，金额 ¥299.00",
        polished,
        {"order_title": "无线蓝牙耳机 Pro"},
    )
    assert result.ok


def test_coordinator_puts_both_order_titles_into_polish_facts(make_coordinator):
    c = make_coordinator()
    output = type(
        "O",
        (),
        {
            "state": __import__("app.agent.flow", fromlist=["State"]).State.AWAIT_ORDER,
            "slots": {"order": {"title": "真丝内衣套装"}, "pending_order_title": "定制刻字保温杯"},
            "ticket": None,
            "citations": [],
        },
    )()
    facts = c._build_facts(output)
    assert facts["order_title"] == "真丝内衣套装"
    assert facts["pending_order_title"] == "定制刻字保温杯"


def test_end_to_end_dropped_title_falls_back_to_the_template(make_coordinator):
    from app.llm import OpenAICompatibleLLM
    from app.usage import measured_usage

    c = make_coordinator()
    c.llm = OpenAICompatibleLLM("https://example.invalid/v1", "key", "test-model")

    def fake_chat(messages, temperature=0.2, **kwargs):
        system = messages[0]["content"] if messages else ""
        if "意图识别器" in system:
            return (
                '{"intent":"return","order_id":null,"reason":"no_reason","confidence":0.9}',
                measured_usage(10, 10, 10, 1.0, "test-model"),
            )
        prompt = messages[-1]["content"]
        marker = "待改写话术（唯一事实来源，必须原样保留其中所有数字、单号、条款编号与结论）：\n"
        draft = prompt.split(marker, 1)[1].split("\n\n背景数据")[0]
        return (
            draft.replace("（无线蓝牙耳机 Pro）", ""),
            measured_usage(10, 10, 10, 1.0, "test-model"),
        )

    c.llm._chat = fake_chat
    result = c.chat("b12-e2e", "u_1001", "耳机用不上了，走售后流程")
    assert "无线蓝牙耳机 Pro" in result.reply
    assert result.meta["grounding"]["fallback"] is True
    assert "missing_phrase" in result.meta["grounding"]["violations"]


# ---------------------------------------------------------------------------
# holdout4 纪律：B11/B12 修复后才建，不与任何旧集重叠、不进 48 条全量
# ---------------------------------------------------------------------------
def _sequences(cases: list[dict]) -> set[str]:
    return {"||".join(case["messages"]) for case in cases}


def test_holdout4_is_a_clean_new_split():
    cases = load_holdout4()
    assert len(cases) == 16
    assert len({case["id"] for case in cases}) == 16
    assert all(case["split"] == "holdout4" for case in cases)
    assert not {case["id"] for case in cases} & {case["id"] for case in load_all_cases()}
    assert len(load_all_cases()) == 48


def test_holdout4_has_zero_overlap_with_every_existing_set():
    new = _sequences(load_holdout4())
    for loader in (
        load_cases, load_dev2, load_holdout2, load_holdout3,
        load_llm_sample, load_llm_sample2, load_llm_sample3, load_llm_sample4,
        load_llm_sample5, load_llm_sample6, load_llm_sample7,
    ):
        assert not new & _sequences(loader()), loader.__name__


def test_holdout4_phrases_were_not_copied_into_the_rules():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sources = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in ("app/llm.py", "app/agent/advice.py", "app/agent/flow.py", "app/slots.py")
    )
    for case in load_holdout4():
        for message in case["messages"]:
            if len(message) < 8 or not any("\u4e00" <= ch <= "\u9fff" for ch in message):
                continue
            assert message not in sources, message


def test_holdout4_loader_points_to_its_own_file():
    assert HOLDOUT4_PATH.name == "cases_holdout4.json"
    assert HOLDOUT4_PATH.exists()
