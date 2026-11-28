"""B7：reason 槽位的证据校验（类别 × 问题描述的语义错配）。

B6 的 x09 暴露的是一条**承诺性错误**：用户说「保温杯刻字了能退吗」，理解层把「刻字」
判成质量问题（reason=quality），代码照单全收，于是绕过「定制类目不支持七天无理由」，
把 7 天期限放宽到 15 天，还宣称「运费由商家承担」。

根因是槽位缺证据约束：intent 有 B1 的状态硬护栏，reason 却由模型自由填写。
修复见 `app/slots.py`——只做减法（无证据则降级），不升级、不碰 intent。

这里分四层锁住：
1. 纯函数口径（证据 / 属性 / 状态 / 否定 / 不跨标点 / 只减不加）；
2. 规则分类器（降级并在 slots 留痕；有证据时不被误伤）；
3. 理解层硬护栏（假模型说 quality → 降级，usage 仍记实测）；
4. 端到端（同一句话在修复前会进待确认态，修复后必须转人工且引用 R-03）。
"""

from __future__ import annotations

from datetime import date

import pytest

from app.agent.coordinator import Coordinator
from app.config import Settings
from app.llm import OpenAICompatibleLLM, RuleLLM
from app.slots import (
    RULE_ATTRIBUTE_ONLY,
    RULE_NEGATED,
    RULE_NO_EVIDENCE,
    RULE_NOT_CLAIMED,
    RULE_SUPPORTED,
    defect_evidence,
    validate_reason,
)
from app.usage import measured_usage

PERSONALIZED_CUP = "O20261118004"  # 定制刻字保温杯，签收 2 天
EARPHONE = "O20261115001"          # 数码配件，签收 7 天


@pytest.fixture()
def make_coordinator(tmp_path):
    created: list[Coordinator] = []

    def make(**overrides):
        base = dict(
            db_path=tmp_path / f"b7_{len(created)}.db",
            frozen_today=date(2026, 11, 22),
            llm_mode="rules",
            agent_mode="multi",
            grounding_enabled=True,
            auth_secret="",
        )
        base.update(overrides)
        coordinator = Coordinator(Settings.load(**base))
        created.append(coordinator)
        return coordinator

    yield make
    for coordinator in created:
        coordinator.close()


def _fake_understand(c: Coordinator, payload: str):
    """理解层恒定返回 payload；表达层返回原文（与既有护栏测试同一写法）。"""
    c.llm = OpenAICompatibleLLM("https://example.invalid/v1", "key", "test-model")

    def fake_chat(messages, temperature=0.2, **kwargs):
        system = messages[0]["content"] if messages else ""
        if "意图识别器" in system:
            return payload, measured_usage(50, 10, 60, 1.0, "test-model")
        lines = (messages[-1]["content"] if messages else "").split("\n")
        return (lines[1] if len(lines) > 1 else ""), measured_usage(50, 10, 60, 1.0, "test-model")

    c.llm._chat = fake_chat


# --- 1. 纯函数口径 --------------------------------------------------------
def test_personalization_is_attribute_not_defect():
    """「刻字」描述的是定制属性，不是缺陷——这是 B6 x09 的根因。"""
    verdict = validate_reason("quality", "O20261118004 保温杯刻字了能退吗")
    assert verdict.reason == "no_reason"
    assert verdict.rule == RULE_ATTRIBUTE_ONLY
    assert verdict.attributes == ("刻字",)
    assert verdict.evidence == ()


def test_customization_done_badly_is_a_real_defect():
    """属性词不构成证据，但「刻歪了」是——不能因为句子里有「刻字」就一律放过。"""
    verdict = validate_reason("quality", "杯子刻字刻歪了，字都糊了")
    assert verdict.reason == "quality"
    assert verdict.rule == RULE_SUPPORTED
    assert "刻歪" in verdict.evidence


def test_usage_condition_is_not_defect_evidence():
    """已拆封 / 洗过 / 穿过属于商品现状，不是质量缺陷。"""
    for text in ("收到时包装已拆封", "内衣洗过了", "这件穿过了"):
        verdict = validate_reason("quality", text)
        assert verdict.reason == "no_reason", text
        assert verdict.rule == RULE_ATTRIBUTE_ONLY, text


def test_plain_defect_keeps_quality():
    for text in ("耳机坏了", "屏幕碎了", "右耳没声音", "杯子漏水了", "收到的是错发的"):
        verdict = validate_reason("quality", text)
        assert verdict.reason == "quality", text
        assert verdict.rule == RULE_SUPPORTED, text


def test_negated_defect_is_not_evidence():
    verdict = validate_reason("quality", "这个没有破损，就是不喜欢")
    assert verdict.reason == "no_reason"
    assert verdict.rule == RULE_NEGATED
    assert defect_evidence("这个没有破损") == ()


def test_negated_meta_phrase_is_not_evidence():
    """「不是质量问题」里的「质量」不能当成质量问题。"""
    verdict = validate_reason("quality", "不是质量问题，就是不想要了")
    assert verdict.reason == "no_reason"
    assert verdict.rule == RULE_NEGATED


def test_one_negated_hit_does_not_void_the_other():
    """按命中点逐个判否定，不按整句判：漏发仍然成立。"""
    verdict = validate_reason("quality", "不是质量问题，是漏发了")
    assert verdict.reason == "quality"
    assert verdict.evidence == ("漏发",)


def test_negation_does_not_cross_punctuation():
    """「没有破损，屏幕裂了」：前一句的否定不作用到后一句。"""
    verdict = validate_reason("quality", "没有破损，屏幕裂了")
    assert verdict.reason == "quality"
    assert verdict.evidence == ("裂了",)


def test_guard_never_grants_a_channel():
    """只做减法：模型说 no_reason 时，本层不会擅自升级成 quality。"""
    for text in ("屏幕碎了", "耳机坏了，右耳没声音"):
        verdict = validate_reason("no_reason", text)
        assert verdict.reason == "no_reason"
        assert verdict.rule == RULE_NOT_CLAIMED
        assert verdict.corrected is False


def test_unknown_reason_value_is_normalised_to_no_reason():
    verdict = validate_reason("quality_issue", "耳机坏了")
    assert verdict.reason == "no_reason"
    assert verdict.rule == RULE_NOT_CLAIMED


def test_empty_text_has_no_evidence():
    verdict = validate_reason("quality", "")
    assert verdict.reason == "no_reason"
    assert verdict.rule == RULE_NO_EVIDENCE


# --- 2. 规则分类器 --------------------------------------------------------
def test_rules_classifier_no_longer_sees_bare_quality_word_as_evidence():
    """B8：规则侧的质量判定与缺陷证据表同源，「质量怎么样」不再被当成缺陷。

    原先 detect_reason 有一张独立小词表（含裸词「质量」），「这个质量怎么样，能退吗」
    会被判成 quality，再被槽位校验降级、记一条 reason_correction。词表统一后，
    规则侧根本不会产生无证据的 quality，也就没有「先错判再降级」这一跳。
    降级留痕的路径仍然存在，但只对模型输出生效（见下方理解层护栏用例）。
    """
    understanding = RuleLLM().classify("O20261115001 这个质量怎么样，能退吗", "IDLE")
    assert understanding.reason == "no_reason"
    assert "reason_correction" not in understanding.slots


def test_rules_classifier_accepts_defect_evidence_from_the_same_table():
    """同源之后，B7 缺陷表里更细的说法（刻歪 / 掉色 / 发霉）规则侧也能认。"""
    for text in ("O20261118004 杯子刻歪了能退吗", "O20261115002 洗了一次就掉色", "蓝莓发霉了"):
        assert RuleLLM().classify(text, "IDLE").reason == "quality", text


def test_rules_classifier_no_correction_key_when_supported():
    understanding = RuleLLM().classify("O20261115001 耳机坏了想退货", "IDLE")
    assert understanding.reason == "quality"
    assert understanding.slots == {}


# --- 3. 理解层硬护栏 ------------------------------------------------------
def test_llm_quality_claim_without_evidence_is_downgraded():
    coordinator_llm = OpenAICompatibleLLM("https://example.invalid/v1", "key", "test-model")

    def fake_chat(messages, temperature=0.2, **kwargs):
        return (
            '{"intent":"return","order_id":"O20261118004","reason":"quality","confidence":0.9}',
            measured_usage(50, 10, 60, 1.0, "test-model"),
        )

    coordinator_llm._chat = fake_chat
    understanding, usage = coordinator_llm.classify_with_usage(
        "O20261118004 保温杯刻字了能退吗", "IDLE"
    )

    assert understanding.intent == "return"           # 意图不动，归 B1 管
    assert understanding.reason == "no_reason"        # 结论被降级
    assert understanding.source == "llm-slot-guard"
    assert understanding.slots["reason_correction"]["attributes"] == ["刻字"]
    # 真实调用确实发生了，用量不能被修正抹掉
    assert usage.measured is True and usage.total_tokens == 60


def test_llm_quality_claim_with_evidence_passes_through():
    coordinator_llm = OpenAICompatibleLLM("https://example.invalid/v1", "key", "test-model")

    def fake_chat(messages, temperature=0.2, **kwargs):
        return (
            '{"intent":"return","order_id":"O20261118004","reason":"quality","confidence":0.9}',
            measured_usage(50, 10, 60, 1.0, "test-model"),
        )

    coordinator_llm._chat = fake_chat
    understanding, usage = coordinator_llm.classify_with_usage("O20261118004 杯子漏水了", "IDLE")

    assert understanding.reason == "quality"
    assert understanding.source == "llm"
    assert understanding.slots == {}
    assert usage.measured is True


def test_state_guard_and_slot_guard_stack():
    """两层护栏独立生效：意图归位（state_guard）+ 结论降级（slot_guard）。"""
    coordinator_llm = OpenAICompatibleLLM("https://example.invalid/v1", "key", "test-model")

    def fake_chat(messages, temperature=0.2, **kwargs):
        return (
            '{"intent":"hesitate","order_id":"O20261118004","reason":"quality","confidence":0.7}',
            measured_usage(50, 10, 60, 1.0, "test-model"),
        )

    coordinator_llm._chat = fake_chat
    understanding, _ = coordinator_llm.classify_with_usage("O20261118004 保温杯刻字了能退吗", "IDLE")

    assert understanding.intent == "return"
    assert understanding.source == "llm-state-guard"
    assert understanding.slots["intent_correction"]["from"] == "hesitate"
    assert understanding.reason == "no_reason"


# --- 4. 端到端 ------------------------------------------------------------
def test_end_to_end_personalized_item_goes_to_human(make_coordinator):
    """修复前：AWAIT_CONFIRM + R-02/R-05（错给 15 天通道）；修复后：HUMAN + R-03。"""
    c = make_coordinator()
    _fake_understand(
        c, '{"intent":"return","order_id":"O20261118004","reason":"quality","confidence":0.9}'
    )
    result = c.chat("b7-e2e-1", "u_1001", "O20261118004 保温杯刻字了能退吗")

    assert result.state == "HUMAN"
    assert "不支持七天无理由退货" in result.reply
    assert "R-03" in [c["clause_id"] for c in result.citations]
    assert result.ticket is None
    step = result.trace[0]["detail"]
    assert step["reason"] == "no_reason"
    assert step["reason_correction"]["rule"] == RULE_ATTRIBUTE_ONLY


def test_end_to_end_real_defect_still_takes_quality_channel(make_coordinator):
    """反向保护：真实缺陷不能被降级误伤，仍走 15 天质量通道。"""
    c = make_coordinator()
    _fake_understand(
        c, '{"intent":"return","order_id":"O20261118004","reason":"quality","confidence":0.9}'
    )
    result = c.chat("b7-e2e-2", "u_1001", "O20261118004 杯子里有一股异味，能退吗")

    assert result.state == "AWAIT_CONFIRM"
    assert "符合质量问题退货条件" in result.reply
    assert "R-02" in [c["clause_id"] for c in result.citations]
    assert "reason_correction" not in result.trace[0]["detail"]


def test_end_to_end_negated_defect_does_not_get_quality_window(make_coordinator):
    c = make_coordinator()
    _fake_understand(
        c, '{"intent":"return","order_id":"O20261118004","reason":"quality","confidence":0.9}'
    )
    result = c.chat("b7-e2e-3", "u_1001", "O20261118004 杯子没有破损，就是不喜欢这个刻字")

    assert result.state == "HUMAN"
    assert "不支持七天无理由退货" in result.reply
    assert result.ticket is None


# --- 5. B7 样本纪律 -------------------------------------------------------
def _seq(case: dict) -> str:
    return "||".join(case["messages"])


def test_b7_sample_structure():
    from app.eval.runner import LLM_SAMPLE7_PATH, load_llm_sample7

    sample = load_llm_sample7()
    assert len(sample) == 11
    assert sum(len(c["messages"]) for c in sample) == 12
    assert all(c["split"] == "llm_sample7" for c in sample)
    assert len({c["id"] for c in sample}) == 11
    assert LLM_SAMPLE7_PATH.name == "cases_llm_sample7.json"
    for case in sample:
        assert len(case["steps"]) == len(case["messages"]), case["id"]
        assert case["expect"] and case["facts"] and case["forbid"], case["id"]
        assert case.get("focus"), case["id"]


def test_b7_sample_keeps_the_paired_design():
    """只测减法的护栏最容易把召回一起砍掉，样本必须成对。"""
    from app.eval.runner import load_llm_sample7

    sample = {c["id"]: c for c in load_llm_sample7()}
    downgraded = ("y01", "y02", "y03", "y05")   # 不该给质量通道
    kept = ("y04", "y06", "y07")                # 必须给质量通道

    def step_phrases(cid):
        # facts 只对最终回复生效（y04 第二轮是建单确认），逐步断言在 steps 里
        return [p for step in sample[cid]["steps"] for p in step.get("reply_contains", [])]

    for cid in downgraded:
        assert "不支持七天无理由退货" in sample[cid]["facts"], cid
        assert "符合质量问题退货条件" in sample[cid]["forbid"], cid
    for cid in kept:
        assert "符合质量问题退货条件" in step_phrases(cid), cid


def test_b7_zero_overlap_with_all_earlier_rounds():
    from app.eval.runner import (
        load_llm_sample, load_llm_sample2, load_llm_sample3, load_llm_sample4,
        load_llm_sample5, load_llm_sample6, load_llm_sample7,
    )

    b7 = load_llm_sample7()
    earlier = (
        load_llm_sample() + load_llm_sample2() + load_llm_sample3()
        + load_llm_sample4() + load_llm_sample5() + load_llm_sample6()
    )
    assert not ({c["id"] for c in b7} & {c["id"] for c in earlier})
    earlier_seqs = {_seq(c) for c in earlier}
    for case in b7:
        assert _seq(case) not in earlier_seqs, case["id"]


def test_b7_never_touches_holdout2_and_stays_out_of_the_48():
    from app.eval.runner import load_all_cases, load_holdout2, load_llm_sample7

    b7 = load_llm_sample7()
    h2 = load_holdout2()
    assert not ({c["id"] for c in b7} & {c["id"] for c in h2})
    assert all(not c["source_case"].startswith("k") for c in b7)
    h2_seqs = {_seq(c) for c in h2}
    for case in b7:
        assert _seq(case) not in h2_seqs, case["id"]
    # 定向探针不并入 48 条全量口径
    assert len(load_all_cases()) == 48


def test_b7_sample_passes_on_rule_side():
    """规则侧必须 100%：探针集不是用来量规则引擎的，锚点错了就测不出模型退化。"""
    from app.eval.runner import load_llm_sample7, run

    report = run("multi", cases=load_llm_sample7(), settings_overrides={"llm_mode": "rules"})
    assert report["success_rate"] == 100.0
