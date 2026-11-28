"""意图召回重构的回归测试：语义信号层 + 规则分类器。

旧实现靠「整句短语表」枚举说法，换个动词补语 / 倒装 / 政策问句就漏判。
这里锁住的是结构性能力（语素组合），不是某几条评测案例的原句。
"""

import pytest

from app.llm import RuleLLM, semantic_signals

LLM = RuleLLM()


@pytest.mark.parametrize(
    "text",
    [
        "我想把耳机退了",
        "能退吗 O20261115001",
        "我想退 O20261115002",
        "刚收到的耳机想退掉，能办吗",
        "耳机有瑕疵，想走售后 O20261115001",
        "东西坏了能换吗 O20261115001",
        "屏幕有裂痕，能给换一个吗 O20261115001",
    ],
)
def test_return_verb_variants_recognized_by_both_classifiers(text):
    """动词「退」+ 补语、倒装、换货、质量问题带单号：两类分类器都要识别。"""
    for focused in (True, False):
        assert LLM.classify(text, focused=focused).intent == "return", (text, focused)


@pytest.mark.parametrize(
    "text",
    [
        "退货要自己付邮费吗",
        "帮我查一下退货期限",
        "无理由退货从哪天开始算",
        "运费要我出吗",
        "退货政策是几天？",
        "几天之内可以退",
        "寄回商品的运费算谁的",
        "R-06 那笔退款一般几天能到账",
        "七天从什么时候算起",
    ],
)
def test_policy_questions_are_not_stolen_by_return_keyword(text):
    """句子里带「退货」不等于要办退货：疑问 + 政策维度词必须优先判 policy。"""
    for focused in (True, False):
        assert LLM.classify(text, focused=focused).intent == "policy", (text, focused)


@pytest.mark.parametrize(
    "text",
    [
        "查下快递到哪了 O20261120005",
        "帮我看下 O20261115001 的物流",
        "我的快递到哪了 O20261120005",
        "查一下 O20261120005 的物流状态",
    ],
)
def test_logistics_beats_bare_order_id(text):
    for focused in (True, False):
        assert LLM.classify(text, focused=focused).intent == "logistics", (text, focused)


def test_question_is_not_a_confirmation():
    """AWAIT_CONFIRM 下「可以退吗」是疑问，不能被 CONFIRM_WORDS 里的「可以」误建单。"""
    assert LLM.classify("可以退吗", state="AWAIT_CONFIRM", focused=True).intent != "confirm"
    assert LLM.classify("确认", state="AWAIT_CONFIRM", focused=True).intent == "confirm"


@pytest.mark.parametrize(
    "text",
    ["这个单子我不想要了", "耳机拆开后觉得不合适，不打算要了"],
)
def test_colloquial_return_is_recognized_in_both_modes(text):
    """不含「退」字的纯口语：B11 起两种分类器共用词表，不再人为制造召回差。"""
    assert LLM.classify(text, focused=True).intent == "return"
    assert LLM.classify(text, focused=False).intent == "return"


def test_unrelated_question_stays_unknown():
    for text in ("你们公司附近有什么好吃的", "今天天气怎么样"):
        for focused in (True, False):
            assert LLM.classify(text, focused=focused).intent == "unknown", (text, focused)


def test_signals_are_structural_not_phrase_memorization():
    policy = semantic_signals("帮我查一下退货期限", focused=True)
    assert policy["consult"] and policy["policy_topic"] and policy["policy_question"]
    assert not policy["has_order_id"]

    action = semantic_signals("能退吗 O20261115001", focused=True)
    assert action["is_question"] and action["has_order_id"]
    assert not action["policy_question"] and action["return_verb"]


def test_negation_still_wins_in_await_confirm():
    for text in ("先别确认", "再等等，先不提交", "不确认", "还没确认好"):
        assert LLM.classify(text, state="AWAIT_CONFIRM", focused=True).intent == "hesitate", text


@pytest.mark.parametrize(
    "text",
    ["屏幕有裂痕，能给换一个吗 O20261115001", "耳机有破损，我要退货 O20261115001"],
)
def test_quality_and_exchange_set_quality_reason(text):
    understanding = LLM.classify(text, focused=True)
    assert understanding.intent == "return"
    assert understanding.reason == "quality"
