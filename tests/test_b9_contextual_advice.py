"""B9：追问要承接上文，情形不明时用条款原文答，长答案不能被压成一句话。

背景（用户实测截图）：问「保温杯刻字能退吗」，得到的是「好的，请提供订单号」——
用户问的那件事一个字没答。B8 把「先答题再要单」做了出来，但还留着三处「固定答案」：

1. 追问不接上文——「为什么不能退」被当成新问题，重新甩一遍「两条通道」；
2. 情形不明只能套通用话术——问「吊牌要留着吗」，明明条款里有凭证要求却不引用；
3. 润色层可能把一屏答案压成一句——事实没编错，但用户问的内容被删干净了。

这里分五层锁住：追问判定 / 上下文承接（并区分「为什么」和「能不能」）/
条款检索作答 / 压缩硬闸 / 端到端对话。
"""

from __future__ import annotations

from datetime import date

import pytest

from app.agent.advice import build_advice, is_followup
from app.domain.rag import Clause, build_index
from app.grounding import verify

# ---------------------------------------------------------------------------
# 1. 追问判定：短 + 指代词才算「接着说上一句」
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    ["为什么不能退", "那还能退吗", "这算什么规定", "凭什么", "到底怎么算完好", "那怎么办"],
)
def test_referential_short_questions_are_followups(message):
    assert is_followup(message)


@pytest.mark.parametrize(
    "message",
    [
        "",
        "保温杯刻字能退吗",          # 带情形词的新问题（长）
        "我上周买的那个蓝色的保温杯刻了字想退",  # 长句里的「那」不算指代
        "你们几点上班",              # 新话题，没有指代词
    ],
)
def test_new_topics_are_not_followups(message):
    assert not is_followup(message)


# ---------------------------------------------------------------------------
# 2. 上下文承接：接着上一轮的情形答，并且「为什么」≠「能不能」
# ---------------------------------------------------------------------------
def test_followup_uses_previous_topic_instead_of_generic_blurb():
    advice = build_advice("为什么不能退", context_key="custom_only", context_evidence="刻字")

    assert advice.key == "custom_only"
    assert advice.basis == "context"
    assert advice.contextual is True
    assert "刻字" in advice.body                 # 复述他问的那件事
    assert "两条通道" not in advice.body          # 不再退回通用话术
    assert "R-03" in advice.clauses


def test_why_and_can_questions_get_different_answers():
    why = build_advice("为什么不能退", context_key="custom_only", context_evidence="刻字")
    can = build_advice("那还能退吗", context_key="custom_only", context_evidence="刻字")

    assert why.body != can.body
    # 「为什么」回答判定依据（按类目排除），「能不能」回答还有没有别的路（质量通道）
    assert "按类目排除" in why.body
    assert "质量" in can.body


def test_followup_keeps_the_evidence_from_context_when_this_turn_has_none():
    """「为什么不能退」里没有「刻字」二字，也要能说出他问的是哪件事。"""
    advice = build_advice("为什么不能退", context_key="used", context_evidence="洗过")
    assert "洗过" in advice.body
    assert "二次销售" in advice.body


def test_unrelated_new_question_does_not_hijack_previous_topic():
    """上一轮聊过刻字，这一句问的是新话题 —— 不能拿旧话题糊上去。"""
    advice = build_advice("你们几点上班", context_key="custom_only", context_evidence="刻字")
    assert advice.key == "unknown"
    assert advice.contextual is False


# ---------------------------------------------------------------------------
# 3. 情形不明：引用真实条款，而不是固定话术
# ---------------------------------------------------------------------------
def test_unknown_situation_answers_from_retrieved_clause():
    advice = build_advice(
        "吊牌要留着吗",
        clause_hits=[("R-01", "七天无理由退货", "凭证要求：需保持吊牌、包装、赠品完整")],
    )

    assert advice.basis == "clauses"
    assert advice.key == "unknown"
    assert "R-01" in advice.body
    assert "吊牌" in advice.body
    assert "两条通道" not in advice.body
    assert advice.clauses == ("R-01",)
    assert advice.informative is True


def test_unknown_without_any_clause_hit_is_honest_not_invented():
    advice = build_advice("你们几点上班", clause_hits=[])

    assert advice.basis == "generic"
    assert advice.informative is False   # 允许上层换成「查不到 + 转人工」的说法
    assert "两条通道" in advice.body      # 兜底仍是政策内的两种情况，不编造


def test_concrete_situation_wins_over_retrieval():
    """原话里有情形词时按情形答，检索只用来兜「情形不明」。"""
    advice = build_advice(
        "杯子洗过了还能退吗",
        clause_hits=[("R-06", "退款时效", "余额支付实时到账")],
    )
    assert advice.basis == "situation"
    assert advice.key == "used"


# ---------------------------------------------------------------------------
# 4. 条款摘句：挑回答这个问句的那一句
# ---------------------------------------------------------------------------
def test_clause_salient_picks_the_sentence_that_answers_the_question():
    from app.config import Settings

    index = build_index(Settings.load().policy_path)
    clause = next(c for c in index.clauses if c.clause_id == "R-01")

    assert "吊牌" in clause.salient("吊牌要留着吗")
    assert "7 天" in clause.salient("几天内可以退")


def test_salient_falls_back_to_text_when_clause_is_one_sentence():
    clause = Clause("R-99", "短条款", "一句话说完，没有分句。")
    assert clause.salient("随便问") != ""


# ---------------------------------------------------------------------------
# 5. 压缩硬闸：长答案不能被润色压成一句话
# ---------------------------------------------------------------------------
_LONG_DRAFT = (
    "你问的是「刻字」这种情况，先给结论：\n"
    "· 定制 / 刻字的商品不适用七天无理由退货\n"
    "· 定制类（刻字、刻名、印字）在 R-03 的排除清单里，所以「刻了字不想要了」不能无理由退。\n"
    "· 但如果商品本身有质量问题（刻坏、破损、故障），仍按质量问题处理：15 天内可退换。\n\n"
    "订单号发我就能核你这单的具体期限（例如 O20261115001）。"
)


def test_polish_that_squeezes_the_answer_into_one_line_is_rejected():
    result = verify(_LONG_DRAFT, "好的，请提供订单号，例如 O20261115001。")

    assert result.ok is False
    assert "information_loss" in result.violations
    assert result.reply == _LONG_DRAFT          # 整体回退，不拼接
    assert result.detail["keep_ratio"] < 0.55


def test_mild_rewrite_still_passes():
    polished = _LONG_DRAFT.replace("先给结论：", "先给结论——").replace("所以", "因此")

    result = verify(_LONG_DRAFT, polished)
    assert result.ok is True
    assert result.used_polished is True


def test_short_draft_is_exempt_from_compression_check():
    draft = "查到了订单 O20261115001，已签收。"
    result = verify(draft, "查到订单 O20261115001，已签收。")
    assert "information_loss" not in result.violations


# ---------------------------------------------------------------------------
# 6. 端到端：截图里的那段对话
# ---------------------------------------------------------------------------
@pytest.fixture()
def make_coordinator(tmp_path):
    from app.agent.coordinator import Coordinator
    from app.config import Settings

    created: list[Coordinator] = []

    def make(**overrides):
        base = dict(
            db_path=tmp_path / f"b9_{len(created)}.db",
            frozen_today=date(2026, 11, 27),
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


def test_end_to_end_followup_continues_the_topic(make_coordinator):
    c = make_coordinator()
    first = c.chat("b9-e2e-1", "u_1001", "保温杯刻字能退吗")
    assert first.state == "AWAIT_ORDER"
    assert first.meta["advice"] == "custom_only"

    second = c.chat("b9-e2e-1", "u_1001", "为什么不能退")
    assert second.state == "AWAIT_ORDER"
    assert second.meta["advice_basis"] == "context"
    assert "刻字" in second.reply
    assert "两条通道" not in second.reply
    assert "R-03" in [item["clause_id"] for item in second.citations]


def test_end_to_end_unknown_question_answers_with_the_clause(make_coordinator):
    c = make_coordinator()
    result = c.chat("b9-e2e-2", "u_1001", "吊牌要留着吗")

    assert result.meta["advice_basis"] == "clauses"
    assert "【R-01】" in result.reply
    assert "吊牌" in result.reply


def test_end_to_end_off_topic_question_is_not_faked(make_coordinator):
    """政策里没有的问题，不猜、也不复读「还在等你发订单号」。"""
    c = make_coordinator()
    c.chat("b9-e2e-3", "u_1001", "我要退货")
    result = c.chat("b9-e2e-3", "u_1001", "你们几点上班")

    assert "没找到对应内容" in result.reply
    assert "转人工" in result.reply
    assert "还在等你发订单号" not in result.reply


# ---------------------------------------------------------------------------
# 7. 订单号槽位格式关：模型写的 "null" 不能拿去查订单
# ---------------------------------------------------------------------------
def test_order_id_guard_rejects_model_null_strings():
    from app.slots import normalize_order_id

    for junk in [None, "", "null", "None", "undefined", "未知", "不知道", "abc"]:
        assert normalize_order_id(junk) is None, junk
    assert normalize_order_id(" O 20261115001 ") == "O20261115001"
    assert normalize_order_id("o20261115001。") == "O20261115001"


def _fake_understand(c, payload: str):
    """理解层恒定返回 payload；表达层返回原文（沿用 B7 的写法）。"""
    from app.llm import OpenAICompatibleLLM
    from app.usage import measured_usage

    c.llm = OpenAICompatibleLLM("https://example.invalid/v1", "key", "test-model")

    def fake_chat(messages, temperature=0.2, **kwargs):
        system = messages[0]["content"] if messages else ""
        if "意图识别器" in system:
            return payload, measured_usage(50, 10, 60, 1.0, "test-model")
        lines = (messages[-1]["content"] if messages else "").split("\n")
        return (lines[1] if len(lines) > 1 else ""), measured_usage(50, 10, 60, 1.0, "test-model")

    c.llm._chat = fake_chat


def test_end_to_end_null_order_id_is_not_looked_up(make_coordinator):
    """真实模型对「没有单号」常写字符串 "null"：不能变成「没有查到订单 null」。"""
    c = make_coordinator()
    _fake_understand(c, '{"intent":"policy","order_id":"null","reason":"no_reason","confidence":0.9}')
    result = c.chat("b9-e2e-4", "u_1001", "为什么不能退")

    assert "null" not in result.reply.lower()
    assert result.trace[0]["detail"]["order_correction"]["rule"] == "format_guard"


def test_end_to_end_policy_intent_followup_stays_on_topic(make_coordinator):
    """真实模型把「为什么不能退」判成 policy：仍要接着刻字那件事答，不能重开话题。"""
    c = make_coordinator()
    first = c.chat("b9-e2e-5", "u_1001", "保温杯刻字能退吗")
    assert first.meta["advice"] == "custom_only"

    _fake_understand(c, '{"intent":"policy","order_id":"null","reason":"no_reason","confidence":0.9}')
    second = c.chat("b9-e2e-5", "u_1001", "为什么不能退")

    assert second.meta["advice_basis"] == "context"
    assert "刻字" in second.reply
    assert "按平台规则" not in second.reply   # 没有掉回通用条款检索答案
