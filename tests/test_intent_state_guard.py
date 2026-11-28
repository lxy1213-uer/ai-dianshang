"""LLM 意图状态一致性硬护栏的回归测试。

B1 真实模型抽样暴露的两个理解层退化：
- IDLE 状态「这个单子我不想要了」被模型判成 hesitate → 状态机从未启动；
- AWAIT_CONFIRM 状态「先别确认」被模型判成 unknown → 走进兜底话术。

修复分两层：提示词写状态约束与 few-shot（软约束），以及
`app/llm.py needs_state_correction()` 硬护栏——模型意图与状态机职权冲突时，
该轮意图降级给规则引擎重判，usage 仍记这次真实调用的实测口径。

这里全部用假模型（monkeypatch _chat），不访问外网。
"""

from __future__ import annotations

from datetime import date

import pytest

from app.agent.coordinator import Coordinator
from app.config import Settings
from app.llm import OpenAICompatibleLLM, needs_state_correction
from app.usage import measured_usage


@pytest.fixture()
def make_coordinator(tmp_path):
    created: list[Coordinator] = []

    def make(**overrides):
        base = dict(
            db_path=tmp_path / f"guard_{len(created)}.db",
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


def _fake_llm(c: Coordinator, intent_json: str):
    """给 coordinator 换一个理解层恒定返回 intent_json 的假模型（表达层照旧）。"""
    c.llm = OpenAICompatibleLLM("https://example.invalid/v1", "key", "test-model")

    def fake_chat(messages, temperature=0.2, **kwargs):
        # 理解层请求才返回指定意图；表达层（润色）请求返回原文不动
        if messages and messages[0].get("role") == "system" and "意图识别器" in messages[0]["content"]:
            return intent_json, measured_usage(50, 10, 60, 1.0, "test-model")
        lines = (messages[-1]["content"] if messages else "").split("\n")
        return (lines[1] if len(lines) > 1 else ""), measured_usage(50, 10, 60, 1.0, "test-model")

    c.llm._chat = fake_chat
    return c.llm


# --- 纯函数口径 -----------------------------------------------------------
def test_hesitate_outside_waiting_state_needs_correction():
    assert needs_state_correction("hesitate", "IDLE", "这个单子我不想要了")
    assert needs_state_correction("hesitate", "AWAIT_ORDER", "再想想")
    assert not needs_state_correction("hesitate", "AWAIT_CONFIRM", "先别确认")


def test_confirm_without_pending_application_needs_correction():
    assert needs_state_correction("confirm", "IDLE", "好的")
    assert not needs_state_correction("confirm", "AWAIT_CONFIRM", "好的")
    assert not needs_state_correction("confirm", "CREATING", "好的")
    assert not needs_state_correction("confirm", "TICKETED", "好的")


def test_negation_in_waiting_state_needs_correction_when_model_missed_it():
    # 模型没认出否定词（unknown / policy 等）→ 规则引擎兜底
    assert needs_state_correction("unknown", "AWAIT_CONFIRM", "先别确认")
    assert needs_state_correction("policy", "AWAIT_CONFIRM", "先别提交")
    # 模型自己认出来了就不修正
    assert not needs_state_correction("hesitate", "AWAIT_CONFIRM", "先别提交")
    assert not needs_state_correction("cancel", "AWAIT_CONFIRM", "先别提交")
    # 非否定语境不修正
    assert not needs_state_correction("unknown", "AWAIT_CONFIRM", "这笔订单能退多少")


# --- 硬护栏生效：意图被规则引擎重判 ---------------------------------------
def test_hesitate_in_idle_is_reclassified_as_return():
    """B1 s04/s05 根因：非等待态判 hesitate 会吞掉纯口语退货诉求。"""
    llm = OpenAICompatibleLLM("https://example.invalid/v1", "key", "test-model")

    def fake_chat(messages, temperature=0.2, **kwargs):
        return '{"intent":"hesitate","order_id":null,"reason":"no_reason","confidence":0.7}', \
            measured_usage(50, 10, 60, 1.0, "test-model")

    llm._chat = fake_chat
    understanding, usage = llm.classify_with_usage("这个单子我不想要了", "IDLE")

    # 规则引擎重判：COLLOQUIAL_RETURN_WORDS 命中「不想要」→ return
    assert understanding.intent == "return"
    assert understanding.source == "llm-state-guard"
    assert understanding.slots["intent_correction"]["from"] == "hesitate"
    # 真实调用确实发生了，usage 仍是实测口径，不能因为修正而抹掉
    assert usage.measured is True
    assert usage.total_tokens == 60


def test_unknown_negation_in_await_confirm_is_reclassified_as_hesitate():
    llm = OpenAICompatibleLLM("https://example.invalid/v1", "key", "test-model")

    def fake_chat(messages, temperature=0.2, **kwargs):
        return '{"intent":"unknown","order_id":null,"reason":"no_reason","confidence":0.3}', \
            measured_usage(50, 10, 60, 1.0, "test-model")

    llm._chat = fake_chat
    understanding, usage = llm.classify_with_usage("先别确认", "AWAIT_CONFIRM")

    assert understanding.intent == "hesitate"
    assert understanding.source == "llm-state-guard"
    assert usage.measured is True


def test_consistent_llm_intent_passes_through_untouched():
    llm = OpenAICompatibleLLM("https://example.invalid/v1", "key", "test-model")

    def fake_chat(messages, temperature=0.2, **kwargs):
        return '{"intent":"return","order_id":"O20261115001","reason":"no_reason","confidence":0.9}', \
            measured_usage(50, 10, 60, 1.0, "test-model")

    llm._chat = fake_chat
    understanding, usage = llm.classify_with_usage("我要退货 O20261115001", "IDLE")

    assert understanding.intent == "return"
    assert understanding.source == "llm"
    assert understanding.slots == {}
    assert usage.measured is True


# --- 端到端：状态机必须被启动，不能掉回 IDLE -------------------------------
def test_state_machine_starts_even_if_model_says_hesitate(make_coordinator):
    """B1 s04 端到端：第一轮判 hesitate → 兜底 → 状态机从未启动。

    有了硬护栏后，IDLE 状态的 hesitate 被重判成 return，流程应进入 AWAIT_ORDER。
    """
    c = make_coordinator(llm_mode="llm", llm_api_key="test-key")
    _fake_llm(c, '{"intent":"hesitate","order_id":null,"reason":"no_reason","confidence":0.7}')

    result = c.chat("guard-e2e-1", "u_1001", "这个单子我不想要了")

    assert result.state == "AWAIT_ORDER"
    assert "订单号" in result.reply
    # trace 首步是接待 Agent：意图是修正后的 return，而不是模型的 hesitate
    assert result.trace[0]["detail"]["intent"] == "return"
    # usage 仍是真实调用的实测口径
    assert result.usage["basis"] == "measured"


def test_negated_confirm_in_await_confirm_keeps_state(make_coordinator):
    """B1 s05 端到端：先建到 AWAIT_CONFIRM，再发否定确认，状态不得被冲掉。"""
    c = make_coordinator(llm_mode="llm", llm_api_key="test-key")
    # 前两轮模型表现正常，第三轮「失误」判成 unknown
    answers = iter([
        '{"intent":"return","order_id":null,"reason":"no_reason","confidence":0.9}',
        '{"intent":"order","order_id":"O20261115001","reason":"no_reason","confidence":0.9}',
        '{"intent":"unknown","order_id":null,"reason":"no_reason","confidence":0.3}',
    ])
    c.llm = OpenAICompatibleLLM("https://example.invalid/v1", "key", "test-model")

    def fake_chat(messages, temperature=0.2, **kwargs):
        if messages and messages[0].get("role") == "system" and "意图识别器" in messages[0]["content"]:
            return next(answers), measured_usage(50, 10, 60, 1.0, "test-model")
        lines = (messages[-1]["content"] if messages else "").split("\n")
        return (lines[1] if len(lines) > 1 else ""), measured_usage(50, 10, 60, 1.0, "test-model")

    c.llm._chat = fake_chat

    c.chat("guard-e2e-2", "u_1001", "我要退货")
    c.chat("guard-e2e-2", "u_1001", "O20261115001")
    result = c.chat("guard-e2e-2", "u_1001", "先别确认")

    assert result.state == "AWAIT_CONFIRM"
    assert "先不提交" in result.reply
    assert result.ticket is None
    assert result.trace[0]["detail"]["intent"] == "hesitate"
