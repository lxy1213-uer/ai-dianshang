"""多模型路由：理解层走免费模型、表达层走主模型的回归测试。

设计要点（与 B1/B2/B3 的评测纪律一致，全是离线桩，不打外网）：
- 未配置专用模型时，行为与单模型完全一致（向后兼容，旧测试已覆盖）；
- 配置齐全时，classify 走专用实例、polish 走主实例，两者互不串账；
- 计费按 step 的实际模型取价：免费模型单价 0，主模型照常计费；
- 配置缺项时整体回退主模型，绝不带病上路。
"""

from __future__ import annotations

import pytest

from app.agent.coordinator import Coordinator
from app.config import Settings
from app.llm import OpenAICompatibleLLM
from app.usage import (
    REFERENCE_PRICES_CNY_PER_1K,
    resolve_model_price,
)

FREE_MODEL = "glm-4-flash"
MAIN_MODEL = "claude-haiku-4-5"


def _full_settings(**overrides) -> Settings:
    base = dict(
        llm_mode="llm",
        agent_mode="multi",
        llm_api_key="sk-main-key",
        llm_model=MAIN_MODEL,
        classify_base_url="https://open.bigmodel.cn/api/paas/v4",
        classify_api_key="sk-free-key",
        classify_model=FREE_MODEL,
    )
    base.update(overrides)  # 允许覆盖（包括置空）任一字段
    return Settings.load(**base)


# --- 配置层 ---------------------------------------------------------------

def test_classify_model_enabled_only_when_all_fields_present():
    s = _full_settings()
    assert s.classify_model_enabled is True

    # 缺任一项就回退主模型
    for missing in ("classify_base_url", "classify_api_key", "classify_model"):
        s = _full_settings(**{missing: ""})
        assert s.classify_model_enabled is False, missing

    # llm 模式没开也不算路由生效
    s = _full_settings(llm_mode="rules")
    assert s.classify_model_enabled is False


def test_health_describes_route_without_leaking_keys():
    s = _full_settings()
    health = s.describe()
    assert health["classify_routed"] is True
    assert health["classify_model"] == FREE_MODEL
    assert health["model"] == MAIN_MODEL
    assert "sk-free-key" not in str(health)
    assert "sk-main-key" not in str(health)


def test_free_models_have_zero_reference_price():
    for name in ("glm-4-flash", "glm-4.7-flash", "Qwen/Qwen3-8B"):
        assert REFERENCE_PRICES_CNY_PER_1K[name] == (0.0, 0.0)


def test_resolve_model_price_classify_uses_dedicated_price():
    s = _full_settings(classify_price_in_per_1k=0.0, classify_price_out_per_1k=0.0)
    pricing = resolve_model_price(s, FREE_MODEL, role="classify")
    assert pricing["source"] == "env_classify"
    assert pricing["in_per_1k"] == 0.0
    assert pricing["multiplier"] == 1.0  # 免费模型不套主网关倍率


def test_resolve_model_price_classify_falls_back_to_reference_table():
    s = _full_settings(price_multiplier=0.3)
    # 未配 CLASSIFY_PRICE_* → 按模型名查参考价，免费模型为 0，且不套倍率
    pricing = resolve_model_price(s, FREE_MODEL, role="classify")
    assert pricing["source"] == "reference_table"
    assert pricing["in_per_1k"] == 0.0
    assert pricing["multiplier"] == 1.0


def test_resolve_model_price_main_keeps_gateway_multiplier():
    s = _full_settings(price_multiplier=0.3)
    pricing = resolve_model_price(s, MAIN_MODEL, role="main")
    assert pricing["multiplier"] == 0.3
    assert pricing["in_per_1k"] == REFERENCE_PRICES_CNY_PER_1K[MAIN_MODEL][0]


# --- 路由层 ---------------------------------------------------------------

def test_coordinator_builds_two_distinct_llm_instances(make_coordinator):
    c = make_coordinator(
        llm_mode="llm", llm_api_key="sk-main-key", llm_model=MAIN_MODEL,
        classify_base_url="https://open.bigmodel.cn/api/paas/v4",
        classify_api_key="sk-free-key", classify_model=FREE_MODEL,
    )
    assert c._classify_routed is True
    assert c.llm.model == MAIN_MODEL
    assert c.llm_classify.model == FREE_MODEL
    assert c.llm_classify is not c.llm


def test_coordinator_unrouted_reuses_same_instance(make_coordinator):
    c = make_coordinator(llm_mode="llm", llm_api_key="k", llm_model=MAIN_MODEL)
    assert c._classify_routed is False
    assert c.llm_classify is c.llm


# --- 计费层：端到端用桩替换两个实例的 _chat -------------------------------

def test_routed_chat_bills_each_model_separately(make_coordinator, monkeypatch):
    from app.usage import measured_usage

    c = make_coordinator(
        llm_mode="llm", llm_api_key="sk-main-key", llm_model=MAIN_MODEL,
        price_in_per_1k=0.0072, price_out_per_1k=0.036, price_multiplier=1.0,
        classify_base_url="https://open.bigmodel.cn/api/paas/v4",
        classify_api_key="sk-free-key", classify_model=FREE_MODEL,
    )
    # 两个实例都是 OpenAICompatibleLLM，分别替换 _chat
    def fake_classify(messages, temperature=0.2, **kwargs):
        return '{"intent":"greeting","confidence":0.9}', measured_usage(1000, 100, 1100, 5, FREE_MODEL)

    def fake_polish(messages, temperature=0.2, **kwargs):
        return "您好！", measured_usage(100, 20, 120, 5, MAIN_MODEL)

    monkeypatch.setattr(c.llm_classify, "_chat", fake_classify)
    monkeypatch.setattr(c.llm, "_chat", fake_polish)

    result = c.chat("route-1", "u_1001", "你好")
    usage = result.usage

    # 理解层免费：1100 token 但费用 0；表达层主模型：0.0072*0.1 + 0.036*0.02 = 0.00144
    assert usage["measured"]["total_tokens"] == 1220
    assert usage["estimated_cost_cny"] == pytest.approx(0.00144)
    by_model = usage["pricing_by_model"]
    assert "classify:glm-4-flash" in by_model
    assert "main:claude-haiku-4-5" in by_model
    assert by_model["classify:glm-4-flash"]["source"] in ("reference_table", "env_classify")

    # trace 里理解层的「接待 Agent」与「表达层」各自标注了角色与模型
    # （注意 trace 里可能有多个「接待 Agent」，取第一个带 role 的理解层步骤）
    understand_step = next(s for s in result.trace if s["detail"].get("role") == "classify")
    express_step = next(s for s in result.trace if s["detail"].get("role") == "main")
    assert understand_step["agent"] == "接待 Agent"
    assert understand_step["detail"]["model"] == FREE_MODEL
    assert express_step["agent"] == "表达层"
    assert express_step["detail"]["model"] == MAIN_MODEL
