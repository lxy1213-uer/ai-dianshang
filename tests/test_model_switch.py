"""运行时模型切换：任一层可切到任一已配置模型，计费按实际模型名走。

全是离线桩，不打外网。锁住三条纪律：
- 选项脱敏，api_key 永不出现在对外视图；
- 切换只换实例与 active 标记，不重启、不改会话状态；
- 切换后计费按「实际模型」取价，Claude 切到理解层不再误算免费。
"""

from __future__ import annotations

import pytest

from app.model_registry import build_registry
from app.usage import measured_usage, REFERENCE_PRICES_CNY_PER_1K

MAIN = "claude-haiku-4-5"
FREE = "glm-4-flash"


def _coordinator(make_coordinator, **extra):
    base = dict(
        llm_mode="llm", llm_api_key="sk-main-key", llm_model=MAIN,
        price_in_per_1k=0.0072, price_out_per_1k=0.036, price_multiplier=1.0,
        classify_base_url="https://open.bigmodel.cn/api/paas/v4",
        classify_api_key="sk-free-key", classify_model=FREE,
    )
    base.update(extra)
    return make_coordinator(**base)


# --- 注册表与脱敏 ---------------------------------------------------------

def test_registry_contains_both_models(make_coordinator):
    c = _coordinator(make_coordinator)
    view = c.models_view()
    assert view["switchable"] is True
    assert {m["id"] for m in view["options"]} == {MAIN, FREE}
    assert view["active"] == {"main": MAIN, "classify": FREE}


def test_registry_never_exposes_api_key(make_coordinator):
    c = _coordinator(make_coordinator)
    assert "sk-main-key" not in str(c.models_view())
    assert "sk-free-key" not in str(c.models_view())
    assert "sk-main-key" not in str(c.health())


def test_registry_empty_in_rules_mode(make_coordinator):
    c = make_coordinator(llm_mode="rules")
    view = c.models_view()
    assert view["options"] == []
    assert view["switchable"] is False
    assert c.health()["model"] == "rule-engine"


# --- 切换行为 -------------------------------------------------------------

def test_switch_main_to_free_model(make_coordinator):
    c = _coordinator(make_coordinator)
    c.switch_model("main", FREE)
    assert c.llm.model == FREE
    assert c.models_view()["active"]["main"] == FREE
    assert c.health()["model"] == FREE
    assert c.models_view()["active"]["classify"] == FREE  # 理解层不动


def test_switch_classify_to_paid_model(make_coordinator):
    c = _coordinator(make_coordinator)
    c.switch_model("classify", MAIN)
    assert c.llm_classify.model == MAIN
    assert c.llm_classify is not c.llm  # 独立实例
    assert c.models_view()["active"]["classify"] == MAIN
    assert c.health()["classify_routed"] is True


def test_switch_unknown_role_or_model_rejected(make_coordinator):
    c = _coordinator(make_coordinator)
    with pytest.raises(ValueError):
        c.switch_model("bogus", MAIN)
    with pytest.raises(ValueError):
        c.switch_model("main", "not-a-model")


def test_switch_does_not_change_session_state(make_coordinator):
    c = _coordinator(make_coordinator)
    c.chat("sw-1", "u_1001", "你好")
    before = c.session_view("sw-1", "u_1001")["state"]
    c.switch_model("main", FREE)
    after = c.session_view("sw-1", "u_1001")["state"]
    assert before == after


# --- 计费随切换改变 -------------------------------------------------------

def test_billing_follows_actual_model_after_switch(make_coordinator, monkeypatch):
    """Claude 切到理解层后按 Claude 价计费，不再误算成免费。"""
    c = _coordinator(make_coordinator)
    c.switch_model("classify", MAIN)   # 理解层切到付费 Claude
    c.switch_model("main", FREE)       # 表达层切到免费 GLM

    def fake_classify(messages, temperature=0.2, **kwargs):
        return '{"intent":"greeting","confidence":0.9}', measured_usage(1000, 100, 1100, 5, MAIN)

    def fake_polish(messages, temperature=0.2, **kwargs):
        return "您好！", measured_usage(100, 20, 120, 5, FREE)

    monkeypatch.setattr(c.llm_classify, "_chat", fake_classify)
    monkeypatch.setattr(c.llm, "_chat", fake_polish)

    result = c.chat("sw-bill", "u_1001", "你好")
    usage = result.usage
    by_model = usage["pricing_by_model"]

    # 理解层现在是 Claude：1000 输入 + 100 输出按主模型单价计费
    assert by_model["classify:" + MAIN]["source"].startswith("env")
    # 表达层现在是免费 GLM：单价 0
    assert by_model["main:" + FREE]["source"] == "reference_table"
    assert REFERENCE_PRICES_CNY_PER_1K[FREE] == (0.0, 0.0)
    # 总费用 = Claude 那部分（0.0072*1 + 0.036*0.1），GLM 部分为 0
    assert usage["estimated_cost_cny"] == pytest.approx(0.0108)


# --- 注册表构建 -----------------------------------------------------------

def test_build_registry_from_settings(settings):
    from app.config import Settings

    s = Settings.load(
        llm_mode="llm", llm_api_key="k", llm_model=MAIN,
        classify_base_url="https://open.bigmodel.cn/api/paas/v4",
        classify_api_key="k2", classify_model=FREE,
    )
    registry = build_registry(s)
    assert set(registry) == {MAIN, FREE}
    assert registry[FREE].free is True
    assert registry[MAIN].free is False
