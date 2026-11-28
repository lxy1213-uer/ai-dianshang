"""多 Agent 真实编排 + Token 真实计费口径的回归测试。

锁住两件事：
1. trace 里下游 Agent 不再是「ms: 0 且没有 detail」的空壳——每一步都有
   perf_counter 实测耗时、真实工具调用链、上下游 handoff；
2. 只有 API 响应里的 usage 才算实测并计费；规则模式 / 降级 / 无 usage
   响应一律只有字符估算，实测 token 与费用为 0。
"""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from app.agent.coordinator import Coordinator
from app.config import Settings
from app.llm import OpenAICompatibleLLM
from app.usage import measured_usage


@pytest.fixture()
def make_coordinator(tmp_path):
    created: list[Coordinator] = []

    def make(**overrides):
        base = dict(
            db_path=tmp_path / f"orch_{len(created)}.db",
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


class _FakeResponse:
    """模拟 httpx.Response 的最小接口（含状态码与文本，便于验证错误分支）。"""

    def __init__(self, payload, status_code: int = 200, text: str = ""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _chat_payload(content: str, usage: dict | None = None) -> dict:
    payload = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        payload["usage"] = usage
    return payload


# ---------------------------------------------------------------------------
# 1. 真实编排
# ---------------------------------------------------------------------------
def test_multi_trace_has_real_durations_and_tool_chain(make_coordinator):
    c = make_coordinator()
    result = c.chat("s1", "u_1001", "退货政策是几天")

    assert result.agent_path == ["接待 Agent", "Supervisor", "政策 Agent"]
    assert len(result.trace) == 4  # 接待 / Supervisor / 政策 / 表达层
    for step in result.trace:
        assert isinstance(step["ms"], float), step
        assert step["ms"] >= 0
        assert "usage" in step and "display_tokens" in step["usage"]
        assert step["detail"], "不能存在没有 detail 的空壳步骤"

    policy = result.trace[2]
    assert policy["agent"] == "政策 Agent"
    assert [t["tool"] for t in policy["detail"]["tools"]] == ["search_policy"]
    assert policy["detail"]["tools"][0]["ok"] is True
    assert policy["detail"]["result"]["citations"]
    assert policy["detail"]["result"]["state"] == "IDLE"
    assert result.citations, "政策 Agent 检索的条款必须真正成为回复引用"


def test_handoff_context_is_consumed_by_downstream(make_coordinator):
    c = make_coordinator()
    result = c.chat("s2", "u_1001", "我要退货 O20261115001")

    handoff = result.trace[1]["detail"]["context"]
    assert handoff["order_id"] == "O20261115001"
    assert handoff["to"] == "售后 Agent"
    assert result.trace[2]["detail"]["handoff"]["order_id"] == "O20261115001"
    assert any(t["tool"] == "check_return_eligibility" for t in result.trace[2]["detail"]["tools"])


def test_orchestration_meta_records_dependencies_and_real_ms(make_coordinator):
    c = make_coordinator()
    result = c.chat("s3", "u_1001", "退货政策是几天")
    orchestration = result.meta["orchestration"]

    assert orchestration["mode"] == "multi"
    assert [s["agent"] for s in orchestration["steps"]] == [
        "接待 Agent", "Supervisor", "政策 Agent", "表达层"
    ]
    assert [s["depends_on"] for s in orchestration["steps"]] == [
        None, "接待 Agent", "Supervisor", "政策 Agent"
    ]
    assert orchestration["steps"][2]["tool_calls"] == 1
    assert orchestration["sum_step_ms"] > 0


def test_single_agent_also_runs_the_real_chain(make_coordinator):
    c = make_coordinator(agent_mode="single")
    result = c.chat("s4", "u_1001", "我要退货 O20261115001")

    assert result.agent_path == ["通用 Agent"]
    assert len(result.trace) == 2
    assert result.trace[0]["detail"]["intent"] == "return"
    assert result.trace[0]["detail"]["tools"]
    assert result.usage["measured"]["total_tokens"] == 0


def test_failed_tool_call_is_recorded_in_trace(make_coordinator):
    c = make_coordinator()
    result = c.chat("s5", "u_1001", "我要退货 O99999999999")
    tools = result.trace[2]["detail"]["tools"]
    assert tools and tools[0]["ok"] is False
    assert tools[0]["error"] == "not_found"


def test_all_steps_report_usage_not_placeholder_zeros(make_coordinator):
    c = make_coordinator()
    result = c.chat("s6", "u_1001", "你好")
    for step in result.trace:
        assert step["usage"]["source"] in ("rules", "code", "llm", "rules-fallback", "custom")
        assert step["tokens"] == step["usage"]["display_tokens"]
    # 不再有「ms 恰好是 int 0 且 action 是办理」的模拟步骤
    assert not any(
        step["ms"] == 0 and not step["detail"] for step in result.trace
    )


# ---------------------------------------------------------------------------
# 2. 计费口径：规则模式
# ---------------------------------------------------------------------------
def test_rules_mode_is_estimated_and_never_billable(make_coordinator):
    c = make_coordinator()
    result = c.chat("s7", "u_1001", "退货政策是几天")
    usage = result.usage

    assert usage["basis"] == "estimated"
    assert usage["billing_mode"] == "rules"
    assert usage["model"] == "rule-engine"
    assert usage["measured"]["calls"] == 0
    assert usage["measured"]["total_tokens"] == 0
    assert usage["estimated_tokens"] > 0
    assert usage["estimated_cost_cny"] == 0.0
    assert "字符估算" in usage["note"]
    assert len(usage["by_agent"]) == len(result.trace)
    for step in result.trace:
        assert step["usage"]["measured"] is False
        assert step["usage"]["billable"] is False
    assert result.tokens == sum(step["tokens"] for step in result.trace)


def test_usage_api_aggregates_turns_and_checks_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "usage_api.db"))
    monkeypatch.setenv("AGENT_TODAY", "2026-11-22")
    monkeypatch.setenv("AGENT_MODE", "multi")
    monkeypatch.setenv("AGENT_LLM_MODE", "rules")
    from app.main import app

    with TestClient(app) as client:
        first = client.post(
            "/api/chat",
            json={"session_id": "usage-s", "user_id": "u_1001", "message": "退货政策是几天"},
        )
        assert first.status_code == 200
        assert first.json()["usage"]["measured"]["total_tokens"] == 0
        client.post(
            "/api/chat",
            json={"session_id": "usage-s", "user_id": "u_1001", "message": "运费谁承担"},
        )
        body = client.get("/api/usage", params={"session_id": "usage-s", "user_id": "u_1001"})
        assert body.status_code == 200
        data = body.json()
        assert data["turns"] == 2
        assert data["measured"]["total_tokens"] == 0
        assert data["estimated_tokens"] > 0
        assert data["estimated_cost_cny"] == 0.0
        assert client.get(
            "/api/usage", params={"session_id": "usage-s", "user_id": "u_9999"}
        ).status_code == 403
        assert client.get(
            "/api/usage", params={"session_id": "usage-s"}
        ).status_code == 401  # 不带身份不得读取用量


# ---------------------------------------------------------------------------
# 3. 计费口径：LLM 实测 / 降级 / 单价
# ---------------------------------------------------------------------------
def test_llm_response_usage_is_the_only_measured_source(monkeypatch):
    llm = OpenAICompatibleLLM("https://example.invalid/v1", "key", "gpt-4o-mini")
    payload = _chat_payload(
        '{"intent":"return","order_id":null,"reason":"no_reason","confidence":0.9}',
        {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
    )
    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: _FakeResponse(payload))
    understanding, usage = llm.classify_with_usage("我要退货")

    assert understanding.intent == "return"
    assert understanding.tokens == 150
    assert usage.measured is True and usage.billable is True
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (120, 30, 150)
    assert usage.source == "llm"


def test_missing_usage_field_is_not_billable(monkeypatch):
    llm = OpenAICompatibleLLM("https://example.invalid/v1", "key", "gpt-4o-mini")
    monkeypatch.setattr(
        "httpx.post",
        lambda *args, **kwargs: _FakeResponse(_chat_payload('{"intent":"policy"}')),
    )
    _, usage = llm.classify_with_usage("退货政策是几天")
    assert usage.measured is False
    assert usage.billable is False
    assert "usage" in usage.note


def test_gateway_error_surfaces_status_and_body(monkeypatch):
    """第三方网关 401 必须暴露状态码与响应体，否则只能看到「降级」没法排查。"""
    llm = OpenAICompatibleLLM("https://gateway.invalid/v1", "bad-key", "claude-haiku-4-5")
    monkeypatch.setattr(
        "httpx.post",
        lambda *a, **k: _FakeResponse({}, status_code=401, text='{"code":"INVALID_API_KEY"}'),
    )
    with pytest.raises(RuntimeError) as exc:
        llm._chat([{"role": "user", "content": "hi"}])
    assert "401" in str(exc.value)
    assert "INVALID_API_KEY" in str(exc.value)


def test_anthropic_style_content_parts_are_flattened(monkeypatch):
    """聚合网关把 Anthropic 响应转成 OpenAI 格式时，content 可能是分段数组。"""
    llm = OpenAICompatibleLLM("https://gateway.invalid/v1", "key", "claude-haiku-4-5")
    payload = {
        "choices": [{"message": {"content": [{"type": "text", "text": "收"}, {"type": "text", "text": "到"}]}}],
        "usage": {"input_tokens": 9, "output_tokens": 2},
    }
    monkeypatch.setattr("httpx.post", lambda *a, **k: _FakeResponse(payload))
    text, usage = llm._chat([{"role": "user", "content": "hi"}])
    assert text == "收到"
    # input/output 命名也要能读成实测用量
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (9, 2, 11)
    assert usage.measured is True


def test_end_to_end_measured_usage_and_cost(make_coordinator, monkeypatch):
    c = make_coordinator(llm_mode="llm", llm_api_key="test-key")
    c.llm = OpenAICompatibleLLM("https://example.invalid/v1", "key", "gpt-4o-mini")
    content = '{"intent":"policy","order_id":null,"reason":"no_reason","confidence":0.9}'

    def fake_chat(messages, temperature=0.2, **kwargs):
        return content, measured_usage(100, 20, 120, 1.5, "gpt-4o-mini")

    monkeypatch.setattr(c.llm, "_chat", fake_chat)
    result = c.chat("s8", "u_1001", "退货政策是几天")
    usage = result.usage

    assert usage["basis"] == "measured"
    assert usage["billing_mode"] == "llm"
    assert usage["measured"]["calls"] == 2  # 接待 Agent + 表达层
    assert usage["measured"]["total_tokens"] == 240
    assert usage["measured"]["prompt_tokens"] == 200
    assert usage["measured"]["completion_tokens"] == 40
    assert usage["pricing"]["source"] == "reference_table"
    assert usage["estimated_cost_cny"] > 0
    assert sum(1 for s in result.trace if s["usage"]["measured"]) == 2
    assert result.reply, "护栏回退后仍要有回复"


def test_env_price_override_changes_cost(make_coordinator, monkeypatch):
    c = make_coordinator(
        llm_mode="llm", llm_api_key="test-key",
        price_in_per_1k=1.0, price_out_per_1k=2.0,
    )
    c.llm = OpenAICompatibleLLM("https://example.invalid/v1", "key", "gpt-4o-mini")

    def fake_chat(messages, temperature=0.2, **kwargs):
        # 必须返回合法 JSON，否则 classify 会走降级，实测调用数就不是 2 次
        return '{"intent":"greeting"}', measured_usage(100, 20, 120, 1.0, "gpt-4o-mini")

    monkeypatch.setattr(c.llm, "_chat", fake_chat)
    usage = c.chat("s9", "u_1001", "你好").usage

    assert usage["pricing"]["source"] == "env"
    # 两次调用：prompt 200 / completion 40 → 0.2 + 0.08
    assert usage["estimated_cost_cny"] == pytest.approx(0.28)


def test_gateway_multiplier_scales_cost(make_coordinator, monkeypatch):
    """聚合网关按「倍率」计费（如 0.3x）：单价可以是官方价，金额再乘倍率。"""
    inline = make_coordinator(
        llm_mode="llm", llm_api_key="test-key",
        price_in_per_1k=1.0, price_out_per_1k=2.0, price_multiplier=1.0,
    )
    scaled = make_coordinator(
        llm_mode="llm", llm_api_key="test-key",
        price_in_per_1k=1.0, price_out_per_1k=2.0, price_multiplier=0.3,
    )

    def fake_chat(messages, temperature=0.2, **kwargs):
        return '{"intent":"greeting"}', measured_usage(100, 20, 120, 1.0, "gpt-4o-mini")

    costs = []
    for inst, session in ((inline, "m-inline"), (scaled, "m-scaled")):
        inst.llm = OpenAICompatibleLLM("https://example.invalid/v1", "key", "gpt-4o-mini")
        monkeypatch.setattr(inst.llm, "_chat", fake_chat)
        usage = inst.chat(session, "u_1001", "你好").usage
        costs.append(usage["estimated_cost_cny"])
        # token 数是实测的，倍率只影响金额，不能改 token
        assert usage["measured"]["total_tokens"] == 240
        assert usage["pricing"]["multiplier"] == pytest.approx(inst.settings.price_multiplier or 1.0)

    assert costs[0] == pytest.approx(0.28)
    assert costs[1] == pytest.approx(0.28 * 0.3, rel=1e-3)


def test_unknown_model_without_price_reports_no_cost(make_coordinator, monkeypatch):
    c = make_coordinator(llm_mode="llm", llm_api_key="test-key", llm_model="some-custom-model")
    c.llm = OpenAICompatibleLLM("https://example.invalid/v1", "key", "some-custom-model")

    def fake_chat(messages, temperature=0.2, **kwargs):
        return '{"intent":"greeting"}', measured_usage(100, 20, 120, 1.0, "some-custom-model")

    monkeypatch.setattr(c.llm, "_chat", fake_chat)
    usage = c.chat("s10", "u_1001", "你好").usage
    assert usage["pricing"]["source"] == "unset"
    assert usage["pricing"]["in_per_1k"] is None
    assert usage["estimated_cost_cny"] is None
    assert usage["measured"]["total_tokens"] == 240  # 实测 token 仍如实上报


def test_fallback_never_bills(make_coordinator, monkeypatch):
    c = make_coordinator(llm_mode="llm", llm_api_key="test-key")
    c.llm = OpenAICompatibleLLM("https://example.invalid/v1", "key", "gpt-4o-mini")

    def exploding_chat(messages, temperature=0.2, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(c.llm, "_chat", exploding_chat)
    result = c.chat("s11", "u_1001", "我要退货 O20261115001")

    assert result.reply and result.state
    assert result.usage["measured"]["total_tokens"] == 0
    assert result.usage["estimated_cost_cny"] == 0.0
    fallback_steps = [s for s in result.trace if s["usage"]["source"] == "rules-fallback"]
    assert len(fallback_steps) == 2
    assert all(s["usage"]["measured"] is False for s in fallback_steps)


def test_polish_prompt_avoids_leak_labels_and_caps_length(monkeypatch):
    """第三方模型会复述「事实：」「原始话术：」标签，导致整段被 prompt_leak 丢弃。

    这两项在真实网关上实测复现过，用请求体断言锁住修复。
    """
    captured: dict = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs.get("json", {}))
        return _FakeResponse(_chat_payload("改写后的客服话术"))

    monkeypatch.setattr("httpx.post", fake_post)
    llm = OpenAICompatibleLLM("https://gateway.invalid/v1", "key", "claude-haiku-4-5")
    text, usage = llm.polish_with_usage("原文含 7 天与 R-01", {"window_days": 7})

    assert text == "改写后的客服话术"
    assert captured["max_tokens"] == 300
    user = captured["messages"][-1]["content"]
    assert "原文含 7 天与 R-01" in user
    assert "原始话术：" not in user and "事实：" not in user
    assert "不要输出" in user
    assert usage.measured is False  # _chat_payload 没给 usage，符合「不计费」口径


def test_classify_caps_json_output(monkeypatch):
    captured: dict = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs.get("json", {}))
        return _FakeResponse(
            _chat_payload('{"intent":"return"}', {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        )

    monkeypatch.setattr("httpx.post", fake_post)
    llm = OpenAICompatibleLLM("https://gateway.invalid/v1", "key", "claude-haiku-4-5")
    understanding, usage = llm.classify_with_usage("我要退货")
    assert understanding.intent == "return"
    assert captured["max_tokens"] == 200
    assert usage.total_tokens == 15
