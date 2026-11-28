"""测试公共夹具：固定日期、隔离数据库。"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest

# 测试必须与外网隔离：开发者本地 .env 可能配了真实 LLM（AGENT_LLM_MODE=llm），
# 放行的话测试会真的去打外部 API——慢、花钱，而且结果不可复现。
# config.load_dotenv 用的是 setdefault，所以这里先占位即可压住 .env 的同名配置。
# 需要在测试里验证 LLM 链路的，用 monkeypatch.setenv 显式覆盖，或直接构造假模型。
os.environ["AGENT_LLM_MODE"] = "rules"
os.environ["LLM_API_KEY"] = ""
# 同理：本地 .env 里的真实单价/倍率会让金额断言随机器漂移，测试统一从不配置单价起步，
# 需要验证计价的用例自己显式传参（见 test_orchestration_and_billing.py）。
os.environ["LLM_PRICE_IN_PER_1K"] = ""
os.environ["LLM_PRICE_OUT_PER_1K"] = ""
os.environ["LLM_PRICE_MULTIPLIER"] = ""
# 理解层专用模型（CLASSIFY_*）同理：本地 .env 配了免费模型时，若不压住，
# 原本该走规则模式的测试会真的去打智谱接口。需要验证路由的用例自行 monkeypatch。
os.environ["CLASSIFY_BASE_URL"] = ""
os.environ["CLASSIFY_API_KEY"] = ""
os.environ["CLASSIFY_MODEL"] = ""
os.environ["CLASSIFY_PRICE_IN_PER_1K"] = ""
os.environ["CLASSIFY_PRICE_OUT_PER_1K"] = ""

from app.agent.coordinator import Coordinator  # noqa: E402
from app.config import Settings  # noqa: E402


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings.load(db_path=tmp_path / "test.db", frozen_today=date(2026, 11, 22))


@pytest.fixture()
def coordinator(settings: Settings):
    instance = Coordinator(settings)
    yield instance
    instance.close()


@pytest.fixture()
def make_coordinator(tmp_path: Path):
    created: list[Coordinator] = []

    def factory(name: str = "test.db", **overrides) -> Coordinator:
        options = {"db_path": tmp_path / name, "frozen_today": date(2026, 11, 22)}
        options.update(overrides)
        instance = Coordinator(Settings.load(**options))
        created.append(instance)
        return instance

    yield factory
    for instance in created:
        instance.close()


def chat(coordinator: Coordinator, session_id: str, *messages: str, user_id: str = "u_1001"):
    result = None
    for message in messages:
        result = coordinator.chat(session_id, user_id, message)
    return result
