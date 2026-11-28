"""可切换模型注册表：把 .env 里配置的两套网关抽象成「可选项 + 定价」。

设计要点：
1. 两个模型（主网关 Claude / 理解层免费 GLM）都注册为可选，任一层都能选任一个；
2. 每个模型自带 base_url / api_key / timeout / 单价 / 倍率，切换时整体套用，
   不会出现「Claude 切到理解层却还算免费」的串账；
3. 对外展示一律脱敏，api_key 只留在服务端内存，绝不进 /api/models 响应。
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .usage import resolve_model_price, resolve_price


@dataclass(frozen=True)
class ModelProfile:
    """一个可切换的模型选项：连接参数与计费参数绑定在一起。"""

    id: str                      # 角色无关的展示/选择标识，就是模型名
    label: str                   # 界面短名
    base_url: str
    api_key: str                 # 仅在服务端内存，to_public() 永不包含
    timeout: float
    price_in_per_1k: float | None
    price_out_per_1k: float | None
    multiplier: float
    price_source: str
    free: bool                   # 单价恒为 0 的免费模型

    def to_public(self) -> dict:
        """给前端的选项：脱敏，只给展示需要的字段。"""
        return {
            "id": self.id,
            "label": self.label,
            "free": self.free,
            "price_source": self.price_source,
            "multiplier": self.multiplier,
        }


def _label_for(model: str, free: bool) -> str:
    """给模型一个短名；已知模型用友好别名，未知模型原样展示。"""
    known = {
        "claude-haiku-4-5": "Claude Haiku 4.5",
        "glm-4-flash-250414": "GLM-4-Flash",
        "glm-4-flash": "GLM-4-Flash",
        "glm-4.7-flash": "GLM-4.7-Flash",
        "Qwen/Qwen3-8B": "Qwen3-8B",
    }
    name = known.get(model, model)
    return f"{name}（免费）" if free else name


def _profile(
    model: str,
    base_url: str,
    api_key: str,
    timeout: float,
    pricing: dict,
) -> ModelProfile:
    pin = pricing.get("in_per_1k")
    pout = pricing.get("out_per_1k")
    free = pin == 0 and pout == 0
    return ModelProfile(
        id=model,
        label=_label_for(model, free),
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        price_in_per_1k=pin,
        price_out_per_1k=pout,
        multiplier=float(pricing.get("multiplier") or 1.0),
        price_source=pricing.get("source", "unset"),
        free=free,
    )


def build_registry(settings: Settings) -> dict[str, ModelProfile]:
    """从配置派生可切换模型表。

    - 主模型（表达层默认）：llm_* + resolve_price()
    - 理解层模型（可选）：classify_* + resolve_model_price(role=classify)
    - 同名模型去重：保留先注册者（主模型优先），避免一个 id 两套参数。
    - 至少包含主模型；llm 未启用时返回空表（此时界面不显示切换器）。
    """
    registry: dict[str, ModelProfile] = {}
    if not settings.llm_enabled:
        return registry

    main_pricing = resolve_price(settings)
    main = _profile(
        settings.llm_model, settings.llm_base_url, settings.llm_api_key,
        settings.llm_timeout, main_pricing,
    )
    registry[main.id] = main

    if settings.classify_model_enabled and settings.classify_model not in registry:
        classify_pricing = resolve_model_price(settings, settings.classify_model, role="classify")
        classify = _profile(
            settings.classify_model, settings.classify_base_url, settings.classify_api_key,
            settings.classify_timeout, classify_pricing,
        )
        registry[classify.id] = classify

    return registry
