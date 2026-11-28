"""Token 计量与计费口径（真实编排配套）。

三条硬规则，避免把「字符估算」冒充成「账单」：

1. **只有 API 响应里的 usage 字段算实测**（measured）。规则模式、调用降级、
   或兼容接口不返回 usage 时，一律只有 prompt 字符估算（estimated），
   标记 billable=False，**不产生费用**。
2. **estimated_cost_cny 是估算费用**：实测 token × 可配置单价，不是供应商账单。
   单价来自环境变量或内置参考价目表；都没有时返回 None（不编造金额）。
3. 规则模式的 `tokens` 字段保留为对比用的估算值，但 `measured.total_tokens`
   恒为 0、`estimated_cost_cny` 为 0，报告里两套口径必须同时出现。
"""

from __future__ import annotations

from dataclasses import dataclass

# 参考价目表：人民币 / 每 1K tokens（input, output）。
# 仅用于让 llm 模式能演示金额换算；按 1 USD ≈ 7.2 CNY 粗略换算，
# **不是官方报价、不是账单**，实际以供应商账单为准，可用环境变量覆盖。
REFERENCE_PRICES_CNY_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00108, 0.00432),   # $0.15 / $0.60 per 1M
    "gpt-4o": (0.01800, 0.07200),        # $2.50 / $10.00 per 1M
    "deepseek-chat": (0.00200, 0.00800), # ¥2 / ¥8 per 1M（示例）
    "claude-haiku-4-5": (0.00720, 0.03600),  # $1 / $5 per 1M × 7.2（示例换算）
    # 免费模型：单价 0，实测 token 仍上报，但估算费用恒为 0。
    # 免费不代表无限制——限速/额度由供应商决定，项目侧只保证「不虚报金额」。
    "glm-4-flash": (0.0, 0.0),           # 智谱 GLM-4-Flash，永久免费（旧别名）
    "glm-4-flash-250414": (0.0, 0.0),    # 智谱 GLM-4-Flash 当前可用版本，免费
    "glm-4.7-flash": (0.0, 0.0),         # 智谱 GLM-4.7-Flash，免费（高峰期可能 429）
    "Qwen/Qwen3-8B": (0.0, 0.0),         # 硅基流动免费小模型
    "Qwen/Qwen2.5-7B-Instruct": (0.0, 0.0),
}

# 意图识别专用模型用哪套单价字段：理解层可用 CLASSIFY_PRICE_* 单独指定（免费填 0），
# 未指定时回退到主模型的单价配置，再回退到内置参考价目表。
_CLASSIFY_PRICE_FIELDS = ("classify_price_in_per_1k", "classify_price_out_per_1k")


@dataclass
class Usage:
    """一次模型调用（或一段本地代码）的用量记录。"""

    source: str = "code"                 # rules | llm | rules-fallback | code | custom
    measured: bool = False               # 是否来自 API usage 字段
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_tokens: int = 0            # 非实测时的字符估算
    latency_ms: float = 0.0
    model: str = ""
    note: str = ""

    @property
    def billable(self) -> bool:
        """只有拿到真实 usage 且 token 数 > 0 才计费。"""
        return self.measured and self.total_tokens > 0

    @property
    def display_tokens(self) -> int:
        """兼容旧字段：实测时给实测值，否则给估算值。"""
        return self.total_tokens if self.measured else self.estimated_tokens

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "measured": self.measured,
            "billable": self.billable,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_tokens": self.estimated_tokens,
            "display_tokens": self.display_tokens,
            "latency_ms": round(self.latency_ms, 3),
            "model": self.model,
            "note": self.note,
        }


def estimated_usage(
    source: str,
    tokens: int,
    latency_ms: float = 0.0,
    model: str = "",
    note: str = "",
) -> Usage:
    """构造「无实测用量」记录：只有字符估算，不计费。"""
    return Usage(
        source=source,
        measured=False,
        estimated_tokens=int(tokens or 0),
        latency_ms=latency_ms,
        model=model,
        note=note or "无模型调用，tokens 为 prompt 字符估算，不计入费用",
    )


def measured_usage(
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    latency_ms: float,
    model: str,
    note: str = "",
) -> Usage:
    """构造「实测用量」记录：token 数来自 /chat/completions 的 usage 字段。"""
    return Usage(
        source="llm",
        measured=True,
        prompt_tokens=int(prompt_tokens or 0),
        completion_tokens=int(completion_tokens or 0),
        total_tokens=int(total_tokens or 0),
        latency_ms=latency_ms,
        model=model,
        note=note or "token 数来自 API usage 字段",
    )


def _multiplier(settings) -> float:
    """网关倍率（如聚合平台的 0.3x）。缺省 1.0，非法值也退回 1.0。"""
    raw = getattr(settings, "price_multiplier", None)
    try:
        value = float(raw) if raw is not None else 1.0
    except (TypeError, ValueError):
        return 1.0
    return value if value > 0 else 1.0


def resolve_price(settings) -> dict:
    """决定单价来源；未配置时明确返回 None，不编造金额。

    支持「倍率」：第三方聚合网关（如诺联）按官方价 × 分组倍率计费，
    因此单价可以是「官方参考价」，再乘 LLM_PRICE_MULTIPLIER。
    """
    multiplier = _multiplier(settings)
    configured_in = getattr(settings, "price_in_per_1k", None)
    configured_out = getattr(settings, "price_out_per_1k", None)
    if configured_in is not None and configured_out is not None:
        return {
            "in_per_1k": configured_in,
            "out_per_1k": configured_out,
            "multiplier": multiplier,
            "source": "env",
            "note": "来自 LLM_PRICE_IN_PER_1K / LLM_PRICE_OUT_PER_1K"
            + (f" × 倍率 {multiplier}" if multiplier != 1.0 else ""),
        }
    model = settings.llm_model
    reference = REFERENCE_PRICES_CNY_PER_1K.get(model)
    if reference:
        return {
            "in_per_1k": reference[0],
            "out_per_1k": reference[1],
            "multiplier": multiplier,
            "source": "reference_table",
            "note": "内置参考价（示例，非官方账单），可用环境变量覆盖"
            + (f" × 倍率 {multiplier}" if multiplier != 1.0 else ""),
        }
    return {
        "in_per_1k": None,
        "out_per_1k": None,
        "multiplier": multiplier,
        "source": "unset",
        "note": "未配置单价，只报实测 token，不报金额",
    }


def resolve_model_price(settings, model: str, *, role: str = "main") -> dict:
    """按「模型 + 角色」取价，供多模型路由逐步骤计费。

    - role="classify"：优先用 CLASSIFY_PRICE_IN/OUT_PER_1K（免费模型填 0），
      未配置则回退到主模型单价，再回退参考价目表；
    - role="main"：主模型口径，与 resolve_price() 完全一致。

    倍率只对主模型生效：免费模型走独立供应商，不该套聚合网关的倍率。
    """
    if role == "classify":
        cin = getattr(settings, "classify_price_in_per_1k", None)
        cout = getattr(settings, "classify_price_out_per_1k", None)
        if cin is not None and cout is not None:
            return {
                "in_per_1k": cin, "out_per_1k": cout, "multiplier": 1.0,
                "source": "env_classify", "note": "来自 CLASSIFY_PRICE_IN/OUT_PER_1K（理解层专用）",
            }
        # 未单独配置：优先参考价目表按模型名取（免费模型这里是 0）
        reference = REFERENCE_PRICES_CNY_PER_1K.get(model)
        if reference:
            return {
                "in_per_1k": reference[0], "out_per_1k": reference[1], "multiplier": 1.0,
                "source": "reference_table", "note": "内置参考价（免费模型为 0），不套主网关倍率",
            }

    return resolve_price(settings)


def compute_cost(prompt_tokens: int, completion_tokens: int, pricing: dict) -> float | None:
    """按单价换算估算费用（元）；缺单价时返回 None。"""
    price_in = pricing.get("in_per_1k")
    price_out = pricing.get("out_per_1k")
    if price_in is None or price_out is None:
        return None
    multiplier = pricing.get("multiplier") or 1.0
    cost = (prompt_tokens / 1000 * price_in + completion_tokens / 1000 * price_out) * multiplier
    return round(cost, 6)


def build_turn_usage(
    trace: list[dict],
    settings,
    model_pricing: dict[str, dict] | None = None,
) -> dict:
    """把一轮里每个 Agent 步骤的 usage 汇总成可对外报告的计费口径。

    trace 里每个 step 都带 `usage`（Usage.to_dict()）与真实 `ms`。

    `model_pricing` 是可选的「模型名 → 定价」表（运行时切换模型时由注册表提供）：
    传入后按 trace 里的实际模型名取价，保证 Claude 切到理解层不会误算免费；
    不传时完全沿用原来的按 role 取价逻辑，旧调用与测试行为不变。
    """
    billing_mode = "llm" if getattr(settings, "llm_enabled", False) else "rules"
    model = settings.llm_model if billing_mode == "llm" else "rule-engine"
    pricing = resolve_price(settings)  # 主模型口径，兼容旧的 pricing 字段断言

    # 每条 step 带上 role / model：多模型路由时逐步骤按各自单价计费。
    # 未路由时所有 step 都是 main，累加结果与旧逻辑完全一致。
    step_usages = [
        (step.get("agent", ""), step.get("usage") or {}, step.get("ms", 0),
         (step.get("detail") or {}).get("role", "main"),
         (step.get("detail") or {}).get("model") or (step.get("usage") or {}).get("model", ""))
        for step in trace
    ]
    measured_steps = [(a, u, role, m) for a, u, _, role, m in step_usages if u.get("measured")]
    prompt = sum(int(u.get("prompt_tokens", 0)) for _, u, _, _ in measured_steps)
    completion = sum(int(u.get("completion_tokens", 0)) for _, u, _, _ in measured_steps)
    total = sum(int(u.get("total_tokens", 0)) for _, u, _, _ in measured_steps)
    estimated = sum(int(u.get("estimated_tokens", 0)) for _, u, _, _, _ in step_usages)

    # 逐步骤计费：每个模型用自己的单价；任何一步缺单价则该步费用未知，
    # 整轮保守返回 None（不虚报金额），但实测 token 仍如实上报。
    cost_parts: list[float] = []
    pricing_by_model: dict[str, dict] = {}
    if total:
        for agent, u, role, step_model in measured_steps:
            p_tokens = int(u.get("prompt_tokens", 0))
            c_tokens = int(u.get("completion_tokens", 0))
            overridden = (model_pricing or {}).get(step_model)
            step_pricing = overridden if overridden is not None else resolve_model_price(
                settings, step_model, role=role
            )
            key = f"{role}:{step_model}"
            pricing_by_model[key] = {
                "agent": agent, "role": role, "model": step_model,
                "source": step_pricing["source"],
                "multiplier": step_pricing.get("multiplier", 1.0),
                "prompt_tokens": p_tokens, "completion_tokens": c_tokens,
            }
            part = compute_cost(p_tokens, c_tokens, step_pricing)
            if part is None:
                cost = None
                break
            cost_parts.append(part)
        else:
            cost = round(sum(cost_parts), 6)
    else:
        # 规则模式 / 全部降级 / 无 usage：费用明确为 0，而不是「未知」
        cost = 0.0

    if billing_mode == "rules":
        note = (
            "规则模式没有任何模型调用：下面 reported tokens 是 prompt 字符估算，"
            "实测 token 为 0、费用为 0。本数字只用于对比提示词规模，不代表账单。"
        )
    elif not measured_steps:
        note = "本轮没有拿到任何 API usage（全部降级或未返回 usage），因此不计费用；估算 token 仅供参考。"
    elif len(pricing_by_model) > 1:
        note = (
            "多模型路由：每个步骤按各自模型的单价计费（理解层可用免费模型，单价 0）；"
            "费用为估算值，非供应商账单；降级/无 usage 的调用不计费。"
        )
    else:
        note = "费用 = 实测 token × 配置单价，为估算值，非供应商账单；降级/无 usage 的调用不计费。"

    return {
        "basis": "measured" if measured_steps else ("estimated" if estimated else "none"),
        "billing_mode": billing_mode,
        "model": model,
        "pricing": pricing,
        "pricing_by_model": pricing_by_model,
        "measured": {
            "calls": len(measured_steps),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        },
        "estimated_tokens": estimated,
        "estimated_cost_cny": cost,
        "by_agent": [
            {
                "agent": agent,
                "source": u.get("source", "code"),
                "measured": bool(u.get("measured")),
                "total_tokens": int(u.get("total_tokens", 0)),
                "estimated_tokens": int(u.get("estimated_tokens", 0)),
                "latency_ms": ms,
                "role": role,
                "model": step_model,
            }
            for agent, u, ms, role, step_model in step_usages
        ],
        "note": note,
    }


def summarize_turns(usages: list[dict]) -> dict:
    """给 /api/usage 用：把多轮的 turn usage 累加成会话级口径。"""
    prompt = sum(int(u.get("measured", {}).get("prompt_tokens", 0)) for u in usages)
    completion = sum(int(u.get("measured", {}).get("completion_tokens", 0)) for u in usages)
    total = sum(int(u.get("measured", {}).get("total_tokens", 0)) for u in usages)
    calls = sum(int(u.get("measured", {}).get("calls", 0)) for u in usages)
    estimated = sum(int(u.get("estimated_tokens", 0)) for u in usages)
    costs = [u.get("estimated_cost_cny") for u in usages if isinstance(u.get("estimated_cost_cny"), (int, float))]
    basis = usages[-1].get("basis") if usages else "none"
    mode = usages[-1].get("billing_mode") if usages else "rules"
    return {
        "turns": len(usages),
        "billing_mode": mode,
        "basis": basis,
        "measured": {
            "calls": calls,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        },
        "estimated_tokens": estimated,
        "estimated_cost_cny": round(sum(costs), 6) if costs else 0.0,
        "note": "实测 token 来自 API usage；估算 token 为 prompt 字符估算，不计费",
    }
