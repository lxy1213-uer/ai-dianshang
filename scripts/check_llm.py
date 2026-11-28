"""第三方 OpenAI 兼容网关的接入自检：python scripts/check_llm.py

不启动服务、不建工单，只验证三件事：
1. 网络能否连到 LLM_BASE_URL（DNS / TLS / 代理）
2. key 是否被网关接受（401 通常是 key 或域名不对）
3. 模型名是否可用，且响应里是否带 usage（决定能不能计费）

多模型路由（理解层可用免费模型）：
    python scripts/check_llm.py               # 检查主模型（表达层）
    python scripts/check_llm.py --role classify  # 检查理解层专用模型

输出里的 key 始终脱敏，可以直接贴给别人排查。
"""

import sys
import time
from pathlib import Path

ROLE = "classify" if "--role" in sys.argv and sys.argv[sys.argv.index("--role") + 1:sys.argv.index("--role") + 2] == ["classify"] else "main"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from app.config import Settings  # noqa: E402
from app.llm import OpenAICompatibleLLM  # noqa: E402
from app.usage import resolve_price  # noqa: E402


def mask(key: str) -> str:
    if not key:
        return "(空)"
    if len(key) <= 12:
        return key[:4] + "…"
    return f"{key[:8]}…{key[-4:]}（{len(key)} 字符）"


def hint(status: int, body: str) -> str:
    low = body.lower()
    if status == 401 or "invalid_api_key" in low or "unauthorized" in low:
        return (
            "key 没有被该网关接受。按顺序排查：\n"
            "    1) 浏览器地址栏的站点域名，是否和 LLM_BASE_URL 一致（不是同名站点的营销页）\n"
            "    2) 在控制台用「复制所选令牌」重新复制完整 key，注意 l/I/1、O/0 混淆\n"
            "    3) 令牌是否被禁用、是否绑定了 IP 白名单"
        )
    if status == 404:
        return (
            "路径不存在。多数网关是 https://<域名>/v1，但有的只接受 https://<域名>/v1/chat/completions；\n"
            "    也可能是该网关不提供 OpenAI 兼容的 /chat/completions，只提供 /v1/messages。"
        )
    if status == 429:
        return "被限流或额度用尽：看控制台余额与该令牌的剩余额度。"
    if status in (500, 502, 503, 504):
        return "网关上游异常，稍后重试；持续失败说明该模型渠道不可用。"
    return ""


def main() -> int:
    settings = Settings.load()
    role_label = "理解层专用模型（classify）" if ROLE == "classify" else "主模型（表达层）"
    print("=" * 66)
    print(f"接入自检：OpenAI 兼容网关 · {role_label}")
    print("=" * 66)

    if ROLE == "classify":
        # 理解层专用模型：检查前先确认配置齐全，否则给出明确指引
        if not settings.classify_model_enabled:
            print("\n[!] 理解层专用模型未配置齐全。需要同时设置：")
            print("    CLASSIFY_BASE_URL / CLASSIFY_API_KEY / CLASSIFY_MODEL，且 AGENT_LLM_MODE=llm。")
            print("    当前未启用时，理解层会跟随主模型（这是正常的默认行为）。")
            return 2
        base_url = settings.classify_base_url
        api_key = settings.classify_api_key
        model = settings.classify_model
        timeout = settings.classify_timeout
        from app.usage import resolve_model_price
        pricing = resolve_model_price(settings, model, role="classify")
    else:
        base_url = settings.llm_base_url
        api_key = settings.llm_api_key
        model = settings.llm_model
        timeout = settings.llm_timeout
        pricing = resolve_price(settings)

    print(f"模式        : {settings.llm_mode}（llm_enabled={settings.llm_enabled}）")
    print(f"Base URL    : {base_url}")
    print(f"模型        : {model}")
    print(f"API Key     : {mask(api_key)}")
    print(f"计价口径    : {pricing['source']} | 倍率 {pricing.get('multiplier')} | {pricing['note']}")
    if not settings.llm_enabled:
        print("\n[!] AGENT_LLM_MODE 不是 llm 或 LLM_API_KEY 为空，服务会走规则模式。")
        return 2

    headers = {"Authorization": f"Bearer {api_key}"}
    ok = True

    # 1) 模型列表：不是所有网关都实现，失败只提示不算致命
    print("\n[1/2] GET /models")
    try:
        r = httpx.get(f"{base_url}/models", headers=headers, timeout=15)
        if r.status_code == 200:
            ids = [m.get("id") for m in (r.json().get("data") or [])]
            print(f"  200，共 {len(ids)} 个模型；目标模型{'在' if model in ids else '不在'}列表中")
            if ids and model not in ids:
                # /models 只作参考：部分网关不列免费/旧版模型，但 /chat/completions 仍可调用。
                # 能不能用，以下面的真实对话自检为准，不因为它没进列表就判失败。
                print("  [i] 目标模型未出现在列表里：免费或旧版模型可能不列出，以对话自检为准。")
                print(f"  列表样例：{ids[:10]}")
        else:
            print(f"  {r.status_code} {r.text[:200]}")
            if model and r.status_code != 404:
                ok = False
    except Exception as exc:  # 网络层失败单独提示
        print(f"  连接失败：{type(exc).__name__}: {exc}")
        ok = False

    # 2) 真正打一次 chat，这是唯一能确认「能对话 + 能计费」的检查
    print("\n[2/2] POST /chat/completions")
    llm = OpenAICompatibleLLM(base_url, api_key, model, timeout)
    started = time.perf_counter()
    try:
        content, usage = llm._chat([{"role": "user", "content": "只回复两个字：收到"}])
        ms = (time.perf_counter() - started) * 1000
        print(f"  200，{ms:.0f} ms，回复：{content[:80]!r}")
        print(f"  实测用量：measured={usage.measured} prompt={usage.prompt_tokens} "
              f"completion={usage.completion_tokens} total={usage.total_tokens}")
        if usage.note:
            print(f"  备注：{usage.note}")
        if not usage.measured:
            print("  [!] 响应没带 usage：可以对话，但无法计费，只能记字符估算。")
        # 计价能否落地，取决于单价与倍率是否齐备
        if pricing["in_per_1k"] is None:
            print("  [i] 未配置单价，金额会返回 null（不编造数字）。")
        else:
            from app.usage import compute_cost

            cost = compute_cost(usage.prompt_tokens, usage.completion_tokens, pricing)
            print(f"  本次估算费用：{cost} 元（按 {pricing['source']} × 倍率 {pricing.get('multiplier')}）")
    except Exception as exc:
        detail = str(exc)
        print(f"  失败：{type(exc).__name__}: {detail[:300]}")
        status = 0
        if detail.startswith("HTTP "):
            try:
                status = int(detail.split()[1])
            except (IndexError, ValueError):
                status = 0
        advice = hint(status, detail)
        if advice:
            print("  排查建议：\n    " + advice)
        ok = False

    print("\n" + "=" * 66)
    print("结论：" + ("全部通过，可以切到 llm 模式" if ok else "未通过，按上面的建议修配置后再跑一次"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
