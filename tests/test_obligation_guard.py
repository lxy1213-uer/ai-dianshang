"""义务强度翻转护栏（路线图「需要 → 建议这类」）验收。

背景：B5 之前护栏能拦数字/单号/结论反转，但把 draft 的强制要求
（「需提供问题照片」）改成建议（「建议提供问题照片」）时，所有硬事实都还在、
读起来也通顺，护栏却不管——用户可能因此不提供凭证。本文件锁住这类语义幻觉
被拦、且合法语气缓和（请/麻烦）不被误杀。
"""

from __future__ import annotations

import re

import pytest

from app.grounding import (
    _obligation_flips,
    _compact,
    verify,
)

# ---------------------------------------------------------------------------
# 1. 义务弱化：强制 → 建议/可以/可选，必须拦
# ---------------------------------------------------------------------------

WEAKENING_CASES = [
    (
        "需提供→建议提供",
        "需提供问题照片或视频，必要时配合寄回检测。",
        "建议提供问题照片或视频，必要时配合寄回检测。",
        "弱化：提供",
    ),
    (
        "需保持→可以保持",
        "需保持吊牌、包装、赠品完整。",
        "可以保持吊牌、包装、赠品完整。",
        "弱化：保持",
    ),
    (
        "需上传→可选上传",
        "需上传问题凭证。",
        "可选上传问题凭证。",
        "弱化：上传",
    ),
    (
        "必须支付→建议支付",
        "这笔必须支付运费。",
        "这笔建议支付运费。",
        "弱化：支付",
    ),
    (
        "应当填写→可填写",
        "应当填写收货地址。",
        "可填写收货地址。",
        "弱化：填写",
    ),
    (
        "务必保留→自愿保留",
        "务必保留购物凭证。",
        "自愿保留购物凭证。",
        "弱化：保留",
    ),
    (
        "需承担→可以承担",
        "质量问题退货运费由商家承担，消费者需承担运费。",
        "质量问题退货运费由商家承担，消费者可以承担运费。",
        "弱化：承担",
    ),
]


@pytest.mark.parametrize("name,draft,polished,flip", WEAKENING_CASES, ids=[c[0] for c in WEAKENING_CASES])
def test_obligation_weakened_is_rejected(name, draft, polished, flip):
    result = verify(draft, polished)
    assert not result.ok
    assert "obligation_flip" in result.violations
    assert flip in result.detail["obligation_flips"]
    assert result.reply == draft                       # 整体回退
    assert result.used_polished is False


# ---------------------------------------------------------------------------
# 2. 反向强化：建议 → 强制，也要拦（把可选说成必须同样改变义务性质）
# ---------------------------------------------------------------------------

STRENGTHENING_CASES = [
    ("建议提供→必须提供", "建议提供问题照片。", "必须提供问题照片。", "强化：提供"),
    ("可以上传→需上传", "可以上传凭证。", "需上传凭证。", "强化：上传"),
]


@pytest.mark.parametrize("name,draft,polished,flip", STRENGTHENING_CASES, ids=[c[0] for c in STRENGTHENING_CASES])
def test_obligation_strengthened_is_rejected(name, draft, polished, flip):
    result = verify(draft, polished)
    assert not result.ok
    assert "obligation_flip" in result.violations
    assert flip in result.detail["obligation_flips"]


# ---------------------------------------------------------------------------
# 3. 合法改写：不能误杀
# ---------------------------------------------------------------------------

LEGAL_CASES = [
    (
        "无需→无需（强义务否定不误判）",
        "无需说明理由，但需保持吊牌完整。",
        "无需说明理由，但需保持吊牌完整～",
    ),
    (
        "不需要→不必（强义务否定不误判）",
        "不需要上传凭证。",
        "不必上传凭证。",
    ),
    (
        "需→请（语气缓和，请不在弱化语素表）",
        "需提供问题照片。",
        "请提供问题照片。",
    ),
    (
        "义务保留+额外建议（义务没丢）",
        "需提供问题照片。",
        "需提供问题照片，建议提供清晰照。",
    ),
    (
        "非义务动作（办理不在动作表）",
        "你可以直接办理退货。",
        "您可以直接办理退货。",
    ),
    (
        "泛化建议无动作（建议：换个订单）",
        "建议：换个订单再试。",
        "建议：换个订单再试哈。",
    ),
    (
        "真实 draft: 建议：上传凭证并提交退货申请",
        "这笔订单暂时不能直接办理退货：\n\n【不适用七天无理由】定制类商品不适用。\n\n建议：上传凭证并提交退货申请。",
        "这笔订单暂时不能直接办理退货：\n\n【不适用七天无理由】定制类商品不适用。\n\n建议：上传凭证并提交退货申请～",
    ),
    (
        "真实 draft: 需上传问题凭证（质量问题详情）",
        "在 15 天质量问题期限内（截止 2026-11-22）。退货运费由商家承担，需上传问题凭证。",
        "在 15 天质量问题期限内（截止 2026-11-22）。退货运费由商家承担，需上传问题凭证～",
    ),
]


@pytest.mark.parametrize("name,draft,polished", LEGAL_CASES, ids=[c[0] for c in LEGAL_CASES])
def test_legal_rewrites_pass(name, draft, polished):
    result = verify(draft, polished)
    assert result.ok, f"{name}: {result.violations} / {result.detail.get('obligation_flips')}"
    assert result.used_polished is True


# ---------------------------------------------------------------------------
# 4. 语素组合层面：未见说法也能覆盖，但仅在同一动作上对齐
# ---------------------------------------------------------------------------

def test_unseen_weakening_phrases_still_detected():
    """不枚举整句：任意强/弱语素 + 同动作都该被识别。"""
    cases = [
        ("须给出凭证", "建议给出凭证"),
        ("需要提交申请", "可以提交申请"),
        ("应当寄回商品", "不妨寄回商品"),
        ("务必附上照片", "可附上照片"),
    ]
    for draft, polished in cases:
        flips = _obligation_flips(_compact(draft), _compact(polished))
        assert flips, (draft, polished)


def test_different_actions_do_not_cross_match():
    """动作必须对齐：需提供→建议上传 不该互相放行（各自动作丢失另有校验）。"""
    flips = _obligation_flips(_compact("需提供照片"), _compact("建议上传照片"))
    assert flips == []


def test_no_false_positive_from_policy_snippet():
    """政策片段自带的「可申请退货」是许可，不是义务强化，不该误判。"""
    draft = "普通类目商品在签收后 7 天内，可申请无理由退货。无需说明理由。"
    polished = "普通类目商品在签收后 7 天内，可申请无理由退货。无需说明理由～"
    assert verify(draft, polished).ok


# ---------------------------------------------------------------------------
# 5. 真实链路：合法润色不被误杀（义务护栏接入 verify 后仍可正常采用润色）
# ---------------------------------------------------------------------------

def test_coordinator_keeps_legal_polish_with_obligation_guard(make_coordinator):
    """真实链路：礼貌语气润色不涉及义务强度，护栏应放行润色版本。"""
    import re

    from tests.test_grounding import MutatingLLM

    c = make_coordinator("obligation_guard.db")
    c.llm = MutatingLLM(lambda draft, ctx: re.sub(r"请把订单号发给我", "麻烦把订单号发给我", draft))

    result = c.chat("s_obligation", "u_1001", "我要退货")
    guard = result.meta["grounding"]
    assert guard["detail"]["checked"] is True
    assert "obligation_flip" not in guard["violations"]
    assert guard["used_polished"] is True
    assert "麻烦把订单号发给我" in result.reply
