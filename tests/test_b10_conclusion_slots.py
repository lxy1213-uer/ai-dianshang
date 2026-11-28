"""B10：结论短语的语义槽 + 极性硬闸。

B6 x01（「超过7天退货期限」vs 断言「已超过」）与 B7 y11（「还在待发货状态」
vs 断言「还未发货」）连续两轮在同一处失分：断言与结论短语的具体措辞耦合。
继续往整词表加成员是追着模型跑；这里把状态类结论抽成语义槽，并配反向断言
自检——语义等价放行、极性反转绝不放行。
"""

from __future__ import annotations

import pytest

from app.eval.runner import (
    _CONCLUSION_SLOT_SPECS,
    _contains_by_slot,
    _contains_fact,
    _forbidden_present,
)

# ---------------------------------------------------------------------------
# 1. 等价说法放行
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "reply",
    [
        "已超过退货期限",
        "超过7天退货期限",          # B6 x01
        "该订单超出无理由退货期限",
        "已经超期了",
        "这笔申请过了期限",
    ],
)
def test_exceeded_equivalents_pass(reply):
    assert _contains_by_slot(reply, "已超过")


@pytest.mark.parametrize(
    "reply",
    [
        "还未发货",
        "这笔订单还在待发货状态",    # B7 y11
        "目前没发货，稍后安排",
        "尚未发出，可以取消",
        "还没发呢",
    ],
)
def test_unshipped_equivalents_pass(reply):
    assert _contains_by_slot(reply, "还未发货")


# ---------------------------------------------------------------------------
# 2. 极性反转硬闸：含全部字面要素也不能放行
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "reply",
    [
        "未超过7天退货期限",
        "没有超过期限，在可退范围内",
        "还在期限内，不算超期",
        "未到期，可以办理",
    ],
)
def test_negated_exceeded_never_passes(reply):
    assert not _contains_by_slot(reply, "已超过")
    assert not _contains_fact(reply, "已超过")


@pytest.mark.parametrize(
    "reply",
    [
        "已经发货，无法取消",
        "已签收，请走退货流程",
        "已发货，之前显示待发货",   # 全局闸：现在的状态是「已发货」
        "已经签收，不是待发货了",
    ],
)
def test_shipped_never_passes_as_unshipped(reply):
    assert not _contains_by_slot(reply, "还未发货")
    assert not _contains_fact(reply, "还未发货")


# ---------------------------------------------------------------------------
# 3. 精确匹配仍权威，槽不覆盖无关词 / 数字 / 单号
# ---------------------------------------------------------------------------
def test_exact_match_still_wins():
    assert _contains_fact("已超过", "已超过")
    assert _contains_fact("还未发货", "还未发货")


@pytest.mark.parametrize("keyword", ["已超过15天", "R-06", "运费由商家承担", "随便什么"])
def test_unrelated_keywords_are_not_matched_by_slots(keyword):
    assert not _contains_by_slot("超过期限，待发货，R-01", keyword)


def test_slot_table_is_strictly_scoped():
    assert set(_CONCLUSION_SLOT_SPECS) == {"已超过", "还未发货"}


def test_forbidden_assertion_uses_the_same_gate():
    """禁词侧同样不能被极性翻转绕过，也不能把相反状态误判为禁词出现。"""
    assert _forbidden_present("超过7天退货期限", "已超过")
    assert not _forbidden_present("未超过7天退货期限", "已超过")
    assert _forbidden_present("还在待发货状态", "还未发货")
    assert not _forbidden_present("已经发货", "还未发货")


# ---------------------------------------------------------------------------
# 4. 跨小句不误杀：别的事实里的否定不参与本槽判定
# ---------------------------------------------------------------------------
def test_cross_clause_negation_does_not_kill_positive():
    assert _contains_by_slot("该订单未签收，但已经超过退货期限", "已超过")
    assert _contains_by_slot("订单已创建，目前还在待发货状态", "还未发货")
