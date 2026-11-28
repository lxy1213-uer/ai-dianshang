"""B4 职责边界守卫：严格断言层放宽同义，但护栏与幻觉检测不放宽。

分层原则（B4 刻意维持）：
- **护栏（grounding）**：管事实安全，保守策略，连等价改写也可能回退，
  绝不因为「评测口径放宽了」就跟着放宽——那是把语义判断带回表达层。
- **评测断言（_contains_fact）**：只管「事实等价的措辞差异算不算通过」，
  这是评分口径问题，不影响线上回复的正确性。

本文件端到端验证这个边界：同义改写能被严格口径接受（B4 修复价值），
而语义反转仍被护栏和幻觉检测拦下（安全底线）。
"""

from __future__ import annotations

import re

from app.eval.runner import load_llm_sample4, run
from app.llm import RuleLLM


class _PolishLLM(RuleLLM):
    """离线假模型：classify 走规则，polish 按注入变换改写话术。"""

    def __init__(self, transform):
        self.transform = transform

    def polish(self, draft: str, context: dict | None = None):
        return self.transform(draft), 321


def _hook(transform):
    def install(coordinator):
        coordinator.llm = _PolishLLM(transform)

    return install


def _only_case(case_id: str):
    return [c for c in load_llm_sample4() if c["id"] == case_id]


# ---------------------------------------------------------------------------
# 1) 端到端：同义改写被严格口径接受
# ---------------------------------------------------------------------------
def test_synonym_rewrite_passes_strict_eval():
    """B3 u10 失败同款：把「没有查到订单」换成「未查询到订单」，B4 口径应通过。"""
    report = run(
        "multi",
        cases=_only_case("v01"),
        settings_overrides={"llm_mode": "rules"},
        coordinator_hook=_hook(lambda d: d.replace("没有查到订单", "未查询到订单")),
    )
    row = report["cases"][0]
    assert row["passed"] is True, row["failures"]
    assert "missing_fact" not in row["categories"]


def test_all_four_synonym_groups_pass_strict_eval():
    """四组同义词典的等价改写都应通过（在各自对应的样本上）。"""
    transforms = {
        "v01": lambda d: d.replace("没有查到订单", "未查询到订单"),
        "v03": lambda d: d.replace("尚未签收", "还未签收"),
        "v04": lambda d: d.replace("不能重复申请", "无法重复申请").replace("已退款", "已经退款"),
    }
    for case_id, transform in transforms.items():
        report = run(
            "multi",
            cases=_only_case(case_id),
            settings_overrides={"llm_mode": "rules"},
            coordinator_hook=_hook(transform),
        )
        row = report["cases"][0]
        assert row["passed"] is True, (case_id, row["failures"])


def test_structural_phrase_equivalent_still_rejected_by_guard():
    """护栏不放宽：「运费由商家承担」整句被压缩掉，即使事实等价也回退。"""
    report = run(
        "multi",
        cases=_only_case("v06"),
        settings_overrides={"llm_mode": "rules"},
        coordinator_hook=_hook(
            lambda d: re.sub(r"退货运费由商家承担", "运费商家出", d)
            if "退货运费由商家承担" in d
            else re.sub(r"运费由商家承担", "运费商家出", d)
        ),
    )
    row = report["cases"][0]
    # 护栏回退到模板原文，因此必含事实仍在——但护栏必须记录回退，
    # 证明「等价改写」在表达层没有被无条件放行。
    assert "missing_phrase" in " ".join(row["failures"]) or row["passed"] is True
    # 直接验证护栏判定，避免只靠评测侧面推断
    from app.grounding import verify

    result = verify("按政策，运费由商家承担。", "按政策，运费商家出。")
    assert result.ok is False
    assert "missing_phrase" in result.violations


# ---------------------------------------------------------------------------
# 2) 语义反转仍被护栏拦下
# ---------------------------------------------------------------------------
def test_negation_flip_still_falls_back_to_template():
    """把「没有查到订单」反转成「已查到订单」，护栏必须回退到模板。"""
    from app.grounding import verify

    draft = "没有查到订单 O99999999999。请确认订单号是否完整。"
    polished = "已查到订单 O99999999999。"
    result = verify(draft, polished)
    assert result.ok is False
    assert "negation_flip" in result.violations
    assert "没有查到订单" in result.reply  # 回到模板原文


def test_fabricated_order_id_still_blocked_under_b4():
    """同义口径不影响幻觉检测：编造订单号仍被护栏拦。"""
    from app.grounding import verify

    draft = "没有查到订单 O99999999999。"
    polished = "没有查到订单 O11111111111。"
    result = verify(draft, polished)
    assert result.ok is False
    assert "fabricated_id" in result.violations
