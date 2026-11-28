"""B5 语素信号组合断言 + 样本纪律 + 护栏边界。

不调用外部 API（conftest 强制规则模式）。锁住四件事：
1. 信号组合放行 B4 词典漏掉的等价说法（未查到 / 未能查询到 / 不支持重复申请）；
2. 极性语素是硬门槛——否定反转、缺极性一律不放行；
3. 数字 / 单号 / 条款号 / 结构性短语不在信号规格里，精确匹配行为不变；
4. B5 样本结构与零重叠纪律，且生产护栏仍拦语义反转。
"""

from __future__ import annotations

import json

import pytest

from app.eval.runner import (
    LLM_SAMPLE5_PATH,
    _FACT_SIGNAL_SPECS,
    _contains,
    _contains_fact,
    load_all_cases,
    load_holdout2,
    load_llm_sample,
    load_llm_sample2,
    load_llm_sample3,
    load_llm_sample4,
    load_llm_sample5,
    run,
)


# ---------------------------------------------------------------------------
# 1) 信号规格结构
# ---------------------------------------------------------------------------
def test_signal_specs_cover_four_conclusions():
    assert set(_FACT_SIGNAL_SPECS) == {
        "没有查到订单", "尚未签收", "已退款", "不能重复申请",
    }
    for key, groups in _FACT_SIGNAL_SPECS.items():
        assert len(groups) >= 2, key  # 至少「极性 + 动作」两组
        for group in groups:
            assert group and all(w and w.strip() for w in group), key


def test_every_signal_group_requires_a_polarity_gate():
    """每组第一组都应是极性语素：没有它就没有放行门槛。"""
    polarity_first = {
        "没有查到订单": ("没有", "未", "没"),
        "尚未签收": ("尚未", "还没", "未", "没"),
        "已退款": ("已", "完成"),
        "不能重复申请": ("不能", "不可", "无法", "不支持"),
    }
    for key, expected in polarity_first.items():
        first = _FACT_SIGNAL_SPECS[key][0]
        assert any(any(tok in w for w in first) for tok in expected), (key, first)


# ---------------------------------------------------------------------------
# 2) 放行：B4 漏掉的等价说法
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "keyword, text",
    [
        ("没有查到订单", "我们未查到订单 O88888888888，请核实订单号。"),
        ("没有查到订单", "未能查询到订单 O77777777777。"),
        ("没有查到订单", "没查到这个订单，请重新确认。"),
        ("没有查到订单", "查无订单，麻烦核对一下编号。"),
        ("尚未签收", "订单还没收到呢，暂时不能退。"),
        ("尚未签收", "该单还未签收。"),
        ("已退款", "这笔已完成退款处理。"),
        ("已退款", "系统显示已经退款。"),
        ("不能重复申请", "系统不支持重复申请退货。"),
        ("不能重复申请", "该订单无法再次申请。"),
        ("不能重复申请", "不允许重复提交退货申请。"),
    ],
)
def test_signals_allow_unseen_equivalent_wording(keyword, text):
    assert _contains_fact(text, keyword) is True


# ---------------------------------------------------------------------------
# 3) 拦截：极性反转 / 缺极性 / 数字 / 单号 / 结构短语
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "keyword, text",
    [
        ("没有查到订单", "已查到订单 O999。"),
        ("没有查到订单", "查到订单 O999。"),
        ("尚未签收", "订单已经签收。"),
        ("尚未签收", "订单已签收。"),
        ("已退款", "这笔未退款。"),
        ("已退款", "还没有退款。"),
        ("不能重复申请", "可以重复申请退货。"),
        ("不能重复申请", "支持再次申请。"),
    ],
)
def test_signals_reject_polarity_flip(keyword, text):
    assert _contains_fact(text, keyword) is False


@pytest.mark.parametrize(
    "keyword, text",
    [
        ("7 天", "签收后8天"),
        ("7 天", "签收后六天"),
        ("R-05", "R-06"),
        ("O20261115001", "O20261115002"),
        ("运费由商家承担", "运费商家出"),
        ("退货运费由商家承担", "运费商家承担"),
    ],
)
def test_signals_never_relax_ids_numbers_or_structure(keyword, text):
    """这些断言不在信号规格里，必须走精确匹配（原文/去空白）。"""
    assert keyword not in _FACT_SIGNAL_SPECS
    assert _contains_fact(text, keyword) is False


def test_signals_preserve_whitespace_normalization():
    assert _contains_fact("签收后7天内可申请", "7 天") is True
    assert _contains_fact("第7日24:00前提交", "第 7 日") is True


def test_forbidden_path_still_uses_strict_contains():
    """`_contains` 本身保持严格子串语义（禁词判定与必含判定的第一跳）。

    B6 起禁词断言改走 `_contains_fact`（改写说法同样算命中，见 test_b6_*），
    但底层的精确匹配没有被削弱：`_contains` 仍然只认原文 / 去空白。
    """
    assert _contains("订单还没签收", "尚未签收") is False
    assert _contains_fact("订单还没签收", "尚未签收") is True


# ---------------------------------------------------------------------------
# 4) B5 样本纪律
# ---------------------------------------------------------------------------
def _seq(case):
    return "||".join(case["messages"])


def test_b5_sample_structure():
    sample = load_llm_sample5()
    assert len(sample) == 10
    assert sum(len(c["messages"]) for c in sample) == 11
    assert all(c["split"] == "llm_sample5" for c in sample)
    for case in sample:
        assert len(case["steps"]) == len(case["messages"]), case["id"]
        assert case["expect"] and case.get("source_case") and case.get("focus")
    assert len({c["id"] for c in sample}) == 10
    assert LLM_SAMPLE5_PATH.name == "cases_llm_sample5.json"
    assert len(json.loads(LLM_SAMPLE5_PATH.read_text(encoding="utf-8"))) == 10


def test_b5_zero_overlap_with_all_earlier_rounds():
    b5 = load_llm_sample5()
    earlier = (
        load_llm_sample() + load_llm_sample2()
        + load_llm_sample3() + load_llm_sample4()
    )
    assert not ({c["id"] for c in b5} & {c["id"] for c in earlier})
    earlier_seqs = {_seq(c) for c in earlier}
    for case in b5:
        assert _seq(case) not in earlier_seqs, case["id"]


def test_b5_never_touches_holdout2():
    b5 = load_llm_sample5()
    h2 = load_holdout2()
    h2_ids = {c["id"] for c in h2}
    h2_seqs = {_seq(c) for c in h2}
    assert not ({c["id"] for c in b5} & h2_ids)
    assert all(c["source_case"] not in h2_ids for c in b5)
    assert all(not c["source_case"].startswith("k") for c in b5)
    for case in b5:
        assert _seq(case) not in h2_seqs


def test_b5_not_part_of_full_eval():
    assert len(load_all_cases()) == 48
    assert not ({c["id"] for c in load_all_cases()} & {c["id"] for c in load_llm_sample5()})


def test_b5_rules_baseline_is_reproducible():
    report = run("multi", cases=load_llm_sample5(), settings_overrides={"llm_mode": "rules"})
    assert report["total"] == 10
    assert report["success_rate"] == 100.0
    assert report["hallucination_rate"] == 0.0
    assert report["handoff_loss"] == 0
    assert set(report["by_split"]) == {"llm_sample5"}


# ---------------------------------------------------------------------------
# 5) 端到端：等价改写被严格口径接住，反转被护栏拦下
# ---------------------------------------------------------------------------
def _only(case_id):
    return [c for c in load_llm_sample5() if c["id"] == case_id]


class _InjectLLM:
    """离线假模型：改写话术，classify/polish 都由它接管。"""

    def __init__(self, transform):
        self.transform = transform

    def classify(self, text, state="IDLE", focused=True):
        from app.llm import RuleLLM
        return RuleLLM().classify(text, state=state, focused=focused)

    def polish(self, draft, context=None):
        return self.transform(draft), 321


def _hook(transform):
    return lambda coordinator: setattr(coordinator, "llm", _InjectLLM(transform))


def test_end_to_end_b4_failed_wording_now_passes():
    """B4 失败的「未能查询到订单」在 B5 严格口径下应通过。"""
    report = run(
        "multi", cases=_only("w01"),
        settings_overrides={"llm_mode": "rules"},
        coordinator_hook=_hook(lambda d: d.replace("没有查到订单", "未能查询到订单")),
    )
    row = report["cases"][0]
    assert row["passed"] is True, row["failures"]


def test_end_to_end_polarity_flip_still_falls_back():
    """语义反转被护栏回退到模板，等价改写则不被护栏误杀。"""
    from app.grounding import verify

    flipped = verify(
        "没有查到订单 O999。",
        "已经查到订单 O999。",
    )
    assert flipped.ok is False
    assert "negation_flip" in flipped.violations
    assert "没有查到订单" in flipped.reply

    # 非结构性短语的等价改写，护栏本就不拦（事实交给断言层判定）
    equiv = verify("没有查到订单 O999。", "未能查询到订单 O999。")
    assert equiv.ok is True
