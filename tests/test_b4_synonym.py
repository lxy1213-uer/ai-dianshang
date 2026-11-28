"""B4 受控同义断言口径 + 定向探针样本纪律。

这里**不调用外部 API**（conftest 已强制规则模式），锁住三件事：
1. 严格口径必含事实断言支持受控同义等价，但否定反转 / 数字 / 单号 /
   结构性短语一律不放宽；禁词断言完全不受影响；
2. B4 探针集结构完整、与 B1/B2/B3 零重叠、不碰 holdout2、不并入 48 条；
3. 端到端：把规则模板里的结论短语换成同义说法，严格口径仍能通过；
   把结论反转（查无订单→已查到）则被幻觉/反转护栏拦下。
"""

from __future__ import annotations

import json

import pytest

from app.eval.runner import (
    LLM_SAMPLE4_PATH,
    _EQUIVALENT_PHRASES,
    _contains,
    _contains_fact,
    load_all_cases,
    load_holdout2,
    load_llm_sample,
    load_llm_sample2,
    load_llm_sample3,
    load_llm_sample4,
    run,
)


# ---------------------------------------------------------------------------
# 1) 同义词典本身的结构约束
# ---------------------------------------------------------------------------
def test_synonym_groups_are_closed_and_nonempty():
    assert len(_EQUIVALENT_PHRASES) >= 4
    seen: set[str] = set()
    for group in _EQUIVALENT_PHRASES:
        assert len(group) >= 2, group
        assert len({m for m in group}) == len(group), f"组内重复：{group}"
        for member in group:
            assert member and member.strip(), group
            seen.add(member)
    assert len(seen) == sum(len(g) for g in _EQUIVALENT_PHRASES), "同一短语不能出现在多组"


def test_synonym_groups_do_not_cross_negation_polarity():
    """同义组不能同时含肯定与否定：否则「查到订单」会放行「没有查到订单」断言。"""
    for group in _EQUIVALENT_PHRASES:
        neg_markers = ("没有", "未", "没", "不能", "不可", "无法")
        has_neg = [any(m in word for m in neg_markers) for word in group]
        assert all(has_neg) or not any(has_neg), f"同义组极性不一致：{group}"


# ---------------------------------------------------------------------------
# 2) _contains_fact 放行 / 拦截边界
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "keyword, text",
    [
        ("没有查到订单", "没有查到订单 O999"),
        ("没有查到订单", "未查询到订单 O999"),
        ("没有查到订单", "没查到订单 O999"),
        ("没有查到订单", "未查到该订单 O999"),
        ("尚未签收", "订单尚未签收"),
        ("尚未签收", "订单还未签收"),
        ("尚未签收", "订单还没签收"),
        ("已退款", "这笔已经退款了"),
        ("已退款", "已完成退款处理"),
        ("不能重复申请", "无法重复申请退货"),
        ("不能重复申请", "不可重复申请退货"),
    ],
)
def test_contains_fact_allows_equivalent(keyword, text):
    assert _contains_fact(text, keyword) is True


@pytest.mark.parametrize(
    "keyword, text",
    [
        ("没有查到订单", "已查到订单 O999"),       # 否定反转
        ("没有查到订单", "查到订单 O999"),         # 丢了否定词
        ("尚未签收", "已经签收"),                   # 极性反转
        ("已退款", "未退款"),                       # 极性反转
        ("运费由商家承担", "运费商家出"),           # 结构性短语不在同义词典
        ("R-05", "R-06"),                           # 单号不放宽
        ("O20261115001", "O20261115002"),           # 订单号不放宽
    ],
)
def test_contains_fact_rejects_non_equivalent(keyword, text):
    assert _contains_fact(text, keyword) is False


@pytest.mark.parametrize(
    "keyword, text",
    [
        ("7 天", "签收后8天"),
        ("7 天", "签收后六天"),
        ("第 7 日", "第 8 日"),
        ("1-3 个工作日", "1-2 个工作日"),
    ],
)
def test_contains_fact_never_relaxes_numbers(keyword, text):
    """数字断言永远只走原文 / 空白归一，绝不被同义词典放宽。"""
    assert _contains_fact(text, keyword) is False
    assert _contains(text, keyword) is False


def test_contains_fact_matches_whitespace_only_as_before():
    """兼容 B1 的空白归一：7 天 ↔ 7天 仍通过。"""
    assert _contains_fact("签收后7天内可申请", "7 天") is True
    assert _contains_fact("第7日24:00前提交", "第 7 日") is True


def test_forbidden_assertion_path_is_not_relaxed():
    """禁词断言走 `_contains`（未改）：同义词典里的词不会让禁词漏判。

    若禁词写「尚未签收」，回复「还没签收」在原口径下不算命中；这正是
    B4 的设计取舍——放宽只在必含侧，避免放宽禁词检测造成安全漏判。
    """
    assert _contains("订单还没签收", "尚未签收") is False
    assert _contains_fact("订单还没签收", "尚未签收") is True


def test_legacy_empty_keyword_still_true():
    assert _contains_fact("任意文本", "") is True
    assert _contains("任意文本", "") is True


# ---------------------------------------------------------------------------
# 3) B4 样本纪律
# ---------------------------------------------------------------------------
def test_b4_sample_structure_and_size():
    sample = load_llm_sample4()
    assert len(sample) == 10
    assert sum(len(c["messages"]) for c in sample) == 13
    assert all(c["split"] == "llm_sample4" for c in sample)
    for case in sample:
        assert len(case["steps"]) == len(case["messages"]), case["id"]
        assert case["expect"], case["id"]
        assert case.get("source_case"), case["id"]
        assert case.get("focus"), case["id"]
    assert len({c["id"] for c in sample}) == 10


def test_b4_sample_file_path_and_valid_json():
    assert LLM_SAMPLE4_PATH.name == "cases_llm_sample4.json"
    raw = json.loads(LLM_SAMPLE4_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, list) and len(raw) == 10


def _distinctive(seq: list[str]) -> str:
    """消息序列的指纹：通用短句（「我要退货」/「确认」）单看会重复，
    但整段对话序列是案例唯一的；用它判定「整句照抄」而不是逐条比通用词。"""
    return "||".join(seq)


def test_b4_zero_overlap_with_earlier_rounds_by_id_and_sequence():
    """与 B1/B2/B3 零重叠：id 不重复，整段消息序列也不重复。"""
    b4 = load_llm_sample4()
    earlier = load_llm_sample() + load_llm_sample2() + load_llm_sample3()
    earlier_ids = {c["id"] for c in earlier}
    earlier_seqs = {_distinctive(c["messages"]) for c in earlier}
    assert not ({c["id"] for c in b4} & earlier_ids)
    for case in b4:
        assert _distinctive(case["messages"]) not in earlier_seqs, case["id"]


def test_b4_never_touches_holdout2():
    b4 = load_llm_sample4()
    holdout2 = load_holdout2()
    holdout_ids = {c["id"] for c in holdout2}
    holdout_seqs = {_distinctive(c["messages"]) for c in holdout2}
    assert not ({c["id"] for c in b4} & holdout_ids)
    assert all(c["source_case"] not in holdout_ids for c in b4)
    assert all(not c["source_case"].startswith("k") for c in b4)
    for case in b4:
        assert _distinctive(case["messages"]) not in holdout_seqs, case["id"]


def test_b4_is_not_part_of_full_eval():
    all_cases = load_all_cases()
    assert len(all_cases) == 48
    assert {c["split"] for c in all_cases} == {"dev", "dev2", "holdout2"}
    assert not ({c["id"] for c in all_cases} & {c["id"] for c in load_llm_sample4()})


def test_b4_rules_baseline_is_reproducible():
    """规则侧在 B4 样本上 10/10 通过，是同义口径修复的锚点。"""
    report = run("multi", cases=load_llm_sample4(), settings_overrides={"llm_mode": "rules"})
    assert report["total"] == 10
    assert report["success_rate"] == 100.0
    assert report["hallucination_rate"] == 0.0
    assert report["handoff_loss"] == 0
    assert set(report["by_split"]) == {"llm_sample4"}
