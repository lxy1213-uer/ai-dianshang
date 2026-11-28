"""B6 完成型结论（建单确认）规格 + 禁词同口径 + 不可见字符清洗。

上游证据（B5 实测失败 w10）：真实模型把模板「售后工单已创建：RT…」润色成
「售后工单已成功创建：RT…」，整词断言因插入「成功」漏判。B5 那批样本已用于
定位失败，按纪律不再复用，本轮另开 10 条新措辞探针（llm_sample6）。

本文件不调用外部 API（conftest 强制规则模式），锁住五件事：
1. 完成型规格按「动作邻域窗口 + 完成语素 + 对象语素」判定，接得住插入语改写；
2. 否定硬闸是硬门槛——「工单没有创建成功」绝不放行，且不跨小句误判；
3. 禁词断言与必含断言同口径：改写说法同样算「出现了该禁词」；
4. 不可见字符（零宽 / 词连接符）在护栏入口被清洗，用户拿到的回复干净；
5. B6 样本结构与零重叠纪律，规则侧基线可复现，全量 48 条无退化。
"""

from __future__ import annotations

import json

import pytest

from app.eval.runner import (
    LLM_SAMPLE6_PATH,
    _COMPLETION_SIGNAL_SPECS,
    _contains,
    _contains_fact,
    _forbidden_present,
    load_all_cases,
    load_holdout2,
    load_llm_sample,
    load_llm_sample2,
    load_llm_sample3,
    load_llm_sample4,
    load_llm_sample5,
    load_llm_sample6,
    run,
)
from app.grounding import strip_invisible, verify

JUNK = "\u200b\u200c\u2060\ufeff\u200d"


# ---------------------------------------------------------------------------
# 1) 规格结构
# ---------------------------------------------------------------------------
def test_completion_specs_share_one_fact():
    assert set(_COMPLETION_SIGNAL_SPECS) == {
        "工单已创建", "已创建工单", "售后工单已创建", "工单创建成功",
    }
    specs = list(_COMPLETION_SIGNAL_SPECS.values())
    assert all(spec is specs[0] for spec in specs), "别名必须指向同一份规格"
    spec = specs[0]
    assert spec["objects"] and spec["actions"] and spec["done"]
    assert set(spec["gates"]) == set(spec["actions"]), "每个动作都要有否定邻接正则"


# ---------------------------------------------------------------------------
# 2) 放行：B5 漏掉的插入语 / 倒装改写
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "售后工单已成功创建：RT20261126DC85F7",
        "您的售后工单已成功创建，工单号 RT20261126DC85F7",
        "已为您创建售后工单：RT20261126DC85F7",
        "工单创建成功，编号 RT20261126DC85F7",
        "工单已经创建好了，RT20261126DC85F7",
        "已成功生成售后工单 RT20261126DC85F7",
        "该售后单已完成创建",
        "建单成功，工单号是 RT20261126DC85F7",
        "售后工单已开单",
    ],
)
def test_completion_allows_rewritten_wording(text):
    assert _contains_fact(text, "工单已创建") is True


@pytest.mark.parametrize("keyword", ["已创建工单", "售后工单已创建", "工单创建成功"])
def test_completion_alias_keys_behave_identically(keyword):
    assert _contains_fact("已为您创建售后工单", keyword) is True
    assert _contains_fact("工单没有创建成功", keyword) is False


# ---------------------------------------------------------------------------
# 3) 否定硬闸：含全部字面要素的否定说法绝不放行
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "工单没有创建成功，请稍后重试",
        "没有成功创建工单，请重试",
        "未能创建工单：订单状态异常",
        "无法创建工单，请联系人工",
        "工单尚未创建",
        "工单创建失败",
        "创建工单失败，工单号暂无",
        "暂不创建工单",
        "工单没能生成",
        "系统返回：工单创建不成功",
    ],
)
def test_completion_rejects_negated_wording(text):
    assert _contains_fact(text, "工单已创建") is False


def test_negation_gate_does_not_cross_sentence_boundary():
    """句首的「未」属于另一个事实，不能把后面的建单确认也否掉。"""
    assert _contains_fact("该订单未签收，工单已创建", "工单已创建") is True
    assert _contains_fact("订单未查询到物流，但售后工单已成功创建", "工单已创建") is True
    assert _contains_fact("工单未创建", "工单已创建") is False


def test_completion_requires_done_and_object():
    """没有完成语素 / 对象语素的「将创建」「建单时效」不算已创建。"""
    assert _contains_fact("确认后即可创建工单", "工单已创建") is False
    assert _contains_fact("我会创建售后工单，请回复确认", "工单已创建") is False
    assert _contains_fact("建单需要 1 个工作日", "工单已创建") is False
    assert _contains_fact("已创建订单记录", "工单已创建") is False


def test_completion_is_authoritative():
    """完成型规格命中即具权威性：判否后不得回落到更宽松的路径。"""
    assert _contains_fact("工单没有创建成功", "工单已创建") is False
    # 精确匹配仍然是第一跳：模板原文永远通过
    assert _contains_fact("售后工单已创建：RT20261126DC85F7", "工单已创建") is True


# ---------------------------------------------------------------------------
# 4) 禁词同口径：改写说法同样算命中（安全侧不能漏检）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "reply",
    [
        "售后工单已成功创建：RT20261126DC85F7",
        "已为您创建售后工单，工单号 RT20261126DC85F7",
        "工单创建成功，RT20261126DC85F7",
    ],
)
def test_forbidden_detects_rewritten_claim(reply):
    assert _forbidden_present(reply, "工单已创建") is True


@pytest.mark.parametrize(
    "reply",
    [
        "我会创建售后工单，请回复「确认」；回复「取消」放弃本次申请。",
        "工单没有创建成功，请稍后重试。",
        "这笔订单暂时不能直接办理退货。",
    ],
)
def test_forbidden_does_not_misfire(reply):
    assert _forbidden_present(reply, "工单已创建") is False


def test_forbidden_keeps_strict_behaviour_for_other_words():
    """非完成型禁词仍是精确子串：不因 B6 引入模糊匹配。"""
    assert _contains("运费由商家承担", "运费由商家承担") is True
    assert _forbidden_present("运费由商家承担", "运费需自理") is False
    assert _forbidden_present("O20261115002", "O20261115001") is False


# ---------------------------------------------------------------------------
# 5) 不可见字符清洗
# ---------------------------------------------------------------------------
def test_strip_invisible_counts_and_cleans():
    cleaned, removed = strip_invisible("工单已创建" + JUNK * 3)
    assert cleaned == "工单已创建"
    assert removed == len(JUNK) * 3
    assert strip_invisible("") == ("", 0)


def test_verify_returns_clean_reply_and_records_strip():
    draft = "售后工单已创建：RT20261126DC85F7\n订单 O20261115001 ｜ 七天无理由退货"
    polished = "售后工单已成功创建：RT20261126DC85F7，订单 O20261115001 ｜ 七天无理由退货。" + JUNK * 5
    result = verify(draft, polished)
    assert result.ok is True, result.violations
    assert result.used_polished is True
    assert not any(ch in result.reply for ch in JUNK)
    assert result.detail.get("invisible_stripped") == len(JUNK) * 5


def test_verify_strips_junk_even_when_unchanged():
    draft = "售后工单已创建：RT20261126DC85F7"
    result = verify(draft, draft + JUNK * 2)
    assert result.ok is True
    assert result.used_polished is False
    assert result.reply == draft
    assert result.detail.get("invisible_stripped") == len(JUNK) * 2


# ---------------------------------------------------------------------------
# 6) B6 样本纪律
# ---------------------------------------------------------------------------
def _seq(case):
    return "||".join(case["messages"])


def test_b6_sample_structure():
    sample = load_llm_sample6()
    assert len(sample) == 10
    assert sum(len(c["messages"]) for c in sample) == 14
    assert all(c["split"] == "llm_sample6" for c in sample)
    for case in sample:
        assert len(case["steps"]) == len(case["messages"]), case["id"]
        assert case["expect"] and case.get("source_case") and case.get("focus")
        assert LLM_SAMPLE6_PATH.name == "cases_llm_sample6.json"
    assert len({c["id"] for c in sample}) == 10
    assert len(json.loads(LLM_SAMPLE6_PATH.read_text(encoding="utf-8"))) == 10


def test_b6_zero_overlap_with_all_earlier_rounds():
    b6 = load_llm_sample6()
    earlier = (
        load_llm_sample() + load_llm_sample2() + load_llm_sample3()
        + load_llm_sample4() + load_llm_sample5()
    )
    assert not ({c["id"] for c in b6} & {c["id"] for c in earlier})
    earlier_seqs = {_seq(c) for c in earlier}
    for case in b6:
        assert _seq(case) not in earlier_seqs, case["id"]


def test_b6_never_touches_holdout2():
    b6 = load_llm_sample6()
    h2 = load_holdout2()
    assert not ({c["id"] for c in b6} & {c["id"] for c in h2})
    assert all(c["source_case"] not in {h["id"] for h in h2} for c in b6)
    assert all(not c["source_case"].startswith("k") for c in b6)
    h2_seqs = {_seq(c) for c in h2}
    for case in b6:
        assert _seq(case) not in h2_seqs


def test_b6_not_part_of_full_eval():
    assert len(load_all_cases()) == 48
    assert not ({c["id"] for c in load_all_cases()} & {c["id"] for c in load_llm_sample6()})


def test_b6_rules_baseline_is_reproducible():
    report = run("multi", cases=load_llm_sample6(), settings_overrides={"llm_mode": "rules"})
    assert report["total"] == 10
    assert report["success_rate"] == 100.0
    assert report["hallucination_rate"] == 0.0
    assert report["handoff_loss"] == 0
    assert set(report["by_split"]) == {"llm_sample6"}


def test_full_eval_has_no_regression_after_forbidden_switch():
    """禁词改走同一判定后，48 条全量严格口径不得出现退化。"""
    report = run("multi", cases=load_all_cases(), settings_overrides={"llm_mode": "rules"})
    assert report["total"] == 48
    assert report["success_rate"] == 100.0


# ---------------------------------------------------------------------------
# 7) 端到端：B5 真实改写被接住，未建单说法仍判失败
# ---------------------------------------------------------------------------
class _InjectLLM:
    """离线假模型：polish 由 transform 决定，理解层仍走规则。"""

    def __init__(self, transform):
        self.transform = transform

    def classify(self, text, state="IDLE", focused=True):
        from app.llm import RuleLLM
        return RuleLLM().classify(text, state=state, focused=focused)

    def polish(self, draft, context=None):
        return self.transform(draft), 321


def _hook(transform):
    return lambda coordinator: setattr(coordinator, "llm", _InjectLLM(transform))


def _ticket_cases():
    return [c for c in load_llm_sample6() if c["id"] in ("x02", "x03", "x04", "x06")]


@pytest.mark.parametrize(
    "old,new",
    [
        ("售后工单已创建", "售后工单已成功创建"),
        ("售后工单已创建：", "已为您创建售后工单："),
        ("售后工单已创建：", "工单创建成功："),
    ],
)
def test_end_to_end_b5_failed_wording_now_passes(old, new):
    report = run(
        "multi", cases=_ticket_cases(),
        settings_overrides={"llm_mode": "rules"},
        coordinator_hook=_hook(lambda d: d.replace(old, new)),
    )
    assert report["success_rate"] == 100.0, [
        (c["id"], c["failures"]) for c in report["cases"] if not c["passed"]
    ]


def test_end_to_end_negated_claim_is_still_caught():
    """把建单确认改写成「未创建成功」必须判失败——护栏不管这个方向，断言层要管。"""
    report = run(
        "multi", cases=_ticket_cases(),
        settings_overrides={"llm_mode": "rules"},
        coordinator_hook=_hook(lambda d: d.replace("售后工单已创建：", "工单未创建成功：")),
    )
    assert report["success_rate"] == 0.0
    assert all(
        any("工单已创建" in f for f in c["failures"])
        for c in report["cases"]
    )


def test_end_to_end_invisible_junk_does_not_break_assertions():
    report = run(
        "multi", cases=_ticket_cases(),
        settings_overrides={"llm_mode": "rules"},
        coordinator_hook=_hook(lambda d: d + JUNK * 4),
    )
    assert report["success_rate"] == 100.0
