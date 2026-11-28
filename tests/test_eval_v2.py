"""新评测口径的回归测试：逐步断言、答案正确性、幻觉检测、事件留痕、留出集。"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.eval.runner import (
    _Checker,
    _check_final,
    _check_hallucinations,
    compare_full,
    load_all_cases,
    load_cases,
    load_dev2,
    load_holdout2,
    run,
    sensitivity_demo,
)

AUTHORITY = {
    "clause_ids": {f"R-0{i}" for i in range(1, 9)},
    "known_clause_ids": {f"R-0{i}" for i in range(1, 9)},
    "long_ids": {"O20261115001", "RT20261122AAAAAA"},
    "amounts": {299.0},
    "ticket_ids": {"RT20261122AAAAAA"},
}


def _result(reply: str, *, ticket=None, citations=None, kinds=()):
    return SimpleNamespace(
        reply=reply,
        citations=citations or [],
        ticket=ticket,
        trace=[{"detail": {"intent": "return", "order_id": None}}],
        slots={},
        meta={},
        state="IDLE",
    )


def _coordinator(kinds=()):
    return SimpleNamespace(
        db=SimpleNamespace(list_events=lambda session_id: [{"kind": k} for k in kinds])
    )


@pytest.fixture(scope="module")
def full():
    return compare_full()


# ---------------------------------------------------------------------------
# 数据集结构
# ---------------------------------------------------------------------------
def test_splits_are_separated_and_holdout2_is_untouched():
    dev, dev2, holdout2 = load_cases(), load_dev2(), load_holdout2()
    assert len(dev) >= 12 and len(dev2) >= 12 and len(holdout2) >= 12
    ids = [{c["id"] for c in group} for group in (dev, dev2, holdout2)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert all(c.get("split") == "dev" for c in dev)
    assert all(c.get("split") == "dev2" for c in dev2)
    assert all(c.get("split") == "holdout2" for c in holdout2)
    for case in load_all_cases():
        assert len(case.get("steps", [])) == len(case["messages"]), case["id"]
        assert case["expect"]
    # 三段措辞互不复用；holdout2 尤其不能抄 dev / dev2
    dev_text = {" ".join(c["messages"]) for c in dev}
    dev2_text = {" ".join(c["messages"]) for c in dev2}
    assert all(" ".join(c["messages"]) not in dev_text for c in dev2 + holdout2)
    assert all(" ".join(c["messages"]) not in dev2_text for c in holdout2)
    assert any("我要退货" not in " ".join(c["messages"]) for c in holdout2)


def test_holdout2_phrases_never_leaked_into_classifier_source():
    """留出集整句不得出现在规则引擎源码里——防止「照着留出集加词表」刷分。"""
    source = (Path(__file__).resolve().parents[1] / "app" / "llm.py").read_text(encoding="utf-8")
    for case in load_holdout2():
        for message in case["messages"]:
            if len(message) >= 6:
                assert message not in source, case["id"]


def test_strict_checks_far_more_assertions_than_legacy(full):
    multi = full["multi"]
    assert multi["assertions_total"] > multi["legacy_checks_total"] * 3
    assert multi["legacy_checks_total"] > 0


# ---------------------------------------------------------------------------
# 幻觉检测（纯函数，不依赖主链路）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "reply,expected_category",
    [
        ("依据 R-99 可退", "invalid_citation"),
        ("另见订单 O20261119999", "fabricated_id"),
        ("退款 ¥999.00", "fabricated_amount"),
        ("亲，保证退款～", "ungrounded_promise"),
    ],
)
def test_hallucination_detector_catches_fabrications(reply, expected_category):
    checker = _Checker()
    final = _result(reply)
    _check_hallucinations(checker, [final], final, AUTHORITY)
    assert expected_category in {c for c, _ in checker.failures}


def test_hallucination_detector_allows_grounded_reply():
    checker = _Checker()
    reply = "查到订单 O20261115001，金额 ¥299.00，依据 R-01、R-05。"
    final = _result(reply, citations=[{"clause_id": "R-01"}])
    _check_hallucinations(checker, [final], final, AUTHORITY)
    assert checker.failures == []


def test_hallucination_detector_catches_phantom_and_missing_ticket_id():
    phantom = _Checker()
    final = _result("工单 RT20261122ZZZZZZ 已创建")
    _check_hallucinations(phantom, [final], final, AUTHORITY)
    categories = {c for c, _ in phantom.failures}
    assert "fabricated_ticket" in categories and "phantom_ticket" in categories

    missing = _Checker()
    final2 = _result("售后工单已创建，请等待取件", ticket={"ticket_id": "RT20261122AAAAAA"})
    _check_hallucinations(missing, [final2], final2, AUTHORITY)
    assert "missing_ticket_id" in {c for c, _ in missing.failures}


# ---------------------------------------------------------------------------
# 最终事实 / 禁词 / 事件留痕
# ---------------------------------------------------------------------------
def test_final_facts_forbid_and_events():
    case = {
        "facts": ["O20261115001"],
        "forbid": ["保证退款"],
        "require_events": ["security_violation"],
        "forbid_events": ["ticket_create_attempt"],
    }
    checker = _Checker()
    final = _result("订单 O20261115001 没有权限查看")
    _check_final(checker, case, final, _coordinator(["security_violation"]), "s")
    assert checker.failures == []

    checker2 = _Checker()
    final2 = _result("已为您办理，保证退款")
    _check_final(checker2, case, final2, _coordinator(["ticket_create_attempt"]), "s")
    categories = {c for c, _ in checker2.failures}
    assert {"missing_fact", "forbidden_phrase", "security_leak", "forbidden_event"} <= categories


# ---------------------------------------------------------------------------
# 端到端：逐步断言与口径敏感性
# ---------------------------------------------------------------------------
def test_step_assertion_catches_mid_conversation_regression_that_legacy_misses():
    """第 1 轮状态断言写错（期望 TICKETED），最后一轮仍然正确：
    旧口径只看最后一轮 → 通过；严格口径看每一轮 → 失败。"""
    case = {
        "id": "x_step",
        "name": "中间步骤回归",
        "split": "dev",
        "user_id": "u_1001",
        "messages": ["我要退货", "O20261115001"],
        "expect": {"state": "AWAIT_CONFIRM"},
        "steps": [
            {"state": "TICKETED"},
            {"state": "AWAIT_CONFIRM", "no_ticket": True},
        ],
    }
    report = run("multi", cases=[case], with_detail=True)
    row = report["cases"][0]
    assert row["legacy_passed"] is True
    assert row["passed"] is False
    assert "step_mismatch" in row["categories"]
    assert row["steps_passed"] == 1


def test_sensitivity_demo_proves_strict_is_stricter():
    demo = sensitivity_demo()
    assert demo["legacy_passed"] is True
    assert demo["strict_passed"] is False
    assert {"missing_ticket_id", "fabricated_amount", "ungrounded_promise"} <= set(demo["categories"])


def test_holdout2_is_the_only_generalization_evidence(full):
    """dev / dev2 已参与调参，不作为泛化证据；只有 holdout2 的成绩可以外推。"""
    single, multi = full["single"], full["multi"]
    for report in (single, multi):
        assert set(report["by_split"]) == {"dev", "dev2", "holdout2"}
        assert 0 <= report["by_split"]["holdout2"]["success_rate"] <= 100
    assert multi["by_split"]["dev"]["success_rate"] == 100.0
    # 不硬断言「留出集一定更差」：修复后可能持平，成绩如实记录即可
    assert (
        single["by_split"]["holdout2"]["success_rate"]
        <= multi["by_split"]["holdout2"]["success_rate"]
    )
    # 规则引擎只输出代码模板，不应被幻觉检测器误报
    assert single["hallucination_rate"] == 0.0
    assert multi["hallucination_rate"] == 0.0
    assert single["security_block_rate"] == 100.0
    assert multi["security_block_rate"] == 100.0
    assert single["handoff_loss"] == 0 and multi["handoff_loss"] == 0


def test_reports_keep_both_scoring_views(full):
    for mode in ("single", "multi"):
        report = full[mode]
        assert 0 <= report["success_rate"] <= 100
        assert 0 <= report["legacy_success_rate"] <= 100
        assert 0 <= report["answer_correct_rate"] <= 100
        assert 0 <= report["step_pass_rate"] <= 100
        assert report["assertions_total"] >= report["legacy_checks_total"]
        assert set(report["by_split"]) == {"dev", "dev2", "holdout2"}
        for row in report["cases"]:
            assert row["assertions_passed"] <= row["assertions_total"]
            assert row["split"] in ("dev", "dev2", "holdout2")
            assert isinstance(row["categories"], list)


# --- B1 修复：严格断言的空白归一匹配 ---------------------------------------
# 模型答「7天内」，断言写「7 天」，事实全对只差空格——这类格式差异不应
# 记为 missing_fact；但归一只消除排版差异，不做语义匹配，缺数字仍失败。
def test_contains_is_whitespace_insensitive():
    from app.eval.runner import _contains

    assert _contains("签收后7天内可申请", "7 天")
    assert _contains("第 7 日 24:00 前提交", "第7日")
    assert _contains("已8天，超过期限", "8 天")
    # 原文直接匹配的照旧通过
    assert _contains("还剩 7 天", "7 天")


def test_contains_still_strict_on_real_missing_facts():
    from app.eval.runner import _contains

    # 数字换成中文、缺数字、换数字：都是真缺事实，不得放行
    assert not _contains("签收后六天内可申请", "7 天")
    assert not _contains("签收后内可申请", "7 天")
    assert not _contains("签收后8天内可申请", "7 天")
    assert not _contains("", "7 天")


def test_forbidden_phrase_cannot_bypass_by_inserting_spaces():
    """禁词断言同样走归一匹配：模型不能靠加空格绕过 forbid。"""
    checker = _Checker()
    step = {"reply_not_contains": ["工单已创建"]}
    result = SimpleNamespace(
        reply="工单已 创建，请放心", state="TICKETED", ticket=None,
        citations=[], trace=[], slots={}, meta={},
    )
    from app.eval.runner import _check_step
    _check_step(checker, 1, step, result)
    assert ("forbidden_phrase", "第 1 轮回复不应出现「工单已创建」") in checker.failures


def test_step_contains_tolerates_formatting_difference():
    """B1 s01/s08 场景：必含事实只差空格时，逐步断言应通过。"""
    from app.eval.runner import _check_step

    checker = _Checker()
    step = {"reply_contains": ["7 天", "第 7 日", "R-01"]}
    result = SimpleNamespace(
        reply="按【R-01】规则，签收后7天内可退，第7日24:00前提交有效。",
        state="IDLE", ticket=None, citations=[], trace=[], slots={}, meta={},
    )
    assert _check_step(checker, 1, step, result) is True
    assert checker.failures == []


def test_legacy_scoring_stays_byte_exact_for_comparison():
    """旧口径保留原样子串匹配，继续作为历史对照（B1 报告的 80% 就是它）。"""
    from app.eval.runner import _evaluate_legacy

    case = {"expect": {"reply_contains": ["7 天"]}}
    result = SimpleNamespace(reply="签收后7天内可退", state="IDLE", ticket=None, citations=[], slots={})
    failures = _evaluate_legacy(case, result)
    assert failures == ["回复缺少关键词「7 天」"]
