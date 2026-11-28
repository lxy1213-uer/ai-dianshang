"""B1 抽样基线：数据纪律与诊断逻辑的回归测试。

这里**不调用外部 API**（conftest 已强制规则模式），只锁住三件事：
1. 抽样集结构完整、只来自 dev / dev2、绝不碰 holdout2；
2. 抽样集不并入 48 条全量口径；
3. 口径诊断只做归因，不改动正式成绩。
"""

from __future__ import annotations

import json
from pathlib import Path

from app.eval.runner import (
    LLM_SAMPLE_PATH,
    load_all_cases,
    load_cases,
    load_dev2,
    load_holdout2,
    load_llm_sample,
    run,
)


def test_sample_structure_and_size():
    sample = load_llm_sample()
    assert len(sample) == 10
    assert sum(len(c["messages"]) for c in sample) == 17
    assert all(c["split"] == "llm_sample" for c in sample)
    for case in sample:
        assert len(case["steps"]) == len(case["messages"]), case["id"]
        assert case["expect"], case["id"]
        assert case.get("source_case"), case["id"]
        assert case.get("focus"), case["id"]
    assert len({c["id"] for c in sample}) == 10


def test_sample_comes_only_from_dev_and_dev2():
    """抽样只允许从已参与调参的 dev / dev2 复制，不得新造说法冒充留出集。"""
    sample = load_llm_sample()
    allowed = {c["id"] for c in load_cases() + load_dev2()}
    assert all(c["source_case"] in allowed for c in sample)
    assert all(c["source_case"].startswith(("c", "h")) for c in sample)


def test_sample_never_touches_holdout2():
    """holdout2 是唯一干净的泛化证据，抽样集不得复用它的任何案例或措辞。"""
    sample = load_llm_sample()
    holdout2 = load_holdout2()
    assert not ({c["id"] for c in sample} & {c["id"] for c in holdout2})
    sample_text = {" ".join(c["messages"]) for c in sample}
    assert all(" ".join(c["messages"]) not in sample_text for c in holdout2)
    holdout_ids = {c["id"] for c in holdout2}
    assert all(c["source_case"] not in holdout_ids for c in sample)
    assert all(not c["source_case"].startswith("k") for c in sample)


def test_sample_is_not_part_of_full_eval():
    """llm_sample 不能污染 48 条全量口径。"""
    all_cases = load_all_cases()
    assert len(all_cases) == 48
    assert {c["split"] for c in all_cases} == {"dev", "dev2", "holdout2"}
    assert not ({c["id"] for c in all_cases} & {c["id"] for c in load_llm_sample()})


def test_rules_baseline_on_sample_is_reproducible():
    """规则侧在抽样集上 10/10 通过，是 B1 对照的锚点（不联网）。"""
    report = run(
        "multi",
        cases=load_llm_sample(),
        settings_overrides={"llm_mode": "rules"},
    )
    assert report["total"] == 10
    assert report["success_rate"] == 100.0
    assert report["hallucination_rate"] == 0.0
    assert report["handoff_loss"] == 0
    assert set(report["by_split"]) == {"llm_sample"}


def test_sample_file_is_valid_json_with_expected_path():
    assert LLM_SAMPLE_PATH.name == "cases_llm_sample.json"
    raw = json.loads(LLM_SAMPLE_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, list) and len(raw) == 10


# --- 诊断逻辑：只解释低分，绝不改成绩 -----------------------------------
def _diagnose():
    from scripts.run_llm_baseline import diagnose

    return diagnose


def test_diagnose_flags_whitespace_only_failure_without_changing_score():
    sample = [
        {
            "id": "s01",
            "name": "政策问答",
            "steps": [{"reply_contains": ["7 天", "第 7 日"]}],
            "facts": ["R-01", "7 天"],
            "expect": {},
        }
    ]
    llm = {
        "success_rate": 0.0,
        "passed": 0,
        "total": 1,
        "cases": [
            {
                "id": "s01",
                "passed": False,
                "failures": ["最终回复缺少关键事实「7 天」"],
                "reply_excerpt": "按平台规则【R-01】签收后7天内可申请，第7日24:00前提交视为期限内",
            }
        ],
    }
    d = _diagnose()({}, llm, sample)
    assert d["available"] is True
    assert [x["id"] for x in d["whitespace_only_failures"]] == ["s01"]
    assert d["semantic_or_flow_failures"] == []
    assert d["hypothetical_rate_if_compact"] == 100.0
    # 关键纪律：正式成绩不变
    assert d["formal_success_rate"] == 0.0
    assert llm["success_rate"] == 0.0


def test_diagnose_keeps_real_flow_failure_as_semantic():
    sample = [
        {
            "id": "s04",
            "name": "口语退货",
            "steps": [{"reply_contains": ["订单号"]}],
            "facts": ["订单号"],
            "expect": {"state": "AWAIT_ORDER"},
        }
    ]
    llm = {
        "success_rate": 0.0,
        "passed": 0,
        "total": 1,
        "cases": [
            {
                "id": "s04",
                "passed": False,
                "failures": ["第 1 轮状态应为 AWAIT_ORDER，实际 IDLE", "第 1 轮回复缺少「订单号」"],
                "reply_excerpt": "这句话我还没把握处理准确，可以帮你查订单、查物流或转人工。",
            }
        ],
    }
    d = _diagnose()({}, llm, sample)
    assert [x["id"] for x in d["semantic_or_flow_failures"]] == ["s04"]
    assert d["whitespace_only_failures"] == []
    assert d["hypothetical_rate_if_compact"] == 0.0


def test_diagnose_without_llm_side_is_explicitly_unavailable():
    d = _diagnose()({"success_rate": 100.0}, None, load_llm_sample())
    assert d["available"] is False
    assert "reason" in d


def test_ops_handles_empty_sink():
    from scripts.run_llm_baseline import _ops

    assert _ops([]) == {"turns": 0, "note": "未运行"}


# --- B2 复测样本：另开一轮的纪律 -------------------------------------------
# B1 样本已用于定位失败并驱动修复（意图状态护栏 + 断言空白归一），
# 按「留出集一旦用于调参就永久消耗」的纪律，B2 必须换一批全新案例。
def test_b2_sample_structure_and_size():
    from app.eval.runner import load_llm_sample2

    sample = load_llm_sample2()
    assert len(sample) == 10
    assert sum(len(c["messages"]) for c in sample) == 17
    assert all(c["split"] == "llm_sample2" for c in sample)
    for case in sample:
        assert len(case["steps"]) == len(case["messages"]), case["id"]
        assert case["expect"], case["id"]
        assert case.get("source_case"), case["id"]
    assert len({c["id"] for c in sample}) == 10
    assert not ({c["id"] for c in sample} & {c["id"] for c in load_llm_sample()})


def test_b2_sample_comes_only_from_dev_and_dev2():
    from app.eval.runner import load_llm_sample2

    sample = load_llm_sample2()
    allowed = {c["id"] for c in load_cases() + load_dev2()}
    assert all(c["source_case"] in allowed for c in sample)


def test_b2_zero_overlap_with_b1_by_source_case():
    """B2 与 B1 必须零重叠：同一批案例边测边改就等于照着答案刷分。"""
    from app.eval.runner import load_llm_sample2

    b1_sources = {c["source_case"] for c in load_llm_sample()}
    b2_sources = {c["source_case"] for c in load_llm_sample2()}
    assert not (b1_sources & b2_sources)


def test_b2_never_touches_holdout2():
    from app.eval.runner import load_llm_sample2

    sample = load_llm_sample2()
    holdout2 = load_holdout2()
    holdout2_ids = {c["id"] for c in holdout2}
    assert all(c["source_case"] not in holdout2_ids for c in sample)
    assert all(not c["source_case"].startswith("k") for c in sample)
    assert not ({c["id"] for c in sample} & holdout2_ids)


def test_b2_is_not_part_of_full_eval():
    from app.eval.runner import load_llm_sample2

    all_cases = load_all_cases()
    assert len(all_cases) == 48
    assert {c["split"] for c in all_cases} == {"dev", "dev2", "holdout2"}
    assert not ({c["id"] for c in all_cases} & {c["id"] for c in load_llm_sample2()})


def test_b2_rules_baseline_is_reproducible():
    """规则侧在 B2 样本上 10/10 通过（含空白归一后的断言引擎），是 B2 对照的锚点。"""
    from app.eval.runner import load_llm_sample2

    report = run(
        "multi",
        cases=load_llm_sample2(),
        settings_overrides={"llm_mode": "rules"},
    )
    assert report["total"] == 10
    assert report["success_rate"] == 100.0
    assert report["hallucination_rate"] == 0.0
    assert report["handoff_loss"] == 0
    assert set(report["by_split"]) == {"llm_sample2"}


def test_state_guard_prompt_examples_do_not_leak_eval_sentences():
    """提示词 few-shot 例句是新造的：任何评测集里有意义的整句（≥4 字）
    不得出现在 classify_with_usage 的源码里（防止照着样本刷分）。
    4 字以下的超短句（如「确认」）是业务通用词，任何意图提示词都必然出现，不在此列。
    """
    from app.eval.runner import load_llm_sample, load_llm_sample2
    from app.llm import OpenAICompatibleLLM
    import inspect

    src = inspect.getsource(OpenAICompatibleLLM.classify_with_usage)
    sentences = set()
    for loader in (load_cases, load_dev2, load_holdout2, load_llm_sample, load_llm_sample2):
        for case in loader():
            sentences.update(m for m in case["messages"] if len(m) >= 4)
    leaked = [s for s in sentences if s in src]
    assert leaked == []


# --- B3 复测样本：润色丢词修复后的另开一轮纪律 -------------------------------
# B2 样本已用于定位 t08/t09 失败（润色丢结构性短语）并驱动修复
# （grounding REQUIRED_PHRASES + missing_phrase 校验），按纪律 B3 必须再换新案例。
def test_b3_sample_structure_and_size():
    from app.eval.runner import load_llm_sample3

    sample = load_llm_sample3()
    assert len(sample) == 10
    assert sum(len(c["messages"]) for c in sample) == 14
    assert all(c["split"] == "llm_sample3" for c in sample)
    for case in sample:
        assert len(case["steps"]) == len(case["messages"]), case["id"]
        assert case["expect"], case["id"]
        assert case.get("source_case"), case["id"]
    assert len({c["id"] for c in sample}) == 10


def test_b3_comes_only_from_dev_and_dev2():
    from app.eval.runner import load_llm_sample3

    sample = load_llm_sample3()
    allowed = {c["id"] for c in load_cases() + load_dev2()}
    assert all(c["source_case"] in allowed for c in sample)


def test_b3_zero_overlap_with_b1_and_b2_by_source_case():
    from app.eval.runner import load_llm_sample, load_llm_sample2, load_llm_sample3

    b1 = {c["source_case"] for c in load_llm_sample()}
    b2 = {c["source_case"] for c in load_llm_sample2()}
    b3 = {c["source_case"] for c in load_llm_sample3()}
    assert not (b3 & b1)
    assert not (b3 & b2)


def test_b3_never_touches_holdout2():
    from app.eval.runner import load_llm_sample3

    sample = load_llm_sample3()
    holdout2_ids = {c["id"] for c in load_holdout2()}
    assert all(c["source_case"] not in holdout2_ids for c in sample)
    assert not ({c["id"] for c in sample} & holdout2_ids)


def test_b3_is_not_part_of_full_eval():
    from app.eval.runner import load_llm_sample3

    all_cases = load_all_cases()
    assert len(all_cases) == 48
    assert {c["split"] for c in all_cases} == {"dev", "dev2", "holdout2"}
    assert not ({c["id"] for c in all_cases} & {c["id"] for c in load_llm_sample3()})


def test_b3_rules_baseline_is_reproducible():
    """规则侧在 B3 样本上 10/10 通过，是 B3 对照的锚点。"""
    from app.eval.runner import load_llm_sample3

    report = run(
        "multi",
        cases=load_llm_sample3(),
        settings_overrides={"llm_mode": "rules"},
    )
    assert report["total"] == 10
    assert report["success_rate"] == 100.0
    assert report["hallucination_rate"] == 0.0
    assert set(report["by_split"]) == {"llm_sample3"}
