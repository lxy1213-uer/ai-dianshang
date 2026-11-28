"""第 7 周验收：评测集可跑通，多 Agent 不弱于单 Agent。"""

from app.eval.runner import load_cases, run


def test_dataset_shape():
    cases = load_cases()
    assert len(cases) >= 12
    for case in cases:
        assert case["id"] and case["messages"] and case["expect"]


def test_multi_agent_success_rate():
    report = run("multi", with_detail=True)
    assert report["success_rate"] >= 90.0, report["cases"]
    assert report["handoff_loss"] == 0


def test_multi_not_worse_than_single():
    single = run("single", with_detail=False)
    multi = run("multi", with_detail=False)
    assert multi["success_rate"] >= single["success_rate"]


def test_colloquial_gap_is_closed_by_b11():
    """c13 曾是单/多 Agent 的口语召回差距来源；B11 让两种分类器共用口语词表后消失。

    拆 Agent 的价值改为由 trace 里的真实分工（理解层 / 核单 / 表达层）说明，
    不再靠「单 Agent 故意漏判」的人造差距——那本来就是模拟出来的边界，不是产品目标。
    """
    cases = [c for c in load_cases() if c["id"] == "c13"]
    single = run("single", cases=cases, with_detail=False)
    multi = run("multi", cases=cases, with_detail=False)
    assert single["success_rate"] == 100.0
    assert multi["success_rate"] >= single["success_rate"]
