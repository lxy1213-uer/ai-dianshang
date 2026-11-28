"""第二批留出集的样本纪律：不与任何已调参集合重叠、不进 48 条全量口径。

留出集一旦被用来定位失败并据此改代码，就永久失效。这里只锁「隔离性」，
不锁分数——`run_eval.py --holdout3` 每次跑出来的成绩本身才是泛化证据。
"""

from __future__ import annotations

from pathlib import Path

from app.eval.runner import (
    HOLDOUT3_PATH,
    load_all_cases,
    load_cases,
    load_dev2,
    load_holdout2,
    load_holdout3,
    load_llm_sample,
)

_ROOT = Path(__file__).resolve().parents[1]


def _sequences(cases: list[dict]) -> set[str]:
    return {"||".join(case["messages"]) for case in cases}


def test_holdout3_size_and_structure():
    cases = load_holdout3()
    assert len(cases) == 16
    assert len({case["id"] for case in cases}) == 16
    assert all(case["split"] == "holdout3" for case in cases)
    assert all(case["messages"] and case.get("expect") for case in cases)


def test_holdout3_does_not_join_the_tuned_full_set():
    ids = {case["id"] for case in load_all_cases()}
    assert not ids & {case["id"] for case in load_holdout3()}
    assert len(load_all_cases()) == 48  # 历史口径保持不变


def test_holdout3_has_zero_overlap_with_every_existing_case_set():
    new = _sequences(load_holdout3())
    for loader in (load_cases, load_dev2, load_holdout2, load_llm_sample):
        assert not new & _sequences(loader()), loader.__name__


def test_holdout3_messages_were_not_written_into_source_as_rules():
    """新措辞不能先被写进规则/提示词里——那就是拿留出集调参。"""
    sources = "\n".join(
        (_ROOT / relative).read_text(encoding="utf-8")
        for relative in ("app/llm.py", "app/agent/advice.py", "app/slots.py")
    )
    for case in load_holdout3():
        for message in case["messages"]:
            # 纯单号 / 确认词是流程通用词，不算「整句措辞」；只查带中文的自然语言。
            if len(message) < 8 or not any("\u4e00" <= ch <= "\u9fff" for ch in message):
                continue
            assert message not in sources, message


def test_holdout3_file_is_tracked_next_to_other_splits():
    assert HOLDOUT3_PATH.name == "cases_holdout3.json"
    assert HOLDOUT3_PATH.exists()
