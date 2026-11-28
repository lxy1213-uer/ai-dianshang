"""第 3 周验收：政策条款检索、引用与无命中转人工。"""

from app.domain.rag import PolicyIndex, parse_policy


def test_policy_is_split_into_clauses(settings):
    clauses = parse_policy(settings.policy_path)
    assert len(clauses) == 8
    assert clauses[0].clause_id == "R-01"
    assert "七天无理由" in clauses[0].title


def test_retrieve_return_window_with_citation(settings):
    index = PolicyIndex.from_file(settings.policy_path)
    answer = index.answer("退货政策是几天？")
    assert answer["need_human"] is False
    assert answer["citations"][0]["clause_id"] == "R-01"
    assert "R-01" in answer["text"]


def test_retrieve_shipping_fee(settings):
    index = PolicyIndex.from_file(settings.policy_path)
    hits = index.search("退货运费谁承担")
    assert hits[0][0].clause_id == "R-05"


def test_retrieve_quality_clause(settings):
    index = PolicyIndex.from_file(settings.policy_path)
    hits = index.search("耳机有破损能退吗")
    assert hits[0][0].clause_id == "R-02"


def test_no_hit_goes_human(settings):
    index = PolicyIndex.from_file(settings.policy_path)
    answer = index.answer("今天天气怎么样")
    assert answer["need_human"] is True
    assert answer["citations"] == []


def test_snippet_is_trimmed(settings):
    clause = parse_policy(settings.policy_path)[0]
    assert len(clause.snippet(30)) <= 31
