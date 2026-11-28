"""架构进阶 · 可插拔向量重排验收。

纪律：默认 off 必须与课程基线逐字节一致（排序 + 分数），
开启 hybrid 才允许改变候选集/排序，且必须确定性可复现。
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from app.config import DATA_DIR, Settings
from app.domain.rag import Clause, PolicyIndex, build_index
from app.rerank import (
    HashingVectorReranker,
    NoOpReranker,
    _cosine,
    _vector,
    build_reranker,
)

POLICY = DATA_DIR / "policies.md"


# ---------- 1. 默认零配置不变 ----------

def test_default_build_index_has_no_reranker():
    idx = build_index(POLICY)
    assert idx.reranker is None


def test_default_scores_and_order_unchanged():
    """off 时分数仍是原始 BM25 分（不是融合分），Top-1 与基线一致。"""
    idx = build_index(POLICY)
    hits = idx.search("运费谁承担", top_k=3)
    assert hits[0][0].clause_id == "R-05"
    assert hits[0][1] > 1.0  # BM25 原始分；融合分被归一化到 <= 1
    assert idx.search("退货政策是几天", top_k=3)[0][0].clause_id == "R-01"


def test_build_reranker_modes():
    assert build_reranker(SimpleNamespace(rerank_mode="off", rerank_weight=0.4)) is None
    assert build_reranker(SimpleNamespace(rerank_mode="", rerank_weight=0.4)) is None
    assert build_reranker(SimpleNamespace(rerank_mode="unknown", rerank_weight=0.4)) is None
    r = build_reranker(SimpleNamespace(rerank_mode="HYBRID", rerank_weight=0.7))
    assert isinstance(r, HashingVectorReranker)
    assert r.weight == 0.7


def test_settings_describe_exposes_rerank():
    s = Settings.load(rerank_mode="hybrid", rerank_weight=0.6)
    d = s.describe()
    assert d["rerank_mode"] == "hybrid"
    assert d["rerank_weight"] == 0.6


# ---------- 2. 向量基元 ----------

def test_vector_is_deterministic_and_normalized():
    a = _vector("运费由商家承担")
    b = _vector("运费由商家承担")
    assert a == b
    assert math.isclose(math.sqrt(sum(v * v for v in a.values())), 1.0, abs_tol=1e-9)
    assert math.isclose(_cosine(a, b), 1.0, abs_tol=1e-9)


def test_vector_differs_across_processes_safe_hash():
    """用 blake2b 而非内置 hash：不同文本得到不同向量，且不依赖 PYTHONHASHSEED。"""
    a = _vector("运费由商家承担")
    b = _vector("七天无理由退货期限")
    assert a != b
    assert -1.0 <= _cosine(a, b) <= 1.0


# ---------- 3. 重排真的能改变顺序（防「假重排」） ----------

def _fake_pool():
    a = Clause("R-A", "退款到账时效", "退款款项将在若干个工作日内原路退回，具体到账时间以支付渠道为准")
    b = Clause("R-B", "运费谁承担", "运费由商家承担，运费由商家承担，运费由商家承担，运费由商家承担")
    return [(a, 12.0), (b, 6.0)]


def test_weight_zero_keeps_bm25_order():
    out = HashingVectorReranker(0.0).rerank("运费由商家承担 运费谁出", _fake_pool(), 2)
    assert [c.clause_id for c, _ in out] == ["R-A", "R-B"]


def test_high_weight_flips_top1_by_vector_similarity():
    """关键守卫：若两侧分数不做池内归一化，权重再大也压不动排序 → 空壳。"""
    q = "运费由商家承担 运费谁出"
    assert [c.clause_id for c, _ in HashingVectorReranker(0.4).rerank(q, _fake_pool(), 2)] == ["R-B", "R-A"]
    assert [c.clause_id for c, _ in HashingVectorReranker(0.8).rerank(q, _fake_pool(), 2)] == ["R-B", "R-A"]


def test_rerank_is_deterministic():
    r = HashingVectorReranker(0.6)
    q, pool = "运费由商家承担", _fake_pool()
    assert r.rerank(q, pool, 2) == r.rerank(q, pool, 2)


def test_rerank_empty_candidates():
    assert HashingVectorReranker().rerank("任意问句", [], 3) == []
    assert NoOpReranker().rerank("任意问句", [], 3) == []


def test_custom_reranker_is_pluggable():
    """接口可注入任意实现（含未来真实 embedding 服务）。"""
    class Reverse:
        def rerank(self, query, candidates, top_k):
            return list(reversed(candidates))[:top_k]

    idx = PolicyIndex.from_file(POLICY, reranker=Reverse())
    hits = idx.search("质量问题要什么凭证", top_k=3)
    assert hits
    assert hits[0][0].clause_id != build_index(POLICY).search("质量问题要什么凭证", top_k=3)[0][0].clause_id


# ---------- 4. hybrid 端到端 ----------

def test_hybrid_search_is_reproducible_and_widens_recall():
    idx = build_index(POLICY, reranker=HashingVectorReranker(0.4))
    q = "定制商品能退吗"
    a = [(c.clause_id, s) for c, s in idx.search(q, top_k=3)]
    b = [(c.clause_id, s) for c, s in idx.search(q, top_k=3)]
    assert a == b
    assert a  # 至少命中一条
    assert all(0.0 <= s <= 1.0 for _, s in a)  # 融合分已归一化
    assert len(a) >= len(build_index(POLICY).search(q, top_k=3))  # 候选池更宽


def test_hybrid_no_hit_still_routes_to_human():
    idx = build_index(POLICY, reranker=HashingVectorReranker())
    out = idx.answer("今天天气怎么样")
    assert out["need_human"] is True
    assert out["citations"] == []


def test_hybrid_citations_still_carry_clause_id_and_score():
    idx = build_index(POLICY, reranker=HashingVectorReranker())
    out = idx.answer("运费谁承担")
    assert out["need_human"] is False
    assert out["citations"][0]["clause_id"] == "R-05"
    assert isinstance(out["citations"][0]["score"], float)


def test_min_score_is_respected_in_rerank_mode():
    idx = build_index(POLICY, reranker=HashingVectorReranker())
    assert idx.search("运费谁承担", top_k=3, min_score=1e9) == []


# ---------- 5. Coordinator 组装 ----------

def test_coordinator_injects_reranker_only_when_configured(tmp_path):
    from app.agent.coordinator import Coordinator

    off = Coordinator(Settings.load(
        llm_mode="rules", db_path=tmp_path / "off.db", rerank_mode="off",
    ))
    assert off.index.reranker is None
    off.close()

    on = Coordinator(Settings.load(
        llm_mode="rules", db_path=tmp_path / "on.db", rerank_mode="hybrid", rerank_weight=0.5,
    ))
    assert isinstance(on.index.reranker, HashingVectorReranker)
    on.close()
