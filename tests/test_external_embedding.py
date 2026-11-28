"""真实 embedding 重排：注入假客户端，零网络，验证重排/缓存/降级。"""

import pytest

from app.config import Settings
from app.rerank import (
    EmbeddingReranker,
    HttpEmbeddingClient,
    _cosine_dense,
    build_reranker,
)
from app.domain.rag import Clause, PolicyIndex


class _Clause:
    def __init__(self, cid, title, text):
        self.clause_id, self.title, self.text = cid, title, text


DOCS = [
    _Clause("R-A", "退款时效", "质检通过后 1-3 个工作日发起退款"),
    _Clause("R-B", "运费承担", "质量问题运费由商家承担，无理由运费由消费者承担"),
    _Clause("R-C", "七天无理由", "签收次日起七天内可无理由退货，商品需完好"),
]

# 假 embedding：每个条款/查询映射成 3 维向量，语义由关键词决定
VECTORS = {
    "运费承担 质量问题运费由商家承担，无理由运费由消费者承担": [1.0, 0.0, 0.0],
    "退款时效 质检通过后 1-3 个工作日发起退款": [0.0, 1.0, 0.0],
    "七天无理由 签收次日起七天内可无理由退货，商品需完好": [0.0, 0.0, 1.0],
    "路费谁出": [1.0, 0.0, 0.0],
    "无理由": [0.0, 0.0, 1.0],
}


class FakeEmbeddingClient:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("service down")
        out = []
        for text in texts:
            out.append(VECTORS.get(text, [0.2, 0.2, 0.2]))
        return out


def _candidates():
    # 检索链路传入的候选池已按 BM25 分数降序排列
    return [(DOCS[0], 6.0), (DOCS[1], 5.0), (DOCS[2], 4.0)]


def test_cosine_dense():
    assert _cosine_dense([1, 0], [1, 0]) == pytest.approx(1.0)
    assert _cosine_dense([1, 0], [0, 1]) == pytest.approx(0.0)
    assert _cosine_dense([], [1]) == 0.0


def test_embedding_reranker_puts_semantic_match_first():
    client = FakeEmbeddingClient()
    ranker = EmbeddingReranker(client, weight=0.8)
    result = ranker.rerank("路费谁出", _candidates(), top_k=3)
    assert result[0][0].clause_id == "R-B"   # BM25 第 2，语义第 1
    assert [r[0].clause_id for r in result] == ["R-B", "R-A", "R-C"]


def test_embedding_reranker_caches_doc_vectors():
    client = FakeEmbeddingClient()
    ranker = EmbeddingReranker(client)
    ranker.rerank("路费谁出", _candidates(), 3)
    first_calls = len(client.calls)
    ranker.rerank("无理由", _candidates(), 3)
    # 第二轮只查 query，不重复编码条款
    assert len(client.calls) == first_calls + 1
    assert all(len(texts) == 1 for texts in client.calls[1:])


def test_embedding_reranker_falls_back_to_bm25_on_service_failure():
    ranker = EmbeddingReranker(FakeEmbeddingClient(fail=True))
    result = ranker.rerank("路费谁出", _candidates(), 2)
    assert [r[0].clause_id for r in result] == ["R-A", "R-B"]  # 原 BM25 顺序


def test_embedding_reranker_handles_empty_candidates():
    assert EmbeddingReranker(FakeEmbeddingClient()).rerank("x", [], 3) == []


def test_build_reranker_modes():
    assert build_reranker(Settings.load(rerank_mode="off")) is None
    assert build_reranker(Settings.load(rerank_mode="weird")) is None
    from app.rerank import HashingVectorReranker
    assert isinstance(build_reranker(Settings.load(rerank_mode="hybrid")), HashingVectorReranker)
    # embedding 缺 URL 时降级 hashing，不静默关闭
    assert isinstance(build_reranker(Settings.load(rerank_mode="embedding")), HashingVectorReranker)
    real = build_reranker(Settings.load(
        rerank_mode="embedding", embedding_base_url="https://api.example.com/v1",
        embedding_api_key="k", embedding_model="m",
    ))
    assert isinstance(real, EmbeddingReranker)
    assert isinstance(real.client, HttpEmbeddingClient)


def test_http_embedding_client_parses_openai_shape(monkeypatch):
    import json as _json
    body = _json.dumps({"data": [
        {"index": 1, "embedding": [0.0, 1.0]},
        {"index": 0, "embedding": [1.0, 0.0]},
    ]}).encode()
    seen = {}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return body

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["auth"] = req.headers.get("Authorization")
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = HttpEmbeddingClient("https://api.example.com/v1", "sk-123", "m")
    vectors = client.embed(["a", "b"])
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]   # 按 index 排序
    assert seen["url"] == "https://api.example.com/v1/embeddings"
    assert seen["auth"] == "Bearer sk-123"


def test_http_embedding_client_raises_on_bad_response(monkeypatch):
    body = b'{"data": []}'

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return body

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: _Resp())
    client = HttpEmbeddingClient("https://api.example.com/v1")
    with pytest.raises(RuntimeError):
        client.embed(["a"])
