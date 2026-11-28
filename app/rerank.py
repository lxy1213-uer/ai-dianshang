"""可插拔重排层：BM25 召回候选池 → 向量相似度重排。

设计取舍（课程项目要零配置可跑）：
- 不引 faiss / transformers / 向量库，用「字符二元组 → signed feature hashing 向量 →
  cosine 相似度」的纯 stdlib 实现。它是**确定性、可离线复现的真向量**，不是空壳：
  不同问句得到不同排序，可与 BM25 融合后改变 Top-K 顺序。
- 默认不注入重排器（NoOp），`PolicyIndex` 行为与课程基线逐字节一致；
  只有 `AGENT_RERANK=hybrid` 才启用，重排后的 score 是融合分而非原始 BM25 分。
- 接口是 Protocol，调用方可注入任意自定义重排器（含真实 embedding 服务），
  检索链路「切分 / 召回 / 阈值 / 重排 / 引用」分层保持不变。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any, Protocol

DIM = 256

ClauseLike = Any  # 只需 .title / .text / .clause_id，duck typing 便于注入假条款测试


def _tokens(text: str) -> list[str]:
    """中文友好的字符二元组 + 单字；与 BM25 侧的分词口径一致。"""
    s = re.sub(r"\s+", "", (text or "").lower())
    if not s:
        return []
    if len(s) == 1:
        return [s]
    return [s[i:i + 2] for i in range(len(s) - 1)] + list(s)


def _vector(text: str, dim: int = DIM) -> dict[int, float]:
    """signed feature hashing：把稀疏 token 投影到定长向量并 L2 归一化。

    blake2b 而非内置 hash()：后者进程间随机化，会让「可复现」失效。
    """
    vec: dict[int, float] = defaultdict(float)
    for token in _tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        n = int.from_bytes(digest, "big")
        vec[n % dim] += 1.0 if (n >> 63) & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def _cosine(a: dict[int, float], b: dict[int, float]) -> float:
    """两个已归一化稀疏向量的点积，范围 [-1, 1]。"""
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[tuple[ClauseLike, float]], top_k: int
    ) -> list[tuple[ClauseLike, float]]:
        """输入「BM25 已排序的候选池」，输出重排后的 (条款, 分数)。"""
        ...


class NoOpReranker:
    """默认实现：保持 BM25 排序与分数原样（零配置基线）。"""

    def rerank(self, query, candidates, top_k):
        return list(candidates[:top_k])


class HashingVectorReranker:
    """BM25 归一化分 + hashing 向量 cosine 的线性融合重排。

    负相关按 0 处理（不做倒扣），避免把明显不相关的条款因随机碰撞抬上来。
    同分时按原 BM25 分降序，保证结果稳定可复现。
    """

    def __init__(self, weight: float = 0.4, dim: int = DIM):
        self.weight = min(1.0, max(0.0, float(weight)))
        self.dim = dim

    def rerank(self, query, candidates, top_k):
        if not candidates:
            return []
        qvec = _vector(query, self.dim)
        rows: list[tuple[ClauseLike, float, float]] = []
        for clause, bm25 in candidates:
            dvec = _vector(f"{clause.title} {clause.title} {clause.text}", self.dim)
            rows.append((clause, bm25, max(0.0, _cosine(qvec, dvec))))
        # 两侧都做**池内归一化**到 [0, 1]：否则 BM25 原始分（8~14）与 cosine
        # （0~1）量级差一个数量级，权重再大也压不动排序，重排会退化成空壳。
        max_bm25 = max(row[1] for row in rows) or 1.0
        max_sim = max(row[2] for row in rows) or 1.0
        scored: list[tuple[ClauseLike, float, float]] = []
        for clause, bm25, sim in rows:
            fused = ((1.0 - self.weight) * (bm25 / max_bm25)
                      + self.weight * (sim / max_sim))
            scored.append((clause, round(fused, 3), bm25))
        # 同分按原 BM25 降序，保证稳定可复现
        scored.sort(key=lambda row: (row[1], row[2]), reverse=True)
        return [(clause, fused) for clause, fused, _ in scored[:top_k]]


# --- 真实 embedding -------------------------------------------------------
# 与其余可插拔组件同一纪律：协议 + 注入式客户端 + 零配置默认关闭。
# HTTP 客户端用标准库 urllib 延迟构造，不引入硬依赖；测试注入假客户端。


class EmbeddingClient(Protocol):
    """把一批文本编码成等长向量；顺序与输入一致。"""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HttpEmbeddingClient:
    """OpenAI 兼容 `/v1/embeddings`：POST JSON，取 data[].embedding。

    失败时向上抛异常，由 `EmbeddingReranker` 降级为 BM25 原序——
    检索层的可用性不能被外部 embedding 服务拖垮。
    """

    def __init__(self, base_url: str, api_key: str = "", model: str = "", timeout: float = 5.0):
        self.url = (base_url or "").rstrip("/") + "/embeddings"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        body = json.dumps({"input": texts, "model": self.model}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, OSError) as exc:
            raise RuntimeError(f"embedding 服务不可用：{type(exc).__name__}") from exc
        rows = sorted(data.get("data") or [], key=lambda row: row.get("index", 0))
        vectors = [[float(x) for x in row.get("embedding") or []] for row in rows]
        if len(vectors) != len(texts) or any(not v for v in vectors):
            raise RuntimeError("embedding 服务返回维度不合法")
        return vectors


def _cosine_dense(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class EmbeddingReranker:
    """真实语义重排：条款向量预计算并缓存，查询向量每次算一次。

    - 客户端可注入（`client=`），测试用假实现零网络；
    - 任一 embedding 调用失败 → 返回 BM25 原序 Top-K（可用性优先）；
    - 与 `HashingVectorReranker` 相同的融合公式与稳定同分排序。
    """

    def __init__(self, client: EmbeddingClient, weight: float = 0.6):
        self.client = client
        self.weight = min(1.0, max(0.0, float(weight)))
        self._doc_vectors: dict[str, list[float]] = {}

    def _ensure_doc_vectors(self, candidates) -> None:
        missing = [clause for clause, _ in candidates if clause.clause_id not in self._doc_vectors]
        if not missing:
            return
        texts = [f"{c.title} {c.text}" for c in missing]
        vectors = self.client.embed(texts)
        for clause, vector in zip(missing, vectors):
            self._doc_vectors[clause.clause_id] = vector

    def rerank(self, query, candidates, top_k):
        if not candidates:
            return []
        try:
            self._ensure_doc_vectors(candidates)
            qvec = self.client.embed([str(query)])[0]
        except Exception:
            # embedding 服务异常：绝不阻断检索，退回 BM25 原序
            return [(clause, round(score, 3)) for clause, score in candidates[:top_k]]
        rows = [
            (clause, score, max(0.0, _cosine_dense(qvec, self._doc_vectors.get(clause.clause_id, []))))
            for clause, score in candidates
        ]
        max_bm25 = max(row[1] for row in rows) or 1.0
        max_sim = max(row[2] for row in rows) or 1.0
        scored = [
            (clause, round((1 - self.weight) * (bm25 / max_bm25) + self.weight * (sim / max_sim), 3), bm25)
            for clause, bm25, sim in rows
        ]
        scored.sort(key=lambda row: (row[1], row[2]), reverse=True)
        return [(clause, fused) for clause, fused, _ in scored[:top_k]]


def build_reranker(settings):
    """按配置构造重排器；未知模式一律降级为 off（不破坏基线）。"""
    mode = (getattr(settings, "rerank_mode", "off") or "off").lower()
    if mode in ("", "off", "none", "bm25"):
        return None
    if mode == "hybrid":
        return HashingVectorReranker(getattr(settings, "rerank_weight", 0.4))
    if mode == "embedding":
        base_url = getattr(settings, "embedding_base_url", "") or ""
        if not base_url:
            # 缺 URL 时降级到 hashing，而不是静默关闭重排
            return HashingVectorReranker(getattr(settings, "rerank_weight", 0.4))
        client = HttpEmbeddingClient(
            base_url,
            getattr(settings, "embedding_api_key", "") or "",
            getattr(settings, "embedding_model", "") or "text-embedding-3-small",
            getattr(settings, "embedding_timeout", 5.0),
        )
        return EmbeddingReranker(client, getattr(settings, "rerank_weight", 0.6))
    return None
