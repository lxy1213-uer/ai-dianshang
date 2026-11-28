"""轻量 RAG：政策按条款切分 → 建立检索索引 → Top-K 命中并引用条款编号。

真实项目里可以换成向量库 + Embeddings，这里实现「字符二元组 + BM25 + 问句同义扩展」，
零依赖、可离线、可复现，检索链路（切分 / 索引 / Top-K / 阈值 / 引用 / 无命中转人工）与线上一致。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# 用户口语 → 政策用语，解决"问法和条款写法不一致"导致的召回失败
QUERY_EXPANSIONS: list[tuple[tuple[str, ...], str]] = [
    (("几天", "多久", "期限", "时效", "什么时候", "算时间"), "期限 天内 起算 第 7 日"),
    (("破损", "坏了", "故障", "瑕疵", "漏发", "错发", "质量", "不符", "不能用"), "质量问题 破损 凭证"),
    (("不想要", "不喜欢", "不合适", "买错"), "无理由"),
    (("运费", "邮费", "谁承担", "谁出"), "运费 承担"),
    (("到账", "退钱"), "退款 到账 工作日"),
    (("工单", "进度", "重复", "建单"), "售后工单 进度 幂等"),
    (("人工", "客服", "投诉"), "转人工"),
    (("内衣", "定制", "生鲜", "虚拟"), "不支持 类目"),
]


@dataclass
class Clause:
    clause_id: str
    title: str
    text: str

    @property
    def label(self) -> str:
        return f"【{self.clause_id}】{self.title}"

    def snippet(self, limit: int = 90) -> str:
        body = re.sub(r"\s+", " ", self.text).strip()
        return body[:limit] + ("…" if len(body) > limit else "")

    def salient(self, query: str, limit: int = 70) -> str:
        """挑出条款里**回答这个问句**的那一句，而不是永远截开头。

        条款是分句写的（适用 / 起算方式 / 凭证要求），固定截前 60 字永远只返回
        「适用：…」，用户问「怎么算完好」也只能听到适用范围。这里按与问句的
        字符二元组重合度挑句，重合度相同则取靠前的句子（保持稳定）。
        """
        sentences = [s.strip() for s in re.split(r"[。；\n]+", self.text) if s.strip()]
        if not sentences:
            return self.snippet(limit)
        query_grams = set(_grams(query))
        best, best_score = sentences[0], -1
        for sentence in sentences:
            score = len(query_grams & set(_grams(sentence)))
            if score > best_score:
                best, best_score = sentence, score
        best = re.sub(r"\s+", " ", best)
        return best[:limit] + ("…" if len(best) > limit else "")


# --- 引用完整性收口（B13）-------------------------------------------------
# holdout4 p04：答复里写了「运费由商家承担」（这是 R-05 的结论），引用却只有 R-02
# （R-02 的正文里顺带提了一句运费，检索时权重偏向它）。结论一旦落到话术里，
# 就必须把真正承载它的条款一起列出，否则用户点开引用找不到依据。
# 这张表只做**加法**：命中结论短语就补上应引条款，绝不删已引条款、不改话术。
# 语素取条款里的结论性表述，不用整句，避免换个措辞就漏。
# 刻意只收两组「结论性承诺」：运费谁承担（R-05）与能否重复申请（R-06）。
# 这两类结论一旦写进话术就必须有条款支撑，且措辞独一，不会误伤别的答复。
# 其余条款（未发货 / 转人工 / 七天无理由…）由状态机按分支精确引用，
# 通用词（"全额退款" "签收" "转人工"）不进表，避免给撤单答复补出 R-01/R-08。
_CLAUSE_FACT_MARKERS: dict[str, tuple[str, ...]] = {
    "R-05": ("运费由商家承担", "运费由消费者承担", "运费需自理", "运费谁承担",
             "运费由谁承担", "无需消费者垫付"),
    "R-06": ("不能重复申请", "重复申请退款", "不能再次申请"),
}


def complete_citations(text: str, cited_ids, clauses=None) -> list[str]:
    """按答复正文里的事实短语补齐应引条款（只加不减）。

    - `text`：最终要展示给用户的话术；
    - `cited_ids`：已有的条款号（list / tuple / set / dict 均可）；
    - `clauses`：可选的条款清单，用于过滤不存在的条款号；默认不过滤。

    返回去重后的条款号列表。原顺序保持不变，新补的按条款表顺序追加。
    """
    items = list(cited_ids or [])
    if items and isinstance(items[0], dict):
        ids = [c.get("clause_id") for c in items]
    else:
        ids = items
    result = [str(c) for c in ids if c]
    known = {c.clause_id for c in clauses} if clauses is not None else None
    body = text or ""
    for clause_id, markers in _CLAUSE_FACT_MARKERS.items():
        if known is not None and clause_id not in known:
            continue
        if clause_id in result:
            continue
        if any(marker in body for marker in markers):
            result.append(clause_id)
    return result


_CLAUSE_HEAD = re.compile(r"^###\s+(R-\d+)\s*(.*)$", re.MULTILINE)


def parse_policy(path: Path) -> list[Clause]:
    text = Path(path).read_text(encoding="utf-8")
    matches = list(_CLAUSE_HEAD.finditer(text))
    clauses: list[Clause] = []
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        clauses.append(
            Clause(
                clause_id=match.group(1),
                title=match.group(2).strip(),
                text=text[match.end():end].strip(),
            )
        )
    return clauses


_NORMALIZE = re.compile(r"[^\u4e00-\u9fa5a-zA-Z0-9]+")


def _normalize(text: str) -> str:
    return _NORMALIZE.sub("", text or "").lower()


def _grams(text: str) -> Counter:
    """中文用字符二元组，英文/数字保留整词，兼顾语义与实现成本。"""
    norm = _normalize(text)
    grams: Counter = Counter()
    for i in range(len(norm) - 1):
        grams[norm[i:i + 2]] += 1
    for token in re.findall(r"[a-zA-Z0-9]{2,}", (text or "").lower()):
        grams[token] += 2
    return grams


def expand_query(query: str) -> str:
    extra = [target for triggers, target in QUERY_EXPANSIONS if any(t in query for t in triggers)]
    return query + " " + " ".join(extra)


class PolicyIndex:
    """政策索引：BM25 打分，标题重复一次以提高权重。"""

    K1 = 1.2
    B = 0.55

    def __init__(self, clauses: list[Clause], reranker=None):
        if not clauses:
            raise ValueError("政策库为空，检查 policies.md 的条款格式（### R-01 标题）")
        self.clauses = clauses
        # 可插拔重排器：None（默认）= 纯 BM25，行为与课程基线完全一致；
        # 注入后走「BM25 召回宽候选池 → 重排 → Top-K」两阶段链路。
        self.reranker = reranker
        self._docs: list[Counter] = []
        self._lengths: list[int] = []
        self._df: Counter = Counter()
        for clause in clauses:
            grams = _grams(f"{clause.title} {clause.title} {clause.text}")
            self._docs.append(grams)
            self._lengths.append(sum(grams.values()) or 1)
            for gram in grams:
                self._df[gram] += 1
        self._avgdl = sum(self._lengths) / len(self._lengths)
        self._total = len(clauses)

    @classmethod
    def from_file(cls, path: Path, reranker=None) -> "PolicyIndex":
        return cls(parse_policy(path), reranker=reranker)

    def _idf(self, gram: str) -> float:
        df = self._df.get(gram, 0)
        return math.log((self._total - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 3, min_score: float | None = None) -> list[tuple[Clause, float]]:
        terms = _grams(expand_query(query))
        if not terms:
            return []
        raw: list[tuple[Clause, float]] = []
        for clause, grams, dl in zip(self.clauses, self._docs, self._lengths):
            score = 0.0
            for term, qf in terms.items():
                tf = grams.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.K1 * (1 - self.B + self.B * dl / self._avgdl)
                score += self._idf(term) * qf * (tf * (self.K1 + 1)) / denom
            if score > 0:
                raw.append((clause, score))
        if not raw:
            return []
        raw.sort(key=lambda item: item[1], reverse=True)
        best = raw[0][1]
        # 相对阈值：只有明显低于第一名的条款才丢弃，避免短问句被绝对阈值一刀切
        cutoff = min_score if min_score is not None else max(1.0, best * 0.35)
        if self.reranker is None:
            return [(clause, round(score, 3)) for clause, score in raw if score >= cutoff][:top_k]
        # 重排模式：先用更宽的阈值召回候选池，再交给重排器定序。
        # 注意此时返回的 score 是融合分（不是原始 BM25 分），调用方只用于展示与排序。
        pool = [(clause, score) for clause, score in raw
                if score >= (min_score if min_score is not None else max(0.0, best * 0.15))]
        return self.reranker.rerank(query, pool, top_k)

    def answer(self, query: str, top_k: int = 2) -> dict:
        """返回带条款引用的答复素材；没有命中就让调用方转人工。"""
        hits = self.search(query, top_k=top_k)
        if not hits:
            return {
                "need_human": True,
                "text": "这个问题我在现行政策里没有找到明确条款，为避免给错信息，建议转人工确认。",
                "citations": [],
            }
        citations = [
            {
                "clause_id": clause.clause_id,
                "title": clause.title,
                "score": score,
                "snippet": clause.snippet(),
            }
            for clause, score in hits
        ]
        label = "、".join(
            f"【{c.clause_id}】《{c.title}》" for c in [h[0] for h in hits[:2]]
        )
        return {
            "need_human": False,
            "text": f"按平台规则 {label}：{hits[0][0].snippet(130)}",
            "citations": citations,
        }


def build_index(path: Path, reranker=None) -> PolicyIndex:
    """索引工厂：调用方按需注入重排器；不注入 = 纯 BM25（零配置默认）。"""
    return PolicyIndex.from_file(path, reranker=reranker)
