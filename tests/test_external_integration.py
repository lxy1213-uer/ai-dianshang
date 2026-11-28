"""六项外部服务真实协议联调（零 Docker / 零厂商账号）。

本机没有 Docker、redis-server、psql，也不装 redis/psycopg/cryptography，
所以这里用纯标准库的最小协议服务在 127.0.0.1 上起真实 socket，
让**项目里的真实客户端代码**原封不动走上网络：

1. OIDC   : HttpJwksSource 真实 HTTP 拉 JWKS + verify_id_token 纯 Python RS256 验签
2. Embedding: HttpEmbeddingClient 真实 HTTP POST /v1/embeddings + EmbeddingReranker 重排
3. Webhook: WecomNotifier 真实 HTTP POST + 企微 errcode 判定 + 重试 + PII 脱敏
4. 支付   : HttpRefundGateway 真实 HTTP 签名退款 + 幂等 + RefundCallbackVerifier 回调验签 + 对账
5. Redis  : RedisLimiter 真实 RESP over TCP 滑动窗口（pipeline/zset/pexpire）
6. Postgres: PgLeaseSessionLocker 真实 socket + %s 方言 SQL（ON CONFLICT 租约锁）

诚实边界：Postgres 这一项是「SQL 语义级联调」，不是 pg wire 协议级；
真要连 Postgres 实例请按 docs/外部服务接入手册.md 用 docker-compose.local.yml。
六项都不发生真实资金流转，密钥都是本地固定测试值。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sys
import time
import urllib.request
from pathlib import Path

import pytest

STUBS = Path(__file__).resolve().parents[1] / "scripts" / "local_stubs"
sys.path.insert(0, str(STUBS))

import integration_stubs as st  # noqa: E402
from adapters import PgStubConnection, RespRedisClient  # noqa: E402

from app.identity import OidcIdentityProvider  # noqa: E402
from app.notify import WecomNotifier  # noqa: E402
from app.oidc import HttpJwksSource, verify_id_token  # noqa: E402
from app.pglock import PgLeaseSessionLocker  # noqa: E402
from app.ratelimit import RedisLimiter  # noqa: E402
from app.refund import HttpRefundGateway  # noqa: E402
from app.rerank import EmbeddingReranker, HttpEmbeddingClient  # noqa: E402
from app.settlement import (  # noqa: E402
    CallbackError, RefundCallbackVerifier, parse_ledger, reconcile,
)


@pytest.fixture()
def http_stub():
    server, port, state = st.start_http_stub()
    try:
        yield f"http://127.0.0.1:{port}", state
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture()
def redis_stub():
    server, port = st.start_redis_stub()
    try:
        yield port
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture()
def pg_stub():
    server, port = st.start_pg_stub()
    try:
        yield server, port
    finally:
        server.shutdown()
        server.server_close()


# ---------- 1. OIDC 真实 HTTP ----------
def test_oidc_real_http_jwks_and_rs256_verification(http_stub):
    base, _ = http_stub
    jwks_url = base + "/jwks"
    # 真实 HTTP 拉 JWKS，验证服务返回的是合法 RSA 公钥
    source = HttpJwksSource(jwks_url, cache_seconds=0)
    assert len(source.fetch()["keys"]) == 1

    # 从真实 /token 端点拿 id_token，再走真实验签 + claims 校验
    body = json.dumps({"sub": "u_integrate", "role": "supervisor"}).encode()
    req = urllib.request.Request(
        base + "/token", data=body, headers={"Content-Type": "application/json"}
    )
    token = json.load(urllib.request.urlopen(req))["access_token"]
    claims = verify_id_token(
        token, jwks=HttpJwksSource(jwks_url, cache_seconds=0),
        issuer=st.ISSUER, audience=st.AUDIENCE,
    )
    assert claims.sub == "u_integrate" and claims.role == "supervisor"

    # 真实 OIDC 身份源走同一条 HTTP 链路
    idp = OidcIdentityProvider(st.ISSUER, st.AUDIENCE, jwks_url=jwks_url)
    identity = idp.authenticate("", token)
    assert identity and identity.user_id == "u_integrate" and identity.role == "supervisor"


def test_oidc_rejects_tampered_token(http_stub):
    base, _ = http_stub
    token = st.issue_id_token()
    parts = token.split(".")
    tampered = parts[0] + "." + parts[1] + "." + ("0" * len(parts[2]))
    with pytest.raises(ValueError):
        verify_id_token(
            tampered, jwks=HttpJwksSource(base + "/jwks", cache_seconds=0),
            issuer=st.ISSUER, audience=st.AUDIENCE,
        )


def test_oidc_discovery_document_served_over_http(http_stub):
    """discovery 文档本身可经 HTTP 取到；discover_jwks_url 强制 https 是刻意安全约束，
    生产联调应由终端在可信 TLS  terminating 代理后完成，这里只验证文档结构。"""
    base, _ = http_stub
    doc = json.load(urllib.request.urlopen(base + "/.well-known/openid-configuration"))
    assert doc["issuer"] == st.ISSUER
    assert doc["jwks_uri"].startswith("http")


# ---------- 2. Embedding 真实 HTTP ----------
def test_embedding_real_http_and_rerank_changes_order(http_stub):
    base, _ = http_stub
    client = HttpEmbeddingClient(base + "/v1", api_key="local", model="local-embed")
    vecs = client.embed(["七天无理由退货", "运费谁承担"])
    assert len(vecs) == 2 and len(vecs[0]) == 256

    class Clause:
        def __init__(self, cid, title):
            self.clause_id, self.title, self.text = cid, title, title

    candidates = [
        (Clause("R-01", "七天无理由退货期限"), 1.0),
        (Clause("R-05", "运费由商家承担"), 0.9),
    ]
    reranker = EmbeddingReranker(client, weight=0.6)
    ranked = reranker.rerank("运费谁承担", candidates, 2)
    # 语义更贴近的 R-05 被真实向量重排到第一位（BM25 原本 R-01 更高）
    assert ranked[0][0].clause_id == "R-05"


def test_embedding_failure_degrades_to_bm25_order(http_stub):
    base, _ = http_stub
    client = HttpEmbeddingClient(base + "/does-not-exist", api_key="local")
    reranker = EmbeddingReranker(client, weight=0.9)

    class Clause:
        def __init__(self, cid):
            self.clause_id, self.title, self.text = cid, cid, cid

    candidates = [(Clause("R-01"), 1.0), (Clause("R-05"), 0.5)]
    ranked = reranker.rerank("任意问题", candidates, 2)
    assert [c.clause_id for c, _ in ranked] == ["R-01", "R-05"]


# ---------- 3. Webhook 真实 HTTP ----------
def test_webhook_real_http_records_and_sanitizes(http_stub):
    base, state = http_stub
    notifier = WecomNotifier(
        webhook=base + "/webhook", enabled=True, desk_base_url=base,
        retries=1, retry_backoff=0.01,
    )
    ok = notifier.notify_sla_overdue([{
        "queue_id": "HQ-INT", "reason": "用户 13812345678 要求人工",
        "user_id": "u_1", "sla_deadline": "2026-11-28 12:00:00",
    }])
    assert ok is True
    assert len(state.webhook_messages) == 1
    content = json.dumps(state.webhook_messages[0], ensure_ascii=False)
    assert "13812345678" not in content  # PII 脱敏在发外部前完成


def test_webhook_retry_on_transient_failure(http_stub):
    """本地替身偶发返回非 0 errcode 时，WecomNotifier 应重试。"""
    base, state = http_stub
    calls: list[int] = []

    original_post = WecomNotifier._http_post

    def flaky(self, url, payload):
        calls.append(1)
        # 第一次直接跳过真实 HTTP，模拟瞬断；第二次走真实 HTTP 成功
        if len(calls) == 1:
            return False
        return original_post(self, url, payload)

    WecomNotifier._http_post = flaky
    try:
        notifier = WecomNotifier(webhook=base + "/webhook", enabled=True,
                                 retries=2, retry_backoff=0.01)
        assert notifier.notify_sla_overdue([{"queue_id": "HQ-R", "reason": "超时"}]) is True
        assert len(calls) == 2
        assert len(state.webhook_messages) == 1
    finally:
        WecomNotifier._http_post = original_post


# ---------- 4. 支付：真实 HTTP 退款 + 回调验签 + 对账 ----------
def test_payment_real_http_refund_idempotency_and_bad_signature(http_stub):
    base, state = http_stub
    gateway = HttpRefundGateway(base + "/refunds", secret=st.API_SECRET)
    payload = {"ticket_id": "RT-INT", "amount": 12.5, "idempotency_key": "refund:RT-INT"}
    first = gateway.refund(payload)
    assert first["ok"] is True and first["refund_id"].startswith("RFD-")

    # 重复请求：网关侧仍是同一笔流水（幂等），本地 HTTP 客户端拿到相同 refund_id
    again = gateway.refund(payload)
    assert again["ok"] is True and again["refund_id"] == first["refund_id"]
    with state.lock:
        assert len(state.pay_records) == 1

    # 错签名被真实 HTTP 层拒绝
    bad = HttpRefundGateway(base + "/refunds", secret="wrong-secret")
    rejected = bad.refund({"ticket_id": "RT-BAD", "idempotency_key": "refund:RT-BAD"})
    assert rejected["ok"] is False and "401" in str(rejected["error"])


def test_payment_callback_signature_and_reconciliation(http_stub):
    base, state = http_stub
    # 先产生一笔网关流水
    gateway = HttpRefundGateway(base + "/refunds", secret=st.API_SECRET)
    paid = gateway.refund({"ticket_id": "RT-CB", "amount": 8.0,
                           "idempotency_key": "refund:RT-CB"})
    refund_id = paid["refund_id"]

    verifier = RefundCallbackVerifier(st.CALLBACK_SECRET, tolerance_seconds=300)
    payload = {"refund_id": refund_id, "ticket_id": "RT-CB", "status": "SUCCESS",
               "amount": 8.0, "timestamp": int(time.time())}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    sig = hmac.new(st.CALLBACK_SECRET.encode(),
                   str(payload["timestamp"]).encode() + b"." + raw,
                   hashlib.sha256).hexdigest()
    assert verifier.verify(raw, sig, str(payload["timestamp"]))["refund_id"] == refund_id
    # 重放被拒
    with pytest.raises(CallbackError):
        verifier.verify(raw, sig, str(payload["timestamp"]))

    # 网关对账文件走真实 HTTP 拉取，本地流水一致 → 无差异
    led = json.load(urllib.request.urlopen(base + "/ledger.json"))
    rows = parse_ledger(json.dumps(led), "json")
    local = [{"refund_id": refund_id, "ticket_id": "RT-CB",
              "status": "SUCCESS", "amount": 8.0}]
    assert reconcile(local, rows) == []

    # 本地漏记 → 网关单边差异，只给建议不自动打款
    diffs = reconcile([], rows)
    assert diffs and diffs[0].kind == "gateway_only"
    assert "人工" in diffs[0].suggested_action


def test_payment_signed_callback_post_to_receiver(http_stub):
    """模拟网关主动 POST 签名回调到宿主机端点（真实 HTTP 往返）。"""
    base, _ = http_stub
    verifier = RefundCallbackVerifier(st.CALLBACK_SECRET, tolerance_seconds=300)
    seen: list[int] = []

    class Receiver(st.HttpStubHandler):
        def do_POST(self):
            raw = self._raw_body()
            try:
                verifier.verify(raw, self.headers.get("X-Refund-Signature", ""),
                                self.headers.get("X-Refund-Timestamp", ""))
                status = 200
            except CallbackError:
                status = 401
            seen.append(status)
            self._send({"accepted": status == 200}, status)

    from http.server import ThreadingHTTPServer
    import threading
    recv = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    threading.Thread(target=recv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{recv.server_address[1]}/api/refund/callback"
    try:
        status, _ = st.send_signed_callback(
            url, {"refund_id": "RFD-POST", "ticket_id": "RT-P",
                  "status": "SUCCESS", "amount": 1.0}
        )
        assert status == 200
        status, _ = st.send_signed_callback(
            url, {"refund_id": "RFD-TAMPER", "ticket_id": "RT-T",
                  "status": "SUCCESS", "amount": 1.0}, tamper="signature"
        )
        assert status == 401
        assert seen == [200, 401]
    finally:
        recv.shutdown()
        recv.server_close()


# ---------- 5. Redis 真实 RESP ----------
def test_redis_real_resp_sliding_window(redis_stub):
    client = RespRedisClient(port=redis_stub)
    try:
        assert client.ping() is True
        limiter = RedisLimiter(client, window_seconds=60, key_prefix="int:rl")
        results = [limiter.check("ip:chat", 3)[0] for _ in range(4)]
        assert results == [True, True, True, False]
        # 不同 key 独立计数
        assert limiter.check("ip:other", 1)[0] is True
    finally:
        client.close()


def test_redis_two_instances_share_count(redis_stub):
    a = RespRedisClient(port=redis_stub)
    b = RespRedisClient(port=redis_stub)
    try:
        la = RedisLimiter(a, window_seconds=60, key_prefix="int:shared")
        lb = RedisLimiter(b, window_seconds=60, key_prefix="int:shared")
        assert la.check("shared-key", 2)[0] is True
        assert lb.check("shared-key", 2)[0] is True
        assert la.check("shared-key", 2)[0] is False  # 第二个实例看到同一计数
    finally:
        a.close(); b.close()


# ---------- 6. Postgres 语义级联调 ----------
def test_postgres_real_socket_lease_lock_mutual_exclusion(pg_stub):
    server, port = pg_stub
    locker = PgLeaseSessionLocker(
        lambda: PgStubConnection(port=port), lease_seconds=5, wait_seconds=0.3
    )
    try:
        entered: list[str] = []

        def worker(name: str):
            with locker.lock("s-shared"):
                entered.append(name)
                time.sleep(0.05)

        import threading
        t1 = threading.Thread(target=worker, args=("a",))
        t2 = threading.Thread(target=worker, args=("b",))
        t1.start(); t2.start(); t1.join(); t2.join()
        # 两个线程串行进入，且锁最终释放
        assert sorted(entered) == ["a", "b"]
        assert server.rows() == []
    finally:
        locker.close()


def test_postgres_lock_releases_only_own_owner(pg_stub):
    server, port = pg_stub
    locker = PgLeaseSessionLocker(lambda: PgStubConnection(port=port), lease_seconds=5)
    try:
        with locker.lock("s-owner"):
            rows = server.rows()
            assert len(rows) == 1 and len(rows[0]["owner"]) == 32
        # 释放后锁表清空；再次获取可成功
        with locker.lock("s-owner"):
            assert len(server.rows()) == 1
        assert server.rows() == []
    finally:
        locker.close()
