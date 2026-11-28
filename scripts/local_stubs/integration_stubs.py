"""六项外部服务的本地最小协议实现（零第三方依赖，真实 socket）。

用途：在没有 Docker / 厂商账号的机器上，用真实网络 I/O 跑通六项客户端的联调。
与 docker-compose.local.yml 里的 Keycloak/Redis/Postgres 替身互补：那些更接近
产品，本模块更轻、可随 pytest 一键启动，验证的是「协议与客户端代码是对的」。

覆盖的服务与协议：
1. OIDC IdP  : HTTP GET  /.well-known/openid-configuration、/jwks；POST /token 签发 RS256 id_token
2. Embedding : HTTP POST /v1/embeddings（OpenAI 兼容）
3. 企微 webhook: HTTP POST /webhook → {"errcode": 0}，落 /messages 供查询
4. 支付网关  : HTTP POST /refunds（验签 + 幂等）、/__local/send-callback、GET /ledger.json
5. Redis     : RESP over TCP（PING/ZADD/ZCARD/ZREMRANGEBYSCORE/ZRANGE/PEXPIRE/INFO/MULTI/EXEC）
6. Postgres  : 轻量 length-prefix JSON over TCP，sqlite 执行 %s→? 的租约锁 SQL
               —— 注意：这是 SQL 语义级联调，不是 pg wire 协议级；真实 pg 需 docker-compose

安全边界：
- 私钥是测试专用的 1024-bit RSA，签名只在进程内用于验证验签链路，绝不用于生产；
- API 密钥/回调密钥都是本地固定值；回调签名口径与 RefundCallbackVerifier 完全一致；
- 所有服务只绑 127.0.0.1，端口 0 由系统分配，避免端口冲突。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import socket
import socketserver
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

# ---------- OIDC 测试密钥（与 tests/test_external_idp.py 同源，仅用于验签自测） ----------
N_INT = 74403846017565288313067730518049027329323646381999316218501720924448077061848964658700741608363159031126857081500578212866209590940778451974983437804601011578429314482282715767301966467533409153286899862644704633244381093096522367172133135542659096828743391286661161431484009575743722039318976809214725868003
D_INT = 23180458794065154901448295187718770160521998440997635511991213174777314150004305360351431132941071167394726153014004393705880150568820886406536946047184691043871117653296196830071116965980955147771428885471772360258107102354958655122140946284962456876764539142744329440142763238379748588965076408621732245345
E_INT = 65537
_K = (N_INT.bit_length() + 7) // 8
_DIGEST_INFO = bytes.fromhex("3031300d060960864801650304020105000420")
API_SECRET = "local-api-secret"
CALLBACK_SECRET = "local-callback-secret"

ISSUER = "local-idp"          # OIDC issuer（不含 http scheme，便于本地联调）
AUDIENCE = "ecom-agent"
KID = "local-key"


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def jwks() -> dict:
    n = _b64e(N_INT.to_bytes(_K, "big"))
    e = _b64e(E_INT.to_bytes((E_INT.bit_length() + 7) // 8, "big"))
    return {
        "keys": [
            {"kty": "RSA", "kid": KID, "alg": "RS256", "use": "sig", "n": n, "e": e}
        ]
    }


def issue_id_token(claims: dict | None = None) -> str:
    """用测试私钥签发 RS256 id_token；签名口径与 oidc.py 的验签完全对应。"""
    now = int(time.time())
    payload = {
        "iss": ISSUER, "aud": AUDIENCE, "sub": "u_local_oidc",
        "preferred_username": "local-admin", "role": "supervisor",
        "iat": now, "nbf": now - 5, "exp": now + 600,
    }
    if claims:
        payload.update(claims)
    header = {"alg": "RS256", "typ": "JWT", "kid": KID}
    h = _b64e(json.dumps(header, separators=(",", ":")).encode())
    p = _b64e(json.dumps(payload, separators=(",", ":")).encode())
    signed = f"{h}.{p}".encode("ascii")
    digest = hashlib.sha256(signed).digest()
    em = b"\x00\x01" + b"\xff" * (_K - len(_DIGEST_INFO) - len(digest) - 3) + b"\x00" + _DIGEST_INFO + digest
    sig = pow(int.from_bytes(em, "big"), D_INT, N_INT).to_bytes(_K, "big")
    return f"{h}.{p}.{_b64e(sig)}"


# ---------- embedding 确定性向量（与 rerank 的 hashing 口径一致） ----------
def _embed(text: str, dim: int = 256) -> list[float]:
    vec = [0.0] * dim
    s = "".join((text or "").lower().split())
    if not s:
        return vec
    toks = [s[i:i + 2] for i in range(len(s) - 1)] + list(s) if len(s) > 1 else [s]
    for tok in toks:
        n = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "big")
        vec[n % dim] += 1.0 if (n >> 63) & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


# =====================================================================
# HTTP stub：OIDC + Embedding + Webhook + Payment（同一端口，按 path 分发）
# =====================================================================
class _State:
    """跨请求共享的内存状态（替代 Docker 卷里的日志文件）。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.webhook_messages: list[dict] = []
        self.pay_records: dict[str, dict] = {}
        self.callbacks: list[dict] = []


class HttpStubHandler(BaseHTTPRequestHandler):
    state: _State = _State()          # 类属性：同一 server 实例共享
    issuer_host: str = "127.0.0.1"

    def log_message(self, fmt: str, *args) -> None:
        return

    # ---------- helpers ----------
    def _send(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _raw_body(self) -> bytes:
        try:
            n = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            n = 0
        return self.rfile.read(n) if n else b""

    @property
    def _origin(self) -> str:
        return f"http://{self.headers.get('Host') or '127.0.0.1'}"

    # ---------- GET ----------
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._send({"ok": True, "service": "integration-http-stub"})
        elif path == "/.well-known/openid-configuration":
            self._send({
                "issuer": ISSUER,
                "authorization_endpoint": self._origin + "/authorize",
                "token_endpoint": self._origin + "/token",
                "jwks_uri": self._origin + "/jwks",
                "response_types_supported": ["code"],
            })
        elif path == "/jwks":
            self._send(jwks())
        elif path == "/messages":
            with self.state.lock:
                rows = list(self.state.webhook_messages)
            self._send({"count": len(rows), "messages": rows[-50:]})
        elif path == "/pay/records":
            with self.state.lock:
                rows = list(self.state.pay_records.values())
            self._send({"count": len(rows), "records": rows})
        elif path == "/ledger.json":
            with self.state.lock:
                rows = list(self.state.pay_records.values())
            self._send(rows)
        elif path == "/ledger.csv":
            with self.state.lock:
                rows = list(self.state.pay_records.values())
            header = "refund_id,ticket_id,status,amount\n"
            text = header + "".join(
                f"{r['refund_id']},{r['ticket_id']},{r['status']},{r.get('amount') or ''}\n"
                for r in rows
            )
            body = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/callbacks":
            with self.state.lock:
                rows = list(self.state.callbacks)
            self._send({"count": len(rows), "items": rows})
        else:
            self._send({"detail": "not found"}, 404)

    # ---------- POST ----------
    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        raw = self._raw_body()
        if path == "/token":
            # 极简授权：任意 JSON body 都签一个 token；claims 可由 body 覆盖 sub/role
            try:
                claims = json.loads(raw.decode("utf-8")) if raw else {}
            except ValueError:
                claims = {}
            allow = {"sub", "preferred_username", "role", "aud", "iss"}
            selected = {k: v for k, v in claims.items() if k in allow}
            self._send({"access_token": issue_id_token(selected), "token_type": "Bearer",
                        "expires_in": 600})
        elif path == "/v1/embeddings":
            try:
                body = json.loads(raw.decode("utf-8"))
            except ValueError:
                self._send({"detail": "invalid json"}, 400)
                return
            texts = body.get("input") or []
            data = [{"object": "embedding", "index": i, "embedding": _embed(str(t))}
                    for i, t in enumerate(texts)]
            self._send({"object": "list", "model": body.get("model", "local-embed"), "data": data})
        elif path == "/webhook":
            try:
                body = json.loads(raw.decode("utf-8"))
            except ValueError:
                body = {"raw": raw.decode("utf-8", errors="replace")}
            with self.state.lock:
                self.state.webhook_messages.append({"at": time.time(), "body": body})
            self._send({"errcode": 0, "errmsg": "ok"})
        elif path == "/refunds":
            self._handle_refund(raw)
        else:
            self._send({"detail": "not found"}, 404)

    def _handle_refund(self, raw: bytes) -> None:
        sig = self.headers.get("X-Refund-Signature", "")
        expected = hmac.new(API_SECRET.encode(), raw, hashlib.sha256).hexdigest()
        if not sig or not hmac.compare_digest(sig, expected):
            self._send({"ok": False, "error": "bad_signature"}, 401)
            return
        try:
            body = json.loads(raw.decode("utf-8"))
        except ValueError:
            self._send({"ok": False, "error": "bad_json"}, 400)
            return
        key = str(body.get("idempotency_key") or "")
        with self.state.lock:
            if key and key in self.state.pay_records:
                self._send({"ok": True, "duplicate": True,
                            "refund_id": self.state.pay_records[key]["refund_id"],
                            "gateway": "http-local"})
                return
            digest = hashlib.blake2b(key.encode(), digest_size=8).hexdigest().upper()
            rec = {
                "refund_id": f"RFD-{digest}", "ticket_id": body.get("ticket_id"),
                "status": "SUCCESS", "amount": body.get("amount"),
                "idempotency_key": key,
            }
            if key:
                self.state.pay_records[key] = rec
        self._send({"ok": True, "duplicate": False, "refund_id": rec["refund_id"],
                    "gateway": "http-local"})


def start_http_stub() -> tuple[ThreadingHTTPServer, int, _State]:
    HttpStubHandler.state = _State()  # 每次启动重置，避免多次测试间串扰
    server = ThreadingHTTPServer(("127.0.0.1", 0), HttpStubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1], HttpStubHandler.state


# =====================================================================
# RESP stub：Redis 协议子集（真实 socket）
# =====================================================================
class _RespStore:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.zsets: dict[str, dict[str, float]] = {}
        self.expire: dict[str, float] = {}   # key -> monotonic deadline

    def _purge(self, key: str) -> None:
        exp = self.expire.get(key)
        if exp is not None and time.monotonic() > exp:
            self.zsets.pop(key, None)
            self.expire.pop(key, None)

    def execute(self, parts: list[bytes]) -> Any:
        cmd = parts[0].decode("ascii").upper() if parts else ""
        with self.lock:
            if cmd == "PING":
                return "PONG"
            if cmd == "INFO":
                return "# Server\r\nredis_mode:localstub\r\n"
            if cmd == "FLUSHALL":
                self.zsets.clear(); self.expire.clear(); return "OK"
            key = parts[1].decode() if len(parts) > 1 else ""
            self._purge(key)
            if cmd == "ZADD":
                zs = self.zsets.setdefault(key, {})
                added = 0
                for i in range(2, len(parts) - 1, 2):
                    score = float(parts[i]); member = parts[i + 1].decode()
                    if member not in zs:
                        added += 1
                    zs[member] = score
                return added
            if cmd == "ZCARD":
                return len(self.zsets.get(key, {}))
            if cmd == "ZREMRANGEBYSCORE":
                lo, hi = float(parts[2]), float(parts[3])
                zs = self.zsets.get(key, {})
                gone = [m for m, s in zs.items() if lo <= s <= hi]
                for m in gone:
                    zs.pop(m, None)
                return len(gone)
            if cmd == "PEXPIRE":
                ms = int(parts[2])
                if key in self.zsets:
                    self.expire[key] = time.monotonic() + ms / 1000.0
                    return 1
                return 0
            if cmd == "ZRANGE":
                items = sorted(self.zsets.get(key, {}).items(), key=lambda kv: (kv[1], kv[0]))
                start = int(parts[2]); stop = int(parts[3])
                withscores = len(parts) > 4 and parts[4].decode("ascii").upper() == "WITHSCORES"
                selected = items[start:] if stop == -1 else items[start:stop + 1]
                out: list[bytes] = []
                for member, score in selected:
                    out.append(member.encode())
                    if withscores:
                        out.append(repr(score).encode())
                return out
            return None


class RespStubHandler(socketserver.StreamRequestHandler):
    store: _RespStore   # 由 start_redis_stub 赋类属性（构造即 handle，实例赋值太晚）

    def _read_line(self) -> bytes:
        line = self.rfile.readline()
        return line.rstrip(b"\r\n")

    def _read_command(self) -> list[bytes] | None:
        head = self.rfile.readline()
        if not head:
            return None
        head = head.strip()
        if not head.startswith(b"*"):
            # inline command
            return head.split()
        count = int(head[1:])
        parts = []
        for _ in range(count):
            marker = self.rfile.readline().strip()
            if not marker.startswith(b"$"):
                return None
            n = int(marker[1:])
            parts.append(self.rfile.read(n + 2)[:n] if n >= 0 else b"")
        return parts

    @staticmethod
    def _encode(value: Any) -> bytes:
        if isinstance(value, list):
            return b"*" + str(len(value)).encode() + b"\r\n" + b"".join(
                b"$" + str(len(v)).encode() + b"\r\n" + v + b"\r\n" for v in value
            )
        if isinstance(value, str):
            return b"+" + value.encode() + b"\r\n"
        if value is None:
            return b"$-1\r\n"
        return b":" + str(int(value)).encode() + b"\r\n"

    def handle(self) -> None:
        while True:
            parts = self._read_command()
            if parts is None:
                return
            cmd = parts[0].decode("ascii").upper() if parts else ""
            if cmd == "MULTI":
                self.wfile.write(b"+OK\r\n"); self.wfile.flush(); continue
            if cmd == "EXEC":
                # 客户端 pipeline 逐条发命令；这里返回 OK 占位即可（计数由逐条回复给出）
                self.wfile.write(b"+OK\r\n"); self.wfile.flush(); continue
            result = self.store.execute(parts)
            self.wfile.write(self._encode(result))
            self.wfile.flush()


class RespStubServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), RespStubHandler)
        self.store = _RespStore()
        RespStubHandler.store = self.store  # 类属性：构造即 handle，必须在 accept 前就位


def start_redis_stub() -> tuple[RespStubServer, int]:
    server = RespStubServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


# =====================================================================
# PG 语义 stub：length-prefix JSON over TCP + sqlite 执行 %s 方言
# =====================================================================
class _PgSemantics:
    def __init__(self) -> None:
        import sqlite3
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()

    def execute(self, sql: str, params: list) -> dict:
        # psycopg 的 %s → sqlite 的 ?（字符串里的 %% 还原为 %）
        converted = sql.replace("%%", "\x00").replace("%s", "?").replace("\x00", "%")
        with self.lock:
            cur = self.conn.execute(converted, tuple(params))
            if converted.lstrip().upper().startswith(("SELECT", "WITH")):
                rows = [dict(r) for r in cur.fetchall()]
            else:
                rows = []
            self.conn.commit()
            return {"rowcount": cur.rowcount, "rows": rows}


class PgStubHandler(socketserver.StreamRequestHandler):
    engine: _PgSemantics   # 由 PgStubServer 赋类属性（构造即 handle，实例赋值太晚）

    def handle(self) -> None:
        while True:
            head = self.rfile.read(4)
            if len(head) < 4:
                return
            size = struct.unpack("!I", head)[0]
            raw = self.rfile.read(size)
            if not raw:
                return
            req = json.loads(raw.decode("utf-8"))
            if req.get("op") == "close":
                self.wfile.write(struct.pack("!I", 2) + b"{}")
                self.wfile.flush()
                return
            try:
                resp = self.engine.execute(req.get("sql", ""), req.get("params") or [])
                body = json.dumps(resp, ensure_ascii=False).encode("utf-8")
            except Exception as exc:  # 语义错误也回传，由客户端决定如何降级
                body = json.dumps({"error": f"{type(exc).__name__}: {exc}"},
                                  ensure_ascii=False).encode("utf-8")
            self.wfile.write(struct.pack("!I", len(body)) + body)
            self.wfile.flush()


class PgStubServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), PgStubHandler)
        self.engine = _PgSemantics()
        PgStubHandler.engine = self.engine  # 类属性：构造即 handle，必须在 accept 前就位

    def rows(self) -> list[dict]:
        with self.engine.lock:
            cur = self.engine.conn.execute("SELECT session_id, owner, expires_at FROM session_locks")
            return [dict(r) for r in cur.fetchall()]


def start_pg_stub() -> tuple[PgStubServer, int]:
    server = PgStubServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


# =====================================================================
# 回调主动投递：模拟支付网关给宿主机发签名回调（真实 HTTP）
# =====================================================================
def send_signed_callback(target_url: str, payload: dict, *, tamper: str = "",
                         secret: str = CALLBACK_SECRET) -> tuple[int, str]:
    import urllib.error
    import urllib.request

    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    if tamper == "body":
        raw = raw.replace(b"SUCCESS", b"FAILED")
    ts = int(time.time())
    if tamper == "replay":
        ts -= 3600
    sig = hmac.new(secret.encode(), f"{ts}.".encode("ascii") + raw, hashlib.sha256).hexdigest()
    if tamper == "signature":
        sig = "0" * 64
    req = urllib.request.Request(
        target_url, data=raw, method="POST",
        headers={"Content-Type": "application/json",
                 "X-Refund-Signature": sig, "X-Refund-Timestamp": str(ts)},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
