"""把项目真实客户端接到本地最小协议 stub 上的薄适配器。

- `RespRedisClient`：说 RESP 协议，满足 RedisLimiter 需要的 pipeline/z* 接口；
- `PgStubConnection`：说 length-prefix JSON，满足 PgLeaseSessionLocker 的 execute/commit/close 契约。

适配器只做「协议翻译」，不含任何业务判定：限流的滑动窗口逻辑、租约锁的
ON CONFLICT / owner 校验全部由 app.ratelimit / app.pglock 的真实代码执行。
"""

from __future__ import annotations

import json
import socket
import struct


# ---------- RESP 客户端（RedisLimiter 注入用） ----------
def _resp_command(*parts: str) -> bytes:
    out = b"*" + str(len(parts)).encode() + b"\r\n"
    for p in parts:
        raw = p.encode("utf-8")
        out += b"$" + str(len(raw)).encode() + b"\r\n" + raw + b"\r\n"
    return out


class _RespReader:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.buf = b""

    def _line(self) -> bytes:
        while b"\r\n" not in self.buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("RESP connection closed")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\r\n", 1)
        return line

    def read(self):
        line = self._line()
        if not line:
            return None
        tag, body = line[:1], line[1:]
        if tag == b"+":
            return body.decode("utf-8")
        if tag == b":":
            return int(body)
        if tag == b"-":
            raise RuntimeError(body.decode("utf-8", errors="replace"))
        if tag == b"$":
            n = int(body)
            if n < 0:
                return None
            while len(self.buf) < n + 2:
                self.buf += self.sock.recv(4096)
            val, self.buf = self.buf[:n], self.buf[n + 2:]
            return val.decode("utf-8")
        if tag == b"*":
            n = int(body)
            return [self.read() for _ in range(n)]
        return line.decode("utf-8")


class RespRedisClient:
    """最小 RESP client：实现 RedisLimiter 使用的 zset 命令与 pipeline。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 6379, timeout: float = 5.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.reader = _RespReader(self.sock)

    def _cmd(self, *parts: str):
        self.sock.sendall(_resp_command(*parts))
        return self.reader.read()

    def ping(self) -> bool:
        return self._cmd("PING") == "PONG"

    def zadd(self, key: str, mapping: dict[str, float]) -> int:
        parts = ["ZADD", key]
        for member, score in mapping.items():
            parts += [repr(score), member]
        return int(self._cmd(*parts))

    def zcard(self, key: str) -> int:
        return int(self._cmd("ZCARD", key))

    def zremrangebyscore(self, key: str, lo: str | float, hi: str | float) -> int:
        return int(self._cmd("ZREMRANGEBYSCORE", key, str(lo), str(hi)))

    def pexpire(self, key: str, ms: int) -> int:
        return int(self._cmd("PEXPIRE", key, str(ms)))

    def zrange(self, key: str, start: int, stop: int, withscores: bool = False) -> list[tuple[str, float]]:
        parts = ["ZRANGE", key, str(start), str(stop)]
        if withscores:
            parts.append("WITHSCORES")
        rows = self._cmd(*parts) or []
        if withscores:
            return [(rows[i], float(rows[i + 1])) for i in range(0, len(rows), 2)]
        return [(row, 0.0) for row in rows]

    def flushall(self) -> None:
        self._cmd("FLUSHALL")

    def pipeline(self, transaction: bool = True):
        return _RespPipeline(self)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class _RespPipeline:
    """按顺序缓存命令，execute 时逐条发到同一 socket 并收集结果。

    RedisLimiter 关心的是 execute() 返回的列表（results[2] 为 zcard），
    这里不启用 MULTI/EXEC 事务语义——验证的是限流算法与其网络往返，
    事务语义由 Redis 服务端本身保证，不在本模拟范围。
    """

    def __init__(self, client: RespRedisClient):
        self.client = client
        self._queue: list[tuple[tuple[str, ...], object]] = []

    def zremrangebyscore(self, key, lo, hi):
        self._queue.append((("ZREMRANGEBYSCORE", key, str(lo), str(hi)), None))
        return self

    def zadd(self, key, mapping):
        parts = ["ZADD", key]
        for member, score in mapping.items():
            parts += [repr(score), member]
        self._queue.append((tuple(parts), None))
        return self

    def zcard(self, key):
        self._queue.append((("ZCARD", key), None))
        return self

    def pexpire(self, key, ms):
        self._queue.append((("PEXPIRE", key, str(ms)), None))
        return self

    def execute(self) -> list:
        results = []
        for parts, _ in self._queue:
            self.client.sock.sendall(_resp_command(*parts))
            results.append(self.client.reader.read())
        self._queue.clear()
        return results


# ---------- PG 语义连接（PgLeaseSessionLocker 注入用） ----------
class _Cursor:
    def __init__(self, payload: dict):
        self.rowcount = int(payload.get("rowcount", 0))
        self._rows = payload.get("rows") or []

    def fetchall(self) -> list[dict]:
        return list(self._rows)


class PgStubConnection:
    """满足 app.pglock._Connection 契约：execute(sql, params) / commit() / close()。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 5432, timeout: float = 5.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)

    def _request(self, payload: dict) -> dict:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.sock.sendall(struct.pack("!I", len(raw)) + raw)
        head = self._recv_exact(4)
        size = struct.unpack("!I", head)[0]
        return json.loads(self._recv_exact(size).decode("utf-8"))

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("PG stub connection closed")
            buf += chunk
        return buf

    def execute(self, sql: str, params: tuple = ()) -> _Cursor:
        resp = self._request({"op": "execute", "sql": sql, "params": list(params or ())})
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return _Cursor(resp)

    def commit(self) -> None:
        # stub 在 execute 内已提交；commit 保留契约以适配真实代码路径
        return None

    def close(self) -> None:
        try:
            self._request({"op": "close"})
        except Exception:
            pass
        try:
            self.sock.close()
        except OSError:
            pass
