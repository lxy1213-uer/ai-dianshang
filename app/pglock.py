"""Postgres 租约锁：与 SQLite 版同一 SETNX-with-TTL 语义，可注入连接。

与 `locking.DbLeaseSessionLocker` 的区别只有方言与驱动：
- 占位符 `%s`（psycopg），时间列 `TIMESTAMPTZ`；
- 连接由外部注入（`connect()` 工厂），测试用假连接零 Postgres 依赖；
- `psycopg` 延迟导入，零配置演示完全不需要安装。

注意：**锁后端配置成 pg 但驱动缺失时直接报错，绝不静默退回进程内锁**——
多实例部署下静默回退会破坏会话串行化，属于正确性事故，不是可用性降级。
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Protocol

PG_LOCK_DDL = """
CREATE TABLE IF NOT EXISTS session_locks (
    session_id  TEXT PRIMARY KEY,
    owner       TEXT NOT NULL,
    expires_at  TIMESTAMPTZ NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL
);
"""

# 接受 psycopg 的游标风格：connection.execute(sql, params) -> cursor(rowcount, fetchall)
_ACQUIRE_SQL = """
INSERT INTO session_locks(session_id, owner, expires_at, updated_at)
VALUES (%s, %s, %s, %s)
ON CONFLICT (session_id) DO UPDATE SET
    owner      = excluded.owner,
    expires_at = excluded.expires_at,
    updated_at = excluded.updated_at
WHERE session_locks.expires_at <= %s
"""
_RENEW_SQL = "UPDATE session_locks SET expires_at = %s, updated_at = %s WHERE session_id = %s AND owner = %s"
_RELEASE_SQL = "DELETE FROM session_locks WHERE session_id = %s AND owner = %s"
_INSPECT_SQL = "SELECT session_id, owner, expires_at, updated_at FROM session_locks"


class LockTimeout(RuntimeError):
    pass


class _Connection(Protocol):
    def execute(self, sql: str, params: tuple = ()): ...

    def commit(self) -> None: ...

    def close(self) -> None: ...


class PgLeaseSessionLocker:
    """Postgres 会话租约锁（自动续期 + 只释放自己的 owner）。"""

    backend = "pg"

    def __init__(
        self,
        connect,
        lease_seconds: float = 30.0,
        wait_seconds: float = 5.0,
        poll_interval: float = 0.02,
    ):
        self.lease = max(0.2, float(lease_seconds))
        self.wait = max(0.0, float(wait_seconds))
        self.poll = max(0.005, float(poll_interval))
        self._guard = threading.RLock()
        self._conn: _Connection = connect()
        with self._guard:
            self._conn.execute(PG_LOCK_DDL)
            self._conn.commit()

    @staticmethod
    def _now() -> str:
        # 用 UTC + 微秒：TIMESTAMPTZ 列接受带时区的 ISO 字符串，字典序与时间一致。
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f%z")

    def _stamp(self, offset_seconds: float) -> str:
        return (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).strftime(
            "%Y-%m-%d %H:%M:%S.%f%z"
        )

    def _try_acquire(self, session_id: str, owner: str) -> bool:
        now = self._now()
        with self._guard:
            cur = self._conn.execute(
                _ACQUIRE_SQL,
                (session_id, owner, self._stamp(self.lease), now, now),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def _renew(self, session_id: str, owner: str) -> None:
        with self._guard:
            self._conn.execute(
                _RENEW_SQL,
                (self._stamp(self.lease), self._now(), session_id, owner),
            )
            self._conn.commit()

    def _release(self, session_id: str, owner: str) -> None:
        with self._guard:
            self._conn.execute(_RELEASE_SQL, (session_id, owner))
            self._conn.commit()

    def _renew_loop(self, session_id: str, owner: str, stop: threading.Event) -> None:
        interval = max(0.05, self.lease / 3)
        while not stop.wait(interval):
            try:
                self._renew(session_id, owner)
            except Exception:
                return  # 续期失败不抛到业务线程，租约到期后自然释放

    @contextmanager
    def lock(self, session_id: str):
        owner = uuid.uuid4().hex
        deadline = time.monotonic() + self.wait
        while not self._try_acquire(session_id, owner):
            if time.monotonic() >= deadline:
                raise LockTimeout(f"会话 {session_id} 正在处理中，请稍后重试")
            time.sleep(self.poll)
        stop = threading.Event()
        keeper = threading.Thread(
            target=self._renew_loop, args=(session_id, owner, stop),
            daemon=True, name=f"pg-lease-{session_id[:8]}",
        )
        keeper.start()
        try:
            yield
        finally:
            stop.set()
            keeper.join(timeout=1.0)
            try:
                self._release(session_id, owner)
            except Exception:
                pass  # 锁表异常不能掩盖业务结果

    def inspect(self) -> list[dict]:
        with self._guard:
            cur = self._conn.execute(_INSPECT_SQL)
            return [dict(row) for row in cur.fetchall()]

    def close(self) -> None:
        with self._guard:
            self._conn.close()


def build_pg_session_locker(
    dsn: str, lease_seconds: float = 30.0, wait_seconds: float = 5.0, connect=None
) -> PgLeaseSessionLocker:
    """`connect` 可注入（测试）；未注入时延迟 import psycopg。"""
    if connect is None:
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "AGENT_LOCK_BACKEND=pg 需要 psycopg：pip install 'psycopg[binary]'"
            ) from exc
        connect = lambda: psycopg.connect(dsn)
    return PgLeaseSessionLocker(connect, lease_seconds, wait_seconds)
