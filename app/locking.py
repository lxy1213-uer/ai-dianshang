"""会话锁可插拔层：进程内锁（默认） / 数据库租约锁（跨进程）。

为什么需要：
`Coordinator.chat()` 把「读 session → 状态机 → 写 session」整轮串行化，
原实现是进程内 `threading.Lock` 字典——多 worker / 多进程部署时各管各的，
同一会话的并发请求仍会互相覆盖状态。

DB 租约锁（标准 SETNX-with-TTL 思路，落在现有 SQLite 上，零新依赖）：
- 原子抢占：`INSERT ... ON CONFLICT(session_id) DO UPDATE ... WHERE expires_at <= now`，
  owner token 区分持有者；SQLite / Postgres 都有同样的 upsert 语义（方言差异见 README）。
- 崩溃自愈：TTL 过期后任何实例都能抢占，无需人工清理（与 `recover_creating()` 同一哲学）。
- 自动续期：后台线程在持锁期间按 lease/3 续租，避免 LLM 长对话（实测单轮可达 69s）
  把锁「饿死」；释放时只删自己 owner 的行，不误删已被他人抢占的锁。
- 幂等建单仍由 `tickets.idempotency_key UNIQUE` 在数据库层保证，跨进程同样只落一条。

诚实边界：租约依赖各实例本机时钟（跨机器需 NTP）；本项目未内置 Postgres 驱动，
`AGENT_LOCK_BACKEND=db` 跑在现有 SQLite 上，SQL 迁移路径写在 README。
默认 `inprocess`，零配置行为与课程基线完全一致。
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

# 必须带微秒：秒级时间戳会把亚秒租约（短 TTL 测试、高并发续期）截断成
# 「写入即过期」，导致刚抢到的锁被下一个实例立刻抢占、续期形同虚设。
# 等宽格式保证字典序比较与时间先后一致。
TIME_FMT = "%Y-%m-%d %H:%M:%S.%f"

LEASE_DDL = """
CREATE TABLE IF NOT EXISTS session_locks (
    session_id  TEXT PRIMARY KEY,
    owner       TEXT NOT NULL,
    expires_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""


class LockTimeout(RuntimeError):
    """等待超过 wait_seconds 仍未拿到锁；调用方可重试，不会静默并发执行。"""


class SessionLocker(Protocol):
    backend: str

    def lock(self, session_id: str):
        """返回上下文管理器；进入时持有锁，退出时释放。"""
        ...

    def close(self) -> None:
        ...


class InProcessSessionLocker:
    """默认实现：单进程内按 session_id 串行化（原有行为）。"""

    backend = "inprocess"

    def __init__(self):
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    @contextmanager
    def lock(self, session_id: str):
        with self._guard:
            lk = self._locks.setdefault(session_id, threading.Lock())
        with lk:
            yield

    def close(self) -> None:
        return None


class DbLeaseSessionLocker:
    """基于数据库租约的跨进程会话锁（独立连接 + 自动续期）。"""

    backend = "db"

    def __init__(
        self,
        db_path: Path | str,
        lease_seconds: float = 30.0,
        wait_seconds: float = 5.0,
        poll_interval: float = 0.02,
    ):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lease = max(0.2, float(lease_seconds))
        self.wait = max(0.0, float(wait_seconds))
        self.poll = max(0.005, float(poll_interval))
        self._guard = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        with self._guard:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(LEASE_DDL)
            self._conn.commit()

    @staticmethod
    def _stamp(dt: datetime) -> str:
        return dt.strftime(TIME_FMT)

    def _try_acquire(self, session_id: str, owner: str) -> bool:
        now = datetime.now()
        expires = self._stamp(now + timedelta(seconds=self.lease))
        now_s = self._stamp(now)
        with self._guard:
            # WHERE 只作用在「已存在的那一行」：租约未过期则整句不改动，rowcount=0。
            cur = self._conn.execute(
                """
                INSERT INTO session_locks(session_id, owner, expires_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    owner      = excluded.owner,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
                WHERE session_locks.expires_at <= ?
                """,
                (session_id, owner, expires, now_s, now_s),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def _renew(self, session_id: str, owner: str) -> None:
        now = datetime.now()
        expires = self._stamp(now + timedelta(seconds=self.lease))
        with self._guard:
            self._conn.execute(
                """UPDATE session_locks SET expires_at = ?, updated_at = ?
                   WHERE session_id = ? AND owner = ?""",
                (expires, self._stamp(now), session_id, owner),
            )
            self._conn.commit()

    def _release(self, session_id: str, owner: str) -> None:
        with self._guard:
            self._conn.execute(
                "DELETE FROM session_locks WHERE session_id = ? AND owner = ?",
                (session_id, owner),
            )
            self._conn.commit()

    def _renew_loop(self, session_id: str, owner: str, stop: threading.Event) -> None:
        interval = max(0.05, self.lease / 3)
        while not stop.wait(interval):
            try:
                self._renew(session_id, owner)
            except sqlite3.Error:
                return  # 续期失败不抛到业务线程；租约到期后自然释放，由上层重试

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
            daemon=True, name=f"lease-renew-{session_id[:8]}",
        )
        keeper.start()
        try:
            yield
        finally:
            stop.set()
            keeper.join(timeout=1.0)
            try:
                self._release(session_id, owner)
            except sqlite3.Error:
                pass  # 锁表异常不能掩盖业务结果；TTL 到期后会自动释放

    def inspect(self) -> list[dict]:
        """诊断用：返回当前所有租约（测试也用它验证 owner 隔离）。"""
        with self._guard:
            rows = self._conn.execute(
                "SELECT session_id, owner, expires_at, updated_at FROM session_locks"
            ).fetchall()
            return [dict(r) for r in rows]

    def close(self) -> None:
        with self._guard:
            self._conn.close()


def build_session_locker(settings) -> SessionLocker:
    """按配置构造锁后端；未知值一律降级为 inprocess（不破坏零配置基线）。"""
    mode = (getattr(settings, "lock_backend", "inprocess") or "inprocess").lower()
    if mode == "pg":
        # 显式配置 pg 却缺驱动时**报错**，不静默降级：多实例场景下
        # 悄悄退回进程内锁会破坏会话串行化（正确性事故）。
        from .pglock import build_pg_session_locker

        return build_pg_session_locker(
            getattr(settings, "pg_dsn", "") or "",
            getattr(settings, "lock_lease_seconds", 30.0),
            getattr(settings, "lock_wait_seconds", 5.0),
        )
    if mode == "db":
        return DbLeaseSessionLocker(
            settings.db_path,
            getattr(settings, "lock_lease_seconds", 30.0),
            getattr(settings, "lock_wait_seconds", 5.0),
        )
    return InProcessSessionLocker()
