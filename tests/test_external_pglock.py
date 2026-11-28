"""Postgres 租约锁：注入假连接验证 SQL 语义，零 Postgres 依赖。"""

import pytest

from app.config import Settings
from app.locking import build_session_locker
from app.pglock import (
    _ACQUIRE_SQL,
    _RELEASE_SQL,
    _RENEW_SQL,
    PgLeaseSessionLocker,
    build_pg_session_locker,
)


class _Cursor:
    def __init__(self, rows=None, rowcount=0):
        self.rows = rows or []
        self.rowcount = rowcount

    def fetchall(self):
        return self.rows


class FakeConn:
    """记录 SQL；ON CONFLICT 语义由 rowcount 脚本驱动。"""

    def __init__(self, acquire_results=(True,)):
        self.calls = []
        self.rows = {}
        self.acquire_results = list(acquire_results)
        self.closed = False

    def execute(self, sql, params=()):
        self.calls.append((sql.strip(), tuple(params)))
        if "INSERT INTO session_locks" in sql:
            ok = self.acquire_results.pop(0) if self.acquire_results else False
            if ok:
                sid, owner, expires, _, _ = params
                self.rows[sid] = {"session_id": sid, "owner": owner, "expires_at": expires, "updated_at": expires}
            return _Cursor(rowcount=1 if ok else 0)
        if "UPDATE session_locks" in sql:
            _, _, sid, owner = params
            row = self.rows.get(sid)
            if row and row["owner"] == owner:
                row["expires_at"] = params[0]
                return _Cursor(rowcount=1)
            return _Cursor(rowcount=0)
        if "DELETE FROM session_locks" in sql:
            sid, owner = params
            row = self.rows.get(sid)
            if row and row["owner"] == owner:
                del self.rows[sid]
                return _Cursor(rowcount=1)
            return _Cursor(rowcount=0)
        if "SELECT" in sql:
            return _Cursor(rows=list(self.rows.values()))
        return _Cursor()

    def commit(self):
        self.calls.append(("COMMIT", ()))

    def close(self):
        self.closed = True


def conn_factory(conn):
    return lambda: conn


def test_acquire_uses_upsert_with_postgres_placeholders():
    conn = FakeConn()
    locker = PgLeaseSessionLocker(conn_factory(conn), lease_seconds=1.0, wait_seconds=0)
    with locker.lock("s-1"):
        assert "ON CONFLICT (session_id) DO UPDATE" in _ACQUIRE_SQL
        assert "%s" in _ACQUIRE_SQL and "?" not in _ACQUIRE_SQL
        assert "WHERE session_locks.expires_at <= %s" in _ACQUIRE_SQL
        assert "TIMESTAMPTZ" in conn.calls[0][0]
    assert locker.inspect() == []   # 已按 owner 释放


def test_lock_timeout_when_another_owner_holds_lease():
    conn = FakeConn(acquire_results=(False, False, False))
    locker = PgLeaseSessionLocker(conn_factory(conn), lease_seconds=1.0, wait_seconds=0)
    with pytest.raises(RuntimeError, match="正在处理中"):
        with locker.lock("s-busy"):
            pass


def test_renew_and_release_are_owner_scoped():
    conn = FakeConn()
    locker = PgLeaseSessionLocker(conn_factory(conn), lease_seconds=1.0)
    with locker.lock("s-2"):
        before = dict(conn.rows["s-2"])
        real_owner = before["owner"]
        locker._renew("s-2", "someone-else")  # 别人的 owner 不能续期
        assert conn.rows["s-2"] == before
        locker._renew("s-2", real_owner)
        assert conn.rows["s-2"]["expires_at"] != before["expires_at"]
    assert "s-2" not in conn.rows


def test_build_pg_session_locker_with_injected_connect():
    conn = FakeConn()
    locker = build_pg_session_locker("postgresql://x", connect=conn_factory(conn))
    assert locker.backend == "pg"
    locker.close()
    assert conn.closed


def test_factory_uses_pg_when_configured(monkeypatch):
    import app.pglock as mod

    conn = FakeConn()
    monkeypatch.setattr(mod, "build_pg_session_locker",
                        lambda *a, **k: PgLeaseSessionLocker(conn_factory(conn)))
    s = Settings.load(lock_backend="pg", pg_dsn="postgresql://x")
    assert build_session_locker(s).backend == "pg"


def test_factory_rejects_pg_without_driver(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_psycopg(name, *a, **k):
        if name == "psycopg":
            raise ImportError("no driver")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_psycopg)
    s = Settings.load(lock_backend="pg", pg_dsn="postgresql://x")
    with pytest.raises(RuntimeError, match="psycopg"):
        build_session_locker(s)


def test_default_lock_backend_is_inprocess():
    assert build_session_locker(Settings.load()).backend == "inprocess"
