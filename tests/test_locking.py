"""架构进阶 · 可插拔会话锁验收（默认 inprocess 零变化 / db 租约跨实例）。"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from threading import Thread

import pytest

from app.agent.coordinator import Coordinator
from app.config import Settings
from app.locking import (
    TIME_FMT,
    DbLeaseSessionLocker,
    InProcessSessionLocker,
    LockTimeout,
    build_session_locker,
)


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "locks.db"


# ---------- 1. 默认行为不变 ----------

def test_default_backend_is_inprocess(db_path):
    s = Settings.load(llm_mode="rules", db_path=db_path, lock_backend="inprocess")
    locker = build_session_locker(s)
    assert isinstance(locker, InProcessSessionLocker)
    assert locker.backend == "inprocess"
    locker.close()


def test_unknown_backend_falls_back_to_inprocess(db_path):
    s = Settings.load(llm_mode="rules", db_path=db_path, lock_backend="redis-typo")
    assert isinstance(build_session_locker(s), InProcessSessionLocker)


def test_settings_describe_exposes_lock_backend():
    d = Settings.load(lock_backend="db").describe()
    assert d["lock_backend"] == "db"


def test_inprocess_locker_serializes_same_session():
    """进程内锁：同一 session 的两段临界区不得交错。"""
    locker = InProcessSessionLocker()
    order: list[str] = []

    def worker(name: str):
        with locker.lock("s1"):
            order.append(f"{name}-in")
            time.sleep(0.05)
            order.append(f"{name}-out")

    t1, t2 = Thread(target=worker, args=("A",)), Thread(target=worker, args=("B",))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert len(order) == 4
    assert order[0].split("-")[0] == order[1].split("-")[0]  # in/out 未被另一线程插入
    assert order[2].split("-")[0] == order[3].split("-")[0]


# ---------- 2. DB 租约锁：跨实例互斥 ----------

def test_db_backend_is_built_from_settings(db_path):
    s = Settings.load(llm_mode="rules", db_path=db_path, lock_backend="db", lock_lease_seconds=2.0)
    locker = build_session_locker(s)
    try:
        assert isinstance(locker, DbLeaseSessionLocker)
        assert locker.backend == "db"
    finally:
        locker.close()


def test_two_instances_exclude_each_other(db_path):
    """两个独立连接（模拟两个 worker）争用同一会话：第二个必须等待超时。"""
    a = DbLeaseSessionLocker(db_path, lease_seconds=2.0, wait_seconds=0.15)
    b = DbLeaseSessionLocker(db_path, lease_seconds=2.0, wait_seconds=0.15)
    try:
        with a.lock("s1"):
            with pytest.raises(LockTimeout):
                with b.lock("s1"):
                    pytest.fail("同一会话被两个实例同时进入临界区")
        with b.lock("s1"):  # 释放后可再获取
            pass
        assert a.inspect() == []
    finally:
        a.close(); b.close()


def test_release_only_deletes_own_owner(db_path):
    a = DbLeaseSessionLocker(db_path, lease_seconds=2.0, wait_seconds=0.15)
    b = DbLeaseSessionLocker(db_path, lease_seconds=2.0, wait_seconds=0.15)
    try:
        with a.lock("s1"):
            b._release("s1", "someone-else")
            assert len(a.inspect()) == 1  # 他人的释放不能删掉我持有的租约
        assert a.inspect() == []
    finally:
        a.close(); b.close()


def test_expired_lease_can_be_reclaimed(db_path):
    """崩溃自愈：持有者进程死掉留下过期租约，新实例可抢占。"""
    b = DbLeaseSessionLocker(db_path, lease_seconds=0.3, wait_seconds=0.1)
    try:
        past = (datetime.now() - timedelta(seconds=10)).strftime(TIME_FMT)
        with b._guard:
            b._conn.execute(
                "INSERT INTO session_locks(session_id, owner, expires_at, updated_at) VALUES (?,?,?,?)",
                ("s1", "dead-owner", past, past),
            )
            b._conn.commit()
        assert b._try_acquire("s1", "new-owner") is True
        assert b.inspect()[0]["owner"] == "new-owner"
        assert b._try_acquire("s1", "third-owner") is False  # 未过期则不可抢占
    finally:
        b.close()


def test_auto_renewal_keeps_lock_during_long_task(db_path):
    """长对话（LLM 单轮可达数十秒）不得把租约饿死。"""
    a = DbLeaseSessionLocker(db_path, lease_seconds=0.3, wait_seconds=0.1)
    b = DbLeaseSessionLocker(db_path, lease_seconds=0.3, wait_seconds=0.1)
    try:
        with a.lock("s1"):
            time.sleep(0.8)  # 远超 lease
            with pytest.raises(LockTimeout):
                with b.lock("s1"):
                    pytest.fail("续期失效，长任务期间锁被抢占")
        with b.lock("s1"):
            pass
        assert a.inspect() == []
    finally:
        a.close(); b.close()


def test_lock_timeout_is_raised_not_silently_skipped(db_path):
    a = DbLeaseSessionLocker(db_path, lease_seconds=5.0, wait_seconds=0.1)
    b = DbLeaseSessionLocker(db_path, lease_seconds=5.0, wait_seconds=0.1)
    try:
        with a.lock("s1"):
            with pytest.raises(LockTimeout) as exc:
                with b.lock("s1"):
                    pass
            assert "正在处理中" in str(exc.value)
    finally:
        a.close(); b.close()


# ---------- 3. Coordinator 端到端 ----------

def test_coordinator_with_db_lock_runs_chat_and_releases(tmp_path):
    s = Settings.load(
        llm_mode="rules", db_path=tmp_path / "coord.db",
        lock_backend="db", lock_lease_seconds=2.0,
    )
    c = Coordinator(s)
    try:
        assert c._locker.backend == "db"
        r = c.chat("s_lock", "u_1001", "我要退货")
        assert r.reply
        assert c._locker.inspect() == []  # 每轮退出后租约必须释放
        r2 = c.chat("s_lock", "u_1001", "O20261115001")
        assert r2.state == "AWAIT_CONFIRM"
    finally:
        c.close()


def test_coordinator_default_locker_is_inprocess(tmp_path):
    c = Coordinator(Settings.load(llm_mode="rules", db_path=tmp_path / "plain.db"))
    try:
        assert c._locker.backend == "inprocess"
        assert c.chat("s", "u_1001", "我要退货").reply
    finally:
        c.close()
