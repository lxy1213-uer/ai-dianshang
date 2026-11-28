"""事项 A：工单状态回写的状态机、幂等、并发与审计测试。"""

from __future__ import annotations

import threading

import pytest

from app.domain.status import ALLOWED, TERMINAL, TicketStatus, can_transition
from app.domain.tickets import TicketError


def _make_ticket(db, key="k-status"):
    return db.insert_ticket(key, "s-status", "u_1001", "O20261115001", "no_reason", "req-status")[0]


def test_status_enum_and_whitelist():
    assert [s.value for s in TicketStatus] == [
        "CREATED", "REVIEWING", "APPROVED", "RECEIVED", "REFUNDED", "REJECTED", "CLOSED"
    ]
    assert can_transition(TicketStatus.CREATED, TicketStatus.REVIEWING)
    assert not can_transition(TicketStatus.CREATED, TicketStatus.REFUNDED)
    assert TERMINAL == {TicketStatus.REFUNDED, TicketStatus.REJECTED, TicketStatus.CLOSED}
    assert ALLOWED[TicketStatus.CLOSED] == set()


def test_created_ticket_starts_at_created(make_coordinator):
    c = make_coordinator()
    t = _make_ticket(c.db)
    assert t["status"] == "CREATED"
    assert t["version"] == 1


def test_valid_transition_increments_version_and_timeline(make_coordinator):
    c = make_coordinator()
    t = _make_ticket(c.db)
    r = c.tickets.update_status(t["ticket_id"], "REVIEWING", operator="agent1")
    assert r["status"] == "REVIEWING"
    assert r["version"] == 2
    assert r["updated_at"]
    assert r["next_step"]
    assert len(r["timeline"]) == 1
    assert r["timeline"][0]["payload"]["operator"] == "agent1"


def test_full_happy_path_to_refunded(make_coordinator):
    c = make_coordinator()
    t = _make_ticket(c.db)
    for target in ("REVIEWING", "APPROVED", "RECEIVED", "REFUNDED"):
        r = c.tickets.update_status(t["ticket_id"], target)
    assert r["status"] == "REFUNDED"
    assert r["is_terminal"] is True
    assert len(r["timeline"]) == 4


def test_invalid_transition_does_not_persist(make_coordinator):
    c = make_coordinator()
    t = _make_ticket(c.db)
    with pytest.raises(TicketError) as e:
        c.tickets.update_status(t["ticket_id"], "REFUNDED")
    assert e.value.code == "invalid_transition"
    assert c.db.get_ticket(t["ticket_id"])["status"] == "CREATED"
    assert c.db.ticket_timeline(t["ticket_id"]) == []


def test_terminal_state_cannot_transition(make_coordinator):
    c = make_coordinator()
    t = _make_ticket(c.db)
    for target in ("REVIEWING", "APPROVED", "RECEIVED", "REFUNDED"):
        c.tickets.update_status(t["ticket_id"], target)
    with pytest.raises(TicketError) as e:
        c.tickets.update_status(t["ticket_id"], "CLOSED")
    assert e.value.code == "invalid_transition"


def test_same_idempotency_key_records_once(make_coordinator):
    c = make_coordinator()
    t = _make_ticket(c.db)
    first = c.tickets.update_status(t["ticket_id"], "REVIEWING", operator="a", idempotency_key="idem-x")
    second = c.tickets.update_status(t["ticket_id"], "REVIEWING", operator="a", idempotency_key="idem-x")
    assert first["version"] == second["version"] == 2
    assert len(second["timeline"]) == 1


def test_same_target_status_is_idempotent(make_coordinator):
    c = make_coordinator()
    t = _make_ticket(c.db)
    c.tickets.update_status(t["ticket_id"], "REVIEWING")
    again = c.tickets.update_status(t["ticket_id"], "REVIEWING")
    assert again["version"] == 2
    assert len(again["timeline"]) == 1


def test_optimistic_lock_only_one_wins(make_coordinator):
    """验证乐观锁的核心不变量：两个请求基于同一 version 提交，只有一个能写成功。"""
    c = make_coordinator()
    t = _make_ticket(c.db, key="k-concurrent")
    tid = t["ticket_id"]
    results = []

    # 直接测 db 层条件更新：都传 expected_version=1，模拟"都基于 v1 快照"。
    def worker(target):
        updated, result = c.db.update_ticket_status(
            tid, target, "w-" + target, "下一步", f"idem-{target}", expected_version=1
        )
        results.append(result)

    barrier = threading.Barrier(2)

    def synced(target):
        barrier.wait()
        worker(target)

    threads = [threading.Thread(target=synced, args=("REVIEWING",)),
               threading.Thread(target=synced, args=("REJECTED",))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert sorted(results) == ["conflict", "updated"]
    final = c.db.get_ticket(tid)
    assert final["status"] in ("REVIEWING", "REJECTED")
    assert final["version"] == 2
    assert len(c.db.ticket_timeline(tid)) == 1


def test_service_rejects_stale_expected_version(make_coordinator):
    """Service 层：expected_version 与现状不符时抛 version_conflict。"""
    c = make_coordinator()
    t = _make_ticket(c.db, key="k-stale")
    c.tickets.update_status(t["ticket_id"], "REVIEWING")  # 现在 v2
    with pytest.raises(TicketError) as e:
        c.tickets.update_status(t["ticket_id"], "APPROVED", expected_version=1)
    assert e.value.code == "version_conflict"


def test_unknown_ticket_and_status(make_coordinator):
    c = make_coordinator()
    with pytest.raises(TicketError) as e:
        c.tickets.detail("NO-SUCH-TICKET")
    assert e.value.code == "not_found"
    t = _make_ticket(c.db, key="k-unknown")
    with pytest.raises(TicketError) as e:
        c.tickets.update_status(t["ticket_id"], "BOGUS")
    assert e.value.code == "invalid_status"
