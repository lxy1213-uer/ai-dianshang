"""SLA 超时通知验收：企微 webhook 降级、去重、脱敏、事件留痕。"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.notify import WecomNotifier, build_notifier


class FakeNotifier:
    """记录调用，替换真实 HTTP 发送。"""

    def __init__(self):
        self.calls: list[list[dict]] = []
        self.configured = True

    def notify_sla_overdue(self, records):
        self.calls.append(list(records))
        return True


# ---------------------------------------------------------------------------
# 1. WecomNotifier 单元：开关、URL、payload、降级、脱敏
# ---------------------------------------------------------------------------

def test_no_webhook_is_noop():
    n = WecomNotifier(webhook="", enabled=True)
    assert not n.configured
    assert n.notify_sla_overdue([{"queue_id": "q1"}]) is False


def test_disabled_is_noop():
    n = WecomNotifier(webhook="abc123", enabled=False)
    assert not n.configured
    assert n.notify_sla_overdue([{"queue_id": "q1"}]) is False


def test_key_expands_to_wecom_url():
    n = WecomNotifier(webhook="abc123")
    assert n.webhook.endswith("key=abc123")
    full = WecomNotifier(webhook="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xyz")
    assert full.webhook.startswith("https://")


def test_payload_markdown_has_queue_and_no_pii():
    n = WecomNotifier(webhook="abc", desk_base_url="https://desk.example")
    captured = {}

    def fake_send(url, payload):
        captured["url"] = url
        captured["payload"] = payload
        return True

    n._send = fake_send
    ok = n.notify_sla_overdue([
        {"queue_id": "q1", "user_id": "u_1001", "reason": "手机号 13812345678 的问题", "sla_deadline": "2026-11-26 19:00:00"}
    ])
    assert ok is True
    p = captured["payload"]
    assert p["msgtype"] == "markdown"
    content = p["markdown"]["content"]
    assert "q1" in content
    assert "https://desk.example/desk#queue" in content
    assert "13812345678" not in content          # PII 已打码
    assert "138****5678" in content


def test_send_failure_degrades_without_raising():
    n = WecomNotifier(webhook="abc")

    def boom(url, payload):
        raise OSError("network down")

    n._send = boom
    assert n.notify_sla_overdue([{"queue_id": "q1"}]) is False


def test_empty_records_no_send():
    n = WecomNotifier(webhook="abc")
    n._send = lambda url, payload: (_ for _ in ()).throw(AssertionError("不应发送"))
    assert n.notify_sla_overdue([]) is False


def test_build_notifier_from_settings():
    class S:
        wecom_webhook = "k"
        sla_notify_enabled = True
        desk_base_url = ""
    assert build_notifier(S()).configured is True


# ---------------------------------------------------------------------------
# 2. AgentDesk 集成：新超时通知一次、去重、事件留痕
# ---------------------------------------------------------------------------

def test_check_sla_notifies_once_and_logs_event(make_coordinator):
    c = make_coordinator()
    fake = FakeNotifier()
    c.desk.notifier = fake

    rec = c.desk.enqueue(
        {"session_id": "s_sla", "user_id": "u_1001", "state": "HUMAN"},
        {}, "测试超时", recent_turns=[],
    )
    qid = rec["queue_id"]
    # 把 deadline 改到过去
    past = (datetime.now() - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
    with c.db._lock:
        c.db._conn.execute("UPDATE human_queue SET sla_deadline = ? WHERE queue_id = ?", (past, qid))
        c.db._conn.commit()

    first = c.desk.check_sla()
    assert [r["queue_id"] for r in first] == [qid]
    assert len(fake.calls) == 1
    assert c.desk.last_notified == [qid]

    # 第二次不再通知（overdue 已置 1，mark_overdue_records 不再返回）
    second = c.desk.check_sla()
    assert second == []
    assert len(fake.calls) == 1

    events = [e["kind"] for e in c.db.list_events("s_sla")]
    assert "sla_breach" in events


def test_list_queue_auto_checks_sla(make_coordinator):
    c = make_coordinator()
    fake = FakeNotifier()
    c.desk.notifier = fake
    rec = c.desk.enqueue(
        {"session_id": "s_sla2", "user_id": "u_1001", "state": "HUMAN"}, {}, "超时", []
    )
    past = (datetime.now() - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
    with c.db._lock:
        c.db._conn.execute("UPDATE human_queue SET sla_deadline = ? WHERE queue_id = ?", (past, rec["queue_id"]))
        c.db._conn.commit()

    items = c.desk.list_queue(only_overdue=True)
    assert any(i["queue_id"] == rec["queue_id"] for i in items)
    assert len(fake.calls) == 1


def test_no_notifier_when_webhook_absent(make_coordinator):
    c = make_coordinator()  # 默认无 webhook
    assert c.notifier.configured is False
    rec = c.desk.enqueue(
        {"session_id": "s_sla3", "user_id": "u_1001", "state": "HUMAN"}, {}, "超时", []
    )
    past = (datetime.now() - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
    with c.db._lock:
        c.db._conn.execute("UPDATE human_queue SET sla_deadline = ? WHERE queue_id = ?", (past, rec["queue_id"]))
        c.db._conn.commit()
    # 不应抛异常，通知只是 no-op
    assert len(c.desk.check_sla()) == 1


# --- 可插拔协议与重试（外部依赖骨架） ---------------------------------------
def test_notifier_protocol_is_satisfied_by_wecom_notifier():
    from app.notify import Notifier, WecomNotifier
    n = WecomNotifier(webhook="abc")
    assert hasattr(n, "configured") and hasattr(n, "notify_sla_overdue")
    assert set(Notifier.__annotations__) <= {"configured"}


def test_notifier_retries_then_succeeds():
    from app.notify import WecomNotifier
    calls = []
    sleeps = []
    n = WecomNotifier(webhook="abc", retries=2, retry_backoff=0.1, sleep=sleeps.append)

    def flaky(url, payload):
        calls.append(url)
        if len(calls) < 3:
            raise OSError("network")
        return True

    n._send = flaky
    assert n.notify_sla_overdue([{"queue_id": "q1"}]) is True
    assert len(calls) == 3
    assert sleeps == [0.1, 0.2]


def test_notifier_retries_exhausted_still_degrades():
    from app.notify import WecomNotifier
    calls = []
    n = WecomNotifier(webhook="abc", retries=2, sleep=lambda _: None)
    n._send = lambda url, payload: calls.append(url) or False
    assert n.notify_sla_overdue([{"queue_id": "q1"}]) is False
    assert len(calls) == 3


def test_build_notifier_passes_retry_config():
    from app.config import Settings
    from app.notify import build_notifier
    s = Settings(sla_notify_retries=3)
    assert build_notifier(s).retries == 3
