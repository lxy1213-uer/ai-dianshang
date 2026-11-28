"""事项 C：埋点漏斗的事件字典、脱敏、幂等、聚合与降级测试。"""

from __future__ import annotations

from app.domain.events import FUNNEL_STEPS, EventName, sanitize


def test_event_dict_covers_six_steps():
    assert len(FUNNEL_STEPS) == 6
    assert FUNNEL_STEPS[0] == EventName.SESSION_START.value
    assert FUNNEL_STEPS[-1] == EventName.TICKET_CREATED.value


def test_sanitize_removes_pii_keys():
    payload = {
        "order_id": "O20261115001",
        "phone": "13812345678",
        "address": "北京市朝阳区某路1号",
        "receiver": "张三",
    }
    clean = sanitize(payload)
    assert clean["order_id"] == "O20261115001"
    assert "phone" not in clean
    assert "address" not in clean
    assert "receiver" not in clean


def test_sanitize_masks_phone_and_id_in_text():
    clean = sanitize({"note": "联系 13812345678，身份证 110101199001011234"})
    assert "13812345678" not in clean["note"]
    assert "110101199001011234" not in clean["note"]
    assert "****" in clean["note"]


def test_track_is_idempotent(make_coordinator):
    c = make_coordinator()
    f = c.funnel
    assert f.track("s1", "u1", EventName.SESSION_START.value, {}, event_id="e1") is True
    assert f.track("s1", "u1", EventName.SESSION_START.value, {}, event_id="e1") is False


def test_funnel_counts_unique_sessions(make_coordinator):
    c = make_coordinator()
    f = c.funnel
    # 同一会话多次 intent 事件只算一次；两个会话各一次
    f.track("s1", "u1", EventName.SESSION_START.value, {}, event_id="s1:start")
    f.track("s1", "u1", EventName.SESSION_START.value, {}, event_id="s1:start2")
    f.track("s2", "u2", EventName.SESSION_START.value, {}, event_id="s2:start")
    f.track("s1", "u1", EventName.INTENT_RECOGNIZED.value, {"intent": "return"}, event_id="s1:intent")
    f.track("s1", "u1", EventName.ORDER_QUERIED.value, {}, event_id="s1:order")
    f.track("s2", "u2", EventName.ORDER_QUERIED.value, {}, event_id="s2:order")

    report = f.funnel()
    counts = {s["event"]: s["sessions"] for s in report["steps"]}
    # 累积语义：到达查单的会话必然经过会话开始与意图识别
    assert counts[EventName.SESSION_START.value] == 2
    assert counts[EventName.INTENT_RECOGNIZED.value] == 2
    assert counts[EventName.ORDER_QUERIED.value] == 2
    assert report["total_sessions"] == 2
    # 漏斗单调不增
    seq = [s["sessions"] for s in report["steps"]]
    assert seq == sorted(seq, reverse=True)


def test_funnel_conversion_rates(make_coordinator):
    c = make_coordinator()
    f = c.funnel
    for sid in ("a", "b", "c"):
        f.track(sid, "u", EventName.SESSION_START.value, {}, event_id=f"{sid}:start")
    f.track("a", "u", EventName.ORDER_QUERIED.value, {}, event_id="a:order")
    report = f.funnel()
    order_step = next(s for s in report["steps"] if s["event"] == EventName.ORDER_QUERIED.value)
    assert order_step["conversion_overall"] == round(1 / 3, 4)


def test_funnel_date_filter(make_coordinator):
    c = make_coordinator()
    f = c.funnel
    f.track("s1", "u", EventName.SESSION_START.value, {}, event_id="s1:start")
    # 未来日期区间应查不到
    report = f.funnel(date_from="2099-01-01", date_to="2099-12-31")
    assert report["total_sessions"] == 0


def test_branches_recorded(make_coordinator):
    c = make_coordinator()
    f = c.funnel
    f.track("s1", "u", EventName.HANDED_OFF.value, {}, event_id="s1:ho")
    f.track("s2", "u", EventName.HANDED_OFF.value, {}, event_id="s2:ho")
    report = f.funnel()
    assert report["branches"][EventName.HANDED_OFF.value] == 2


def test_track_failure_degrades(make_coordinator, monkeypatch):
    c = make_coordinator()
    monkeypatch.setattr(c.funnel.db, "log_metric_event", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
    # 埋点失败返回 False，不抛异常，主链路不受影响
    assert c.funnel.track("s", "u", "x", {}, event_id="e") is False


def test_chat_emits_funnel_events(make_coordinator):
    c = make_coordinator()
    c.chat("s-funnel", "u_1001", "我要退货")
    events = c.db.list_metric_events()
    kinds = {e["kind"] for e in events}
    assert EventName.SESSION_START.value in kinds
    assert EventName.INTENT_RECOGNIZED.value in kinds
