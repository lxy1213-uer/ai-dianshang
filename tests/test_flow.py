"""第 4-6 周验收：工具校验、状态机、越权、确认失效、幂等建单、持久化。"""

import pytest

from app.agent.flow import State, TransitionError
from app.agent.tools import ToolError, validate_arguments, TOOL_SCHEMAS
from app.domain.tickets import TicketService
from tests.conftest import chat

SID = "s_flow"


def test_happy_path_creates_ticket(coordinator):
    result = chat(coordinator, SID, "我要退货", "O20261115001", "确认")
    assert result.state == "TICKETED"
    assert result.ticket is not None
    assert result.ticket["ticket_id"].startswith("RT")
    assert result.slots["ticket_id"] == result.ticket["ticket_id"]
    assert len(coordinator.db.list_tickets(SID)) == 1


def test_asks_for_order_number_first(coordinator):
    result = chat(coordinator, SID, "我要退货")
    assert result.state == "AWAIT_ORDER"
    assert "订单号" in result.reply


def test_confirm_summary_has_deadline_and_policy(coordinator):
    result = chat(coordinator, SID, "我要退货", "O20261115001")
    assert result.state == "AWAIT_CONFIRM"
    assert "今天到期" in result.reply
    assert {c["clause_id"] for c in result.citations} >= {"R-01"}


def test_expired_order_goes_human(coordinator):
    result = chat(coordinator, SID, "我要退货", "O20261115002")
    assert result.state == "HUMAN"
    assert "超过" in result.reply


def test_forbidden_order_is_blocked_and_logged(coordinator):
    result = chat(coordinator, SID, "我要退货", "O20261112008")
    assert result.state == "AWAIT_ORDER"
    assert "不在当前登录账号" in result.reply
    events = coordinator.db.list_events(SID)
    assert any(e["kind"] == "security_violation" for e in events)
    assert coordinator.db.count_tickets() == 0


def test_ticket_cannot_be_created_without_confirmation(coordinator):
    """服务端必须校验确认，客户端伪造 confirmed=False 直接拒绝。"""
    chat(coordinator, SID, "我要退货", "O20261115001")
    session = coordinator.db.get_session(SID)
    token = session["slots"]["confirm_token"]
    with pytest.raises(ToolError) as exc:
        coordinator.tools.call(
            "create_return_ticket",
            {"order_id": "O20261115001", "confirm_token": token},
            {
                "user_id": "u_1001",
                "session_id": SID,
                "session": {"state": session["state"], "slots": session["slots"]},
                "confirmed": False,
            },
        )
    assert exc.value.code == "not_confirmed"
    assert coordinator.db.count_tickets() == 0


def test_switching_order_invalidates_old_confirmation(coordinator):
    """换单后旧确认令牌失效，不能拿旧令牌建单。"""
    chat(coordinator, SID, "我要退货", "O20261115001")
    old_token = coordinator.db.get_session(SID)["slots"]["confirm_token"]
    chat(coordinator, SID, "换成 O20261116013")
    session = coordinator.db.get_session(SID)
    assert session["slots"]["order_id"] == "O20261116013"
    assert session["slots"]["confirm_token"] != old_token

    with pytest.raises(ToolError) as exc:
        coordinator.tools.call(
            "create_return_ticket",
            {"order_id": "O20261115001", "confirm_token": old_token},
            {
                "user_id": "u_1001",
                "session_id": SID,
                "session": {"state": session["state"], "slots": session["slots"]},
                "confirmed": True,
            },
        )
    assert exc.value.code in ("stale_confirmation", "order_mismatch")
    assert any(e["kind"] == "confirm_invalidated" for e in coordinator.db.list_events(SID))


def test_idempotent_retry_returns_same_ticket(make_coordinator):
    """模拟响应丢失：用户重试后只返回原工单，不重复建单。"""
    c = make_coordinator("loss.db", simulate_response_loss=True)
    first = chat(c, "s_loss", "我要退货", "O20261115001", "确认")
    assert first.state == "AWAIT_CONFIRM"
    assert first.meta.get("retryable") is True

    second = chat(c, "s_loss", "确认")
    assert second.state == "TICKETED"
    assert second.meta["created"] is False
    assert len(c.db.list_tickets("s_loss")) == 1


def test_ticket_service_is_idempotent(make_coordinator):
    c = make_coordinator("dup.db")
    service = TicketService(c.db, c.settings)
    kwargs = dict(
        session_id="s_dup",
        user_id="u_1001",
        order_id="O20261115001",
        confirm_token="token-x",
        reason="no_reason",
        confirmed=True,
    )
    first, created_1 = service.create(**kwargs)
    second, created_2 = service.create(**kwargs, attempt_no=2)
    assert created_1 is True
    assert created_2 is False
    assert first["ticket_id"] == second["ticket_id"]
    assert c.db.count_tickets() == 1


def test_session_survives_restart(make_coordinator):
    """第 8 周验收：进程重启后会话与工单都还在。"""
    first = make_coordinator("restart.db")
    chat(first, "s_restart", "我要退货", "O20261115001")
    view = first.session_view("s_restart", "u_1001")
    assert view["state"] == "AWAIT_CONFIRM"

    second = make_coordinator("restart.db")
    result = chat(second, "s_restart", "确认")
    assert result.state == "TICKETED"
    assert len(second.db.list_tickets("s_restart")) == 1


def test_session_reset_when_user_changes(coordinator):
    chat(coordinator, SID, "我要退货", "O20261115001")
    result = chat(coordinator, SID, "你好", user_id="u_1002")
    view = coordinator.session_view(SID, "u_1002")
    assert view["user_id"] == "u_1002"
    assert view["slots"].get("order_id") is None
    assert result.state in ("IDLE", "AWAIT_ORDER")


def test_illegal_transition_is_rejected(coordinator):
    session = coordinator.db.get_session("s_new") or coordinator.db.save_session(
        "s_new", "u_1001", State.IDLE.value, {}
    )
    with pytest.raises(TransitionError):
        coordinator.flow._save(session, State.CREATING, {})


def test_logistics_tool_timeout_degrades(make_coordinator):
    c = make_coordinator("timeout.db", simulate_tool_timeout=True)
    result = chat(c, "s_to", "我的快递到哪了 O20261115001")
    assert result.state != "ERROR"
    assert "超时" in result.reply
    assert result.meta.get("degraded") is True or "超时" in result.reply


def test_tool_argument_validation(coordinator):
    schema = next(s for s in TOOL_SCHEMAS if s["name"] == "check_return_eligibility")
    with pytest.raises(ToolError):
        validate_arguments(schema, {})
    with pytest.raises(ToolError):
        validate_arguments(schema, {"order_id": "O1", "reason": "whatever"})
    with pytest.raises(ToolError):
        coordinator.tools.call("query_order", {"order_id": 123}, {"user_id": "u_1001"})


def test_unknown_tool_rejected(coordinator):
    with pytest.raises(ToolError) as exc:
        coordinator.tools.call("drop_database", {}, {"user_id": "u_1001"})
    assert exc.value.code == "unknown_tool"
