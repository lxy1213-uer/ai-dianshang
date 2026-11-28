"""Agent 工具层：JSON Schema 定义 + 服务端校验 + 统一的异常语义（第 4 周）。

关键点：模型只负责"选工具、填参数"，参数合法性、订单归属、用户确认一律由后端校验。
"""

from __future__ import annotations

import time
from typing import Any, Callable

from ..config import Settings
from ..domain.orders import Order, evaluate_return
from ..domain.rag import PolicyIndex
from ..domain.tickets import TicketError, TicketService


class ToolError(Exception):
    code = "tool_error"

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ToolValidationError(ToolError):
    def __init__(self, message: str):
        super().__init__("invalid_arguments", message)


class ToolTimeout(ToolError):
    def __init__(self, message: str = "工具调用超时"):
        super().__init__("timeout", message)


TOOL_SCHEMAS: list[dict] = [
    {
        "name": "query_order",
        "description": "按订单号查询订单基础信息。后端会校验订单是否属于当前登录用户。",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "订单号，例如 O20261115001"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "list_my_orders",
        "description": "列出当前登录用户自己的订单（不接收 user_id 参数，身份由服务端确定）。可按需按状态过滤。",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "返回条数，最多 50"},
                "status": {"type": "string", "description": "可选状态过滤：paid/shipped/delivered/refunded/closed"},
            },
            "required": [],
        },
    },
    {
        "name": "query_logistics",
        "description": "查询订单的物流轨迹与最新状态。",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "check_return_eligibility",
        "description": "判断订单是否符合退货条件，返回是否可退、剩余天数、截止日期与依据条款。",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "reason": {"type": "string", "enum": ["no_reason", "quality"]},
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "检索售后政策条款，返回带条款编号的引用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "create_return_ticket",
        "description": "在用户明确确认后创建售后工单。需携带服务端下发的确认令牌，幂等调用只返回一个工单。",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "confirm_token": {"type": "string"},
                "reason": {"type": "string", "enum": ["no_reason", "quality"]},
            },
            "required": ["order_id", "confirm_token"],
        },
    },
    {
        "name": "query_ticket_status",
        "description": "查询售后工单的真实状态、更新时间与下一步说明。仅返回库中已记录的事实，不含预计到账时间或金额。",
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string", "description": "工单号，例如 RT20261125001ABC；不填则用当前会话最近一张工单"},
            },
            "required": [],
        },
    },
]

_TYPE_MAP = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def validate_arguments(schema: dict, args: dict) -> None:
    params = schema.get("parameters", {})
    properties = params.get("properties", {})
    if not isinstance(args, dict):
        raise ToolValidationError("参数必须是对象")
    for key in params.get("required", []):
        if key not in args or args[key] in (None, ""):
            raise ToolValidationError(f"缺少必填参数：{key}")
    for key, value in args.items():
        if key not in properties:
            raise ToolValidationError(f"不支持的参数：{key}")
        expected = properties[key].get("type")
        py_type = _TYPE_MAP.get(expected)
        if py_type and not isinstance(value, py_type):
            raise ToolValidationError(f"参数 {key} 类型应为 {expected}")
        allowed = properties[key].get("enum")
        if allowed and value not in allowed:
            raise ToolValidationError(f"参数 {key} 取值必须是 {allowed}")
        if expected == "integer" and isinstance(value, int):
            if "minimum" in properties[key] and value < properties[key]["minimum"]:
                raise ToolValidationError(f"参数 {key} 不能小于 {properties[key]['minimum']}")
            if "maximum" in properties[key] and value > properties[key]["maximum"]:
                raise ToolValidationError(f"参数 {key} 不能大于 {properties[key]['maximum']}")


class ToolRegistry:
    def __init__(
        self,
        orders: dict[str, Order],
        logistics: dict,
        index: PolicyIndex,
        settings: Settings,
        tickets: TicketService,
        db=None,
        on_call: Callable[[str, dict, bool, float, str | None], None] | None = None,
        order_source: Any = None,
    ):
        """on_call：真实编排的观测钩子，每次工具调用都上报名称/成败/真实耗时。"""
        self.orders = orders
        self.logistics = logistics
        self.index = index
        self.settings = settings
        self.tickets = tickets
        self.db = db
        self.on_call = on_call
        self.order_source = order_source
        self._handlers: dict[str, Callable[[dict, dict], dict]] = {
            "query_order": self._query_order,
            "list_my_orders": self._list_my_orders,
            "query_logistics": self._query_logistics,
            "check_return_eligibility": self._check_eligibility,
            "search_policy": self._search_policy,
            "create_return_ticket": self._create_ticket,
            "query_ticket_status": self._query_ticket_status,
        }

    @property
    def schemas(self) -> list[dict]:
        return TOOL_SCHEMAS

    def call(self, name: str, args: dict, ctx: dict) -> dict:
        """真实执行工具，并把调用链上报给编排层（成功与失败都记录）。

        耗时用 perf_counter 实测，而不是轨迹里写死的 0。
        """
        started = time.perf_counter()
        error_code: str | None = None
        try:
            schema = next((item for item in TOOL_SCHEMAS if item["name"] == name), None)
            if schema is None:
                raise ToolError("unknown_tool", f"不存在的工具：{name}")
            validate_arguments(schema, args)
            handler = self._handlers[name]
            try:
                result = handler(args, ctx)
            except ToolError:
                raise
            except TicketError as exc:
                raise ToolError(exc.code, exc.message) from exc
            if self.db is not None:
                self.db.log_event(
                    ctx.get("session_id"),
                    ctx.get("request_id"),
                    "tool_call",
                    {"tool": name, "args": args, "ok": True},
                )
            return result
        except ToolError as exc:
            error_code = exc.code
            raise
        except Exception:
            error_code = "unexpected"
            raise
        finally:
            if self.on_call is not None:
                self.on_call(
                    name,
                    args,
                    error_code is None,
                    round((time.perf_counter() - started) * 1000, 3),
                    error_code,
                )

    # ---------- 具体工具实现 ----------
    def _assert_source_ready(self) -> None:
        error = getattr(self.order_source, "error", "")
        if error:
            raise ToolError("order_source_unavailable", f"订单源暂不可用：{error}")

    def _load_owned_order(self, order_id: str, ctx: dict) -> Order:
        self._assert_source_ready()
        order = self.orders.get((order_id or "").strip().upper())
        if order is None:
            raise ToolError("not_found", "查不到这个订单号，请核对后重新提供")
        if order.user_id != ctx.get("user_id"):
            raise ToolError("forbidden", "该订单不属于当前账号，无法查看或办理")
        return order

    def _query_order(self, args: dict, ctx: dict) -> dict:
        order = self._load_owned_order(args["order_id"], ctx)
        return {"order": order.to_public()}

    def _list_my_orders(self, args: dict, ctx: dict) -> dict:
        """只按服务端身份过滤；调用方即使传 user_id 也会在参数校验阶段被拒绝。"""
        self._assert_source_ready()
        user_id = ctx.get("user_id") or ""
        if not user_id:
            raise ToolError("forbidden", "缺少登录身份，无法查看订单列表")
        rows = [order for order in self.orders.values() if order.user_id == user_id]
        status = (args.get("status") or "").strip()
        if status:
            rows = [order for order in rows if order.status == status]
        rows.sort(
            key=lambda order: (order.signed_at.isoformat() if order.signed_at else "", order.order_id),
            reverse=True,
        )
        limit = max(1, min(int(args.get("limit", 50)), 50))
        return {"user_id": user_id, "orders": [order.to_public() for order in rows[:limit]]}

    def _query_logistics(self, args: dict, ctx: dict) -> dict:
        order = self._load_owned_order(args["order_id"], ctx)
        if self.settings.simulate_tool_timeout:
            raise ToolTimeout("物流服务超时，已触发降级")
        info = self.logistics.get(order.order_id)
        if not info:
            return {
                "order_id": order.order_id,
                "status": order.status_label,
                "message": "暂无物流轨迹，可能还未揽收",
            }
        return {
            "order_id": order.order_id,
            "company": info.get("company", ""),
            "tracking_no": info.get("tracking_no", ""),
            "status": info.get("status", ""),
            "nodes": info.get("nodes", [])[-3:],
        }

    def _check_eligibility(self, args: dict, ctx: dict) -> dict:
        order = self._load_owned_order(args["order_id"], ctx)
        eligibility = evaluate_return(
            order,
            self.settings.current_date(),
            args.get("reason", "no_reason"),
            no_reason_days=self.settings.return_window_days,
            quality_days=self.settings.quality_window_days,
        )
        return {"order": order.to_public(), "eligibility": eligibility.to_dict()}

    def _search_policy(self, args: dict, ctx: dict) -> dict:
        return self.index.answer(args["query"], top_k=int(args.get("top_k", 2)))

    def _create_ticket(self, args: dict, ctx: dict) -> dict:
        order = self._load_owned_order(args["order_id"], ctx)

        # 服务端三重校验：状态合法 + 令牌是当前有效令牌 + 订单与令牌一致
        session = ctx.get("session") or {}
        slots = session.get("slots") or {}
        if session.get("state") not in ("AWAIT_CONFIRM", "CREATING"):
            raise ToolError("invalid_state", "当前会话不在确认环节，无法建单")
        if slots.get("confirm_token") != args["confirm_token"]:
            raise ToolError("stale_confirmation", "该确认已失效（订单可能已更换），请重新核对后确认")
        if slots.get("order_id") != order.order_id:
            raise ToolError("order_mismatch", "确认的订单与当前订单不一致，请重新确认")

        ticket, created = self.tickets.create(
            session_id=ctx.get("session_id", ""),
            user_id=ctx.get("user_id", ""),
            order_id=order.order_id,
            confirm_token=args["confirm_token"],
            reason=args.get("reason", "no_reason"),
            request_id=ctx.get("request_id"),
            confirmed=bool(ctx.get("confirmed")),
            attempt_no=int(ctx.get("attempt_no", 1)),
        )
        return {"ticket": ticket, "created": created}

    def _query_ticket_status(self, args: dict, ctx: dict) -> dict:
        """查工单真实状态：默认取当前会话最近一张，也可指定工单号（校验归属）。"""
        ticket_id = (args.get("ticket_id") or "").strip()
        session_id = ctx.get("session_id", "")
        user_id = ctx.get("user_id", "")
        if ticket_id:
            ticket = self.db.get_ticket(ticket_id) if self.db else None
            if not ticket or ticket["user_id"] != user_id:
                raise ToolError("not_found", "查不到这张工单，请核对工单号")
        else:
            tickets = self.db.list_tickets(session_id=session_id, user_id=user_id, limit=1) if self.db else []
            ticket = tickets[0] if tickets else None
        if not ticket:
            raise ToolError("not_found", "当前还没有售后工单，办理完成后即可查询进度")
        view = self.tickets.detail(ticket["ticket_id"])
        return {"ticket": {
            "ticket_id": view["ticket_id"],
            "status": view["status"],
            "status_label": view["status_label"],
            "updated_at": view["updated_at"],
            "next_step": view["next_step"],
        }}
