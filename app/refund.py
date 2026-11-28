"""真实交易退款对接（可插拔网关 + 绝不重复打款的领域编排）。

分层：
- `RefundGateway`：外部支付/退款系统的适配器。默认 `SandboxRefundGateway`（零配置、
  确定性、不向外发请求）；配置 `AGENT_REFUND_API_URL` 后走 `HttpRefundGateway`
  （HMAC-SHA256 签名 + Idempotency-Key + 超时 + 失败降级）。
- `RefundService`：领域编排。状态机前置校验（只允许 RECEIVED → REFUNDED）、
  `refund_requests` 表的原子占位与幂等短路、成功才推进状态并写 `refund_executed` 事件。

安全底线（这是资金类接口，比一般接口更严）：
- 幂等键由服务端固定为 `refund:<ticket_id>`，**不接受调用方传入**，避免换个 key 重复打款；
- 网关调用前先原子占位（PENDING + claim_id），并发第二个请求只会拿到 busy，不会再次打款；
- 网关失败 → 记 FAILED 且**不推进状态**，允许按同一工单重试；
- 状态已是 REFUNDED 但查不到 SUCCESS 流水 → 视为数据不一致，拒绝再打款（宁可人工介入）；
- 本项目仍是演示：金额由调用方给出（默认 None），不发生真实资金流转；接真实支付网关
  需要额外的对账、回调验签与事务补偿，见 README 边界说明。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import urllib.error
import urllib.request
import uuid
from typing import Any, Protocol

from .domain.events import EventName
from .domain.status import TicketStatus, parse


class RefundError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class RefundGateway(Protocol):
    gateway: str

    def refund(self, payload: dict) -> dict:
        """返回 {"ok", "refund_id", "error", "gateway"}；失败不得抛异常。"""
        ...


class SandboxRefundGateway:
    """默认沙箱：确定性 refund_id、记录调用明细，绝不产生外部请求或资金流转。"""

    gateway = "sandbox"

    def __init__(self):
        self.calls: list[dict] = []

    def refund(self, payload: dict) -> dict:
        self.calls.append(dict(payload))
        digest = hashlib.blake2b(
            payload["idempotency_key"].encode("utf-8"), digest_size=8
        ).hexdigest().upper()
        return {
            "ok": True,
            "refund_id": f"RFD-{digest}",
            "error": None,
            "gateway": self.gateway,
        }


class HttpRefundGateway:
    """真实 HTTP 退款网关适配器：签名 + 幂等键 + 超时 + 失败降级。"""

    gateway = "http"

    def __init__(self, url: str, secret: str = "", timeout: float = 5.0):
        self.url = url
        self.secret = secret
        self.timeout = timeout

    def refund(self, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        signature = hmac.new(
            self.secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        req = urllib.request.Request(
            self.url,
            data=body.encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Refund-Signature": signature,
                "Idempotency-Key": payload["idempotency_key"],
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw) if raw else {}
                ok = 200 <= resp.status < 300 and data.get("ok", True) is not False
                refund_id = data.get("refund_id") or (
                    f"HTTP-{payload['ticket_id'][-8:]}" if ok else None
                )
                return {
                    "ok": ok,
                    "refund_id": refund_id,
                    "error": None if ok else str(data.get("error", "网关返回失败")),
                    "gateway": self.gateway,
                }
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                ValueError, OSError) as exc:
            return {"ok": False, "refund_id": None,
                    "error": f"{type(exc).__name__}: {exc}"[:160], "gateway": self.gateway}


def build_refund_gateway(settings) -> RefundGateway:
    url = getattr(settings, "refund_api_url", "") or ""
    if not url:
        return SandboxRefundGateway()
    return HttpRefundGateway(
        url,
        getattr(settings, "refund_api_secret", "") or "",
        getattr(settings, "refund_timeout", 5.0),
    )


class RefundService:
    """退款编排：幂等占位 → 网关调用 → 记流水 → 推进状态 → 埋点。"""

    def __init__(self, db, tickets, gateway: RefundGateway):
        self.db = db
        self.tickets = tickets
        self.gateway = gateway

    def status(self, ticket_id: str) -> dict | None:
        return self.db.get_refund(ticket_id)

    def _result(self, ticket_id: str, row: dict, *, idempotent: bool = False) -> dict:
        return {
            "ticket_id": ticket_id,
            "status": row["status"],
            "refund_id": row.get("refund_id"),
            "gateway": row.get("gateway"),
            "idempotent": idempotent,
            "ticket": self.tickets.detail(ticket_id),
        }

    def execute(
        self,
        ticket_id: str,
        *,
        operator: str = "system",
        amount: float | None = None,
        currency: str = "CNY",
    ) -> dict:
        ticket = self.db.get_ticket(ticket_id)
        if not ticket:
            raise RefundError("not_found", f"工单不存在：{ticket_id}")

        current = parse(ticket["status"]) or TicketStatus.CREATED
        if current is TicketStatus.REFUNDED:
            row = self.db.get_refund(ticket_id)
            if row and row["status"] == "SUCCESS":
                return self._result(ticket_id, row, idempotent=True)
            # 状态与流水不一致：绝不再打一次款
            raise RefundError("inconsistent", "工单已标记已退款但缺少成功流水，拒绝重复打款，请人工核查")
        if current is not TicketStatus.RECEIVED:
            raise RefundError(
                "invalid_state",
                f"只有「{TicketStatus.RECEIVED.label}」可执行退款，当前「{current.label}」",
            )

        # 幂等键由服务端固定，调用方无法绕过
        idempotency_key = f"refund:{ticket_id}"
        claim_id = uuid.uuid4().hex
        action, row = self.db.reserve_refund(
            ticket_id, idempotency_key, claim_id, operator, self.gateway.gateway
        )
        if action == "success":
            return self._result(ticket_id, row, idempotent=True)
        if action == "busy":
            raise RefundError("in_progress", "同一工单退款正在处理中，请稍后查询结果")

        # 已占位（claimed）：现在才真正调用外部网关
        response = self.gateway.refund({
            "ticket_id": ticket_id,
            "order_id": ticket["order_id"],
            "amount": amount,
            "currency": currency,
            "idempotency_key": idempotency_key,
            "operator": operator,
        })
        if not response.get("ok"):
            self.db.finish_refund(ticket_id, claim_id, "FAILED", error=response.get("error"))
            raise RefundError("gateway_failed", f"退款网关调用失败：{response.get('error')}")

        self.db.finish_refund(
            ticket_id, claim_id, "SUCCESS", refund_id=response.get("refund_id"),
            amount=amount,
        )
        # 先落 SUCCESS 流水再推进状态：即使状态推进异常，重入也会幂等短路，不会二次打款。
        try:
            self.tickets.update_status(
                ticket_id, TicketStatus.REFUNDED.value,
                operator=operator, idempotency_key=idempotency_key,
            )
        except Exception as exc:  # 状态推进失败不掩盖「已打款」事实，交人工对账
            self.db.log_event(
                ticket["session_id"], ticket["request_id"],
                EventName.REFUND_STATUS_FAILED.value,
                {"ticket_id": ticket_id, "refund_id": response.get("refund_id"),
                 "error": type(exc).__name__},
            )
        self.db.log_event(
            ticket["session_id"], ticket["request_id"],
            EventName.REFUND_EXECUTED.value,
            {"ticket_id": ticket_id, "order_id": ticket["order_id"],
             "refund_id": response.get("refund_id"), "gateway": self.gateway.gateway,
             "operator": operator},
        )
        return self._result(ticket_id, self.db.get_refund(ticket_id))
