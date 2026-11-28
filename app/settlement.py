"""退款回调验签与对账（真实资金闭环的本地骨架）。

现有 `refund.py` 只做了「请求签名 + 幂等键 + 服务端占位」，还缺三件资金安全的事：

1. **回调验签**：网关异步通知「退款成功/失败」时，必须验 HMAC 签名（含时间戳
   防重放）才可信；裸 POST 一律拒绝。
2. **对账解析**：网关给的对账文件（CSV 或 JSON）按 `refund_id` 与本地流水比对，
   列出四类差异：本地成功网关没有 / 网关成功本地没有 / 状态不一致 / 金额不一致。
3. **补偿动作**：差异只生成**建议动作清单**，绝不自动重复打款——资金类操作的
   补救必须人工确认，这是项目一贯的「宁可人工介入，不可自动二次扣款」。

全部纯标准库；验签密钥由配置注入，测试用固定密钥与假数据，零外部依赖。
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import time
from dataclasses import dataclass
from typing import Iterable


class CallbackError(ValueError):
    """回调验签/格式不合法；调用方应返回 4xx，绝不推进退款状态。"""


@dataclass(frozen=True)
class CallbackEvent:
    refund_id: str
    ticket_id: str
    status: str              # SUCCESS / FAILED
    amount: float | None
    timestamp: int

    @classmethod
    def from_payload(cls, payload: dict) -> "CallbackEvent":
        if not isinstance(payload, dict):
            raise CallbackError("回调 body 不是 JSON 对象")
        refund_id = str(payload.get("refund_id") or "").strip()
        ticket_id = str(payload.get("ticket_id") or "").strip()
        status = str(payload.get("status") or "").upper()
        if not refund_id or not ticket_id:
            raise CallbackError("回调缺少 refund_id / ticket_id")
        if status not in ("SUCCESS", "FAILED"):
            raise CallbackError("回调状态不合法")
        try:
            amount = None if payload.get("amount") is None else float(payload["amount"])
            timestamp = int(payload.get("timestamp") or 0)
        except (TypeError, ValueError) as exc:
            raise CallbackError("回调金额/时间戳不合法") from exc
        return cls(refund_id, ticket_id, status, amount, timestamp)


class RefundCallbackVerifier:
    """HMAC-SHA256 回调验签 + 时间戳窗口防重放 + 幂等去重。

    签名口径：`signature = HMAC(secret, timestamp + "." + raw_body)`，
    通过请求头 `X-Refund-Signature` / `X-Refund-Timestamp` 传递。
    """

    HEADER_SIGNATURE = "X-Refund-Signature"
    HEADER_TIMESTAMP = "X-Refund-Timestamp"

    def __init__(self, secret: str, tolerance_seconds: int = 300, now_fn=None):
        self.secret = secret
        self.tolerance = max(0, int(tolerance_seconds))
        self.now_fn = now_fn or time.time
        self._seen: dict[str, float] = {}

    def verify(self, raw_body: bytes, signature: str, timestamp: str) -> dict:
        if not self.secret:
            raise CallbackError("未配置回调密钥，拒绝所有回调")
        try:
            ts = int(timestamp)
        except (TypeError, ValueError) as exc:
            raise CallbackError("回调时间戳不合法") from exc
        if abs(int(self.now_fn()) - ts) > self.tolerance:
            raise CallbackError("回调时间戳超出允许窗口（疑似重放）")
        expected = hmac.new(
            self.secret.encode("utf-8"),
            f"{ts}.".encode("ascii") + raw_body,
            hashlib.sha256,
        ).hexdigest()
        if not signature or not hmac.compare_digest(str(signature), expected):
            raise CallbackError("回调签名校验失败")
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise CallbackError("回调 body 不是合法 JSON") from exc
        event = CallbackEvent.from_payload(payload)
        # 幂等：同一 refund_id + status 在窗口内只处理一次
        dedup_key = f"{event.refund_id}:{event.status}"
        if dedup_key in self._seen:
            raise CallbackError("重复回调（幂等去重）")
        self._seen[dedup_key] = float(self.now_fn())
        return payload


# --- 对账 -------------------------------------------------------------------
@dataclass(frozen=True)
class LedgerRow:
    refund_id: str
    ticket_id: str
    status: str
    amount: float | None = None

    @classmethod
    def from_dict(cls, row: dict) -> "LedgerRow":
        return cls(
            refund_id=str(row.get("refund_id") or "").strip(),
            ticket_id=str(row.get("ticket_id") or "").strip(),
            status=str(row.get("status") or "").upper(),
            amount=None if row.get("amount") in (None, "") else float(row["amount"]),
        )


@dataclass(frozen=True)
class Difference:
    kind: str            # local_only / gateway_only / status_mismatch / amount_mismatch
    refund_id: str
    ticket_id: str
    detail: str
    suggested_action: str


def parse_ledger(text: str, fmt: str = "auto") -> list[LedgerRow]:
    """解析网关对账文件；支持 CSV（带表头）与 JSON 数组。"""
    text = (text or "").strip()
    if not text:
        return []
    if fmt == "json" or (fmt == "auto" and text.startswith("[")):
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise ValueError("JSON 对账文件必须是数组")
        return [LedgerRow.from_dict(row) for row in rows if row.get("refund_id")]
    reader = csv.DictReader(io.StringIO(text))
    required = {"refund_id", "ticket_id", "status"}
    if not required.issubset(set(reader.fieldnames or [])):
        raise ValueError(f"CSV 对账文件缺少列：{sorted(required - set(reader.fieldnames or []))}")
    return [LedgerRow.from_dict(row) for row in reader if row.get("refund_id")]


def reconcile(
    local: Iterable[dict], gateway: Iterable[LedgerRow], *, amount_tolerance: float = 0.01
) -> list[Difference]:
    """按 refund_id 比对本地流水与网关对账单，输出差异与建议动作。

    只产出建议，不修改任何状态；重复打款必须由人工复核后执行。
    """
    local_by_id = {str(row.get("refund_id") or "").strip(): row for row in local
                   if row.get("refund_id") and str(row.get("status") or "").upper() != "FAILED"}
    gateway_by_id = {row.refund_id: row for row in gateway}
    diffs: list[Difference] = []
    for refund_id in sorted(set(local_by_id) | set(gateway_by_id)):
        lrow, grow = local_by_id.get(refund_id), gateway_by_id.get(refund_id)
        if lrow and not grow:
            diffs.append(Difference(
                "local_only", refund_id, str(lrow.get("ticket_id") or ""),
                "本地标记成功但网关对账单缺失", "人工核查网关是否漏记；禁止自动重打款",
            ))
            continue
        if grow and not lrow:
            diffs.append(Difference(
                "gateway_only", refund_id, grow.ticket_id,
                "网关有退款记录但本地无成功流水", "人工补录流水并核对工单状态",
            ))
            continue
        lstatus = str(lrow.get("status") or "").upper()
        if lstatus != grow.status:
            diffs.append(Difference(
                "status_mismatch", refund_id, grow.ticket_id,
                f"本地 {lstatus} / 网关 {grow.status}", "人工确认实际资金状态后修正本地状态",
            ))
            continue
        lamount = lrow.get("amount")
        if lamount is not None and grow.amount is not None:
            if abs(float(lamount) - grow.amount) > amount_tolerance:
                diffs.append(Difference(
                    "amount_mismatch", refund_id, grow.ticket_id,
                    f"本地 {lamount} / 网关 {grow.amount}", "人工核对金额差异，必要时发起差额退款",
                ))
    return diffs


def differences_to_dict(diffs: Iterable[Difference]) -> list[dict]:
    return [{"kind": d.kind, "refund_id": d.refund_id, "ticket_id": d.ticket_id,
             "detail": d.detail, "suggested_action": d.suggested_action} for d in diffs]
