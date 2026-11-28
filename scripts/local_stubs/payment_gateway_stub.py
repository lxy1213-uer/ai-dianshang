"""支付网关本地替身：退款 API + 回调签名模拟器 + 对账单导出。

零依赖（仅标准库）。它做的是「像真网关一样通信」，不发生任何资金流转。

    POST /refunds                 被 app.refund.HttpRefundGateway 调用
    POST /__local/send-callback   主动给宿主机 /api/refund/callback 发签名回调
    GET  /ledger.csv              生成对账文件，供 /api/refund/reconcile 上传
    GET  /records                 查看网关侧流水

启动示例见 docker-compose.local.yml。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import io
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_API_SECRET = "local-api-secret"
_CALLBACK_SECRET = "local-callback-secret"
_APP_CALLBACK = "http://host.docker.internal:8000/api/refund/callback"
_LOG_PATH = Path("/data/payment.log")
_RECORDS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _append(record: dict) -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _sign(secret: str, raw: bytes) -> str:
    """退款请求签名口径：HMAC(secret, raw_body)（与 HttpRefundGateway 一致）。"""
    return hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()


def _sign_callback(secret: str, timestamp: int, raw: bytes) -> str:
    """回调签名口径：HMAC(secret, timestamp + "." + raw_body)（与 RefundCallbackVerifier 一致）。"""
    prefix = f"{timestamp}.".encode("ascii")
    return hmac.new(secret.encode("utf-8"), prefix + raw, hashlib.sha256).hexdigest()


def _post_callback(url: str, raw: bytes, signature: str, timestamp: str) -> dict:
    req = urllib.request.Request(
        url,
        data=raw,
        headers={
            "Content-Type": "application/json",
            "X-Refund-Signature": signature,
            "X-Refund-Timestamp": timestamp,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {"status": resp.status, "body": resp.read().decode("utf-8", errors="replace")}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "body": exc.read().decode("utf-8", errors="replace")}
    except Exception as exc:
        return {"status": 0, "body": f"{type(exc).__name__}: {exc}"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def _json(self, payload, status: int = 200) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _text(self, text: str, content_type: str = "text/plain; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = 0
        return self.rfile.read(length) if length else b""

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._json({"ok": True, "service": "payment-gateway-stub",
                        "app_callback": _APP_CALLBACK})
        elif path == "/records":
            with _LOCK:
                self._json({"count": len(_RECORDS), "records": list(_RECORDS.values())})
        elif path in ("/ledger.csv", "/ledger.json"):
            with _LOCK:
                rows = list(_RECORDS.values())
            if path.endswith(".json"):
                self._json(rows)
                return
            out = io.StringIO()
            writer = csv.DictWriter(
                out, fieldnames=["refund_id", "ticket_id", "status", "amount"],
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "refund_id": row.get("refund_id", ""),
                    "ticket_id": row.get("ticket_id", ""),
                    "status": row.get("status", ""),
                    "amount": row.get("amount") if row.get("amount") is not None else "",
                })
            self._text(out.getvalue(), "text/csv; charset=utf-8")
        else:
            self._json({"detail": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/refunds":
            self._refund()
        elif path == "/__local/send-callback":
            self._send_callback()
        else:
            self._json({"detail": "not found"}, 404)

    def _refund(self) -> None:
        raw = self._body()
        signature = self.headers.get("X-Refund-Signature", "")
        if not signature or not hmac.compare_digest(signature, _sign(_API_SECRET, raw)):
            _append({"event": "refund_rejected", "reason": "bad signature"})
            self._json({"ok": False, "error": "bad signature"}, 401)
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
        except ValueError:
            self._json({"ok": False, "error": "invalid JSON"}, 400)
            return
        key = str(payload.get("idempotency_key") or "")
        if not key:
            self._json({"ok": False, "error": "missing idempotency_key"}, 400)
            return
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12].upper()
        refund_id = f"RFD-{digest}"
        with _LOCK:
            duplicate = refund_id in _RECORDS
            if not duplicate:
                _RECORDS[refund_id] = {
                    "refund_id": refund_id,
                    "ticket_id": payload.get("ticket_id", ""),
                    "status": "SUCCESS",
                    "amount": payload.get("amount"),
                    "idempotency_key": key,
                }
        _append({"event": "refund_accepted", "refund_id": refund_id,
                 "duplicate": duplicate, "payload": payload})
        if "simulate=fail" in self.path:
            self._json({"ok": False, "error": "simulated gateway failure",
                        "refund_id": refund_id}, 500)
            return
        self._json({"ok": True, "refund_id": refund_id, "duplicate": duplicate})

    def _send_callback(self) -> None:
        raw_in = self._body()
        try:
            spec = json.loads(raw_in.decode("utf-8")) if raw_in else {}
        except ValueError:
            self._json({"error": "invalid JSON"}, 400)
            return
        refund_id = str(spec.get("refund_id") or "RFD-LOCAL-DEMO")
        ticket_id = str(spec.get("ticket_id") or "T-LOCAL-DEMO")
        status = str(spec.get("status") or "SUCCESS").upper()
        amount = spec.get("amount", 99.5)
        tamper = str(spec.get("tamper") or "none").lower()
        ts = spec.get("timestamp") or int(time.time())
        payload = {"refund_id": refund_id, "ticket_id": ticket_id,
                   "status": status, "amount": amount, "timestamp": int(ts)}
        signed_raw = _json_bytes(payload)
        signature = _sign_callback(_CALLBACK_SECRET, int(ts), signed_raw)
        out_raw, out_sig, out_ts = signed_raw, signature, str(int(ts))
        if tamper == "body":
            out_raw = _json_bytes({**payload, "amount": 0.01})
        elif tamper == "timestamp":
            out_ts = str(int(ts) + 1000)
        elif tamper == "signature":
            out_sig = "0" * 64
        attempts = [_post_callback(_APP_CALLBACK, out_raw, out_sig, out_ts)]
        if tamper == "replay":
            attempts.append(_post_callback(_APP_CALLBACK, signed_raw, signature, str(int(ts))))
        self._json({"tamper": tamper, "url": _APP_CALLBACK,
                    "sent": payload, "attempts": attempts})


def main() -> None:
    global _API_SECRET, _CALLBACK_SECRET, _APP_CALLBACK, _LOG_PATH
    parser = argparse.ArgumentParser(description="支付网关本地替身")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--api-secret", default=_API_SECRET)
    parser.add_argument("--callback-secret", default=_CALLBACK_SECRET)
    parser.add_argument("--app-callback", default=_APP_CALLBACK)
    parser.add_argument("--log", default=str(_LOG_PATH))
    args = parser.parse_args()
    _API_SECRET = args.api_secret
    _CALLBACK_SECRET = args.callback_secret
    _APP_CALLBACK = args.app_callback
    _LOG_PATH = Path(args.log)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"payment gateway stub listening on :{args.port}, callback={_APP_CALLBACK}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
