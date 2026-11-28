"""本地替身冒烟测试：不依赖 Docker，直接在线程里起两个假服务。"""

from __future__ import annotations

import hashlib
import hmac
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

STUBS = Path(__file__).resolve().parents[1] / "scripts" / "local_stubs"
sys.path.insert(0, str(STUBS))

import payment_gateway_stub as pay  # noqa: E402
import webhook_sink as hook  # noqa: E402

from app.settlement import CallbackError, RefundCallbackVerifier  # noqa: E402


def _request(url: str, payload: dict | None = None, headers: dict | None = None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _post_raw(url: str, raw: bytes, headers: dict):
    req = urllib.request.Request(url, data=raw, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def test_webhook_sink_records_and_returns_wecom_ok(tmp_path):
    hook._LOG_PATH = tmp_path / "webhook.log"
    server, port = _serve(hook.Handler)
    try:
        status, body = _request(f"http://127.0.0.1:{port}/webhook", {"msgtype": "markdown"})
        assert status == 200 and body["errcode"] == 0
        status, shown = _request(f"http://127.0.0.1:{port}/messages")
        assert status == 200 and shown["count"] == 1
        assert shown["messages"][0]["body"]["msgtype"] == "markdown"
    finally:
        server.shutdown()
        server.server_close()


def test_payment_gateway_hmac_and_idempotency(tmp_path):
    pay._API_SECRET = "test-api"
    pay._LOG_PATH = tmp_path / "payment.log"
    pay._RECORDS.clear()
    server, port = _serve(pay.Handler)
    try:
        url = f"http://127.0.0.1:{port}/refunds"
        good = {"ticket_id": "T-1", "amount": 10.5, "idempotency_key": "refund:T-1"}
        raw = json.dumps(good, sort_keys=True, separators=(",", ":")).encode("utf-8")
        sig = hmac.new(b"test-api", raw, hashlib.sha256).hexdigest()
        status, body = _post_raw(url, raw, {"X-Refund-Signature": sig,
                                            "Content-Type": "application/json"})
        assert status == 200 and body["ok"] and body["refund_id"].startswith("RFD-")
        first_id = body["refund_id"]
        status, again = _post_raw(url, raw, {"X-Refund-Signature": sig,
                                             "Content-Type": "application/json"})
        assert status == 200 and again["duplicate"] and again["refund_id"] == first_id
        status, bad = _post_raw(url, raw, {"X-Refund-Signature": "0" * 64,
                                           "Content-Type": "application/json"})
        assert status == 401 and not bad["ok"]
        _, ledger = _request(f"http://127.0.0.1:{port}/ledger.json")
        assert ledger[0]["refund_id"] == first_id
        assert ledger[0]["amount"] == 10.5
    finally:
        server.shutdown()
        server.server_close()


def test_signed_callback_and_tamper_paths(tmp_path):
    pay._CALLBACK_SECRET = "test-callback"
    pay._LOG_PATH = tmp_path / "payment.log"
    verifier = RefundCallbackVerifier("test-callback", tolerance_seconds=300)
    seen: list[dict] = []

    class AppHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length)
            try:
                verifier.verify(raw, self.headers.get("X-Refund-Signature", ""),
                                self.headers.get("X-Refund-Timestamp", ""))
                status, body = 200, {"accepted": True, "requires_manual_review": True}
            except CallbackError:
                status, body = 401, {"detail": "rejected"}
            seen.append({"status": status})
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    app_server, app_port = _serve(AppHandler)
    pay_server, pay_port = _serve(pay.Handler)
    pay._APP_CALLBACK = f"http://127.0.0.1:{app_port}/api/refund/callback"
    try:
        url = f"http://127.0.0.1:{pay_port}/__local/send-callback"
        status, body = _request(url, {"refund_id": "RFD-OK", "ticket_id": "T-1", "amount": 9.5})
        assert status == 200 and body["attempts"][0]["status"] == 200

        status, tampered = _request(url, {"refund_id": "RFD-BAD", "ticket_id": "T-2",
                                          "tamper": "body"})
        assert status == 200 and tampered["attempts"][0]["status"] == 401

        status, replay = _request(url, {"refund_id": "RFD-REPLAY", "ticket_id": "T-3",
                                        "tamper": "replay"})
        assert status == 200
        assert [a["status"] for a in replay["attempts"]] == [200, 401]
    finally:
        app_server.shutdown()
        app_server.server_close()
        pay_server.shutdown()
        pay_server.server_close()
