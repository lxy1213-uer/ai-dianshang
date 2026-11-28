"""企微 webhook 本地替身：接收通知、落盘、提供调试查询。

零依赖（仅标准库），返回企微格式 {"errcode": 0} 以通过 WecomNotifier 的成功判定。

    启动：python scripts/local_stubs/webhook_sink.py --port 8081 --log /data/webhook.log
    查看：curl http://127.0.0.1:8081/messages
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_LOCK = threading.Lock()
_LOG_PATH = Path("/data/webhook.log")


def _append(record: dict) -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_all() -> list[dict]:
    if not _LOG_PATH.exists():
        return []
    with _LOCK:
        lines = _LOG_PATH.read_text(encoding="utf-8").splitlines()
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        # 容器日志已经够多，替身不再打印每条访问，避免刷屏。
        return

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] == "/health":
            self._json({"ok": True, "service": "webhook-sink"})
        elif self.path.split("?", 1)[0] == "/messages":
            self._json({"count": len(_read_all()), "messages": _read_all()[-50:]})
        else:
            self._json({"detail": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else None
        except ValueError:
            body = {"raw": raw.decode("utf-8", errors="replace")}
        _append({
            "ts": int(time.time()),
            "path": self.path,
            "headers": {k: v for k, v in self.headers.items()
                        if k.lower() not in ("content-length", "host")},
            "body": body,
        })
        # 企微机器人成功响应格式，WecomNotifier._http_post 会检查 errcode == 0。
        self._json({"errcode": 0, "errmsg": "ok"})


def main() -> None:
    global _LOG_PATH
    parser = argparse.ArgumentParser(description="企微 webhook 本地替身")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--log", default="/data/webhook.log")
    args = parser.parse_args()
    _LOG_PATH = Path(args.log)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"webhook sink listening on :{args.port}, log={_LOG_PATH}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
