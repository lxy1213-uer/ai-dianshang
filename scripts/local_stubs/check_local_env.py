"""本地替身环境自检：逐项探测六项依赖，并给出 .env 配置提示。

    python scripts/local_stubs/check_local_env.py

不修改任何文件；只报告连通性和缺失的可选依赖。
"""

from __future__ import annotations

import importlib.util
import socket
import urllib.request

CHECKS = [
    ("Redis", "tcp", "127.0.0.1", 6379),
    ("Postgres", "tcp", "127.0.0.1", 5432),
    ("Keycloak", "http", "http://localhost:8080/realms/ecom-agent"),
    ("Ollama", "http", "http://127.0.0.1:11434/api/tags"),
    ("企微 webhook 替身", "http", "http://127.0.0.1:8081/health"),
    ("支付网关替身", "http", "http://127.0.0.1:8082/health"),
    ("应用本体", "http", "http://127.0.0.1:8000/api/health"),
]


def tcp_ok(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=4) as resp:
            return 200 <= resp.status < 400
    except Exception:
        return False


def main() -> None:
    print(f"{'服务':<20}{'状态':<8}地址")
    print("-" * 64)
    for name, kind, *rest in CHECKS:
        ok = tcp_ok(rest[0], rest[1]) if kind == "tcp" else http_ok(rest[0])
        target = f"{rest[0]}:{rest[1]}" if kind == "tcp" else rest[0]
        print(f"{name:<20}{'OK' if ok else 'DOWN':<8}{target}")
    print("-" * 64)
    for package, hint in [
        ("redis", "Redis 多实例限流：pip install redis"),
        ("psycopg", "Postgres 租约锁：pip install 'psycopg[binary]'"),
    ]:
        installed = importlib.util.find_spec(package) is not None
        print(f"{package:<20}{'OK' if installed else 'MISSING':<8}{hint}")
    print("\n配置模板：deploy/local/.env.local.example")
    print("应用配置：python scripts/local_stubs/apply_local_env.py --apply")


if __name__ == "__main__":
    main()
