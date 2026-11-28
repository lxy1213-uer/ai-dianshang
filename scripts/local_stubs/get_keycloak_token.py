"""从本地 Keycloak 取 id_token（Authorization Code 太重，联调先用 password grant）。

用法：
    python scripts/local_stubs/get_keycloak_token.py                  # 默认 agent01/pwd
    python scripts/local_stubs/get_keycloak_token.py boss01 pwd

取到的 token 放到 /api/auth/login 的 password 字段（username 填同名账号）：
    curl -X POST http://127.0.0.1:8000/api/auth/login \
      -H 'Content-Type: application/json' \
      -d '{"username":"agent01","password":"<id_token>"}'
"""

from __future__ import annotations

import argparse
import base64
import json
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_URL = "http://localhost:8080/realms/ecom-agent/protocol/openid-connect/token"


def _decode_payload(token: str) -> dict:
    part = token.split(".")[1]
    part += "=" * (-len(part) % 4)
    return json.loads(base64.urlsafe_b64decode(part.encode("ascii")))


def main() -> None:
    parser = argparse.ArgumentParser(description="获取本地 Keycloak id_token")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--client", default="ecom-agent")
    parser.add_argument("username", nargs="?", default="agent01")
    parser.add_argument("password", nargs="?", default="pwd")
    args = parser.parse_args()

    data = urllib.parse.urlencode({
        "grant_type": "password",
        "client_id": args.client,
        "username": args.username,
        "password": args.password,
        "scope": "openid profile roles",
    }).encode("ascii")
    req = urllib.request.Request(
        args.url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Keycloak 返回 {exc.code}: {exc.read().decode('utf-8', errors='replace')}")
    token = result.get("id_token")
    if not token:
        raise SystemExit(f"响应里没有 id_token：{result}")
    print("--- claims（未验签，仅调试用）---")
    print(json.dumps(_decode_payload(token), ensure_ascii=False, indent=2))
    print("--- id_token（给 /api/auth/login 的 password 字段）---")
    print(token)


if __name__ == "__main__":
    main()
