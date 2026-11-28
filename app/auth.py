"""演示级身份层：HMAC 签名的 user token。

两种模式：
- demo（默认，未配置 AGENT_AUTH_SECRET）：沿用显式 user_id，但所有查询接口仍强制归属校验；
- signed（配置密钥）：user_id 只从签名 token 解析，忽略请求里自带的 user_id。

注意：/api/auth/token 是演示用签发入口，本身不校验密码。真实系统应由登录态或 SSO
换取 token，本模块只负责"token 不可伪造 + 服务端按 token 归属校验"这两件事。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def issue_token(user_id: str, secret: str, ttl_seconds: int = 86400) -> str:
    payload = _b64e(
        json.dumps(
            {"uid": user_id, "exp": int(time.time()) + ttl_seconds},
            separators=(",", ":"),
        ).encode("utf-8")
    )
    signature = _b64e(hmac.new(secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256).digest())
    return payload + "." + signature


def _verify_payload(token: str | None, secret: str) -> dict | None:
    """校验签名与有效期，返回 payload；任何异常都视为未认证。"""
    if not token or not secret:
        return None
    try:
        payload, signature = token.split(".", 1)
        expected = _b64e(hmac.new(secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            return None
        data = json.loads(_b64d(payload))
        if int(data.get("exp", 0)) < int(time.time()):
            return None
        return data
    except Exception:
        return None


def verify_token(token: str | None, secret: str) -> str | None:
    """校验用户 token（typ 缺省或 user），返回 user_id；坐席 token 不能冒充用户。"""
    data = _verify_payload(token, secret)
    if not data or data.get("typ") == "staff":
        return None
    return str(data.get("uid") or "") or None


def issue_staff_token(name: str, role: str, secret: str, ttl_seconds: int = 28800) -> str:
    """签发坐席 token，携带角色；与用户 token 用 typ 字段隔离，互不冒充。"""
    payload = _b64e(
        json.dumps(
            {"sub": name, "role": role, "typ": "staff", "exp": int(time.time()) + ttl_seconds},
            separators=(",", ":"),
        ).encode("utf-8")
    )
    signature = _b64e(hmac.new(secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256).digest())
    return payload + "." + signature


def verify_staff_token(token: str | None, secret: str) -> dict | None:
    """校验坐席 token，返回 {sub, role}；用户 token 不能被当作坐席 token。"""
    data = _verify_payload(token, secret)
    if not data or data.get("typ") != "staff":
        return None
    return {"sub": str(data.get("sub") or ""), "role": str(data.get("role") or "")}


def resolve_user(request, claimed: str | None, settings) -> str | None:
    """从请求中解析身份：signed 模式只信 token，demo 模式沿用显式 user_id。"""
    if settings.signed_auth_enabled:
        header = request.headers.get("authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        return verify_token(token, settings.auth_secret)
    return (claimed or "").strip() or None
