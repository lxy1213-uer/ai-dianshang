"""OIDC id_token 验签（纯标准库，零硬依赖）。

当前 `identity.py` 的 `sso` 模式是 HMAC 自签自验的**沙箱**，不是真 IdP。
本模块实现真正的 OIDC 登录校验：前端完成授权码流程后，把 IdP 签发的
`id_token`（JWT, RS256）放到登录请求的 `credential` 字段，服务端：

1. 取 JWKS：优先用配置的 `jwks_url`，否则从 `issuer` 拉 OIDC discovery
   文档里的 `jwks_uri`；HTTP 客户端可注入，测试用假客户端零网络。
2. 验签：按 `kid` 找 key，用纯 Python 实现 RSA PKCS#1 v1.5 + SHA-256
   （JWK 只给 n/e，验签只需公钥，不需要 cryptography）。
3. 验 claims：`iss` / `aud` / `exp` / `nbf` 全部符合才认，任何异常都拒绝。

设计纪律：与项目其余可插拔组件一致——未知配置一律降级/拒绝，绝不
「验证失败也放行」；密钥与 token 不写日志；不引入 cryptography / PyJWT
这类硬依赖，零配置演示与现有测试不受影响。生产若要换用 PyJWT，只需
实现同一 `JwksSource` + `verify_id_token` 契约。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

# PKCS#1 v1.5 的 DigestInfo 前缀：SHA-256 = 3031300d060960864801650304020105000420
_SHA256_DIGEST_INFO_PREFIX = bytes.fromhex("3031300d060960864801650304020105000420")
_DEFAULT_DISCOVERY_SUFFIX = "/.well-known/openid-configuration"
_ALLOWED_ALGS = {"RS256"}


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _int_from_b64(text: str) -> int:
    return int.from_bytes(_b64d(text), "big")


class JwksSource(Protocol):
    def fetch(self) -> dict:
        """返回 JWKS JSON（{"keys": [...]}）；失败应抛异常，由上层拒绝登录。"""
        ...


class HttpJwksSource:
    """默认 JWKS 源：urllib GET + 缓存，URL 不写日志。"""

    def __init__(self, url: str, timeout: float = 5.0, cache_seconds: float = 300.0):
        self.url = url
        self.timeout = timeout
        self.cache_seconds = max(0.0, float(cache_seconds))
        self._cached: dict | None = None
        self._fetched_at = 0.0

    def fetch(self) -> dict:
        now = time.monotonic()
        if self._cached is not None and now - self._fetched_at < self.cache_seconds:
            return self._cached
        with urllib.request.urlopen(self.url, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("keys"), list):
            raise ValueError("JWKS 格式不合法")
        self._cached = data
        self._fetched_at = now
        return data


class StaticJwksSource:
    """注入式 JWKS 源：测试与离线联调用，零网络。"""

    def __init__(self, jwks: dict):
        self.jwks = jwks

    def fetch(self) -> dict:
        return self.jwks


@dataclass(frozen=True)
class IdTokenClaims:
    sub: str
    issuer: str
    audience: str | tuple[str, ...]
    expires_at: int
    not_before: int
    raw: dict

    @property
    def username(self) -> str:
        return str(self.raw.get("preferred_username") or self.raw.get("email") or self.sub)

    @property
    def role(self) -> str:
        """role 是可选 claim；缺省给 viewer，交由坐席 RBAC 进一步判定。"""
        value = self.raw.get("role") or self.raw.get("app_role")
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        return str(value or "viewer")


def discover_jwks_url(issuer: str, timeout: float = 5.0) -> str:
    """从 OIDC discovery 文档取 jwks_uri；issuer 末尾斜杠统一去掉。"""
    base = (issuer or "").rstrip("/")
    if not base:
        raise ValueError("OIDC issuer 未配置")
    url = base + _DEFAULT_DISCOVERY_SUFFIX
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        doc = json.loads(resp.read().decode("utf-8"))
    jwks_uri = doc.get("jwks_uri")
    if not isinstance(jwks_uri, str) or not jwks_uri.startswith("https://"):
        raise ValueError("OIDC discovery 未返回合法 jwks_uri")
    return jwks_uri


def _rsa_pkcs1v15_verify(n: int, e: int, message: bytes, signature: bytes) -> bool:
    """RSASSA-PKCS1-v1_5 + SHA-256 验签，纯 Python 实现。

    JWK 只暴露 n/e，验签是公钥运算 `s^e mod n`，不需要私钥或第三方库。
    严格校验整块 encoded message：0x00 01 FF...FF 00 || DigestInfo(sha256)。
    """
    if n <= 0 or e <= 1 or not signature:
        return False
    k = (n.bit_length() + 7) // 8
    if len(signature) != k:
        return False
    s = int.from_bytes(signature, "big")
    if s >= n:
        return False
    em = pow(s, e, n).to_bytes(k, "big")
    digest = hashlib.sha256(message).digest()
    expected_tail = _SHA256_DIGEST_INFO_PREFIX + digest
    if len(em) < len(expected_tail) + 11 or em[0] != 0x00 or em[1] != 0x01:
        return False
    # PS 必须全 0xFF，直到 0x00 分隔符
    sep = em.find(b"\x00", 2)
    if sep < 10 or any(b != 0xFF for b in em[2:sep]):
        return False
    return hmac.compare_digest(em[sep + 1:], expected_tail)


def verify_id_token(
    token: str,
    *,
    jwks: JwksSource,
    issuer: str,
    audience: str,
    leeway_seconds: int = 60,
    now: float | None = None,
) -> IdTokenClaims:
    """验 id_token 的签名与关键 claims；任何失败都抛 ValueError。"""
    parts = (token or "").split(".")
    if len(parts) != 3:
        raise ValueError("id_token 格式不合法")
    try:
        header = json.loads(_b64d(parts[0]))
        payload = json.loads(_b64d(parts[1]))
    except Exception as exc:
        raise ValueError("id_token 解析失败") from exc
    if not isinstance(header, dict) or not isinstance(payload, dict):
        raise ValueError("id_token 解析失败")
    if header.get("alg") not in _ALLOWED_ALGS:
        raise ValueError("不支持的签名算法（仅接受 RS256）")

    keys = jwks.fetch().get("keys") or []
    kid = header.get("kid")
    matched = [k for k in keys if k.get("kty") == "RSA" and (not kid or k.get("kid") == kid)]
    if len(matched) != 1:
        raise ValueError("JWKS 中找不到唯一匹配的 RSA 公钥")
    key = matched[0]
    try:
        n, e = _int_from_b64(key["n"]), _int_from_b64(key["e"])
    except Exception as exc:
        raise ValueError("JWK 公钥参数不合法") from exc
    signed = f"{parts[0]}.{parts[1]}".encode("ascii")
    if not _rsa_pkcs1v15_verify(n, e, signed, _b64d(parts[2])):
        raise ValueError("id_token 签名校验失败")

    current = int(time.time() if now is None else now)
    if str(payload.get("iss") or "") != (issuer or "").rstrip("/"):
        raise ValueError("id_token issuer 不匹配")
    aud = payload.get("aud")
    audiences = (aud,) if isinstance(aud, str) else tuple(aud or [])
    if audience and audience not in audiences:
        raise ValueError("id_token audience 不匹配")
    if int(payload.get("exp") or 0) + int(leeway_seconds) < current:
        raise ValueError("id_token 已过期")
    if int(payload.get("nbf") or 0) > current + int(leeway_seconds):
        raise ValueError("id_token 尚未生效")
    sub = str(payload.get("sub") or "").strip()
    if not sub:
        raise ValueError("id_token 缺少 sub")
    return IdTokenClaims(
        sub=sub, issuer=str(payload["iss"]), audience=audiences,
        expires_at=int(payload["exp"]), not_before=int(payload.get("nbf") or 0),
        raw=payload,
    )
