"""身份源协议：账号表（演示）与 SSO 签名码（沙箱）两种实现。

`app/auth.py` 只负责「token 不可伪造 + 服务端按 token 归属校验」，不负责
「你是谁」。本模块把后者抽成 `IdentityProvider`：真实部署注入 OIDC/SAML
实现即可，聊天、工单、坐席逻辑都不用改。

零配置默认仍是账号表登录（与既有行为逐字节一致）；`AGENT_IDP_PROVIDER=sso`
时改用签名码，`password` 字段传由 IdP 签发的 code。这里不引入外部 IdP 依赖，
沙箱签发函数 `issue_sso_code()` 用于联调与测试，生产应替换为真实 IdP 校验。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Protocol

from .auth import _b64d, _b64e
from .authz import parse_accounts
from .config import Settings

import hashlib
import hmac


@dataclass(frozen=True)
class Identity:
    user_id: str
    username: str
    role: str


class IdentityProvider(Protocol):
    def authenticate(self, username: str, credential: str) -> Identity | None: ...


class AccountIdentityProvider:
    """AGENT_STAFF_ACCOUNTS 账号表；保留既有演示登录语义。"""

    def __init__(self, accounts_raw: str = ""):
        self.accounts = parse_accounts(accounts_raw)

    def authenticate(self, username: str, credential: str) -> Identity | None:
        account = self.accounts.get((username or "").strip())
        if not account or account["password"] != credential:
            return None
        return Identity(
            user_id=account.get("user_id") or "u_1001",
            username=(username or "").strip(),
            role=account["role"],
        )


def issue_sso_code(user_id: str, secret: str, role: str = "agent", ttl_seconds: int = 300) -> str:
    """沙箱 IdP：签发一次性登录码（HMAC + 过期时间），供协议联调使用。"""
    payload = _b64e(
        json.dumps(
            {"uid": user_id, "role": role, "exp": int(time.time()) + ttl_seconds},
            separators=(",", ":"),
        ).encode("utf-8")
    )
    signature = _b64e(hmac.new(secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256).digest())
    return payload + "." + signature


class SsoCodeIdentityProvider:
    """校验外部 IdP 传来的签名码；不认识 / 过期 / 串改一律拒绝。"""

    def __init__(self, secret: str):
        self.secret = secret

    def authenticate(self, username: str, credential: str) -> Identity | None:
        if not self.secret or not credential:
            return None
        try:
            payload, signature = credential.split(".", 1)
            expected = _b64e(
                hmac.new(self.secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256).digest()
            )
            if not hmac.compare_digest(signature, expected):
                return None
            data = json.loads(_b64d(payload))
            if int(data.get("exp", 0)) < int(time.time()):
                return None
            user_id = str(data.get("uid") or "").strip()
            if not user_id:
                return None
            if (username or "").strip() and (username or "").strip() != user_id:
                return None
            return Identity(user_id=user_id, username=user_id, role=str(data.get("role") or "viewer"))
        except Exception:
            return None


class OidcIdentityProvider:
    """真正对接 OIDC IdP：credential 是 RS256 id_token（JWT）。

    配置 `AGENT_IDP_PROVIDER=oidc` 时启用；`issuer` / `audience` 必填，
    `jwks_url` 留空则从 `<issuer>/.well-known/openid-configuration` 发现。
    HTTP 拉取可注入（测试用假 JWKS，不依赖真实 IdP 或网络）。
    """

    def __init__(
        self,
        issuer: str,
        audience: str,
        jwks_url: str = "",
        *,
        jwks_source=None,
        http_timeout: float = 5.0,
    ):
        from .oidc import HttpJwksSource, discover_jwks_url, verify_id_token

        if not issuer or not audience:
            raise ValueError("OIDC 身份源需要同时配置 issuer 与 audience")
        self.issuer = issuer
        self.audience = audience
        self._verify = verify_id_token
        if jwks_source is not None:
            self.jwks = jwks_source
        else:
            url = jwks_url or discover_jwks_url(issuer, timeout=http_timeout)
            self.jwks = HttpJwksSource(url, timeout=http_timeout)

    def authenticate(self, username: str, credential: str) -> Identity | None:
        try:
            claims = self._verify(
                credential, jwks=self.jwks, issuer=self.issuer, audience=self.audience
            )
        except Exception:
            return None
        return Identity(
            user_id=claims.sub, username=claims.username, role=claims.role
        )


def build_identity_provider(settings: Settings) -> IdentityProvider:
    mode = str(getattr(settings, "idp_provider", "account") or "account").lower()
    if mode == "oidc":
        return OidcIdentityProvider(
            getattr(settings, "oidc_issuer", ""),
            getattr(settings, "oidc_audience", ""),
            getattr(settings, "oidc_jwks_url", ""),
        )
    if mode == "sso":
        return SsoCodeIdentityProvider(settings.auth_secret)
    return AccountIdentityProvider(settings.staff_accounts)
