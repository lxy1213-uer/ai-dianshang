"""OIDC 身份源：纯标准库 RS256 验签 + 假 IdP，零网络零外部依赖。"""

import base64
import hashlib
import json
import time

import pytest

from app.oidc import (
    IdTokenClaims,
    StaticJwksSource,
    _rsa_pkcs1v15_verify,
    discover_jwks_url,
    verify_id_token,
)

# 测试专用固定 1024-bit RSA 密钥对（仅用于验签自测，绝不用于生产）
N_INT = 74403846017565288313067730518049027329323646381999316218501720924448077061848964658700741608363159031126857081500578212866209590940778451974983437804601011578429314482282715767301966467533409153286899862644704633244381093096522367172133135542659096828743391286661161431484009575743722039318976809214725868003
D_INT = 23180458794065154901448295187718770160521998440997635511991213174777314150004305360351431132941071167394726153014004393705880150568820886406536946047184691043871117653296196830071116965980955147771428885471772360258107102354958655122140946284962456876764539142744329440142763238379748588965076408621732245345
N_B64 = base64.urlsafe_b64encode(N_INT.to_bytes((N_INT.bit_length() + 7) // 8, "big")).decode().rstrip("=")
JWKS = {"keys": [{"kty": "RSA", "kid": "test-key", "alg": "RS256", "n": N_B64, "e": "AQAB"}]}
ISSUER = "https://idp.example.com"
AUDIENCE = "ecom-agent"
DIGEST_INFO = bytes.fromhex("3031300d060960864801650304020105000420")


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _sign(payload: dict, *, n=N_INT, d=D_INT, alg="RS256", kid="test-key") -> str:
    header = {"alg": alg, "typ": "JWT", "kid": kid}
    h = _b64e(json.dumps(header, separators=(",", ":")).encode())
    p = _b64e(json.dumps(payload, separators=(",", ":")).encode())
    signed = f"{h}.{p}".encode("ascii")
    k = (n.bit_length() + 7) // 8
    em = b"\x00\x01" + b"\xff" * (k - len(DIGEST_INFO) - hashlib.sha256(signed).digest().__len__() - 3) + b"\x00" + DIGEST_INFO + hashlib.sha256(signed).digest()
    sig = pow(int.from_bytes(em, "big"), d, n).to_bytes(k, "big")
    return f"{h}.{p}.{_b64e(sig)}"


def _claims(**overrides) -> dict:
    now = int(time.time())
    data = {
        "iss": ISSUER, "aud": AUDIENCE, "sub": "u_9001",
        "preferred_username": "ops@example.com", "role": "agent",
        "iat": now, "nbf": now - 5, "exp": now + 300,
    }
    data.update(overrides)
    return data


def _verify(token: str):
    return verify_id_token(
        token, jwks=StaticJwksSource(JWKS), issuer=ISSUER, audience=AUDIENCE
    )


# ---------- 底层 RSA 验签 ----------
def test_rsa_pkcs1v15_verifies_valid_signature():
    n = N_INT
    e = 65537
    message = b"header.payload"
    k = (n.bit_length() + 7) // 8
    digest = hashlib.sha256(message).digest()
    em = b"\x00\x01" + b"\xff" * (k - len(DIGEST_INFO) - len(digest) - 3) + b"\x00" + DIGEST_INFO + digest
    sig = pow(int.from_bytes(em, "big"), D_INT, n).to_bytes(k, "big")
    assert _rsa_pkcs1v15_verify(n, e, message, sig) is True


def test_rsa_pkcs1v15_rejects_tampered_signature_and_message():
    n = N_INT
    e = 65537
    message = b"header.payload"
    k = (n.bit_length() + 7) // 8
    digest = hashlib.sha256(message).digest()
    em = b"\x00\x01" + b"\xff" * (k - len(DIGEST_INFO) - len(digest) - 3) + b"\x00" + DIGEST_INFO + digest
    sig = pow(int.from_bytes(em, "big"), D_INT, n).to_bytes(k, "big")
    assert _rsa_pkcs1v15_verify(n, e, b"header.payloaD", sig) is False
    assert _rsa_pkcs1v15_verify(n, e, message, b"\x00" * k) is False
    assert _rsa_pkcs1v15_verify(n, e, message, sig + b"\x00") is False


# ---------- id_token 验签与 claims ----------
def test_verify_id_token_accepts_valid_token():
    claims = _verify(_sign(_claims()))
    assert isinstance(claims, IdTokenClaims)
    assert claims.sub == "u_9001"
    assert claims.username == "ops@example.com"
    assert claims.role == "agent"


@pytest.mark.parametrize(
    "bad_payload,reason",
    [
        ({"iss": "https://evil.example.com"}, "issuer"),
        ({"aud": "another-app"}, "audience"),
        ({"exp": int(time.time()) - 1000}, "expired"),
        ({"nbf": int(time.time()) + 1000}, "not-before"),
        ({"sub": ""}, "sub"),
    ],
)
def test_verify_id_token_rejects_bad_claims(bad_payload, reason):
    with pytest.raises(ValueError):
        _verify(_sign(_claims(**bad_payload)))


def test_verify_id_token_rejects_wrong_algorithm_and_kid():
    with pytest.raises(ValueError):
        _verify(_sign(_claims(), alg="HS256"))
    with pytest.raises(ValueError):
        _verify(_sign(_claims(), kid="unknown-key"))


def test_verify_id_token_rejects_malformed_token():
    for token in ("", "abc", "a.b", "a.b.c.d", "!!!.!!!.!!!"):
        with pytest.raises(ValueError):
            _verify(token)


def test_verify_id_token_accepts_audience_list():
    claims = _verify(_sign(_claims(aud=["other", AUDIENCE])))
    assert AUDIENCE in claims.audience


def test_discover_jwks_url_parses_discovery_document(monkeypatch):
    doc = json.dumps({"jwks_uri": "https://idp.example.com/keys"}).encode()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout: _FakeResponse(doc),
    )
    assert discover_jwks_url(ISSUER + "/") == "https://idp.example.com/keys"


def test_discover_jwks_url_rejects_http_jwks_uri(monkeypatch):
    doc = json.dumps({"jwks_uri": "http://idp.example.com/keys"}).encode()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout: _FakeResponse(doc),
    )
    with pytest.raises(ValueError):
        discover_jwks_url(ISSUER)


class _FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


# ---------- 工厂与端到端 ----------
def test_build_identity_provider_uses_oidc_when_configured():
    from app.config import Settings
    from app.identity import OidcIdentityProvider, build_identity_provider

    s = Settings.load(
        idp_provider="oidc", oidc_issuer=ISSUER, oidc_audience=AUDIENCE,
        oidc_jwks_url="https://idp.example.com/keys",
    )
    provider = build_identity_provider(s)
    assert isinstance(provider, OidcIdentityProvider)


def test_oidc_identity_provider_authenticates_with_injected_jwks():
    from app.identity import OidcIdentityProvider

    provider = OidcIdentityProvider(
        ISSUER, AUDIENCE, jwks_source=StaticJwksSource(JWKS)
    )
    identity = provider.authenticate("", _sign(_claims()))
    assert identity is not None
    assert identity.user_id == "u_9001"
    assert identity.role == "agent"


def test_oidc_identity_provider_rejects_invalid_token():
    from app.identity import OidcIdentityProvider

    provider = OidcIdentityProvider(
        ISSUER, AUDIENCE, jwks_source=StaticJwksSource(JWKS)
    )
    assert provider.authenticate("", "not-a-jwt") is None
