"""可插拔后端协议：限流 / 身份源 / SLA 调度。

三项都遵循同一条纪律：默认实现零配置、不向外发请求；真实后端由配置切换、
由接口注入，测试用假对象验证「多实例共享 / 拒绝伪造 / 定时触发」，不依赖
Redis、IdP 或企微 webhook 是否真的可用。
"""

from __future__ import annotations

import threading

from fastapi.testclient import TestClient

from app.config import Settings
from app.identity import AccountIdentityProvider, SsoCodeIdentityProvider, issue_sso_code
from app.main import app
from app.ratelimit import RedisLimiter, SlidingWindowLimiter, build_limiter
from app.sla_scheduler import BackgroundScheduler, ManualScheduler, build_scheduler

# ---------------------------------------------------------------------------
# 1. 限流：多实例共享计数（Redis 协议）
# ---------------------------------------------------------------------------
class _FakePipeline:
    def __init__(self, store: dict):
        self.store = store
        self.results: list = []
        self.key = ""

    def zremrangebyscore(self, key, _min, max_score):
        self.key = key
        values = [v for v in self.store.get(key, {}).values() if v > max_score]
        self.store.setdefault(key, {})
        for member in list(self.store[key]):
            if self.store[key][member] <= max_score:
                del self.store[key][member]
        self.results.append(len(values))

    def zadd(self, key, mapping):
        self.key = key
        self.store.setdefault(key, {}).update(mapping)
        self.results.append(1)

    def zcard(self, key):
        self.results.append(len(self.store.get(key, {})))

    def pexpire(self, key, ttl):
        self.results.append(True)

    def execute(self):
        return self.results


class _FakeRedis:
    def __init__(self):
        self.store: dict[str, dict[str, float]] = {}

    def pipeline(self):
        return _FakePipeline(self.store)

    def zrange(self, key, start, end, withscores=False):
        items = sorted(self.store.get(key, {}).items(), key=lambda item: item[1])
        return list(items[start:end + 1])


def test_two_redis_backed_instances_share_one_counter():
    shared = _FakeRedis()
    first = RedisLimiter(shared)
    second = RedisLimiter(shared)

    assert first.check("203.0.113.1:chat", 1) == (True, 0)
    allowed, retry = second.check("203.0.113.1:chat", 1)
    assert allowed is False
    assert retry >= 1


def test_redis_limiter_is_scoped_by_key_and_disabled_by_zero_limit():
    client = _FakeRedis()
    limiter = RedisLimiter(client)
    assert limiter.check("a", 1)[0] is True
    assert limiter.check("b", 1)[0] is True
    assert limiter.check("b", 0) == (True, 0)


def test_build_limiter_defaults_to_inprocess_and_falls_back_on_redis_failure(tmp_path):
    settings = Settings.load(db_path=tmp_path / "rl.db", rate_limit_backend="memory")
    assert isinstance(build_limiter(settings), SlidingWindowLimiter)

    broken = Settings.load(db_path=tmp_path / "rl2.db", rate_limit_backend="redis", redis_url="")
    assert isinstance(build_limiter(broken), SlidingWindowLimiter)


# ---------------------------------------------------------------------------
# 2. 身份源：账号表（默认）与 SSO 签名码（沙箱）
# ---------------------------------------------------------------------------
_ACCOUNTS = "lead:pwd:supervisor:u_1001;view:pwd2:viewer:u_1002"


def test_account_provider_keeps_demo_login_semantics():
    provider = AccountIdentityProvider(_ACCOUNTS)
    identity = provider.authenticate("lead", "pwd")
    assert identity is not None
    assert (identity.user_id, identity.role) == ("u_1001", "supervisor")
    assert provider.authenticate("lead", "wrong") is None
    assert provider.authenticate("nobody", "pwd") is None


def test_sso_provider_accepts_signed_code_and_rejects_tampering():
    provider = SsoCodeIdentityProvider("secret")
    code = issue_sso_code("u_1007", "secret", role="agent")
    identity = provider.authenticate("u_1007", code)
    assert identity is not None
    assert (identity.user_id, identity.role) == ("u_1007", "agent")

    assert provider.authenticate("other_user", code) is None
    assert provider.authenticate("u_1007", code + "x") is None
    assert provider.authenticate("u_1007", "null") is None
    assert SsoCodeIdentityProvider("").authenticate("u_1007", code) is None


def test_expired_sso_code_is_rejected():
    provider = SsoCodeIdentityProvider("secret")
    expired = issue_sso_code("u_1007", "secret", ttl_seconds=-10)
    assert provider.authenticate("u_1007", expired) is None


# ---------------------------------------------------------------------------
# 3. SLA 调度：默认不动，后台模式真的会调用扫描函数
# ---------------------------------------------------------------------------
def test_manual_scheduler_is_a_noop():
    calls: list[int] = []
    scheduler = ManualScheduler()
    scheduler.start(lambda: calls.append(1))
    scheduler.stop()
    assert calls == []


def test_background_scheduler_triggers_scan_and_stops_cleanly():
    fired = threading.Event()
    calls: list[int] = []
    scheduler = BackgroundScheduler(interval_seconds=0.05, run_immediately=True)

    def scan():
        calls.append(1)
        fired.set()

    scheduler.start(scan)
    assert fired.wait(1.0) is True
    scheduler.stop()
    assert calls
    assert scheduler._thread is None


def test_build_scheduler_respects_config(tmp_path):
    manual = Settings.load(db_path=tmp_path / "s1.db", sla_scheduler="manual")
    background = Settings.load(db_path=tmp_path / "s2.db", sla_scheduler="background", sla_scan_seconds=7)
    assert isinstance(build_scheduler(manual), ManualScheduler)
    scheduler = build_scheduler(background)
    assert isinstance(scheduler, BackgroundScheduler)
    assert scheduler.interval == 7.0


# ---------------------------------------------------------------------------
# 4. 默认服务未受影响：健康接口仍可用，新增配置可见
# ---------------------------------------------------------------------------
def test_health_exposes_new_backend_configuration():
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["idp_provider"] == "account"
    assert body["rate_limit_backend"] == "memory"
    assert body["sla_scheduler"] == "manual"
