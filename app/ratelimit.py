"""限流：进程内滑动窗口（默认）与可注入的 Redis 后端。

默认关闭（rate_limit_per_min=0），不影响课程演示与现有测试；
配置后按 client IP + 端点组计数，超限返回 429 并给出 Retry-After。
Redis 后端**不 import redis 作为硬依赖**：客户端由调用方注入，测试用假客户端
即可验证多实例共享同一计数；连接失败时回退到进程内限流（可用性优先）。
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Protocol

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)


class Limiter(Protocol):
    """限流协议：进程内与 Redis 两个实现共用同一接口。"""

    def check(self, key: str, limit: int) -> tuple[bool, int]: ...



class SlidingWindowLimiter:
    def __init__(self, window_seconds: int = 60):
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int) -> tuple[bool, int]:
        """返回 (是否放行, 建议重试秒数)。limit<=0 表示不限制。"""
        if limit <= 0:
            return True, 0
        now = time.monotonic()
        with self._lock:
            bucket = self._hits[key]
            while bucket and bucket[0] <= now - self.window:
                bucket.popleft()
            if len(bucket) >= limit:
                retry = max(1, int(self.window - (now - bucket[0])) + 1)
                return False, retry
            bucket.append(now)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


class RedisLimiter:
    """多实例共享计数：Redis sorted set 滑动窗口，client 由外部注入。

    用 pipeline 把「清理过期 → 写入本次 → 计数 → 设 TTL」打包，避免并发下
    多个实例各自算一遍；member 带 uuid，防止同一毫秒的请求被当作同一条。
    """

    def __init__(self, client, window_seconds: int = 60, key_prefix: str = "agent:rl"):
        self.client = client
        self.window = window_seconds
        self.prefix = key_prefix

    def check(self, key: str, limit: int) -> tuple[bool, int]:
        if limit <= 0:
            return True, 0
        now = time.time()
        redis_key = f"{self.prefix}:{key}"
        pipe = self.client.pipeline()
        pipe.zremrangebyscore(redis_key, 0, now - self.window)
        pipe.zadd(redis_key, {f"{now}:{uuid.uuid4().hex}": now})
        pipe.zcard(redis_key)
        pipe.pexpire(redis_key, int(self.window * 1000))
        results = pipe.execute()
        count = int(results[2])
        if count <= limit:
            return True, 0
        # 最老一条的分数决定还要等多久
        oldest = self.client.zrange(redis_key, 0, 0, withscores=True)
        retry = max(1, int(self.window - (now - float(oldest[0][1]))) + 1) if oldest else 1
        return False, retry


def build_limiter(settings) -> Limiter:
    """按配置选择后端；Redis 连接不可用则退回进程内实现（不阻断聊天）。"""
    backend = str(getattr(settings, "rate_limit_backend", "memory") or "memory").lower()
    if backend != "redis":
        return SlidingWindowLimiter()
    try:
        import redis  # 延迟导入：零配置演示不要求安装 redis-py
        client = redis.from_url(getattr(settings, "redis_url", "") or "redis://127.0.0.1:6379/0")
        client.ping()
        return RedisLimiter(client)
    except Exception as exc:
        logger.warning("Redis 限流后端不可用，回退进程内限流：%s", type(exc).__name__)
        return SlidingWindowLimiter()


def client_ip(request: Request) -> str:
    # 不信任 X-Forwarded-For（可伪造）；多实例部署应在可信反代层终止并改写。
    return request.client.host if request.client else "unknown"


def enforce(request: Request, scope: str, default_limit: int) -> None:
    """在端点开头调用：按配置限流，关闭时直接返回。"""
    settings = request.app.state.settings
    limit = int(getattr(settings, "rate_limit_per_min", 0) or 0)
    endpoint_limit = int(getattr(settings, f"rate_limit_{scope}", 0) or 0)
    if endpoint_limit > 0:
        limit = endpoint_limit
    if limit <= 0:
        return
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    allowed, retry = limiter.check(f"{client_ip(request)}:{scope}", limit)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"请求过于频繁，请 {retry} 秒后重试",
            headers={"Retry-After": str(retry)},
        )
