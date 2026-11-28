"""一键启动六项本地真实协议替身（零 Docker / 零外部依赖）。

    python scripts/local_stubs/run_integration.py

启动后会打印每个服务的端口与对应的 .env 配置片段。默认绑定 127.0.0.1，
按 Ctrl+C 停止。服务只做协议联调，不发生真实资金流转。

注意：Postgres 是 SQL 语义级替身（sqlite 执行 %s 方言），不是 pg wire 协议；
要验证真实 psycopg 驱动请改用 docker-compose.local.yml 里的 Postgres 16。
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import integration_stubs as st  # noqa: E402


def main() -> None:
    http, http_port, _ = st.start_http_stub()
    redis, redis_port = st.start_redis_stub()
    pg, pg_port = st.start_pg_stub()

    base = f"http://127.0.0.1:{http_port}"
    print("=" * 68)
    print("六项本地真实协议替身已启动（仅监听 127.0.0.1）")
    print("=" * 68)
    print(f"  HTTP (OIDC / Embedding / Webhook / 支付网关) : {base}")
    print(f"  RESP (Redis)                                : 127.0.0.1:{redis_port}")
    print(f"  SQL  (Postgres 语义替身)                     : 127.0.0.1:{pg_port}")
    print("-" * 68)
    print("可写入 .env 的联调配置（示例，端口以本次输出为准）：")
    print(f"  AGENT_IDP_PROVIDER=oidc")
    print(f"  AGENT_OIDC_ISSUER={st.ISSUER}")
    print(f"  AGENT_OIDC_AUDIENCE={st.AUDIENCE}")
    print(f"  AGENT_OIDC_JWKS_URL={base}/jwks")
    print(f"  AGENT_EMBEDDING_BASE_URL={base}/v1")
    print(f"  AGENT_RERANK=embedding")
    print(f"  AGENT_WECOM_WEBHOOK={base}/webhook")
    print(f"  AGENT_REFUND_API_URL={base}/refunds")
    print(f"  AGENT_REFUND_API_SECRET={st.API_SECRET}")
    print(f"  AGENT_REFUND_CALLBACK_SECRET={st.CALLBACK_SECRET}")
    print(f"  AGENT_RATE_LIMIT_BACKEND=redis   # 需先 pip install redis（redis-py）")
    print(f"  AGENT_REDIS_URL=redis://127.0.0.1:{redis_port}/0")
    print("-" * 68)
    print("验证：python -m pytest tests/test_external_integration.py tests/test_app_integration.py -q")
    print("注意①：PG 语义替身不接 psycopg，真驱动验收请用 docker-compose.local.yml。")
    print("注意②：应用内 Redis 后端走 redis-py；未安装时 build_limiter 会回退内存限流。")
    print("       本脚本的 RESP 替身直接由 RedisLimiter + 适配器在测试中验证，不依赖 redis-py。")
    print("=" * 68)

    stop = threading.Event()
    try:
        while not stop.wait(1):
            pass
    except KeyboardInterrupt:
        print("\n正在停止替身服务…")
    finally:
        http.shutdown(); http.server_close()
        redis.shutdown(); redis.server_close()
        pg.shutdown(); pg.server_close()


if __name__ == "__main__":
    main()
