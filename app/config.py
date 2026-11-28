"""集中配置：全部通过环境变量或 .env 覆盖，部署时不需要改代码。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DATA_DIR = APP_DIR / "data"


def load_dotenv(path: Path) -> None:
    """极简 .env 读取，避免为课程项目引入额外依赖。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # 已存在的真实环境变量优先，方便 CI / 线上覆盖
        os.environ.setdefault(key, value)


load_dotenv(PROJECT_DIR / ".env")


def _env_str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name) or default)
    except ValueError:
        return default


def _env_float(name: str) -> float | None:
    """可选浮点配置；留空/非法都返回 None（不编造默认值）。"""
    value = _env_str(name)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _env_float_or(name: str, default: float) -> float:
    """带默认值的浮点配置；留空 / 非法回落 default（区别于可选的 _env_float）。"""
    value = _env_float(name)
    return default if value is None else value


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env_str(name).lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on", "y"}


def _env_date(name: str) -> date | None:
    value = _env_str(name)
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


@dataclass
class Settings:
    """一个实例 = 一套运行参数，测试时可以整体替换。"""

    llm_mode: str = "rules"          # rules | llm
    agent_mode: str = "multi"        # multi | single

    # --- 主模型（表达层 / 话术润色，以及单 Agent 路径）-------------------
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_timeout: float = 30.0

    # --- 理解层专用模型（意图识别，可选）---------------------------------
    # 意图识别只输出一个 JSON，任务轻、可用免费模型（如智谱 glm-4-flash），
    # 主模型继续负责润色等关键表达。留空则理解层也走主模型（完全向后兼容）。
    # 五个字段必须一起配齐才生效；缺任意一个就整体回退到主模型。
    classify_base_url: str = ""
    classify_api_key: str = ""
    classify_model: str = ""
    classify_timeout: float = 30.0
    classify_price_in_per_1k: float | None = None   # 免费模型填 0，避免虚报费用
    classify_price_out_per_1k: float | None = None

    # Token 计费单价（人民币 / 每 1K tokens，主模型用）。留空则使用 app/usage.py
    # 的参考价目表，两者都没有时只报实测 token、不报金额；参考价不是供应商账单。
    price_in_per_1k: float | None = None
    price_out_per_1k: float | None = None
    # 网关倍率：聚合平台（如诺联）按官方价 × 分组倍率计费，缺省 1.0
    price_multiplier: float | None = None

    db_path: Path = PROJECT_DIR / "data" / "agent.db"
    orders_path: Path = DATA_DIR / "orders.json"
    logistics_path: Path = DATA_DIR / "logistics.json"
    policy_path: Path = DATA_DIR / "policies.md"

    # --- 订单源：默认读 orders.json；切 csv/http 后才接外部订单 ------------
    # user_id 必须映射到本系统登录身份，不能由订单源自行"猜用户"。
    order_source: str = "json"                 # AGENT_ORDER_SOURCE=json|csv|http
    order_source_path: str = ""                # AGENT_ORDER_SOURCE_PATH；空则用 orders_path
    order_source_url: str = ""                 # AGENT_ORDER_SOURCE_URL（http 模式必填）
    order_source_token: str = ""               # AGENT_ORDER_SOURCE_TOKEN（Bearer）
    order_source_timeout: float = 5.0          # AGENT_ORDER_SOURCE_TIMEOUT
    order_source_channel: str = "DOUYIN"       # AGENT_ORDER_SOURCE_CHANNEL
    order_source_field_map: str = ""           # AGENT_ORDER_SOURCE_FIELD_MAP（JSON 对象）

    return_window_days: int = 7
    quality_window_days: int = 15
    frozen_today: date | None = None

    simulate_response_loss: bool = False
    simulate_tool_timeout: bool = False

    # 表达层后置事实校验：polish 输出与代码模板不对齐时整体回退（默认开）
    grounding_enabled: bool = True

    # 身份与确认令牌：配置 AGENT_AUTH_SECRET 后启用签名 token，user_id 不再信任客户端传参
    auth_secret: str = ""
    idp_provider: str = "account"        # AGENT_IDP_PROVIDER=account|sso|oidc
    oidc_issuer: str = ""                 # AGENT_OIDC_ISSUER（oidc 模式必填）
    oidc_audience: str = ""               # AGENT_OIDC_AUDIENCE（oidc 模式必填）
    oidc_jwks_url: str = ""               # AGENT_OIDC_JWKS_URL，留空走 discovery
    auth_ttl_seconds: int = 86400
    confirm_token_ttl_seconds: int = 600

    # --- 安全底座：RBAC + 限流（默认全关，保持 demo 行为不变）--------------
    rbac_enabled: bool = False           # AGENT_RBAC_ENABLED=1 开启坐席角色鉴权
    staff_accounts: str = ""             # AGENT_STAFF_ACCOUNTS: name:password:role[:user_id];...
    staff_ttl_seconds: int = 28800       # 坐席 token 有效期（默认 8 小时）
    rate_limit_per_min: int = 0          # 全局每分钟上限，0=关闭
    rate_limit_chat: int = 0             # 聊天接口单独上限，0=跟随全局
    rate_limit_auth: int = 0             # 登录/签发接口上限，0=跟随全局
    rate_limit_switch: int = 0           # 模型切换上限，0=跟随全局
    rate_limit_backend: str = "memory"   # AGENT_RATE_LIMIT_BACKEND=memory|redis
    redis_url: str = ""                  # AGENT_REDIS_URL（Redis 后端才用）

    # --- 产品闭环：SLA 超时通知（企业微信群机器人）------------------------
    # 填 webhook 完整 URL（含 key）或仅 key 即启用；留空则只记事件、不推送。
    wecom_webhook: str = ""              # AGENT_WECOM_WEBHOOK
    sla_notify_enabled: bool = True      # AGENT_SLA_NOTIFY=0 关闭推送
    sla_notify_retries: int = 0          # AGENT_SLA_NOTIFY_RETRIES，指数退避次数
    sla_alert_minutes: int = 5           # 超时阈值（与 AgentDesk.SLA_MINUTES 对齐）
    sla_scheduler: str = "manual"        # AGENT_SLA_SCHEDULER=manual|background
    sla_scan_seconds: int = 60           # AGENT_SLA_SCAN_SECONDS，后台扫描间隔
    desk_base_url: str = ""              # 告警卡片里的坐席台链接前缀

    # --- 架构进阶：可插拔检索 / 分布式锁 / 退款网关 ----------------------
    # 三项默认都走「零配置、不向外发请求」的实现，配置后才切换后端。
    rerank_mode: str = "off"             # AGENT_RERANK=off|hybrid|embedding
    rerank_weight: float = 0.4           # AGENT_RERANK_WEIGHT，向量相似度融合权重
    embedding_base_url: str = ""         # AGENT_EMBEDDING_BASE_URL，OpenAI 兼容 /v1
    embedding_api_key: str = ""          # AGENT_EMBEDDING_API_KEY
    embedding_model: str = ""            # AGENT_EMBEDDING_MODEL
    embedding_timeout: float = 5.0       # AGENT_EMBEDDING_TIMEOUT
    lock_backend: str = "inprocess"      # AGENT_LOCK_BACKEND=inprocess|db|pg
    pg_dsn: str = ""                     # AGENT_PG_DSN（pg 后端必填）
    lock_lease_seconds: float = 30.0     # AGENT_LOCK_LEASE_SECONDS，租约 TTL（LLM 单轮可达 69s，按实际调大）
    lock_wait_seconds: float = 5.0       # AGENT_LOCK_WAIT_SECONDS，抢锁等待上限
    refund_api_url: str = ""            # AGENT_REFUND_API_URL，留空=沙箱网关
    refund_api_secret: str = ""         # AGENT_REFUND_API_SECRET，HMAC 签名密钥
    refund_timeout: float = 5.0         # AGENT_REFUND_TIMEOUT
    refund_callback_secret: str = ""    # AGENT_REFUND_CALLBACK_SECRET，回调 HMAC 密钥
    refund_callback_tolerance: int = 300  # AGENT_REFUND_CALLBACK_TOLERANCE，防重放窗口秒

    app_title: str = "小奈 · 电商售后 Agent"

    @classmethod
    def load(cls, **overrides) -> "Settings":
        settings = cls(
            llm_mode=_env_str("AGENT_LLM_MODE", "rules").lower(),
            agent_mode=_env_str("AGENT_MODE", "multi").lower(),
            llm_base_url=_env_str("LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_api_key=_env_str("LLM_API_KEY"),
            llm_model=_env_str("LLM_MODEL", "gpt-4o-mini"),
            llm_timeout=float(_env_str("LLM_TIMEOUT", "30") or 30),
            classify_base_url=_env_str("CLASSIFY_BASE_URL"),
            classify_api_key=_env_str("CLASSIFY_API_KEY"),
            classify_model=_env_str("CLASSIFY_MODEL"),
            classify_timeout=float(_env_str("CLASSIFY_TIMEOUT", "30") or 30),
            classify_price_in_per_1k=_env_float("CLASSIFY_PRICE_IN_PER_1K"),
            classify_price_out_per_1k=_env_float("CLASSIFY_PRICE_OUT_PER_1K"),
            price_in_per_1k=_env_float("LLM_PRICE_IN_PER_1K"),
            price_out_per_1k=_env_float("LLM_PRICE_OUT_PER_1K"),
            price_multiplier=_env_float("LLM_PRICE_MULTIPLIER"),
            db_path=Path(_env_str("DB_PATH", str(PROJECT_DIR / "data" / "agent.db"))),
            orders_path=Path(_env_str("ORDERS_PATH", str(DATA_DIR / "orders.json"))),
            logistics_path=Path(_env_str("LOGISTICS_PATH", str(DATA_DIR / "logistics.json"))),
            policy_path=Path(_env_str("POLICY_PATH", str(DATA_DIR / "policies.md"))),
            order_source=_env_str("AGENT_ORDER_SOURCE", "json").lower(),
            order_source_path=_env_str("AGENT_ORDER_SOURCE_PATH"),
            order_source_url=_env_str("AGENT_ORDER_SOURCE_URL"),
            order_source_token=_env_str("AGENT_ORDER_SOURCE_TOKEN"),
            order_source_timeout=_env_float_or("AGENT_ORDER_SOURCE_TIMEOUT", 5.0),
            order_source_channel=_env_str("AGENT_ORDER_SOURCE_CHANNEL", "DOUYIN"),
            order_source_field_map=_env_str("AGENT_ORDER_SOURCE_FIELD_MAP"),
            return_window_days=_env_int("RETURN_WINDOW_DAYS", 7),
            quality_window_days=_env_int("QUALITY_WINDOW_DAYS", 15),
            frozen_today=_env_date("AGENT_TODAY"),
            simulate_response_loss=_env_bool("SIMULATE_RESPONSE_LOSS", False),
            simulate_tool_timeout=_env_bool("SIMULATE_TOOL_TIMEOUT", False),
            grounding_enabled=_env_bool("AGENT_GROUNDING", True),
            auth_secret=_env_str("AGENT_AUTH_SECRET"),
            idp_provider=_env_str("AGENT_IDP_PROVIDER", "account").lower(),
            oidc_issuer=_env_str("AGENT_OIDC_ISSUER"),
            oidc_audience=_env_str("AGENT_OIDC_AUDIENCE"),
            oidc_jwks_url=_env_str("AGENT_OIDC_JWKS_URL"),
            auth_ttl_seconds=_env_int("AUTH_TTL_SECONDS", 86400),
            confirm_token_ttl_seconds=_env_int("CONFIRM_TOKEN_TTL_SECONDS", 600),
            rbac_enabled=_env_bool("AGENT_RBAC_ENABLED", False),
            staff_accounts=_env_str("AGENT_STAFF_ACCOUNTS"),
            staff_ttl_seconds=_env_int("STAFF_TTL_SECONDS", 28800),
            rate_limit_per_min=_env_int("AGENT_RATE_LIMIT_PER_MIN", 0),
            rate_limit_chat=_env_int("AGENT_RATE_LIMIT_CHAT", 0),
            rate_limit_auth=_env_int("AGENT_RATE_LIMIT_AUTH", 0),
            rate_limit_switch=_env_int("AGENT_RATE_LIMIT_SWITCH", 0),
            rate_limit_backend=_env_str("AGENT_RATE_LIMIT_BACKEND", "memory").lower(),
            redis_url=_env_str("AGENT_REDIS_URL"),
            wecom_webhook=_env_str("AGENT_WECOM_WEBHOOK"),
            sla_notify_enabled=_env_bool("AGENT_SLA_NOTIFY", True),
            sla_notify_retries=_env_int("AGENT_SLA_NOTIFY_RETRIES", 0),
            sla_alert_minutes=_env_int("AGENT_SLA_ALERT_MINUTES", 5),
            sla_scheduler=_env_str("AGENT_SLA_SCHEDULER", "manual").lower(),
            sla_scan_seconds=_env_int("AGENT_SLA_SCAN_SECONDS", 60),
            desk_base_url=_env_str("AGENT_DESK_BASE_URL"),
            rerank_mode=_env_str("AGENT_RERANK", "off").lower(),
            rerank_weight=_env_float_or("AGENT_RERANK_WEIGHT", 0.4),
            embedding_base_url=_env_str("AGENT_EMBEDDING_BASE_URL"),
            embedding_api_key=_env_str("AGENT_EMBEDDING_API_KEY"),
            embedding_model=_env_str("AGENT_EMBEDDING_MODEL"),
            embedding_timeout=_env_float_or("AGENT_EMBEDDING_TIMEOUT", 5.0),
            lock_backend=_env_str("AGENT_LOCK_BACKEND", "inprocess").lower(),
            pg_dsn=_env_str("AGENT_PG_DSN"),
            lock_lease_seconds=_env_float_or("AGENT_LOCK_LEASE_SECONDS", 30.0),
            lock_wait_seconds=_env_float_or("AGENT_LOCK_WAIT_SECONDS", 5.0),
            refund_api_url=_env_str("AGENT_REFUND_API_URL"),
            refund_api_secret=_env_str("AGENT_REFUND_API_SECRET"),
            refund_timeout=_env_float_or("AGENT_REFUND_TIMEOUT", 5.0),
            refund_callback_secret=_env_str("AGENT_REFUND_CALLBACK_SECRET"),
            refund_callback_tolerance=_env_int("AGENT_REFUND_CALLBACK_TOLERANCE", 300),
        )
        for key, value in overrides.items():
            setattr(settings, key, value)
        return settings

    def current_date(self) -> date:
        return self.frozen_today or date.today()

    @property
    def llm_enabled(self) -> bool:
        return self.llm_mode == "llm" and bool(self.llm_api_key)

    @property
    def classify_model_enabled(self) -> bool:
        """理解层专用模型是否配置齐全。缺任一项就走主模型，绝不带病上路。"""
        return bool(self.llm_enabled and self.classify_base_url and self.classify_api_key and self.classify_model)

    @property
    def signed_auth_enabled(self) -> bool:
        return bool(self.auth_secret)

    def describe(self) -> dict:
        """给 /api/health 用的可观测信息（不暴露密钥）。"""
        return {
            "llm_mode": "llm" if self.llm_enabled else "rules",
            "agent_mode": self.agent_mode,
            "model": self.llm_model if self.llm_enabled else "rule-engine",
            "classify_model": (
                self.classify_model if self.classify_model_enabled
                else (self.llm_model if self.llm_enabled else "rule-engine")
            ),
            "classify_routed": self.classify_model_enabled,
            "pricing_configured": self.price_in_per_1k is not None and self.price_out_per_1k is not None,
            "price_multiplier": self.price_multiplier if self.price_multiplier is not None else 1.0,
            "return_window_days": self.return_window_days,
            "quality_window_days": self.quality_window_days,
            "today": self.current_date().isoformat(),
            "simulate_response_loss": self.simulate_response_loss,
            "simulate_tool_timeout": self.simulate_tool_timeout,
            "grounding_enabled": self.grounding_enabled,
            "auth_mode": "signed" if self.signed_auth_enabled else "demo",
            "order_source": self.order_source,
            "order_source_configured": self.order_source != "json" or str(self.orders_path) != str(DATA_DIR / "orders.json"),
            "confirm_token_ttl_seconds": self.confirm_token_ttl_seconds,
            "rbac_enabled": self.rbac_enabled,
            "idp_provider": self.idp_provider,
            "rate_limit_backend": self.rate_limit_backend,
            "rate_limit_per_min": self.rate_limit_per_min,
            "sla_notify_enabled": bool(self.wecom_webhook) and self.sla_notify_enabled,
            "sla_alert_minutes": self.sla_alert_minutes,
            "sla_scheduler": self.sla_scheduler,
            "rerank_mode": self.rerank_mode,
            "rerank_weight": self.rerank_weight,
            "lock_backend": self.lock_backend,
            "refund_gateway": "http" if self.refund_api_url else "sandbox",
            "oidc_configured": bool(self.oidc_issuer and self.oidc_audience),
            "embedding_configured": bool(self.embedding_base_url),
            "refund_callback_configured": bool(self.refund_callback_secret),
        }
