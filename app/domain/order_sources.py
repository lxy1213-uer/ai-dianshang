"""可插拔订单源：把平台订单 / 离线导出文件转换成内部 Order。

设计边界
--------
1. 默认 json 源继续读 app/data/orders.json，零配置行为完全不变；
2. csv 源用于个人演示：把抖音/抖店导出的表格直接接进来，不需要开放平台权限；
3. http 源是真实 API 的通用骨架：返回 list 或 {"orders": [...]} 即可，
   字段通过别名表映射，Bearer token 只在服务端内存里使用；
4. 归属字段 user_id 必须存在且不可伪造——它必须映射到本系统登录身份，
   否则该条记录直接拒绝，绝不"猜一个用户"；
5. 外部源失败时记 error 并返回空字典，工具/诊断/列表接口会明确报
   "订单源不可用"，不会把系统故障伪装成"查不到这个订单"。

新增一个平台时只需要加一个 OrderSource 实现，后面的核单、资格判定、
状态机、工单和诊断层都不用改。
"""

from __future__ import annotations

import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from .orders import Order, parse_dt


class OrderSourceError(Exception):
    """订单源配置或数据错误；配置错误 fail fast，运行错误记入 error。"""


class OrderSource(Protocol):
    mode: str
    error: str

    def load(self) -> dict[str, Order]: ...

    def describe(self) -> dict: ...


# 平台字段 → 内部字段。CSV 表头、JSON key 都走同一张表。
# AGENT_ORDER_SOURCE_FIELD_MAP 可覆盖：{"order_id": "out_order_no", ...}
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "order_id": ("order_id", "order_no", "out_order_no", "id", "订单号"),
    "user_id": ("user_id", "buyer_id", "open_id", "buyer_open_id", "用户ID", "买家ID"),
    "title": ("title", "product_name", "goods_name", "商品标题", "商品名称"),
    "category": ("category", "category_name", "类目", "商品类目"),
    "amount": ("amount", "pay_amount", "paid_amount", "total_amount", "实付金额", "支付金额"),
    "status": ("status", "order_status", "订单状态"),
    "signed_at": ("signed_at", "sign_time", "receive_time", "签收时间"),
    "signed_days_ago": ("signed_days_ago",),
    "address": ("address", "receiver_address", "收货地址"),
    "sku": ("sku", "spec", "sku_name", "规格"),
    "channel": ("channel", "platform", "渠道", "平台"),
}

# 平台状态 → 内部状态。未知状态直接拒绝，避免把"退款中"当成已签收。
STATUS_MAP: dict[str, str] = {
    "paid": "paid", "待发货": "paid", "待支付": "paid", "pay_success": "paid", "已支付": "paid",
    "shipped": "shipped", "已发货": "shipped", "配送中": "shipped", "shipping": "shipped",
    "delivered": "delivered", "已签收": "delivered", "已完成": "delivered",
    "signed": "delivered", "received": "delivered", "receive_success": "delivered",
    "refunded": "refunded", "已退款": "refunded", "退款成功": "refunded", "refund_success": "refunded",
    "closed": "closed", "已关闭": "closed", "已取消": "closed", "canceled": "closed", "cancelled": "closed",
}

# 类目兜底：导出表常没有标准类目，从标题里识别会直接影响 R-03 判定，必须保守。
CATEGORY_HINTS: tuple[tuple[str, str], ...] = (
    ("定制", "定制"), ("刻字", "定制"), (" DIY ", "定制"),
    ("内衣", "内衣"), ("文胸", "内衣"), ("内裤", "内衣"),
    ("生鲜", "生鲜"), ("水果", "生鲜"), ("蓝莓", "生鲜"), ("草莓", "生鲜"), ("车厘子", "生鲜"),
    ("虚拟", "虚拟"), ("充值", "虚拟"), ("会员", "虚拟"), ("课程", "虚拟"),
)

_VALID_STATUSES = {"paid", "shipped", "delivered", "refunded", "closed"}
_JUNK = {"", "null", "none", "nan", "-", "未知"}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    return str(value).strip()


def _fields(field_map: dict[str, str] | None) -> dict[str, tuple[str, ...]]:
    merged = {key: tuple(names) for key, names in FIELD_ALIASES.items()}
    for canonical, source_name in (field_map or {}).items():
        if canonical not in merged:
            raise OrderSourceError(f"不支持的字段映射目标：{canonical}")
        name = _clean(source_name)
        if name:
            merged[canonical] = (name,) + tuple(x for x in merged[canonical] if x != name)
    return merged


def _pick(record: dict, canonical: str, aliases: dict[str, tuple[str, ...]]) -> str:
    for name in aliases.get(canonical, ()):
        value = _clean(record.get(name))
        if value and value.lower() not in _JUNK:
            return value
    return ""


def _map_category(raw: str, title: str) -> str:
    category = _clean(raw)
    if category:
        return category
    upper_title = f" {title} "
    for needle, mapped in CATEGORY_HINTS:
        if needle.strip() in upper_title:
            return mapped
    return "其他"


def _map_status(raw: str) -> str:
    value = _clean(raw)
    mapped = STATUS_MAP.get(value.lower()) or STATUS_MAP.get(value)
    if mapped and mapped in _VALID_STATUSES:
        return mapped
    raise OrderSourceError(f"无法识别的订单状态：{value or '(空)'}")


def _map_amount(raw: str) -> float:
    value = _clean(raw).replace("¥", "").replace("￥", "").replace(",", "")
    try:
        amount = float(value)
    except ValueError as exc:
        raise OrderSourceError(f"金额格式错误：{value or '(空)'}") from exc
    if amount < 0:
        raise OrderSourceError(f"金额不能为负：{value}")
    return amount


def record_to_order(
    record: dict,
    today: date,
    *,
    default_channel: str = "APP",
    field_map: dict[str, str] | None = None,
) -> Order:
    """把一条平台原始记录转成内部 Order；关键事实缺失或非法时 fail closed。"""
    if not isinstance(record, dict):
        raise OrderSourceError("订单记录必须是对象")

    aliases = _fields(field_map)
    order_id = _pick(record, "order_id", aliases).upper()
    user_id = _pick(record, "user_id", aliases)
    title = _pick(record, "title", aliases)
    if not order_id:
        raise OrderSourceError("订单缺少 order_id（或别名 order_no / id）")
    if not user_id:
        raise OrderSourceError(f"订单 {order_id} 缺少 user_id：不能猜测归属账号")
    if not title:
        raise OrderSourceError(f"订单 {order_id} 缺少商品标题")

    signed_at = parse_dt(_pick(record, "signed_at", aliases))
    days_ago = _pick(record, "signed_days_ago", aliases)
    if signed_at is None and days_ago:
        try:
            signed_at = datetime.combine(
                today - timedelta(days=int(days_ago)), datetime.min.time()
            ).replace(hour=10, minute=30)
        except ValueError as exc:
            raise OrderSourceError(f"订单 {order_id} 的 signed_days_ago 非法：{days_ago}") from exc

    return Order(
        order_id=order_id,
        user_id=user_id,
        title=title,
        category=_map_category(_pick(record, "category", aliases), title),
        amount=_map_amount(_pick(record, "amount", aliases)),
        status=_map_status(_pick(record, "status", aliases)),
        signed_at=signed_at,
        address=_pick(record, "address", aliases),
        sku=_pick(record, "sku", aliases),
        channel=_pick(record, "channel", aliases) or default_channel,
    )


def _records_to_orders(records: list[dict], today: date, **kwargs) -> dict[str, Order]:
    orders: dict[str, Order] = {}
    for index, record in enumerate(records, start=1):
        order = record_to_order(record, today, **kwargs)
        if order.order_id in orders:
            raise OrderSourceError(f"订单号重复：{order.order_id}（第 {index} 条）")
        orders[order.order_id] = order
    return orders


def _redact_url(url: str) -> str:
    """健康状态只暴露域名和路径，不泄露 query/token。"""
    parsed = urlparse(url)
    netloc = parsed.hostname or ""
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))


class JsonOrderSource:
    """默认源：沿用既有 orders.json，保持所有演示与评测口径不变。"""

    mode = "json"

    def __init__(self, path: Path, today: date):
        self.path = Path(path)
        self.today = today
        self.error = ""
        self.loaded = 0

    def load(self) -> dict[str, Order]:
        from .orders import load_orders

        orders = load_orders(self.path, self.today)
        self.loaded = len(orders)
        self.error = ""
        return orders

    def describe(self) -> dict:
        return {"mode": self.mode, "path": str(self.path), "loaded": self.loaded, "error": self.error}


class CsvOrderSource:
    """离线导入源：直接读平台导出的 CSV，适合没有开放平台权限的个人演示。"""

    mode = "csv"

    def __init__(
        self, path: Path, today: date, *,
        default_channel: str = "DOUYIN",
        field_map: dict[str, str] | None = None,
    ):
        self.path = Path(path)
        self.today = today
        self.default_channel = default_channel
        self.field_map = field_map or {}
        self.error = ""
        self.loaded = 0

    def load(self) -> dict[str, Order]:
        try:
            with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            orders = _records_to_orders(
                rows, self.today,
                default_channel=self.default_channel,
                field_map=self.field_map,
            )
        except (OSError, csv.Error, OrderSourceError) as exc:
            self.loaded = 0
            self.error = f"CSV 订单源加载失败：{exc}"
            return {}
        self.error = ""
        self.loaded = len(orders)
        return orders

    def describe(self) -> dict:
        return {
            "mode": self.mode, "path": str(self.path), "loaded": self.loaded,
            "channel": self.default_channel, "error": self.error,
        }


class HttpOrderSource:
    """真实订单 API 骨架：GET JSON，Bearer token，失败时明确降级为空。"""

    mode = "http"

    def __init__(
        self, url: str, today: date, *,
        token: str = "", timeout: float = 5.0,
        default_channel: str = "DOUYIN",
        field_map: dict[str, str] | None = None,
        opener=None,
    ):
        if not url:
            raise OrderSourceError("http 订单源必须配置 AGENT_ORDER_SOURCE_URL")
        self.url = url
        self.token = token
        self.timeout = timeout
        self.today = today
        self.default_channel = default_channel
        self.field_map = field_map or {}
        self._opener = opener or urlopen
        self.error = ""
        self.loaded = 0

    def load(self) -> dict[str, Order]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            request = Request(self.url, headers=headers, method="GET")
            with self._opener(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, dict):
                payload = next(
                    (payload[key] for key in ("orders", "data", "items", "list")
                     if isinstance(payload.get(key), list)),
                    None,
                )
            if not isinstance(payload, list):
                raise OrderSourceError("接口响应需为订单数组或含 orders/data/items/list 的对象")
            orders = _records_to_orders(
                payload, self.today,
                default_channel=self.default_channel,
                field_map=self.field_map,
            )
        except (OSError, HTTPError, URLError, ValueError, OrderSourceError) as exc:
            self.loaded = 0
            self.error = f"HTTP 订单源加载失败：{exc}"
            return {}
        self.error = ""
        self.loaded = len(orders)
        return orders

    def describe(self) -> dict:
        return {
            "mode": self.mode, "endpoint": _redact_url(self.url), "loaded": self.loaded,
            "channel": self.default_channel, "token_configured": bool(self.token),
            "error": self.error,
        }


def build_order_source(settings: Any) -> OrderSource:
    """配置驱动的订单源工厂；未知 mode 直接报错，不静默退回演示数据。"""
    mode = (getattr(settings, "order_source", "json") or "json").lower()
    path = getattr(settings, "order_source_path", "") or str(getattr(settings, "orders_path", ""))
    field_map = getattr(settings, "order_source_field_map", "") or ""
    if isinstance(field_map, str) and field_map.strip():
        try:
            parsed_map = json.loads(field_map)
        except ValueError as exc:
            raise OrderSourceError(f"AGENT_ORDER_SOURCE_FIELD_MAP 不是合法 JSON：{exc}") from exc
        if not isinstance(parsed_map, dict):
            raise OrderSourceError("AGENT_ORDER_SOURCE_FIELD_MAP 必须是 JSON 对象")
    else:
        parsed_map = {}

    common = {
        "default_channel": getattr(settings, "order_source_channel", "DOUYIN") or "DOUYIN",
        "field_map": parsed_map,
    }
    today = settings.current_date()
    if mode == "json":
        return JsonOrderSource(Path(path), today)
    if mode == "csv":
        return CsvOrderSource(Path(path), today, **common)
    if mode == "http":
        return HttpOrderSource(
            getattr(settings, "order_source_url", "") or "",
            today,
            token=getattr(settings, "order_source_token", "") or "",
            timeout=float(getattr(settings, "order_source_timeout", 5.0) or 5.0),
            **common,
        )
    raise OrderSourceError(f"未知的订单源模式：{mode}（可用 json/csv/http）")
