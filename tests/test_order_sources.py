"""订单源接入验收：默认零配置不变，CSV/HTTP 映射、失败降级与字段纪律。"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from app.config import Settings
from app.domain.order_sources import (
    OrderSourceError,
    build_order_source,
    record_to_order,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "app" / "data" / "orders_douyin_sample.csv"


def _settings(tmp_path: Path, **overrides) -> Settings:
    options = {
        "db_path": tmp_path / "orders.db",
        "frozen_today": date(2026, 11, 22),
    }
    options.update(overrides)
    return Settings.load(**options)


def test_default_json_source_keeps_demo_behavior(tmp_path):
    source = build_order_source(_settings(tmp_path))
    orders = source.load()
    assert source.describe()["mode"] == "json"
    assert not source.error
    assert "O20261115001" in orders
    assert orders["O20261115001"].title == "无线蓝牙耳机 Pro"


def test_csv_source_maps_douyin_sample_headers(tmp_path):
    source = build_order_source(_settings(
        tmp_path,
        order_source="csv",
        order_source_path=str(SAMPLE_CSV),
    ))
    orders = source.load()
    assert not source.error, source.error
    assert len(orders) == 4
    first = orders["DY20261101001"]
    assert first.user_id == "u_1001"
    assert first.title == "无线蓝牙耳机 Pro"
    assert first.status == "delivered"
    assert first.signed_at is not None
    assert source.describe()["loaded"] == 4


def test_csv_source_maps_status_and_category_from_platform_values(tmp_path):
    source = build_order_source(_settings(
        tmp_path,
        order_source="csv",
        order_source_path=str(SAMPLE_CSV),
    ))
    orders = source.load()
    assert orders["DY20261101002"].category == "定制"
    assert orders["DY20261101003"].category == "生鲜"
    assert orders["DY20261101004"].status == "shipped"
    assert orders["DY20261101004"].signed_at is None


def test_csv_source_rejects_record_without_user_id(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "order_no,product_name,pay_amount,order_status\n"
        "DYBAD001,耳机,10,已签收\n",
        encoding="utf-8",
    )
    source = build_order_source(_settings(
        tmp_path, order_source="csv", order_source_path=str(path)
    ))
    assert source.load() == {}
    assert "user_id" in source.error


def test_csv_source_rejects_unknown_status(tmp_path):
    path = tmp_path / "bad_status.csv"
    path.write_text(
        "order_no,buyer_id,product_name,pay_amount,order_status\n"
        "DYBAD002,u_1001,耳机,10,退款中\n",
        encoding="utf-8",
    )
    source = build_order_source(_settings(
        tmp_path, order_source="csv", order_source_path=str(path)
    ))
    assert source.load() == {}
    assert "无法识别的订单状态" in source.error


def test_field_map_override_supports_custom_headers(tmp_path):
    path = tmp_path / "custom.csv"
    path.write_text(
        "订单编号,买家账号,商品名称,实付金额,订单状态,签收时间\n"
        "DY-CUSTOM-1,u_1009,定制项链,99,已完成,2026-11-20 10:30\n",
        encoding="utf-8",
    )
    field_map = json.dumps({
        "order_id": "订单编号",
        "user_id": "买家账号",
        "title": "商品名称",
        "amount": "实付金额",
        "status": "订单状态",
        "signed_at": "签收时间",
    }, ensure_ascii=False)
    source = build_order_source(_settings(
        tmp_path,
        order_source="csv",
        order_source_path=str(path),
        order_source_field_map=field_map,
    ))
    order = source.load()["DY-CUSTOM-1"]
    assert order.user_id == "u_1009"
    assert order.category == "定制"
    assert order.status == "delivered"


def test_record_to_order_does_not_guess_owner():
    with pytest.raises(OrderSourceError, match="user_id"):
        record_to_order(
            {"order_id": "DY1", "title": "耳机", "amount": 10, "status": "paid"},
            date(2026, 11, 22),
        )


class _Response:
    def __init__(self, payload):
        self._raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._raw


def test_http_source_maps_api_payload_and_redacts_endpoint(tmp_path):
    payload = {
        "orders": [{
            "out_order_no": "DYHTTP1",
            "buyer_open_id": "u_1001",
            "goods_name": "无线耳机",
            "pay_amount": "199.00",
            "order_status": "SIGNED",
            "receive_time": "2026-11-20 10:30",
        }]
    }
    seen = {}

    def opener(request, timeout):
        seen["url"] = request.full_url
        seen["auth"] = request.headers.get("Authorization")
        seen["timeout"] = timeout
        return _Response(payload)

    source = build_order_source(_settings(
        tmp_path,
        order_source="http",
        order_source_url="https://api.example.com/orders?token=secret",
        order_source_token="bearer-secret",
        order_source_timeout=7,
    ))
    source._opener = opener
    orders = source.load()
    assert not source.error, source.error
    assert orders["DYHTTP1"].status == "delivered"
    assert orders["DYHTTP1"].user_id == "u_1001"
    assert seen["auth"] == "Bearer bearer-secret"
    assert seen["timeout"] == 7
    assert "secret" not in source.describe()["endpoint"]


def test_http_source_failure_degrades_with_explicit_error(tmp_path):
    def broken(_request, timeout=None):
        raise OSError("connection refused")

    source = build_order_source(_settings(
        tmp_path,
        order_source="http",
        order_source_url="https://api.example.com/orders",
    ))
    source._opener = broken
    assert source.load() == {}
    assert "订单源加载失败" in source.error
    assert source.describe()["error"] == source.error


def test_unknown_source_mode_fails_fast(tmp_path):
    with pytest.raises(OrderSourceError, match="未知的订单源模式"):
        build_order_source(_settings(tmp_path, order_source="mysql"))


def test_import_script_dry_run_converts_sample():
    result = subprocess.run(
        [sys.executable, "scripts/import_orders.py",
         "--input", "app/data/orders_douyin_sample.csv", "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    records = json.loads(result.stdout)
    assert len(records) == 4
    assert records[0]["order_id"] == "DY20261101001"
    assert records[0]["user_id"] == "u_1001"
