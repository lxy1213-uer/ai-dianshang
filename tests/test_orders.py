"""第 1 周验收：订单查询、期限判断、边界值。"""

from datetime import date

from app.domain.orders import (
    CODE_CATEGORY_EXCLUDED,
    CODE_DATA_MISSING,
    CODE_ELIGIBLE_NO_REASON,
    CODE_NOT_SHIPPED,
    CODE_NOT_SIGNED,
    CODE_ORDER_CLOSED,
    CODE_OUT_OF_WINDOW,
    REASON_NO_REASON,
    REASON_QUALITY,
    evaluate_return,
    mask_address,
)
from app.config import Settings

TODAY = date(2026, 11, 22)


def load(settings: Settings):
    from app.domain.orders import load_orders

    return load_orders(settings.orders_path, TODAY)


def test_order_lookup_by_id(settings):
    orders = load(settings)
    order = orders["O20261115001"]
    assert order.title == "无线蓝牙耳机 Pro"
    assert order.amount == 299.0
    assert order.signed_at is not None


def test_deadline_boundary_day7_eligible(settings):
    """签收第 7 天（今天到期）仍可退。"""
    order = load(settings)["O20261115001"]
    result = evaluate_return(order, TODAY)
    assert result.code == CODE_ELIGIBLE_NO_REASON
    assert result.remaining_days == 0
    assert result.deadline == date(2026, 11, 22)


def test_deadline_boundary_day8_rejected(settings):
    """签收第 8 天（超期 1 天）不再受理无理由退货。"""
    order = load(settings)["O20261115002"]
    result = evaluate_return(order, TODAY)
    assert result.code == CODE_OUT_OF_WINDOW
    assert result.days_since_sign == 8
    assert result.requires_human is True


def test_unsigned_order(settings):
    order = load(settings)["O20261120005"]
    result = evaluate_return(order, TODAY)
    assert result.code == CODE_NOT_SIGNED
    assert result.eligible is False


def test_missing_signed_time_goes_human(settings):
    """已签收但缺签收时间：数据异常，不能自动拒绝。"""
    order = load(settings)["O20261119010"]
    result = evaluate_return(order, TODAY)
    assert result.code == CODE_DATA_MISSING
    assert result.requires_human is True


def test_category_excluded_for_no_reason(settings):
    order = load(settings)["O20261118004"]  # 定制
    result = evaluate_return(order, TODAY, REASON_NO_REASON)
    assert result.code == CODE_CATEGORY_EXCLUDED
    assert result.requires_human is True


def test_quality_reason_allows_excluded_category(settings):
    """类目受限但属于质量问题：按 15 天质量问题通道，可以退。"""
    order = load(settings)["O20261118004"]
    result = evaluate_return(order, TODAY, REASON_QUALITY)
    assert result.eligible is True
    assert result.code == "ELIGIBLE_QUALITY"
    assert result.window_days == 15


def test_refunded_and_unshipped_orders(settings):
    orders = load(settings)
    assert evaluate_return(orders["O20261114011"], TODAY).code == CODE_ORDER_CLOSED
    assert evaluate_return(orders["O20261121007"], TODAY).code == CODE_NOT_SHIPPED


def test_address_is_masked(settings):
    order = load(settings)["O20261115001"]
    public = order.to_public()
    assert "号" not in public["address"]
    assert public["address"].endswith("****")
    assert mask_address("") == ""


def test_quality_window_is_15_days(settings):
    """签收 8 天的订单，质量问题仍在 15 天内。"""
    order = load(settings)["O20261115002"]
    result = evaluate_return(order, TODAY, REASON_QUALITY)
    assert result.eligible is True
    assert result.days_since_sign == 8
