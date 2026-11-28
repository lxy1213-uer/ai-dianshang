#!/usr/bin/env python3
"""把平台导出的订单 CSV 转成项目内部 orders.json（离线导入，不需要开放平台权限）。

用法：
    python scripts/import_orders.py --input app/data/orders_douyin_sample.csv --output data/orders_imported.json
    python scripts/import_orders.py --input orders.csv --dry-run

字段默认支持 order_no / buyer_id / product_name / pay_amount / order_status / sign_time 等
平台常见表头；如导出表头不同，用 --field-map 覆盖，例如：
    --field-map '{"order_id":"订单编号","user_id":"买家账号","title":"商品名称"}'

导入后启动应用：
    ORDERS_PATH=data/orders_imported.json python -m uvicorn app.main:app --reload
或：
    AGENT_ORDER_SOURCE=csv AGENT_ORDER_SOURCE_PATH=orders.csv python -m uvicorn app.main:app --reload
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.domain.order_sources import build_order_source  # noqa: E402


def _to_record(order) -> dict:
    return {
        "order_id": order.order_id,
        "user_id": order.user_id,
        "title": order.title,
        "sku": order.sku,
        "category": order.category,
        "amount": round(order.amount, 2),
        "status": order.status,
        "signed_at": order.signed_at.strftime("%Y-%m-%d %H:%M") if order.signed_at else None,
        "address": order.address,
        "channel": order.channel,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="平台订单 CSV → 内部 orders.json")
    parser.add_argument("--input", required=True, help="平台导出的 CSV 路径")
    parser.add_argument("--output", default="data/orders_imported.json", help="输出的内部 JSON 路径")
    parser.add_argument("--channel", default="DOUYIN", help="默认渠道标识")
    parser.add_argument("--field-map", default="", help="JSON 对象：内部字段 → 平台表头")
    parser.add_argument("--dry-run", action="store_true", help="只预览转换结果，不写文件")
    args = parser.parse_args()

    settings = Settings.load(
        order_source="csv",
        order_source_path=str(Path(args.input)),
        order_source_channel=args.channel,
        order_source_field_map=args.field_map,
    )
    source = build_order_source(settings)
    orders = source.load()
    if source.error:
        print(source.error, file=sys.stderr)
        return 1

    records = [_to_record(order) for order in orders.values()]
    if args.dry_run:
        print(json.dumps(records, ensure_ascii=False, indent=2))
        print(f"\n[preview] {len(records)} 条订单，未写入文件", file=sys.stderr)
        return 0

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已导入 {len(records)} 条订单 → {output}")
    print("启动应用：ORDERS_PATH=" + str(output) + " python -m uvicorn app.main:app --reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
