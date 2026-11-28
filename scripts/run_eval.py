"""命令行评测：python scripts/run_eval.py

默认跑「dev + dev2 + holdout2」三段的完整对比，并附一次口径敏感性自检。
报告写入 data/eval_report.json，可直接作为验收材料。
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 评测默认走规则模式：可复现、零成本。开发者本地 .env 里若配了真实 LLM，
# 不隔离的话 48 条案例会真的打外部 API（慢、花钱、结果还不可复现）。
# 确实想在 LLM 模式下评测，加 --llm。config.load_dotenv 用 setdefault，先占位即可压住 .env。
if "--llm" not in sys.argv:
    os.environ["AGENT_LLM_MODE"] = "rules"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

USE_LLM = "--llm" in sys.argv


def report_path(name: str) -> Path:
    """真实模型结果写独立报告，绝不覆盖规则模式的历史基线。"""
    suffix = "_llm" if USE_LLM else ""
    return Path(__file__).resolve().parents[1] / "data" / f"{name}{suffix}.json"


from app.eval.runner import (  # noqa: E402
    compare_full,
    compare_holdout3,
    compare_holdout4,
    compare_holdout5,
    format_report,
    load_cases,
    load_dev2,
    load_holdout2,
    load_holdout3,
    load_holdout4,
    load_holdout5,
    sensitivity_demo,
)


def main() -> int:
    # 第二批留出集单独评测：不并入 48 条全量口径，保持历史可比。
    if "--holdout3" in sys.argv:
        result = compare_holdout3()
        single, multi = result["single"], result["multi"]
        for mode in ("single", "multi"):
            print("=" * 88)
            print(format_report(result[mode]))
        out = report_path("eval_holdout3_report")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "mode": "llm" if USE_LLM else "rules",
                    "methodology": "与 48 条全量相同的严格口径；case 不参与任何调参",
                    "cases": len(load_holdout3()),
                    "single": single,
                    "multi": multi,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"报告已写入：{out}")
        return 0 if multi["success_rate"] >= single["success_rate"] else 1

    # 第三批留出集：B11/B12 修复后新建，holdout3 已因定位失败而被污染。
    if "--holdout4" in sys.argv:
        result = compare_holdout4()
        single, multi = result["single"], result["multi"]
        for mode in ("single", "multi"):
            print("=" * 88)
            print(format_report(result[mode]))
        out = report_path("eval_holdout4_report")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "mode": "llm" if USE_LLM else "rules",
                    "methodology": "与 48 条全量相同的严格口径；case 不参与任何调参",
                    "cases": len(load_holdout4()),
                    "single": single,
                    "multi": multi,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"报告已写入：{out}")
        return 0 if multi["success_rate"] >= single["success_rate"] else 1

    # 第四批留出集：B12/B13 修复后新建，holdout4 已因定位失败而被污染。
    if "--holdout5" in sys.argv:
        result = compare_holdout5()
        single, multi = result["single"], result["multi"]
        for mode in ("single", "multi"):
            print("=" * 88)
            print(format_report(result[mode]))
        out = report_path("eval_holdout5_report")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "mode": "llm" if USE_LLM else "rules",
                    "methodology": "与 48 条全量相同的严格口径；case 不参与任何调参",
                    "cases": len(load_holdout5()),
                    "single": single,
                    "multi": multi,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"报告已写入：{out}")
        return 0 if multi["success_rate"] >= single["success_rate"] else 1

    result = compare_full()
    single, multi = result["single"], result["multi"]

    for mode in ("single", "multi"):
        print("=" * 88)
        print(format_report(result[mode]))
        print()

    print("=" * 88)
    print("对比结论（严格口径）")
    for split in ("dev", "dev2", "holdout2"):
        s = single["by_split"].get(split, {})
        m = multi["by_split"].get(split, {})
        print(
            f"  [{split:7s}] 成功率 单 {s.get('success_rate')}% → 多 {m.get('success_rate')}%"
            f" ｜ 断言正确率 {s.get('answer_correct_rate')}% → {m.get('answer_correct_rate')}%"
            f" ｜ 幻觉 {s.get('hallucination_rate')}% → {m.get('hallucination_rate')}%"
        )
    print(f"  [全量   ] 成功率 单 {single['success_rate']}% → 多 {multi['success_rate']}%")
    print(f"  平均耗时：{single['avg_latency_ms']} ms → {multi['avg_latency_ms']} ms")
    print(f"  平均估算 Token（对比提示词规模）：{single['avg_tokens']} → {multi['avg_tokens']}")
    print(f"  API usage 实测 Token 均值：{single['avg_measured_tokens']} → {multi['avg_measured_tokens']}")
    print(f"  估算费用合计：¥{single['total_estimated_cost_cny']} → ¥{multi['total_estimated_cost_cny']}（规则模式恒为 0）")
    print(f"  转交丢信息：单 {single['handoff_loss']} 次 / 多 {multi['handoff_loss']} 次")
    print(f"  越权拦截率：单 {single['security_block_rate']}% / 多 {multi['security_block_rate']}%")
    print(f"  检查项：严格 {multi['assertions_total']} 项 / 旧口径 {multi['legacy_checks_total']} 项")
    print("  数据集口径：dev / dev2 已参与调参；holdout2 未参与任何调参，是唯一泛化证据")
    print("  注：dev2 是原留出集，在修意图召回那轮被消耗，高分不能外推")

    print()
    print("=" * 88)
    print("口径敏感性自检（同一个坏模型，两套口径同时判）")
    demo = sensitivity_demo()
    print(f"  案例 {demo['case_id']}：{demo['description']}")
    print(
        f"    旧口径：{'通过' if demo['legacy_passed'] else '失败'}"
        f" ｜ 严格口径：{'通过' if demo['strict_passed'] else '失败'}"
    )
    print(f"    严格口径检出：{demo['categories']}")
    for failure in demo["strict_failures"]:
        print(f"      - {failure}")
    print(f"    模型输出摘录：{demo['reply_excerpt'][:110]!r}")

    out = Path(__file__).resolve().parents[1] / "data" / "eval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "methodology": {
            "strict": "逐步断言 + 答案事实/禁词 + 幻觉检测 + 事件留痕，全部通过才算通过",
            "legacy": "仅最后一轮 state / ticket / citation 布尔 / 关键词",
            "dev": "开发集，参与规则调参；高分只说明实现与断言自洽",
            "dev2": "原留出集，修意图召回时用于定位/验证缺陷，已被污染，不再算留出集",
            "holdout2": "修复完成后新建的全新表达，未参与任何调参，是唯一泛化证据",
            "hallucination": "回复中的条款号/订单号/工单号/金额必须来自权威集合；无工单不得出现工单号",
            "token_note": "报告里的 avg_tokens 是 prompt 字符估算，仅供对比提示词规模；avg_measured_tokens 来自 API usage 字段，规则模式恒为 0",
            "billing_note": "只有 API 响应带 usage 才算实测并计费；降级/无 usage/规则模式只有估算且费用为 0；estimated_cost_cny 是按可配置单价换算的估算费用，不是供应商账单",
            "orchestration": "下游 Agent 真实执行工具链与状态机，trace 每步耗时为 perf_counter 实测值，工具调用回填 detail.tools",
        },
        "dev_cases": len(load_cases()),
        "dev2_cases": len(load_dev2()),
        "holdout2_cases": len(load_holdout2()),
        "billing": {
            "mode": multi["billing_basis"],
            "avg_measured_tokens": {"single": single["avg_measured_tokens"], "multi": multi["avg_measured_tokens"]},
            "avg_estimated_tokens": {"single": single["avg_tokens"], "multi": multi["avg_tokens"]},
            "total_estimated_cost_cny": {
                "single": single["total_estimated_cost_cny"],
                "multi": multi["total_estimated_cost_cny"],
            },
            "note": "规则模式没有模型调用：实测 token 与费用均为 0；估算 token 不代表账单",
        },
        "single": single,
        "multi": multi,
        "sensitivity": demo,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入：{out}")
    return 0 if multi["success_rate"] >= single["success_rate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
