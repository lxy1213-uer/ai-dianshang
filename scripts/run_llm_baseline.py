"""真实模型 vs 规则引擎的抽样基线（B1 / B2 复测共用脚本）。

    python scripts/run_llm_baseline.py                # B1：跑 rules + llm 两侧（真实调用，需费用）
    python scripts/run_llm_baseline.py --rules-only    # 只跑规则侧，离线自检脚本逻辑
    python scripts/run_llm_baseline.py --analyze-only  # 复用已生成报告做口径诊断，不调 API
    python scripts/run_llm_baseline.py --sample2       # B2：另开一轮新建样本（B1 修复后复测）
    python scripts/run_llm_baseline.py --sample3       # B3：另开一轮新建样本（B2 修复后复测）
    python scripts/run_llm_baseline.py --sample4       # B4：同义断言口径修复后的定向探针复测
    python scripts/run_llm_baseline.py --sample5       # B5：语素信号组合修复后的定向探针复测
    python scripts/run_llm_baseline.py --sample6       # B6：完成型结论（建单确认）规格修复后复测
    python scripts/run_llm_baseline.py --sample7       # B7：reason 槽位证据校验修复后复测

为什么要有这个脚本
------------------
README 里那些 100% 成绩全部来自规则引擎，真实模型接进来后从未在严格评测上跑过。
全量 48 条 x 2 模式 x 多轮 = 数百次调用，几十分钟加费用，不划算；这里抽 10 条、
17 轮覆盖各意图类别，量出「接了模型到底变好还是变差」。

B2 与 B1 的区别：B1 首次量出基线（60%），暴露两个问题（意图状态约束缺失、
断言与规则模板的空格耦合），修复后**另开一轮、换用 dev / dev2 中 B1 未用过
的案例**做复测——B1 那批已用于定位失败并驱动修复，按纪律不能边测边改。

评测纪律（写死在报告里，不是口号）
--------------------------------
- 抽样集来自已参与过调参的 dev / dev2，**不是留出集、不是泛化证据**；
  只能回答「同一批题，两种实现分别答成什么样」，不能外推。
- 刻意不含 holdout2 的任何案例，也不并入 load_all_cases()，保护唯一干净的留出集。
- 修复动作只基于上一轮的失败结论；本轮测完不因分数难看继续改规则词表或提示词。
- 真实模型有随机性且只跑一次，分数是「一次观测」不是稳定指标。
- 前置校验确保真的发生了 LLM 调用（basis=measured）；若全部静默降级，结果判为无效。
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import tempfile
from collections import Counter
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent.coordinator import Coordinator  # noqa: E402
from app.config import Settings  # noqa: E402
from app.eval.runner import (  # noqa: E402
    load_llm_sample, load_llm_sample2, load_llm_sample3, load_llm_sample4,
    load_llm_sample5, load_llm_sample6, load_llm_sample7, run,
)

RULES_ONLY = "--rules-only" in sys.argv
SAMPLE2 = "--sample2" in sys.argv
SAMPLE3 = "--sample3" in sys.argv
SAMPLE4 = "--sample4" in sys.argv
SAMPLE5 = "--sample5" in sys.argv
SAMPLE6 = "--sample6" in sys.argv
SAMPLE7 = "--sample7" in sys.argv
FROZEN_TODAY = date(2026, 11, 22)

# B1 = 首次基线；B2 / B3 = 逐轮修复后另开一轮的复测，样本互零重叠，
# B4 = 同义断言口径修复后的定向探针集（dev/dev2 已耗尽，改用新建样本）。
# 报告分开落盘避免互相覆盖。
_ROUND_BY_FLAG = {
    "--sample2": ("B2", load_llm_sample2),
    "--sample3": ("B3", load_llm_sample3),
    "--sample4": ("B4", load_llm_sample4),
    "--sample5": ("B5", load_llm_sample5),
    "--sample6": ("B6", load_llm_sample6),
    "--sample7": ("B7", load_llm_sample7),
}
if SAMPLE7:
    ROUND, load_sample = "B7", load_llm_sample7
elif SAMPLE6:
    ROUND, load_sample = "B6", load_llm_sample6
elif SAMPLE5:
    ROUND, load_sample = "B5", load_llm_sample5
elif SAMPLE4:
    ROUND, load_sample = "B4", load_llm_sample4
elif SAMPLE3:
    ROUND, load_sample = "B3", load_llm_sample3
elif SAMPLE2:
    ROUND, load_sample = "B2", load_llm_sample2
else:
    ROUND, load_sample = "B1", load_llm_sample
REPORT_NAME = {
    "B1": "llm_baseline_report.json", "B2": "llm_baseline2_report.json",
    "B3": "llm_baseline3_report.json", "B4": "llm_baseline4_report.json",
    "B5": "llm_baseline5_report.json", "B6": "llm_baseline6_report.json",
    "B7": "llm_baseline7_report.json",
}
REPORT_PATH = ROOT / "data" / REPORT_NAME[ROUND]


# --------------------------------------------------------------------------
# 前置校验：确认网关真的返回了 usage，而不是全部降级成规则引擎
# --------------------------------------------------------------------------
def preflight() -> dict:
    settings = Settings.load(llm_mode="llm")
    if not settings.llm_enabled:
        return {"ok": False, "reason": "AGENT_LLM_MODE 未启用或 LLM_API_KEY 为空"}
    with tempfile.TemporaryDirectory(prefix="agn_pf_") as tmp:
        s = Settings.load(
            llm_mode="llm",
            db_path=Path(tmp) / "preflight.db",
            frozen_today=FROZEN_TODAY,
        )
        coordinator = Coordinator(s)
        try:
            result = coordinator.chat("preflight_b1", "u_1001", "退货政策是几天？")
        finally:
            coordinator.close()
    usage = result.usage or {}
    measured = usage.get("measured") or {}
    ok = usage.get("basis") == "measured" and measured.get("calls", 0) > 0
    return {
        "ok": ok,
        "model": usage.get("model"),
        "billing_mode": usage.get("billing_mode"),
        "basis": usage.get("basis"),
        "measured_calls": measured.get("calls", 0),
        "measured_tokens": measured.get("total_tokens", 0),
        "latency_ms": result.latency_ms,
        "pricing": usage.get("pricing"),
        "reply_excerpt": (result.reply or "")[:100],
        "reason": None if ok else "响应缺少 usage 或全部降级，无法作为 LLM 基线",
    }


# --------------------------------------------------------------------------
# 逐轮观测：护栏回退 / 真实调用 / 实测 token / 耗时，评测器本身不用改
# --------------------------------------------------------------------------
def _observe(result) -> dict:
    meta = result.meta or {}
    grounding = meta.get("grounding") or {}
    usage = result.usage or {}
    measured = usage.get("measured") or {}
    steps = result.trace or []
    sources = [str((step.get("usage") or {}).get("source") or "") for step in steps]
    return {
        "state": result.state,
        "latency_ms": result.latency_ms,
        "grounding_checked": bool(grounding),
        "grounding_fallback": bool(grounding.get("fallback")),
        "grounding_used_polished": bool(grounding.get("used_polished")),
        "grounding_violations": list(grounding.get("violations") or []),
        "basis": usage.get("basis"),
        "billing_mode": usage.get("billing_mode"),
        "measured_calls": int(measured.get("calls", 0)),
        "measured_total_tokens": int(measured.get("total_tokens", 0)),
        "estimated_cost_cny": usage.get("estimated_cost_cny"),
        "sources": sources,
        "llm_steps": sum(1 for s in sources if s == "llm"),
        "fallback_steps": sum(1 for s in sources if "fallback" in s),
        "reply_excerpt": (result.reply or "")[:200],
    }


def _hook(sink: list[dict]):
    """包装 coordinator.chat，把每轮实测数据收进 sink。

    chat 是唯一顶层入口，内部不会再递归调用它，因此每轮恰好记录一次。
    """

    def install(coordinator: Coordinator) -> None:
        original = coordinator.chat

        def observing(session_id, user_id, message):
            result = original(session_id, user_id, message)
            sink.append(_observe(result))
            return result

        coordinator.chat = observing

    return install


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------
def _ops(sink: list[dict]) -> dict:
    turns = len(sink)
    if not turns:
        return {"turns": 0, "note": "未运行"}
    latencies = [row["latency_ms"] for row in sink]
    violations = Counter(v for row in sink for v in row["grounding_violations"])
    checked = sum(1 for row in sink if row["grounding_checked"])
    fallback = sum(1 for row in sink if row["grounding_fallback"])
    polished = sum(1 for row in sink if row["grounding_used_polished"])
    costs = [row["estimated_cost_cny"] for row in sink if isinstance(row["estimated_cost_cny"], (int, float))]
    return {
        "turns": turns,
        "grounding_checked_turns": checked,
        "grounding_fallback_turns": fallback,
        "grounding_fallback_rate": round(fallback / checked * 100, 1) if checked else None,
        "polish_adopted_turns": polished,
        "polish_adopted_rate": round(polished / checked * 100, 1) if checked else None,
        "grounding_violations": dict(sorted(violations.items())),
        "measured_calls": sum(row["measured_calls"] for row in sink),
        "measured_total_tokens": sum(row["measured_total_tokens"] for row in sink),
        "estimated_cost_cny": round(sum(costs), 6),
        "llm_steps": sum(row["llm_steps"] for row in sink),
        "rules_fallback_steps": sum(row["fallback_steps"] for row in sink),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 1),
            "min": min(latencies),
            "max": max(latencies),
            "total": sum(latencies),
            "median": round(statistics.median(latencies), 1),
        },
    }


def _case_index(report: dict) -> dict:
    return {row["id"]: row for row in report.get("cases", [])}


def _comparison(rules: dict, llm: dict | None, sample: list[dict]) -> dict:
    r_idx = _case_index(rules)
    groups: dict[str, list] = {
        "both_pass": [], "both_fail": [], "regressions": [], "improvements": [],
    }
    if llm is None:
        return groups
    l_idx = _case_index(llm)
    by_id = {case["id"]: case for case in sample}
    for cid in sorted(by_id):
        r, l = r_idx.get(cid, {}), l_idx.get(cid, {})
        rp, lp = bool(r.get("passed")), bool(l.get("passed"))
        key = "both_pass" if rp and lp else "both_fail" if not rp and not lp else (
            "regressions" if rp else "improvements"
        )
        groups[key].append(
            {
                "id": cid,
                "name": by_id[cid].get("name"),
                "focus": by_id[cid].get("focus"),
                "source_case": by_id[cid].get("source_case"),
                "rules_passed": rp,
                "llm_passed": lp,
                "rules_failures": r.get("failures", []),
                "llm_failures": l.get("failures", []),
                "llm_reply_excerpt": l.get("reply_excerpt", ""),
            }
        )
    return groups


def _pct(value) -> str:
    return "—" if value is None else f"{value}%"


def _conclusion(rules: dict, llm: dict | None, ops_llm: dict | None) -> dict:
    if llm is None:
        return {"verdict": "仅跑了规则侧（--rules-only），不构成对照结论"}
    delta = round(llm["success_rate"] - rules["success_rate"], 1)
    if delta < 0:
        verdict = "真实模型在这批抽样上更差"
    elif delta > 0:
        verdict = "真实模型在这批抽样上更好"
    else:
        verdict = "两种实现在这批抽样上持平"
    note = (
        "这只是 10 条抽样的一次观测，样本来自已调参的 dev / dev2 或定向探针，不是泛化证据；"
        "模型有随机性，分数是一次观测不是稳定指标。"
    )
    if SAMPLE7:
        note += (
            "本轮为 B6 修复后的复测（定向探针集）：修复 = reason 槽位证据校验"
            "（app/slots.py validate_reason，接入 app/llm.py 两条分类路径）。"
            "声称质量问题但用户原话里没有缺陷证据时降级为 no_reason——「刻字 / 定制」"
            "属个性化属性、「已拆封 / 洗过」属使用状态、被否定的缺陷（「没有破损」）"
            "都不算证据；只做减法、不升级，intent 仍归 B1 状态护栏。修复动作只基于"
            "B6 的失败归因（x09），未参考本轮结果。"
        )
    elif SAMPLE6:
        note += (
            "本轮为 B5 修复后的复测（定向探针集）：修复 = 完成型结论规格"
            "（app/eval/runner.py _COMPLETION_SIGNAL_SPECS / _contains_by_completion），"
            "建单确认按「动作邻域窗口 + 完成语素 + 对象语素」判定，并加否定硬闸"
            "（工单没有创建成功 / 未能创建工单 一律不放行）；同时把禁词断言改为同一判定，"
            "让「工单已成功创建」在禁词侧同样算命中；护栏入口清洗零宽等不可见字符"
            "（app/grounding.py strip_invisible）。修复动作只基于 B5 的失败结论，"
            "未参考本轮结果。"
        )
    elif SAMPLE5:
        note += (
            "本轮为 B4 修复后的复测（定向探针集）：修复 = 语素信号组合，严格口径"
            "必含事实断言按「极性语素 + 动作语素」组合判定（app/eval/runner.py "
            "_FACT_SIGNAL_SPECS / _contains_by_signals），覆盖枚举词典漏掉的"
            "「未查到订单 / 未能查询到订单 / 不支持重复申请」等说法。极性语素是硬门槛，"
            "反转与数字/单号/结构短语仍不放宽，护栏同步补全「已经查到」反转词。"
            "修复动作只基于 B4 的失败归因，未参考本轮结果。"
        )
    elif SAMPLE4:
        note += (
            "本轮为 B3 修复后的复测（定向探针集）：修复 = 受控同义词典，严格口径的"
            "必含事实断言支持事实等价改写（app/eval/runner.py _contains_fact，"
            "禁词断言与护栏均不放宽）。B3 唯一失败 u10「没有查到订单→未查询到订单」"
            "属事实等价、非幻觉，本轮专测这类同义改写是否被正确接受。修复动作只基于"
            "B3 的失败结论，未参考本轮结果。"
        )
    elif SAMPLE3:
        note += (
            "本轮为 B2 修复后的复测（另开一轮、新建样本）：修复 = 润色结构性短语护栏"
            "（app/grounding.py REQUIRED_PHRASES + missing_phrase 校验，丢「转人工 / "
            "交接信息 / 运费承担方」等结论性短语即回退模板）+ polish 提示词软约束。"
            "修复动作只基于 B2 的失败结论，未参考本轮结果。"
        )
    elif SAMPLE2:
        note += (
            "本轮为 B1 修复后的复测（另开一轮、新建样本）：修复 = LLM 意图状态硬护栏"
            "（app/llm.py needs_state_correction）+ 提示词状态约束与 few-shot +"
            "严格断言空白归一（app/eval/runner.py _contains）。修复动作只基于 B1"
            "的失败结论，未参考本轮结果。"
        )
    else:
        note += "本次结果不用于调参。"
    return {
        "verdict": verdict,
        "delta_success_rate_pp": delta,
        "rules_success_rate": rules["success_rate"],
        "llm_success_rate": llm["success_rate"],
        "note": note,
        "measured_calls": (ops_llm or {}).get("measured_calls", 0),
    }


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def diagnose(rules: dict, llm: dict | None, sample: list[dict]) -> dict:
    """口径敏感性诊断：把失败拆成「空格/格式差异」与「真实语义或流程退化」。

    这是**诊断性复算，不是正式成绩**：不改任何断言、不改规则、不改提示词，
    也不回写 success_rate。目的是解释低分从哪来，避免把 60% 误读成「模型答错了」。

    只基于报告里的最终回复摘录（前 200 字）复算；多轮案例的中间轮不参与，
    因此 whitespace_only 的条数是保守下界。
    """
    if llm is None:
        return {"available": False, "reason": "未运行 LLM 侧（--rules-only），无失败可诊断"}
    l_idx = _case_index(llm)
    whitespace_only: list[dict] = []
    semantic: list[dict] = []
    for case in sample:
        row = l_idx.get(case["id"])
        if not row or row.get("passed"):
            continue
        excerpt = row.get("reply_excerpt", "") or ""
        failures = row.get("failures", []) or []
        # 只复算最终轮：reply_excerpt 是最后一轮的回复，拿中间轮的必含词去比会误判
        steps = case.get("steps", [])
        final_step = steps[-1] if steps else {}
        required: list[str] = []
        required += list(final_step.get("reply_contains") or [])
        required += list(case.get("facts") or [])
        required += list((case.get("expect") or {}).get("reply_contains") or [])
        required = list(dict.fromkeys(required))
        missing_strict = [x for x in required if x not in excerpt]
        missing_compact = [x for x in required if _compact(x) not in _compact(excerpt)]
        flow_signals = ("状态应为", "意图应为", "工单", "事件", "meta", "不得", "应含", "缺少条款")
        has_flow_failure = any(signal in " ".join(failures) for signal in flow_signals)
        if not has_flow_failure and missing_strict and not missing_compact:
            whitespace_only.append(
                {"id": case["id"], "name": case["name"], "recovered_by_compact": missing_strict}
            )
        else:
            semantic.append(
                {
                    "id": case["id"],
                    "name": case["name"],
                    "missing_after_compact": missing_compact,
                    "failures": failures[:4],
                }
            )
    hypothetical = round((llm["passed"] + len(whitespace_only)) / llm["total"] * 100, 1)
    return {
        "available": True,
        "formal_success_rate": llm["success_rate"],
        "whitespace_only_failures": whitespace_only,
        "semantic_or_flow_failures": semantic,
        "hypothetical_rate_if_compact": hypothetical,
        "note": (
            "诊断性复算，不是正式成绩：hypothetical_rate 只是说明「若评测器采用去空白匹配，"
            "这几条格式差异案例会通过」，正式成绩仍按现有严格口径记为 "
            f"{llm['success_rate']}%。本次不修改任何断言/规则/提示词，不用于调参。"
            "仅基于最终回复前 200 字摘录与最终轮断言，中间轮状态/意图错误不在此复算内。"
        ),
    }


def _print_diagnosis(d: dict) -> None:
    print("=" * 90)
    print("口径敏感性诊断（解释低分来源，不改变正式成绩）")
    print("=" * 90)
    if not d.get("available"):
        print(f"  {d.get('reason')}")
        return
    w, s = d["whitespace_only_failures"], d["semantic_or_flow_failures"]
    print(f"  仅空格/格式差异导致失败：{len(w)} 条  " + ", ".join(x["id"] for x in w))
    for item in w:
        print(f"      - {item['id']} 去空白后可匹配：{item['recovered_by_compact']}")
    print(f"  真实语义 / 流程退化：{len(s)} 条  " + ", ".join(x["id"] for x in s))
    for item in s:
        print(f"      - {item['id']} {item['name']}")
        if item["missing_after_compact"]:
            print(f"        去空白后仍缺失：{item['missing_after_compact']}")
        print(f"        失败项：{'; '.join(item['failures'][:3])}")
    print(f"  若改为去空白匹配，成功率将是 {d['hypothetical_rate_if_compact']}%（仅诊断，正式成绩仍为 {d['formal_success_rate']}%）")
    print(f"  {d['note']}")
    print()


def analyze_existing() -> int:
    """复用已有报告做离线诊断，不重跑模型、不产生 API 费用。"""
    if not REPORT_PATH.exists():
        print(f"找不到报告：{REPORT_PATH}，请先完整跑一次")
        return 2
    payload = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    diagnosis = diagnose(payload.get("rules"), payload.get("llm"), load_sample())
    payload["diagnosis"] = diagnosis
    REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_diagnosis(diagnosis)
    print(f"诊断结果已回写：{REPORT_PATH}（success_rate 等正式成绩未改动）")
    return 0


def _print_sample(sample: list[dict]) -> None:
    turns = sum(len(case["messages"]) for case in sample)
    print("=" * 90)
    print(f"{ROUND} · LLM 模式抽样评测（真实模型 vs 规则引擎）")
    print("=" * 90)
    print(f"样本：{len(sample)} 条 / {turns} 轮 ｜ split={sample[0].get('split')} ｜ agent_mode=multi ｜ 严格四层断言")
    if SAMPLE5:
        print("来源：新建定向探针集 → 验证语素信号组合修复（B4 枚举词典漏说法），不是泛化证据；不含 holdout2")
    elif SAMPLE4:
        print("来源：新建定向探针集（dev / dev2 未用案例已耗尽）→ 验证同义断言口径修复，不是泛化证据；不含 holdout2")
    elif SAMPLE3:
        print("来源：dev / dev2 中 B1/B2 均未用过的案例（均已参与调参）→ 修复后复测，不是泛化证据；不含 holdout2")
    elif SAMPLE2:
        print("来源：dev / dev2 中 B1 未用过的案例（均已参与调参）→ 修复后复测，不是泛化证据；不含 holdout2")
    else:
        print("来源：dev / dev2（均已参与调参）→ 只做能力对照，不是泛化证据；不含 holdout2")
    for case in sample:
        print(f"  {case['id']}  [{case['source_case']}]  {case['focus']}  · {len(case['messages'])} 轮")
    print()


def _print_comparison(rules: dict, llm: dict | None, ops: dict, comparison: dict) -> None:
    print("=" * 90)
    print("对照结果（严格口径）")
    print("=" * 90)
    rows = [
        ("严格成功率", "success_rate", "%"),
        ("旧口径成功率", "legacy_success_rate", "%"),
        ("答案正确率", "answer_correct_rate", "%"),
        ("逐步通过率", "step_pass_rate", "%"),
        ("幻觉率", "hallucination_rate", "%"),
        ("越权拦截率", "security_block_rate", "%"),
        ("平均单案例耗时(ms)", "avg_latency_ms", ""),
        ("平均估算 Token", "avg_tokens", ""),
        ("API 实测 Token 均值", "avg_measured_tokens", ""),
        ("估算费用合计(¥)", "total_estimated_cost_cny", ""),
        ("转交丢信息", "handoff_loss", ""),
    ]
    print(f"  {'指标':<24}{'规则引擎':>14}{'真实模型':>14}")
    print("  " + "-" * 52)
    for label, key, unit in rows:
        rv = rules.get(key)
        lv = llm.get(key) if llm else None
        r_text = "—" if rv is None else f"{rv}{unit}"
        l_text = "—" if lv is None or llm is None else f"{lv}{unit}"
        print(f"  {label:<24}{r_text:>14}{l_text:>14}")
    print(f"  {'严格 / 旧口径检查项':<24}{rules['assertions_total']:>7}/{rules['legacy_checks_total']:<6}"
          f"{(str(llm['assertions_total']) + '/' + str(llm['legacy_checks_total'])) if llm else '—':>14}")
    print()

    if llm is None:
        print("  （--rules-only：未跑 LLM 侧）")
        return

    print("  逐案例差异：")
    for key, title in (
        ("regressions", "变差（规则通过 → 模型失败）"),
        ("improvements", "变好（规则失败 → 模型通过）"),
        ("both_fail", "两侧都失败"),
        ("both_pass", "两侧都通过"),
    ):
        items = comparison[key]
        print(f"    {title}：{len(items)} 条  " + ", ".join(x["id"] for x in items))
        if key in ("regressions", "both_fail"):
            for item in items:
                fails = item["llm_failures"][:3]
                print(f"      - {item['id']} {item['name']}")
                if fails:
                    print(f"        模型侧失败：{'; '.join(fails)}")
                if item["llm_reply_excerpt"]:
                    print(f"        回复摘录：{item['llm_reply_excerpt'][:100]!r}")
    print()

    print("  LLM 侧运营指标（规则引擎无模型调用，不参与比较）：")
    o = ops["llm"]
    print(f"    实际 LLM 调用：{o['measured_calls']} 次 ｜ 降级步骤：{o['rules_fallback_steps']}")
    print(f"    实测 Token 合计：{o['measured_total_tokens']} ｜ 估算费用：¥{o['estimated_cost_cny']}（估算，非账单）")
    lat = o["latency_ms"]
    print(f"    单轮耗时：均值 {lat['mean']} ms / 中位 {lat['median']} / 最小 {lat['min']} / 最大 {lat['max']} ｜ 合计 {lat['total']} ms")
    print(f"    护栏回退率：{_pct(o['grounding_fallback_rate'])}（{o['grounding_fallback_turns']}/{o['grounding_checked_turns']} 轮）"
          f" ｜ 润色采用率：{_pct(o['polish_adopted_rate'])}")
    print(f"    护栏违规分布：{o['grounding_violations'] or '无'}")
    print()

    conclusion = _conclusion(rules, llm, o)
    print("=" * 90)
    print(f"结论：{conclusion['verdict']}（严格成功率 {conclusion['rules_success_rate']}% → "
          f"{conclusion['llm_success_rate']}%，差 {conclusion['delta_success_rate_pp']:+} pp）")
    print("=" * 90)
    print(f"  {conclusion['note']}")
    if o["measured_calls"] == 0:
        print("  ⚠ 警告：本次没有任何实测 LLM 调用，结果不能代表真实模型，请检查网关/降级。")


def main() -> int:
    if "--analyze-only" in sys.argv:
        return analyze_existing()
    sample = load_sample()
    _print_sample(sample)

    rules_sink: list[dict] = []
    rules = run(
        "multi",
        cases=sample,
        settings_overrides={"llm_mode": "rules"},
        coordinator_hook=_hook(rules_sink),
    )

    llm = None
    llm_sink: list[dict] = []
    pf = {"ok": None, "reason": "--rules-only，跳过真实模型"}
    if not RULES_ONLY:
        print("前置校验：确认网关可用且响应带 usage ...")
        pf = preflight()
        print(f"  model={pf.get('model')} basis={pf.get('basis')} "
              f"measured_calls={pf.get('measured_calls')} latency={pf.get('latency_ms')} ms")
        if not pf["ok"]:
            print(f"  ✗ {pf.get('reason')} —— 中止，不产出会误导人的 LLM 基线")
            return 2
        print(f"  ✓ 真实模型链路可用，开始抽样评测（{len(sample)} 条 / {sum(len(c['messages']) for c in sample)} 轮，约数分钟）\n")
        llm = run(
            "multi",
            cases=sample,
            settings_overrides={"llm_mode": "llm"},
            coordinator_hook=_hook(llm_sink),
        )

    ops = {"rules": _ops(rules_sink), "llm": _ops(llm_sink)}
    comparison = _comparison(rules, llm, sample)
    _print_comparison(rules, llm, ops, comparison)
    diagnosis = diagnose(rules, llm, sample)
    _print_diagnosis(diagnosis)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "methodology": {
            "purpose": (
                "B6 修复后的定向探针复测：验证 reason 槽位证据校验挡住「个性化属性 / 使用状态 /"
                "否定缺陷」被误判成质量问题，同时不误伤真实缺陷的质量通道"
                if SAMPLE7 else
                "B5 修复后的定向探针复测：验证完成型结论（建单确认）规格接住插入语改写，"
                "且禁词侧与不可见字符清洗同时生效" if SAMPLE6 else
                "B4 修复后的定向探针复测：验证语素信号组合接住枚举词典漏掉的等价说法" if SAMPLE5 else
                "B3 修复后的定向探针复测：新建专门覆盖结论短语同义改写的样本集，"
                "验证受控同义词典让事实等价改写被严格口径正确接受" if SAMPLE4 else
                "B2 修复后另开一轮的复测：换用 B1/B2 均未用过的 dev / dev2 案例，"
                "验证润色结构性短语护栏的修复效果" if SAMPLE3 else
                "B1 修复后另开一轮的复测：换用 B1 未用过的 dev / dev2 案例，"
                "验证意图状态护栏与断言空白归一的修复效果" if SAMPLE2 else
                "第一次量出真实模型 vs 规则引擎的严格评测对照基线（B1）"
            ),
            "round": ROUND,
            "fixes_between_rounds": (
                [
                    "app/slots.py validate_reason + defect_evidence：reason 槽位证据校验"
                    "（缺陷语素 + 否定邻域；个性化属性 / 使用状态单列，不算证据）",
                    "app/llm.py RuleLLM._classify_rules / OpenAICompatibleLLM.classify_with_usage："
                    "两条分类路径接入同一校验，降级记入 slots.reason_correction 并在 trace 留痕",
                    "app/agent/coordinator.py：接待 Agent 步骤输出 reason_correction，"
                    "护栏动作对上层可见",
                ] if SAMPLE7 else
                [
                    "app/eval/runner.py _COMPLETION_SIGNAL_SPECS + _contains_by_completion："
                    "完成型结论按动作邻域窗口（完成语素 + 对象语素）判定，否定语素紧邻动作时硬拦",
                    "app/eval/runner.py _forbidden_present：禁词断言与必含断言同一判定，"
                    "「工单已成功创建」在禁词侧同样算命中",
                    "app/grounding.py strip_invisible：护栏入口清洗零宽 / 方向控制符等不可见字符",
                ] if SAMPLE6 else
                [
                    "app/eval/runner.py _FACT_SIGNAL_SPECS + _contains_by_signals：严格口径"
                    "必含事实断言按语素信号组合判定，覆盖枚举词典外的等价说法",
                    "app/grounding.py _NEGATION_FLIPS：补全「已经查到 / 查到了订单」反转词（安全底线）",
                ] if SAMPLE5 else
                [
                    "app/eval/runner.py _contains_fact + _EQUIVALENT_PHRASES：严格口径必含"
                    "事实断言支持受控同义等价（禁词断言与生产护栏均不放宽）",
                ] if SAMPLE4 else
                [
                    "app/grounding.py REQUIRED_PHRASES + missing_phrase：润色丢结构性短语硬校验",
                    "app/llm.py polish 提示词：关键短语不得压缩省略（软约束）",
                ] if SAMPLE3 else
                [
                    "app/llm.py needs_state_correction：LLM 意图状态一致性硬护栏",
                    "app/llm.py classify 提示词：状态约束 + few-shot（例句全部新造）",
                    "app/eval/runner.py _contains：严格断言空白归一匹配（旧口径保持原样）",
                ] if SAMPLE2 else []
            ),
            "strict": "逐步断言 + 答案事实/禁词 + 幻觉检测 + 事件留痕，全部通过才算通过",
            "sample_split": sample[0].get("split"),
            "not_generalization": "不是留出集；不能外推泛化能力；holdout2 未被触碰",
            "no_tuning": "修复动作只基于上一轮失败结论，未参考本轮结果；本轮测完不再改规则/提示词",
            "single_run": "真实模型只跑一次且有随机性，分数是一次观测不是稳定指标",
            "billing": "estimated_cost_cny 按参考价 x 倍率换算，是估算费用不是供应商账单",
            "guard": "前置校验要求 basis=measured 且 measured_calls>0，否则中止",
        },
        "config": {
            "agent_mode": "multi",
            "frozen_today": FROZEN_TODAY.isoformat(),
            "rules_only": RULES_ONLY,
            "sample_cases": len(sample),
            "sample_turns": sum(len(c["messages"]) for c in sample),
            "sample_focus": [c["focus"] for c in sample],
            "sample_sources": [c["source_case"] for c in sample],
        },
        "preflight": pf,
        "ops": ops,
        "comparison": comparison,
        "diagnosis": diagnosis,
        "conclusion": _conclusion(rules, llm, ops.get("llm")),
        "rules": rules,
        "llm": llm,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入：{REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
