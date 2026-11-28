"""holdout5 · 真实模型首次基线（独立报告，不覆盖规则模式口径）。

    python scripts/run_holdout5_llm.py --smoke   # 先验 1 条，确认真的走了 LLM
    python scripts/run_holdout5_llm.py           # 全量 16 条 x 单/多两种模式

设计要点
--------
- 不修改 .env：只在当前进程里把「主模型」指向已验证可用的理解层免费模型
  （CLASSIFY_BASE_URL / CLASSIFY_API_KEY / CLASSIFY_MODEL）。环境变量优先于 .env，
  进程结束即恢复，长期配置不受影响。
- 报告写入 data/eval_holdout5_report_llm.json，与规则模式的
  data/eval_holdout5_report.json 完全隔离。
- 前置校验：billing_basis 必须是 measured（响应带 usage），否则判为无效基线，
  绝不把「静默降级到规则引擎」的结果当成真实模型成绩。
- 评测纪律：holdout5 本轮只做首次观测，失败不用于调参；一旦据此改规则/提示词，
  该批次即视为污染，必须另建零重叠留出集。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _read_env() -> dict[str, str]:
    """直接读 .env，避免在设置覆盖变量前 import app.config。"""
    result: dict[str, str] = {}
    path = ROOT / ".env"
    if not path.exists():
        return result
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


_env = _read_env()
_model = _env.get("CLASSIFY_MODEL", "")
_key = _env.get("CLASSIFY_API_KEY", "")
if not _model or not _key:
    raise SystemExit("缺少 CLASSIFY_MODEL / CLASSIFY_API_KEY，无法运行真实模型基线")

# 环境变量优先于 .env（config.load_dotenv 用 setdefault）；只在进程内生效
import os  # noqa: E402

os.environ["AGENT_LLM_MODE"] = "llm"
os.environ["LLM_BASE_URL"] = _env.get("CLASSIFY_BASE_URL", "")
os.environ["LLM_API_KEY"] = _key
os.environ["LLM_MODEL"] = _model
os.environ["LLM_TIMEOUT"] = _env.get("CLASSIFY_TIMEOUT", "60") or "60"
os.environ["LLM_PRICE_IN_PER_1K"] = "0"
os.environ["LLM_PRICE_OUT_PER_1K"] = "0"

from app.eval.runner import compare_holdout5, format_report, load_holdout5, run  # noqa: E402

SMOKE = "--smoke" in sys.argv


def real_calls_happened(report: dict) -> bool:
    """runner 的 billing_basis 是列表（如 ['measured']），需兼容集合判定。"""
    basis = report.get("billing_basis")
    measured = "measured" in basis if isinstance(basis, (list, tuple, set)) else basis == "measured"
    return measured and report.get("avg_measured_tokens", 0) > 0


def main() -> int:
    if SMOKE:
        cases = load_holdout5()[:1]
        report = run("multi", cases=cases)
        print(format_report(report))
        basis = report.get("billing_basis")
        measured = report.get("avg_measured_tokens", 0)
        print(f"\n真实调用前置校验：basis={basis}, measured_tokens={measured}")
        return 0 if real_calls_happened(report) else 1

    result = compare_holdout5()
    single, multi = result["single"], result["multi"]
    for mode in ("single", "multi"):
        print("=" * 88)
        print(format_report(result[mode]))

    # 硬校验：静默降级的结果不能当真实模型基线
    for label, report in (("single", single), ("multi", multi)):
        if not real_calls_happened(report):
            print(f"[✗] {label} 未发生真实模型调用（basis={report.get('billing_basis')}），基线无效")
            return 1

    rules_path = ROOT / "data" / "eval_holdout5_report.json"
    rules = json.loads(rules_path.read_text(encoding="utf-8")) if rules_path.exists() else {}
    comparison = {
        mode: {
            "rules": rules.get(mode, {}).get("success_rate"),
            "llm": result[mode].get("success_rate"),
        }
        for mode in ("single", "multi")
    }

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "llm",
        "model": _model,
        "base_url": os.environ["LLM_BASE_URL"],
        "methodology": (
            "holdout5 首次真实模型基线；与 48 条全量及规则模式报告完全隔离；"
            "本轮只做观测，不用失败案例调参（否则该留出集即污染）"
        ),
        "billing_basis": {m: r.get("billing_basis") for m, r in result.items()},
        "cases": len(load_holdout5()),
        "rules_baseline": {
            "source": "data/eval_holdout5_report.json",
            "single": rules.get("single", {}).get("success_rate"),
            "multi": rules.get("multi", {}).get("success_rate"),
        },
        "comparison": comparison,
        "single": single,
        "multi": multi,
    }
    out = ROOT / "data" / "eval_holdout5_report_llm.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 88)
    print(
        f"规则模式（历史基线）：单 {comparison['single']['rules']}% / 多 {comparison['multi']['rules']}%"
    )
    print(f"真实模型（{_model}）：单 {single['success_rate']}% / 多 {multi['success_rate']}%")
    print(f"报告已写入：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
