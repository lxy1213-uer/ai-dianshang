"""评测器 v2：同一批案例在指定 Agent 模式下跑一遍，输出宽松 / 严格两套口径。

为什么要两套口径
----------------
旧口径（`legacy_success_rate`）只看最后一轮的 state、是否建单、citation 布尔值与关键词。
它有三个盲点：多轮中间步骤走错也可能通过；`citation: true` 不代表条款号有效或答案正确；
案例文本与规则词表同源（照着答案出题），分数不能外推。

严格口径（`success_rate`）= 下面四层断言全部通过：

1. **逐步**：`steps` 与 `messages` 一一对应，每轮检查 state / intent / 必含与必不含文本 /
   条款号 / 工单 / meta，不再只看最后一轮。
2. **答案正确性**：`facts` 必含事实（含工单号自动检查）、`forbid` 禁词与语义反转。
3. **幻觉检测**（全局自动，无需在案例里声明）：回复中的条款号、订单号、工单号、金额
   必须来自权威集合（知识库 / 订单库 / 工单表 / 用户本人在本会话发过的编号）；
   citations 的 clause_id 必须真实存在；无工单时不得出现工单号；有工单时工单号必须出现在回复里；
   政策外承诺词零容忍。
4. **可追溯**：`require_events` / `forbid_events` 校验会话事件日志，覆盖越权留痕、
   换单令牌失效、幂等建单尝试、转人工交接等「只靠回复看不出来」的行为。

数据集划分（评测纪律）
----------------------
- `cases.json`（dev）：开发集，参与调参。
- `cases_holdout.json`（dev2）：最初是留出集，但在「修意图召回」那轮被用于
  定位与验证缺陷，**已被污染、不再具备留出意义**，因此改标 dev2。它的分数
  只能说明修复覆盖了已知说法，不能当泛化证据。
- `cases_holdout2.json`（holdout2）：修复完成后新建，措辞全新，写作时未参考
  任何失败结果、未据此改动一行规则，**不参与任何调参**，才是泛化能力的证据。
- `cases_llm_sample.json`（llm_sample）：B1 抽样基线专用，**从 dev / dev2 复制**，
  只用于「真实模型 vs 规则引擎」的相对对照。它不是留出集、不是泛化证据，
  不并入 `load_all_cases()`，以免污染 48 条全量口径。刻意不含 holdout2 的任何案例。
- `cases_llm_sample2.json`（llm_sample2）：B2 复测集。B1 样本已用于定位失败并
  驱动修复（意图状态护栏 + 断言空白归一），按纪律不再边测边改；本轮换用
  dev / dev2 中 B1 未用过的案例另开一轮，与 B1 零重叠。

一次针对留出集失败案例的调参，就会永久消耗掉那批留出集；这是必须补第二批的原因。

指标
----
- `success_rate`：严格口径通过率（案例级，全部断言通过才算通过）
- `legacy_success_rate`：旧口径通过率，保留用于对照
- `answer_correct_rate`：断言级通过率（所有检查项逐项计数）
- `step_pass_rate`：逐步断言通过率（一个 step 内全部断言通过才算该 step 通过）
- `hallucination_rate`：出现幻觉类违规的案例占比
- `security_block_rate`：要求越权拦截的案例中确实拦住的比例
- `handoff_loss`：接待已识别订单号，但办理后上下文里丢失
- 耗时与 Token 口径不变；多 Agent 按各步提示词累加，单 Agent 按大一统提示词累加
"""

from __future__ import annotations

import json
import re
import statistics
import tempfile
import time
from collections import Counter
from datetime import date
from pathlib import Path

from ..agent.coordinator import Coordinator
from ..config import Settings
from ..grounding import FORBIDDEN_PROMISES

CASES_PATH = Path(__file__).with_name("cases.json")
HOLDOUT_PATH = Path(__file__).with_name("cases_holdout.json")
HOLDOUT2_PATH = Path(__file__).with_name("cases_holdout2.json")
HOLDOUT3_PATH = Path(__file__).with_name("cases_holdout3.json")
HOLDOUT4_PATH = Path(__file__).with_name("cases_holdout4.json")
HOLDOUT5_PATH = Path(__file__).with_name("cases_holdout5.json")
LLM_SAMPLE_PATH = Path(__file__).with_name("cases_llm_sample.json")
LLM_SAMPLE2_PATH = Path(__file__).with_name("cases_llm_sample2.json")
LLM_SAMPLE3_PATH = Path(__file__).with_name("cases_llm_sample3.json")
LLM_SAMPLE4_PATH = Path(__file__).with_name("cases_llm_sample4.json")
LLM_SAMPLE5_PATH = Path(__file__).with_name("cases_llm_sample5.json")
LLM_SAMPLE6_PATH = Path(__file__).with_name("cases_llm_sample6.json")
LLM_SAMPLE7_PATH = Path(__file__).with_name("cases_llm_sample7.json")

# 幻觉类违规：用于 hallucination_rate 与报告里的分类统计
HALLUCINATION_CATEGORIES = (
    "fabricated_id", "fabricated_ticket", "fabricated_amount",
    "invalid_citation", "phantom_ticket", "ungrounded_promise", "missing_ticket_id",
)

_CLAUSE_RE = re.compile(r"R-\d{2}", re.I)
_TICKET_RE = re.compile(r"(?<![A-Za-z0-9])RT\d{8}[A-Za-z0-9]{6}", re.I)
_AMOUNT_RE = re.compile(r"¥\s*(\d+(?:\.\d+)?)")
# 长编号：订单号 O...、工单号 RT...、运单号 JD.../SF... 都在这个形态里
_LONG_ID_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{1,3}\d{8,}[A-Za-z0-9]*", re.I)


def load_cases(path: Path | None = None) -> list[dict]:
    """默认加载开发集（dev）。"""
    return json.loads(Path(path or CASES_PATH).read_text(encoding="utf-8"))


def load_holdout(path: Path | None = None) -> list[dict]:
    """历史别名：cases_holdout.json 现标 dev2（已被调参污染）。"""
    return json.loads(Path(path or HOLDOUT_PATH).read_text(encoding="utf-8"))


# 语义更准确的名字，供新代码与文档使用
load_dev2 = load_holdout


def load_holdout2(path: Path | None = None) -> list[dict]:
    """第一批未参与调参的留出集（已被 B1–B7 的观测反复引用，谨慎外推）。"""
    return json.loads(Path(path or HOLDOUT2_PATH).read_text(encoding="utf-8"))


def load_holdout3(path: Path | None = None) -> list[dict]:
    """第二批完全未参与调参的留出集（2026-11-27 新建）。

    不并入 `load_all_cases()`：48 条全量口径要保持历史可比；单独用
    `compare_holdout3()` / `run_eval.py --holdout3` 评测，作为新的泛化证据。
    """
    return json.loads(Path(path or HOLDOUT3_PATH).read_text(encoding="utf-8"))


def load_holdout4(path: Path | None = None) -> list[dict]:
    """第三批留出集（B11/B12 修复后新建）。

    holdout3 的失败结论已用于定位口语召回缺口，按纪律视为已污染；
    本批与其零重叠，作为下一轮泛化证据，同样不并入 48 条全量。
    """
    return json.loads(Path(path or HOLDOUT4_PATH).read_text(encoding="utf-8"))


def load_holdout5(path: Path | None = None) -> list[dict]:
    """第四批留出集（B12 状态型问句 + B13 引用完整性修复后新建）。

    holdout4 的两条失败已用于定位修复方向，按纪律视为污染；本批与其零重叠，
    同样不并入 48 条全量，用 `compare_holdout5()` / `run_eval.py --holdout5` 单独评测。
    """
    return json.loads(Path(path or HOLDOUT5_PATH).read_text(encoding="utf-8"))


def load_llm_sample(path: Path | None = None) -> list[dict]:
    """B1 抽样基线集（llm_sample）。刻意不并入 load_all_cases()。

    这批案例从 dev / dev2 复制而来（均已参与过调参），只用于比较
    「同一批题，真实模型与规则引擎分别答成什么样」，不能外推为泛化能力。
    """
    return json.loads(Path(path or LLM_SAMPLE_PATH).read_text(encoding="utf-8"))


def load_llm_sample2(path: Path | None = None) -> list[dict]:
    """B2 复测集（llm_sample2）：B1 暴露问题修复后另开一轮的新建样本。

    与 B1 样本零重叠（案例全部换新），同样只从已参与调参的 dev / dev2
    复制、不含 holdout2、不并入 load_all_cases()。B1 那批已用于定位
    失败并驱动修复，按「留出集一旦用于调参就永久消耗」的纪律不再复用。
    """
    return json.loads(Path(path or LLM_SAMPLE2_PATH).read_text(encoding="utf-8"))


def load_llm_sample3(path: Path | None = None) -> list[dict]:
    """B3 复测集（llm_sample3）：B2 暴露问题修复后另开一轮的新建样本。

    与 B1 / B2 样本零重叠（dev / dev2 剩余未用案例），不含 holdout2、
    不并入 load_all_cases()。B2 那批（t08/t09 润色丢结构性短语）已用于
    定位失败并驱动修复（grounding 关键短语校验），按纪律不再复用。
    """
    return json.loads(Path(path or LLM_SAMPLE3_PATH).read_text(encoding="utf-8"))


def load_llm_sample4(path: Path | None = None) -> list[dict]:
    """B4 复测集（llm_sample4）：同义断言口径修复后另开一轮的定向探针样本。

    dev / dev2 的未用案例已耗尽（仅剩 h07 / h14），按文档规划改用「扩展评测集」
    策略：这批 10 条为**新建的定向探针**，消息措辞全新，专门覆盖受控同义词典
    里的四类结论短语（查无订单 / 尚未签收 / 已退款 / 不能重复申请）及「不放宽」
    场景（数字 / 结构性短语 / 工单号）。与 B1/B2/B3 零重叠、不含 holdout2、
    不并入 load_all_cases()。它同样不是泛化证据，只验证同义口径修复是否生效。
    """
    return json.loads(Path(path or LLM_SAMPLE4_PATH).read_text(encoding="utf-8"))


def load_llm_sample5(path: Path | None = None) -> list[dict]:
    """B5 复测集（llm_sample5）：语素信号组合修复后的定向探针样本。

    dev / dev2 已耗尽，沿用 B4 的「新建定向探针」策略：10 条消息措辞全新，
    与 B1–B4 零重叠、不含 holdout2、不并入 load_all_cases()。断言仍写规范
    结论短语（保证规则侧锚点通过），靠真实模型的等价改写验证信号组合是否
    接住了 B4 词典漏掉的说法。它不是泛化证据。
    """
    return json.loads(Path(path or LLM_SAMPLE5_PATH).read_text(encoding="utf-8"))


def load_llm_sample6(path: Path | None = None) -> list[dict]:
    """B6 复测集（llm_sample6）：建单确认（完成型）规格修复后的定向探针样本。

    上游证据来自 B5 实测失败 w10：真实模型把模板「售后工单已创建：RT…」润色成
    「售后工单已成功创建：RT…」，整词断言因插入「成功」漏判。B5 那批已用于
    定位失败并驱动修复，按纪律不再复用，因此另开这批 10 条新措辞探针：
    与 B1–B5 零重叠、不含 holdout2、不并入 load_all_cases()。
    断言仍写规范结论短语（规则侧锚点必须通过），由真实模型的等价改写检验
    完成型规格是否接得住；它同样不是泛化证据。
    """
    return json.loads(Path(path or LLM_SAMPLE6_PATH).read_text(encoding="utf-8"))


def load_llm_sample7(path: Path | None = None) -> list[dict]:
    """B7 复测集（llm_sample7）：reason 槽位证据校验修复后的定向探针样本。

    上游证据来自 B6 实测失败 x09：「保温杯刻字了能退吗」被理解层判成质量问题
    （reason=quality），绕过「定制类目不支持七天无理由」，错给 15 天质量通道。
    B6 那批已用于定位失败并驱动修复，按纪律不再复用，因此另开这批 11 条新措辞
    探针：与 B1–B6 零重叠、不含 holdout2、不并入 load_all_cases()。

    设计上刻意成对：y01/y02/y03/y05 是「不该给质量通道」的四种说法（个性化属性、
    用户自述质量没问题、使用状态、否定缺陷），y04/y06/y07 是「必须给质量通道」的
    反向保护（真发霉、超 7 天掉色、定制做坏了）。只测减法的护栏最容易把召回一起
    砍掉，成对样本才能在下一轮看出代价。
    """
    return json.loads(Path(path or LLM_SAMPLE7_PATH).read_text(encoding="utf-8"))


def load_all_cases() -> list[dict]:
    return load_cases() + load_holdout() + load_holdout2()


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _contains(text: str, keyword: str) -> bool:
    """子串断言的空白不敏感匹配：先原文匹配，再对去空白副本匹配。

    B1 真实模型抽样暴露的脆性：模型答「7天内」，断言写「7 天」，事实全对
    只差一个空格，严格口径误判为 missing_fact。去空白匹配只消除排版差异，
    不做模糊或语义匹配——「7 天」不会匹配「六天」，数字缺失仍然失败。
    仅用于严格口径（step / final 断言）；旧口径保留原样，继续作为历史对照。

    B4 起，必含事实断言走 `_contains_fact`（本函数 + 受控同义）；B6 起禁词断言
    也走同一判定（`_forbidden_present`），因为改写说法同样必须被视为「出现了该事实」。
    旧口径（legacy）仍走本函数，保持历史分数可比。
    """
    if not keyword:
        return True
    if keyword in (text or ""):
        return True
    return _compact(keyword) in _compact(text or "")


# B4：受控同义等价集。只收录「事实完全一致、仅换措辞」的结论性短语，
# 绝不收录会改变真值的表达（否定反转 / 数字 / 单号一律不放宽）。
# 每条等价组都是闭环：组内任意成员互相可替换且不改变事实结论。
_EQUIVALENT_PHRASES: tuple[tuple[str, ...], ...] = (
    ("没有查到订单", "未查询到订单", "没查到订单", "未查到该订单"),
    ("尚未签收", "还未签收", "未签收", "还没签收"),
    ("已退款", "已经退款", "已完成退款"),
    ("不能重复申请", "不可重复申请", "无法重复申请"),
)

# 展开为 compact 形式的成员 -> 规范化代表词，便于 O(1) 命中；启动时构建一次。
_EQUIVALENT_INDEX: dict[str, str] = {
    _compact(member): _compact(group[0])
    for group in _EQUIVALENT_PHRASES
    for member in group
}

# B5：语素信号组合（与 llm.py semantic_signals 同一哲学）。
# B4 枚举整词在复测里被「未查到订单 / 未能查询到订单 / 不支持重复申请」绕过——
# 枚举永远追不上措辞创造力。改为拆成语素组：每组代表一个必须出现的语义面
# （极性 + 动作），组内任一语素命中即可，全部组都命中才算通过。
#
# 设计要点（防止「信号」变成放水）：
# - 每组都必须命中，**极性组是硬门槛**：没有否定/完成语素就绝不放行，
#   于是「查到订单」不会通过「没有查到订单」断言，「已签收」不会通过「尚未签收」；
# - 语素只覆盖这四类结论性短语，数字 / 单号 / 条款号 / 结构性短语（运费承担方）
#   根本不在这里，行为与精确匹配完全相同；
# - 不做编辑距离、不做模糊、不引语义模型，纯确定性字符串包含，可离线复现。
_FACT_SIGNAL_SPECS: dict[str, tuple[tuple[str, ...], ...]] = {
    # 查无订单：否定极性 + 查动作（覆盖 未查到 / 未能查询到 / 没查到 / 查无）
    "没有查到订单": (
        ("没有", "未", "没", "未能", "没能", "查无"),
        ("查到", "查询到", "查", "找到", "检索", "搜到"),
    ),
    # 尚未签收：否定极性 + 签收动作（覆盖 还未签收 / 没签收 / 未收到）
    "尚未签收": (
        ("尚未", "还没", "还未", "没有", "未", "没"),
        ("签收", "收到"),
    ),
    # 已退款：完成极性 + 退款动作（覆盖 已经退款 / 已完成退款）
    "已退款": (
        ("已", "已经", "完成"),
        ("退款", "退钱", "退回"),
    ),
    # 不能重复申请：否定极性 + 重复动作（覆盖 不支持 / 无法 / 不允许 重复申请）
    "不能重复申请": (
        ("不能", "不可", "无法", "不支持", "不允许", "不予", "不再"),
        ("重复", "再次", "二次", "再申请"),
    ),
}


def _contains_by_signals(text: str, keyword: str) -> bool:
    """B5：语素信号组合匹配。断言关键词必须整体命中一个信号规格的 key。"""
    key = _compact(keyword)
    groups = _FACT_SIGNAL_SPECS.get(key)
    if groups is None:
        return False
    compact_text = _compact(text)
    return all(any(_compact(sig) in compact_text for sig in group) for group in groups)


# B10：**结论短语的语义槽**（B6 x01「超过7天退货期限」、B7 y11「还在待发货状态」
# 连续两轮因同一类措辞耦合失分。继续给整词表加成员是错的解法——模型换一种说法
# 就再失一次。这里把「期限已过」「尚未发货」抽成语义槽：槽内有等价说法集合，
# 槽外有**同一小句内的极性闸**，语义等价放行、极性反转绝不放行。
#
# 与 B5 语素信号的区别：B5 判的是「否定 + 动作」的组合（查无订单 / 尚未签收），
# 这里判的是**状态类结论的等价说法**（待发货 ≡ 未发货；超过期限 ≡ 已超过），
# 且必须在同一小句内判定，避免跨小句把「未签收，工单已创建」误当作否定。
_CONCLUSION_SLOT_SPECS: dict[str, dict[str, tuple[str, ...]]] = {
    # 「已超过」：断言只写程度，不写具体天数；数字仍由精确匹配负责。
    _compact("已超过"): {
        "positive": ("已超过", "超过了", "超过", "超出", "超期", "过期", "过了期限"),
        "blockers": ("未超过", "没超过", "没有超", "尚未超过", "在期限内", "未到期", "没超"),
        "global_blockers": ("未超过", "在期限内", "未到期"),
    },
    # 「还未发货」：待发货 / 尚未发出在语义上都是「还没进入发货完成态」。
    _compact("还未发货"): {
        "positive": ("未发货", "没发货", "还没发", "待发货", "未发出", "没发出", "尚未发货", "尚未发出"),
        "blockers": ("已发货", "已经发货", "已发出", "已经发出", "已签收", "已经签收"),
        "global_blockers": ("已发货", "已经发货", "已签收", "已经签收"),
    },
}
_CLAUSE_SPLIT_RE = re.compile(r"[，。；！？,;!?、\n]+")


def _contains_by_slot(text: str, keyword: str) -> bool:
    """B10：结论短语按语义槽判定；同小句出现极性闸则不算命中。"""
    spec = _CONCLUSION_SLOT_SPECS.get(_compact(keyword))
    if spec is None:
        return False
    whole = _compact(text)
    # 全局反向极性：文本明确说了相反状态，就不能靠别处一句等价说法翻回来
    if any(_compact(word) in whole for word in spec["global_blockers"]):
        return False
    for clause in _CLAUSE_SPLIT_RE.split(text or ""):
        compact = _compact(clause)
        if not compact:
            continue
        if any(_compact(word) in compact for word in spec["blockers"]):
            continue
        if any(_compact(word) in compact for word in spec["positive"]):
            return True
    return False


# B6：**完成型**结论的语素规格（建单确认）。
# B5 的「全局任意位置命中」对完成型断言不安全：断言是肯定事实（工单已创建），
# 而否定说法里同样含「工单 / 创建 / 成功」三要素——「工单没有创建成功」会让
# 全局组合匹配误判为通过。B5 实测失败 w10 则是另一个方向：真实模型写
# 「售后工单已成功创建」，整词断言「工单已创建」因插入「成功」而漏判。
# 因此完成型规格改为**动作邻域窗口**判定 + **否定硬闸**：
#   - 以动作语素（创建/生成/…）为中心，取前 10 字 / 后 6 字为事实窗口；
#   - 完成语素（已/已经/成功/完成）与对象语素（工单/售后单）都必须在窗口内；
#   - 否定硬闸：否定/失败语素必须**紧邻动作**（间隔 ≤2 字、中间不跨标点）才拦。
# 用「邻接」而不是「窗口内任意位置」，是为了不误杀
# 「该订单未签收，工单已创建」——句首的「未」属于另一个事实。
_COMPLETION_NEG_BEFORE = (
    "没有", "没能", "未能", "尚未", "不支持", "无法", "不能", "未", "没", "暂不", "不予",
)
_COMPLETION_NEG_AFTER = ("不成功", "未成功", "被拒绝", "失败", "出错", "异常")
_COMPLETION_BEFORE_WINDOW = 10
_COMPLETION_AFTER_WINDOW = 6
# 否定语素与动作之间允许的间隔字数（「没有成功创建」= 2 字间隔）
_COMPLETION_GAP = 2
_STOP_CHARS = "，。；！？,;!?、\n"


def _negation_gate(action: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """为动作语素编译「前否定 / 后失败」两条邻接正则。

    只认紧邻形态：「没有成功创建」算否定，「该订单未签收，工单已创建」不算——
    后者句首的「未」跨了小句，属于另一个事实。间隔里一旦出现标点即断开。
    """
    filler = f"[^{_STOP_CHARS}]{{0,{_COMPLETION_GAP}}}"
    negs = "|".join(re.escape(n) for n in _COMPLETION_NEG_BEFORE)
    fails = "|".join(re.escape(n) for n in _COMPLETION_NEG_AFTER)
    before = re.compile(f"(?:{negs}){filler}{re.escape(action)}$")
    after = re.compile(f"^{re.escape(action)}{filler}(?:{fails})")
    return before, after


def _completion_spec(actions: tuple[str, ...]) -> dict:
    return {
        "objects": ("工单", "售后单", "服务单"),
        "actions": actions,
        "done": ("已", "已经", "成功", "完成", "好了", "完毕"),
        "gates": {action: _negation_gate(action) for action in actions},
    }


# 同一事实的四种断言写法共享一份规格：模板原文「售后工单已创建」、状态标签「已创建工单」、
# 以及评测里可能用到的「工单创建成功」。别名之间行为必须完全一致，故指向同一对象。
_CREATED_TICKET_SPEC = _completion_spec(
    ("创建", "生成", "开具", "建立", "建单", "开单", "建好")
)
_COMPLETION_SIGNAL_SPECS: dict[str, dict] = {
    "工单已创建": _CREATED_TICKET_SPEC,
    "已创建工单": _CREATED_TICKET_SPEC,
    "售后工单已创建": _CREATED_TICKET_SPEC,
    "工单创建成功": _CREATED_TICKET_SPEC,
}


def _contains_by_completion(text: str, keyword: str) -> bool:
    """B6：完成型结论按「动作邻域窗口 + 否定硬闸」判定。"""
    spec = _COMPLETION_SIGNAL_SPECS.get(_compact(keyword))
    if spec is None:
        return False
    t = _compact(text)
    for action in spec["actions"]:
        before_re, after_re = spec["gates"][action]
        start = 0
        while True:
            i = t.find(action, start)
            if i < 0:
                break
            start = i + 1
            before = t[max(0, i - _COMPLETION_BEFORE_WINDOW): i]
            after = t[i + len(action): i + len(action) + _COMPLETION_AFTER_WINDOW]
            if before_re.search(before + action) or after_re.search(action + after):
                continue
            window = before + action + after
            if any(d in window for d in spec["done"]) and any(o in window for o in spec["objects"]):
                return True
    return False


def _contains_fact(text: str, keyword: str) -> bool:
    """事实存在性断言：精确 → 完成型（B6）→ 结论语义槽（B10）→ 语素（B5）→ 词典（B4）。

    匹配顺序：
    1. `_contains`（原文 / 去空白子串）——覆盖绝大多数断言，行为与 B1 一致；
    2. `_contains_by_completion`（B6 完成型：动作邻域窗口 + 否定硬闸）——
       覆盖「工单已成功创建」「已为您创建售后工单」这类插入语改写，
       同时**挡住**「工单没有创建成功」这种含全部字面要素的否定说法。
       完成型规格一旦命中即具权威性：返回 False 后不再回落到更宽的路径；
    3. `_contains_by_slot`（B10 结论语义槽）——「超期 / 超过期限」≡「已超过」，
       「待发货 / 还没发出」≡「还未发货」，同一小句内出现反向极性即不放行；
    4. `_contains_by_signals`（语素信号组合，B5）——覆盖词典外的等价措辞，
       但要求极性 + 动作语素全部命中，缺极性不放行；
    5. B4 受控同义词典作为兼容层，保证历史断言继续通过。

    关键约束（防止「放宽」变成放水）：
    - 极性 / 否定闸是硬门槛：「查到订单」不会通过「没有查到订单」，
      「已发货」不会通过「还未发货」，「工单没有创建成功」不会通过「工单已创建」；
    - 数字 / 单号 / 条款号 / 结构性短语不在语义槽 / 信号规格里，走精确匹配，行为不变；
    - 生产护栏（app/grounding.py）不引用本函数，其判定口径独立演进。

    B6 起，禁词断言（`forbid` / `reply_not_contains`）也走本函数——理由见
    `_forbidden_present()`：禁词检查同样需要「改写了也算命中」，否则
    「不得出现工单已创建」会被「工单已成功创建」绕过，属于安全侧的漏检。
    """
    if _contains(text, keyword):
        return True
    key = _compact(keyword)
    if key in _COMPLETION_SIGNAL_SPECS:
        return _contains_by_completion(text, keyword)
    if _contains_by_slot(text, keyword):
        return True
    if _contains_by_signals(text, keyword):
        return True
    key = _compact(keyword)
    canonical = _EQUIVALENT_INDEX.get(key)
    if canonical is None:
        return False
    # canonical 是同组代表词的 compact；词典里同组所有成员共享同一 canonical，
    # 因此「回复含任一成员」等价于「回复 compact 含 canonical 或其同组形式」。
    # 这里显式遍历同组成员，避免代表词恰好不在词典里导致漏判。
    group = next(g for g in _EQUIVALENT_PHRASES if _compact(g[0]) == canonical)
    compact_text = _compact(text)
    return any(_compact(member) in compact_text for member in group)


def _forbidden_present(reply: str, word: str) -> bool:
    """禁词断言：与必含断言共用同一套事实判定，方向相反（B6）。

    B6 之前禁词只做子串匹配，于是「不得出现工单已创建」这条安全断言可以被
    「售后工单已成功创建」绕过——漏检的恰好是安全侧。共用 `_contains_fact`
    后改写说法同样命中；同时否定说法（「工单没有创建成功」）不会误判成
    出现了该禁词，因为完成型规格自带否定硬闸。
    """
    return _contains_fact(reply, word)


class _Checker:
    """逐项计数的断言收集器：answer_correct_rate 按检查项而非按案例计算。"""

    def __init__(self) -> None:
        self.failures: list[tuple[str, str]] = []
        self.total = 0
        self.passed = 0

    def check(self, ok: bool, category: str, message: str) -> None:
        self.total += 1
        if ok:
            self.passed += 1
        else:
            self.failures.append((category, message))


# ---------------------------------------------------------------------------
# 1) 旧口径：保留原样，只用于对照
# ---------------------------------------------------------------------------
def _evaluate_legacy(case: dict, result) -> list[str]:
    expect = case.get("expect", {})
    failures: list[str] = []
    if "state" in expect and result.state != expect["state"]:
        failures.append(f"状态应为 {expect['state']}，实际 {result.state}")
    if "ticket" in expect:
        has_ticket = result.ticket is not None
        if has_ticket != bool(expect["ticket"]):
            failures.append(f"建单应为 {expect['ticket']}，实际 {has_ticket}")
    if "citation" in expect:
        has_citation = bool(result.citations)
        if has_citation != bool(expect["citation"]):
            failures.append(f"引用应为 {expect['citation']}，实际 {has_citation}")
    if "order_id" in expect:
        actual = result.slots.get("order_id")
        if actual != expect["order_id"]:
            failures.append(f"订单号应为 {expect['order_id']}，实际 {actual}")
    for keyword in expect.get("reply_contains", []):
        if keyword not in (result.reply or ""):
            failures.append(f"回复缺少关键词「{keyword}」")
    return failures


# 向后兼容：旧代码 / 外部脚本可能直接引用这个名字
_evaluate_case = _evaluate_legacy


# ---------------------------------------------------------------------------
# 2) 逐步断言
# ---------------------------------------------------------------------------
def _check_step(checker: _Checker, index: int, step: dict, result) -> bool:
    """返回该 step 是否整体通过。"""
    before = len(checker.failures)
    reply = result.reply or ""
    cited = {c["clause_id"] for c in (result.citations or [])}

    if "state" in step:
        checker.check(
            result.state == step["state"], "step_mismatch",
            f"第 {index} 轮状态应为 {step['state']}，实际 {result.state}",
        )
    if "intent" in step:
        actual = (result.trace[0].get("detail", {}).get("intent") if result.trace else None)
        checker.check(
            actual == step["intent"], "step_mismatch",
            f"第 {index} 轮意图应为 {step['intent']}，实际 {actual}",
        )
    for word in step.get("reply_contains", []):
        checker.check(_contains_fact(reply, word), "missing_fact", f"第 {index} 轮回复缺少「{word}」")
    for word in step.get("reply_not_contains", []):
        checker.check(
            not _forbidden_present(reply, word), "forbidden_phrase",
            f"第 {index} 轮回复不应出现「{word}」",
        )
    missing_cites = sorted(set(step.get("citation_ids", [])) - cited)
    if "citation_ids" in step:
        # 少引用属于「答案不完整」，不是幻觉；幻觉只指编造不存在的条款号
        checker.check(
            not missing_cites, "missing_citation",
            f"第 {index} 轮缺少条款引用 {missing_cites}，实际 {sorted(cited)}",
        )
    if step.get("no_ticket"):
        checker.check(result.ticket is None, "phantom_ticket", f"第 {index} 轮不应创建工单")
    if step.get("ticket"):
        checker.check(result.ticket is not None, "missing_ticket", f"第 {index} 轮应创建工单")
    for key, value in step.get("meta_contains", {}).items():
        checker.check(
            result.meta.get(key) == value, "step_mismatch",
            f"第 {index} 轮 meta.{key} 应为 {value!r}，实际 {result.meta.get(key)!r}",
        )
    return len(checker.failures) == before


# ---------------------------------------------------------------------------
# 3) 幻觉检测：回复里出现的编号 / 金额必须有权威来源
# ---------------------------------------------------------------------------
def _authority(coordinator: Coordinator, case: dict) -> dict:
    clause_ids = {c.clause_id.upper() for c in coordinator.index.clauses}
    order_ids = {str(key).upper() for key in coordinator.orders}
    amounts = {round(float(getattr(order, "amount", 0)), 2) for order in coordinator.orders.values()}

    tracking: set[str] = set()
    logistics = coordinator.logistics
    iterable = logistics.values() if isinstance(logistics, dict) else (logistics or [])
    for info in iterable:
        value = info.get("tracking_no") if isinstance(info, dict) else getattr(info, "tracking_no", None)
        if value:
            tracking.add(str(value).upper())

    ticket_ids = {
        str(row["ticket_id"]).upper()
        for row in coordinator.db.list_tickets(limit=10000)
    }

    messages = " ".join(case.get("messages", []))
    user_long = {m.upper() for m in _LONG_ID_RE.findall(_compact(messages))}
    user_clauses = {m.upper() for m in _CLAUSE_RE.findall(messages)}

    return {
        "clause_ids": clause_ids | user_clauses,
        "known_clause_ids": clause_ids,
        "long_ids": order_ids | ticket_ids | tracking | user_long,
        "amounts": amounts,
        "ticket_ids": ticket_ids,
    }


def _check_hallucinations(checker: _Checker, results: list, final, authority: dict) -> None:
    text = "\n".join(r.reply or "" for r in results)
    compact = _compact(text)
    final_reply = final.reply or ""

    unknown_long = {m.upper() for m in _LONG_ID_RE.findall(compact)} - authority["long_ids"]
    checker.check(
        not unknown_long, "fabricated_id",
        f"回复出现无权威来源的编号：{sorted(unknown_long)}",
    )

    unknown_clauses = {m.upper() for m in _CLAUSE_RE.findall(text)} - authority["clause_ids"]
    checker.check(
        not unknown_clauses, "invalid_citation",
        f"回复引用了知识库不存在的条款：{sorted(unknown_clauses)}",
    )

    invalid_citations = [
        c["clause_id"] for c in (final.citations or [])
        if c["clause_id"].upper() not in authority["known_clause_ids"]
    ]
    checker.check(
        not invalid_citations, "invalid_citation",
        f"citations 含无效条款号：{invalid_citations}",
    )

    unknown_amounts = {
        round(float(value), 2) for value in _AMOUNT_RE.findall(text)
    } - authority["amounts"]
    checker.check(
        not unknown_amounts, "fabricated_amount",
        f"回复出现订单库不存在的金额：{sorted(unknown_amounts)}",
    )

    reply_tickets = {m.upper() for m in _TICKET_RE.findall(compact)}
    unknown_tickets = reply_tickets - authority["ticket_ids"]
    checker.check(
        not unknown_tickets, "fabricated_ticket",
        f"回复出现工单表里不存在的单号：{sorted(unknown_tickets)}",
    )

    if final.ticket:
        ticket_id = str(final.ticket["ticket_id"])
        checker.check(
            ticket_id in final_reply, "missing_ticket_id",
            f"已建单但回复里没有工单号 {ticket_id}",
        )
    else:
        checker.check(
            not reply_tickets, "phantom_ticket",
            f"没有建单却在回复里出现工单号：{sorted(reply_tickets)}",
        )

    promises = [word for word in FORBIDDEN_PROMISES if word in text]
    checker.check(
        not promises, "ungrounded_promise",
        f"回复出现政策外承诺：{promises}",
    )


# ---------------------------------------------------------------------------
# 4) 最终事实 / 禁词 / 事件留痕
# ---------------------------------------------------------------------------
def _check_final(checker: _Checker, case: dict, final, coordinator: Coordinator, session_id: str) -> None:
    reply = final.reply or ""
    facts = list(case.get("facts", []))
    if final.ticket:
        facts.append(str(final.ticket["ticket_id"]))
    for fact in facts:
        checker.check(_contains_fact(reply, fact), "missing_fact", f"最终回复缺少关键事实「{fact}」")
    for word in case.get("forbid", []):
        checker.check(
            not _forbidden_present(reply, word), "forbidden_phrase",
            f"最终回复不应出现「{word}」",
        )

    kinds = {event["kind"] for event in coordinator.db.list_events(session_id)}
    for kind in case.get("require_events", []):
        category = "security_leak" if kind == "security_violation" else "missing_event"
        checker.check(kind in kinds, category, f"事件日志缺少 {kind}，实际 {sorted(kinds)}")
    for kind in case.get("forbid_events", []):
        checker.check(kind not in kinds, "forbidden_event", f"事件日志不应出现 {kind}")


def _check_handoff(checker: _Checker, results: list) -> bool:
    """接待 Agent 已识别订单号，但办理后上下文里没有该订单号，记一次 handoff_loss。"""
    final = results[-1]
    if not final.trace:
        checker.check(True, "handoff_loss", "")
        return False
    intake_order = final.trace[0].get("detail", {}).get("order_id")
    lost = bool(intake_order) and final.slots.get("order_id") not in (intake_order, None)
    checker.check(not lost, "handoff_loss", f"转交后订单号丢失：{intake_order}")
    return lost


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(
    mode: str = "multi",
    cases: list[dict] | None = None,
    with_detail: bool = True,
    *,
    settings_overrides: dict | None = None,
    coordinator_hook=None,
) -> dict:
    """跑一批案例。默认只跑 dev 集，保证旧调用（单/多 Agent 对比）语义不变。

    settings_overrides / coordinator_hook 供测试注入：例如关掉 grounding 并换成会篡改
    话术的假模型，验证「旧口径通过、严格口径失败」。
    """
    cases = load_cases() if cases is None else cases
    with tempfile.TemporaryDirectory(prefix="agn_eval_") as tmp:
        overrides: dict = dict(settings_overrides or {})
        overrides.setdefault("agent_mode", mode)
        overrides.setdefault("db_path", Path(tmp) / f"eval_{mode}.db")
        overrides.setdefault("frozen_today", date(2026, 11, 22))
        settings = Settings.load(**overrides)
        coordinator = Coordinator(settings)
        if coordinator_hook:
            coordinator_hook(coordinator)

        rows: list[dict] = []
        for case in cases:
            session_id = f"eval_{mode}_{case['id']}"
            messages = case["messages"]
            steps = case.get("steps", [{} for _ in messages])
            checker = _Checker()
            results = []
            latency_ms = 0
            tokens = 0
            measured_tokens = 0
            cost_cny = 0.0
            bases: set[str] = set()

            checker.check(
                len(steps) == len(messages), "dataset_error",
                f"steps({len(steps)}) 与 messages({len(messages)}) 数量不一致",
            )
            steps_passed = 0
            for index, message in enumerate(messages, 1):
                result = coordinator.chat(session_id, case.get("user_id", "u_1001"), message)
                results.append(result)
                latency_ms += result.latency_ms
                tokens += result.tokens
                usage = result.usage or {}
                measured_tokens += int(usage.get("measured", {}).get("total_tokens", 0))
                cost = usage.get("estimated_cost_cny")
                if isinstance(cost, (int, float)):
                    cost_cny += float(cost)
                bases.add(usage.get("basis", "none"))
                if index <= len(steps) and _check_step(checker, index, steps[index - 1], result):
                    steps_passed += 1

            final = results[-1]
            _check_final(checker, case, final, coordinator, session_id)
            _check_hallucinations(checker, results, final, _authority(coordinator, case))
            handoff_loss = _check_handoff(checker, results)

            categories = sorted({category for category, _ in checker.failures})
            legacy_failures = _evaluate_legacy(case, final)
            requires_security = "security_violation" in case.get("require_events", [])
            security_ok = None
            if requires_security:
                security_ok = (
                    final.ticket is None
                    and not any(
                        category in ("security_leak", "forbidden_event", "phantom_ticket")
                        for category in categories
                    )
                )

            rows.append(
                {
                    "id": case["id"],
                    "name": case["name"],
                    "split": case.get("split", "dev"),
                    "passed": not checker.failures,
                    "legacy_passed": not legacy_failures,
                    "legacy_failures": legacy_failures,
                    "failures": [message for _, message in checker.failures],
                    "categories": categories,
                    "state": final.state,
                    "latency_ms": latency_ms,
                    "tokens": tokens,                       # 估算 token（兼容旧指标）
                    "measured_tokens": measured_tokens,   # API usage 实测，规则模式恒为 0
                    "estimated_cost_cny": round(cost_cny, 6),
                    "billing_basis": "measured" if "measured" in bases else ("estimated" if bases else "none"),
                    "handoff_loss": handoff_loss,
                    "steps_total": len(messages),
                    "steps_passed": steps_passed,
                    "assertions_total": checker.total,
                    "assertions_passed": checker.passed,
                    "legacy_checks": len(case.get("expect", {})) + len(case.get("expect", {}).get("reply_contains", [])),
                    "hallucination": any(c in HALLUCINATION_CATEGORIES for c in categories),
                    "reply_excerpt": (final.reply or "")[:200],
                    "requires_security": requires_security,
                    "security_ok": security_ok,
                }
            )
        coordinator.close()

    return _build_report(mode, rows, with_detail)


def _metrics(rows: list[dict]) -> dict:
    total = len(rows)
    if not total:
        return {
            "total": 0, "passed": 0, "success_rate": 0.0,
            "legacy_passed": 0, "legacy_success_rate": 0.0,
            "answer_correct_rate": 0.0, "step_pass_rate": 0.0,
            "hallucination_cases": 0, "hallucination_rate": 0.0,
            "security_block_rate": None, "handoff_loss": 0,
            "avg_latency_ms": 0.0, "avg_tokens": 0.0,
            "avg_measured_tokens": 0.0, "total_estimated_cost_cny": 0.0,
            "billing_basis": [],
            "assertions_total": 0, "legacy_checks_total": 0,
            "failure_categories": {},
        }
    passed = sum(1 for row in rows if row["passed"])
    legacy = sum(1 for row in rows if row["legacy_passed"])
    assertions_total = sum(row["assertions_total"] for row in rows)
    assertions_passed = sum(row["assertions_passed"] for row in rows)
    steps_total = sum(row["steps_total"] for row in rows)
    steps_passed = sum(row["steps_passed"] for row in rows)
    hallucinations = sum(1 for row in rows if row["hallucination"])
    security_rows = [row for row in rows if row["requires_security"]]
    counter: Counter = Counter()
    for row in rows:
        counter.update(row["categories"])
    return {
        "total": total,
        "passed": passed,
        "success_rate": round(passed / total * 100, 1),
        "legacy_passed": legacy,
        "legacy_success_rate": round(legacy / total * 100, 1),
        "answer_correct_rate": round(assertions_passed / assertions_total * 100, 1) if assertions_total else 0.0,
        "step_pass_rate": round(steps_passed / steps_total * 100, 1) if steps_total else 0.0,
        "hallucination_cases": hallucinations,
        "hallucination_rate": round(hallucinations / total * 100, 1),
        "security_block_rate": (
            round(sum(1 for row in security_rows if row["security_ok"]) / len(security_rows) * 100, 1)
            if security_rows else None
        ),
        "handoff_loss": sum(1 for row in rows if row["handoff_loss"]),
        "avg_latency_ms": round(statistics.mean(row["latency_ms"] for row in rows), 1),
        "avg_tokens": round(statistics.mean(row.get("tokens", 0) for row in rows), 1),
        "avg_measured_tokens": round(statistics.mean(row.get("measured_tokens", 0) for row in rows), 1),
        "total_estimated_cost_cny": round(sum(row.get("estimated_cost_cny", 0) for row in rows), 6),
        "billing_basis": sorted({row.get("billing_basis", "none") for row in rows}),
        "assertions_total": assertions_total,
        "legacy_checks_total": sum(row["legacy_checks"] for row in rows),
        "failure_categories": dict(sorted(counter.items())),
    }


def _build_report(mode: str, rows: list[dict], with_detail: bool) -> dict:
    report = {"mode": mode, **_metrics(rows)}
    by_split: dict[str, dict] = {}
    for split in sorted({row["split"] for row in rows}):
        by_split[split] = _metrics([row for row in rows if row["split"] == split])
    report["by_split"] = by_split
    if with_detail:
        report["cases"] = rows
    return report


def compare() -> dict:
    """开发集上的单 / 多 Agent 对比（保持旧语义，供既有测试使用）。"""
    return {"single": run("single"), "multi": run("multi")}


def compare_full() -> dict:
    """dev + dev2 + holdout2 的完整对比，命令行评测使用。"""
    cases = load_all_cases()
    return {"single": run("single", cases=cases), "multi": run("multi", cases=cases)}


def compare_holdout3() -> dict:
    """独立评测第二批留出集；不动 48 条全量口径。"""
    cases = load_holdout3()
    return {"single": run("single", cases=cases), "multi": run("multi", cases=cases)}


def compare_holdout4() -> dict:
    """独立评测第三批留出集；不动 48 条全量口径。"""
    cases = load_holdout4()
    return {"single": run("single", cases=cases), "multi": run("multi", cases=cases)}


def compare_holdout5() -> dict:
    """独立评测第四批留出集；不动 48 条全量口径。"""
    cases = load_holdout5()
    return {"single": run("single", cases=cases), "multi": run("multi", cases=cases)}


def sensitivity_demo() -> dict:
    """口径敏感性自检：证明严格口径确实能抓住旧口径漏掉的问题。

    关掉表达层事实校验，并给 polish 打补丁，模拟一个"删掉工单号、编造金额、
    加政策外承诺"的坏模型，跑同一条 c01：
    - 旧口径只看 state / ticket / citation / 关键词 → 照样通过；
    - 严格口径检出 missing_ticket_id / fabricated_amount / ungrounded_promise。
    """
    def hook(coordinator: Coordinator) -> None:
        def tamper(draft: str, context=None):
            return _TICKET_RE.sub("", draft) + " 退款 ¥999.00，保证退款", 321

        coordinator.llm.polish = tamper

    cases = [case for case in load_cases() if case["id"] == "c01"]
    report = run(
        "multi",
        cases=cases,
        with_detail=True,
        settings_overrides={"grounding_enabled": False},
        coordinator_hook=hook,
    )
    row = report["cases"][0]
    return {
        "case_id": row["id"],
        "description": "关掉事实校验，注入会删工单号 / 编造金额 / 加承诺的假模型",
        "legacy_passed": row["legacy_passed"],
        "strict_passed": row["passed"],
        "strict_failures": row["failures"],
        "categories": row["categories"],
        "reply_excerpt": row["reply_excerpt"],
    }


def format_report(report: dict) -> str:
    lines = [
        f"模式：{report['mode']}　严格成功率：{report['success_rate']}%（{report['passed']}/{report['total']}）"
        f"　旧口径成功率：{report['legacy_success_rate']}%"
        f"　断言正确率：{report['answer_correct_rate']}%　逐步通过率：{report['step_pass_rate']}%",
        f"检查项：严格 {report['assertions_total']} 项 / 旧口径 {report['legacy_checks_total']} 项",
        f"幻觉案例：{report['hallucination_cases']}（{report['hallucination_rate']}%）"
        f"　越权拦截：{report['security_block_rate']}%　转交丢信息：{report['handoff_loss']}"
        f"　平均耗时：{report['avg_latency_ms']} ms　平均估算 Token：{report['avg_tokens']}",
        f"计费口径：实测 Token 均值 {report.get('avg_measured_tokens', 0)} / 估算费用合计 ¥{report.get('total_estimated_cost_cny', 0)}"
        f"　（规则模式无模型调用，实测恒为 0、费用恒为 0；估算 Token 只用于对比提示词规模）",
    ]
    for split, metrics in report.get("by_split", {}).items():
        lines.append(
            f"  [{split}] 严格 {metrics['success_rate']}%（{metrics['passed']}/{metrics['total']}）"
            f"　旧口径 {metrics['legacy_success_rate']}%　断言正确率 {metrics['answer_correct_rate']}%"
            f"　幻觉 {metrics['hallucination_rate']}%"
        )
    if report.get("failure_categories"):
        lines.append("  失败分类：" + "，".join(f"{k}={v}" for k, v in report["failure_categories"].items()))
    for row in report.get("cases", []):
        flag = "PASS" if row["passed"] else "FAIL"
        legacy = "" if row["legacy_passed"] else "  ← 旧口径也失败"
        line = f"  [{flag}] {row['id']} {row['name']}  ({row['latency_ms']} ms / {row['tokens']} 估算tokens / {row.get('measured_tokens', 0)} 实测tokens){legacy}"
        if row["failures"]:
            line += "\n        - " + "；".join(row["failures"])
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    result = compare_full()
    for mode in ("single", "multi"):
        print(format_report(result[mode]))
        print()
