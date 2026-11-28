"""表达层后置事实校验（Grounding Guard）。

设计原则：**模型只负责"怎么说"，事实只能来自代码生成的 draft**。

`polish()` 的 system prompt 只是"软约束"，模型仍可能删掉工单号、改金额、编一个条款号。
本模块在 polish 之后做一次纯函数校验，把输出与 draft 做事实对齐：

    必须保留  : draft 里的标识符（订单号 / 工单号 / 条款号）与关键数值（金额 / 日期 / 时效）
    不得新增  : draft 与结构化 facts 之外的标识符、draft 之外的数字
    不得越权  : 政策外的承诺话术、prompt 泄漏、空输出、异常长文

任一违规 → **整体回退到 draft**（不做部分修补，保守且可解释），并把违规原因写进
trace / meta / 事件日志，保证"代码负责事实"这条原则在 LLM 模式下也成立。

规则模式下 polished == draft，直接短路，零开销、零误杀。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1. 标识符：唯一编号类事实，必须原样保留，也不允许凭空新增
#    用 ASCII 字母数字边界而非 \b —— 中文属于 \w，"订单O2026..." 里 \b 不成立。
# ---------------------------------------------------------------------------
# \s* 容忍「O 20261115001」这类排版空格；compact 后等价于无空格版本
_IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9])RT\s*\d{8}\s*[A-Za-z0-9]{6}(?![A-Za-z0-9])"  # 工单 RT20261122A1B2C3
    r"|(?<![A-Za-z0-9])R\s*-\s*\d{2}(?![A-Za-z0-9])"                  # 条款 R-01
    r"|(?<![A-Za-z0-9])[A-Za-z]{1,3}\s*\d{8,}[A-Za-z0-9]*(?![A-Za-z0-9])"  # 订单 O20261115001
    r"|(?<![0-9])\d{12,}(?![0-9])",                              # 纯数字长单号（运单等）
    re.I,
)

# ---------------------------------------------------------------------------
# 2. 中文数字归一：允许「还剩 3 天」被改写成「还剩三天」这类等价表达
# ---------------------------------------------------------------------------
_CN_DIGITS = {
    "零": "0", "〇": "0", "一": "1", "二": "2", "两": "2", "三": "3",
    "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
}
_CN_NUM_RE = re.compile(
    r"([零〇一二两三四五六七八九十]+)\s*(?=(?:个?工作日|分钟|小时|天|日|周|个月|元|块|件|次|%|％))"
)
# 「一到三个工作日」这类范围表达要先整体归一，否则「一」不会被单独转换
_CN_RANGE_RE = re.compile(
    r"([零〇一二两三四五六七八九十]+)\s*([-~至到])\s*([零〇一二两三四五六七八九十]+)"
    r"\s*(?=(?:个?工作日|分钟|小时|天|日|周|个月|元|块|件|次))"
)


def _cn_to_arabic(token: str) -> str:
    if token == "十":
        return "10"
    if "十" in token:
        left, _, right = token.partition("十")
        tens = int(_CN_DIGITS.get(left, "1")) if left else 1
        ones = int(_CN_DIGITS[right]) if right else 0
        return str(tens * 10 + ones)
    return "".join(_CN_DIGITS.get(ch, ch) for ch in token)


def _normalize_numbers(text: str) -> str:
    text = _CN_RANGE_RE.sub(
        lambda m: f"{_cn_to_arabic(m.group(1))}{m.group(2)}{_cn_to_arabic(m.group(3))}", text or ""
    )
    return _CN_NUM_RE.sub(lambda m: _cn_to_arabic(m.group(1)), text)


def _strip_identifiers(text: str) -> str:
    """数值比较前先剔除单号：RT...A1B2C3 里的零散 1/2/3 会撑大允许数字集合，
    让「1-3 个工作日」被偷改成「1-2」而漏检。单号本身由标识符检查覆盖。

    注意在**原文**上替换而非 compact 文本：压掉空白会把「2026-11-15 10:30」
    拼成「2026-11-1510:30」，把 15 和 10 误合成 1510，导致合法改写被误杀。
    """
    return _IDENTIFIER_RE.sub(" ", text or "")


def _numbers(text: str) -> set[float | int]:
    """抽取文本中所有数值，归一为可比较的标量（299.00 与 299 视为同一事实）。"""
    result: set[float | int] = set()
    for raw in re.findall(r"\d+(?:\.\d+)?", _normalize_numbers(_strip_identifiers(text))):
        value = float(raw)
        result.add(int(value) if value.is_integer() else round(value, 4))
    return result


# ---------------------------------------------------------------------------
# 3. 关键数值：金额 / 日期 / 期限 / 时效，删掉任何一个都算丢事实
#    时间窗口（9:00-20:00）不列入，避免「9 点到 20 点」这类合法改写被误杀；
#    但新增数字的检查仍会覆盖它（模型编 8:00-22:00 会因 8/22 不在 draft 被拒）。
# ---------------------------------------------------------------------------
_CRITICAL_RE = re.compile(
    r"[¥￥]\s*\d+(?:\.\d+)?"                                             # 金额
    r"|\d{4}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}\s*日?"              # 日期
    r"|\d+\s*[-~至到]\s*\d+\s*个?工作日"                                  # 1-3 个工作日
    r"|第\s*\d+\s*天"                                                     # 第 7 天
    r"|\d+\s*(?:个?工作日|分钟|小时|个月|周|天|日)"                          # 时效 / 天数
    r"|\d+(?:\.\d+)?\s*[%％]",                                            # 百分比
)

# ---------------------------------------------------------------------------
# 4. 政策外承诺：system prompt 明确禁止，这里做兜底
# ---------------------------------------------------------------------------
FORBIDDEN_PROMISES = (
    "保证退款", "无条件", "全额赔付", "包赔", "先行赔付", "绝对", "百分百", "100%",
    "随时可退", "无限期", "免费退", "立刻到账", "马上到账", "秒退", "双倍赔偿",
    "假一赔十", "包退包换", "无条件退款",
)

_PROMPT_LEAK_MARKERS = ("原始话术", "事实：", "```", "<|", "system prompt")

# 不可见字符：B4/B5 真实模型输出里 6/10 条回复末尾都挂着一串
# 「‌​⁠」类零宽 / 词连接符（每条约 36 个），用户侧看不见但会污染复制粘贴、
# 也会干扰下游断言匹配。润色文本一律先清洗再做校验，校验通过后原样返回清洗结果。
_INVISIBLE_RE = re.compile(
    "[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]"
)


def strip_invisible(text: str) -> tuple[str, int]:
    """去掉零宽空格 / 方向控制符 / 词连接符等不可见字符，返回 (清洗文本, 去掉个数)。"""
    cleaned = _INVISIBLE_RE.sub("", text or "")
    return cleaned, len(text or "") - len(cleaned)

# 语义反转：把「拒绝 / 未创建」改写成「已办理」是最危险的幻觉，单独兜底
_NEGATION_FLIPS = (
    ("没有创建工单", ("已创建工单", "工单已创建", "成功创建工单", "工单已经生成")),
    ("没有重复建单", ("重复建单", "创建了两个工单", "又建了一张工单")),
    ("没有权限", ("有权限", "已为您办理", "已经帮您办好了")),
    ("暂时不能直接办理", ("可以直接办理", "已为您办理退货", "退货已受理")),
    # B5 测试暴露：模型可能把「没有查到」反转成「已经查到 / 查到了订单」，
    # 只写「已查到订单」会漏。补全肯定式反转说法，守住安全底线。
    # 注意不能写裸「查到订单」——它是原文「没有查到订单」的子串，会误杀保留原文的合法润色。
    ("没有查到订单", ("已查到订单", "已经查到订单", "查到了订单", "成功查到订单", "订单信息如下")),
)

# 义务强度翻转：把 draft 的强制要求（需/必须）改成可选建议（建议/可以/可），
# 会改变用户的义务性质——用户可能因此不提供凭证、不保持吊牌，是最容易漏掉的
# 语义幻觉（数字、单号都还在，读起来也通顺）。沿用 B5 的语素组合哲学：
# 强制语素 + 动作 vs 弱化语素 + 同一动作，不枚举整句，覆盖未见说法。
# 负向后顾 (?<![无不]) 排除「无需 / 不需要」把强义务否定掉的情况。
_OBLIGATION_STRONG = ("需", "须", "必须", "应当", "务必", "需要")
_OBLIGATION_WEAK = ("可", "可以", "可选", "建议", "不妨", "自愿", "随意")
_OBLIGATION_ACTIONS = (
    "提供", "上传", "保持", "保留", "配合", "寄回", "填写", "提交",
    "承担", "支付", "保证", "附上", "发送", "给出",
)


def _obligation_re(particles: tuple[str, ...]) -> re.Pattern[str]:
    # 长语素排前，避免「可」抢在「可以」前匹配；动作作为命名组便于按动作对齐
    ordered = sorted(particles, key=len, reverse=True)
    return re.compile(
        r"(?<![无不])(?:"
        + "|".join(re.escape(p) for p in ordered)
        + r")(?P<action>"
        + "|".join(re.escape(a) for a in _OBLIGATION_ACTIONS)
        + r")"
    )


_OBLIGATION_STRONG_RE = _obligation_re(_OBLIGATION_STRONG)
_OBLIGATION_WEAK_RE = _obligation_re(_OBLIGATION_WEAK)


def _obligation_actions(pattern: re.Pattern[str], text: str) -> set[str]:
    return {m.group("action") for m in pattern.finditer(text)}


def _obligation_flips(draft_compact: str, polished_compact: str) -> list[str]:
    """返回义务强度被翻转的动作列表（弱化/强化各记为一条）。

    判定按「同一动作」对齐：draft 说「需提供照片」、polish 说「建议提供照片」
    → 弱化；反向 draft「建议提供」、polish「必须提供」→ 强化。若 polish 仍保留
    原强度（如「需提供照片，建议提供清晰照」），说明义务未丢，不判违规。
    """
    flips: list[str] = []
    d_strong = _obligation_actions(_OBLIGATION_STRONG_RE, draft_compact)
    d_weak = _obligation_actions(_OBLIGATION_WEAK_RE, draft_compact)
    p_strong = _obligation_actions(_OBLIGATION_STRONG_RE, polished_compact)
    p_weak = _obligation_actions(_OBLIGATION_WEAK_RE, polished_compact)
    for action in sorted(d_strong - p_strong):
        if action in p_weak:
            flips.append(f"弱化：{action}")
    for action in sorted(d_weak - p_weak):
        if action in p_strong:
            flips.append(f"强化：{action}")
    return flips

# ---------------------------------------------------------------------------
# 5. 结构性事实短语：模板里承载关键结论的整句，润色不得压缩丢掉
#    B2 真实模型复测暴露：模型把「交接信息：…」压缩成「已为您转接」、
#    把「运费由商家承担」整句改没——事实没编造，但信息完整性受损；
#    标识符 / 数值检查管不住这类词，这里单独校验（compact 匹配容忍排版空格）。
#    只列「丢了会误导用户」的短语，宁少勿滥：语气词、客套话不在此列，
#    避免合法改写被误杀导致护栏频繁回退。
# ---------------------------------------------------------------------------
REQUIRED_PHRASES = (
    "转人工",            # 转交提示本身：丢了用户不知道已转人工
    "交接信息",          # 人工交接上下文：丢了用户会重复叙述
    "运费由商家承担",    # 运费承担方（质量通道）
    "运费由消费者承担",  # 运费承担方（无理由通道）
    "运费需自理",        # 运费承担方（无理由通道模板变体）
)

# ---------------------------------------------------------------------------
# 6. 信息压缩：长答案不能被润色压成一句话
#    B8 的答案是有层次的（结论 + 条款要点 + 下一步追问），而润色提示词又要求
#    「200 字以内」。用户截图里那条「好的，请提供订单号」就是被压到只剩追问的
#    样子——事实一个没编错，但用户问的事被删干净了，标识符/数值检查看不出来。
#    只拦「显著变短」（< 55%），措辞精简照常放行，避免误杀合法改写。
# ---------------------------------------------------------------------------
_COMPRESSION_MIN_DRAFT = 120
_COMPRESSION_MIN_KEEP = 0.55


@dataclass
class GroundingResult:
    ok: bool
    reply: str
    used_polished: bool = False
    violations: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "used_polished": self.used_polished,
            "violations": self.violations,
            "detail": self.detail,
        }


def _compact(text: str) -> str:
    """去掉空白后再抽标识符：允许模型把单号排版成 'O 20261115001'。"""
    return re.sub(r"\s+", "", text or "")


def _identifiers(text: str) -> set[str]:
    return {m.group(0).upper() for m in _IDENTIFIER_RE.finditer(_compact(text))}


def _fact_identifiers(facts: dict | None) -> set[str]:
    """结构化 facts 里的标识符属于"允许出现"，但不强制要求出现在回复中。"""
    parts: list[str] = []
    for value in (facts or {}).values():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple, set)):
            parts.extend(str(item) for item in value)
        elif isinstance(value, (int, float)):
            parts.append(str(value))
    return _identifiers(" ".join(parts))


def _critical_numbers(text: str) -> set[float | int]:
    result: set[float | int] = set()
    for match in _CRITICAL_RE.finditer(_normalize_numbers(_strip_identifiers(text))):
        result |= _numbers(match.group(0))
    return result


def verify(
    draft: str,
    polished: str,
    facts: dict | None = None,
    *,
    enabled: bool = True,
) -> GroundingResult:
    """校验 polish 输出；违规时返回 draft 作为最终回复。

    - draft    : 状态机模板生成的原文，唯一事实来源
    - polished : 模型润色后的文本
    - facts    : 结构化上下文，用于放宽"允许出现的标识符"集合
    - enabled  : 关闭时只做透传（用于对照演示）
    """
    draft = draft or ""
    polished = polished if polished is not None else ""
    # 不可见字符清洗放在最前面：既让用户拿到的回复干净，也避免它们把
    # 「工单已创建」这类短语切断（零宽字符不算空白，compact 也去不掉）。
    polished, stripped = strip_invisible(polished)
    draft, _ = strip_invisible(draft)
    used = polished.strip() != draft.strip()

    if not enabled:
        result = GroundingResult(True, polished or draft, used, [], {"checked": False, "disabled": True})
        if stripped:
            result.detail["invisible_stripped"] = stripped
        return result
    if not used:
        # 规则模式 / 模型降级：没有改写，无需校验
        result = GroundingResult(True, draft, False, [], {"checked": False, "unchanged": True})
        if stripped:
            result.detail["invisible_stripped"] = stripped
        return result

    violations: list[str] = []
    detail: dict = {"checked": True}
    if stripped:
        detail["invisible_stripped"] = stripped

    # --- 结构异常 ---
    if not polished.strip():
        violations.append("empty_output")
    if len(polished) > max(600, int(len(draft) * 2.5)):
        violations.append("overlong")
    leaked = [marker for marker in _PROMPT_LEAK_MARKERS if marker in polished]
    if leaked:
        violations.append("prompt_leak")
        # 记录具体命中项：接第三方网关时能看到模型到底复述了哪段，而不只是「违规」
        detail["prompt_leak_markers"] = leaked

    # --- 标识符：必须保留 draft 里的，不得造 draft/facts 之外的 ---
    draft_ids = _identifiers(draft)
    polished_ids = _identifiers(polished)
    allowed_ids = draft_ids | _fact_identifiers(facts)
    missing_ids = sorted(draft_ids - polished_ids)
    extra_ids = sorted(polished_ids - allowed_ids)
    if missing_ids:
        violations.append("missing_fact")
    if extra_ids:
        violations.append("fabricated_id")
    detail.update(
        {
            "required_ids": sorted(draft_ids),
            "found_ids": sorted(polished_ids),
            "missing_ids": missing_ids,
            "extra_ids": extra_ids,
        }
    )

    # --- 数值：不得新增 draft 里没有的数字；关键数值不得被删 ---
    draft_numbers = _numbers(draft)
    polished_numbers = _numbers(polished)
    extra_numbers = sorted(polished_numbers - draft_numbers, key=str)
    missing_numbers = sorted(_critical_numbers(draft) - polished_numbers, key=str)
    if missing_numbers:
        violations.append("missing_fact")
    if extra_numbers:
        violations.append("fabricated_number")
    detail.update(
        {
            "draft_numbers": sorted(draft_numbers, key=str),
            "polished_numbers": sorted(polished_numbers, key=str),
            "missing_numbers": missing_numbers,
            "extra_numbers": extra_numbers,
        }
    )

    # --- 结构性事实短语：draft 里出现的关键短语必须保留 ---
    # 与标识符 / 数值同构：compact 匹配只消除排版差异，不做语义等价——
    # 模型把「运费由商家承担」改写成「运费商家出」属于丢了校验短语，
    # 保守回退整段到 draft，宁可少一次润色也不丢结论。
    draft_compact = _compact(draft)
    polished_compact = _compact(polished)
    # B12：商品名是结构性事实，不是可改写的措辞。B6 x05 实测模型把
    # 「查到订单 O… · 智能手表 S2」里的商品名整段删掉，单号还在，原有的
    # 标识符 / 数值 / 固定短语三道闸全都看不出来。facts 里给出的 order_title
    # 与 pending_order_title 因此也按同一口径强制保留（compact 容忍排版空格，
    # 不做「手表 S2 ≈ 智能手表 S2」这类语义等价放行——宁可少一次润色）。
    structural_phrases = tuple(
        str(facts.get(key) or "")
        for key in ("order_title", "pending_order_title")
    ) if facts else ()
    missing_phrases = [
        phrase
        for phrase in (*REQUIRED_PHRASES, *structural_phrases)
        if phrase and _compact(phrase) in draft_compact and _compact(phrase) not in polished_compact
    ]
    if missing_phrases:
        violations.append("missing_phrase")
        detail["missing_phrases"] = missing_phrases

    # --- 信息压缩：长答案被压成一句话 ---
    if len(draft_compact) >= _COMPRESSION_MIN_DRAFT and len(polished_compact) < len(draft_compact):
        keep_ratio = len(polished_compact) / len(draft_compact)
        detail["keep_ratio"] = round(keep_ratio, 3)
        if keep_ratio < _COMPRESSION_MIN_KEEP:
            violations.append("information_loss")

    # --- 政策外承诺 ---
    promises = [word for word in FORBIDDEN_PROMISES if word in polished and word not in draft]
    if promises:
        violations.append("ungrounded_promise")
        detail["promises"] = promises

    # --- 语义反转 ---
    flips = [
        positive
        for negative, positives in _NEGATION_FLIPS
        if negative in draft
        for positive in positives
        if positive in polished
    ]
    if flips:
        violations.append("negation_flip")
        detail["flips"] = flips

    # --- 义务强度翻转：需/必须 ↔ 建议/可以，改变用户义务性质 ---
    # 用 compact 消除排版空格（「建 议提供」不该因空格漏检），但不做模糊匹配
    obligation_flips = _obligation_flips(draft_compact, polished_compact)
    if obligation_flips:
        violations.append("obligation_flip")
        detail["obligation_flips"] = obligation_flips

    # 去重且保持顺序，便于日志阅读
    violations = list(dict.fromkeys(violations))
    detail["violations"] = violations

    if violations:
        return GroundingResult(False, draft, False, violations, detail)
    return GroundingResult(True, polished, True, [], detail)
