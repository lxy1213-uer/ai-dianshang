"""槽位证据校验：理解层给出的结论必须能被用户原话支持。

## 为什么需要这一层（B7）

B6 的定向探针 x09 暴露了一个**承诺性错误**：用户说「O20261118004 保温杯刻字了能退吗」，
理解层把「刻字」判成质量问题（`reason=quality`），代码照单全收，于是：

- 绕过类目限制：该订单类目是「定制」，按 R-03 不支持七天无理由；
- 走错通道：`quality` 把期限从 7 天放宽到 15 天；
- 直接进待确认态：回复「符合质量问题退货条件、还剩 13 天、运费由商家承担」。

真实模型在 B6 上就是这么错的（状态 AWAIT_CONFIRM，引用 R-02/R-05），而规则引擎在同一句上
判对（R-03 + 转人工）。差别不在模型笨，而在**槽位没有证据约束**：`intent` 有 B1 的
状态硬护栏，`reason` 却是模型自由填写、代码无条件采信。

`reason` 在这里不是标签，是**承诺**——它决定走哪个通道、给几天、运费谁承担、是否受类目
限制。凡是承诺性结论都必须有证据，这与 B1「状态是代码的职权」是同一条原则。

## 这一层做什么、不做什么

- 只做**减法**：声称质量问题但原话里没有缺陷证据时，降级为 `no_reason` 并记录证据缺口。
  降级是保守方向（少给一个通道），不会凭空放宽期限、不会把有缺陷的商品判成无理由。
- **不升级**：原话里有「屏幕碎了」但模型说 `no_reason` 时，本层不擅自改成 `quality`——
  给通道就是给承诺，证据不足时宁可让用户补充，也不替用户宣布资格。这条留给下一轮（B8）。
- **不碰 intent / 状态**：那是 B1 护栏的职责，两层互不交叉，避免一个函数同时管两件事。

## 三个容易混的语义

1. **缺陷**（破损/故障/漏发/变质…）→ 支持 `quality`；
2. **个性化属性**（刻字/定制/绣名…）→ 描述的是商品属性，不是缺陷，「刻字了」不等于「刻坏了」；
3. **使用状态**（已拆封/洗过/穿过…）→ 描述的是商品现状，也不构成质量证据。

第 2、3 类是售后场景里最常见的误判来源，所以单独列一张表并在结论里注明命中了哪一类，
而不是笼统地说「没找到证据」。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .domain.orders import REASON_NO_REASON, REASON_QUALITY

# --- 缺陷证据 -------------------------------------------------------------
# 只收「指向商品本身有问题」的说法。刻意避开会出现在别的语境里的词：
# 「不符合退货条件」里的「不符」不是缺陷，「裂变」这类也不是——本层判「有证据」
# 就等于放行质量通道，误收就是给错通道（不安全方向）。
_DEFECT_RE = re.compile(
    "破损|损坏|破坏|破洞|破了|摔坏|坏了|坏掉|坏了一半|是坏的|有坏|"
    "裂了|裂开|裂纹|开裂|裂|碎了|压坏|变形|磨花|掉漆|"
    "划痕|瑕疵|污渍|脏了|褪色|掉色|"
    "故障|失灵|开不了机|不能开机|无法开机|不能用了|不能用|用不了|不好用|"
    "不工作|没反应|没声音|没有声音|不出声|不响|充不进电|无法充电|死机|黑屏|闪屏|"
    "漏水|漏液|渗水|漏了|"
    "漏发|少发|少件|错发|发错|寄错|货不对板|与描述不符|描述不符|和图片不一样|"
    "发霉|变质|异味|发臭|酸了|烂了|不新鲜|过期|"
    "刻歪|歪了|糊了|晕开|"
    "质量问题"
)

# --- 个性化属性：描述「这件商品被定制过」，不是缺陷 -------------------------
_PERSONALIZATION_RE = re.compile(
    "刻字|刻了字|刻名|刻了名|刻上|定制|订制|定制款|私人定制|"
    "绣字|绣名|刺绣|印字|印名|加了名字|打上名字|专属"
)

# --- 使用状态：描述「这件商品现在什么样」，也不是缺陷 -----------------------
_CONDITION_RE = re.compile(
    "拆封|拆了|已拆|开箱|洗过|洗了|穿过|穿过了|用过了|用过|试过|"
    "吊牌剪了|剪了吊牌|没吊牌|无吊牌|包装扔了|包装丢了"
)

# 否定语素：只认紧邻在前的形态，窗口 3 字以内且不跨标点。
# 「不是质量问题，是漏发」里「质量问题」被否定，但「漏发」仍然成立——按命中点逐个判，
# 不按整句判，否则一出现否定词就把整句证据作废。
_NEGATORS = ("没有", "没能", "并未", "并不", "不是", "不算", "没", "未", "无", "非", "不")
_NEG_WINDOW = 3
_STOP = "，。；！？,;!?、\n  "

# 结论码：调用方按码决定要不要向上层暴露「模型被降级」这件事
RULE_NOT_CLAIMED = "not_claimed"          # 本来就没声称质量问题，不干预
RULE_SUPPORTED = "supported"              # 声称质量问题且原话有缺陷证据
RULE_NO_EVIDENCE = "no_evidence"          # 声称质量问题但原话没有任何缺陷线索
RULE_ATTRIBUTE_ONLY = "attribute_only"    # 只有个性化属性 / 使用状态，属于属性描述
RULE_NEGATED = "negated"                  # 提到的缺陷全部被否定（「没有破损」）


@dataclass(frozen=True)
class ReasonVerdict:
    """一次 reason 槽位校验的结论（不可变，便于断言与留痕）。"""

    claimed: str
    reason: str
    rule: str
    evidence: tuple[str, ...] = ()
    attributes: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()

    @property
    def corrected(self) -> bool:
        return self.claimed != self.reason

    def to_dict(self) -> dict:
        return {
            "claimed": self.claimed,
            "reason": self.reason,
            "rule": self.rule,
            "evidence": list(self.evidence),
            "attributes": list(self.attributes),
            "conditions": list(self.conditions),
        }


@dataclass
class _Hit:
    word: str
    start: int
    negated: bool = False


def _scan(pattern: re.Pattern[str], text: str) -> list[_Hit]:
    hits: list[_Hit] = []
    for match in pattern.finditer(text):
        prefix = text[max(0, match.start() - _NEG_WINDOW): match.start()]
        # 间隔里出现标点即视为跨小句，否定不再作用于后一个词（只看最后一个小句）
        cut = max((prefix.rfind(ch) for ch in _STOP), default=-1)
        if cut >= 0:
            prefix = prefix[cut + 1:]
        negated = prefix.endswith(_NEGATORS)
        hits.append(_Hit(match.group(0), match.start(), negated))
    return hits


def _dedupe(words: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(words))


def defect_evidence(text: str) -> tuple[str, ...]:
    """返回原话中**未被否定**的缺陷语素（按出现顺序去重）。"""
    return _dedupe([h.word for h in _scan(_DEFECT_RE, text or "") if not h.negated])


def validate_reason(claimed: str, text: str) -> ReasonVerdict:
    """校验 `reason` 槽位是否被原话支持；不支持则降级（只做减法）。"""
    claimed = claimed if claimed in (REASON_NO_REASON, REASON_QUALITY) else REASON_NO_REASON
    raw = text or ""

    if claimed != REASON_QUALITY:
        return ReasonVerdict(claimed=claimed, reason=claimed, rule=RULE_NOT_CLAIMED)

    defect_hits = _scan(_DEFECT_RE, raw)
    evidence = _dedupe([h.word for h in defect_hits if not h.negated])
    attributes = _dedupe([h.word for h in _scan(_PERSONALIZATION_RE, raw)])
    conditions = _dedupe([h.word for h in _scan(_CONDITION_RE, raw)])

    if evidence:
        return ReasonVerdict(
            claimed=claimed, reason=REASON_QUALITY, rule=RULE_SUPPORTED,
            evidence=evidence, attributes=attributes, conditions=conditions,
        )

    if defect_hits:
        # 命中了但全部被否定（「没有破损」/「不是质量问题」）
        rule = RULE_NEGATED
    elif attributes or conditions:
        rule = RULE_ATTRIBUTE_ONLY
    else:
        rule = RULE_NO_EVIDENCE

    return ReasonVerdict(
        claimed=claimed, reason=REASON_NO_REASON, rule=rule,
        evidence=(), attributes=attributes, conditions=conditions,
    )


# --- 订单号槽位：格式与「空值」---------------------------------------------
# B9 实测：真实模型对「没有单号」的表达常常是 JSON 里的 "null" / "none"。
# 这些字符串在 Python 里是**真值**，代码无条件采信就会拿 "null" 去查订单，
# 用户看到的是「没有查到订单 null」——和「这单为什么不能退」毫无关系。
# 与 reason 同一条原则：槽位先过格式关，再交给业务。
_JUNK_ORDER_IDS = {
    "null", "none", "nil", "undefined", "nan", "false", "true", "n/a", "na",
    "未知", "无", "没有", "空", "不清楚", "不知道", "不记得",
}
_ORDER_ID_RE = re.compile(r"[A-Za-z0-9]{5,}")
_TRAILING_PUNCT = " \t\"'`。，,;；:：!！?？、"


def normalize_order_id(raw: object) -> str | None:
    """把模型给的「订单号」洗成可用值；洗不出来就返回 None（宁可追问，不可乱查）。

    只做格式关，不判断订单是否存在——「格式对但查不到」要照常回「订单不存在」，
    那是用户能自己纠正的信息；把 "null" 当单号查，用户只会看到一句莫名其妙的报错。
    """
    if raw is None:
        return None
    text = str(raw).strip().strip(_TRAILING_PUNCT)
    text = re.sub(r"[\s\u3000]+", "", text)
    if not text or text.lower() in _JUNK_ORDER_IDS:
        return None
    if not _ORDER_ID_RE.fullmatch(text):
        return None
    return text.upper()


# --- 状态型问句证据（B12）--------------------------------------------------
# holdout4 p07 暴露：真实模型把「这笔 O… 的钱还能退一次吗」判成 policy，
# 于是按通用政策检索答成「退款时效」，既没核单也没说「不能重复申请」。
# 这类句子的关键不是「问句」形式，而是**已经发生的退款状态 + 再次申请意图**：
# 它针对的是某一笔具体订单，答案由订单状态决定，必须走核单，不能当通用规则答。
#
# 与 B7 同一条原则：证据由**两组语素组合**给出，单看「退款」两字会误伤
# 「退款多久到账」这类真政策问句，所以要求「状态事实」与「再次申请」同时出现，
# 或其中一组 + 明确的订单指向（单号 / 这笔 / 该订单）。
_REFUND_STATE_RE = re.compile(
    "已退款|已经退款|退过款|退款成功|钱退了|款退了|退到账|已经退了|已经退过|"
    "退过一次|之前退过|上次退过|退完了|退款已完成"
)
_REAPPLY_INTENT_RE = re.compile(
    "再退|又退|还能退|还能申请|重新申请|重复申请|二次|再一次|退一次|申请一次|"
    "再申请|退第二次|退多一次"
)
_ORDER_REFERENCE_RE = re.compile(
    "这笔|这单|这个订单|该订单|我那单|我的订单|刚才那单|[A-Za-z]{1,2}\\d{8,}|\\b\\d{10,}\\b"
)

RULE_STATUS_QUESTION = "status_reapply"
RULE_STATE_ONLY = "state_without_reapply"
RULE_REAPPLY_ONLY = "reapply_without_state"
RULE_NOT_STATUS_QUESTION = "not_status_question"


@dataclass(frozen=True)
class StatusQuestionVerdict:
    """状态型问句判定：是否需要把 policy 降级给规则引擎重判。"""

    is_status_question: bool
    rule: str
    state_hits: tuple[str, ...] = ()
    reapply_hits: tuple[str, ...] = ()
    order_ref: bool = False

    def to_dict(self) -> dict:
        return {
            "is_status_question": self.is_status_question,
            "rule": self.rule,
            "state_hits": list(self.state_hits),
            "reapply_hits": list(self.reapply_hits),
            "order_ref": self.order_ref,
        }


def status_question_evidence(text: str) -> StatusQuestionVerdict:
    """判定一句问话是不是「针对某笔订单的已退款状态 / 再次申请」。

    触发条件（保守，避免误伤真政策问句）：
    - 状态事实 + 再次申请：如「已经退款了还能再申请一次吗」；
    - 订单指向 + 上述任一组：如「这笔 O… 的钱还能退一次吗」「这单之前退过款」。
    单纯「退款多久到账」「退货政策是几天」两组都不命中，不会被纠正。
    """
    raw = text or ""
    state_hits = tuple(dict.fromkeys(m.group(0) for m in _REFUND_STATE_RE.finditer(raw)))
    reapply_hits = tuple(dict.fromkeys(m.group(0) for m in _REAPPLY_INTENT_RE.finditer(raw)))
    order_ref = bool(_ORDER_REFERENCE_RE.search(raw))

    if state_hits and reapply_hits:
        return StatusQuestionVerdict(True, RULE_STATUS_QUESTION, state_hits, reapply_hits, order_ref)
    if order_ref and state_hits:
        return StatusQuestionVerdict(True, RULE_STATE_ONLY, state_hits, reapply_hits, order_ref)
    if order_ref and reapply_hits:
        return StatusQuestionVerdict(True, RULE_REAPPLY_ONLY, state_hits, reapply_hits, order_ref)
    return StatusQuestionVerdict(False, RULE_NOT_STATUS_QUESTION, state_hits, reapply_hits, order_ref)
