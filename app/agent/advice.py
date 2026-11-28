"""无订单号时的「先答题、再要单」预答层。

问题背景（用户实测）：问「保温杯刻字了能退吗」时，流程只看到 intent=return
且没有订单号，就回一句「请提供订单号」。用户的疑问一个字都没被回答，被反问
了一个他还得先去查的东西——这是固定答案，不是客服。

本层做三件事，都只依赖用户原话与政策条款，不依赖模型：
1. `read_situation()`：按原话里的**情形词**判断问的是哪一类（定制 / 缺陷 /
   使用状态 / 受限类目 / 发货状态 / 无理由），给出对应条款的针对性结论；
2. `is_followup()` + `build_advice(context_key=…)`：指代性追问（「为什么不能退」）
   接着上一轮的情形答，并补一条上一轮没说过的新信息（分界点 / 凭证 / 下一步）；
3. `match_order()`：话里提到商品特征时，若在**本人订单**里能唯一命中，就直接
   关联那一单，连单号都不用问；不能唯一命中就不猜（宁可追问，不可认错单）。

两条纪律：
- 所有结论只复述 `app/data/policies.md` 的条款，不新增承诺（不写「一定能退」）；
- 无订单号时无法知道签收时间，所以只回答「规则上怎么判」，期限留给核单。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from ..domain.orders import Order

# ---------------------------------------------------------------------------
# 1. 情形识别：词表按「同一类语义」分组，组内是并列说法
# ---------------------------------------------------------------------------

# 缺陷：指向商品本身有问题（R-02）。与 B7 的 slots 表同源，但这里只用于
# 「解释规则」，不参与资格判定，所以可以收得宽一点。
_DEFECT_RE = re.compile(
    r"破损|损坏|坏了|坏掉|破了|破洞|摔坏|裂了|裂开|开裂|碎了|压坏|变形|掉漆|磨花|"
    r"故障|失灵|开不了机|不能开机|无法开机|不能用|用不了|不好用|不工作|没反应|"
    r"没声音|没有声音|不出声|不响|充不进电|无法充电|死机|黑屏|闪屏|"
    r"漏水|漏了|渗水|不保温|掉色|褪色|起球|开线|脱线|缩水|"
    r"发霉|霉了|变质|异味|发臭|酸了|烂了|不新鲜|过期|"
    r"漏发|少发|错发|发错|与描述不符|货不对板|质量问题|瑕疵|次品"
)
# 个性化属性：商品属性，不是缺陷（R-03 定制类）
_CUSTOM_RE = re.compile(
    r"刻字|刻了字|刻名|刻上|刻的|激光刻|定制|订制|diy|绣字|绣名|刺绣|印字|印名|"
    r"加了名字|打上名字|专属|私人订制"
)
# 使用状态：商品的现状，不是缺陷，但会破坏「完好、不影响二次销售」
_USED_RE = re.compile(
    r"洗过|洗了|水洗|穿过|穿过了|用过|用过了|拆封|拆开|已开封|开封了|剪标|剪了吊牌|"
    r"吊牌没了|包装扔了|包装丢了|试穿"
)
# 受限类目：R-03 明确排除的清单
_RESTRICTED_RE = re.compile(r"内衣|内裤|文胸|生鲜|水果|蔬菜|海鲜|虚拟|充值|会员卡|激活")
# 发货状态：R-04
_UNSHIPPED_RE = re.compile(r"没发货|还没发|未发货|待发货|没收到|未收到|没签收|拒收|没派送|物流不动")
# 无理由动机：R-01
_NO_REASON_RE = re.compile(
    r"不喜欢|不想要|不想要了|买错|拍错|多拍|重复下单|尺码|大小不合|不合身|"
    r"颜色不对|和想象|没想象|后悔|七天无理由|无理由|"
    # B11：口语里「用不上 / 闲置」和「不想要」是同一类动机，此前会落到
    # unknown 分支去检索条款，答成 R-04 未发货规则，跟用户问的完全不搭。
    r"用不上|用不着|闲置|吃灰|多买了|拍错了|下错单"
)


# 指代性追问：原话里没有新的情形词，但明显在追问上一句的结论
# （「为什么不能退」「那还能退吗」）。这类句子按「上文情形」续答；不设这条，
# 用户追问一句就会被当成新问题，重新甩一遍通用话术——B8 残留的固定答案。
_REFERENTIAL_RE = re.compile(
    r"这|那|它|刚才|刚刚|上面|前面|到底|凭什么|为什么|为啥|怎么办|咋办|"
    r"行不行|能不能|可以吗|退得了|什么意思|啥意思|所以|还是|具体"
)
_FOLLOWUP_MAX = 14  # 追问都短；长句里的「这」多半是新信息，不当作指代


def is_followup(text: str) -> bool:
    """是否是「接着说上一句」的追问（长短 + 指代词双条件，避免误接新问题）。"""
    t = (text or "").strip()
    return bool(t) and len(t) <= _FOLLOWUP_MAX and bool(_REFERENTIAL_RE.search(t))


@dataclass(frozen=True)
class Situation:
    key: str
    headline: str
    notes: tuple[str, ...]
    clauses: tuple[str, ...]
    followup: str = ""   # 用户追问「为什么 / 那还能退吗」时补的那一层说明


_SITUATIONS: dict[str, Situation] = {
    "custom_defect": Situation(
        key="custom_defect",
        headline="定制类商品不走七天无理由，但「做坏了」算质量问题",
        notes=(
            "只是不喜欢这个刻字 / 不想要这个定制：R-03 把定制类排除在七天无理由之外，这条走不通。",
            "但刻歪、掉字、掉漆、漏水这类是商品本身的问题，按质量问题算，签收后 15 天内可以退换，运费由商家承担。",
        ),
        clauses=("R-03", "R-02", "R-05"),
        followup=(
            "刻字是定制品，「不想要了」这条走不通；但「做坏了」属于质量问题，能退。"
            "分界点是「商品被做坏」而不是「你不喜欢了」，凭证是刻歪、掉字、漏水的照片或视频。"
        ),
    ),
    "custom_only": Situation(
        key="custom_only",
        headline="定制 / 刻字的商品不适用七天无理由退货",
        notes=(
            "定制类（刻字、刻名、印字、专属设计）在 R-03 的排除清单里，所以「刻了字不想要了」不能无理由退。",
            "但如果商品本身有质量问题（刻坏、破损、故障），仍按质量问题处理：15 天内可退换，运费商家承担。",
        ),
        clauses=("R-03", "R-02"),
        followup=(
            "刻字 / 定制在 R-03 的排除清单里，「不想要了」退不了；能走的是质量问题通道——"
            "刻坏、破损、漏水按 R-02 算，签收后 15 天内可退换、运费商家承担，要照片或视频做凭证。"
        ),
    ),
    "defect": Situation(
        key="defect",
        headline="你说的属于质量问题，走 R-02 这条通道更宽",
        notes=(
            "破损、功能故障、错发漏发、与描述不符，签收后 15 天内可申请退货或换货，且不受「不支持无理由退货类目」限制。",
            "需要提供问题照片或视频，必要时配合寄回检测；退货运费由商家承担。",
        ),
        clauses=("R-02", "R-05"),
        followup=(
            "破损 / 故障属于 R-02 质量问题：签收后 15 天内可退换，运费商家承担，"
            "不受「不支持无理由退货类目」限制；需要照片或视频，必要时配合寄回检测。"
        ),
    ),
    "restricted": Situation(
        key="restricted",
        headline="这个类目不在七天无理由范围内",
        notes=(
            "内衣、生鲜食品、虚拟商品、定制类属于 R-03 的排除清单，不能走七天无理由。",
            "如果确实有质量问题（发霉、破损、错发），按 R-02 的 15 天质量通道处理，仍然可以退换。",
        ),
        clauses=("R-03", "R-02"),
        followup=(
            "这一类目在 R-03 的排除清单里，七天无理由走不通；"
            "确有质量问题（发霉、破损、错发）就按 R-02 的 15 天通道处理，仍可退换。"
        ),
    ),
    "unshipped": Situation(
        key="unshipped",
        headline="还没发货的话不用退货，直接取消订单就行",
        notes=(
            "已付款未发货可以申请取消订单，全额退款，不用走退货流程。",
            "已发货未签收可以拒收，或者签收后按七天无理由申请。",
        ),
        clauses=("R-04",),
        followup=(
            "已付款未发货直接取消订单、全额退款，不用走退货；"
            "已发货未签收可以拒收，签收后想退就走七天无理由。"
        ),
    ),
    "used": Situation(
        key="used",
        headline="「已经用过」会影响无理由退货，但不影响质量问题",
        notes=(
            "七天无理由要求商品完好、不影响二次销售，洗过、穿过、拆封使用通常就不满足了。",
            "如果是质量原因（掉色、破损、故障），不受这条限制，15 天内仍可申请。",
        ),
        clauses=("R-01", "R-02"),
        followup=(
            "「拆封 / 洗过」破坏的是「商品完好、不影响二次销售」这个前提，所以无理由这条走不通；"
            "如果原因是质量（掉色、破损、故障），不受这条限制，15 天内仍可申请。"
        ),
    ),
    "no_reason": Situation(
        key="no_reason",
        headline="普通类目走七天无理由，条件是签收 7 天内、商品完好",
        notes=(
            "签收当日不计入，自签收次日起算第 1 天，第 7 天 24:00 前提交都在期限内。",
            "吊牌、包装、赠品要保持完整；无理由退货的运费由你承担。",
        ),
        clauses=("R-01", "R-05"),
        followup=(
            "普通类目：签收次日起算 7 天、商品完好、运费自理；"
            "超期或影响二次销售就只能协商，协商不成我帮你转人工。"
        ),
    ),
    "unknown": Situation(
        key="unknown",
        headline="退货有两条通道，看你这单属于哪一种",
        notes=(
            "七天无理由：普通类目、签收 7 天内、商品完好，运费自理。",
            "质量问题：破损 / 故障 / 错发漏发 / 与描述不符，签收 15 天内，运费商家承担，不受类目限制。",
        ),
        clauses=("R-01", "R-02"),
    ),
}

# 命中证据的展示顺序：先说什么，用户就觉得「你在听我讲」
_EVIDENCE_ORDER = (
    ("custom_defect", "custom"),
    ("custom_only", "custom"),
    ("defect", "defect"),
    ("restricted", "restricted"),
    ("unshipped", "unshipped"),
    ("used", "used"),
    ("no_reason", "no_reason"),
)

_PATTERNS = {
    "defect": _DEFECT_RE,
    "custom": _CUSTOM_RE,
    "used": _USED_RE,
    "restricted": _RESTRICTED_RE,
    "unshipped": _UNSHIPPED_RE,
    "no_reason": _NO_REASON_RE,
}


def read_situation(message: str) -> str:
    """按原话判定情形；顺序即优先级（定制+缺陷要合并说，不能只说其中一半）。"""
    text = message or ""
    hit = {name: bool(pattern.search(text)) for name, pattern in _PATTERNS.items()}
    if hit["custom"] and hit["defect"]:
        return "custom_defect"
    if hit["custom"]:
        return "custom_only"
    if hit["defect"]:
        return "defect"
    if hit["restricted"]:
        return "restricted"
    if hit["unshipped"]:
        return "unshipped"
    if hit["used"]:
        return "used"
    if hit["no_reason"]:
        return "no_reason"
    return "unknown"


def evidence_of(message: str, key: str) -> str:
    """取出用户原话里命中的那个词，用来「复述他的问题」，而不是泛泛作答。"""
    text = message or ""
    order = [name for k, name in _EVIDENCE_ORDER if k == key] or []
    for name in order:
        match = _PATTERNS[name].search(text)
        if match:
            return match.group(0)
    if key == "custom_defect":
        match = _DEFECT_RE.search(text)
        if match:
            return match.group(0)
    return ""


def _extra_notes(message: str, key: str) -> tuple[str, ...]:
    """补充说明：主情形说清了，但原话里还有别的相关事实时别漏掉。"""
    text = message or ""
    notes: list[str] = []
    if key != "used" and _USED_RE.search(text):
        notes.append("另外，已经洗过 / 拆封使用会破坏「不影响二次销售」，无理由这条路基本走不通。")
    return tuple(notes)


@dataclass(frozen=True)
class Advice:
    key: str
    headline: str
    body: str          # 规则说明本体（不含「要单」那句，便于换成更具体的追问）
    ask: str           # 默认的「要单」追问
    clauses: tuple[str, ...]
    evidence: str = ""       # 用户原话里命中的词，用来复述他的问题
    basis: str = "situation"  # situation / context / clauses / generic：这一版答案的依据
    contextual: bool = False  # 是否承接上一轮情形续答

    @property
    def text(self) -> str:
        return f"{self.body}\n\n{self.ask}"

    @property
    def informative(self) -> bool:
        """是否比「我还不确定」更有信息量——决定它能否替代固定的兜底话术。"""
        return self.key != "unknown" or self.basis == "clauses"

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "headline": self.headline,
            "clauses": list(self.clauses),
            "basis": self.basis,
            "text": self.text,
        }


_ASK_ORDER = (
    "具体到你这单，还要看签收时间和订单状态——请把订单号发给我"
    "（在「我的订单」里可以复制，例如 O20261115001），我按你的订单核一遍期限内还能不能办。"
)
_ASK_ORDER_GENERIC = (
    "想办的话请把订单号发给我（在「我的订单」里可以复制，例如 O20261115001），"
    "我按你的订单核一遍期限再给结论。"
)

_NEXT_STEP_QUALITY = (
    "下一步：把订单号发我算剩余期限；有照片或视频先发过来，我按质量问题通道建单。"
)
_NEXT_STEP_GENERIC = "下一步：把订单号发我，我按你的签收时间算还剩几天。"
_ASK_ORDER_CONTEXT = (
    "订单号发我就能核你这单的具体期限（「我的订单」里可以复制，例如 O20261115001）。"
)

# 「为什么」问的是判定依据，和「能不能 / 怎么办」不是同一个问题：
# 前者要回答「卡在哪一条规则」，后者要回答「还有没有别的路」。同一个模板
# 套两种追问，用户会立刻发现你在复读——这里按问法分叉。
_WHY_RE = re.compile(r"为什么|为啥|凭什么|什么原因|怎么会|依据")

_WHY_NOTES: dict[str, str] = {
    "custom_only": (
        "R-03 是「按类目排除」，不是「按行为排除」：定制类被直接列进不支持七天无理由的清单，"
        "所以不看你有没有拆封、也不看还剩几天，这类商品本身就不在 7 天无理由的范围内；"
        "条款给的唯一例外是质量问题，按 R-02 处理。"
    ),
    "custom_defect": (
        "R-03 按类目把定制类排除在七天无理由之外，所以「不想要了」没有依据；"
        "但 R-02 没有排除定制类——只要证明是「做坏了」（刻歪、掉字、漏水），就走质量通道。"
    ),
    "restricted": (
        "R-03 直接按类目排除（内衣、生鲜、虚拟、定制），不看商品状态；"
        "条款唯一的例外是同一条里写的「如存在质量问题，按 R-02 处理」。"
    ),
    "used": (
        "R-01 的前置条件是「商品完好、不影响二次销售」，洗过、拆封破坏的正是这个前提，"
        "所以不是期限到了，是条件不成立——质量原因另算，走 R-02。"
    ),
    "unshipped": (
        "没发货就不存在「退货」这件事：R-04 允许直接取消订单、全额退款，不需要走退货校验。"
    ),
    "no_reason": (
        "R-01 要求三个条件同时成立：签收 7 天内、商品完好、不影响二次销售；"
        "你这一单卡在哪一条，要按签收时间核过才知道。"
    ),
    "defect": (
        "R-02 把破损、功能故障、错发漏发、与描述不符单列成一条通道，"
        "所以它不受类目限制，期限也比无理由宽（15 天），运费由商家承担。"
    ),
}


def _clause_bullets(hits) -> list[str]:
    """把检索到的条款写成「编号 + 标题 + 要点」，谈的是具体条款而不是通用说明。"""
    bullets: list[str] = []
    for clause_id, title, snippet in hits[:2]:
        body = (snippet or "").strip().rstrip("…")
        bullets.append(f"· 【{clause_id}】{title}：{body}")
    return bullets


def build_advice(
    message: str, *, reason: str = "", claimed_quality_dropped: bool = False,
    context_key: str = "", context_evidence: str = "", clause_hits=(),
) -> Advice:
    """按原话生成「针对性规则说明 + 要单」。

    三种依据，优先级从高到低：
    1. `situation`：原话里有情形词（刻字 / 破了 / 洗过…）→ 按那类条款答；
    2. `context`：原话是指代性追问（「为什么不能退」）→ 接着上一轮的情形答，
       并给一条上一轮没说过的新信息（分界点 / 凭证要求 / 下一步），不复读原话；
    3. `clauses`：情形不明 → 用原话检索条款，引用真实条款标题与要点作答，
       而不是固定甩「两条通道」的通用话术。

    `claimed_quality_dropped`：模型声称质量问题但原话里没有缺陷证据，被 B7 槽位
    校验降级过——这时要明确告诉用户「按哪条口径判的」，否则他会以为自己被敷衍。
    """
    text = message or ""
    raw_key = read_situation(text)
    key = raw_key
    contextual = False
    if (
        raw_key == "unknown"
        and context_key in _SITUATIONS
        and context_key != "unknown"
        and is_followup(text)
    ):
        key = context_key
        contextual = True

    situation = _SITUATIONS[key]
    evidence = (context_evidence if contextual else "") or evidence_of(text, key)

    if contextual:
        label = evidence or situation.headline
        basis = "context"
        if _WHY_RE.search(text):
            intro = f"接着刚才说的「{label}」，你问的是为什么——卡在条款这一层："
            main = _WHY_NOTES.get(key) or situation.followup
        else:
            intro = f"接着刚才说的「{label}」，直接说结论："
            main = situation.followup
        lines = [
            intro,
            f"· {main}",
            f"· {_NEXT_STEP_QUALITY if 'R-02' in situation.clauses else _NEXT_STEP_GENERIC}",
        ]
    elif raw_key == "unknown" and clause_hits:
        basis = "clauses"
        lines = ["你这句话里没有明确的情形词，我按你问的内容去条款里查了一下："]
        lines.extend(_clause_bullets(clause_hits))
        lines.append("· 能不能办还要按你这单的签收时间算，条款本身不代替订单状态。")
    else:
        basis = "generic" if raw_key == "unknown" else "situation"
        if raw_key == "unknown":
            intro = "这句里还没说到具体情形，我先把两条通道摆清楚："
        elif evidence:
            intro = f"你问的是「{evidence}」这种情况，先给结论："
        else:
            intro = "先按你说的情况给结论："
        lines = [intro, f"· {situation.headline}"]
        lines.extend(f"· {note}" for note in situation.notes)
        lines.extend(f"· {note}" for note in _extra_notes(text, key))

    if claimed_quality_dropped and not contextual:
        lines.append(
            "· 你这句话里没有描述具体损坏，我按无理由的口径判断；"
            "如果确实有破损、故障、错发，直接把症状说清楚，我改走质量问题通道。"
        )

    if basis == "clauses":
        clauses = tuple(clause_id for clause_id, _, _ in list(clause_hits)[:2]) or ("R-01",)
        ask = _ASK_ORDER_GENERIC
    elif contextual:
        clauses = situation.clauses
        ask = _ASK_ORDER_CONTEXT
    else:
        clauses = situation.clauses
        ask = _ASK_ORDER_GENERIC if key == "unknown" else _ASK_ORDER
    return Advice(
        key=key, headline=situation.headline, body="\n".join(lines), ask=ask,
        clauses=clauses, evidence=evidence, basis=basis, contextual=contextual,
    )


# ---------------------------------------------------------------------------
# 2. 订单识别：话里提到商品特征时，能唯一命中就不问单号
# ---------------------------------------------------------------------------

_SPACE_RE = re.compile(r"[\s\u3000]+")
_GENERIC_TOKENS = {
    "套装", "正品", "官方", "旗舰", "包邮", "新款", "礼盒", "组合", "小样", "两件", "一件",
}
_TOKEN_MIN, _TOKEN_MAX = 2, 5


@dataclass(frozen=True)
class OrderMatch:
    order: Order
    token: str


def _tokens(title: str, category: str) -> set[str]:
    norm = _SPACE_RE.sub("", title or "")
    tokens: set[str] = set()
    for size in range(_TOKEN_MIN, _TOKEN_MAX + 1):
        for start in range(len(norm) - size + 1):
            token = norm[start:start + size]
            if token in _GENERIC_TOKENS:
                continue
            tokens.add(token)
    if category:
        tokens.add(category)
    return tokens


def _norm(text: str) -> str:
    return _SPACE_RE.sub("", text or "")


def match_order(orders: Iterable[Order], user_id: str, message: str) -> OrderMatch | None:
    """从**本人**订单里认出唯一一单；认不出或有歧义都返回 None。

    唯一性按「该 token 只出现在这一单」判定：本人有两单都含「保温杯」时，
    「保温杯」不算证据，宁可追问也不猜——认错单比多问一句贵得多。
    """
    text = _norm(message)
    if not text:
        return None
    own = [o for o in orders if o.user_id == user_id]
    if not own:
        return None

    best: list[tuple[int, Order, str]] = []
    for order in own:
        title = _norm(order.title)
        others = [o for o in own if o is not order]
        order_best = 0
        order_token = ""
        # 同长度 token 必须稳定挑选：set 的迭代顺序受 hash seed 影响，
        # 否则同一句话在不同进程里会拿不同的词去认单（诊断页实测到）。
        for token in sorted(_tokens(order.title, order.category), key=lambda t: (-len(t), t)):
            if len(token) <= order_best or token not in text:
                continue
            if any(token in _norm(o.title) or token == o.category for o in others):
                continue  # 别的订单也符合 → 不是唯一证据
            if token in order.category and not title:
                continue
            order_best, order_token = len(token), token
        if order_best:
            best.append((order_best, order, order_token))

    if not best:
        return None
    best.sort(key=lambda item: (-item[0], item[1].order_id))
    if len(best) > 1 and best[0][0] == best[1][0]:
        return None  # 打平：两单都能对上，不猜
    return OrderMatch(order=best[0][1], token=best[0][2])
