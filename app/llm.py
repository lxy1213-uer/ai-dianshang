"""LLM 抽象层：理解（意图/槽位）+ 表达（措辞润色）。

设计原则：**模型负责理解与表达，代码负责规则与状态**。
- rules 模式：内置规则引擎，零依赖、可离线、结果可复现，用于演示与回归测试；
- llm 模式：调用 OpenAI 兼容的 /chat/completions，失败时自动降级到规则引擎，绝不让接口 500。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from .slots import (
    defect_evidence,
    normalize_order_id,
    status_question_evidence,
    validate_reason,
)
from .usage import Usage, estimated_usage, measured_usage

INTENTS = ("return", "policy", "logistics", "order", "confirm", "hesitate", "cancel", "human", "greeting", "unknown")

# 仍在使用的基础类别词（退货/政策意图已下沉到下方语义特征层）
LOGISTICS_WORDS_FOCUSED = ("物流", "快递", "到哪", "运单", "配送", "签收了吗", "什么时候到")
ORDER_WORDS_FOCUSED = ("订单", "查单", "我的单", "单号", "买的东西")
HUMAN_WORDS = ("人工", "转人工", "真人", "投诉", "找客服")
GREETING_WORDS = ("你好", "您好", "在吗", "hi", "hello", "在么")
CONFIRM_WORDS = ("确认", "确定", "好的", "可以", "没问题", "是的", "提交", "就这样")
# 否定 / 犹豫表达必须先于确认词判断，否则「不确认」「还没确认好」会被当成同意建单
CONFIRM_NEGATIONS = (
    "不确认", "没确认", "不确定", "别确认", "不要确认", "先别", "暂不", "等等",
    "再想想", "稍等", "不急", "先不", "不提交", "不要提交", "还没", "再说",
    "考虑一下", "待会", "晚点", "回头",
)
# B11：撤单是用户口语里极常见的诉求（「还没发出去，能不能撤掉」），
# 原先只有「取消 / 算了 / 不退了」三种说法，实测通用分类器整句漏判。
# 「撤回 / 撤销」限定在订单语境由调用方判断——这里只收动作语素，避免把
# 「撤回刚才那句话」误当成取消售后申请。
CANCEL_WORDS = ("取消", "算了", "不退了", "先不", "不要了", "撤掉", "撤单", "撤销订单", "撤回订单")
# 质量问题判定复用 B7 的缺陷证据表（app/slots.py）。原先这里另有一张独立小词表
# （质量/破损/坏了/故障/漏发/错发/不符/瑕疵/裂/不能用），与缺陷表不同源：
# 真实说法「刻歪了」「掉色了」「发霉了」前者判 no_reason、后者能识别，导致
# 「刻歪的保温杯」被当成无理由 → 撞上定制类目 → 直接拒绝。B8 起统一为一张表。
QUALITY_EVIDENCE_SOURCE = "app/slots.py::defect_evidence"

# --- 语义特征层 -----------------------------------------------------------
# 旧实现用「整句短语表」枚举用户说法，泛化很差：换个动词补语（"退了"）、
# 倒装，或把「退货」塞进一句政策问句，就会漏判或抢错意图。
# 现在拆成可组合的语素信号；这里刻意不收录任何评测案例的原句。
#
# 政策维度词只描述「规则的哪一方面」，刻意不含「退」字本身——
# 否则「我要退货」会被当成在问规则。
POLICY_TOPICS = (
    "政策", "规则", "条款", "期限", "时效", "几天", "多久", "什么时候",
    "运费", "邮费", "包邮", "谁承担", "谁出", "自己付", "到账", "退钱",
    "无理由", "起算", "从哪天", "条件", "范围", "类目", "凭证", "工作日",
)
POLICY_DOC_WORDS = ("政策", "条款", "规则")
QUESTION_MARKERS = (
    "吗", "么", "?", "？", "怎么", "如何", "多久", "几天", "什么时候",
    "是不是", "能不能", "可不可以", "要不要", "是否", "谁", "哪", "有没有",
)
CONSULT_MARKERS = (
    "查", "查询", "问一下", "咨询", "请问", "说说", "讲讲", "解释",
    "了解", "想知道", "看下", "看一下", "帮我看", "告诉我",
)
# 动词「退」：一个字即可覆盖 退了 / 退掉 / 想退 / 能退 / 退货 / 退款 / 退换
RETURN_VERB_RE = re.compile(r"退")
EXCHANGE_WORDS = ("能换", "想换", "换货", "换个", "换成", "换一个", "退换", "换吗")
# 纯口语、不含「退」字的诉求。B11 起两种分类器共用同一张表：
# 原设计故意让通用分类器更窄以模拟「大一统提示词覆盖不足」，但 holdout3 的首次
# 基线证实这不只是模拟差异——真实用户就是这么说话的。收窄带来的漏判会把整句
# 落到 unknown（既不答政策也不办理），比多认一条更糟。仍然只收语素，不收整句。
COLLOQUIAL_RETURN_WORDS = (
    "不想要", "不要了", "不喜欢", "买错", "不合适", "售后", "不打算要",
    "用不上", "用不着", "闲置", "吃灰", "多买了", "拍错了", "下错单",
    "申请售后", "走售后", "办售后", "再申请", "申请一次", "申请退钱",
    "退款申请", "退钱申请",
)

# 单 Agent 的物流词更窄：模拟「大一统提示词」对边缘说法覆盖不足
LOGISTICS_WORDS_GENERIC = ("物流", "快递")

# --- 意图状态一致性硬护栏 -------------------------------------------------
# 状态是代码的职权：hesitate / confirm 只在「等待确认」的流程里有意义。
# B1 真实模型抽样暴露：模型把 IDLE 状态的「这个单子我不想要了」判成 hesitate、
# 把 AWAIT_CONFIRM 状态的「先别确认」判成 unknown，导致状态机从未启动。
# 提示词里的状态约束是软约束；这个纯函数是硬护栏，与表达层 grounding 同构：
# 模型越权改写流程语义时，这一轮意图降级给规则引擎重判（usage 仍记实测）。
WAITING_STATES = ("AWAIT_CONFIRM", "CREATING")
CONFIRM_INTENT_STATES = ("AWAIT_CONFIRM", "CREATING", "TICKETED")


def needs_state_correction(intent: str, state: str, text: str) -> bool:
    """判断模型的意图是否与状态机职权冲突，需要规则引擎重判。"""
    waiting = state in WAITING_STATES
    if intent == "hesitate" and not waiting:
        # 非等待态的「犹豫」多半是主诉求被误判（如纯口语退货被吞掉）
        return True
    if intent == "confirm" and state not in CONFIRM_INTENT_STATES:
        # 没有等待确认的申请，就谈不上「确认」；误判会凭空建单
        return True
    if waiting and intent not in ("hesitate", "confirm", "cancel", "human") \
            and _match(text or "", CONFIRM_NEGATIONS):
        # 等待确认时用户说了否定/犹豫词，模型却没认出来 → 规则引擎兜底
        # （规则里否定确认优先于确认词，重判绝不会误建单）
        return True
    return False


def needs_status_question_correction(intent: str, text: str) -> bool:
    """状态型问句护栏（B12）：把「针对某单的退款状态追问」从 policy 拉回核单路径。

    holdout4 p07：模型看到「钱还能退一次吗」的问句形式判成 policy，于是答成通用
    退款时效，既没核单也没说「不能重复申请」。但这类问题的答案由订单状态决定，
    属于办理路径（return + 单号 → check_return_eligibility），不能当政策咨询。
    只对 `policy` / `unknown` 做减法——真政策问句无状态事实与再次申请语素，不触发。
    """
    if intent not in ("policy", "unknown"):
        return False
    return status_question_evidence(text or "").is_status_question

ORDER_ID_PATTERN = re.compile(r"[A-Za-z]{1,2}\d{8,}|\b\d{10,}\b")


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：中文约 1.5 字/token，用于离线模式的成本对比。"""
    if not text:
        return 0
    return max(1, round(len(text) / 1.5))


def extract_order_id(text: str) -> str | None:
    if not text:
        return None
    match = ORDER_ID_PATTERN.search(text.replace(" ", ""))
    return match.group(0).upper() if match else None


def detect_reason(text: str) -> str:
    """质量问题的判定复用 `app.slots.defect_evidence`（单一事实来源）。"""
    return "quality" if defect_evidence(text) else "no_reason"


@dataclass
class Understanding:
    intent: str = "unknown"
    order_id: str | None = None
    reason: str = "no_reason"
    confidence: float = 0.6
    source: str = "rules"
    tokens: int = 0
    slots: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "order_id": self.order_id,
            "reason": self.reason,
            "confidence": self.confidence,
            "source": self.source,
            "tokens": self.tokens,
        }


def _match(text: str, words: tuple[str, ...]) -> str | None:
    for word in words:
        if word in text:
            return word
    return None


def _has(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def semantic_signals(text: str, *, focused: bool = True) -> dict:
    """把一句话拆成可组合的语义信号，供两种分类器共用。

    只认语素（动词「退」、疑问标记、政策维度词），不认整句，
    因此没见过的说法也能靠组合命中。focused 只影响纯口语词的覆盖宽度。
    """
    low = (text or "").lower()
    # 运行时拼装：通用词表定义在下方，避免在模块加载期引用未定义常量
    logistics_words = tuple(dict.fromkeys(LOGISTICS_WORDS_FOCUSED + LOGISTICS_WORDS_GENERIC))
    has_order_id = extract_order_id(text) is not None
    quality = detect_reason(text) == "quality"
    is_question = _has(low, QUESTION_MARKERS)
    consult = _has(low, CONSULT_MARKERS)
    policy_topic = _has(low, POLICY_TOPICS)

    return_verb = bool(RETURN_VERB_RE.search(low)) or _has(low, EXCHANGE_WORDS)
    # B11：口语诉求两种模式共用一张表（此前通用模式漏掉整类说法）。
    # focused 保留为参数以兼容调用方与对照实验，但不再制造召回差异。
    if return_verb is False:
        return_verb = _has(low, COLLOQUIAL_RETURN_WORDS)
    # 缺陷陈述本身就是售后诉求（「蓝莓有一盒发霉了」没有「退」字也没有单号）。
    # 政策问句在分类器里先于 return 判定，所以「坏了运费谁承担」不会被抢走。
    if quality:
        return_verb = True

    return {
        "has_order_id": has_order_id,
        "quality": quality,
        "is_question": is_question,
        "consult": consult,
        "policy_topic": policy_topic,
        # 明确问物流时不算政策咨询（"快递几天能到"）
        "policy_question": (
            (is_question or consult or _has(low, POLICY_DOC_WORDS))
            and policy_topic
            and not _has(low, logistics_words)
        ),
        "return_verb": return_verb,
    }


class RuleLLM:
    """离线规则引擎。focused=True 模拟"专职 Agent 的聚焦提示词"，False 模拟"单 Agent 的大一统提示词"。"""

    name = "rules"

    def _classify_rules(self, text: str, state: str = "IDLE", focused: bool = True) -> Understanding:
        """规则分类 + **reason 槽位证据校验**（B7）。

        规则引擎靠关键词判「质量问题」，「质量怎么样」「不是质量问题」「刻字了」这类说法
        都会踩中同一个词表。槽位校验只做减法：没有缺陷证据就降级成 no_reason，
        避免凭空多给一条 15 天质量通道。降级记录在 slots.reason_correction。
        """
        understanding = self._classify_rules_raw(text, state, focused)
        verdict = validate_reason(understanding.reason, text or "")
        if verdict.corrected:
            understanding.reason = verdict.reason
            understanding.slots["reason_correction"] = verdict.to_dict()
        return understanding

    def _classify_rules_raw(self, text: str, state: str = "IDLE", focused: bool = True) -> Understanding:
        raw = (text or "").strip()
        low = raw.lower()
        prompt = self._prompt(raw, state, focused)

        order_id = extract_order_id(raw)
        tokens = estimate_tokens(prompt)
        reason = detect_reason(raw)

        # B12：状态型问句（「已退款 / 到账了还能再申请」）——答案由订单状态决定，
        # 必须走核单，不能被「到账 / 退款」这类政策维度词抢成 policy。
        # 与模型路径共用同一份证据函数，两种分类器口径一致（B11 共表同构）。
        # 只在 confirm/human/cancel 之后、policy 判定之前拦截；记入 trace 留痕。
        status_verdict = status_question_evidence(raw)
        if status_verdict.is_status_question:
            understanding = Understanding("return", order_id, reason, 0.88, "rules", tokens)
            understanding.slots["status_question_evidence"] = status_verdict.to_dict()
            return understanding

        # 否定/犹豫只在等待确认时拦截；其余状态说「不确认」不会触发任何建单动作
        if state == "AWAIT_CONFIRM" and _match(raw, CONFIRM_NEGATIONS):
            return Understanding("hesitate", order_id, reason, 0.9, "rules", tokens)
        # CREATING / TICKETED 也要能识别「确认」：分别用于崩溃续办与并发重试返回原工单。
        # 疑问句（"可以退吗"）不能当成确认，否则会误建单。
        if (
            state in ("AWAIT_CONFIRM", "CREATING", "TICKETED")
            and _match(raw, CONFIRM_WORDS)
            and not order_id
            and not _has(low, QUESTION_MARKERS)
        ):
            return Understanding("confirm", None, reason, 0.95, "rules", tokens)
        if _match(raw, CANCEL_WORDS):
            return Understanding("cancel", order_id, reason, 0.9, "rules", tokens)
        if _match(raw, HUMAN_WORDS):
            return Understanding("human", order_id, reason, 0.92, "rules", tokens)

        signals = semantic_signals(raw, focused=focused)
        logistics_words = LOGISTICS_WORDS_FOCUSED if focused else LOGISTICS_WORDS_GENERIC
        order_words = ORDER_WORDS_FOCUSED if focused else ("订单",)

        # 1) 问规则 ≠ 办退货。必须排在 return 之前：否则
        #    「退货要自己付邮费吗」里的「退货」会把整句抢成办理动作。
        if signals["policy_question"]:
            return Understanding("policy", order_id, "no_reason", 0.86, "rules", tokens)
        # 2) 退货 / 换货 / 质量问题带单号 → 进入核单流程
        if signals["return_verb"]:
            return Understanding("return", order_id, reason, 0.9, "rules", tokens)
        # 3) 物流查询优先于裸订单号（「帮我看下 O... 的物流」）
        if _match(raw, logistics_words):
            return Understanding("logistics", order_id, "no_reason", 0.85, "rules", tokens)
        if order_id:
            return Understanding("order", order_id, reason, 0.85, "rules", tokens)
        if _match(raw, order_words):
            return Understanding("order", order_id, "no_reason", 0.7, "rules", tokens)
        if _match(raw, GREETING_WORDS):
            return Understanding("greeting", None, "no_reason", 0.8, "rules", tokens)
        return Understanding("unknown", order_id, reason, 0.4, "rules", tokens)

    # --- 带用量明细的接口（真实编排与计费使用） ---------------------------
    # 规则引擎没有模型调用，只能给「字符估算」，并明确 measured=False、不计费。
    def classify_with_usage(
        self, text: str, state: str = "IDLE", focused: bool = True
    ) -> tuple[Understanding, Usage]:
        started = time.perf_counter()
        understanding = self._classify_rules(text, state, focused)
        ms = (time.perf_counter() - started) * 1000
        return understanding, estimated_usage(
            "rules", understanding.tokens, ms,
            note="规则引擎本地执行，无模型调用；tokens 为 prompt 字符估算，不计费",
        )

    def classify(self, text: str, state: str = "IDLE", focused: bool = True) -> Understanding:
        """兼容旧签名（评测、单元测试与自定义假模型仍可直接调用）。"""
        return self.classify_with_usage(text, state, focused)[0]

    def polish_with_usage(self, draft: str, context: dict | None = None) -> tuple[str, Usage]:
        started = time.perf_counter()
        text, tokens = self.polish(draft, context)
        ms = (time.perf_counter() - started) * 1000
        return text, estimated_usage(
            "rules", tokens, ms,
            note="规则模式不改写话术；tokens 为文本字符估算，不计费",
        )

    def polish(self, draft: str, context: dict | None = None) -> tuple[str, int]:
        """规则模式不改写文案：客服话术模板本身就是最终表达。"""
        return draft, estimate_tokens(draft)

    @staticmethod
    def _prompt(text: str, state: str, focused: bool) -> str:
        if focused:
            return (
                "你是电商售后客服的接待 Agent。判断用户意图，抽取订单号与退货原因。"
                f"当前会话状态：{state}。只输出 JSON。用户：{text}"
            )
        return (
            "你是电商客服，负责查订单、答政策、办退货、查物流、转人工。"
            "请判断意图并抽取订单号，输出 JSON。"
            f"当前状态：{state}。用户：{text}"
        )


class OpenAICompatibleLLM:
    """OpenAI 兼容接口（OpenAI / DeepSeek / 通义 / 本地 vLLM 都可用）。"""

    name = "llm"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @staticmethod
    def _content_text(message: dict) -> str:
        """提取文本内容：兼容 content 为字符串 / 分段数组 / 缺失三种形态。

        部分聚合网关把 Anthropic 响应转成 OpenAI 格式时，content 会是
        [{"type": "text", "text": "..."}]，直接当字符串用会 TypeError。
        """
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        if content is None:
            return str(message.get("text") or message.get("reasoning_content") or "")
        return str(content)

    def _chat(
        self, messages: list[dict], temperature: float = 0.2, max_tokens: int | None = None
    ) -> tuple[str, Usage]:
        """真实调用 /chat/completions，并从响应 usage 字段采集实测用量。

        关键：只有响应里真的带 usage 才 measured=True；兼容接口不返回时
        明确降级成「字符估算 + 不计费」，绝不拿估算值冒充账单。
        """
        import httpx

        started = time.perf_counter()
        payload: dict = {"model": self.model, "messages": messages, "temperature": temperature}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        if getattr(response, "status_code", 200) >= 400:
            # 带上状态码与响应片段：接第三方网关时 401/404/429 的排查全靠这个
            body = getattr(response, "text", "") or ""
            raise RuntimeError(
                f"HTTP {response.status_code} from {self.base_url}/chat/completions: {body[:300]}"
            )
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"响应缺少 choices，无法取出回复：{str(data)[:300]}")
        content = self._content_text(choices[0].get("message") or {})
        ms = (time.perf_counter() - started) * 1000

        # usage 兼容两种命名：OpenAI 的 prompt/completion 与 Anthropic 的 input/output
        raw_usage = data.get("usage") or {}
        prompt_tokens = int(raw_usage.get("prompt_tokens") or raw_usage.get("input_tokens") or 0)
        completion_tokens = int(
            raw_usage.get("completion_tokens") or raw_usage.get("output_tokens") or 0
        )
        total_tokens = int(raw_usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        echoed = str(data.get("model") or "")
        model_note = f"（网关回显模型 {echoed}）" if echoed and echoed != self.model else ""
        if not (prompt_tokens or completion_tokens or total_tokens):
            return content, estimated_usage(
                "llm",
                estimate_tokens("".join(str(m.get("content", "")) for m in messages) + content),
                ms,
                self.model,
                note="API 响应缺少 usage 字段，无法计费；仅记录字符估算" + model_note,
            )
        return content, measured_usage(
            prompt_tokens, completion_tokens, total_tokens, ms, self.model,
            note="token 数来自 API usage 字段" + model_note,
        )

    def classify_with_usage(
        self, text: str, state: str = "IDLE", focused: bool = True
    ) -> tuple[Understanding, Usage]:
        # 判断口径与状态约束写进提示词（软约束），few-shot 例句全部新造，
        # 不使用任何评测集整句，防止照着样本刷分。
        system = (
            "你是电商售后客服的意图识别器。只输出 JSON，字段："
            '{"intent": "return|policy|logistics|order|confirm|hesitate|cancel|human|greeting|unknown",'
            '"order_id": "订单号或 null", "reason": "no_reason|quality", "confidence": 0.0}'
            "\n判断口径：带疑问/咨询且问的是期限、运费、到账、条件等规则细节 → policy（问规则）；"
            "出现动词「退」、换货诉求，或质量问题且带订单号 → return（办业务）；"
            "用户表达放弃购买、换货或售后诉求（哪怕没说「退」字，例如买错了、"
            "尺码不合适这类口语说法），且不是在提问 → 也算 return；"
            "疑问句不算确认，不能判成 confirm。"
            "\n状态约束（当前会话状态见文末，同一句话在不同状态下意图可能不同）："
            "hesitate 只在状态为 AWAIT_CONFIRM / CREATING 时使用——用户对即将提交的申请"
            "表达犹豫或反悔；其它状态下用户的犹豫表达按主诉求分类，不能因为语气犹豫就判 hesitate。"
            "confirm 只在状态为 AWAIT_CONFIRM / CREATING / TICKETED 且用户明确同意提交时使用。"
            "状态为 AWAIT_CONFIRM / CREATING 时，用户回复里的否定词（先别、不提交、再想想等）→ hesitate。"
            "\n状态型问句不算政策咨询：针对某一笔订单的「已退款还能退一次吗」「这单退过款还能再申请吗」"
            "这类问题，答案由订单状态决定，要按 return 带订单号去核单，不能答成通用的退款时效规则。"
            "只有脱离具体订单、问通用规则本身（退款多久到账、退货要几天）才是 policy。"
            "\n示例："
            "\n- 状态 IDLE，用户「这东西买错了，帮我处理一下」→ return"
            "\n- 状态 IDLE，用户「我就先问问运费规则」→ policy"
            "\n- 状态 AWAIT_CONFIRM，用户「先放着，我再想想」→ hesitate"
            "\n- 状态 AWAIT_CONFIRM，用户「嗯，那就办吧」→ confirm"
            f"\n当前会话状态：{state}"
        )
        try:
            content, usage = self._chat(
                [{"role": "system", "content": system}, {"role": "user", "content": text}],
                max_tokens=200,
            )
            payload = json.loads(_strip_code_fence(content))
            intent = payload.get("intent", "unknown")
            if intent not in INTENTS:
                intent = "unknown"
            raw_order_id = payload.get("order_id")
            order_id = normalize_order_id(raw_order_id)
            reason = payload.get("reason") if payload.get("reason") in ("no_reason", "quality") else "no_reason"
            confidence = float(payload.get("confidence", 0.8))
            source = "llm"
            slots: dict = {}
            if raw_order_id not in (None, "") and order_id is None:
                # 订单号格式护栏（B9）：模型常把「没有单号」写成字符串 "null" / "none"，
                # 这是真值，拿它去查订单会得到「没有查到订单 null」。留痕但不采信。
                slots["order_correction"] = {
                    "from": str(raw_order_id)[:24], "to": None, "rule": "format_guard",
                }
            state_correction = needs_state_correction(intent, state, text or "")
            status_correction = needs_status_question_correction(intent, text or "")
            if state_correction or status_correction:
                # 硬护栏：模型对流程状态的理解越权、或把「已退款再申请」的状态型问句
                # 判成政策咨询时，这一轮意图降级给规则引擎重判。
                # usage 仍是这次真实调用的实测口径（模型确实被调了）；
                # 修正本身记录在 slots.intent_correction，trace 与事件里可见。
                # 两条护栏都命中时按 state_guard 记（状态冲突优先级更高）。
                corrected = RuleLLM()._classify_rules(text or "", state, focused)
                slots["intent_correction"] = {
                    "from": intent,
                    "to": corrected.intent,
                    "state": state,
                    "rule": "state_guard" if state_correction else "status_question_guard",
                }
                if status_correction and not state_correction:
                    slots["status_question_evidence"] = status_question_evidence(text or "").to_dict()
                intent = corrected.intent
                order_id = corrected.order_id
                reason = corrected.reason
                confidence = corrected.confidence
                source = "llm-state-guard" if state_correction else "llm-status-guard"

            # 槽位证据校验（B7）：与状态护栏同构，但管的是「结论有没有证据」。
            # reason 决定走 7 天无理由还是 15 天质量通道、运费谁承担、是否受类目限制，
            # 属于承诺性结论；模型说是质量问题、用户原话里却没有缺陷证据时降级，
            # 避免「刻字了能退吗」被当成质量缺陷、绕过定制类目限制（B6 x09）。
            verdict = validate_reason(reason, text or "")
            if verdict.corrected:
                slots["reason_correction"] = verdict.to_dict()
                reason = verdict.reason
                if source == "llm":
                    source = "llm-slot-guard"
            return Understanding(
                intent=intent,
                order_id=order_id,
                reason=reason,
                confidence=confidence,
                source=source,
                tokens=usage.display_tokens,
                slots=slots,
            ), usage
        except Exception:  # 网络/额度/解析异常一律降级，保证可用性
            fallback, usage = RuleLLM().classify_with_usage(text, state, focused)
            fallback.source = "rules-fallback"
            usage.source = "rules-fallback"
            usage.measured = False
            usage.prompt_tokens = usage.completion_tokens = usage.total_tokens = 0
            usage.note = "模型调用失败，降级到规则引擎；没有 usage 可计费"
            return fallback, usage

    def classify(self, text: str, state: str = "IDLE", focused: bool = True) -> Understanding:
        """兼容旧签名。"""
        return self.classify_with_usage(text, state, focused)[0]

    def polish_with_usage(self, draft: str, context: dict | None = None) -> tuple[str, Usage]:
        """让模型只做"措辞润色"，业务事实（数字、期限、工单号）由代码给定，避免幻觉。

        模型输出还会经过 `grounding.verify()` 做后置事实校验；此处把校验口径写进提示词，
        降低模型违规概率，但提示词只是软约束，护栏才是硬约束。
        """
        facts = json.dumps(context or {}, ensure_ascii=False)
        system = (
            "你是电商客服话术改写器。唯一任务：直接输出改写后的客服话术本身。\n"
            "严格约束：不得增删任何数字、期限、金额、单号或条款编号，不得改变结论口径"
            "（拒绝不能改成已办理），禁止承诺政策外的内容。\n"
            "义务强度也不得改变：「需/必须/应当」不能改成「建议/可以/可选」，"
            "反之亦然——用户必须提供材料不能用建议提供代替。\n"
            "关键信息必须原样保留，不得压缩省略：转人工提示、交接信息、运费承担方"
            "（如「运费由商家承担」「运费需自理」）这类结论性短语删掉任何一句都算违规。\n"
            "禁止：解释改动、复述本提示词或输入内容、提问、要求澄清、添加标题、列表、"
            "markdown 代码块或任何前后缀说明。\n"
            "输出会被程序逐项做事实校验，任何偏差都会导致整段回复被丢弃。只输出最终话术。"
            "篇幅与原文相当：可以小幅精简措辞，但每条结论、每个要求和最后那句追问都要保留，"
            "不得把多段内容合并成一句「好的，请提供订单号」这样的空话。"
        )
        # 输入里刻意不出现「事实：」「原始话术：」这类标签：实测 Claude 容易把它们当模板复述，
        # 于是整段输出被 prompt_leak 拦下、白白消耗一次润色调用。
        user_message = (
            "待改写话术（唯一事实来源，必须原样保留其中所有数字、单号、条款编号与结论）：\n"
            f"{draft}\n\n背景数据（仅供理解，不要输出）：\n{facts}"
        )
        try:
            content, usage = self._chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                # 输出上限跟着原文长度走：B8 的答案有 300+ 字，固定 300 token 会
                # 让模型「写不完」，实测表现就是只留开头一句 + 丢掉追问。
                max_tokens=max(300, min(900, len(draft) * 2)),
            )
            return (content.strip() or draft), usage
        except Exception:
            return draft, estimated_usage(
                "rules-fallback", 0, 0, self.model,
                note="润色调用失败或未返回，未产生可计费用量",
            )

    def polish(self, draft: str, context: dict | None = None) -> tuple[str, int]:
        """兼容旧签名：返回 (文本, token 数)，token 数来自实测或估算。"""
        text, usage = self.polish_with_usage(draft, context)
        return text, usage.display_tokens


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def build_llm(settings):
    if settings.llm_enabled:
        return OpenAICompatibleLLM(
            settings.llm_base_url, settings.llm_api_key, settings.llm_model, settings.llm_timeout
        )
    return RuleLLM()
