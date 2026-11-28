"""多轮办理状态机 + 退货流程（第 5 周核心）。

状态：IDLE → AWAIT_ORDER → AWAIT_CONFIRM → CREATING → TICKETED，异常统一走 HUMAN。
所有状态写入 SQLite，刷新页面、重启进程都能接着办。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Settings
from ..db import Database
from ..domain.orders import REASON_LABEL, REASON_NO_REASON, REASON_QUALITY
from ..domain.rag import PolicyIndex
from ..llm import Understanding
from ..slots import normalize_order_id
from .advice import Advice, build_advice, is_followup, match_order, read_situation
from .tools import ToolError, ToolRegistry


# 认单提议的应答词：只认「明确的肯定」，避免「是不是可以退」被当成确认
_YES_WORDS = ("是", "是的", "是这单", "对", "对的", "嗯", "好", "好的", "可以", "没错", "确认", "就是这单", "就是这个")
_NO_WORDS = ("不是", "不对", "不是这单", "不是我", "不用", "换一单", "另一个")


def _is_affirmative(text: str) -> bool:
    t = (text or "").strip().strip("，。！？,.!?~～ ")
    if not t or any(word in t for word in _NO_WORDS):
        return False
    return t in _YES_WORDS


def _is_negative(text: str) -> bool:
    t = (text or "").strip()
    return any(word in t for word in ("不是", "不对", "不是我", "不是这单", "换一单"))


class State(str, Enum):
    IDLE = "IDLE"
    AWAIT_ORDER = "AWAIT_ORDER"
    AWAIT_CONFIRM = "AWAIT_CONFIRM"
    CREATING = "CREATING"
    TICKETED = "TICKETED"
    HUMAN = "HUMAN"


STATE_LABEL = {
    State.IDLE: "待接入",
    State.AWAIT_ORDER: "等待订单号",
    State.AWAIT_CONFIRM: "等待用户确认",
    State.CREATING: "正在建单",
    State.TICKETED: "已创建工单",
    State.HUMAN: "待人工处理",
}

# 允许的状态迁移，任何未列出的迁移都会被拒绝
ALLOWED: dict[State, set[State]] = {
    State.IDLE: {State.IDLE, State.AWAIT_ORDER, State.AWAIT_CONFIRM, State.TICKETED, State.HUMAN},
    State.AWAIT_ORDER: {State.AWAIT_ORDER, State.AWAIT_CONFIRM, State.TICKETED, State.HUMAN, State.IDLE},
    State.AWAIT_CONFIRM: {State.AWAIT_CONFIRM, State.AWAIT_ORDER, State.CREATING, State.TICKETED, State.IDLE, State.HUMAN},
    State.CREATING: {State.CREATING, State.TICKETED, State.AWAIT_CONFIRM, State.AWAIT_ORDER, State.IDLE, State.HUMAN},
    State.TICKETED: {State.TICKETED, State.AWAIT_ORDER, State.AWAIT_CONFIRM, State.HUMAN, State.IDLE},
    State.HUMAN: {State.HUMAN, State.IDLE, State.AWAIT_ORDER, State.AWAIT_CONFIRM, State.TICKETED},
}


class TransitionError(Exception):
    pass


@dataclass
class FlowOutput:
    reply: str
    state: State
    slots: dict = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    ticket: dict | None = None
    cards: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def public_slots(self) -> dict:
        slots = dict(self.slots)
        slots.pop("confirm_token", None)  # 确认令牌不下发前端
        return slots


class ReturnFlow:
    """对话流程编排：模型只给意图，规则与状态都在这里。"""

    def __init__(
        self, db: Database, tools: ToolRegistry, index: PolicyIndex, settings: Settings,
        on_handoff=None,
    ):
        self.db = db
        self.tools = tools
        self.index = index
        self.settings = settings
        self.on_handoff = on_handoff  # 事项 B：转人工入队回调（可选，保持旧调用兼容）

    # ---------- 会话 ----------
    def ensure_session(self, session_id: str, user_id: str) -> dict:
        session = self.db.get_session(session_id)
        if session and session["user_id"] != user_id:
            # 同一浏览器切换账号：清空上下文，避免跨账号串单
            self.db.log_event(session_id, None, "session_user_switched", {"from": session["user_id"], "to": user_id})
            session = None
        if session is None:
            return self.db.save_session(session_id, user_id, State.IDLE.value, {})
        return session

    def reset(self, session_id: str, user_id: str) -> dict:
        return self.db.save_session(session_id, user_id, State.IDLE.value, {})

    def _save(self, session: dict, to_state: State, slots: dict) -> dict:
        current = State(session["state"])
        if to_state not in ALLOWED[current]:
            raise TransitionError(f"非法状态迁移：{current.value} → {to_state.value}")
        saved = self.db.save_session(session["session_id"], session["user_id"], to_state.value, slots)
        if current != to_state:
            self.db.log_event(
                session["session_id"], None, "state_transition", {"from": current.value, "to": to_state.value}
            )
        return saved

    def _ctx(self, session: dict, slots: dict, *, confirmed: bool = False, request_id: str | None = None, attempt_no: int = 1) -> dict:
        return {
            "user_id": session["user_id"],
            "session_id": session["session_id"],
            "session": {"state": session["state"], "slots": slots},
            "confirmed": confirmed,
            "request_id": request_id,
            "attempt_no": attempt_no,
        }

    def _citations(self, clause_ids: list[str]) -> list[dict]:
        wanted = set(clause_ids)
        return [
            {"clause_id": c.clause_id, "title": c.title, "snippet": c.snippet(70)}
            for c in self.index.clauses
            if c.clause_id in wanted
        ]

    # ---------- 主入口 ----------
    def handle(self, session: dict, message: str, understanding: Understanding, request_id: str | None = None) -> FlowOutput:
        state = State(session["state"])
        slots = dict(session["slots"])
        intent = understanding.intent
        # 订单号先过格式关：模型可能把「没有单号」写成字符串 "null"，那是真值，
        # 拿它去查订单只会回一句「没有查到订单 null」（B9 实测）。
        order_id = normalize_order_id(understanding.order_id)

        # 0) 上一轮我提议过「你问的是这一单吗」：用户回「是」就按那单核单，回「不是」就撤掉提议。
        if state is State.AWAIT_ORDER and not order_id:
            pending = slots.get("pending_order_id")
            if pending and _is_affirmative(message):
                slots.pop("pending_order_id", None)
                slots.pop("pending_order_title", None)
                pending_reason = slots.pop("pending_reason", "") or ""
                # 「是」这一句本身不带理由，不能让 no_reason 盖掉上一轮说过的「坏了」：
                # 只要任一侧判过质量问题，就按质量通道核单。
                if REASON_QUALITY in (understanding.reason, pending_reason):
                    reason = REASON_QUALITY
                else:
                    reason = understanding.reason or pending_reason or slots.get("reason", REASON_NO_REASON)
                return self._begin_return(session, slots, pending, reason, request_id)
            if pending and _is_negative(message):
                slots.pop("pending_order_id", None)
                slots.pop("pending_order_title", None)
                slots.pop("pending_reason", None)
                slots.pop("pending_situation", None)
                saved = self._save(session, State.AWAIT_ORDER, slots)
                return self._remind_order(State.AWAIT_ORDER, saved["slots"], message)

        # 0.5) 指代性追问：刚讲过某一类情形，用户问「为什么不能退」——
        # 真实模型常把这类短句判成 policy，一判 policy 就重开话题、重新检索一遍条款，
        # 上一轮的上下文全丢。这里先按上文情形答，别让用户把话再说一遍。
        context_key = (slots.get("last_advice") or {}).get("key")
        if (
            context_key and context_key != "unknown"
            and state in (State.IDLE, State.AWAIT_ORDER)
            and intent in ("policy", "unknown")
            and is_followup(message)
        ):
            advice = self._advice_for(slots, message)
            saved = self._save(session, state, slots)
            return FlowOutput(
                reply=advice.text,
                state=state,
                slots=saved["slots"],
                suggestions=["我的订单号是 O20261115001", "不知道订单号", "转人工"],
                citations=self._citations(list(advice.clauses)),
                meta={"advice": advice.key, "advice_basis": advice.basis},
            )

        # 1) 等待订单号：用户回的消息基本就是订单号
        if state is State.AWAIT_ORDER and order_id and intent not in ("human", "cancel"):
            slots.pop("pending_order_id", None)
            slots.pop("pending_order_title", None)
            slots.pop("pending_reason", None)
            slots.pop("pending_situation", None)
            return self._begin_return(session, slots, order_id, understanding.reason or slots.get("reason", REASON_NO_REASON), request_id)

        # 2) 等待确认：确认 / 取消 / 换单
        if state is State.AWAIT_CONFIRM:
            if order_id and order_id != slots.get("order_id"):
                return self._begin_return(session, slots, order_id, understanding.reason or slots.get("reason", REASON_NO_REASON), request_id)
            if intent == "hesitate":
                return FlowOutput(
                    reply="好的，先不提交，这笔退货申请还留着。核对完回复「确认」我就建单；不想办了回复「取消」。",
                    state=State.AWAIT_CONFIRM,
                    slots=slots,
                    suggestions=["确认", "取消"],
                )
            if intent == "confirm":
                return self._create_ticket(session, slots, request_id)
            if intent == "cancel":
                slots.pop("confirm_token", None)
                saved = self._save(session, State.IDLE, slots)
                return FlowOutput(
                    reply="好的，这次退货申请已经取消，没有创建工单。需要再办理时告诉我就行。",
                    state=State.IDLE,
                    slots=saved["slots"],
                    suggestions=["我要退货", "查退货政策", "转人工"],
                )
            if intent == "human":
                return self._handoff(session, slots, "用户要求人工协助")
            return FlowOutput(
                reply="这笔退货申请还在等你确认。回复「确认」我就创建工单；回复「取消」可以放弃；也可以直接给我另一个订单号。",
                state=State.AWAIT_CONFIRM,
                slots=slots,
                suggestions=["确认", "取消"],
            )

        # 2.5) 建单中途的残留状态：进程崩溃后不能让用户卡死在这里
        if state is State.CREATING:
            if intent == "cancel":
                slots.pop("confirm_token", None)
                saved = self._save(session, State.AWAIT_CONFIRM, slots)
                return FlowOutput(
                    reply="已取消本次提交，没有创建工单。需要时再回复「确认」。",
                    state=State.AWAIT_CONFIRM,
                    slots=saved["slots"],
                    suggestions=["确认", "取消"],
                )
            if intent == "confirm":
                return self._create_ticket(session, slots, request_id)
            return FlowOutput(
                reply="上一次建单还没完成（可能是网络中断）。工单不会重复创建，回复「确认」继续即可拿到结果。",
                state=State.CREATING,
                slots=slots,
                suggestions=["确认", "取消"],
            )

        # 2.6) 已建单后又收到「确认」：加锁后的并发重试会走到这里，返回原工单
        if state is State.TICKETED and intent == "confirm":
            return self._already_ticketed(session, slots)

        # 2.7) 进度查询：用户明确问"到哪一步/进度/工单状态"且有工单号时，
        # 直接查库里真实状态回答（不新增意图值，避免影响严格评测口径）。
        if self._asks_progress(message) and (slots.get("ticket_id") or self.db.list_tickets(session_id=session["session_id"], user_id=session["user_id"], limit=1)):
            return self._query_progress(session, slots, message)

        # 3) 其它状态按意图路由
        if intent == "return":
            correction = (getattr(understanding, "slots", None) or {}).get("reason_correction")
            return self._begin_return(
                session, slots, order_id, understanding.reason, request_id,
                message=message, claimed_quality_dropped=bool(correction),
            )
        if intent == "order":
            if not order_id and state is State.AWAIT_ORDER:
                return self._remind_order(state, slots)
            return self._show_order(session, slots, order_id or slots.get("order_id"), request_id)
        if intent == "logistics":
            return self._show_logistics(session, slots, order_id or slots.get("order_id"), request_id)
        if intent == "policy":
            return self._answer_policy(session, slots, message, understanding)
        if intent == "human":
            return self._handoff(session, slots, message)
        if intent == "cancel":
            # B11：「键盘还没发出去，能不能撤掉」是**撤订单**，不是「算了不办了」。
            # 两者在规则里共用一个 cancel 意图，这里按「是否带单号」分流：
            # 带单号且确实未发货 → 按 R-04 说明可直接取消订单、全额退款；
            # 不带单号 / 已发货 → 保持原来的「取消本次操作」语义，不误引导。
            if order_id and state in (State.IDLE, State.AWAIT_ORDER):
                return self._cancel_order_guide(session, slots, order_id)
            return FlowOutput(
                reply="好的，已取消本次操作。需要办理时随时叫我。",
                state=state,
                slots=slots,
                suggestions=["我要退货", "退货政策是几天"],
            )
        if intent == "greeting":
            return FlowOutput(
                reply="你好，我是售后客服小奈。你可以直接说需求，比如「我要退货」「查物流」「退货政策是几天」。",
                state=state,
                slots=slots,
                suggestions=["我要退货", "查物流", "退货政策是几天"],
            )
        if state is State.AWAIT_ORDER:
            return self._remind_order(state, slots, message)
        return self._fallback(session, slots, message)

    # ---------- 退货：核单 + 资格判定 ----------
    def _clause_hits(self, message: str) -> list:
        """按用户原话检索条款，供「情形不明」时引用真实条款作答。"""
        try:
            return [
                (clause.clause_id, clause.title, clause.salient(message, 70))
                for clause, _ in self.index.search(message, top_k=2)
            ]
        except Exception:
            return []

    def _advice_for(
        self, slots: dict, message: str, *, reason: str = "",
        claimed_quality_dropped: bool = False, store: bool = True,
    ) -> "Advice":
        """生成答案，并把「这一轮讲的是哪一类」记进上下文，供下一句追问续答。

        没有上下文承接时，「为什么不能退」会被当成一个全新的问题，重新甩一遍
        通用话术——用户会觉得自己刚才那几句话白说了。
        """
        context = slots.get("last_advice") or {}
        hits = self._clause_hits(message) if read_situation(message) == "unknown" else ()
        advice = build_advice(
            message, reason=reason, claimed_quality_dropped=claimed_quality_dropped,
            context_key=context.get("key", ""), context_evidence=context.get("evidence", ""),
            clause_hits=hits,
        )
        if store:
            slots["last_advice"] = {"key": advice.key, "evidence": advice.evidence}
        return advice

    def _answer_then_ask(
        self, session: dict, slots: dict, message: str, *,
        reason: str = "", claimed_quality_dropped: bool = False,
    ) -> tuple[str, "Advice"]:
        """回答用户真正问的那件事，然后要么认出订单请确认，要么请他给单号。

        认单只做到「提议」这一层：`match_order` 命中时把候选写进
        `slots.pending_order_id`，等用户回一句「是」再核单。直接按猜的订单
        办退货，一旦认错单就是替用户把别的东西退了——认错单的代价远高于多问一句。
        """
        advice = self._advice_for(
            slots, message, reason=reason, claimed_quality_dropped=claimed_quality_dropped
        )
        match = match_order(self.tools.orders.values(), session["user_id"], message) if message else None
        if match is None:
            slots.pop("pending_order_id", None)
            slots.pop("pending_order_title", None)
            return advice.text, advice
        slots["pending_order_id"] = match.order.order_id
        # B12：提议话术里带着商品名，润色层要把它当结构性事实保住。
        slots["pending_order_title"] = match.order.title
        # 「坏了 / 刻歪了 / 洗过」这些事实在用户回「是」的那一句里不会再出现，
        # 必须连同提议一起存下来，否则确认后会按无理由口径重算，把质量通道丢掉。
        slots["pending_reason"] = reason or ""
        slots["pending_situation"] = advice.key
        ask = (
            f"你话里的「{match.token}」看着像订单 {match.order.order_id}（{match.order.title}）——"
            "是这一单的话回「是」，我直接按这单核期限；不是你问的那单，把正确订单号发我。"
        )
        return f"{advice.body}\n\n{ask}", advice

    def _begin_return(
        self, session: dict, slots: dict, order_id: str | None, reason: str,
        request_id: str | None, message: str = "", claimed_quality_dropped: bool = False,
    ) -> FlowOutput:
        if not order_id:
            # 没有单号：先把「他问的这件事按规则怎么算」答清楚，再要么认单、要么要单。
            reply, advice = self._answer_then_ask(
                session, slots, message, reason=reason,
                claimed_quality_dropped=claimed_quality_dropped,
            )
            saved = self._save(session, State.AWAIT_ORDER, slots)
            return FlowOutput(
                reply=reply,
                state=State.AWAIT_ORDER,
                slots=saved["slots"],
                suggestions=["我的订单号是 O20261115001", "不知道订单号", "转人工"],
                citations=self._citations(list(advice.clauses)),
                meta={"advice": advice.key, "advice_basis": advice.basis, "order_match": slots.get("pending_order_id")},
            )

        ctx = self._ctx(session, slots, request_id=request_id)
        try:
            result = self.tools.call(
                "check_return_eligibility", {"order_id": order_id, "reason": reason or REASON_NO_REASON}, ctx
            )
        except ToolError as exc:
            return self._tool_error_reply(session, slots, exc, order_id)

        order = result["order"]
        eligibility = result["eligibility"]
        previous_token = slots.get("confirm_token")
        previous_order = slots.get("order_id")

        slots.update(
            {
                "order_id": order["order_id"],
                "reason": eligibility["reason"],
                "eligibility": eligibility,
                "order": order,
            }
        )
        citations = self._citations(eligibility.get("citations", []))

        if not eligibility["eligible"]:
            slots.pop("confirm_token", None)
            if previous_token and previous_order and previous_order != order["order_id"]:
                self.db.log_event(
                    session["session_id"], request_id, "confirm_invalidated",
                    {"order_id": previous_order, "new_order_id": order["order_id"]},
                )
            target = State.HUMAN if eligibility["requires_human"] else State.IDLE
            saved = self._save(session, target, slots)
            reply = f"这笔订单暂时不能直接办理退货：\n\n【{eligibility['title']}】{eligibility['detail']}"
            if eligibility["suggestion"]:
                reply += f"\n\n建议：{eligibility['suggestion']}。"
            if target is State.HUMAN:
                reply += "\n\n已经帮你标记为待人工处理，你把情况补充给人工客服即可，不需要重复描述。"
            return FlowOutput(
                reply=reply,
                state=target,
                slots=saved["slots"],
                suggestions=["转人工", "查退货政策"] if target is State.IDLE else ["人工什么时候联系我", "换一个订单号"],
                citations=citations,
                cards=[{"type": "order", "data": order}],
            )

        # 可退 → 下发一次性确认令牌，进入等待确认
        token = uuid.uuid4().hex
        slots["confirm_token"] = token
        slots["confirm_issued_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        slots.pop("create_attempts", None)
        if previous_token and previous_order and previous_order != order["order_id"]:
            self.db.log_event(
                session["session_id"], request_id, "confirm_invalidated",
                {"order_id": previous_order, "new_order_id": order["order_id"]},
            )
        saved = self._save(session, State.AWAIT_CONFIRM, slots)

        remaining = eligibility["remaining_days"]
        deadline = eligibility["deadline"]
        deadline_text = f"还剩 {remaining} 天（截止 {deadline}）" if remaining and remaining > 0 else f"今天到期（截止 {deadline}）"
        summary = (
            f"查到订单 {order['order_id']} · {order['title']}\n"
            f"金额 ¥{order['amount']:.2f} ｜ {order['status_label']}（{order['signed_label']}）\n"
            f"判定：{eligibility['title']}，{deadline_text}\n"
            f"取件：上门取件 9:00-20:00，{'运费由商家承担' if eligibility['reason'] == REASON_QUALITY else '运费需自理'}\n"
            f"依据：{'、'.join(eligibility.get('citations', [])) or '平台售后政策'}"
        )
        # 系统只有订单数据，不知道商品现在什么状态：用户自己说「洗过 / 拆封」时，
        # 不能因为期限还算得出来就默认能过审——补一句风险提示，别让承诺走在审核前面。
        situation = read_situation(message) if (message or "").strip() else slots.get("pending_situation")
        if eligibility["reason"] == REASON_NO_REASON and situation == "used":
            summary += "\n注意：你提到已经拆封 / 洗过，无理由退货要求商品完好、不影响二次销售，审核环节可能被驳回。"
        return FlowOutput(
            reply=summary + "\n\n确认无误请回复「确认」，我会创建售后工单；回复「取消」放弃本次申请。",
            state=State.AWAIT_CONFIRM,
            slots=saved["slots"],
            suggestions=["确认", "取消", "换成另一个订单"],
            citations=citations,
            cards=[{"type": "order", "data": order}, {"type": "eligibility", "data": eligibility}],
        )

    # ---------- 建单（幂等） ----------
    def _create_ticket(self, session: dict, slots: dict, request_id: str | None) -> FlowOutput:
        token = slots.get("confirm_token")
        order_id = slots.get("order_id")
        if not token or not order_id:
            saved = self._save(session, State.AWAIT_ORDER, slots)
            return FlowOutput(
                reply="确认信息已经失效了，请再把订单号发我一次，我重新帮你核单。",
                state=State.AWAIT_ORDER,
                slots=saved["slots"],
            )

        # 确认令牌有时效：隔很久再回来确认，资格/期限可能已经变了
        if self._confirm_expired(slots.get("confirm_issued_at")):
            slots.pop("confirm_token", None)
            slots.pop("create_attempts", None)
            saved = self._save(session, State.AWAIT_ORDER, slots)
            self.db.log_event(
                session["session_id"], request_id, "confirm_expired", {"order_id": order_id}
            )
            minutes = self.settings.confirm_token_ttl_seconds // 60
            return FlowOutput(
                reply=f"这次确认已经超过 {minutes} 分钟有效期。为保证资格判定准确，请重新发一次订单号让我核单，没有创建工单。",
                state=State.AWAIT_ORDER,
                slots=saved["slots"],
                suggestions=["我的订单号是 O20261115001", "转人工"],
                meta={"error": "confirm_expired"},
            )

        attempt_no = int(slots.get("create_attempts", 0)) + 1
        slots["create_attempts"] = attempt_no
        creating = self._save(session, State.CREATING, slots)

        # 建单前重新复核资格：核单到确认之间订单状态/日期都可能变化
        try:
            recheck = self.tools.call(
                "check_return_eligibility",
                {"order_id": order_id, "reason": slots.get("reason", REASON_NO_REASON)},
                self._ctx(creating, slots, request_id=request_id),
            )
        except ToolError as exc:
            back = self._save(creating, State.AWAIT_CONFIRM, slots)
            return FlowOutput(
                reply=f"确认时复核订单失败：{exc.message}。可以稍后重试，或转人工处理。",
                state=State.AWAIT_CONFIRM,
                slots=back["slots"],
                suggestions=["确认", "转人工"],
                meta={"error": exc.code},
            )
        eligibility = recheck["eligibility"]
        if not eligibility["eligible"]:
            slots.pop("confirm_token", None)
            slots.pop("create_attempts", None)
            target = State.HUMAN if eligibility.get("requires_human") else State.AWAIT_ORDER
            saved = self._save(creating, target, slots)
            self.db.log_event(
                session["session_id"], request_id, "eligibility_changed",
                {"order_id": order_id, "code": eligibility["code"]},
            )
            return FlowOutput(
                reply=f"确认时重新核对了资格，情况有变化：{eligibility['title']}。{eligibility['detail']}\n\n没有创建工单。",
                state=target,
                slots=saved["slots"],
                suggestions=["转人工", "查退货政策"],
                citations=self._citations(eligibility.get("citations", [])),
                meta={"error": "eligibility_changed"},
            )

        ctx = self._ctx(creating, slots, confirmed=True, request_id=request_id, attempt_no=attempt_no)

        try:
            result = self.tools.call(
                "create_return_ticket",
                {"order_id": order_id, "confirm_token": token, "reason": slots.get("reason", REASON_NO_REASON)},
                ctx,
            )
        except ToolError as exc:
            if exc.code in ("response_lost", "timeout"):
                back = self._save(creating, State.AWAIT_CONFIRM, slots)
                return FlowOutput(
                    reply="建单请求超时了，工单可能已经生成。请再回复一次「确认」重试——重试不会重复建单，只会返回同一个工单号。",
                    state=State.AWAIT_CONFIRM,
                    slots=back["slots"],
                    suggestions=["确认"],
                    meta={"retryable": True, "error": exc.code},
                )
            back = self._save(creating, State.AWAIT_CONFIRM, slots)
            return FlowOutput(
                reply=f"建单没有成功：{exc.message}。你可以回复「确认」重试，或转人工处理。",
                state=State.AWAIT_CONFIRM,
                slots=back["slots"],
                suggestions=["确认", "转人工"],
                meta={"error": exc.code},
            )

        ticket = result["ticket"]
        created = result["created"]
        slots["ticket_id"] = ticket["ticket_id"]
        slots.pop("confirm_token", None)
        slots.pop("create_attempts", None)
        saved = self._save(creating, State.TICKETED, slots)

        if created:
            reply = (
                f"售后工单已创建：{ticket['ticket_id']}\n"
                f"订单 {ticket['order_id']} ｜ {REASON_LABEL.get(ticket['reason'], ticket['reason'])}\n\n"
                "接下来快递员会在 1 个工作日内联系取件；仓库签收质检后 1-3 个工作日退款，按原路退回。"
            )
        else:
            reply = (
                f"这笔申请之前已经提交过了，工单号是 {ticket['ticket_id']}，没有重复建单。\n"
                "同一订单的重复确认只会返回原工单，可以放心。"
            )
        return FlowOutput(
            reply=reply,
            state=State.TICKETED,
            slots=saved["slots"],
            suggestions=["查物流", "退货运费谁承担", "再退一个订单"],
            citations=self._citations(["R-05", "R-06", "R-07"]),
            ticket=ticket,
            cards=[{"type": "ticket", "data": ticket}],
            meta={"created": created, "attempt_no": attempt_no},
        )

    def _confirm_expired(self, issued_at: str | None) -> bool:
        """确认令牌有效期；缺少或格式异常的时间按过期处理（保守）。"""
        if not issued_at:
            return True
        try:
            issued = datetime.strptime(issued_at, "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            return True
        return (datetime.now() - issued).total_seconds() > self.settings.confirm_token_ttl_seconds

    def _already_ticketed(self, session: dict, slots: dict) -> FlowOutput:
        tickets = self.db.list_tickets(session["session_id"])
        ticket = tickets[0] if tickets else None
        ticket_id = (ticket or {}).get("ticket_id") or slots.get("ticket_id") or "（见工单列表）"
        return FlowOutput(
            reply=f"这笔申请已经提交过了，工单号是 {ticket_id}，没有重复建单。",
            state=State.TICKETED,
            slots=slots,
            suggestions=["查物流", "退货运费谁承担", "再退一个订单"],
            ticket=ticket,
            cards=[{"type": "ticket", "data": ticket}] if ticket else [],
            meta={"created": False, "duplicate_confirm": True},
        )

    # ---------- 撤单（未发货取消订单）----------
    def _cancel_order_guide(self, session: dict, slots: dict, order_id: str) -> FlowOutput:
        """带单号的「撤掉 / 撤单」按 R-04 给出可执行路径，不创建退货工单。

        这里不替用户真正撤单：系统没有取消订单的工具，承诺「已取消」就是越权。
        只说明规则与入口，状态保持 IDLE，避免误进退货流程。
        """
        state = State(session["state"])
        try:
            order = self.tools.call(
                "query_order", {"order_id": order_id}, self._ctx(session, slots)
            )["order"]
        except ToolError as exc:
            return self._tool_error_reply(session, slots, exc, order_id)
        if order["status"] != "paid":
            saved = self._save(session, State.IDLE, slots)
            return FlowOutput(
                reply=(
                    f"订单 {order_id} 目前是{order['status_label']}，不能按未发货直接取消订单。"
                    "如果还没签收，可以拒收后申请退款；已经签收的话我可以按七天无理由帮你核期限。"
                ),
                state=State.IDLE,
                slots=saved["slots"],
                suggestions=["我要退货", "转人工"],
                citations=self._citations(["R-04", "R-01"]),
                meta={"cancel_guide": "not_unshipped"},
            )
        saved = self._save(session, State.IDLE, slots)
        return FlowOutput(
            reply=(
                f"订单 {order_id}（{order['title']}）还没发货，按平台规则可以直接取消订单、全额退款，"
                "不需要走七天无理由退货。到「我的 - 售后服务」选取消订单即可；"
                "我这边不能直接替你撤单，需要协助可以回复「转人工」。"
            ),
            state=State.IDLE,
            slots=saved["slots"],
            suggestions=["转人工", "查退货政策"],
            citations=self._citations(["R-04"]),
            meta={"cancel_guide": "unshipped"},
        )

    # ---------- 查询类 ----------
    def _show_order(self, session: dict, slots: dict, order_id: str | None, request_id: str | None) -> FlowOutput:
        if not order_id:
            return FlowOutput(
                reply="把订单号发我就能查，例如 O20261115001。也可以直接说「我要退货」，我按流程带你走。",
                state=State(session["state"]),
                slots=slots,
                suggestions=["我要退货", "退货政策是几天"],
            )
        try:
            order = self.tools.call("query_order", {"order_id": order_id}, self._ctx(session, slots, request_id=request_id))["order"]
        except ToolError as exc:
            return self._tool_error_reply(session, slots, exc, order_id)

        slots["order_id"] = order["order_id"]
        slots["order"] = order
        saved = self._save(session, State(session["state"]), slots)
        reply = (
            f"订单 {order['order_id']}\n"
            f"{order['title']} ｜ ¥{order['amount']:.2f} ｜ {order['status_label']}\n"
            f"签收时间：{order['signed_label']}\n"
            f"收货地址：{order['address']}\n\n"
            "如果商品有问题，回复「退货」我可以直接帮你判断资格并建单。"
        )
        return FlowOutput(
            reply=reply,
            state=State(saved["state"]),
            slots=saved["slots"],
            suggestions=["我要退货", "查物流"],
            cards=[{"type": "order", "data": order}],
        )

    def _show_logistics(self, session: dict, slots: dict, order_id: str | None, request_id: str | None) -> FlowOutput:
        if not order_id:
            return FlowOutput(
                reply="告诉我订单号，我帮你查包裹到哪了。",
                state=State(session["state"]),
                slots=slots,
                suggestions=["我的订单号是 O20261115001"],
            )
        try:
            info = self.tools.call("query_logistics", {"order_id": order_id}, self._ctx(session, slots, request_id=request_id))
        except ToolError as exc:
            if exc.code == "timeout":
                return FlowOutput(
                    reply="物流服务暂时超时，没能取到最新轨迹。你可以稍后再问一次，或者回复「转人工」让我把单号带给人工客服。",
                    state=State(session["state"]),
                    slots=slots,
                    suggestions=["再查一次", "转人工"],
                    meta={"degraded": True, "error": exc.code},
                )
            return self._tool_error_reply(session, slots, exc, order_id)

        nodes = info.get("nodes") or []
        lines = "\n".join(f"· {node.get('time', '')} {node.get('text', '')}" for node in nodes)
        reply = (
            f"订单 {info['order_id']} 物流状态：{info.get('status') or '暂无'}\n"
            f"{info.get('company', '')} {info.get('tracking_no', '')}\n"
            + (lines if lines else info.get("message", "暂无轨迹"))
        )
        return FlowOutput(
            reply=reply,
            state=State(session["state"]),
            slots=slots,
            suggestions=["我要退货", "退货政策是几天"],
            cards=[{"type": "logistics", "data": info}],
        )

    def _answer_policy(
        self, session: dict, slots: dict, message: str, understanding=None
    ) -> FlowOutput:
        """条款检索结果由上游「政策 Agent」真实调用工具后传入，下游直接消费。

        没有预置结果时（单 Agent / 自定义调用）仍走本地检索，行为与旧版一致。
        """
        preloaded = (getattr(understanding, "slots", None) or {}).get("policy_answer")
        answer = preloaded if isinstance(preloaded, dict) else self.index.answer(message, top_k=2)
        state = State(session["state"])
        if answer["need_human"]:
            saved = self._save(session, State.HUMAN, slots)
            return FlowOutput(
                reply=answer["text"] + "\n\n已经帮你转人工，订单信息会一并带上。",
                state=State.HUMAN,
                slots=saved["slots"],
                suggestions=["转人工", "我要退货"],
                citations=[],
            )
        return FlowOutput(
            reply=answer["text"] + "\n\n需要的话我可以直接按这条规则帮你办理退货。",
            state=state,
            slots=slots,
            suggestions=["我要退货", "转人工"],
            citations=answer["citations"],
        )

    # ---------- 转人工与兜底 ----------
    def _handoff(self, session: dict, slots: dict, reason: str) -> FlowOutput:
        saved = self._save(session, State.HUMAN, slots)
        order_hint = f"订单 {slots['order_id']} " if slots.get("order_id") else "还没提供订单号，"
        self.db.log_event(
            session["session_id"], None, "handoff", {"reason": reason, "order_id": slots.get("order_id")}
        )
        # 事项 B：真正入队（打包上下文、SLA、工单号）
        if self.on_handoff is not None:
            try:
                recent = [
                    {"kind": e["kind"], "payload": e["payload"]}
                    for e in self.db.list_events(session["session_id"], limit=6)
                ]
                self.on_handoff(session, slots, reason, recent)
            except Exception:
                # 入队失败不能阻断用户侧"已转人工"的响应
                pass
        return FlowOutput(
            reply=(
                "已经转人工客服，预计 1-2 分钟内接入。\n"
                f"交接信息：{order_hint}问题描述「{reason[:40]}」，你不用再重复说一遍。"
            ),
            state=State.HUMAN,
            slots=saved["slots"],
            suggestions=["我要退货", "查退货政策"],
        )

    def _remind_order(self, state: State, slots: dict, message: str = "") -> FlowOutput:
        text = (message or "").strip()
        if any(word in text for word in ("不知道", "找不到", "没有订单", "没订单")):
            reply = (
                "没关系，订单号可以这样找：我的 → 全部订单 → 点开订单详情，页面顶部的「订单编号」就是（以 O 开头的一串数字）。\n"
                "找到后直接发给我，我接着帮你办。"
            )
            return FlowOutput(
                reply=reply,
                state=state,
                slots=slots,
                suggestions=["我的订单号是 O20261115001", "转人工"],
            )
        # 在等单号时用户又问了别的问题：先把问题答了，再提醒单号——
        # 只说「还在等你发订单号」等于把他刚说的话丢掉。
        advice = self._advice_for(slots, text, store=False) if text else None
        if advice is not None and advice.informative:
            return FlowOutput(
                reply=advice.text,
                state=state,
                slots=slots,
                suggestions=["我的订单号是 O20261115001", "不知道订单号", "转人工"],
                citations=self._citations(list(advice.clauses)),
                meta={"advice": advice.key, "advice_basis": advice.basis},
            )
        return FlowOutput(
            reply=(
                "这句我在现行政策条款里没找到对应内容，不猜——需要确认的话回复「转人工」。\n"
                "另外订单号还没收到，发我就能接着办（在「我的订单」里复制，例如 O20261115001）。"
            ),
            state=state,
            slots=slots,
            suggestions=["我的订单号是 O20261115001", "转人工"],
        )

    _PROGRESS_WORDS = ("进度", "到哪一步", "到哪了", "什么状态", "退款到", "审核到", "工单状态", "办得怎么样", "处理到")

    def _asks_progress(self, message: str) -> bool:
        text = (message or "").strip()
        return any(word in text for word in self._PROGRESS_WORDS)

    def _query_progress(self, session: dict, slots: dict, message: str) -> FlowOutput:
        """只引用库里的真实状态与时间戳，不编造金额、日期或到账时间。"""
        try:
            result = self.tools.call(
                "query_ticket_status", {}, self._ctx(session, slots)
            )
            t = result["ticket"]
            reply = (
                f"工单 {t['ticket_id']} 当前状态：{t['status_label']}。\n"
                f"最近更新：{t['updated_at']}。\n"
                f"下一步：{t['next_step']}"
            )
            cards = [{"type": "ticket", "data": t}]
        except ToolError:
            reply = "目前还没查到你的售后工单。办理完成后就能在这里查进度，也可以回复「转人工」让人工帮你确认。"
            cards = []
        return FlowOutput(
            reply=reply,
            state=State(session["state"]),
            slots=slots,
            suggestions=["我要退货", "转人工"],
            cards=cards,
        )

    def _fallback(self, session: dict, slots: dict, message: str) -> FlowOutput:
        """兜底也要「按实际问题回答」：情形认得出就先给结论，别甩一段固定话。

        B8 之前这里固定回「这句话我还没把握处理准确」，用户描述得很具体
        （「蓝莓坏了一半」）也只得到这句——他说的信息一点没用上。
        """
        state = State(session["state"])
        advice = self._advice_for(slots, message) if (message or "").strip() else None
        if advice is not None and advice.informative:
            # 描述的是缺陷（含「定制做坏了」）→ 直接走质量通道核单；
            # 认不出是哪一单时 _begin_return 会退化成「规则说明 + 要单」。
            if advice.key in ("defect", "custom_defect") and state in (State.IDLE, State.AWAIT_ORDER):
                return self._begin_return(
                    session, slots, None, REASON_QUALITY, None, message=message
                )
            saved = self._save(session, state, slots)  # last_advice 要落库，下一句追问才接得上
            return FlowOutput(
                reply=advice.text,
                state=state,
                slots=saved["slots"],
                suggestions=["我的订单号是 O20261115001", "不知道订单号", "转人工"],
                citations=self._citations(list(advice.clauses)),
                meta={"advice": advice.key, "advice_basis": advice.basis},
            )
        return FlowOutput(
            reply=(
                "这句话我还没把握处理准确。我可以帮你：查订单、查物流、判断退货资格、创建售后工单、解释退货政策。\n"
                "也可以直接回复「转人工」。"
            ),
            state=state,
            slots=slots,
            suggestions=["我要退货", "查物流", "退货政策是几天", "转人工"],
        )

    def _tool_error_reply(self, session: dict, slots: dict, exc: ToolError, order_id: str | None) -> FlowOutput:
        state = State(session["state"])
        if exc.code == "not_found":
            saved = self._save(session, State.AWAIT_ORDER, slots)
            return FlowOutput(
                reply=f"没有查到订单 {order_id}。请确认订单号是否完整（一般以 O 开头），或者回复「转人工」让人工帮你查。",
                state=State.AWAIT_ORDER,
                slots=saved["slots"],
                suggestions=["我的订单号是 O20261115001", "转人工"],
            )
        if exc.code == "forbidden":
            self.db.log_event(
                session["session_id"], None, "security_violation",
                {"type": "order_ownership", "order_id": order_id, "user_id": session["user_id"]},
            )
            saved = self._save(session, State.AWAIT_ORDER, slots)
            return FlowOutput(
                reply="这个订单不在当前登录账号下，我没有权限查看或办理。请切换到下单账号，或提供本人名下的订单号。",
                state=State.AWAIT_ORDER,
                slots=saved["slots"],
                suggestions=["我的订单号是 O20261115001", "转人工"],
                meta={"denied": True},
            )
        if exc.code == "timeout":
            return FlowOutput(
                reply="后台服务有点慢，这次没查成。可以再发一次订单号，或回复「转人工」。",
                state=state,
                slots=slots,
                suggestions=["再试一次", "转人工"],
                meta={"degraded": True},
            )
        return FlowOutput(
            reply=f"办理过程中出了点问题：{exc.message}。可以重试或转人工。",
            state=state,
            slots=slots,
            suggestions=["转人工"],
        )
