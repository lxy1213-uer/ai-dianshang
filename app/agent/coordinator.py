"""Agent 编排：Supervisor + 三个专职 Agent（第 7 周），同时保留单 Agent 模式用于对比。

真实编排（不是编排模拟）
------------------------
每个 Agent 步骤都是一次真实函数调用，耗时用 perf_counter 实测后写入 trace；
下游 Agent 真实驱动自己的工具链 / 状态机，工具调用会回填到该步骤的 detail.tools；
Supervisor 把上游槽位打包成 handoff 交给下游，下游消费它（政策 Agent 的条款检索
结果会直接成为状态机的输入，不重复检索）。

用量口径
--------
模型 token 只从 /chat/completions 的 usage 字段采集（measured）；规则模式、
降级、无 usage 响应都只有 prompt 字符估算（estimated），不计费。
详见 app/usage.py 与 README 第 7 节。

单 Agent 路径：一个通用 Agent 同时做意图识别与办理，使用大一统提示词，没有转交环节。
两条路径共用同一套工具与状态机，差异只在"理解"这一层，便于公平对比成功率、耗时与用量。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

from ..config import Settings
from ..db import Database
from ..domain.orders import load_logistics
from ..domain.order_sources import OrderSourceError, build_order_source
from ..domain.rag import build_index, complete_citations
from ..domain.tickets import TicketService
from ..grounding import verify
from ..llm import OpenAICompatibleLLM, build_llm, estimate_tokens
from ..model_registry import ModelProfile, build_registry
from ..usage import Usage, build_turn_usage, estimated_usage, summarize_turns
from .copilot import DeskCopilot
from .diagnosis import AftersaleDiagnosisService
from .flow import ReturnFlow, State
from .tools import ToolRegistry

# 每个 Agent 的提示词骨架：规则模式下用于估算 prompt 规模，LLM 模式下作为 system prompt 口径
AGENT_PROMPTS = {
    "接待 Agent": (
        "你是电商售后客服的接待 Agent。识别用户意图（退货/政策/物流/订单/转人工），"
        "抽取订单号与退货原因，交给下游 Agent 办理。不要直接回答政策细节，也不要承诺办理结果。"
    ),
    "政策 Agent": (
        "你是售后政策 Agent。只能依据检索到的政策条款回答，必须给出条款编号；"
        "没有可引用条款时明确说明并建议转人工，不得自行编造规则。"
    ),
    "售后 Agent": (
        "你是售后办理 Agent。按状态机推进退货流程：核单 → 判资格 → 展示确认摘要 → 建单。"
        "建单必须由用户在服务端确认后触发，重复请求返回原工单。"
    ),
    "通用 Agent": (
        "你是电商客服，负责查订单、答政策、办退货、查物流、转人工。"
        "自行判断用户意图并完成办理，注意订单号、期限与用户确认，不要重复建单。"
    ),
}

ROUTE_MAP = {
    "return": "售后 Agent",
    "order": "售后 Agent",
    "logistics": "售后 Agent",
    "policy": "政策 Agent",
    "confirm": "售后 Agent",
    "hesitate": "售后 Agent",
    "cancel": "售后 Agent",
    "human": "转人工",
    "greeting": "接待 Agent",
    "unknown": "接待 Agent",
}


@dataclass
class ChatResult:
    reply: str
    state: str
    state_label: str
    slots: dict = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    ticket: dict | None = None
    cards: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    request_id: str = ""
    agent_mode: str = "multi"
    agent_path: list[str] = field(default_factory=list)
    latency_ms: int = 0
    tokens: int = 0                       # 兼容旧字段：估算 token 之和（不代表账单）
    usage: dict = field(default_factory=dict)   # 真实计费口径，见 app/usage.py
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "reply": self.reply,
            "state": self.state,
            "state_label": self.state_label,
            "slots": self.slots,
            "trace": self.trace,
            "citations": self.citations,
            "ticket": self.ticket,
            "cards": self.cards,
            "suggestions": self.suggestions,
            "request_id": self.request_id,
            "agent_mode": self.agent_mode,
            "agent_path": self.agent_path,
            "latency_ms": self.latency_ms,
            "tokens": self.tokens,
            "usage": self.usage,
            "meta": self.meta,
        }


class Coordinator:
    def __init__(self, settings: Settings):
        self.settings = settings
        # 线程局部：并发的不同会话各自记录当前 Agent 步骤，工具回调不会串账
        self._local = threading.local()
        # 订单源可插拔：默认仍是 orders.json 零配置；切 csv/http 后后端代码不用改。
        self.order_source = build_order_source(settings)
        self.orders = self.order_source.load()
        self.logistics = load_logistics(settings.logistics_path)
        # 架构进阶：索引工厂按需注入重排器；默认 off 时与课程基线完全一致。
        from ..rerank import build_reranker
        self.index = build_index(settings.policy_path, reranker=build_reranker(settings))
        self.db = Database(settings.db_path)
        self.db.recover_creating()
        # 会话级锁：把「读 session → 状态机 → 写 session」整轮串行化。
        # 可插拔后端：inprocess（默认，单进程）/ db（跨进程 DB 租约锁）。
        from ..locking import build_session_locker
        self._locker = build_session_locker(settings)
        # 主模型负责表达层（润色）与单 Agent 路径；保留 self.llm 名字兼容现有调用/测试。
        self.llm = build_llm(settings)
        # 理解层专用模型（可选）：意图识别只输出 JSON，可用免费模型降低成本。
        # 未配置时标志为 False，_understand 跟随 self.llm（测试替换 c.llm 依旧生效）。
        self._classify_routed = settings.classify_model_enabled
        if self._classify_routed:
            self.llm_classify = OpenAICompatibleLLM(
                settings.classify_base_url,
                settings.classify_api_key,
                settings.classify_model,
                settings.classify_timeout,
            )
        else:
            self.llm_classify = self.llm

        # 可切换模型注册表：把两套配置抽象成选项，任一层可运行时切到任一个模型。
        # 读写 active 用锁保护，避免切换与对话并发时拿到半更新状态。
        self._registry = build_registry(settings)
        self._model_guard = threading.RLock()
        self._active_main = settings.llm_model if self._registry else ""
        self._active_classify = (
            settings.classify_model if self._classify_routed else self._active_main
        )

        self.tickets = TicketService(self.db, settings)
        # 架构进阶：可插拔退款网关 + 幂等编排（默认沙箱，不向外发请求）
        from ..refund import RefundService, build_refund_gateway
        self.refunds = RefundService(self.db, self.tickets, build_refund_gateway(settings))
        self.tools = ToolRegistry(
            self.orders, self.logistics, self.index, settings, self.tickets,
            db=self.db, on_call=self._on_tool_call, order_source=self.order_source,
        )
        # 事项 B：人工坐席工作台，转人工时把会话上下文打包入队
        from ..domain.agent_desk import AgentDesk
        from ..domain.funnel import FunnelService
        from ..notify import build_notifier
        self.notifier = build_notifier(settings)
        self.desk = AgentDesk(self.db, notifier=self.notifier)
        self.funnel = FunnelService(self.db)
        self.flow = ReturnFlow(
            self.db, self.tools, self.index, settings,
            on_handoff=self._enqueue_human,
        )
        # 售后诊断产品层：只读诊断 + 显式确认执行，复用同一套核单与状态机
        self.diagnosis = AftersaleDiagnosisService(self)
        # 坐席 Copilot：只读建议，复用同一套核单与条款口径
        self.copilot = DeskCopilot(self)
        # 决策审计驾驶舱：把诊断/执行/工单/退款/护栏事件串成可审计的决策链
        from ..domain.audit import AuditService
        self.audit = AuditService(self.db)
        # 报告导出：只读渲染诊断报告与审计决策链（HTML 可打印 / xlsx 零依赖）
        from ..domain.export import ReportExporter
        self.exports = ReportExporter(self.diagnosis, self.audit)

    def _enqueue_human(self, session: dict, slots: dict, reason: str, recent_turns: list[dict]) -> None:
        self.desk.enqueue(session, slots, reason, recent_turns=recent_turns)

    # ---------- 订单源：只读视图 ----------
    def ensure_orders_available(self) -> None:
        """外部订单源故障时明确报不可用，绝不伪装成「查不到这个订单」。"""
        error = getattr(self.order_source, "error", "")
        if error:
            raise OrderSourceError(error)

    def list_my_orders(self, user_id: str, limit: int = 50, status: str | None = None) -> list[dict]:
        self.ensure_orders_available()
        rows = [order for order in self.orders.values() if order.user_id == user_id]
        if status:
            rows = [order for order in rows if order.status == status]
        rows.sort(
            key=lambda order: (
                order.signed_at.isoformat() if order.signed_at else "",
                order.order_id,
            ),
            reverse=True,
        )
        return [order.to_public() for order in rows[: max(1, min(int(limit or 50), 50))]]

    # ---------- 对外入口 ----------
    def chat(self, session_id: str, user_id: str, message: str) -> ChatResult:
        """整轮对话按 session_id 串行化，避免并发请求互相覆盖会话状态。"""
        with self._session_lock(session_id):
            return self._chat_locked(session_id, user_id, message)

    @contextmanager
    def _session_lock(self, session_id: str):
        """兼容旧入口：整轮串行化统一委托给可插拔锁后端。"""
        with self._locker.lock(session_id):
            yield

    # ---------- 真实编排的观测基元 ----------
    def _on_tool_call(self, name: str, args: dict, ok: bool, ms: float, error_code: str | None) -> None:
        """工具层回调：把真实工具调用挂到当前正在执行的 Agent 步骤上。"""
        step = getattr(self._local, "active_step", None)
        if step is None:
            return
        entry: dict = {"tool": name, "ok": ok, "ms": ms}
        if error_code:
            entry["error"] = error_code
        step["detail"].setdefault("tools", []).append(entry)

    @contextmanager
    def _agent_step(self, trace: list[dict], agent: str, action: str, detail: dict):
        """把一次真实的 Agent 执行包成 trace 步骤：实测耗时 + 用量 + 工具调用链。

        ms 用浮点保留 3 位，避免亚毫秒步骤被 int() 压成 0；
        不再有「ms: 0 但没有 detail」的空壳步骤。
        """
        step: dict = {"agent": agent, "action": action, "detail": detail, "ms": 0.0, "tokens": 0}
        started = time.perf_counter()
        previous = getattr(self._local, "active_step", None)
        self._local.active_step = step
        try:
            yield step
        finally:
            step["ms"] = round((time.perf_counter() - started) * 1000, 3)
            usage = step.get("usage")
            if isinstance(usage, Usage):
                usage_dict = usage.to_dict()
            elif isinstance(usage, dict):
                usage_dict = dict(usage)
            else:
                usage_dict = estimated_usage("code", 0, step["ms"]).to_dict()
            step["usage"] = usage_dict
            step["tokens"] = usage_dict["display_tokens"]
            self._local.active_step = previous
            trace.append(step)

    def _understand(self, message: str, state: str, focused: bool) -> tuple:
        """理解层：走专用模型（若配置），否则跟随主模型。

        兼容自定义 LLM：有 *_with_usage 就采实测，否则只记估算且不计费。
        未路由时每次读 self.llm，保证测试里替换 c.llm 后理解层也跟着换。
        """
        llm = self.llm_classify if self._classify_routed else self.llm
        fn = getattr(llm, "classify_with_usage", None)
        if callable(fn):
            return fn(message, state, focused)
        understanding = llm.classify(message, state, focused)
        return understanding, estimated_usage(
            getattr(llm, "name", "custom"), understanding.tokens,
            note="自定义 LLM 未提供用量明细，按字符估算记录且不计费",
        )

    def _polish(self, draft: str, facts: dict) -> tuple:
        fn = getattr(self.llm, "polish_with_usage", None)
        if callable(fn):
            return fn(draft, facts)
        text, tokens = self.llm.polish(draft, facts)
        return text, estimated_usage(
            getattr(self.llm, "name", "custom"), tokens,
            note="自定义 LLM 未提供用量明细，按字符估算记录且不计费",
        )

    # ---------- 一轮对话 ----------
    def _chat_locked(self, session_id: str, user_id: str, message: str) -> ChatResult:
        started = time.perf_counter()
        request_id = "req_" + uuid.uuid4().hex[:12]
        session = self.flow.ensure_session(session_id, user_id)
        trace: list[dict] = []

        if self.settings.agent_mode == "single":
            understanding, agent_path, output = self._run_single(message, session, trace, request_id)
        else:
            understanding, agent_path, output = self._run_multi(message, session, trace, request_id)

        # 引用完整性收口（B13）：答复正文里写了某条结论（如「运费由商家承担」），
        # 引用列表就必须含真正承载它的条款（R-05）——否则用户点开引用找不到依据。
        # 放在表达层之前，让 _build_facts 的 clause_ids 与最终话术一致，避免
        # 话术里出现的条款号被 grounding 当成「引用列表外的标识符」而回退。
        completed_ids = complete_citations(
            output.reply, output.citations, self.flow.index.clauses
        )
        if completed_ids != [c["clause_id"] for c in output.citations]:
            output.citations = self.flow._citations(completed_ids)

        # 表达层：真实调用润色 + 后置事实校验，耗时与用量一起记录
        facts = self._build_facts(output)
        draft = output.reply
        with self._agent_step(trace, "表达层", "话术生成", {}) as step:
            polished, polish_usage = self._polish(draft, facts)
            guard = verify(draft, polished, facts, enabled=self.settings.grounding_enabled)
            step["usage"] = polish_usage
            if not guard.ok:
                action = "事实校验回退"
            elif guard.used_polished:
                action = "话术润色"
            else:
                action = "话术模板"
            step["action"] = action
            step["detail"] = {
                "polished": guard.used_polished,
                "grounding_ok": guard.ok,
                "checked": guard.detail.get("checked", False),
                "violations": guard.violations,
                "model": getattr(self.llm, "model", "rules"),
                "role": "main",
            }
        reply = guard.reply

        meta = dict(output.meta)
        if guard.detail.get("checked"):
            meta["grounding"] = {
                "ok": guard.ok,
                "fallback": not guard.ok,
                "used_polished": guard.used_polished,
                "violations": guard.violations,
                # 诊断细节：例如 prompt_leak 具体命中了哪个标记，便于定位第三方模型的输出习惯
                "detail": {k: v for k, v in guard.detail.items() if k != "violations"},
            }
        if not guard.ok:
            self.db.log_event(
                session_id, request_id, "grounding_rejected",
                {
                    "violations": guard.violations,
                    "detail": {
                        key: guard.detail[key]
                        for key in ("missing_ids", "extra_ids", "missing_numbers", "extra_numbers", "promises", "flips")
                        if key in guard.detail
                    },
                },
            )

        usage_report = build_turn_usage(trace, self.settings, self._pricing_table())
        latency_ms = int((time.perf_counter() - started) * 1000)
        tokens = sum(step["tokens"] for step in trace)
        orchestration = {
            "mode": self.settings.agent_mode,
            "steps": [
                {
                    "agent": step["agent"],
                    "action": step["action"],
                    "ms": step["ms"],
                    "depends_on": trace[i - 1]["agent"] if i > 0 else None,
                    "tool_calls": len(step["detail"].get("tools", [])),
                    "usage_source": step["usage"]["source"],
                    "measured": step["usage"]["measured"],
                }
                for i, step in enumerate(trace)
            ],
            "wall_ms": latency_ms,
            "sum_step_ms": round(sum(step["ms"] for step in trace), 3),
        }
        meta["orchestration"] = orchestration

        result = ChatResult(
            reply=reply,
            state=output.state.value,
            state_label=_state_label(output.state),
            slots=output.public_slots(),
            trace=trace,
            citations=output.citations,
            ticket=output.ticket,
            cards=output.cards,
            suggestions=output.suggestions,
            request_id=request_id,
            agent_mode=self.settings.agent_mode,
            agent_path=agent_path,
            latency_ms=latency_ms,
            tokens=tokens,
            usage=usage_report,
            meta=meta,
        )
        self.db.log_event(
            session_id,
            request_id,
            "turn",
            {
                "message": message,
                "state": result.state,
                "agent_mode": self.settings.agent_mode,
                "agent_path": agent_path,
                "tokens": tokens,
                "latency_ms": latency_ms,
                "measured_tokens": usage_report.get("total_tokens", 0),
                "estimated_tokens": usage_report.get("estimated_tokens", 0),
                "estimated_cost_cny": usage_report.get("estimated_cost_cny", 0),
                "handoff_order_id": output.slots.get("order_id"),
                "intent": understanding.intent,
                "usage": {
                    "basis": usage_report["basis"],
                    "billing_mode": usage_report["billing_mode"],
                    "measured": usage_report["measured"],
                    "estimated_tokens": usage_report["estimated_tokens"],
                    "estimated_cost_cny": usage_report["estimated_cost_cny"],
                    "pricing": usage_report["pricing"],
                },
                "orchestration": orchestration,
            },
        )
        # 事项 C：关键节点埋点（写失败不影响主链路，见 FunnelService.track）
        self._track_funnel(session_id, user_id, request_id, understanding.intent, result, trace)
        return result

    def _track_funnel(
        self, session_id: str, user_id: str, request_id: str,
        intent: str, result: ChatResult, trace: list[dict],
    ) -> None:
        """把一轮对话映射到漏斗节点；event_id 幂等，重试/刷新不重复计数。"""
        from ..domain.events import EventName

        f = self.funnel
        # 会话开始：event_id 固定，整段会话只计一次
        f.track(session_id, user_id, EventName.SESSION_START.value,
                {}, event_id=f"{session_id}:session_start")
        # 意图识别：每轮一次
        f.track(session_id, user_id, EventName.INTENT_RECOGNIZED.value,
                {"intent": intent}, event_id=f"{session_id}:intent:{request_id}")

        # 从真实工具调用链映射节点（只看成功的工具）
        called = set()
        for step in trace:
            for tool in step["detail"].get("tools", []):
                if tool.get("ok"):
                    called.add(tool["tool"])
        if "query_order" in called:
            f.track(session_id, user_id, EventName.ORDER_QUERIED.value,
                    {"request_id": request_id}, event_id=f"{session_id}:order:{request_id}")
        if "check_return_eligibility" in called:
            f.track(session_id, user_id, EventName.ELIGIBILITY_CHECKED.value,
                    {"request_id": request_id}, event_id=f"{session_id}:elig:{request_id}")
        if intent == "confirm" or result.state == "CREATING":
            f.track(session_id, user_id, EventName.CONFIRMED.value,
                    {"request_id": request_id}, event_id=f"{session_id}:confirm:{request_id}")
        if result.ticket:
            # 建单以工单号为幂等键：响应丢失重试也只记一次
            f.track(session_id, user_id, EventName.TICKET_CREATED.value,
                    {"ticket_id": result.ticket["ticket_id"]},
                    event_id=f"ticket:{result.ticket['ticket_id']}")
        # 失败 / 转人工分支
        if "query_order" in called and any(
            not t.get("ok") and t["tool"] == "query_order"
            for step in trace for t in step["detail"].get("tools", [])
        ):
            f.track(session_id, user_id, EventName.ORDER_QUERY_FAILED.value,
                    {"request_id": request_id}, event_id=f"{session_id}:orderfail:{request_id}")
        if result.state == "HUMAN":
            f.track(session_id, user_id, EventName.HANDED_OFF.value,
                    {"request_id": request_id}, event_id=f"{session_id}:handoff:{request_id}")

    # ---------- 表达层事实上下文 ----------
    def _build_facts(self, output) -> dict:
        """给 polish 的权威事实清单，同时作为护栏的"允许出现的标识符"来源。

        注意：只允许出现，不强制出现——强制保留的事实一律以 draft 为准，
        避免把 citations 里没写进话术的条款号误判成"丢失"。
        """
        slots = output.slots
        ticket = output.ticket or {}
        order = slots.get("order") or {}
        eligibility = slots.get("eligibility") or {}
        return {
            "state": output.state.value,
            "order_id": slots.get("order_id"),
            # B12：商品名属于结构性事实，护栏据此识别润色丢词。
            "order_title": order.get("title"),
            "pending_order_title": slots.get("pending_order_title"),
            "ticket_id": ticket.get("ticket_id") or slots.get("ticket_id"),
            "amount": order.get("amount"),
            "order_status": order.get("status_label"),
            "signed_at": order.get("signed_label"),
            "deadline": eligibility.get("deadline"),
            "remaining_days": eligibility.get("remaining_days"),
            "eligibility": eligibility.get("title"),
            "clause_ids": [c["clause_id"] for c in output.citations],
        }

    # ---------- 多 Agent：真实调用链 ----------
    def _run_multi(self, message: str, session: dict, trace: list[dict], request_id: str) -> tuple:
        # 1) 接待 Agent：真实调用理解层（LLM 模式是真实模型请求）
        with self._agent_step(trace, "接待 Agent", "识别意图", {}) as step:
            understanding, usage = self._understand(message, session["state"], focused=True)
            step["usage"] = usage
            step["detail"] = {
                "intent": understanding.intent,
                "order_id": understanding.order_id,
                "reason": understanding.reason,
                "confidence": understanding.confidence,
                "model": getattr(self.llm_classify if self._classify_routed else self.llm, "model", "rules"),
                "role": "classify" if self._classify_routed else "main",
            }
            # 槽位被护栏降级时留痕（reason 字段已是最终值，这里给出「为什么改」）
            # 槽位被护栏修正时统一留痕（reason / order_id / intent 三类同一写法）
            for field in ("reason_correction", "order_correction", "intent_correction"):
                if understanding.slots.get(field):
                    step["detail"][field] = understanding.slots[field]

        # 2) Supervisor：真实做路由决策，并把槽位打包成 handoff 交给下游
        with self._agent_step(trace, "Supervisor", "路由", {}) as step:
            target = ROUTE_MAP.get(understanding.intent, "售后 Agent")
            handoff = {
                "to": target,
                "intent": understanding.intent,
                "order_id": understanding.order_id,
                "reason": understanding.reason,
                "session_state": session["state"],
            }
            understanding.slots["handoff"] = handoff
            step["detail"] = {"to": target, "intent": understanding.intent, "context": handoff}
            step["usage"] = estimated_usage(
                "code", estimate_tokens(json.dumps(handoff, ensure_ascii=False)),
                note="Supervisor 本地路由与上下文打包，无模型调用",
            )

        # 3) 下游 Agent：真实执行自己的工具链 / 状态机，不再是 ms=0 的空壳
        with self._agent_step(trace, target, "办理", {}) as step:
            if target == "政策 Agent":
                # 政策 Agent 自己检索条款，结果交给状态机消费，避免重复检索
                answer = self.tools.call(
                    "search_policy",
                    {"query": message, "top_k": 2},
                    {
                        "session_id": session["session_id"],
                        "request_id": request_id,
                        "user_id": session["user_id"],
                    },
                )
                understanding.slots["policy_answer"] = answer
            output = self.flow.handle(session, message, understanding, request_id=request_id)
            step["usage"] = estimated_usage(
                "code", estimate_tokens(AGENT_PROMPTS.get(target, "") + message),
                note="下游 Agent 真实执行工具/状态机；本地代码无模型调用",
            )
            step["detail"] = {
                "intent": understanding.intent,
                "handoff": handoff,
                "tools": step["detail"].get("tools", []),
                "result": {
                    "state": output.state.value,
                    "ticket_id": (output.ticket or {}).get("ticket_id"),
                    "citations": [c["clause_id"] for c in output.citations],
                    "reply_chars": len(output.reply or ""),
                    "cards": len(output.cards),
                },
            }
        return understanding, ["接待 Agent", "Supervisor", target], output

    # ---------- 单 Agent：一个 Agent 真实做完理解 + 办理 ----------
    def _run_single(self, message: str, session: dict, trace: list[dict], request_id: str) -> tuple:
        with self._agent_step(trace, "通用 Agent", "识别并办理", {}) as step:
            understanding, usage = self._understand(message, session["state"], focused=False)
            output = self.flow.handle(session, message, understanding, request_id=request_id)
            step["usage"] = usage
            step["detail"] = {
                "intent": understanding.intent,
                "order_id": understanding.order_id,
                "reason": understanding.reason,
                "tools": step["detail"].get("tools", []),
                "result": {
                    "state": output.state.value,
                    "ticket_id": (output.ticket or {}).get("ticket_id"),
                    "citations": [c["clause_id"] for c in output.citations],
                    "reply_chars": len(output.reply or ""),
                },
            }
            # 槽位被护栏修正时统一留痕（reason / order_id / intent 三类同一写法）
            for field in ("reason_correction", "order_correction", "intent_correction"):
                if understanding.slots.get(field):
                    step["detail"][field] = understanding.slots[field]
        return understanding, ["通用 Agent"], output

    # ---------- 运行时模型切换 ----------
    def _pricing_table(self) -> dict[str, dict]:
        """模型名 → 定价，供计费按实际模型取价（切换后口径仍然正确）。"""
        return {
            model_id: {
                "in_per_1k": p.price_in_per_1k,
                "out_per_1k": p.price_out_per_1k,
                "multiplier": p.multiplier,
                "source": p.price_source,
                "note": "运行时模型注册表定价（按实际模型名取价）",
            }
            for model_id, p in self._registry.items()
        }

    def models_view(self) -> dict:
        """给前端：可选模型 + 两层当前选择。api_key 永不出现。"""
        with self._model_guard:
            options = [p.to_public() for p in self._registry.values()]
            return {
                "options": options,
                "active": {
                    "main": self._active_main,
                    "classify": self._active_classify,
                },
                "switchable": len(options) > 1,
            }

    def switch_model(self, role: str, model_id: str) -> dict:
        """运行时把某一层切到指定模型；整轮对话下一条消息起生效。

        role: main（表达层/润色）或 classify（理解层/意图识别）。
        模型不存在、角色非法、llm 未启用都会抛 ValueError，由接口层转 400。
        """
        if role not in ("main", "classify"):
            raise ValueError("role 必须是 main 或 classify")
        with self._model_guard:
            profile = self._registry.get(model_id)
            if profile is None:
                raise ValueError(f"未知模型：{model_id}")
            instance = OpenAICompatibleLLM(
                profile.base_url, profile.api_key, profile.id, profile.timeout
            )
            if role == "main":
                self.llm = instance
                self._active_main = model_id
            else:
                # 理解层显式选中一个模型后，永远走独立实例（不再「跟随主模型」）
                self.llm_classify = instance
                self._classify_routed = True
                self._active_classify = model_id
            return self.models_view()

    # ---------- 观测 ----------
    def session_view(self, session_id: str, user_id: str) -> dict:
        session = self.db.get_session(session_id)
        if not session or session["user_id"] != user_id:
            return {
                "session_id": session_id, "state": State.IDLE.value,
                "state_label": _state_label(State.IDLE), "slots": {},
                "human_replies": self.desk.user_messages(session_id),
            }
        slots = dict(session["slots"])
        slots.pop("confirm_token", None)
        # 事项 B：坐席回复合并进会话视图，确保用户在原会话看得到且不被 Agent 消息覆盖
        human_replies = self.desk.user_messages(session["session_id"])
        return {
            "session_id": session["session_id"],
            "user_id": session["user_id"],
            "state": session["state"],
            "state_label": _state_label(State(session["state"])),
            "slots": slots,
            "updated_at": session["updated_at"],
            "human_replies": human_replies,
        }

    def usage_view(self, session_id: str) -> dict:
        """会话级用量汇总：从 turn 事件里累加，归属校验由接口层负责。"""
        events = self.db.list_events(session_id, limit=200)
        usages = [
            event["payload"]["usage"]
            for event in events
            if event["kind"] == "turn" and isinstance(event["payload"].get("usage"), dict)
        ]
        usages.reverse()  # list_events 是倒序，恢复成时间顺序
        return {"session_id": session_id, **summarize_turns(usages)}

    def health(self) -> dict:
        view = self.models_view()
        described = self.settings.describe()
        runtime = {}
        if view["switchable"]:
            # 有可切换模型时，覆盖成运行时实际生效的选择（可能与 .env 初始值不同）
            runtime = {
                "model": view["active"]["main"],
                "classify_model": view["active"]["classify"],
                "classify_routed": self._classify_routed,
            }
        return {
            "status": "ok",
            "orders": len(self.orders),
            "order_source_detail": self.order_source.describe(),
            "policy_clauses": len(self.index.clauses),
            "tickets": self.db.count_tickets(),
            **described,
            **runtime,
            "model_options": view["options"],
        }

    def close(self) -> None:
        self._locker.close()
        self.db.close()


def _state_label(state: State) -> str:
    from .flow import STATE_LABEL

    return STATE_LABEL.get(state, state.value)
