"""坐席 Copilot：给人工坐席的只读建议层。

坐席台原本只把交接上下文原样摊开，坐席得自己翻条款、判期限、猜该说什么。
本层把已经做好的诊断能力复用到坐席侧，产出三样东西：

- **推荐答复**：基于用户原话 + 当前订单事实的可采纳草稿（不是固定话术）；
- **条款依据**：真实条款编号与要点，坐席可核对；
- **风险提示**：超期 / 越权 / 承诺越界 / 已退款复议 / 数据缺失 / SLA 超时；
- **建议动作**：建单 / 转人工复核 / 查询到账 / 补充凭证，均需坐席显式执行。

纪律（与售后诊断同源）：
1. **纯只读**：不建单、不退款、不改订单/会话状态、不替坐席发送任何回复；
2. **重新核单**：不信任入队时的快照，按队列里的 user_id + order_id 现查订单并判资格，
   订单不属于该用户就报归属异常，绝不把他人订单喂给坐席；
3. **证据同源**：诉求走 detect_reason + validate_reason，情形走 read_situation，
   与用户侧口径完全一致；
4. **外部订单源故障**：明确报 order_source_unavailable，不伪装成「查不到订单」；
5. **不越权承诺**：推荐答复只复述条款与当前事实，不写「一定能退 / 一定退款」。

一层关键设计：所有事实都来自服务端队列记录与订单源，接口只接 queue_id，
不接受前端传来的 user_id / order_id / 结论——坐席看到什么由服务端决定。
"""

from __future__ import annotations

from typing import Any

from ..domain.order_sources import OrderSourceError
from ..domain.orders import (
    CODE_CATEGORY_EXCLUDED,
    CODE_DATA_MISSING,
    CODE_ELIGIBLE_NO_REASON,
    CODE_ELIGIBLE_QUALITY,
    CODE_NOT_SHIPPED,
    CODE_NOT_SIGNED,
    CODE_ORDER_CLOSED,
    CODE_OUT_OF_WINDOW,
    CATEGORY_NO_REASON_EXCLUDED,
    REASON_LABEL,
    evaluate_return,
)
from ..llm import detect_reason
from ..slots import normalize_order_id, status_question_evidence, validate_reason
from .advice import build_advice, read_situation

# 已退款复查：evaluate_return 对 refunded 也会给出 requires_human 的结论，
# 这里单独用订单状态识别，避免依赖某个具体 code 常量名。
_REFUNDED_STATUS = "refunded"


class CopilotError(Exception):
    """Copilot 对外的稳定错误码，HTTP 层按 status 映射。"""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class DeskCopilot:
    """只读建议服务；持有 coordinator 引用以复用订单源、资格判定与条款。"""

    def __init__(self, coordinator: Any):
        self.coordinator = coordinator

    # ---------- 对外入口 ----------
    def suggest(self, queue_id: str) -> dict:
        record = self.coordinator.desk.detail(queue_id)
        try:
            self.coordinator.ensure_orders_available()
        except OrderSourceError as exc:
            raise CopilotError("order_source_unavailable", str(exc), 503) from exc

        context = record.get("context") or {}
        slots = context.get("slots") or {}
        user_id = (record.get("user_id") or "").strip()
        question = self._last_user_message(record, context)
        order_id = normalize_order_id(slots.get("order_id"))

        payload = self._build(record, context, slots, user_id, question, order_id)
        # 审计：坐席查看建议本身也留痕（含命中风险，便于运营复盘口径）
        self.coordinator.db.log_event(
            record["session_id"], queue_id, "desk_copilot_view",
            {
                "queue_id": queue_id,
                "verdict": payload["summary"]["v_code"],
                "risks": [r["code"] for r in payload["risks"]],
                "actions": [a["id"] for a in payload["actions"]],
            },
        )
        return payload

    # ---------- 组装 ----------
    def _build(
        self, record: dict, context: dict, slots: dict,
        user_id: str, question: str, order_id: str | None,
    ) -> dict:
        order = None
        if order_id:
            order = self.coordinator.orders.get(order_id)
            if order is not None and order.user_id != user_id:
                # 归属异常：不给订单事实、不判资格，直接报风险
                order = None
                ownership_broken = True
            else:
                ownership_broken = False
        else:
            ownership_broken = False

        # 诉求来源优先取交接时已判定的 reason：真实模型路径下它是模型给的值，
        # 坐席看到的口径应和接待 Agent 当时一致；没有才用当前原话重新识别。
        claimed = (slots.get("reason") or "").strip() or detect_reason(question)
        reason_verdict = validate_reason(claimed, question)
        reason = reason_verdict.reason
        situation = read_situation(question)
        advice = build_advice(question, reason=reason)

        risk_list: list[dict] = []
        facts: list[dict] = []
        eligibility: dict | None = None

        if ownership_broken:
            risk_list.append(self._risk(
                "ownership_mismatch", "high",
                "交接单里的订单不属于该会话用户，已屏蔽订单事实；请勿按本单答复，先核实会话归属。",
            ))
        if order is None:
            if not order_id:
                risk_list.append(self._risk(
                    "no_order", "high",
                    "交接上下文没有订单号，无法核期限；先向用户索要订单号，不要承诺可退。",
                ))
            elif not ownership_broken:
                risk_list.append(self._risk(
                    "order_not_found", "high",
                    f"订单 {order_id} 在当前订单源中查不到；不要把它当普通查无订单处理，先核实来源。",
                ))
        else:
            facts.append(self._fact("订单", order.order_id, "订单源"))
            facts.append(self._fact("商品", order.title, "订单源"))
            facts.append(self._fact("订单状态", order.status_label, "订单源"))
            facts.append(self._fact("签收时间", order.signed_label, "订单源"))
            facts.append(self._fact("实付金额", f"¥{order.amount:.2f}", "订单源"))
            if order.category in CATEGORY_NO_REASON_EXCLUDED:
                facts.append(self._fact("类目限制", f"{order.category}（不适用七天无理由）", "R-03"))

            el = evaluate_return(
                order, self.coordinator.settings.current_date(), reason,
                no_reason_days=self.coordinator.settings.return_window_days,
                quality_days=self.coordinator.settings.quality_window_days,
            )
            eligibility = el.to_dict()
            risk_list.extend(self._eligibility_risks(eligibility, order))

        # 与用户侧同源的「已退款复议」和「质量证据不足」提示
        # 注意：status_question_evidence 返回的是 verdict 对象，bool(obj) 恒为 True，
        # 必须读 .is_status_question 字段，不能用「if verdict」判断。
        if status_question_evidence(question).is_status_question:
            risk_list.append(self._risk(
                "refund_reapply", "high",
                "用户问的是已退款订单能否重复申请；按 R-05 不能重复申请，复核前不要承诺再次退款。",
            ))
        if reason_verdict.reason == "no_reason" and reason_verdict.claimed == "quality":
            risk_list.append(self._risk(
                "quality_unproven", "high",
                f"用户自称质量问题但原话缺少缺陷证据（{reason_verdict.rule}），"
                "按无理由口径判断；若要走质量通道先让用户补照片/视频。",
            ))

        if record.get("overdue"):
            risk_list.append(self._risk(
                "sla_overdue", "medium",
                "该会话已超 SLA，建议优先处理或转派；不要因赶时效跳过核单。",
            ))
        if record.get("ticket_id"):
            risk_list.append(self._risk(
                "ticket_exists", "medium",
                f"会话已关联工单 {record['ticket_id']}，建单走幂等，不要重复建单。",
            ))

        recommended = self._recommended_reply(
            question, reason, situation, advice, eligibility, order, ownership_broken,
        )
        clauses = self._clauses(advice, eligibility)
        # 与 B13 引用完整性同原则：答复里的关键事实必须有条款支撑。
        # 退款复议（R-05）与重复申请是明确结论，检索权重未必把它带进 citations。
        risk_codes = {r["code"] for r in risk_list}
        if "refunded" in risk_codes or "refund_reapply" in risk_codes:
            if not any(c["clause_id"] == "R-05" for c in clauses):
                clauses.append({"clause_id": "R-05", "reason": "已退款不可重复申请结论"})
        actions = self._actions(eligibility, order, reason, ownership_broken)

        return {
            "queue_id": record["queue_id"],
            "session_id": record["session_id"],
            "question": question,
            "summary": {
                "v_code": (eligibility or {}).get("code", "NO_ORDER"),
                "v_title": (eligibility or {}).get("title", "未核单：缺少订单"),
                "reason": reason,
                "reason_label": REASON_LABEL.get(reason, reason),
                "situation": situation,
                "grounded": order is not None and not ownership_broken,
            },
            "facts": facts,
            "eligibility": eligibility,
            "recommended_reply": recommended,
            "clauses": clauses,
            "risks": risk_list,
            "actions": actions,
            "boundaries": [
                "本建议只读，不代表已发送、已建单或已退款；任何动作都由坐席显式执行。",
                "推荐答复仅为草稿，发送前请结合订单事实与平台最终审核口径复核。",
                "资金与身份类操作一律人工确认；本页不会自动执行退款或改单。",
            ],
        }

    # ---------- 细节 ----------
    @staticmethod
    def _last_user_message(record: dict, context: dict) -> str:
        """还原用户诉求：把交接打包的多轮 turn.message 拼起来，退化用转人工原因。

        不能用「最后一条 turn.message」：用户常在最后一句只回一个订单号，
        情形词（刻字 / 发霉 / 用不上）在前几轮，取最后一句会丢掉关键证据，
        导致建议走通用口径。拼接后交给与用户侧同源的情形/诉求识别。
        """
        turns = context.get("recent_turns") or []
        texts: list[str] = []
        for turn in turns:
            payload = turn.get("payload") or {}
            text = (payload.get("message") or "").strip() if isinstance(payload, dict) else ""
            # 排除内部事件误带的 message，只保留用户原话（turn 事件）
            if text and turn.get("kind") == "turn" and text not in texts:
                texts.append(text)
        joined = " ".join(texts).strip()
        return joined or (record.get("reason") or "").strip()

    @staticmethod
    def _fact(label: str, value: str, source: str) -> dict:
        return {"label": label, "value": str(value), "source": source}

    @staticmethod
    def _risk(code: str, level: str, message: str) -> dict:
        return {"code": code, "level": level, "message": message}

    @staticmethod
    def _eligibility_risks(elig: dict, order: Any) -> list[dict]:
        code = elig.get("code")
        out: list[dict] = []
        if order.status == _REFUNDED_STATUS:
            # 退款单的 evaluate_return code 是 ORDER_CLOSED，「已退款」是更准确的
            # 坐席提示，且必须独立于 code 判断（否则被 ORDER_CLOSED 分支吞掉）。
            out.append(DeskCopilot._risk(
                "refunded", "high",
                "订单已退款，R-05 不能重复申请；先查询到账进度，不要二次退款。",
            ))
        if code == CODE_OUT_OF_WINDOW:
            out.append(DeskCopilot._risk(
                "out_of_window", "high",
                f"已超过退货期限（签收 {order.signed_label}，剩 {elig.get('remaining_days')} 天）；"
                "不要承诺七天无理由可退，可引导质量问题或协商。",
            ))
        elif code == CODE_CATEGORY_EXCLUDED:
            out.append(DeskCopilot._risk(
                "category_excluded", "high",
                f"类目「{order.category}」不适用七天无理由（R-03）；"
                "确属质量问题（发霉/破损）才按 R-02 15 天通道。",
            ))
        elif code == CODE_DATA_MISSING:
            out.append(DeskCopilot._risk(
                "data_missing", "high",
                "签收时间等关键数据缺失，无法判期限；先补数据或转人工复核，不要臆测天数。",
            ))
        elif code == CODE_NOT_SIGNED:
            out.append(DeskCopilot._risk(
                "not_signed", "medium",
                "订单尚未签收，不能按「签收后 N 天」判；未发货按 R-04 取消，已发货可拒收。",
            ))
        elif code == CODE_NOT_SHIPPED:
            out.append(DeskCopilot._risk(
                "not_shipped", "medium",
                "订单未发货，按 R-04 直接取消订单全额退款即可，不必走退货流程。",
            ))
        elif code == CODE_ORDER_CLOSED:
            out.append(DeskCopilot._risk(
                "order_closed", "high",
                "订单已关闭，无法发起售后；确需处理应转主管复核，不要承诺可退。",
            ))

        if elig.get("requires_human") and code not in (
            CODE_ELIGIBLE_NO_REASON, CODE_ELIGIBLE_QUALITY,
        ):
            out.append(DeskCopilot._risk(
                "needs_review", "medium",
                "资格判定要求人工复核，建议按本页建议动作处理后再答复用户。",
            ))
        return out

    @staticmethod
    def _recommended_reply(
        question: str, reason: str, situation: str, advice: Any,
        eligibility: dict | None, order: Any, ownership_broken: bool,
    ) -> str:
        if ownership_broken:
            return (
                "你好，我需要先核实一下这笔订单的归属信息，才能给你准确答复。"
                "麻烦稍等，我确认后会立即回复你。"
            )
        if order is None:
            return advice.body

        lines: list[str] = []
        if advice.evidence:
            lines.append(f"关于你说的「{advice.evidence}」，我按你这单核了一遍：")
        else:
            lines.append("按你这单的当前记录核了一遍：")
        lines.append(f"· 商品「{order.title}」，{order.status_label}，签收 {order.signed_label}。")
        if eligibility:
            lines.append(f"· 资格结论：{eligibility['title']}。{eligibility['detail']}")
            if eligibility.get("deadline") and eligibility.get("remaining_days") is not None:
                lines.append(
                    f"· 期限：截止 {eligibility['deadline']} 24:00 前提交，"
                    f"目前剩 {eligibility['remaining_days']} 天（签收当日不计入）。"
                )
        # 规则口径用 advice 的情形结论句（headline）。只在情形词真正命中时附加：
        # 转人工时 recent_turns 只保留最近 6 条事件，用户早期诉求常被挤出，
        # 若只剩订单号，advice 会走 unknown 给「两条通道」这种通用开头——
        # 对已经核过单的场景是废话，反而稀释资格结论，所以此时不附加。
        if advice.basis == "situation" and advice.headline:
            lines.append(f"· 规则口径：{advice.headline}")
        if reason == "quality":
            lines.append("· 质量问题请提供问题照片或视频，运费按 R-05 由商家承担。")
        elif eligibility and eligibility.get("code") == CODE_ELIGIBLE_NO_REASON:
            lines.append("· 七天无理由退货的运费需自理，商品需保持完好、吊牌包装齐全。")
        lines.append("以上按当前订单与政策给出，最终以平台审核为准；需要我继续帮你办理吗？")
        return "\n".join(lines)

    @staticmethod
    def _clauses(advice: Any, eligibility: dict | None) -> list[dict]:
        ids: list[str] = []
        for cid in list(advice.clauses) + list((eligibility or {}).get("citations") or []):
            if cid and cid not in ids:
                ids.append(cid)
        return [{"clause_id": cid, "reason": "规则口径 / 资格判定依据"} for cid in ids]

    @staticmethod
    def _actions(
        eligibility: dict | None, order: Any, reason: str, ownership_broken: bool,
    ) -> list[dict]:
        if ownership_broken or order is None:
            return [
                {"id": "verify_ownership", "label": "核实订单归属", "auto": False},
                {"id": "ask_order_id", "label": "向用户索要订单号", "auto": False},
            ]
        code = (eligibility or {}).get("code")
        out: list[dict] = []
        if order.status == _REFUNDED_STATUS:
            # 先于 ORDER_CLOSED 判断：退款单要给「查到账」而不是「解释并协商」
            out.append({"id": "query_refund_status", "label": "查询退款到账进度（R-05）", "auto": False})
            out.append({"id": "send_reply", "label": "采纳草稿并显式发送", "auto": False})
            return out
        if code in (CODE_ELIGIBLE_NO_REASON, CODE_ELIGIBLE_QUALITY):
            out.append({
                "id": "create_ticket", "label": "可引导建退货工单（需坐席确认）", "auto": False,
            })
        elif code in (CODE_OUT_OF_WINDOW, CODE_CATEGORY_EXCLUDED, CODE_ORDER_CLOSED):
            out.append({
                "id": "review_then_reply", "label": "按条款解释并引导协商 / 转人工复核", "auto": False,
            })
        elif code == CODE_DATA_MISSING:
            out.append({"id": "collect_evidence", "label": "补充签收/物流凭证后重判", "auto": False})
        elif code == CODE_NOT_SHIPPED:
            out.append({"id": "guide_cancel_order", "label": "按 R-04 引导取消订单", "auto": False})
        if reason == "quality":
            out.append({"id": "collect_evidence", "label": "要求补充问题照片/视频", "auto": False})
        out.append({"id": "send_reply", "label": "采纳草稿并显式发送", "auto": False})
        return out
