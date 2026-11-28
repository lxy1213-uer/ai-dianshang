"""事项 C：埋点漏斗服务。

- 统一事件名来自 domain.events.EventName；
- 埋点写入幂等（event_id），写失败降级不影响主链路；
- 聚合按会话去重，漏斗各节点人数单调不增，可与会话总数对齐。
"""

from __future__ import annotations

from ..db import Database
from .events import FUNNEL_LABELS, FUNNEL_STEPS, EventName, sanitize


class FunnelService:
    def __init__(self, db: Database):
        self.db = db

    def track(
        self,
        session_id: str,
        user_id: str,
        kind: str,
        payload: dict | None = None,
        event_id: str | None = None,
    ) -> bool:
        """埋点写入；event_id 缺省由 会话+事件+请求 派生，保证重试不重复计数。"""
        eid = event_id or (payload or {}).get("event_id")
        if not eid:
            # 无显式 id 时按 payload 里的 request_id/turn 派生一个稳定 id
            seed = (payload or {}).get("request_id") or ""
            eid = f"{session_id}:{kind}:{seed}"
        try:
            return self.db.log_metric_event(
                session_id, user_id, kind, sanitize(payload or {}), eid
            )
        except Exception:
            # 埋点失败不能影响主链路（聊天仍返回 200）
            return False

    def funnel(
        self, date_from: str | None = None, date_to: str | None = None
    ) -> dict:
        """漏斗聚合：每个节点的去重会话数 + 相邻转化率 + 整体转化率。"""
        events = self.db.list_metric_events(date_from, date_to)
        branches: dict[str, int] = {}
        branch_kinds = {
            EventName.ORDER_QUERY_FAILED.value,
            EventName.INELIGIBLE.value,
            EventName.CANCELLED.value,
            EventName.HANDED_OFF.value,
        }
        # 每个会话取"最深的已到达节点"：漏斗是累积路径，到达后面节点必然经过前面节点。
        # 这样即使埋点乱序/缺失，各节点人数也严格单调不增。
        step_index = {kind: i for i, kind in enumerate(FUNNEL_STEPS)}
        deepest: dict[str, int] = {}
        for ev in events:
            kind = ev["kind"]
            sid = ev["session_id"]
            if kind in step_index:
                deepest[sid] = max(deepest.get(sid, -1), step_index[kind])
            elif kind in branch_kinds:
                branches[kind] = branches.get(kind, 0) + 1

        per_step = [0] * len(FUNNEL_STEPS)
        for depth in deepest.values():
            for i in range(depth + 1):
                per_step[i] += 1

        steps = []
        prev_count = None
        first_count = per_step[0]
        for i, kind in enumerate(FUNNEL_STEPS):
            count = per_step[i]
            steps.append({
                "event": kind,
                "label": FUNNEL_LABELS[kind],
                "sessions": count,
                "conversion_from_prev": None if prev_count in (None, 0) else round(count / prev_count, 4),
                "conversion_overall": None if first_count == 0 else round(count / first_count, 4),
            })
            prev_count = count

        return {
            "from": date_from, "to": date_to,
            "total_sessions": first_count,
            "steps": steps,
            "branches": branches,
        }

    def by_mode(self) -> dict:
        """按 单 Agent / 多 Agent 分组的成功率、耗时、Token 口径。

        从 turn 事件里累加；这是演示看板的聚合视图。
        """
        # 直接查全库 turn 事件（list_events 需要 session_id，这里走全表扫描）
        with self.db._lock:
            rows = self.db._conn.execute(
                "SELECT payload FROM events WHERE kind = 'turn'"
            ).fetchall()
        import json
        groups: dict[str, dict] = {}
        for row in rows:
            payload = json.loads(row["payload"] or "{}")
            usage = payload.get("usage") or {}
            mode = payload.get("agent_mode") or "multi"
            g = groups.setdefault(mode, {"turns": 0, "ok": 0, "ms": 0.0,
                                          "estimated_tokens": 0, "measured_tokens": 0})
            g["turns"] += 1
            if payload.get("ok", True):
                g["ok"] += 1
            g["ms"] += float(payload.get("latency_ms") or 0)
            g["estimated_tokens"] += int(payload.get("estimated_tokens") or 0)
            g["measured_tokens"] += int(payload.get("measured_tokens") or 0)
        for g in groups.values():
            g["success_rate"] = round(g["ok"] / g["turns"], 4) if g["turns"] else 0
            g["avg_ms"] = round(g["ms"] / g["turns"], 2) if g["turns"] else 0
        return groups
