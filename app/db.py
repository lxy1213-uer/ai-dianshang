"""SQLite 持久化：会话状态、售后工单、事件日志（第 5、6、8 周共用）。"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    state      TEXT NOT NULL,
    slots      TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id       TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    session_id      TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    order_id        TEXT NOT NULL,
    reason          TEXT NOT NULL,
    status          TEXT NOT NULL,
    request_id      TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tickets_session ON tickets(session_id);
CREATE INDEX IF NOT EXISTS idx_tickets_order ON tickets(order_id);

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    request_id TEXT,
    kind       TEXT NOT NULL,
    payload    TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    event_id   TEXT,
    ticket_id  TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);

-- 事项 B：人工坐席队列。一个会话最多一条待办，认领需互斥。
CREATE TABLE IF NOT EXISTS human_queue (
    queue_id      TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL UNIQUE,
    user_id       TEXT NOT NULL,
    state         TEXT NOT NULL DEFAULT 'PENDING',   -- PENDING / CLAIMED / RESOLVED
    assignee      TEXT,
    reason        TEXT NOT NULL DEFAULT '',
    context       TEXT NOT NULL DEFAULT '{}',         -- 打包的会话上下文
    sla_deadline  TEXT,                               -- 超时时间
    overdue       INTEGER NOT NULL DEFAULT 0,         -- 是否超时标记
    replies       TEXT NOT NULL DEFAULT '[]',         -- 坐席回复列表
    ticket_id     TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    resolved_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_queue_state ON human_queue(state);
CREATE INDEX IF NOT EXISTS idx_queue_assignee ON human_queue(assignee);

-- 架构进阶：退款流水（ticket_id 主键 = 天然幂等，绝不重复打款）
CREATE TABLE IF NOT EXISTS refund_requests (
    ticket_id       TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    status          TEXT NOT NULL,          -- PENDING / SUCCESS / FAILED
    claim_id        TEXT,                   -- 原子占位者，防止并发双打款
    refund_id       TEXT,
    amount          REAL,
    error           TEXT,
    operator        TEXT NOT NULL,
    gateway         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
"""

# 旧库迁移：CREATE TABLE IF NOT EXISTS 不会给已存在的表补列，这里显式补齐。
_TICKET_MIGRATIONS = (
    ("version", "INTEGER NOT NULL DEFAULT 1"),
    ("updated_at", "TEXT"),
    ("next_step", "TEXT"),
)
_EVENT_MIGRATIONS = (
    ("event_id", "TEXT"),
    ("ticket_id", "TEXT"),
)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Database:
    """单连接 + 线程锁：足够支撑课程项目，且避免 Windows 上多线程写冲突。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """幂等补列：老库没有 version/updated_at/next_step/event_id/ticket_id。"""
        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(tickets)")}
        for name, ddl in _TICKET_MIGRATIONS:
            if name not in existing:
                self._conn.execute(f"ALTER TABLE tickets ADD COLUMN {name} {ddl}")
        # 老数据补齐：updated_at 回填 created_at，version 默认 1，状态值标准化为大写
        self._conn.execute(
            "UPDATE tickets SET updated_at = created_at WHERE updated_at IS NULL OR updated_at = ''"
        )
        self._conn.execute("UPDATE tickets SET status = 'CREATED' WHERE status = 'created'")
        ev_existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(events)")}
        for name, ddl in _EVENT_MIGRATIONS:
            if name not in ev_existing:
                self._conn.execute(f"ALTER TABLE events ADD COLUMN {name} {ddl}")
        # 依赖新列的索引必须在补列之后建（旧库直接 executescript 会因列不存在而失败）
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ticket ON events(ticket_id)")
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_events_event_id ON events(event_id) WHERE event_id IS NOT NULL"
        )
        rf_existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(refund_requests)")}
        if "amount" not in rf_existing:
            self._conn.execute("ALTER TABLE refund_requests ADD COLUMN amount REAL")
        self._conn.commit()

    # ---------- 会话 ----------
    def get_session(self, session_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if not row:
            return None
        return {
            "session_id": row["session_id"],
            "user_id": row["user_id"],
            "state": row["state"],
            "slots": json.loads(row["slots"] or "{}"),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def save_session(self, session_id: str, user_id: str, state: str, slots: dict) -> dict:
        now = _now()
        payload = json.dumps(slots, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sessions (session_id, user_id, state, slots, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    state = excluded.state,
                    slots = excluded.slots,
                    updated_at = excluded.updated_at
                """,
                (session_id, user_id, state, payload, now, now),
            )
            self._conn.commit()
        return {
            "session_id": session_id,
            "user_id": user_id,
            "state": state,
            "slots": slots,
            "updated_at": now,
        }

    def list_sessions(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {
                "session_id": row["session_id"],
                "user_id": row["user_id"],
                "state": row["state"],
                "slots": json.loads(row["slots"] or "{}"),
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    # ---------- 工单 ----------
    def get_ticket_by_key(self, idempotency_key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tickets WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
        return dict(row) if row else None

    def get_ticket(self, ticket_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
        return dict(row) if row else None

    def insert_ticket(
        self,
        idempotency_key: str,
        session_id: str,
        user_id: str,
        order_id: str,
        reason: str,
        request_id: str,
    ) -> tuple[dict, bool]:
        """INSERT OR IGNORE + 唯一约束：并发/重试下只会落一条工单。"""
        ticket_id = "RT" + datetime.now().strftime("%Y%m%d") + uuid.uuid4().hex[:6].upper()
        now = _now()
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO tickets
                    (ticket_id, idempotency_key, session_id, user_id, order_id, reason, status, request_id, created_at, version, updated_at, next_step)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_id, idempotency_key, session_id, user_id, order_id, reason,
                    "CREATED", request_id, now, 1, now, "售后专员会尽快审核你的申请，请留意进度更新。",
                ),
            )
            self._conn.commit()
            created = cursor.rowcount == 1
            if not created:
                row = self._conn.execute(
                    "SELECT * FROM tickets WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
                return dict(row), False
        return {
            "ticket_id": ticket_id,
            "idempotency_key": idempotency_key,
            "session_id": session_id,
            "user_id": user_id,
            "order_id": order_id,
            "reason": reason,
            "status": "CREATED",
            "request_id": request_id,
            "created_at": now,
            "version": 1,
            "updated_at": now,
            "next_step": "售后专员会尽快审核你的申请，请留意进度更新。",
        }, True

    def list_tickets(self, session_id: str | None = None, user_id: str | None = None, limit: int = 20) -> list[dict]:
        """按会话查工单；传入 user_id 时强制过滤归属，避免跨用户读取。"""
        clauses: list[str] = []
        params: list = []
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM tickets{where} ORDER BY created_at DESC LIMIT ?", tuple(params)
            ).fetchall()
        return [dict(row) for row in rows]

    def count_tickets(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) AS c FROM tickets").fetchone()["c"])

    def update_ticket_status(
        self,
        ticket_id: str,
        new_status: str,
        operator: str,
        next_step: str,
        idempotency_key: str,
        expected_version: int,
    ) -> tuple[dict | None, str]:
        """原子状态回写：幂等 + 并发冲突控制，返回 (工单, 结果标记)。

        结果标记：
        - "updated"：状态已变更，version +1，写了审计事件；
        - "idempotent"：同一幂等键重复提交，返回原工单，不重复写审计；
        - "not_found"：工单不存在。

        整个判定+写入在单连接锁内完成；条件 UPDATE 用 version 做乐观锁，
        即使换成多进程也能靠 SQLite 写锁保证并发只有一个生效。
        """
        now = _now()
        with self._lock:
            # 1) 幂等：同一回写键已经处理过，直接返回当前快照
            prior = self._conn.execute(
                "SELECT ticket_id FROM events WHERE event_id = ? LIMIT 1", (idempotency_key,)
            ).fetchone()
            if prior:
                row = self._conn.execute(
                    "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
                ).fetchone()
                return (dict(row) if row else None), "idempotent"

            # 2) 乐观锁条件更新：version 与读到时一致才写
            current = self._conn.execute(
                "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
            if not current:
                return None, "not_found"
            old_status = current["status"]
            old_version = current["version"]
            # expected_version 由调用方在读到快照时记录；并发时只有版本仍匹配才写成功。
            cursor = self._conn.execute(
                """
                UPDATE tickets
                   SET status = ?, version = version + 1, updated_at = ?, next_step = ?
                 WHERE ticket_id = ? AND version = ?
                """,
                (new_status, now, next_step, ticket_id, expected_version),
            )
            if cursor.rowcount == 0:
                # 并发：另一个请求已经先改了这一版
                latest = self._conn.execute(
                    "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
                ).fetchone()
                return (dict(latest) if latest else None), "conflict"

            self._conn.execute(
                """
                INSERT INTO events (session_id, request_id, kind, payload, created_at, event_id, ticket_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    current["session_id"], current["request_id"], "ticket_status_change",
                    json.dumps(
                        {"ticket_id": ticket_id, "from": old_status, "to": new_status,
                         "operator": operator, "version": old_version + 1},
                        ensure_ascii=False,
                    ),
                    now, idempotency_key, ticket_id,
                ),
            )
            self._conn.commit()
            updated = self._conn.execute(
                "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
            return dict(updated), "updated"

    def log_metric_event(
        self, session_id: str, user_id: str, kind: str, payload: dict, event_id: str
    ) -> bool:
        """事项 C：漏斗埋点写入（幂等 event_id），失败由调用方降级。返回是否新写入。"""
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM events WHERE event_id = ? LIMIT 1", (event_id,)
            ).fetchone()
            if exists:
                return False
            self._conn.execute(
                """
                INSERT INTO events (session_id, request_id, kind, payload, created_at, event_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, user_id, kind, json.dumps(payload, ensure_ascii=False), _now(), event_id),
            )
            self._conn.commit()
            return True

    def has_event(self, event_id: str) -> bool:
        with self._lock:
            return self._conn.execute(
                "SELECT 1 FROM events WHERE event_id = ? LIMIT 1", (event_id,)
            ).fetchone() is not None

    def list_metric_events(
        self, date_from: str | None = None, date_to: str | None = None, limit: int = 10000
    ) -> list[dict]:
        """按时间范围筛出漏斗事件（event_id 非空即埋点事件）。"""
        clauses = ["event_id IS NOT NULL"]
        params: list = []
        if date_from:
            clauses.append("created_at >= ?")
            params.append(date_from)
        if date_to:
            # 支持 'YYYY-MM-DD'：当天算到 23:59:59
            to = date_to if len(date_to) > 10 else date_to + " 23:59:59"
            clauses.append("created_at <= ?")
            params.append(to)
        where = " WHERE " + " AND ".join(clauses)
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT session_id, kind, payload, created_at FROM events{where} ORDER BY id ASC LIMIT ?",
                tuple(params),
            ).fetchall()
        return [
            {
                "session_id": row["session_id"], "kind": row["kind"],
                "payload": json.loads(row["payload"] or "{}"), "created_at": row["created_at"],
            }
            for row in rows
        ]

    def ticket_timeline(self, ticket_id: str, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT payload, created_at FROM events
                 WHERE ticket_id = ? AND kind = 'ticket_status_change'
                 ORDER BY id ASC LIMIT ?
                """,
                (ticket_id, limit),
            ).fetchall()
        return [
            {"payload": json.loads(row["payload"] or "{}"), "created_at": row["created_at"]}
            for row in rows
        ]

    # ---------- 事件日志（幂等重试、异常演练留痕） ----------
    def log_event(self, session_id: str | None, request_id: str | None, kind: str, payload: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO events (session_id, request_id, kind, payload, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, request_id, kind, json.dumps(payload, ensure_ascii=False), _now()),
            )
            self._conn.commit()

    def list_events(self, session_id: str, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [
            {
                "kind": row["kind"],
                "request_id": row["request_id"],
                "payload": json.loads(row["payload"] or "{}"),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    # ---------- 决策审计驾驶舱：通用事件查询 ----------
    def query_events(
        self,
        kinds: list[str] | None = None,
        session_id: str | None = None,
        session_prefix: str | None = None,
        request_id: str | None = None,
        ticket_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 500,
        order: str = "asc",
    ) -> list[dict]:
        """审计用通用事件查询：按 kind/会话/请求/工单/时间组合过滤。

        与 list_events 的区别：支持多条件与升/降序，且保留 event_id/ticket_id。
        读出的 payload 由 AuditService 再做一次脱敏兜底（log_event 写入未统一 sanitize）。
        """
        clauses: list[str] = []
        params: list = []
        if kinds:
            clauses.append(f"kind IN ({','.join('?' for _ in kinds)})")
            params.extend(kinds)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if session_prefix:
            clauses.append("session_id LIKE ?")
            params.append(session_prefix + "%")
        if request_id:
            clauses.append("request_id = ?")
            params.append(request_id)
        if ticket_id:
            clauses.append("ticket_id = ?")
            params.append(ticket_id)
        if date_from:
            clauses.append("created_at >= ?")
            params.append(date_from)
        if date_to:
            to = date_to if len(date_to) > 10 else date_to + " 23:59:59"
            clauses.append("created_at <= ?")
            params.append(to)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        direction = "DESC" if order == "desc" else "ASC"
        params.append(max(1, min(int(limit or 500), 10000)))
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, session_id, request_id, kind, payload, created_at, event_id, ticket_id "
                f"FROM events{where} ORDER BY id {direction} LIMIT ?",
                tuple(params),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "request_id": row["request_id"],
                "kind": row["kind"],
                "payload": json.loads(row["payload"] or "{}"),
                "created_at": row["created_at"],
                "event_id": row["event_id"],
                "ticket_id": row["ticket_id"],
            }
            for row in rows
        ]

    def get_ticket_by_session(self, session_id: str) -> dict | None:
        """按会话取工单（诊断决策的建单会话为 diag_<decision_id>_ticket）。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tickets WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    # ---------- 事项 B：人工坐席队列 ----------
    def enqueue_human(
        self,
        session_id: str,
        user_id: str,
        reason: str,
        context: dict,
        sla_deadline: str,
        ticket_id: str | None = None,
    ) -> tuple[dict, bool]:
        """转人工入队；同一会话重复入队返回原记录（幂等）。"""
        now = _now()
        queue_id = "HQ" + datetime.now().strftime("%Y%m%d") + uuid.uuid4().hex[:6].upper()
        with self._lock:
            existing = self._conn.execute(
                "SELECT * FROM human_queue WHERE session_id = ?", (session_id,)
            ).fetchone()
            if existing and existing["state"] != "RESOLVED":
                return self._queue_public(dict(existing)), False
            cursor = self._conn.execute(
                """
                INSERT OR REPLACE INTO human_queue
                    (queue_id, session_id, user_id, state, assignee, reason, context,
                     sla_deadline, overdue, replies, ticket_id, created_at, updated_at, resolved_at)
                VALUES (?, ?, ?, 'PENDING', NULL, ?, ?, ?, 0, '[]', ?, ?, ?, NULL)
                """,
                (
                    queue_id, session_id, user_id, reason,
                    json.dumps(context, ensure_ascii=False), sla_deadline,
                    ticket_id, now, now,
                ),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM human_queue WHERE session_id = ?", (session_id,)
            ).fetchone()
            return self._queue_public(dict(row)), cursor.rowcount > 0

    def claim_human(self, queue_id: str, assignee: str) -> tuple[dict | None, str]:
        """认领：条件 UPDATE 保证同一会话只能一个坐席认领，返回 (记录, 结果)。"""
        now = _now()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM human_queue WHERE queue_id = ?", (queue_id,)
            ).fetchone()
            if not row:
                return None, "not_found"
            if row["state"] == "RESOLVED":
                return dict(row), "resolved"
            if row["state"] == "CLAIMED" and row["assignee"] == assignee:
                return dict(row), "idempotent"
            cursor = self._conn.execute(
                """
                UPDATE human_queue SET state = 'CLAIMED', assignee = ?, updated_at = ?
                 WHERE queue_id = ? AND state = 'PENDING'
                """,
                (assignee, now, queue_id),
            )
            self._conn.commit()
            if cursor.rowcount == 0:
                latest = self._conn.execute(
                    "SELECT * FROM human_queue WHERE queue_id = ?", (queue_id,)
                ).fetchone()
                return dict(latest), "conflict"
            updated = self._conn.execute(
                "SELECT * FROM human_queue WHERE queue_id = ?", (queue_id,)
            ).fetchone()
            return dict(updated), "claimed"

    def reassign_human(self, queue_id: str, assignee: str) -> tuple[dict | None, str]:
        """转派：把待办改派给另一名坐席（仅未归档的可转），返回 (记录, 结果)。"""
        now = _now()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM human_queue WHERE queue_id = ?", (queue_id,)
            ).fetchone()
            if not row:
                return None, "not_found"
            if row["state"] == "RESOLVED":
                return dict(row), "resolved"
            if (row["assignee"] or "") == assignee:
                return dict(row), "idempotent"
            self._conn.execute(
                """
                UPDATE human_queue SET assignee = ?, state = 'CLAIMED', updated_at = ?
                 WHERE queue_id = ?
                """,
                (assignee, now, queue_id),
            )
            self._conn.commit()
            updated = self._conn.execute(
                "SELECT * FROM human_queue WHERE queue_id = ?", (queue_id,)
            ).fetchone()
            return dict(updated), "reassigned"

    def append_human_reply(self, queue_id: str, assignee: str, text: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM human_queue WHERE queue_id = ?", (queue_id,)
            ).fetchone()
            if not row:
                return None
            replies = json.loads(row["replies"] or "[]")
            replies.append({
                "assignee": assignee, "text": text, "at": _now(),
            })
            self._conn.execute(
                "UPDATE human_queue SET replies = ?, updated_at = ? WHERE queue_id = ?",
                (json.dumps(replies, ensure_ascii=False), _now(), queue_id),
            )
            self._conn.commit()
            return {"session_id": row["session_id"], "replies": replies}

    def resolve_human(self, queue_id: str) -> dict | None:
        now = _now()
        with self._lock:
            self._conn.execute(
                """
                UPDATE human_queue SET state = 'RESOLVED', resolved_at = ?, updated_at = ?
                 WHERE queue_id = ? AND state != 'RESOLVED'
                """,
                (now, now, queue_id),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM human_queue WHERE queue_id = ?", (queue_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_human_queue(
        self, state: str | None = None, assignee: str | None = None,
        only_overdue: bool = False, limit: int = 50,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list = []
        if state:
            clauses.append("state = ?")
            params.append(state.upper())
        if assignee:
            clauses.append("assignee = ?")
            params.append(assignee)
        if only_overdue:
            clauses.append("overdue = 1")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM human_queue{where} ORDER BY created_at DESC LIMIT ?", tuple(params)
            ).fetchall()
        return [self._queue_public(dict(row)) for row in rows]

    def get_human_queue(self, queue_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM human_queue WHERE queue_id = ?", (queue_id,)
            ).fetchone()
        return self._queue_public(dict(row)) if row else None

    def get_human_queue_by_session(self, session_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM human_queue WHERE session_id = ?", (session_id,)
            ).fetchone()
        return self._queue_public(dict(row)) if row else None

    def mark_overdue_records(self, now: str | None = None) -> list[dict]:
        """把超时未认领的待办打标记，并返回**本次新超时**的记录（供 SLA 通知）。

        先查后更在同一把锁内完成，返回的记录即"刚刚从 0 变 1"的那些，通知据此去重。
        """
        now = now or _now()
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM human_queue
                 WHERE state = 'PENDING' AND overdue = 0 AND sla_deadline IS NOT NULL
                   AND sla_deadline <= ?
                """,
                (now,),
            ).fetchall()
            records = [self._queue_public(dict(row)) for row in rows]
            if records:
                self._conn.execute(
                    """
                    UPDATE human_queue SET overdue = 1, updated_at = ?
                     WHERE state = 'PENDING' AND overdue = 0 AND sla_deadline IS NOT NULL
                       AND sla_deadline <= ?
                    """,
                    (now, now),
                )
                self._conn.commit()
            return records

    def mark_overdue(self, now: str | None = None) -> int:
        """定时调用：把超时未认领的待办打上标记，返回新超时数量。"""
        return len(self.mark_overdue_records(now))

    @staticmethod
    def _queue_public(row: dict) -> dict:
        row["context"] = json.loads(row.get("context") or "{}")
        row["replies"] = json.loads(row.get("replies") or "[]")
        return row

    def recover_creating(self) -> int:
        """启动时自愈：进程在建单中途崩溃会留下 CREATING 会话，恢复为等待确认。"""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE sessions SET state = ?, updated_at = ? WHERE state = ?",
                ("AWAIT_CONFIRM", _now(), "CREATING"),
            )
            self._conn.commit()
            count = cursor.rowcount
        if count:
            self.log_event(
                None, None, "state_recovered",
                {"from": "CREATING", "to": "AWAIT_CONFIRM", "count": count},
            )
        return count

    # ---------- 架构进阶：退款流水（幂等占位 / 完成） ----------
    def get_refund(self, ticket_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM refund_requests WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
            return dict(row) if row else None

    def all_refunds(self) -> list[dict]:
        """对账用：列出全部退款流水（含 refund_id/status/amount）。"""
        with self._lock:
            return [dict(row) for row in self._conn.execute(
                "SELECT * FROM refund_requests ORDER BY created_at"
            ).fetchall()]

    def reserve_refund(
        self, ticket_id: str, idempotency_key: str, claim_id: str,
        operator: str, gateway: str,
    ) -> tuple[str, dict]:
        """原子占位。返回 (action, row)：success / claimed / busy。

        INSERT OR IGNORE 借主键唯一约束保证并发只有一个调用方能进入 claimed；
        FAILED 的流水允许条件更新抢占重试，PENDING 则只认自己的 claim_id。
        """
        with self._lock:
            now = _now()
            self._conn.execute(
                """INSERT OR IGNORE INTO refund_requests
                    (ticket_id, idempotency_key, status, claim_id, operator,
                     gateway, created_at, updated_at)
                    VALUES (?, ?, 'PENDING', ?, ?, ?, ?, ?)""",
                (ticket_id, idempotency_key, claim_id, operator, gateway, now, now),
            )
            self._conn.commit()
            row = dict(self._conn.execute(
                "SELECT * FROM refund_requests WHERE ticket_id = ?", (ticket_id,)
            ).fetchone())
            if row["status"] == "SUCCESS":
                return "success", row
            if row["status"] == "PENDING":
                return ("claimed", row) if row["claim_id"] == claim_id else ("busy", row)
            # FAILED：原子抢占为 PENDING 以便重试（同样只允许一个调用方）
            self._conn.execute(
                """UPDATE refund_requests SET status = 'PENDING', claim_id = ?, updated_at = ?
                     WHERE ticket_id = ? AND status = 'FAILED'""",
                (claim_id, now, ticket_id),
            )
            self._conn.commit()
            row = dict(self._conn.execute(
                "SELECT * FROM refund_requests WHERE ticket_id = ?", (ticket_id,)
            ).fetchone())
            return ("claimed", row) if row["claim_id"] == claim_id else ("busy", row)

    def finish_refund(
        self, ticket_id: str, claim_id: str, status: str,
        refund_id: str | None = None, error: str | None = None,
        amount: float | None = None,
    ) -> dict | None:
        """只完成自己占位的那笔流水，避免并发请求互相覆盖结果。"""
        with self._lock:
            self._conn.execute(
                """UPDATE refund_requests
                       SET status = ?, refund_id = ?, error = ?, amount = ?, updated_at = ?
                     WHERE ticket_id = ? AND claim_id = ?""",
                (status, refund_id, error, amount, _now(), ticket_id, claim_id),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM refund_requests WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
            return dict(row) if row else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
