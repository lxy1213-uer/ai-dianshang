"""外部通知通道：企业微信群机器人 webhook（SLA 超时告警）。

设计纪律（与项目一贯的降级原则一致）：
- 通知是**旁路**：失败只记日志、绝不影响主流程（转人工/回复/归档都不因通知失败而中断）；
- 使用标准库 urllib，不引入新依赖；超时短（5s），避免拖慢请求；
- webhook 支持完整 URL 或仅 key（自动拼成企微机器人地址），key 不写进日志；
- 发送内容经 events.sanitize 脱敏，不泄漏手机号/地址等 PII；
- 未配置 webhook 或显式关闭时直接跳过（no-op），返回 False。
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Any, Iterable, Protocol

from .domain.events import sanitize

logger = logging.getLogger(__name__)

_WECOM_BASE = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key="


class Notifier(Protocol):
    """通知通道协议：SLA 只依赖这个接口，便于换成飞书/邮件测试。"""

    configured: bool

    def notify_sla_overdue(self, records: Iterable[dict]) -> bool: ...


class WecomNotifier:
    def __init__(self, webhook: str = "", enabled: bool = True,
                 desk_base_url: str = "", timeout: float = 5.0,
                 retries: int = 0, retry_backoff: float = 0.5,
                 sleep=None) -> None:
        raw = (webhook or "").strip()
        self.webhook = raw if raw.startswith("http") else (_WECOM_BASE + raw if raw else "")
        self.enabled = enabled and bool(self.webhook)
        self.desk_base_url = desk_base_url.rstrip("/")
        self.timeout = timeout
        # 默认 0 次重试（保持既有行为）；配置后指数退避，最后一次仍失败才降级。
        self.retries = max(0, int(retries))
        self.retry_backoff = max(0.0, float(retry_backoff))
        self._sleep = sleep or time.sleep
        # 便于测试注入发送函数；默认走 urllib
        self._send = self._http_post

    # ---------- 配置态 ----------
    @property
    def configured(self) -> bool:
        return self.enabled

    # ---------- 业务通知 ----------
    def notify_sla_overdue(self, records: Iterable[dict]) -> bool:
        """把新超时的转人工待办合并成一条告警推送；返回是否成功发送。"""
        if not self.enabled:
            return False
        items = list(records)
        if not items:
            return False
        content = self._build_sla_markdown(items)
        payload = {"msgtype": "markdown", "markdown": {"content": content}}
        attempts = self.retries + 1
        for attempt in range(attempts):
            try:
                ok = self._send(self.webhook, payload)
                if ok:
                    return True
            except Exception as exc:  # 通知失败绝不向上抛
                logger.warning("SLA 通知发送异常（第 %d/%d 次）：%s",
                               attempt + 1, attempts, type(exc).__name__)
                ok = False
            if attempt + 1 < attempts:
                self._sleep(self.retry_backoff * (2 ** attempt))
        logger.warning("SLA 通知发送失败（已降级 %d 次，不影响主流程）", attempts)
        return False

    def _build_sla_markdown(self, records: list[dict]) -> str:
        # 逐条脱敏后再拼装，避免 reason/context 里的 PII 泄漏到外部群
        safe = [sanitize(r) for r in records]
        lines = [
            "# ⏰ 售后转人工超时告警",
            f"> 共 **{len(safe)}** 条待办超过 5 分钟未认领，请及时跟进。",
            "",
        ]
        for r in safe[:20]:  # 群消息不宜过长，最多列 20 条
            qid = r.get("queue_id", "")
            reason = str(r.get("reason") or "未说明")[:60]
            deadline = r.get("sla_deadline") or ""
            uid = r.get("user_id", "")
            link = f"{self.desk_base_url}/desk#queue" if self.desk_base_url else ""
            tail = f" [前往坐席台]({link})" if link else ""
            lines.append(
                f"- `{qid}` · 用户 {uid} · 超时于 {deadline}{tail}\n"
                f"  <font color=\"comment\">原因：{reason}</font>"
            )
        if len(safe) > 20:
            lines.append(f"\n> 另有 {len(safe) - 20} 条未展示，请到坐席台查看。")
        return "\n".join(lines)

    def _http_post(self, url: str, payload: dict) -> bool:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        # 企微成功返回 {"errcode": 0}
        try:
            return json.loads(body).get("errcode") == 0
        except (ValueError, AttributeError):
            return False


def build_notifier(settings: Any) -> Notifier:
    """从 Settings 构造通知器（集中一处，便于各处复用）。"""
    return WecomNotifier(
        webhook=getattr(settings, "wecom_webhook", ""),
        enabled=getattr(settings, "sla_notify_enabled", True),
        desk_base_url=getattr(settings, "desk_base_url", ""),
        retries=getattr(settings, "sla_notify_retries", 0),
    )
