"""SLA 扫描调度协议：默认由拉队列触发，可选独立后台定时任务。

原来的实现只在「拉坐席队列」时顺手扫一次超时——没人打开工作台就不会告警。
这里把调度抽成协议：`ManualScheduler` 保持零配置行为不变；`BackgroundScheduler`
用守护线程按间隔调用扫描函数（真正的超时判定/去重/通知仍在 AgentDesk 里，
调度器只负责「何时调用」）。多实例部署应由外部 cron/K8s Job 调用
`POST /api/desk/sla-check`，不要每个实例都起线程重复推送。
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Protocol

logger = logging.getLogger(__name__)


class SLAScheduler(Protocol):
    def start(self, scan: Callable[[], object]) -> None: ...
    def stop(self) -> None: ...


class ManualScheduler:
    """默认：不额外起线程，由端点/拉队列触发。"""

    def start(self, scan: Callable[[], object]) -> None:
        return None

    def stop(self) -> None:
        return None


class BackgroundScheduler:
    """按固定间隔扫描；线程是 daemon，stop() 会 join 以便测试与优雅退出。"""

    def __init__(self, interval_seconds: float = 60.0, run_immediately: bool = False):
        self.interval = max(1.0, float(interval_seconds))
        self.run_immediately = run_immediately
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, scan: Callable[[], object]) -> None:
        if self._thread is not None:
            return

        def loop() -> None:
            first = True
            while not self._stop.is_set():
                if not first or self.run_immediately:
                    try:
                        scan()
                    except Exception as exc:  # 定时任务绝不能把进程带崩
                        logger.warning("SLA 定时扫描失败（已跳过本轮）：%s", type(exc).__name__)
                first = False
                self._stop.wait(self.interval)

        self._stop.clear()
        self._thread = threading.Thread(target=loop, name="sla-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval))
            self._thread = None


def build_scheduler(settings) -> SLAScheduler:
    mode = str(getattr(settings, "sla_scheduler", "manual") or "manual").lower()
    if mode == "background":
        return BackgroundScheduler(
            interval_seconds=float(getattr(settings, "sla_scan_seconds", 60) or 60),
            run_immediately=False,
        )
    return ManualScheduler()
