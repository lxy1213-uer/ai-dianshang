"""用本地替身参数启动应用，不改动 .env（LLM 密钥等原有环境变量保持不变）。

    python scripts/local_stubs/run_local_app.py

等价于把 deploy/local/.env.local.example 里的变量注入当前进程后运行 uvicorn。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "deploy" / "local" / ".env.local.example"


def inject() -> None:
    for raw in TEMPLATE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()


def main() -> None:
    inject()
    sys.path.insert(0, str(ROOT))
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
