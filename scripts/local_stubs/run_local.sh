#!/usr/bin/env bash
# 用本地替身配置启动应用（不改 .env）。在 Git Bash / Linux / macOS 下运行。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec python scripts/local_stubs/run_local_app.py "$@"
