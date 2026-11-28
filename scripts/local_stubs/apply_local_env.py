"""把本地替身参数合并进项目 .env（保留其他配置，先备份）。

默认只预览差异；确认后加 --apply 才真正写入。

    python scripts/local_stubs/apply_local_env.py            # 预览
    python scripts/local_stubs/apply_local_env.py --apply    # 写入（备份 .env.bak.local）
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".env"
TEMPLATE_PATH = ROOT / "deploy" / "local" / ".env.local.example"
BACKUP_PATH = ROOT / ".env.bak.local"
KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def read_template(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def merge(env_text: str, updates: dict[str, str]) -> tuple[str, dict[str, str]]:
    seen: dict[str, str] = {}
    lines = []
    for raw in env_text.splitlines():
        match = KEY_RE.match(raw)
        if match and match.group(1) in updates:
            key = match.group(1)
            old = match.group(2).strip().strip('"').strip("'")
            new = updates[key]
            seen.setdefault(key, old)
            lines.append(f"{key}={new}")
        else:
            lines.append(raw)
    for key, value in updates.items():
        if key not in seen:
            lines.append(f"{key}={value}")
            seen[key] = "<未配置>"
    changed = {k: (v, updates[k]) for k, v in seen.items() if v != updates[k]}
    return "\n".join(lines) + "\n", changed


def main() -> None:
    parser = argparse.ArgumentParser(description="合并本地替身配置到 .env")
    parser.add_argument("--apply", action="store_true", help="真正写入 .env（默认仅预览）")
    parser.add_argument("--template", default=str(TEMPLATE_PATH))
    args = parser.parse_args()

    template = Path(args.template)
    if not template.exists():
        raise SystemExit(f"找不到模板：{template}")
    updates = read_template(template)
    current = ENV_PATH.read_text(encoding="utf-8") if ENV_PATH.exists() else ""
    merged, changed = merge(current, updates)

    print(f"模板：{template}")
    print(f"目标：{ENV_PATH}")
    print(f"变更 {len(changed)} 项：")
    for key, (old, new) in changed.items():
        masked_old = old if len(old) < 24 else old[:8] + "..."
        masked_new = new if len(new) < 24 else new[:8] + "..."
        print(f"  {key}: {masked_old} -> {masked_new}")
    if not args.apply:
        print("\n这是预览。确认无误后加 --apply 写入。")
        return
    shutil.copyfile(ENV_PATH, BACKUP_PATH)
    ENV_PATH.write_text(merged, encoding="utf-8")
    print(f"\n已写入 {ENV_PATH}；原文件备份在 {BACKUP_PATH}")


if __name__ == "__main__":
    main()
