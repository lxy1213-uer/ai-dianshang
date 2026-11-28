# 用本地替身配置启动应用（不改 .env）。在 Windows PowerShell 下运行。
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location $root
try {
    python scripts/local_stubs/run_local_app.py
}
finally {
    Pop-Location
}
