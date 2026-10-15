#!/usr/bin/env bash
# make_func_md.sh —— 一键重新生成仓库根目录的 func.md。
#
# 用法（在仓库根目录执行）：
#   bash scripts/make_func_md.sh            # 生成
#   bash scripts/make_func_md.sh --check    # 校验（CI 用）
#
# 包装 `python scripts/scan_func_md.py`：
#   - 自动切到本脚本所在仓库的根目录（避免在子目录跑找不到根）；
#   - 默认与 main 分支同步：每次扫描都从头计算，**全量覆盖** func.md；
#   - 通过 `--check` 走只读模式，调用方可在 CI 里用它拦截漂移。

set -euo pipefail

# 推断仓库根：脚本位于 <repo>/scripts/make_func_md.sh
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

cd "$REPO_ROOT"

exec python scripts/scan_func_md.py "$@"
