#!/usr/bin/env bash
# rollout-insight Cursor skill 安装脚本。
#
# 把本仓库的 skill/ 软链到 ~/.cursor/skills/rollout-insight，让 Cursor 能识别。
# 用户手动执行：bash skill/install.sh
#
# 行为：
#   - 默认目标：~/.cursor/skills/rollout-insight
#   - 已存在同名目录/链接 → 提示用户备份后再覆盖（除非 OVERRIDE=1）
#   - 软链指向 SKILL.md 同级目录，即仓库的 skill/ 目录
set -euo pipefail

# 解析仓库根目录：本脚本位于 <repo>/skill/install.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SKILL_DIR="${SCRIPT_DIR}"
LINK_PATH="${HOME}/.cursor/skills/rollout-insight"

echo "[rollout-insight skill install]"
echo "  source : ${SKILL_DIR}"
echo "  target : ${LINK_PATH}"

if [[ ! -f "${SKILL_DIR}/SKILL.md" ]]; then
    echo "[ERROR] 找不到 ${SKILL_DIR}/SKILL.md，请确认仓库结构。" >&2
    exit 1
fi

mkdir -p "${HOME}/.cursor/skills"

if [[ -e "${LINK_PATH}" || -L "${LINK_PATH}" ]]; then
    if [[ "${OVERRIDE:-0}" == "1" ]]; then
        echo "[warn] 已存在 ${LINK_PATH}，按 OVERRIDE=1 覆盖"
        rm -rf "${LINK_PATH}"
    else
        echo "[ERROR] 已存在 ${LINK_PATH}；如确认要覆盖：OVERRIDE=1 bash skill/install.sh" >&2
        exit 1
    fi
fi

ln -sfn "${SKILL_DIR}" "${LINK_PATH}"

echo "[ok] 已创建软链 ${LINK_PATH} -> ${SKILL_DIR}"
echo
echo "  验证：ls -l ${LINK_PATH}"
echo "  卸载：rm ${LINK_PATH}"