#!/usr/bin/env bash
# restart_server.sh —— 安全释放端口后重启 rollout-insight Web 服务。
#
# 解决：ERROR [Errno 98] ... bind on address ('0.0.0.0', 39998): address already in use
#
# 用法（在任意目录执行均可）：
#   bash scripts/restart_server.sh              # 默认：停旧进程 → 前台启动
#   bash scripts/restart_server.sh restart      # 同上
#   bash scripts/restart_server.sh stop         # 只停
#   bash scripts/restart_server.sh start        # 只启（端口占用则先尝试安全释放）
#   bash scripts/restart_server.sh status       # 看谁占用端口、是否像本服务
#   bash scripts/restart_server.sh restart --bg # 后台启动，日志写 log/server.log
#   bash scripts/restart_server.sh restart --force   # 端口上非本服务进程也强制杀掉
#
# 环境变量：
#   ROLLOUT_INSIGHT_PORT   默认 39998
#   ROLLOUT_INSIGHT_PYTHON  默认 python3（找不到则回退 python）
#   ROLLOUT_INSIGHT_HOST    健康检查用，默认 127.0.0.1

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
cd "$REPO_ROOT"

PORT="${ROLLOUT_INSIGHT_PORT:-39998}"
HOST="${ROLLOUT_INSIGHT_HOST:-127.0.0.1}"
LOG_DIR="${REPO_ROOT}/log"
PID_FILE="${LOG_DIR}/server.pid"
LOG_FILE="${LOG_DIR}/server.log"
WAIT_STOP_SEC=8
WAIT_READY_SEC=30

if [[ -n "${ROLLOUT_INSIGHT_PYTHON:-}" ]]; then
  PYTHON="$ROLLOUT_INSIGHT_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  PYTHON="python"
fi

CMD="restart"
BACKGROUND=0
FORCE=0

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
}

is_our_cmd() {
  local cmd="$1"
  # 本仓库 uvicorn / main.py；避免误杀无关进程
  [[ "$cmd" == *"main:app"* ]] && return 0
  [[ "$cmd" == *"main.py"* ]] && return 0
  [[ "$cmd" == *"uvicorn"* && "$cmd" == *"rollout"* ]] && return 0
  return 1
}

listener_pids() {
  local port="$1"
  if ! command -v ss >/dev/null 2>&1; then
    echo "错误：需要 ss（iproute）来查端口占用" >&2
    return 1
  fi
  # 只取 LISTEN；pid= 可能在 users:(("python",pid=123,fd=...))
  ss -ltnp 2>/dev/null | awk -v p=":${port}" '
    $0 ~ p {
      while (match($0, /pid=[0-9]+/)) {
        print substr($0, RSTART+4, RLENGTH-4)
        $0 = substr($0, RSTART+RLENGTH)
      }
    }
  ' | sort -u
}

cmdline_of() {
  local pid="$1"
  if [[ -r "/proc/${pid}/cmdline" ]]; then
    tr '\0' ' ' < "/proc/${pid}/cmdline" | sed 's/[[:space:]]*$//'
  else
    echo ""
  fi
}

print_holders() {
  local pids
  pids="$(listener_pids "$PORT" || true)"
  if [[ -z "${pids}" ]]; then
    echo "端口 ${PORT}：空闲"
    return 0
  fi
  echo "端口 ${PORT} 占用进程："
  local pid cmd ours
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    cmd="$(cmdline_of "$pid")"
    ours="否"
    if is_our_cmd "$cmd"; then
      ours="是（本服务）"
    fi
    echo "  pid=${pid} 本服务=${ours}"
    echo "    cmd=${cmd:-<unknown>}"
  done <<< "$pids"
}

stop_pid() {
  local pid="$1"
  local cmd
  cmd="$(cmdline_of "$pid")"
  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  echo "SIGTERM pid=${pid} ${cmd:0:120}"
  kill -TERM "$pid" 2>/dev/null || true
}

force_kill_pid() {
  local pid="$1"
  if kill -0 "$pid" 2>/dev/null; then
    echo "SIGKILL pid=${pid}"
    kill -KILL "$pid" 2>/dev/null || true
  fi
}

wait_port_free() {
  local deadline=$((SECONDS + WAIT_STOP_SEC))
  while (( SECONDS < deadline )); do
    local left
    left="$(listener_pids "$PORT" || true)"
    if [[ -z "${left}" ]]; then
      return 0
    fi
    sleep 0.2
  done
  return 1
}

reclaim_port() {
  local pids pid cmd
  pids="$(listener_pids "$PORT" || true)"
  if [[ -z "${pids}" ]]; then
    echo "端口 ${PORT} 已空闲"
    return 0
  fi

  local ours=()
  local foreign=()
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    # 不要杀自己（一般不会，但防御）
    if [[ "$pid" == "$$" ]]; then
      continue
    fi
    cmd="$(cmdline_of "$pid")"
    if is_our_cmd "$cmd"; then
      ours+=("$pid")
    else
      foreign+=("$pid")
    fi
  done <<< "$pids"

  if ((${#foreign[@]} > 0)) && [[ "$FORCE" -ne 1 ]]; then
    echo "错误：端口 ${PORT} 被非本服务进程占用，拒绝自动杀掉（加 --force 才强制）：" >&2
    for pid in "${foreign[@]}"; do
      echo "  pid=${pid} cmd=$(cmdline_of "$pid")" >&2
    done
    return 1
  fi

  local targets=("${ours[@]}")
  if [[ "$FORCE" -eq 1 ]]; then
    targets+=("${foreign[@]}")
  fi
  if ((${#targets[@]} == 0)); then
    echo "端口 ${PORT} 上没有可停止的进程"
    return 0
  fi

  for pid in "${targets[@]}"; do
    stop_pid "$pid"
  done

  if wait_port_free; then
    echo "端口 ${PORT} 已释放"
    return 0
  fi

  echo "优雅退出超时，改用 SIGKILL…"
  for pid in "${targets[@]}"; do
    force_kill_pid "$pid"
  done
  sleep 0.3
  if wait_port_free; then
    echo "端口 ${PORT} 已释放"
    return 0
  fi
  echo "错误：端口 ${PORT} 仍被占用：$(listener_pids "$PORT" | tr '\n' ' ')" >&2
  return 1
}

# 也清理我们自己写的 pid 文件里已死 / 仍活的旧进程
stop_pidfile() {
  if [[ ! -f "$PID_FILE" ]]; then
    return 0
  fi
  local old
  old="$(tr -d '[:space:]' < "$PID_FILE" || true)"
  if [[ -n "$old" && "$old" =~ ^[0-9]+$ ]]; then
    if kill -0 "$old" 2>/dev/null; then
      local cmd
      cmd="$(cmdline_of "$old")"
      if is_our_cmd "$cmd" || [[ "$FORCE" -eq 1 ]]; then
        stop_pid "$old"
      else
        echo "警告：pid 文件 ${PID_FILE} 指向非本服务进程 pid=${old}，跳过（可用 --force）"
      fi
    fi
  fi
  rm -f "$PID_FILE"
}

wait_ready() {
  local deadline=$((SECONDS + WAIT_READY_SEC))
  local url="http://${HOST}:${PORT}/healthz"
  while (( SECONDS < deadline )); do
    if command -v curl >/dev/null 2>&1; then
      if curl -fsS --max-time 1 "$url" >/dev/null 2>&1; then
        echo "健康检查通过：${url}"
        return 0
      fi
    else
      # 无 curl 时只确认端口已 LISTEN
      if [[ -n "$(listener_pids "$PORT" || true)" ]]; then
        echo "端口 ${PORT} 已监听（无 curl，跳过 /healthz）"
        return 0
      fi
    fi
    sleep 0.3
  done
  echo "错误：${WAIT_READY_SEC}s 内服务未就绪（${url}）" >&2
  return 1
}

do_stop() {
  echo "==> 停止 rollout-insight（port=${PORT}）"
  stop_pidfile || true
  reclaim_port
}

do_start() {
  echo "==> 启动 rollout-insight（port=${PORT}，python=${PYTHON}）"
  if [[ -n "$(listener_pids "$PORT" || true)" ]]; then
    echo "启动前发现端口仍占用，先释放…"
    reclaim_port
  fi

  mkdir -p "$LOG_DIR"
  export ROLLOUT_INSIGHT_PORT="$PORT"

  if [[ "$BACKGROUND" -eq 1 ]]; then
    # 后台：nohup + pid 文件；stdout/stderr 进 log/server.log
    nohup "$PYTHON" main.py >>"$LOG_FILE" 2>&1 &
    local new_pid=$!
    echo "$new_pid" >"$PID_FILE"
    echo "已后台启动 pid=${new_pid}，日志：${LOG_FILE}"
    wait_ready
    print_holders
  else
    echo "前台启动（Ctrl+C 结束）。后台请加 --bg"
    # 前台：若端口仍冲突，uvicorn 会立刻失败；已在上面 reclaim
    exec "$PYTHON" main.py
  fi
}

do_status() {
  echo "仓库：${REPO_ROOT}"
  echo "Python：${PYTHON}"
  print_holders
  if [[ -f "$PID_FILE" ]]; then
    echo "pid 文件：${PID_FILE} -> $(tr -d '[:space:]' < "$PID_FILE")"
  else
    echo "pid 文件：无"
  fi
  if command -v curl >/dev/null 2>&1; then
    if curl -fsS --max-time 1 "http://${HOST}:${PORT}/healthz" >/dev/null 2>&1; then
      echo "healthz：ok"
    else
      echo "healthz：不可达"
    fi
  fi
}

# ---- 参数 ----
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --bg|--background) BACKGROUND=1; shift ;;
    --force) FORCE=1; shift ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --port=*)
      PORT="${1#*=}"
      shift
      ;;
    start|stop|restart|status)
      CMD="$1"
      shift
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if ((${#POSITIONAL[@]} > 0)); then
  echo "未知参数：${POSITIONAL[*]}" >&2
  usage >&2
  exit 2
fi

case "$CMD" in
  status) do_status ;;
  stop) do_stop ;;
  start) do_start ;;
  restart)
    do_stop
    do_start
    ;;
  *)
    echo "未知命令：${CMD}" >&2
    usage >&2
    exit 2
    ;;
esac
