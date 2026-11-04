# rollout-insight CLI 高频配方（recipes）

> 完整命令树见 `SKILL.md`。本文档是"按场景"的速查手册。

## 全局约定

- 默认 base URL：`http://localhost:39998`（可 `--api-base` 或 `ROLLOUT_INSIGHT_API_BASE`）
- 默认南湖集群：`wuxi`（可 `--cluster x10000`）
- 默认输出格式：`json`（列表场景对人用 `--format table`）
- **写操作默认 dry-run；加 `--submit` 才执行**
- 退出码：`0` 成功；`1` 业务/API 错误；`2` 命令行参数错误；`130` Ctrl-C

## 场景 0：新增 controller / service / db 表 / utils 函数的标准流程

> 本节自 **v0.5.2** 起生效。目标：在新增或修改仓库公开能力时，确保
> `func.md` 同步更新，避免 agent / 后续作者陷入"重复造轮子"。

### 三步走

```text
1. Read func.md 确认无重复实现
2. 实现 controller/service/db/utils/data 代码
3. python scripts/scan_func_md.py  →  func.md 跟着本次 commit
```

### 第 1 步：改前先读 func.md

```bash
# 在仓库根目录
cat func.md
# 或者按目录分段读：
sed -n '/^## controller/,/^## /p' func.md
sed -n '/^## service/,/^## /p' func.md
sed -n '/^## db\/ — 数据访问服务/,/^## /p' func.md
```

判断：

- 同名或近名函数已存在？→ 直接复用，不要新写一个 `xxx_v2`。
- 类似职责的 `DBXxxService` 已有方法？→ 在原 service 上加方法。
- 已有 controller 端点覆盖？→ 不要平行加新端点，先复用或扩字段。

### 第 2 步：按既有约定实现

参考仓库根目录 `.cursor/rules/rollout-insight-conventions.mdc` 与既有代码：

- **ORM**：`db/db_<领域>.py`，类名 `DBXxx`，表名 `ri_<语义>_table`。
- **数据访问**：`db/db_<领域>_service.py`，类名 `DBXxxService`，`@staticmethod` + `with Session()`。
- **业务编排**：`service/<领域>_service.py`（或模块级函数，二者皆可）。
- **路由**：`controller/<领域>_controller.py`，统一返回
  `data.response.Response.success(...)` / `Response.error(...)`，业务错误
  用 `RolloutInsightException`。
- **工具**：`utils/<用途>.py`，模块级函数，禁依赖 db/route。

### 第 3 步：重跑扫描 + commit

```bash
# 重生成 func.md
python scripts/scan_func_md.py

# 验证一致性（CI 也会跑这一步；本机也跑一次心里有底）
python scripts/scan_func_md.py --check
# 期望输出：[scan_func_md] --check OK ...

# 跟代码改动一起 commit（必须把 func.md 一起带上）
git add scripts/scan_func_md.py func.md <你自己的改动>
git commit -m "feat(...): 新增 XXX — 重生成 func.md"
```

> **重要**：`func.md` 必须与代码同步落在同一个 commit，否则 CI 会失败；
> 如果只是内部实现细节改了（私有函数、注释、refactor），可不带 func.md。

### 反例（被 reviewer 打回）

| 场景 | 反例 | 应改 |
|------|------|------|
| 新表 | 在 `db/__init__.py` 直接写 Column | 新建 `db/db_<领域>.py` 与 `db/db_<领域>_service.py`；再跑 `scan_func_md.py` |
| 新路由 | controller 里直接写裸 dict 返回 | 用 `Response.success(...)`；检查 `func.md` §controller 段确认无重复 |
| 新工具函数 | 在 controller 内 inline 写一个文件解析 | 抽到 `utils/parse_xxx.py`；并确保 `func.md` §utils 段录到 |
| 漏改 func.md | 只 commit 了代码，func.md 没动 | 跑 `scan_func_md.py`，把输出 diff 一起 commit |

---

## 场景 1：用户刚跑完一次训练，要扫进 rollout-insight

```bash
# 1) 验证 rollout 根目录结构
ls /mnt/zj-data/.../output/slime/<exp>/rollout_dumps/

# 2) 验证 tokenizer 路径
rollout-insight tokenizer test --path /mnt/zj-data/models/.../hf

# 3) 写 tokenizer_dir 到 config.json（dry-run → 确认 → submit）
rollout-insight tokenizer set --path /mnt/zj-data/models/.../hf
rollout-insight tokenizer set --path /mnt/zj-data/models/.../hf --submit

# 4) 建任务（dry-run → 确认 → submit）
rollout-insight tasks create \
  --task-name slime-021-rollout \
  --rollout-root /mnt/zj-data/.../output/slime/<exp>/rollout_dumps \
  --recursive
rollout-insight tasks create \
  --task-name slime-021-rollout \
  --rollout-root /mnt/zj-data/.../output/slime/<exp>/rollout_dumps \
  --recursive --submit

# 5) 等 service 把扫描完，列任务 / 检视 segment
rollout-insight --format table tasks list --task-name slime-021-rollout
rollout-insight --format table segments list --task-id 1
```

## 场景 2：从南湖 job 一键创建 / 关联 rollout 任务

Web：`/api/tasks/create` → 来源类型选「南湖作业」→ 填 `job_id` → **解析**
（回填任务名 / Rollout 根路径 / tokenizer）→ 创建。

解析规则：`ROLLOUT_DUMP_TEMPLATE` 目录优先；否则 `OUTPUT_DIR/generated/rollout_dumps`；
tokenizer 取 `HF_CHECKPOINT`。API：

```bash
curl -s "http://localhost:39998/api/nanhu/jobs/jb-xxx/resolve-for-task?cluster=x10000"
```

CLI 手工路径（仍可用）：

```bash
# 1) 列出用户的所有运行中训练
rollout-insight --cluster x10000 --format table nanhu jobs --job-type AI_TRAIN

# 2) 取 job id 后看 detail（找出 OUTPUT_DIR / ROLLOUT_DUMP_TEMPLATE）
rollout-insight --cluster x10000 nanhu jobs detail --job-id jb-xxx | head -c 4k

# 3) 把南湖 job 绑到本地 task
rollout-insight nanhu link --job-id jb-xxx --task-name slime-021-rollout
rollout-insight nanhu link --job-id jb-xxx --task-name slime-021-rollout --submit

# 4) 如任务失败：重启南湖 job
rollout-insight --cluster x10000 nanhu restart --job-id jb-xxx
rollout-insight --cluster x10000 nanhu restart --job-id jb-xxx --submit
```

## 场景 3：定位一条失败 segment 的 tool_calls

```bash
# 1) 找低 reward 的 segment
rollout-insight --format table segments search --task-id 1 --min-reward 0 --page-size 100

# 2) 按 sandbox_id 模糊匹配（agent 死循环定位常用）
rollout-insight segments list --task-id 1 --keyword sb-abc123

# 3) 查看 transcript
rollout-insight segments show --segment-id 42
```

## 场景 4：批量重扫

```bash
# 列出所有 RUNNING / FAILED 任务，挨个重扫
for tid in $(rollout-insight tasks list --status FAILED | python -c "
import json, sys
d = json.load(sys.stdin)
print(' '.join(str(t['id']) for t in d['data']['tasks']))
"); do
    echo "重扫 task_id=$tid"
    rollout-insight tasks rescan --task-id "$tid" --submit
done
```

## 场景 5：清理废任务

```bash
# 先 dry-run 看目标
rollout-insight tasks delete --task-id 99

# 确认 → submit（级联删除其 rollout/attempt/segment）
rollout-insight tasks delete --task-id 99 --submit
```

## 输出格式差异

- **`--format json`（默认）**：紧凑 JSON，适合 agent 解析。
- **`--format table`**：把 list-of-dict 渲染成 markdown 表，每列限宽 48 字符。
  单 dict 用「(unprintable)」占位，所以非列表场景（如 segments show）table 退化为 JSON。

## 排障

| 症状 | 原因 | 行动 |
|------|------|------|
| `无法连接 http://localhost:39998/...` | Web service 没起 | `python main.py` 后重试 |
| `找不到 nanhu / nanhu-cli 可执行文件` | nanhu-skill 未装 | `pip install -e /path/to/nanhu-skill` |
| `nanhu jobs list: error: --job-type required` | nanhu-skill 的 list 要求 job_type，或**不**走 list 改用 `running` | 留空 `--job-type`，自动走 `running` |
| `segments search: INVALID_ARGUMENT 当前阶段 2C 仅支持 task_id` | controller 限制 | 加 `--task-id <n>` |
| `--format table` 输出 `(empty)` | payload 不是 list-of-dict | 改 `--format json` 看数据结构 |
| `tokenizer test` 报 `未找到任何 tokenizer 关键文件` | 路径里没有 tokenizer.json 等 | 确认是 HF tokenizer 目录（`tokenizer.json` / `vocab.json` 等） |

---

# v0.4.0 新增 recipes

> 本节对应 v0.4.0 T7/T8：rollout 详情/解析/列表 + agent-transcripts 报告。CLI 完整命令树见 `SKILL.md` §命令树。

## 场景 6：看单个 rollout 详情（pt_path / n_samples / mean_reward / attempts 摘要）

**何时用**：

- 想知道某次 rollout 的 .pt 在哪、有多少 sample、mean_reward 多少
- 列出该 rollout 的 attempts（按 group_index 聚合，每 attempt 的 sandbox/instance/status/mean_reward）

**命令**：

```bash
rollout-insight rollouts show 87
```

**预期输出片段**（`--pretty` 模式）：

```json
{
  "pt_path": "/mnt/zj-data/.../rollout_dumps/rollout_10.pt",
  "mean_reward": 0.15346534653465346,
  "n_samples": 202,
  "n_attempts": 64,
  "instance_count": 64,
  "scanned_at": "2026-10-13 15:56:24",
  "attempts": [
    {
      "primary_id": 18285,
      "group_index": 98,
      "segment_count": 1,
      "mean_reward": 0.0,
      "sandbox_id": "bf70958a-...-a52174a0dfbc",
      "instance_id": "python-poetry_poetry-core_pr327",
      "status": "completed",
      "first_segment_id": 17643
    }
  ]
}
```

**相关端点**：

- `GET /api/rollouts/{id}/json`（plan 契约，404 时回退到 `/api/rollouts/{id}`）

## 场景 7：强制重解析 .pt 并写 turn 表

**何时用**：

- 怀疑 turn 表（`ri_segment_turn_table`）丢了某些 turn 或想刷新 token 估算
- 修了 T3 的解析 service 后想批量回填

**命令**：

```bash
# 不加 --submit 也会触发（parse 自身 idempotent + 用户主动操作），但建议先 dry-run
rollout-insight rollouts parse 87
rollout-insight rollouts parse 87 --tokenizer /mnt/zj-data/models/public/021-32B-Agentic-hf
```

**预期输出片段**：

```json
{
  "rollout_id": 87,
  "pt_path": "/mnt/zj-data/.../rollout_10.pt",
  "force": true,
  "status": "PENDING_PARSE"
}
```

> 后台解析是异步的：返回 `PENDING_PARSE` 表示已加入队列；想看进度去 web UI 重新刷新对应 rollout 详情页 / 查 turn 表行数。

**相关端点**：

- `POST /api/rollouts/{id}/parse?force=true`（body `{"tokenizer_dir": "..."}` 备用）

## 场景 8：列 task 下的 rollouts

**何时用**：

- 想批量看 task #N 下有多少 rollout，每个的均值 reward / 状态 / instance 数
- 决策要不要 rescan / delete 整个 task

**命令**：

```bash
rollout-insight rollouts list --task 1
```

**预期输出片段**（v0.4.0 controller **暂无** `GET /api/tasks/{id}/rollouts` JSON 端点，CLI 自动 fallback 到 `/api/tasks` 拿 task 摘要）：

```json
{
  "task_id": 1,
  "task_summary": {
    "task_name": "smoke-001",
    "status": "READY",
    "rollout_count": 36,
    "attempt_count": 6650,
    "segment_count": 6650,
    "mean_reward": 249,
    "rollout_root": "/mnt/zj-data/.../rollout_dumps"
  },
  "rollouts": [],
  "count": 0,
  "fallback": true,
  "hint": "v0.4.0 controller 未提供 GET /api/tasks/{id}/rollouts JSON 端点；如需逐个 rollout 概览，请用 'rollout-insight rollouts show <primary_id>'"
}
```

> 若 controller 后续补上 list 端点，CLI 会**自动**走真实 list（无需改 CLI）。

**相关端点**：

- `GET /api/tasks/{id}/rollouts?limit=...`（fallback → `GET /api/tasks`）

## 场景 9：出 agent-transcripts 批量报告（token + 隐式 reward）

**何时用**：

- 想知道一组 cursor 对话日志的 turn 分布、token 估算、用户反馈（隐式 reward）走势
- 评估 RL 训练数据里「真实用户接受/拒绝」的分布

**命令**：

```bash
# 目录：自动走 batch 端点，递归扫所有 .jsonl
rollout-insight agent-transcripts report /root/.cursor/projects/mnt-zj-data-data-songquanheng-slime/agent-transcripts

# 单文件：走 report 端点
rollout-insight agent-transcripts report /root/.cursor/projects/.../xxx.jsonl
```

**预期输出片段**（每个文件 1 行摘要）：

```json
{
  "reports": [
    {
      "path": "/root/.../08d589c6-...jsonl",
      "total_events": 70,
      "role_distribution": {"user": 10, "assistant": 51, "?": 9},
      "assistant_turn_count": 51,
      "user_turn_count": 10,
      "tool_call_count": 54,
      "char_distribution": {"mean_chars": 422.76, "max_chars": 3157, ...},
      "implicit_rewards": {"mean_reward": 0.55, "n_feedback": 10, "n_positive": 2, "n_negative": 1, "n_neutral": 7},
      "top_longest_turns": [{"turn_index": 46, "char_count": 3157, "preview": "那我直接给完整结论..."}]
    }
  ],
  "count": 79
}
```

**相关端点**：

- `GET /api/agent-transcripts/report?path=<file>`（单文件）
- `GET /api/agent-transcripts/batch?dir=<dir>`（批量；缺省走 controller 内置的 2 个默认文件）

---

## v0.4.0 排障追加

| 症状 | 原因 | 行动 |
|------|------|------|
| `rollouts show N` 输出 `{"code":51001,"msg":"rollout #N 不存在",...}` | N 不是 `ri_rollout_table.primary_id`（业务键是 `task_id+rollout_id`） | 改用 `rollouts list --task <task_id>` 拿 task 摘要 + 自己查 `db_rollout` |
| `rollouts list --task N` 显示 `"fallback": true` | v0.4.0 controller 暂未提供 `GET /api/tasks/{id}/rollouts` JSON 端点 | 已知；等 T2 worker 后续补端点，CLI 会自动走真实路径 |
| `rollouts parse N` 后 `status: PENDING_PARSE` 但页面没新 turn | 后台异步解析；可能 .pt 大或 sandbox 不可达 | 看 `log/server.log` 查 `parse_rollout` 是否报错；或重试 |
| `agent-transcripts report <path>` 输出 `null` | 路径不存在或 controller 端点变化 | 先 `ls` 确认路径；curl `http://localhost:39998/api/agent-transcripts/report?path=<path>` 排查 |