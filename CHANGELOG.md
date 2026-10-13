# rollout-insight CHANGELOG

## v0.4.0 — 树状浏览 + .pt 深度解析 + agent-transcripts 报告

发布日期：2026-10-13

### 概述

v0.4.0 解决首页 task_list 渲染 bug；补齐「任务 → rollout → segment → turn」三级树状浏览；
接入 `.pt` 深度解析（按 `inspect-rollout-trajectory` 规则：`(group_index, rollout_id)` 复合键
为一个 attempt，每个 attempt 含 N 个 sibling segment，reward 在 sample 级）；并对 agent-transcripts
做 token / turn / 隐式 reward 分布分析。

### 完成情况（按子任务）

| 子任务 | 内容 | commit |
| --- | --- | --- |
| T1 | 修首页 task_list 渲染（删除 `/` redirect + `/api/tasks` Accept 头分流） | `b046441` |
| T2 | rollout 详情页 controller + 模板（`/api/rollouts/{id}` JSON + `/api/tasks/{tid}/rollouts/{rid}` HTML） | `c958823` |
| T3 | `.pt` 解析 utils + service（按 inspect 规则：attempt=(group_index,rollout_id)） | `53e55bc` |
| T4 | 新表 `ri_segment_turn_table`（13 列）+ ORM + Service | `ddcb978` |
| T5 | segment → turn 三级树状手风琴视图（`<details>` 每 turn 一个，含 has_think/has_tool_call 图标） | `f3b9a20` |
| T6 | agent-transcripts token / turn / 隐式 reward 分布报告 + 报告页 | `758b36b` |
| T7 | CLI 新子命令 `rollouts` (show/parse/list) + `agent-transcripts` (report) | `ad8b0d3` |
| T8 | Skill `SKILL.md` + `recipes.md` 更新（v0.4.0 触发关键词 + 4 个 recipe） | `d04bc33` |

### 文件变更（vs v0.3.0）

#### 新建文件（10 个）

- `controller/agent_transcript_controller.py`
- `controller/rollout_controller.py`
- `db/db_segment_turn.py`
- `db/db_segment_turn_service.py`
- `service/agent_transcript_analyzer.py`
- `service/rollout_parse_service.py`
- `templates/agent_transcript_report.html`
- `templates/rollout_detail.html`
- `tests/test_agent_transcript_analyzer.py`
- `utils/rollout_pt_parser.py`

#### 修改文件（15 个）

- `.gitignore`
- `cli/api_client.py`
- `cli/formatters.py`
- `cli/rollout_insight.py`
- `controller/nanhu_controller.py`
- `controller/segment_controller.py`
- `controller/task_controller.py`
- `db/__init__.py`
- `db/db_task.py`
- `main.py`
- `service/rollout_scan_service.py`
- `skill/SKILL.md`
- `skill/recipes.md`
- `templates/segment_transcript.html`
- `templates/task_list.html`

### 升级指南

1. **重装 CLI（拿到新子命令）**

   ```bash
   cd /path/to/rollout-insight
   pip install -e .
   rollout-insight --help   # 应包含 rollouts / agent-transcripts
   ```

2. **更新 Skill（拿到新触发关键词与 recipes）**

   ```bash
   bash skill/install.sh
   # 或强制覆盖：OVERRIDE=1 bash skill/install.sh
   ```

3. **数据库迁移**

   - SQLite 文件：`data/rollout_insight.db`
   - 新表 `ri_segment_turn_table` 会在下次 `Base.metadata.create_all(engine)` 时自动建（v0.4.0 启动触发）
   - 旧 task 数据保留不破坏；turn 数据需用户主动触发 T3 `parse_rollout` 才落库
   - 如果是从 v0.3 直接升级，建议先做一次 `rescan` 让 `ri_rollout_table.n_samples / n_attempts / instance_count`
     字段刷新（v0.4.0 显示依赖）

4. **环境变量（无变化）**

   - `ROLLOUT_INSIGHT_PORT`（默认 39998）
   - `ROLLOUT_INSIGHT_API_BASE`（CLI 用，默认 `http://localhost:39998`）

### 新功能速查

- **首页**：表格化 task 列表，支持 `status` 过滤与 `task_name` 模糊匹配
- **rollout 详情页**：面包屑（任务 > rollout）+ 概览卡 + attempts 列表（含每 attempt 的 sibling segment 数 + 平均 reward）+ 「re-parse .pt」按钮
- **segment 详情页**：新增手风琴 turn 视图（`<details>` 每 turn 一个），4 个统计 chip（turn_count / total_chars / token_estimate / time_cost），上一/下一 segment 导航
- **agent-transcripts 报告**：token 折线、turn 分布、隐式 reward 走势、top-5 最长 turn
- **CLI 新子命令**：
  - `rollout-insight rollouts show <primary_id>` — 单个 rollout 概览
  - `rollout-insight rollouts parse <primary_id> [--tokenizer DIR]` — 强制重解析并落 turn 表
  - `rollout-insight rollouts list --task <task_id>` — 列 task 下的 rollouts（回退到 task 摘要，详见已知问题）
  - `rollout-insight agent-transcripts report <file_or_dir>` — token/reward 报告

### 已知问题（v0.4.0 不修，留 v0.5+ 处理）

1. **attempt_id 与 per-sample rollout_id 的关联错位**

   v0.2 数据建模时 `ri_attempt_table.attempt_id` 是被简化的序号（不是 `.pt` 中 `sample.rollout_id` 的真实业务键），
   而 `.pt` dump 中每个 sample 的 `rollout_id` 是真实业务键（如 784/785/...）。`ri_rollout_table.rollout_id`
   存的是 dump 的序号（10/11/...）。`_persist_turns` 通过 `(rollout_id, group_index, instance_id, segment_index)`
   关联 `ri_segment_table` 时常找不到匹配行（典型日志：`跳过落库`）。

   影响：`parse_rollout` 返回的 `n_attempts / n_segments` 正确，但 `persisted_turns=0`，
   导致 T5 segment → turn 手风琴视图展示为空（除非人工按 `.pt` 真实 `rollout_id` 重建 attempt）。

   解决方案（v0.5+）：扫描时同时记录 `(group_index, instance_id, segment_index, sample.rollout_id)` 到 attempt 表或新增
   `ri_sample_key_table`，让 `parse_rollout` 可精确反查。

2. **`rollouts list` CLI JSON 端点不存在**

   `controller/rollout_controller.py` 只注册了 `/api/tasks/{id}/rollouts/{rid}`（HTML）和 `/api/rollouts/{id}`（JSON），
   但 `/api/tasks/{id}/rollouts`（JSON 列表）未注册。CLI 当前会回退到 task 摘要 + hint 提示。

   解决方案（v0.4.1+）：加 `GET /api/tasks/{id}/rollouts` 返回 rollout 概览列表。

3. **segment `response_text` 未持久化到 turn 表**

   当前 `ri_segment_turn_table` 只存 `body_text` 与 `prompt_summary`，不存整个 `response_text`。
   重解析时如果 DB 已有 `turns` 行，新 turn 会覆盖旧值但 `response_text` 仍是懒加载路径。

   解决方案（v0.5+）：新增 `ri_segment_response_table` 单独存 `response_text` 大字段，按需 fetch。

### 端到端验证（T9）

- 服务启动 OK，`/healthz` 返回 200
- T1 首页：`/` 返回 HTML 含 `<table>`，`/api/tasks` 返回 2 个任务，`?status=READY` 过滤 OK
- T2 rollout 详情：`/api/rollouts/106` 返回 JSON（attempts 数组）、`/api/tasks/1/rollouts/106` 返回 HTML
- T3/T4 `.pt` 解析：`parse_attempts_and_segments` 返回 attempts=64 / segments=202；
  `ri_segment_turn_table` 表 schema 13 列；`parse_rollout()` 返回 `n_attempts / n_segments / persisted_turns`
- T5 segment 视图：`/api/segments/16829` 返回 200；turn 表为空时 `<details>` 数为 0（受已知问题 1 影响）
- T6 agent-transcripts：`/api/agent-transcripts/page` 200；`/api/agent-transcripts/report?path=...` 200 并返回统计；
  `/api/agent-transcripts/batch?dir=...` 200
- T7 CLI：`rollout-insight --help` 出现 `rollouts` 与 `agent-transcripts`；`rollouts show` 输出 JSON；
  `agent-transcripts report <dir>` 输出 reports 数组（count=80）；`tasks list` 不破坏 v0.3
- T8 Skill：`/root/.cursor/skills/rollout-insight` 软链存在；`SKILL.md` 含 `rollouts` 和 `agent-transcripts` 关键词；
  `recipes.md` 存在且含 21 处 `rollouts|agent-transcripts` 关键词

---

## v0.3.0 — 南湖集成 + CLI + Skill

- 南湖（nanhu）调度接入：nanhu 客户端 + 2 张 ORM 表（`ri_nanhu_job_table` / `ri_nanhu_restart_table`）
- rollout-insight CLI（`tasks` / `segments` / `nanhu` / `tokenizer`）
- Cursor Skill 安装脚本与 SKILL.md
