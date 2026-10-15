---
name: rollout-insight
description: >-
  通过 rollout-insight CLI 操作本地 rollout-insight Web 服务：列出/新建/重扫/删除任务、检索与查看 segment、调用南湖（nanhu）作业（jobs / restart / link）、配置 tokenizer；v0.4.0 新增 rollout 详情/解析/列表（rollouts）与 agent-transcripts 报告（agent transcript 报告 / 隐式 reward / cursor transcript 分析）。
  集群默认 wuxi，可切 x10000。写操作默认 dry-run，加 --submit 才执行。
  在用户提到 rollout-insight、rollout 任务、segment transcript、rollout 解析、.pt 字段分布、.pt 字段含义、.pt 数据结构、.pt 转 JSON、.pt 转 markdown、.pt 摘要、.pt 摘要统计、.pt 字段摘要、.pt 字段解释、.pt 字段说明、南湖 job 关联、tokenizer 路径、rollout dump 扫描、查看 agent 对话记录、agent transcript 报告、隐式 reward、cursor transcript 分析时使用。
---

# rollout-insight CLI / Skill

`rollout-insight` 是 Slime rollout 落盘（`.pt`）的解析与作业洞察工具。v0.0–v0.2 已落 FastAPI + SQLite + 任务/Segment ORM；v0.3.0 起配套 CLI 与本 skill。

本 skill 只描述**怎么用 CLI**完成常见任务；实现细节见仓库根目录的 `README.md`。

## 改前先读 func.md / 改后重生成 func.md

> 本节自 **v0.5.2** 起生效。`func.md` 是仓库根目录的**自动生成**能力索引，覆盖
> `controller/`、`service/`、`db/`、`utils/`、`data/` 五个目录里所有公开符号。
> 是 agent / skill 查找"现有可复用实现"的第一站。

- **新增 / 修改** `controller/`、`service/`、`db/`、`utils/`、`data/` 下的
  公开能力**前**，先 **Read** 仓库根目录的 `func.md`，确认是否已有可复用
  实现（同名函数、相似职责的 service 等）。**禁止**在复制粘贴已有逻辑后
  重写一个并行实现。
- **新增 / 修改后**，必须重跑：

  ```bash
  python scripts/scan_func_md.py
  ```

  重新生成 `func.md` 并与本次改动一起 commit。CI / 自动化 agent 在 commit
  前会跑 `python scripts/scan_func_md.py --check`，不一致会失败。
- 当改动只动了内部实现细节（私有函数、内部重组、注释）时，可不重跑；
  但建议养成"改动落定就跑一次"的习惯，避免与他人改动交叉时出现 drift。

## 何时使用

- 列/建/重扫/删除 rollout 任务
- 检索 segment（按 reward / keyword / task 过滤）、查看 segment transcript
- 查/重启南湖（nanhu）作业；把南湖 job 绑定到本地 task
- 修改 / 验证 config.json 的 tokenizer_dir
- **v0.4.0** 看单个 rollout 详情（pt_path / n_samples / mean_reward / attempts 摘要）
- **v0.4.0** 触发 `.pt` 重解析（idempotent）并写 turn 表
- **v0.4.0** 列 task 下的 rollouts（fallback 到 task 摘要）
- **v0.4.0** 看 agent-transcripts 报告（turn 折线 / 隐式 reward / token 分布）

## 命令树（v0.4.0）

```text
rollout-insight [--api-base URL] [--format json|table] [--pretty] [--cluster wuxi|x10000]
  tasks {list,create,rescan,delete}
  segments {list,show,search}
  nanhu {jobs,restart,link}
  tokenizer {set,test}
  rollouts {show,parse,list}                # v0.4.0
  agent-transcripts {report}                # v0.4.0
```

## 前置条件

```bash
# 1. 装包（用户手动执行）
cd /path/to/rollout-insight
pip install -e .

# 2. 安装 skill 软链（用户手动执行；脚本见 skill/install.sh）
bash skill/install.sh

# 3. 启动本地 Web 服务（用户手动执行）
python main.py              # 默认端口 39998
```

> Web API base 默认 `http://localhost:39998`，可走 `--api-base` 或 `ROLLOUT_INSIGHT_API_BASE` 覆盖。

## 全局参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--api-base URL` | `http://localhost:39998` | 本地 Web API |
| `--format {json,table}` | `json` | 输出格式；table 走 markdown 风格表格 |
| `--pretty` | `false` | JSON 是否 indent |
| `--cluster {wuxi,x10000}` | `wuxi` | 南湖集群 |

**写操作默认 dry-run；必须加 `--submit` 才执行。**

## 高频配方

### 任务（tasks）

```bash
# 列任务（带模糊匹配）
rollout-insight --format table tasks list
rollout-insight tasks list --task-name slime --status PENDING

# 新建任务（dry-run）
rollout-insight tasks create \
  --task-name slime-rollout-021 \
  --rollout-root /mnt/zj-data/data/songquanheng/output/slime/021_xxx \
  --tokenizer-dir /mnt/zj-data/models/public/models/021-32B-Agentic-hf \
  --recursive

# 确认无误 → 加 --submit 才落库
rollout-insight tasks create --task-name ... --rollout-root ... --submit

# 重扫任务（触发 RolloutScanService）
rollout-insight tasks rescan --task-id 1
rollout-insight tasks rescan --task-id 1 --submit

# 删除任务（级联删除 rollout/attempt/segment）
rollout-insight tasks delete --task-id 1
rollout-insight tasks delete --task-id 1 --submit
```

### Segment（segments）

```bash
# 列出某任务的 segment（默认走 /api/segments/search）
rollout-insight --format table segments list --task-id 1
rollout-insight segments list --task-id 1 --min-reward 0.8 --keyword sandbox-id-xyz

# 检索（与 list 同义，参数完全相同；为方便 agent 用 'search' 习惯命名）
rollout-insight segments search --task-id 1 --min-reward 0.5

# 查看某 segment transcript（调 /api/segments/{id}/transcript.json）
rollout-insight segments show --segment-id 42
```

### 南湖（nanhu）

透传 `nanhu` / `nanhu-cli`（来自 `nanhu-skill` 项目），CLI 不解析 stdout，原样输出。

```bash
# 列运行中作业（一次拉齐训练+部署）
rollout-insight --cluster wuxi --format table nanhu jobs
rollout-insight --cluster wuxi --format table nanhu jobs --status running

# 仅训练 / 仅部署
rollout-insight --cluster wuxi nanhu jobs --job-type AI_TRAIN
rollout-insight --cluster wuxi nanhu jobs --job-type AI_ONLINE_INFERENCE_SERVICE

# 重启作业（dry-run → submit）
rollout-insight --cluster wuxi nanhu restart --job-id jb-xxx
rollout-insight --cluster wuxi nanhu restart --job-id jb-xxx --submit

# 把南湖 job 绑到本地 task（dry-run）
rollout-insight nanhu link --job-id jb-xxx --task-name slime-rollout-021
```

### Tokenizer

```bash
# 验证路径是否是合法 HF tokenizer 目录
rollout-insight tokenizer test --path /mnt/zj-data/models/public/021-32B-Agentic-hf

# 写 config.json 的 tokenizer_dir（dry-run → submit）
rollout-insight tokenizer set --path /mnt/zj-data/models/public/021-32B-Agentic-hf
rollout-insight tokenizer set --path /mnt/zj-data/models/public/021-32B-Agentic-hf --submit
```

### Rollouts（v0.4.0）

```bash
# 看单个 rollout 概览（pt_path / n_samples / n_attempts / mean_reward / attempts 摘要）
# → GET /api/rollouts/{id}/json（404 时回退 /api/rollouts/{id}）
rollout-insight rollouts show 87

# 触发 .pt 重解析（idempotent；可选 --tokenizer 备用）
# → POST /api/rollouts/{id}/parse
rollout-insight rollouts parse 87 --tokenizer /mnt/zj-data/models/public/021-32B-Agentic-hf

# 列 task 下的 rollouts（v0.4.0 controller 暂无 JSON list 端点，CLI 自动 fallback 到 task 摘要）
# → GET /api/tasks/{id}/rollouts（404 → /api/tasks）
rollout-insight rollouts list --task 1
```

> **写操作约定**：`rollouts parse` 自身 idempotent + 用户主动触发，所以**不加 `--submit` 也默认打印 plan**；带 `--submit` 才真触发。CLI 不会重复写库。

### Agent-Transcripts（v0.4.0 T6 配套）

```bash
# 自动判别：file → /api/agent-transcripts/report?path=...；dir → /api/agent-transcripts/batch?dir=...
# 单文件报告（user 反馈 0/1 + 字符/token 分布 + top 5 最长 turn）
rollout-insight agent-transcripts report /root/.cursor/projects/.../xxx.jsonl

# 批量报告（递归扫目录下所有 .jsonl，给每个出 token/reward 摘要）
rollout-insight agent-transcripts report /root/.cursor/projects/.../agent-transcripts/
```

报告字段：
- `total_events / role_distribution / assistant_turn_count / user_turn_count / tool_call_count`
- `char_distribution`（turn 字符数 + token 估算 + 直方图）
- `implicit_rewards`（user 反馈 0/1 启发式 → mean_reward）
- `top_longest_turns`（最长的 5 个 turn + 预览）

## 对人回复模板

**列表类**（tasks / segments / nanhu jobs）：用 markdown 表，无结果明确写「**当前无匹配记录**」。不要把整页 JSON 贴给用户。

```markdown
### 任务列表（N）
| id | task_name | status | nanhu_job_id | rollout_root |
|----|-----------|--------|--------------|--------------|
| 1  | slime-021 | RUNNING| jb-xxx       | /mnt/...     |
```

**写操作类**：永远先打印 dry-run 摘要、用户确认后再 `--submit`。dry-run 字段示例：

```json
{"action":"create task","api_base":"http://localhost:39998","task_name":"...","rollout_root":"...","submit":false,"dry_run":true,"note":"dry-run：未提交，加 --submit 才执行"}
```

## Agent 约定

1. 读操作直接跑；列表对人用 `--format table`。
2. 写操作必须先 dry-run，等用户确认后再 `--submit`。
3. 不要把南湖 CLI 的 stdout JSON 二次格式化；CLI 已透传 nanhu-cli 输出。
4. Web API 不可达时（`[Errno 111] Connection refused`）→ 提示用户先 `python main.py`。
5. nanhu-cli 找不到 → 提示用户先 `pip install -e /path/to/nanhu-skill`。

## 详细文档

- 命令/参数表：`skill/recipes.md`
- 安装步骤：`skill/install.sh`
- Web API 全集：见仓库 `controller/` 与 `docs/`（v0.2 阶段补齐）
- v0.4.0 计划：根目录 `rollout-insight_v0.4.0_52e49d78.plan.md`（T7→T8 串行链对应本 skill 升级）

## 维护 func.md

> 本节自 **v0.5.2** 起生效。

`func.md` 是仓库根目录的**自动生成**索引文件，**禁止手改**（任何手改都会被下
次扫描覆盖）。它的内容是 `python scripts/scan_func_md.py` 对全仓库
`controller/`、`service/`、`db/`、`utils/`、`data/` 五个目录的扫描结果。

### 工作流

```bash
# 1) 改动 controller / service / db / utils / data 下公开能力
# 2) 跑扫描，重新生成 func.md
python scripts/scan_func_md.py

# 3) 跟代码改动一起 commit（必须）
git add scripts/scan_func_md.py func.md
git commit -m "feat(...): 新增 XXX 功能 + 重生成 func.md"

# 4) CI 阶段的门禁（不许绕过）
python scripts/scan_func_md.py --check   # 退出码 0 才允许合并
```

### 常见问题

- **`--check` 失败：func.md 与扫描结果不一致** → 说明上次 commit 忘了重生成；
  跑 `python scripts/scan_func_md.py` 重新写一次 func.md 再 commit。
- **新增函数未出现在 func.md** → 检查是否在 `__pycache__/ log/ tests/ schedule/`
  等被黑名单命名的目录里（plan p18_a）；或函数是否以下划线开头（私有函数不收录）。
- **想忽略某个目录** → 在 `scripts/scan_func_md.py` 的 `SKIP_DIR_NAMES` /
  `SKIP_DIR_PATTERNS` 里加 pattern；同时更新本节"维护 func.md"以备审计。
- **想验证 CI 阶段命令** → 故意往 `utils/xxx.py` 加个新公共函数，重跑
  `--check`，应该看到退出码 1 与 diff 提示。