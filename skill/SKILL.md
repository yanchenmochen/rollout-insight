---
name: rollout-insight
description: >-
  通过 rollout-insight CLI 操作本地 rollout-insight Web 服务：列出/新建/重扫/删除任务、检索与查看 segment、调用南湖（nanhu）作业（jobs / restart / link）、配置 tokenizer。
  集群默认 wuxi，可切 x10000。写操作默认 dry-run，加 --submit 才执行。
  在用户提到 rollout-insight、rollout 任务、segment transcript、南湖 job 关联、tokenizer 路径、rollout dump 扫描、查看 agent 对话记录时使用。
---

# rollout-insight CLI / Skill

`rollout-insight` 是 Slime rollout 落盘（`.pt`）的解析与作业洞察工具。v0.0–v0.2 已落 FastAPI + SQLite + 任务/Segment ORM；v0.3.0 起配套 CLI 与本 skill。

本 skill 只描述**怎么用 CLI**完成常见任务；实现细节见仓库根目录的 `README.md`。

## 何时使用

- 列/建/重扫/删除 rollout 任务
- 检索 segment（按 reward / keyword / task 过滤）、查看 segment transcript
- 查/重启南湖（nanhu）作业；把南湖 job 绑定到本地 task
- 修改 / 验证 config.json 的 tokenizer_dir

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