# Rollout Insight

> Slime rollout 落盘（`.pt`）的解析与作业洞察工具。

当前阶段：**v0.3.0 阶段 3C** —— CLI 包装层 + Cursor Skill 化。其他 worker 在做：service/（3A 后端）、db/、controller/、templates/（3B 表 UI）。本阶段仅在 `cli/`、`skill/` 目录落代码，**不**修改 service / db / controller / templates。

---

## 1. 功能概览（roadmap）

| 版本  | 内容 |
| ----- | ---- |
| v0.1.0 | 项目骨架：FastAPI 入口、CORS、StaticFiles、错误码、配置、日志、模板、SQLite 元数据 |
| v0.2.0 | 任务表（rollout/Job 元数据入库）、dump 扫描与摘要接口、Jinja 表 UI |
| v0.3.0 | 南湖（nanhu）作业调度集成（3A）、表 UI 增强（3B）、**CLI 包装层 + Cursor Skill（3C）** |

---

## 2. 运行环境

- Python ≥ 3.10
- 推荐使用 `venv` 或 `conda`
- 主要依赖见 `requirements.txt`

---

## 3. 快速开始

```bash
# 1. 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装依赖（**首次启动前手动执行**，CI 不自动安装以避免与并行 worker 冲突）
pip install -r requirements.txt

# 3. 配置（可选）
# 默认读取程序根目录下的 config.json；可通过环境变量覆盖：
#   export ROLLOUT_INSIGHT_CONFIG_PATH=/path/to/another_config.json
#   export ROLLOUT_INSIGHT_PORT=39998

# 4. 启动服务
python main.py
# 浏览器访问 http://<host>:39998/  → 自动跳转 /api/tasks
```

> 端口优先级：`$ROLLOUT_INSIGHT_PORT`（默认 `39998`）。

### 训练曲线（TensorBoard）

- 任务详情「训练曲线」可启动本机 TensorBoard，**新标签**打开（不嵌入 iframe）。
- 顶栏「曲线对比」→ `/api/tensorboard`：一键对比库内全部含 `tbs/` 或 `tf_logs/` 的任务。TensorBoard 左侧 Run 名即任务名（符号链接目录），可按任务名筛选对照。若默认端口上已有其他 TensorBoard，启动时会先停掉再绑定。启动参数含 `--load_fast=false` 与 `--purge_orphaned_data=false`：同一目录里后写入的短事件文件不会把更长的历史 step 截掉。
- TB 端口默认 `6006`，配置项 `tensorboard_port` 或环境变量 `ROLLOUT_INSIGHT_TENSORBOARD_PORT`。
- 需本机已安装 `tensorboard`（见 `requirements.txt`）；无事件目录的任务会被跳过。
- 顶栏「洞察」→ `/api/insights`：先选一个任务，列出该任务当前 TensorBoard scalar、已知评测指标说明、尚未收录的 tag，以及建议的分析流程，并可跳转到 TensorBoard（本机 2.20.0 用查询参数 `?tagFilter=`）。

---

## 4. 目录结构

```
rollout-insight/
├── main.py                    # FastAPI 入口
├── config.json                # 全局配置（program_root / db_path / dump_root 等）
├── requirements.txt
├── pyproject.toml             # v0.3.0 CLI 入口（pip install -e . 生成 rollout-insight 可执行）
├── utils/                     # 工具：配置、日志、解析辅助
│   ├── config_loader.py
│   ├── log_setup.py
│   └── analyze_dump.py        # .pt dump 元数据/摘要/transcript（无依赖外部 skill）
├── data/                      # 数据层
│   ├── response.py            # 统一 JSON 响应（code/msg/data）
│   └── rollout_insight_exception.py   # 业务异常（50xxx/51xxx/52xxx/53xxx/54xxx）
├── controller/                # FastAPI 路由层（其他 worker 维护）
├── service/                   # 业务逻辑层（其他 worker 维护）
├── db/                        # SQLAlchemy engine/session/Base（其他 worker 维护）
├── cli/                       # v0.3.0 阶段 3C：rollout-insight CLI 包装层
│   ├── rollout_insight.py    # argparse 入口
│   ├── api_client.py          # 调本地 Web API
│   ├── nanhu_bridge.py        # 调 nanhu-cli（subprocess）
│   ├── tokenizer_cmd.py       # config.json / tokenizer 路径操作
│   └── formatters.py          # table / json 渲染
├── skill/                     # v0.3.0 阶段 3C：Cursor Skill
│   ├── SKILL.md               # 高频配方（Cursor 自动加载）
│   ├── install.sh             # 把 skill 软链到 ~/.cursor/skills/rollout-insight
│   └── recipes.md             # 按场景的速查手册
├── schedule/                  # 定时任务占位
├── static/                    # 静态资源（.gitkeep）
├── templates/                 # Jinja2 模板（base.html）
├── tests/                     # 单元测试
└── log/                       # 运行日志（默认被 .gitignore）
```

---

## 5. API 速览（v0.2）

| 方法 | 路径 | 说明 |
| ---- | ---- | ---- |
| GET  | `/` | 中文欢迎页（自动 3 秒跳转到 `/api/tasks`） |
| GET  | `/healthz` | 健康检查（返回 version） |
| GET  | `/api/tasks` | 任务列表（带分页/状态过滤） |
| POST | `/api/tasks` | 创建任务（form-encoded） |
| GET  | `/api/tasks/{id}` | 任务详情（HTML 渲染；结局/奖励柱 SSR，类型细分渐进加载） |
| GET  | `/api/tasks/{id}/annotations/meta.json` | 任务详情类型柱渐进加载：待拉取的 rollout 列表 |
| GET  | `/api/tasks/{id}/rollouts/{rollout_pk}/annotations.json` | 单 rollout 类型柱 / 进入训练 / 失败摘要 |
| POST | `/api/tasks/{id}/rescan` | 触发 RolloutScanService.rescan_task |
| POST | `/api/tasks/{id}/delete` | 删除任务（级联） |
| GET  | `/api/tasks/create` | 渲染 task_create.html |
| GET  | `/api/tasks/{id}/failures` | 从任务进入的失败分析 |
| GET  | `/api/tasks/{id}/problem-analysis` | 任务问题分析：通过率曲线与 sglang 响应统计 |
| GET  | `/api/tasks/{id}/problem-analysis/instance?instance_id=` | 一道题每次出现的 rollout 与轨迹 |
| GET  | `/api/tasks/{id}/problem-analysis/pass.json` | 按题目的历次通过率（不含响应统计） |
| GET  | `/api/tasks/{id}/problem-analysis.json` | 通过率与已有响应统计 JSON |
| POST | `/api/tasks/{id}/response-stats` | 按 rollout 分批补响应计数（不写 turn 正文） |
| GET  | `/api/tasks/{id}/reward-diagnosis` | 奖励诊断页：有效梯度组 / 奖励分解 / 行为失败卡片 |
| GET  | `/api/tasks/{id}/reward-diagnosis.json` | 奖励诊断 JSON |
| GET  | `/api/segments/search` | Segment 检索 JSON（可选 `group_index`、业务 `attempt_id`） |
| GET  | `/api/segments/search/page` | Segment 检索页（任务列表按钮指向这里） |
| GET  | `/api/segments/{id}` | Segment transcript（HTML 渲染） |
| GET  | `/api/segments/{id}/transcript.json` | Segment transcript（JSON） |
| GET  | `/api/insights` | 洞察页：当前任务的 TensorBoard scalar、指标说明与分析流程 |
| GET  | `/api/failures` | 失败分析旧入口（`?task_id=` 仍进入该任务；不带 `task_id` 时只提示回任务列表） |
| GET  | `/api/failures/attempts/{attempt_pk}` | 单 attempt 失败详情页（结论 / 评测证据 / 耗时 / 轨迹 / patch） |
| GET  | `/api/failures/tasks/{task_id}/overview.json` | 任务失败总览：每 rollout 结局分布 + 非通过 attempt + 原因聚类 |
| GET  | `/api/failures/attempts/{attempt_pk}/detail.json` | 单 attempt 失败详情（JSON） |
| GET  | `/api/failures/attempts/{attempt_pk}/trajectory.json` | OpenHands 轨迹分页（`offset`/`limit`/`errors_only`） |

`GET /api/segments/search` 在原有任务、reward 下限和关键字之外，可以再按 `group_index` 与业务 `attempt_id` 精确过滤。响应里的 `attempt_id` 仍是 attempt 表主键，另外返回 `group_index` 和 `attempt_biz_id`。页面检索走 `GET /api/segments/search/page`。

### 5.1 失败分析（任务级）

失败分析从任务进入：任务列表打开任务详情后，用页顶「失败分析」进入 `/api/tasks/{task_id}/failures`。任务详情、失败分析、问题分析以及 rollout / attempt / segment 子页的右上角都有同一组跳转。顶栏不再放「失败分析」。旧地址 `/api/failures?task_id=` 以及 `/api/failures/attempts/{pk}` 仍可用；不带 `task_id` 的 `/api/failures` 只说明如何回到任务列表。总览可以按 group、attempt id 精确筛选。它回答"每个 attempt 是怎么失败的"：

- **任务视图**：每个 rollout 一行进度条（通过 / 正常失败 / 超时截断 / 中止 / 其它）+ 主要失败原因；点行即按 rollout 过滤。
- **按聚类**：按失败原因（`abort_reason` → `diagnosis` → agent 结束态）聚类，可下钻到对应 attempt 列表。
- **按 attempt**：逐条列出非通过 attempt，可跳「失败轨迹」或「手风琴定位」。点题目名称切到「按题目」。
- **按题目**：用题目名称（`instance_id`）检索。同一题会在多个 rollout 里各采样一次（通常约 8 条 attempt）。列出每次所在的 rollout、该次通过率，以及最后一次减第一次的变化。通过率与问题分析同一套算法。
- **单 attempt 详情**：结论、首个失败用例、评测 stdout/stderr、耗时分解、同组 8 次尝试、OpenHands 轨迹（默认看结尾，可只看报错步）、patch。
- **与 Rollout 详情页四层手风琴联动**：失败 attempt 行右侧出现「失败轨迹 →」；详情页「在手风琴中定位」跳回 `/api/tasks/{id}/rollouts/{pk}?group=..#attempt-{pk}` 并自动展开高亮。任务详情页的 Rollout 列表新增「失败分析」列。

数据来源为运行目录下的 `attempt_artifacts/`（`summaries/`、`patches/`、`failed_trajectories/`），从 `task.rollout_root` 向上最多 4 层查找；首次读取会在该目录写入索引缓存 `.ri_artifacts_index.json`（按 summaries 目录 mtime/文件数自动失效）。没有 `attempt_artifacts` 的任务回退为 `.pt` 标注的失败原因。

### 5.2 问题分析（任务级）

问题分析从任务进入：任务详情页顶「问题分析」进入 `/api/tasks/{task_id}/problem-analysis`。它回答两件事。

**任务通过分析。** 一道题的一次出现是同一个 `rollout_id` 下的同一个 `instance_id`（通常约 8 条 attempt）。按 `rollout_id` 升序，第 k 次出现就是第 k 个 epoch。通过率 = 该次 reward≈1 的条数 / 该次 attempt 条数；abort 与 timeout 计入分母、不计入分子。变好或变坏只看该题最后一次减第一次：大于 0 为绿，小于 0 为红，相等为灰。只出现一次的题单独计数，不进直方图。题目名可点进 `/api/tasks/{task_id}/problem-analysis/instance?instance_id=`，按出现次序看每次所在的 rollout，以及该次每条 attempt 的 sglang 轨迹。总览有两条：一是按 epoch 把全部题目合在一起的通过率曲线，第 k 个点是各题第 k 次出现的 attempt 加总（分子、分母跨题相加）。单步 raw_reward 只覆盖当步抽到的题目，所以会上下跳；这条曲线按 epoch 看全量题目。该次还没再次出现的题不计入，覆盖不足全部题目时用空心点标出。二是首尾差值直方图，每格 1/8。`ROLLOUT_BATCH_SIZE` 来自 run 目录环境变量，读不到时用每个 rollout 的 group 数众数，只用来核对「每个 epoch 窗口里一题是否只出现一次」，不用来重画曲线。

**响应 fake_response。** 统计 sglang 序列里的 assistant 回复：有 tool_call、无 tool_call、工具名（其中 finish 单独列出）。「最后一条是不是 finish」看该 attempt 最后一条 assistant 回复的最后一个工具名。attempt 维度分开报告「出现过无 tool_call」和「最后一条不是 finish」，并给出两者的并集。合法工具名是 `execute_bash`、`str_replace_editor`、`think`、`finish`；有标签但名字不在其中仍算有 tool_call，另外单列。turn 已全部落库时直接用标记，不读正文。覆盖不全时，页面上的「统计全部响应」按 rollout 从 `.pt` 只累计计数，不把正文写入 turn 表，结果缓存在 `rollout_root/.ri_response_stats.json`，按 pt 的路径、大小和 mtime 失效。

### 5.3 奖励诊断（任务级）

奖励诊断从任务进入：任务详情页顶「奖励诊断」进入 `/api/tasks/{task_id}/reward-diagnosis`。它回答「奖励为什么不涨」。

**有效梯度组。** 按 rollout 把每个 group 分成四类：混合（去掉 abort/timeout 后既有过又有挂，可学习）、全过、全挂、只剩基建。画出可学习组占比，并标出相对上一步下降超过 15 pct 的骤降点。判定与任务详情里的 `n_valid_groups` 一致。

**奖励分解。** 同一 rollout 轴叠四条线：有数值 reward 的原始均值、可测通过率（abort/timeout 不进分母）、超时占比、abort 占比。用来区分能力掉分与基建噪声。

**行为失败卡片。** 对非通过 attempt 按优先级打主类：abort → timeout → 无 tool_call → 最后不是 finish → 过早 finish（已 finish 但工具步很少且 diff 很小）。每张卡片给出现象、占失败 attempt 比例、与可测通过率是否同向、建议杠杆，以及跳到失败详情的样例链接。后三类依赖已落库的 assistant turn 标记。

---

## 6. 错误码约定

| 段位      | 含义     |
| --------- | -------- |
| `50xxx`   | 通用异常 |
| `51xxx`   | 任务     |
| `52xxx`   | 解析     |
| `53xxx`   | 南湖     |
| `54xxx`   | 系统     |

详见 `data/rollout_insight_exception.py`。

---

## 7. 开发约定

- 不在仓库内自动 commit；改动由开发者手动 `git add` / `git commit`。
- 任何路由、调度、外部调用必须按错误码段约定抛 `RolloutInsightException`。
- 所有 API 返回统一使用 `data.Response`（`code`/`msg`/`data`）。

---

## 8. v0.3.0 CLI 与 Cursor Skill（**当前阶段**）

阶段 3C 在 `cli/` 与 `skill/` 目录落地「rollout-insight CLI」和 Cursor skill；其余目录
（service/、db/、controller/、templates/）由其他 worker 维护，本阶段**不**改动。

### 8.1 安装

```bash
# 1) 装包（用户手动执行；生成 rollout-insight 可执行文件）
pip install -e .

# 2) 安装 Cursor Skill 软链（把本仓库 skill/ 软链到 ~/.cursor/skills/）
bash skill/install.sh

# 3) 启动本地 Web 服务（其他 worker 维护的主进程）
python main.py
```

### 8.2 命令树

```
rollout-insight [--api-base URL] [--format json|table] [--cluster {wuxi,x10000}]
    tasks {list,create,rescan,delete}
    segments {list,show,search}
    nanhu {jobs,restart,link}
    tokenizer {set,test}
```

**写操作默认 dry-run；加 `--submit` 才执行。**

### 8.3 高频配方

```bash
# 任务
rollout-insight --format table tasks list
rollout-insight tasks create --task-name foo --rollout-root /mnt/.../rollout_dumps
rollout-insight tasks create --task-name foo --rollout-root /mnt/.../rollout_dumps --submit
rollout-insight tasks rescan --task-id 1 --submit
rollout-insight tasks delete --task-id 1 --submit

# Segment
rollout-insight --format table segments list --task-id 1 --min-reward 0.8
rollout-insight segments search --task-id 1 --keyword sandbox-id-xyz
rollout-insight segments show --segment-id 42

# 南湖（透传 nanhu-cli / nanhu）
rollout-insight --cluster wuxi --format table nanhu jobs --status running
rollout-insight --cluster wuxi nanhu jobs --job-type AI_TRAIN
rollout-insight --cluster wuxi nanhu restart --job-id jb-xxx
rollout-insight --cluster wuxi nanhu restart --job-id jb-xxx --submit
rollout-insight nanhu link --job-id jb-xxx --task-name slime-021-rollout

# Tokenizer
rollout-insight tokenizer test --path /mnt/zj-data/models/.../hf
rollout-insight tokenizer set --path /mnt/zj-data/models/.../hf
rollout-insight tokenizer set --path /mnt/zj-data/models/.../hf --submit
```

### 8.4 Cursor Skill 集成

阶段 3C 同时提供 Cursor skill：

- `skill/SKILL.md` —— Cursor 自动加载的高频配方
- `skill/recipes.md` —— 按场景的速查手册
- `skill/install.sh` —— 创建 `~/.cursor/skills/rollout-insight` 软链

**安装 skill（用户手动执行）：**

```bash
bash skill/install.sh
# 验证：
ls -l ~/.cursor/skills/rollout-insight
# 卸载：
rm ~/.cursor/skills/rollout-insight
```

### 8.5 后续手动扩展（v0.3 之后由用户决策）

- `make install-skill` —— 暂未创建 Makefile；如需一键安装，可加 `Makefile`
  包装 `bash skill/install.sh`
- 链接到 nanhu-skill 项目（`~/.cursor/skills/nanhu-cli`）已通过 PATH 上找
  `nanhu` / `nanhu-cli` 二进制自动完成，无需额外配置

### 8.6 排障速查

| 症状 | 原因 | 行动 |
|------|------|------|
| `无法连接 http://localhost:39998/...` | Web service 没起 | `python main.py` 后重试 |
| `找不到 nanhu / nanhu-cli 可执行文件` | nanhu-skill 未装 | `pip install -e /path/to/nanhu-skill` |
| `nanhu jobs list: error: --job-type required` | list 要求 job_type | 留空 `--job-type`，自动走 `jobs running` |
| `--format table` 输出 `(empty)` | payload 不是 list | 改 `--format json` 看数据结构 |

---

## 9. 后续阶段（roadmap）

- v0.3.0 阶段 3A：南湖调度接入（nanhu_client + 创建/重启/绑定作业）
- v0.3.0 阶段 3B：表 UI 增强（任务列表/详情/segment transcript 的 Jinja 模板）
- v0.4.0+：eval dataset、动态过滤、长期稳定性