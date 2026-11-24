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

# 端口已被占用（Errno 98）时，用重启脚本：先安全释放本服务旧进程再启动
bash scripts/restart_server.sh              # 默认后台重启（脱离终端，可直接关终端）
bash scripts/restart_server.sh restart --fg # 前台重启（Ctrl+C 结束）
bash scripts/restart_server.sh status       # 查看谁占用 39998
```

> 端口优先级：`$ROLLOUT_INSIGHT_PORT`（默认 `39998`）。`restart_server.sh` 默认只杀掉命令行像本服务（`main.py` / `main:app`）的监听进程；其它进程占用时需显式加 `--force`。
>
> **默认后台启动**：脚本用 `setsid` 把服务放进独立会话（`PPID=1`、无控制终端），**关闭终端 / 断开 SSH 都不影响服务**；日志追加到 `log/server.log`，pid 记录在 `log/server.pid`，用 `bash scripts/restart_server.sh stop` 停止。需要前台跑（Ctrl+C 结束）时显式加 `--fg`。

### 训练曲线（TensorBoard）

- 任务详情「训练曲线」可启动本机 TensorBoard，**新标签**打开（不嵌入 iframe）。
- 顶栏「曲线对比」→ `/api/tensorboard`：**多任务曲线对比页**。勾选任意多个任务 + 任意多个指标，即可把它们叠在同一张图里对比，与「单任务训练曲线」完全同款（同一套指标目录、同一套 SVG 折线、逐 step / 按 epoch 两种视图）。页面下方仍保留一键拉起本机 TensorBoard 的入口。
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
│   ├── analyze_dump.py        # .pt dump 元数据/摘要/transcript（无依赖外部 skill）
│   ├── tb_series_reader.py    # 读 TB 事件目录的完整 scalar 序列（多 tag）
│   ├── train_curve_catalog.py # 训练曲线指标目录：默认按钮组 + tag 匹配规则
│   ├── train_curve_epoch.py   # 训练曲线 epoch 聚合（steps_per_epoch / 分箱均值）
│   ├── eval_report_md.py      # 解析 run 目录的 evaluations.md（外部评测台账 + 分数）
│   ├── icon_catalog.py        # 菜单图标清单：路由/动作 → Lucide 图标名 + tooltip
│   └── template_factory.py    # 统一创建 Jinja2Templates（注入全局模板变量，如品牌名）
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
├── static/                    # 静态资源
│   ├── branding/              # 站点图标源图（唯一真源，由 scripts/gen_favicon.py 派生）
│   ├── icons/                 # 离线内置 Lucide SVG 图标（MIT，见 static/icons/LICENSE）
│   ├── favicon.ico            # 浏览器标签页 / 书签图标（内含 16/32/48）
│   ├── ri_icons.css           # 图标 CSS mask（由 scripts/gen_icon_css.py 生成）
│   ├── ri_epoch_chart.js      # 共享 epoch 折线渲染（window.RiEpochChart.render）
│   └── ri_multi_chart.js      # 共享多序列折线渲染（window.RiMultiChart.renderLineChart）
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
| GET  | `/favicon.ico` | 站点图标（多尺寸 ICO，**无需登录**） |
| GET  | `/apple-touch-icon.png` | 180×180 图标（iOS 添加到主屏，**无需登录**） |
| GET  | `/site.webmanifest` | PWA manifest（名称 / 主题色 / 各尺寸图标） |
| GET  | `/api/tasks` | 任务列表（带分页/状态过滤） |
| POST | `/api/tasks` | 创建任务（form-encoded） |
| GET  | `/api/tasks/{id}` | 任务详情（HTML 渲染；结局/奖励柱 SSR，类型细分渐进加载） |
| GET  | `/api/tasks/{id}/annotations/meta.json` | 任务详情类型柱渐进加载：待拉取的 rollout 列表 |
| GET  | `/api/tasks/{id}/rollouts/{rollout_pk}/annotations.json` | 单 rollout 类型柱 / 进入训练 / 失败摘要 |
| POST | `/api/tasks/{id}/rescan` | 触发 RolloutScanService.rescan_task |
| POST | `/api/tasks/{id}/delete` | 删除任务（级联） |
| GET  | `/api/tasks/create` | 渲染 task_create.html |
| GET  | `/api/tasks/{id}/failures` | 从任务进入的失败分析（可带 `cat=` / `reason=` / `agent=` / `rollout_id=` 初始筛选） |
| GET  | `/api/tasks/{id}/problem-analysis` | 任务问题分析：通过率曲线与 sglang 响应统计 |
| GET  | `/api/tasks/{id}/problem-analysis/instance?instance_id=` | 一道题每次出现的 rollout 与轨迹 |
| GET  | `/api/tasks/{id}/problem-analysis/pass.json` | 按题目的历次通过率（不含响应统计） |
| GET  | `/api/tasks/{id}/problem-analysis.json` | 通过率与已有响应统计 JSON |
| POST | `/api/tasks/{id}/response-stats` | 按 rollout 分批补响应计数（不写 turn 正文） |
| GET  | `/api/tasks/{id}/reward-diagnosis` | 奖励诊断页：有效梯度组 / 奖励分解 / 行为失败卡片 |
| GET  | `/api/tasks/{id}/reward-diagnosis.json` | 奖励诊断 JSON |
| GET  | `/api/tasks/{id}/train-curve` | 训练曲线页：指标按钮组（奖励/长度/梯度/entropy_loss/context_exhausted/max_fake_responses/rollout_time）+ 折线（逐 step / 按 epoch 聚合）+ 外部评测分数（`evaluations.md`） |
| GET  | `/api/tasks/{id}/train-curve.json` | 训练曲线 JSON：指标列表 + 每个指标的 step→value 序列（含 epoch 聚合序列）+ `evaluations` 评测台账 |
| POST | `/api/tasks/{id}/train-curve/preference` | 保存曲线指标配置（`scope=global` 用户默认 / `scope=task` 任务追加） |
| GET  | `/api/tensorboard` | 多任务曲线对比页：勾选多个任务 + 指标，叠图对比 |
| GET  | `/api/tensorboard/curve-compare.json` | 多任务曲线叠加数据（`task_ids=1,2,3` / `keys=` / `view=step\|epoch`；`brief=true` 只回任务清单与指标目录） |
| GET  | `/api/cross-task-compare` | 跨任务对比页：选多个任务 + 选基线，比较共同题目的训练过程变化 |
| GET  | `/api/cross-task-compare.json` | 跨任务对比 JSON（`task_ids=1,2,3`，可选 `baseline_task_id=` / `instance_id=`） |
| GET  | `/api/cross-task-compare/instance.json` | 单题在多任务上的逐次出现与通过率 |
| GET  | `/api/segments/search` | Segment 检索 JSON（可选 `group_index`、业务 `attempt_id`） |
| GET  | `/api/segments/search/page` | Segment 检索页（任务列表按钮指向这里） |
| GET  | `/api/segments/{id}` | Segment transcript（HTML 渲染） |
| GET  | `/api/segments/{id}/transcript.json` | Segment transcript（JSON） |
| GET  | `/api/insights` | 洞察页：当前任务的 TensorBoard scalar、指标说明与分析流程 |
| GET  | `/api/failures` | 失败分析旧入口（`?task_id=` 仍进入该任务；不带 `task_id` 时只提示回任务列表） |
| GET  | `/api/failures/attempts/{attempt_pk}` | 单 attempt 失败详情页（结论 / 评测证据 / 耗时 / 轨迹 / patch） |
| GET  | `/api/failures/tasks/{task_id}/overview.json` | 任务失败总览：每 rollout 结局分布 + 非通过 attempt + 原因聚类 + agent 结束态聚类 + 组内失败类型 |
| GET  | `/api/failures/attempts/{attempt_pk}/detail.json` | 单 attempt 失败详情（JSON） |
| GET  | `/api/failures/attempts/{attempt_pk}/trajectory.json` | OpenHands 轨迹分页（`offset`/`limit`/`errors_only`） |

`GET /api/segments/search` 在原有任务、reward 下限和关键字之外，可以再按 `group_index` 与业务 `attempt_id` 精确过滤。响应里的 `attempt_id` 仍是 attempt 表主键，另外返回 `group_index` 和 `attempt_biz_id`。页面检索走 `GET /api/segments/search/page`。

### 5.0 任务详情页

任务详情页（`/api/tasks/{task_id}`）把**全量通过率**放在**训练指标之前**——它表征任务训练进程，是该页最重要的指标。区块内容包括：按 epoch 汇总的全量题目通过率折线、变好 / 不变 / 变坏 / 只出现一次的题目计数、题目与 rollout 概览。

折线绘制逻辑抽到共享模块 `static/ri_epoch_chart.js`（暴露 `window.RiEpochChart.render`），任务详情页与问题分析页共用同一份 SVG 渲染，避免两处口径漂移。

### 5.1 失败分析（任务级）

失败分析从任务进入：任务列表打开任务详情后，用页顶「失败分析」进入 `/api/tasks/{task_id}/failures`。任务详情、失败分析、问题分析以及 rollout / attempt / segment 子页的右上角都有同一组跳转。顶栏不再放「失败分析」。旧地址 `/api/failures?task_id=` 以及 `/api/failures/attempts/{pk}` 仍可用；不带 `task_id` 的 `/api/failures` 只说明如何回到任务列表。总览可以按 group、attempt id 精确筛选。它回答"每个 attempt 是怎么失败的"：

- **任务视图**：每个 rollout 一行进度条（通过 / 正常失败 / 超时截断 / 中止 / 其它）+ 主要失败原因；点行即按 rollout 过滤。
- **按聚类**：按失败原因聚类，可下钻到对应 attempt 列表。失败原因由 `utils/failure_reason_taxonomy.py` 统一归一：artifacts summary 与 `.pt` 注解两处口径一致（`evaluator:tests_failed`、`slime_context_limit_reached`、`agent_time_budget_exceeded`、`exception:SandboxApiException` 等）。
- **Agent 结束态**：单列一节按 `agent_outcome` 聚类（正常结束 / 上下文打满 / 时间预算耗尽 / SDK 报错 / 迭代次数超限 …），带次数与占比。它与评测结论正交——`evaluator:tests_failed` 说明评测没通过，`context_exhausted` / `sdk_error` 说明 agent 自己没跑完。点任一项即跳到该结束态的 attempt 列表。
- **按 attempt**：逐条列出非通过 attempt，含「agent 结束态」列，可跳「失败轨迹」或「手风琴定位」。点题目名称切到「按题目」。
- **按题目**：用题目名称（`instance_id`）检索。同一题会在多个 rollout 里各采样一次（通常约 8 条 attempt）。列出每次所在的 rollout、该次通过率，以及最后一次减第一次的变化。通过率与问题分析同一套算法。
- **单 attempt 详情**：结论、首个失败用例、评测 stdout/stderr、耗时分解、同组 8 次尝试、OpenHands 轨迹（默认看结尾，可只看报错步）、patch。
- **与 Rollout 详情页四层手风琴联动**：失败 attempt 行右侧出现「失败轨迹 →」；详情页「在手风琴中定位」跳回 `/api/tasks/{id}/rollouts/{pk}?group=..#attempt-{pk}` 并自动展开高亮。任务详情页的 Rollout 列表新增「失败分析」列。
- **从任务详情直接跳**：任务详情页「正常失败 / 超时 / abort · 具体类型」清单里的每个原因都是快捷链接，点它跳到本任务的失败分析页并自动按该类型筛选（`/api/tasks/{id}/failures?cat=..&reason=..`）。

数据来源为运行目录下的 `attempt_artifacts/`（`summaries/`、`patches/`、`failed_trajectories/`），从 `task.rollout_root` 向上最多 4 层查找；读取会在该目录写入索引缓存 `.ri_artifacts_index.json`。索引**增量刷新**：目录指纹（mtime + 文件数）未变时直接复用缓存，变了则只重新解析新增 / 被覆盖的 summary（按内容指纹比对），因此训练中持续增长的 run 目录不会每次都全量重扫。没有 `attempt_artifacts` 的任务回退为 `.pt` 标注的失败原因。

**Rollout 详情页的组结果列**：Group 列表概览新增「失败类型」列，列出每个 group 内出现的失败类型；每个类型下的 attempt 主号都是超链接，点击直接定位到页面下方对应 attempt（`#attempt-{pk}`，自动展开并高亮）。「正常失败 / 超时 / abort」三列的数量同样渲染为可点链接，指向本组。

### 5.2 问题分析（任务级）

问题分析从任务进入：任务详情页顶「问题分析」进入 `/api/tasks/{task_id}/problem-analysis`。它回答两件事。

**任务通过分析。** 一道题的一次出现是同一个 `rollout_id` 下的同一个 `instance_id`（通常约 8 条 attempt）。按 `rollout_id` 升序，第 k 次出现就是第 k 个 epoch。通过率 = 该次 reward≈1 的条数 / 该次 attempt 条数；abort 与 timeout 计入分母、不计入分子。变好或变坏只看该题最后一次减第一次：大于 0 为绿，小于 0 为红，相等为灰。只出现一次的题单独计数，不进直方图。筛选里另有「太容易」（每一次出现都全通过，常见是每次 8/8）和「太难」（每一次出现都是 0 通过，常见是每次 0/8）。题目名可点进 `/api/tasks/{task_id}/problem-analysis/instance?instance_id=`，按出现次序看每次所在的 rollout，以及该次每条 attempt 的 sglang 轨迹。总览有两条：一是按 epoch 把全部题目合在一起的通过率曲线，第 k 个点是各题第 k 次出现的 attempt 加总（分子、分母跨题相加）。单步 raw_reward 只覆盖当步抽到的题目，所以会上下跳；这条曲线按 epoch 看全量题目。该次还没再次出现的题不计入，覆盖不足全部题目时用空心点标出。二是首尾差值直方图，每格 1/8。`ROLLOUT_BATCH_SIZE` 来自 run 目录环境变量，读不到时用每个 rollout 的 group 数众数，只用来核对「每个 epoch 窗口里一题是否只出现一次」，不用来重画曲线。

**响应 fake_response。** 统计 sglang 序列里的 assistant 回复：有 tool_call、无 tool_call、工具名（其中 finish 单独列出）。「最后一条是不是 finish」看该 attempt 最后一条 assistant 回复的最后一个工具名。attempt 维度分开报告「出现过无 tool_call」和「最后一条不是 finish」，并给出两者的并集。合法工具名是 `execute_bash`、`str_replace_editor`、`think`、`finish`；有标签但名字不在其中仍算有 tool_call，另外单列。turn 已全部落库时直接用标记，不读正文。覆盖不全时，页面上的「统计全部响应」按 rollout 从 `.pt` 只累计计数，不把正文写入 turn 表，结果缓存在 `rollout_root/.ri_response_stats.json`，按 pt 的路径、大小和 mtime 失效。

**单题明细（题目为中心）。** 点题目名进 `/api/tasks/{task_id}/problem-analysis/instance?instance_id=`，页面围绕**这一道题**组织：

- **本题趋势**：三张折线图，横轴是该题第几次出现（epoch），分别是**通过率**、**响应长度**（该次各 attempt 均值）、**agent 轮数**（均值）。三者量纲不同，分开绘制避免互相压扁。回答"这个题目在训练过程中是变好还是变坏、伴随的响应是变长还是变短"。
- **逐次明细表**：每次出现一张表，逐条 attempt 列出 group、attempt id、结局、reward、**响应长度**、**轮数**与轨迹入口（sglang 轨迹 / 正文）。
- 响应长度与轮数取自 `ri_attempt_table` 的 `response_length` / `agent_turns` 字段。

趋势与折线渲染复用共享模块 `static/ri_multi_chart.js`（与多任务曲线对比同款 SVG）。

### 5.3 奖励诊断（任务级）

奖励诊断从任务进入：任务详情页顶「奖励诊断」进入 `/api/tasks/{task_id}/reward-diagnosis`。它回答「奖励为什么不涨」。

**有效梯度组。** 按 rollout 把每个 group 分成四类：混合（去掉 abort/timeout 后既有过又有挂，可学习）、全过、全挂、只剩基建。画出可学习组占比，并标出相对上一步下降超过 15 pct 的骤降点。判定与任务详情里的 `n_valid_groups` 一致。

**奖励分解。** 同一 rollout 轴叠四条线：有数值 reward 的原始均值、可测通过率（abort/timeout 不进分母）、超时占比、abort 占比。用来区分能力掉分与基建噪声。

**行为失败卡片。** 对非通过 attempt 按优先级打主类：abort → timeout → 无 tool_call → 最后不是 finish → 过早 finish（已 finish 但工具步很少且 diff 很小）。每张卡片给出现象、占失败 attempt 比例、与可测通过率是否同向、建议杠杆，以及跳到失败详情的样例链接。后三类依赖已落库的 assistant turn 标记。

**奖励还能怎么涨。** 奖励诊断页新增杠杆区块，把「奖励为什么不涨」拆成 7 条可执行杠杆，按状态排序（先修 bad，再看 warn）：

| 杠杆 | 含义 | 异常时说明 |
| ---- | ---- | ---- |
| `learnable_ratio` | 有效梯度组（混合组）占比 | 偏低或骤降 → 大量样本同号，梯度趋零，没有上升空间 |
| `measurable_pass_rate` | 可测通过率（去掉 abort / timeout 的分母） | 停滞 → 策略本身没在学 |
| `timeout_ratio` | 超时 / 上下文打满占比 | 升高 → 奖励被截断吃掉，可测样本变少 |
| `abort_ratio` | abort 占比 | 升高 → 奖励不可测，属基建问题而非策略问题 |
| `raw_mean_reward` | 训练实际拿到的奖励 | 明显低于可测通过率 → 差值就是被 abort / timeout 扣掉的部分 |
| `saturated_pool_ratio` | 题目池饱和度（末次最好一次通过率 ≥ 0.875 的题占比） | 过高 → 题目池已被刷穿，剩下样本同号，再怎么训也涨不动 |
| `hard_pool_ratio` | 题目池难度天花板（末次最好一次通过率 ≤ 0.125 的题占比） | 过高 → 题目池太难，大量样本全挂，同样没有学习信号 |

题目池健康度取每道题**最好一次**出现的通过率（而不是最后一次），避免「早期没学会、后来学会了」的题被误判成「太难」。

口径由 `service/reward_diagnosis_service.build_reward_levers` 统一提供，跨任务对比页复用同一份实现。

### 5.4 训练曲线（任务级）

训练曲线从任务进入：任务详情页顶「训练曲线」进入 `/api/tasks/{task_id}/train-curve`。它把该任务 `rollout_root` 下 `tbs/` 或 `tf_logs/` 的 TensorBoard scalar 画成折线。

**指标按钮组。** 默认展示 7 个内置指标：奖励、长度、梯度、entropy_loss、context_exhausted、max_fake_responses、rollout_time。用户可以勾选任意内置指标、追加自定义指标（写 TB tag 名，形如 `custom:my_metric`），并选择保存范围：

- **保存为默认**（`scope=global`）：该用户的全局按钮组，长期生效，之后打开任何任务都按这套配置。
- **保存为本任务**（`scope=task`）：在该任务的按钮组上追加展示项，不影响其它任务。

最终生效的指标 = 用户全局偏好 + 本任务追加项（用户全局偏好一旦保存就完全覆盖内置默认按钮组）。偏好存在 `ri_train_curve_preference_table`。

**一个指标 = 一张图，多条子曲线叠画。** 内置指标先按**标准 tag** 命中（`utils/train_curve_catalog.CURVE_METRICS[key]["series"]`），命中不到才退回锚定到路径段的正则兜底。这样：

| 指标 | 画的曲线 |
| --- | --- |
| 奖励 | `rollout/generated/raw_reward`、`rollout/generated/paired_raw_reward`、`rollout/filtered/raw_reward`（三条同图，不同颜色 + 图例） |
| 长度 | `rollout/response_len/mean` |
| 梯度 | `train/grad_norm` |
| entropy_loss | `train/entropy_loss` |
| context_exhausted | `rollout/outcome/context_exhausted` |
| max_fake_responses | `rollout/outcome/max_fake_responses` |
| rollout_time | `perf/rollout_time` |

正则兜底用 `(^|/)raw_reward$` 这类锚定写法，避免把 `raw_reward_coverage`、`raw_reward_measured_attempt_count` 等派生指标误当成奖励曲线；`rollout_time` 同理排除 `perf/rollout_time/attempt/*` 的分解项。

**逐 step / 按 epoch 两种视图。** 右上角可切换：

- **逐 step**：横轴是 TensorBoard 的 step（一次 rollout 采样 + 一次训练更新），粒度细、抖动大。
- **按 epoch 聚合**：每个 epoch 内所有 step 取平均，看整体趋势。epoch 边界与问题分析页同口径——一个 epoch 覆盖 `题目池大小 ÷ 每轮抽样题数` 个 rollout step（例如 64 题 ÷ 16 = 4）。边界算不出来时该按钮置灰，只保留逐 step 视图。

epoch 聚合逻辑在 `utils/train_curve_epoch.py`（`resolve_steps_per_epoch` / `aggregate_by_epoch` / `epoch_view`），epoch 边界复用 `service.problem_analysis_service.build_pass_trend` 返回的 `steps_per_epoch`，避免两页口径漂移。

**数据口径。** 读取时关闭 TensorBoard 的 orphan purge、给足 scalar 水库容量（TB 的 Reservoir 是随机抽样，容量不足会随机丢点、曲线缺段），长序列由本服务做确定性等距抽样并保留首尾。

**外部评测分数（evaluations.md）。** 训练 run 根目录下若存在 `evaluations.md`，说明这个任务的检查点在持续做外部评测。该文件是评测台账：每行一次评测，记录 training step、评测名、轮次、南湖作业、作业状态与结果。训练曲线页在曲线**上方**渲染「外部评测分数」区块：

- **按评测名分组**（例如 `panel`、`v500`），每组一张折线图，横轴是 training step，纵轴是评测分数。
- **一个评测集内按分数口径分别画线**：同一次评测可能给出**多个评测集（不同题池）**的分数——例如 `panel` 同时给出「429 题的加权分」与「87 题的 holdout 分」。两者题池不同、各自独立成线，共用同一张图的坐标轴；图例标出口径名、题池（如「85/87 题」）、末值与变化量（变绿 / 变红），线型按口径区分（加权分实线 / merged 虚线 / holdout 点线）。
- **分数口径**：`weighted score on N tasks`（加权分）、`merged M/N = x`、`holdout score: x over A of B tasks`（留出分）；`95% selection interval ±x`、`ungraded N`、`first round A/B` 一并展示。
- **逐次记录表**：列出每次评测的 step、轮次、作业状态（finished / running / scheduling / error）、**各口径分数分列**、备注与作业 ID，便于回查南湖作业。

解析逻辑在 `utils/eval_report_md.py`（`parse_evaluations_md` / `extract_scores` / `iter_score_kinds` / `read_evaluations_for_rollout_root`），按口径拆分为多条曲线在 `service/train_curve_service.py` 的 `_build_eval_series`，run 目录解析复用 `utils/tb_run_paths.resolve_run_dir`。文件不存在时该区块整体隐藏，不影响曲线。

### 5.4.1 多任务曲线对比（顶栏）

顶栏「曲线对比」进入 `/api/tensorboard`。它把**多个任务**的训练曲线叠在同一张图里，与单任务训练曲线**同款**：同一套指标目录（`utils/train_curve_catalog.py`）、同一套 epoch 聚合口径、同一套 SVG 折线渲染。

- **勾选任务**：页面上方列出所有含 `tbs/` 或 `tf_logs/` 的任务，带名称筛选、全选 / 清空 / 最近 5 个快捷操作；默认勾选最近 3 个。
- **勾选指标**：与单任务训练曲线共用 7 个内置指标（奖励 / 长度 / 梯度 / entropy_loss / context_exhausted / max_fake_responses / rollout_time），按钮组可多选。
- **叠加规则**：同一指标下，**每个任务 × 每条子曲线**各占一条线（例如「奖励」× 3 个任务 = 9 条线）。配色以**任务**为主轴（每个任务一种颜色），同一任务的多个子指标改用**线型**区分（实线 / 虚线 / 点线），一眼能看出某条线属于哪个任务；图例标出任务名、曲线名与首尾变化量。
- **逐 step / 按 epoch 两种视图**：与单任务训练曲线一致，可直接横向比较不同任务的收敛节奏。

数据接口 `GET /api/tensorboard/curve-compare.json?task_ids=1,2&keys=raw_reward&view=step`；`brief=true` 时只回任务清单与指标目录（页面首屏用，不读事件文件，避免一次加载十几个事件目录）。SVG 折线渲染抽到共享模块 `static/ri_multi_chart.js`（暴露 `window.RiMultiChart.renderLineChart` / `renderSparkline`），与问题分析页的单题趋势图共用。

### 5.5 跨任务对比（多任务）

顶栏「跨任务对比」进入 `/api/cross-task-compare`。它回答：**不同的训练配置下，训练过程中哪些共同的题目变好了，哪些变坏了。**

- **共同题目**：所有选中任务都出现过的 `instance_id`。
- **epoch 口径**：与问题分析页完全一致——第 k 个 epoch = 该题第 k 次出现（按 `rollout_id` 升序）。
- **基线任务由用户主动选择**：任务选择框下方有「基线任务」下拉，只列出参与对比的任务。切换基线会重新计算「相对基线」列并按变化幅度重新排序。请求参数 `baseline_task_id=`（非法或未指定时退回第一个任务）。
- **变好 / 变坏**：按各任务**自身训练轨迹**（末次通过率 − 首次通过率）统计，区分「变好 / 不变 / 变坏」三档；另有「相对基线」一列，比较末次通过率与基线任务之差，回答「哪个配置更强」。**变好用绿色、变差用红色**。
- **每题可展开**：题目列表每行可展开，抽屉里给出各任务的通过率折线（sparkline）、每次出现（epoch）的**通过数量**（`n_pass / n_attempts`，如 `3/8`）、末次通过率与首尾差。
- **单题下钻**：抽屉底部可跳「查看单题明细」，看这一道题在多个任务上的 attempt 级别明细。
- **奖励上升指引**：每个任务给出一组杠杆卡片，复用 `service/reward_diagnosis_service.build_reward_levers` 与单任务奖励诊断页同款口径，指明哪条在拖后腿、该做什么。

### 5.6 界面图标（离线 Lucide）

顶栏、任务内跳转栏、页面按钮都配了图标，用来一眼区分各入口的职责（任务列表 / 曲线对比 / 跨任务对比 / 洞察 / **南湖作业** / 系统设置 / 登录 / 注册 / 设置 / 退出，以及任务详情 / 训练曲线 / 失败分析 / 问题分析 / 奖励诊断）。悬停时还会显示一句话说明。

- **系统设置**：指向 `/api/me/cluster-configs`（兼容旧链 `/api/settings`）。每位登录用户维护**自己的**南湖 `app_key` / `app_secret`，同步与解析不会混用他人凭据。
- **南湖作业**：顶栏入口 `/api/nanhu/jobs/page`；列表展示**创建人**与**算法库名称**，可按任务名称 / 创建人 / 算法库检索。同事共享作业用「批量 UUID / 导入并勾选」（`getJobDetail`，与 `nanhu jobs detail` 同源）。**新建任务**：在「关联南湖 job_id」框粘贴单个或多个 `jb-aitrain-…`——单个点「解析」回填字段；多个点「解析」即批量创建（无需改任务名）；已关联过的 job 返回 `51006` 并跳过，其余继续创建。

- **图标本体**：Lucide SVG（MIT License），已 vendor 到 `static/icons/`，许可声明见 `static/icons/LICENSE`。部署环境是离线内网，CDN 不可达，因此**不走外链**。
- **上色方式**：`static/ri_icons.css` 用 CSS `mask-image` + `currentColor` 实现，图标颜色随所在导航栏 / 按钮的文字色自适应，无需为深色、浅色各存一份资源。
- **清单集中管理**：图标名与提示语统一写在 `utils/icon_catalog.py`（`NAV_ICONS`），模板里只写 `class="ri-icon ri-icon-<name>"`，避免图标语义散落在各个模板里。
- **新增图标**：把 Lucide 的 SVG 放进 `static/icons/`，然后重跑

  ```bash
  python scripts/gen_icon_css.py          # 重新生成 static/ri_icons.css
  python scripts/gen_icon_css.py --check   # CI 校验：CSS 与图标目录是否一致
  ```

### 5.7 站点图标（浏览器标签页 / 书签）

浏览器标签页、书签栏与收藏夹里显示的站点图标，由 `static/branding/ri_icon_source.png`（1024×1024 设计稿）派生。

- **真源唯一**：`static/branding/ri_icon_source.png` 是唯一真源，其余尺寸全部由脚本生成，不手改产物。
- **产物清单**：`static/favicon.ico`（内含 16 / 32 / 48 三档，浏览器自动挑选）、`static/favicon-16x16.png`、`static/favicon-32x32.png`、`static/favicon-48x48.png`、`static/apple-touch-icon.png`（180×180，iOS 添加到主屏）、`static/icon-192.png` 与 `static/icon-512.png`（PWA）。
- **访问路径**：模板 `base.html` 在 `<head>` 里声明图标；另有根路径端点 `/favicon.ico`、`/apple-touch-icon.png`、`/site.webmanifest`（PWA manifest）——这些端点**不要求登录**，否则未登录时页面被 302 到登录页时标签页会退回默认图标。实现见 `controller/site_controller.py`。
- **小尺寸优化**：16px 下原图细节会糊，因此生成时统一做饱和度 / 对比度 / 亮度微调 + UnsharpMask 锐化，并叠一层圆角遮罩，让图标在浅色与深色标签页上边缘都干净。
- **改图标**：替换源图后重跑

  ```bash
  python scripts/gen_favicon.py          # 重新生成全部图标
  python scripts/gen_favicon.py --check   # CI 校验：图标是否与源图一致
  ```

> 依赖 Pillow（`pip install Pillow`）；缺失时脚本会给出明确提示并返回退出码 1。

### 5.8 站点品牌名（「洞见」）

站点品牌名由配置项 **`site_brand_name`** 统一控制（默认「洞见」，见 `config.json`），它决定三处显示：浏览器标签页标题后缀、顶栏 logo 文字、登录 / 注册页标题。

- **为什么是「洞见」而不是「Rollout Insight」**：Chrome 在标签页过窄时会把长标题折叠成**只显示图标**，用户就看不到文字了。短品牌名 + 短页面名（两段式）能稳定显示成「图标 + 文字」。
- **标题格式**：统一为 `<页面名> · <品牌名>`，例如「任务列表 · 洞见」「任务 #11 · 洞见」「洞察 · 洞见」。页面名刻意取短，不带长任务名 —— 任务名几十个字符会把标签撑爆。
- **注入方式**：品牌名是**全局模板变量**，由 `utils/template_factory.py` 的 `create_templates()` 一次性写进 `env.globals`；所有 controller 用它创建模板实例，模板里直接用 `{{ brand }}`。
- **改名**：改 `config.json` 的 `site_brand_name` 后重启服务即可，**不需要改任何模板**。`config_loader` 里的 `site_brand_name` 属性是它的读取入口。

> 顺带说明：各 controller 原先各自 `Jinja2Templates(directory=...)`，全局变量无处安放。现已统一收敛到 `create_templates()`，新增全局模板变量时只改这一处。

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

## 9. 训练奖励诊断方案（设计文档）

面向 slime + sglang 的 agentic RL 训练，把 rollout-insight 从「rollout 数据可视化工具」
升级为「**奖励不涨时的归因与行动引擎**」。

| 文档 | 内容 |
|------|------|
| [docs/PLAN_reward_uplift.md](docs/PLAN_reward_uplift.md) | **v2 主方案**：真实数据验证（11 任务 / 398 rollout / 51314 attempt / 458 个 TB tag）、agent 时间预算退化机制、口径修正、训练侧指标接入、题目池健康度、落地顺序 |
| [docs/PLAN_reward_uplift_specs.md](docs/PLAN_reward_uplift_specs.md) | **配套设计规格**：各模块函数签名、数据契约、判定阈值、页面规格、测试清单、`func.md` 同步清单 |

**当前结论速览**（2026-11-23 实测，详见主方案 §1.5）：

- 本批任务奖励不涨的**第一主因**是 **agent 时间预算耗尽**（60 分钟墙钟）→ 单轮生成耗时膨胀
  → attempt 被杀 → GRPO 组整组丢弃 → 训练样本枯竭，而非 v1 判断的「基建故障」
- 口径修正（有效梯度组 / 通过率分母）与训练侧指标接入依然要做，但属次要问题
- 配套还发现 v1 与初版方案都漏掉的三个维度：训练样本漏斗 `rollout/funnel/*`、
  slime 权威组语义 `rollout/generated/shaped_group_reward/*`、题目池 → 采样密度 → 梯度密度

> 文档为**设计级**，尚无对应代码实现；`func.md` 未变更。

---

## 10. 后续阶段（roadmap）

- v0.3.0 阶段 3A：南湖调度接入（nanhu_client + 创建/重启/绑定作业）
- v0.3.0 阶段 3B：表 UI 增强（任务列表/详情/segment transcript 的 Jinja 模板）
- v0.4.0+：eval dataset、动态过滤、长期稳定性
- 奖励诊断 v2 第一批：agent 时间预算画像 + 组语义口径对齐（见 `docs/PLAN_reward_uplift.md` §4）