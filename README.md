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
| GET  | `/api/tasks/create` | 渲染 task_create.html || GET  | `/api/segments/search` | Segment 检索（task_id 必填） |
| GET  | `/api/segments/{id}` | Segment transcript（HTML 渲染） |
| GET  | `/api/segments/{id}/transcript.json` | Segment transcript（JSON） |

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