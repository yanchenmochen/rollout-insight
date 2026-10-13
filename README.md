# Rollout Insight

> Slime rollout 落盘（`.pt`）的解析与作业洞察工具。

当前阶段：**v0.1.0 骨架** —— 本仓库仅搭建项目骨架与基础 API 占位，不实现核心解析和南湖调用（v0.2 / v0.3 由其他 worker 推进）。

---

## 1. 功能概览（roadmap）

| 版本  | 内容 |
| ----- | ---- |
| v0.1.0 | 项目骨架：FastAPI 入口、CORS、StaticFiles、错误码、配置、日志、模板、SQLite 元数据 |
| v0.2   | 任务表（rollout/Job 元数据入库）、dump 扫描与摘要接口 |
| v0.3   | 南湖（nanhu）作业调度集成、解析与一键重放 |

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
├── utils/                     # 工具：配置、日志、解析辅助
│   ├── config_loader.py
│   ├── log_setup.py
│   └── analyze_dump.py        # 由并行 worker 提供，封装 inspect-rollout-trajectory skill
├── data/                      # 数据层
│   ├── response.py            # 统一 JSON 响应（code/msg/data）
│   └── rollout_insight_exception.py   # 业务异常（50xxx/51xxx/52xxx/53xxx/54xxx）
├── controller/                # FastAPI 路由层
│   └── task_controller.py     # 仅占位：/api/tasks
├── service/                   # 业务逻辑层（占位）
├── db/                        # SQLAlchemy engine/session/Base
├── schedule/                  # 定时任务占位
├── static/                    # 静态资源（.gitkeep）
├── templates/                 # Jinja2 模板（base.html）
├── tests/                     # 单元测试
│   ├── test_smoke.py          # 基础 import smoke 测试
│   └── test_analyze_dump.py   # 由并行 worker 提供
└── log/                       # 运行日志（默认被 .gitignore）
```

---

## 5. API 速览（v0.1.0）

| 方法 | 路径 | 说明 |
| ---- | ---- | ---- |
| GET  | `/` | 中文欢迎页（自动 3 秒跳转到 `/api/tasks`，亦可直接 `/welcome` 302 跳转） |
| GET  | `/api/tasks` | 任务列表（v0.1.0 仅返回版本号与占位） |

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
