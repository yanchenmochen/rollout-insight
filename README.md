# rollout-insight

Slime rollout 落盘轨迹分析与检索 Web 平台（v0.1.0 骨架版）。

## 目标

- 解析 `/mnt/zj-data/data/songquanheng/output/slime/*/rollout_dumps/rollout_*.pt` 落盘数据
- 在 Web 页面上查看任务、Attempt、Segment 摘要
- 渲染每条 segment 的多轮对话（trajectory）
- 接入南湖调度，关联训练任务与 rollout 数据

## 架构

- 后端：FastAPI + SQLAlchemy + SQLite
- 前端：Jinja2 服务端渲染
- 复用：inspect-rollout-trajectory skill（`~/.cursor/skills/inspect-rollout-trajectory/`）
- 后续：nanhu-skill（`~/.cursor/skills/nanhu-skill/`）

参考 `/mnt/zj-data/data/songquanheng/Shell-dsv2/automate` 项目的分层：
`controller → service → db/*_service → ORM`。

## 目录

```
rollout-insight/
├── main.py              # FastAPI 入口
├── config.json          # 全局配置
├── requirements.txt     # 依赖
├── controller/          # FastAPI 路由
├── service/             # 业务流程（v0.2+ 启用）
├── db/                  # ORM + db_*_service
├── data/                # DTO / 异常 / Response
├── utils/               # 配置 / 日志 / 分析工具
├── templates/           # Jinja2 模板
├── static/              # 静态资源
├── schedule/            # 后台任务（v0.2+ 启用）
├── tests/               # 单元测试
├── log/                 # 日志
└── .cursor/rules/       # Cursor 规则（9 份）
```

## 启动

```bash
cd /mnt/zj-data/data/songquanheng/rollout-insight
pip install -r requirements.txt   # v0.1.0 推荐：fastapi uvicorn jinja2 pydantic SQLAlchemy loguru
python main.py                    # 默认 39998
# 浏览器访问 http://localhost:39998/
```

自定义端口：`ROLLOUT_INSIGHT_PORT=40000 python main.py`

## 版本路线

| Tag | 范围 |
|---|---|
| **v0.1.0** | 脚手架、9 份 Cursor 规则、`utils/analyze_dump.py` |
| v0.2.0 | 5 张 SQLite 表（task/rollout/attempt/segment/segment_transcript）+ 任务管理 UI + 多轮对话页 |
| v0.3.0 | 南湖调度接入 + Cursor skill 化 + CLI |

## 关联资源

- 参考项目：`/mnt/zj-data/data/songquanheng/Shell-dsv2/automate`
- 分析脚本：`~/.cursor/skills/inspect-rollout-trajectory/scripts/view_trajectory.py`
- 南湖客户端：`~/.cursor/skills/nanhu-skill/`

## 响应语言

始终使用简体中文。
