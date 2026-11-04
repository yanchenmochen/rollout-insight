# rollout-insight Playwright 用例

服务需已启动：`python main.py`（默认 `http://127.0.0.1:39998`）。
默认账号 `admin / admin123`。

```bash
pip install -r tests/e2e/requirements-e2e.txt
python -m playwright install chromium
cd /mnt/zj-data/data/songquanheng/rollout-insight
pytest tests/e2e -m e2e --browser chromium
```

有界面：`ROLLOUT_INSIGHT_E2E_HEADED=1 pytest tests/e2e -m e2e --browser chromium`

| 模块 | 文件 | 用例 | 断言 |
| --- | --- | --- | --- |
| 登录 | `test_auth.py` | 未登录访问 `/` | 跳到 `/api/auth/login` |
| 登录 | `test_auth.py` | 错误密码 | 出现 `.auth-banner` |
| 登录 | `test_auth.py` | admin 登录 | 进入任务列表，顶栏有「退出」 |
| 登录 | `test_auth.py` | 退出 | 回到登录页 |
| 登录 | `test_auth.py` | `GET /api/auth/me` | JSON 里用户名是 admin |
| 任务 | `test_tasks.py` | 任务列表 | `#taskTableBody`、筛选表单可见 |
| 任务 | `test_tasks.py` | 打开第一条任务 | `#rollouts` 可见 |
| 任务 | `test_tasks.py` | 创建页 | 任务名、rollout_root、提交按钮可见（不提交） |
| Rollout | `test_rollouts.py` | 点任务详情里的 rollout ID | `#treeRoot` 下至少 1 个 group |
| Rollout | `test_rollouts.py` | 展开 group → attempt | 摘要含 tool；prompt 块可展开 |
| Segment | `test_segments.py` | 展开 segment | 发出 `turns.json` 且 200，出现 turn 或空提示 |
| 集群配置 | `test_cluster_config.py` | 新建后删除 | 列表出现临时 `e2e-*`，删除后消失 |
| 南湖 | `test_nanhu.py` | 作业页 | 过滤表单渲染（不点同步/重启） |
| Agent transcript | `test_agent_transcripts.py` | 报告页 | 标题与正文含报告/token 字样 |

写操作（创建任务、重扫、重解析 `.pt`、南湖 restart）默认不放进冒烟，避免扫盘和打调度。
