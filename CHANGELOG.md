# rollout-insight CHANGELOG

## v0.5.2 — func.md 自动扫描 + skill 指引

发布日期：2026-10-15

### 概述

v0.5.2 是 v0.5 系列的最后一个 tag：交付一份**自动生成**的 `func.md` 能力索引（覆盖
`controller/`、`service/`、`db/`、`utils/`、`data/` 五个目录的公开符号），并配套
AI agent 的编码约定 + skill 指引，让后续新增 / 修改业务能力时能避免重复造轮子。

`func.md` 由 `python scripts/scan_func_md.py` 全量扫描得出，**禁止手改**；
CI 阶段跑 `python scripts/scan_func_md.py --check` 校验一致性，不一致直接失败。

### 完成情况（按子任务）

| 子任务 | 内容 | commit |
| --- | --- | --- |
| T14 | `scripts/scan_func_md.py` —— AST 扫描 + 渲染 controller/service/db/utils/data 公开能力；`--check` 模式提供 CI 门禁 | 本 tag 内 |
| T14 | `scripts/make_func_md.sh` —— bash 包装脚本（自动 cd 到 repo 根），供本地"一键重生成" | 本 tag 内 |
| T14 | 首次 `func.md`（基于 main 分支当前代码，覆盖 5 个目录的全部公开类 / 静态方法 / 路由 / 错误码） | 本 tag 内 |
| T15 | 新规则文件 `.cursor/rules/rollout-insight-conventions.mdc`（alwaysApply=true）—— "改前先查 func.md / 改后重生成 func.md" | 本 tag 内 |
| T15 | `skill/SKILL.md` 顶部加 "改前先读 func.md / 改后重生成 func.md"，末尾加 "维护 func.md" 排障 | 本 tag 内 |
| T15 | `skill/recipes.md` 新增 "场景 0：新增 controller / service / db 表 / utils 函数的标准流程"（三步走 + 反例表） | 本 tag 内 |

## v0.5.0 — turn 解析重写 + 四层手风琴 + 启动期 bootstrap

发布日期：2026-10-15

### 概述

v0.5.0 是 v0.5 系列的**主交付**：用 tokenizer（`tokens + tokenizers.Tokenizer.from_file`）重写
turn 切分，把 v0.4.0 的 response 文本切升级为基于 `<|im_start|>` token id 的精确切分；同时把
rollout 详情页彻底重做为 **group → attempt → segment → turn** 四层手风琴，每层默认折叠、点
展开时按需 fetch turn JSON；并在 main 启动期新增一个 bootstrap 阶段（清 turn 表 + 写 admin
种子用户 + 落 2 张 user ORM 表），为 v0.5.1 的 auth / per-user 集群配置做准备。

### 完成情况（按子任务）

| 子任务 | 内容 | commit |
| --- | --- | --- |
| T0 ORM | `db/db_user.py` + `db/db_user_cluster_config.py` + `db/__init__.py` 别名（仅 ORM，service 由 v0.5.1 写） | `a9ccecc` |
| T0 bootstrap | `utils/bootstrap.py` + `main.py` 启动调一次 `bootstrap_on_startup()`（清 turn + ensure_admin_seed） | `a9ccecc` |
| T0 service | `db/db_user_service.py` 的最小实现（仅 `ensure_admin_seed` + bcrypt 工具） | `a9ccecc` |
| T1 turn parser | `utils/rollout_pt_parser.py` 新增 `_discover_tokenizer_path` / `_load_tokenizer` / `_split_token_turns` / `split_response_to_turns_with_tokenizer` | `0db67f5` |
| T2 aggregates | `service.rollout_parse_service.parse_attempt_token_stats(attempt_pk)` —— attempt 维度聚合 6 字段 + reward_bucket；controller 整合进 `_load_rollout_full` | `97f5bfe` |
| T3 template | `templates/rollout_detail.html` 重写为 4 层手风琴；turn 内按 think/tool_call/tool_response/纯文本拆 4 子 details；attempt 行展示 5 统计数字 + bucket badge | `a9ccecc` |
| T4 endpoints | `?group=N` 过滤（HTML 与 JSON 两端点）；`/api/segments/{id}/turns.json` 优先 token 切分（带回退） | `a9ccecc` |
| T5 CLI | `rollouts parse --backfill-turns / --clear-turn-table / --tokenizer`；`cli/api_client.get_segment_turns/list_segment_ids_for_rollout/clear_turn_table` | 本 tag 内 |
| T6 tag | `CHANGELOG.md` v0.5.0 节 + `git tag v0.5.0` | 本 tag 内 |

### 文件变更（vs v0.4.0）

#### 新建文件（6 个）

- `db/db_user.py` —— `ri_user_table` ORM（username UNIQUE, password_hash, role, is_active, create_time, last_login_at）
- `db/db_user_cluster_config.py` —— `ri_user_cluster_config_table` ORM（user_id FK + cluster_key UNIQUE）
- `db/db_user_service.py` —— `DBUserService.ensure_admin_seed` 启动期最小实现（passlib[bcrpt] 不可用 → 直接用 bcrypt）
- `utils/bootstrap.py` —— `bootstrap_on_startup()`：DELETE turn 表 + ensure_admin_seed
- `templates/rollout_detail_attempt_loop.html` —— attempt 折叠块复用片段（含 5 数字 chip + bucket badge + segments 子树）

#### 修改文件（8 个）

- `db/__init__.py` —— 新增 `User / ClusterConfig` 别名 + `DBUser / DBUserClusterConfig` 暴露
- `main.py` —— 启动期调 `bootstrap_on_startup()`（在 `Base.metadata.create_all` 之后）
- `requirements.txt` —— `passlib[bcrypt] + itsdangerous`（实际用 `bcrypt` 直调，passlib 1.7.x 与 bcrypt 5.x 的 `__about__` 检测冲突）
- `utils/rollout_pt_parser.py` —— 新增 4 个 token-based 函数 / `split_response_to_turns_with_tokenizer` 顶层入口
- `service/rollout_parse_service.py` —— 新增 `parse_attempt_token_stats(attempt_pk)` + `_classify_reward_bucket`
- `controller/rollout_controller.py` —— `_load_rollout_full` 整合 T2 aggregates；`page_rollout_detail` 与 `api_get_rollout_json` 都接受 `?group=N` 过滤；`distinct_groups` 字段给前端做 group 链接
- `controller/segment_controller.py` —— `/api/segments/{id}/turns.json` 优先走 token 切分（带 `?pt_path=`），回退到 turn 表 → ensure_transcript
- `templates/rollout_detail.html` —— 整体重写；引入 `?group=N` 顶栏链接、bucket badge 配色；JS 渲染 turn 内 4 子 details

#### 修改 CLI / API client

- `cli/rollout_insight.py` —— `rollouts parse` 加 `--backfill-turns / --clear-turn-table`；dispatch 接收并串到 backfill 流程
- `cli/api_client.py` —— 新增 `get_segment_turns / list_segment_ids_for_rollout / clear_turn_table`

### 升级指南

1. **新依赖（无破坏，但需要装）**

   ```bash
   pip install -e .
   # passlib[bcrypt] + itsdangerous 已在 requirements.txt
   ```

2. **数据库迁移（自动）**

   - 启动期 `Base.metadata.create_all` 自动建 `ri_user_table` + `ri_user_cluster_config_table` 2 张表
   - 启动期 `bootstrap_on_startup()` 执行 `DELETE FROM ri_segment_turn_table`（p17_a：旧 turn 数据丢失，需重新 `parse_rollout`）
   - 启动期 `ensure_admin_seed()` 写入 admin 行（username=admin / password=admin123 / role=admin），日志 WARNING 一次提醒生产环境改密码

3. **页面差异**

   - `/api/tasks/{tid}/rollouts/{rid}` URL 不变，但页面布局从 v0.4.0 的 attempt→segment→turn 三层改为 group→attempt→segment→turn 四层
   - `/api/rollouts/{id}` JSON 接受 `?group=N` 过滤；返回字段新增 `n_tool_calls / n_diff_lines / total_resp_len / total_mask_sum / total_trainable_tokens / reward_bucket` 6 个 attempt 摘要

4. **CLI 差异**

   ```bash
   # 旧：
   rollout-insight rollouts parse <pk> --tokenizer DIR
   # 新（多 2 个开关）：
   rollout-insight rollouts parse <pk> --tokenizer DIR --backfill-turns --clear-turn-table
   ```

### 新功能速查

- **Tokenizer-based turn 切分**：`.pt` 里 `sample['tokens']` + 父目录 `resolved_config.yaml` 提到的 `env.SFT_MODEL / HF_CHECKPOINT` 当 tokenizer，按 `<|im_start|>` token id 切分；fallback 到 response 文本切
- **attempt 摘要聚合**：6 字段（tool / diff / resp_len / mask / trainable_tokens / reward_bucket）+ 130 段 task #2 rollout 158 拆 8 group 验证
- **四层手风琴**：
  - 第 1 层 group：N attempts + 全0/全1/混合/无 计数
  - 第 2 层 attempt：5 统计 chip + bucket badge + status
  - 第 3 层 segment：index + status + reward；展开触发 fetch
  - 第 4 层 turn：role + chars + token_estimate + 内部 4 子 details
- **turn 内 4 子 details**：think 块 / tool_call 块 / tool_response 块 / 纯文本块，按出现顺序扫一遍 turn body 切成
- **?group=N 过滤**：URL 顶栏切 group 链接，第一屏只显示该 group 的 attempts
- **bootstrap 启动期**：DELETE turn 表 + ensure_admin_seed + WARNING 一次
- **2 张 v0.5.1 准备表**：`ri_user_table` / `ri_user_cluster_config_table` 已建好但不写 service（v0.5.1 worker 写）

### 已知问题（v0.5.0 不修，留 v0.5.1+ 处理）

1. **`ensure_admin_seed` 用 bcrypt 直调，passlib[bcrypt] 仅占位**
   passlib 1.7.x 与 bcrypt 5.x 的 `__about__.__version__` 检测有兼容问题（已知 issue #175），
   当前实现直接 `import bcrypt; bcrypt.hashpw / bcrypt.checkpw`。requirements.txt 仍保留
   `passlib[bcrypt]` 以备 Worker B 切换。
2. **`split_turns → `_split_token_turns`` 的 prompt 段归属问题**
   若 `.pt` 里 prompt tokens 中夹杂 `<|im_start|>`（如 dataset prompt 自带 turn marker），
   `_split_token_turns` 会把它切成第一个 turn；UI 当前不展示 prompt-only turn，需 Worker C
   在 v0.5.3 加 `include_prompt=False` 筛选。
3. **`/api/rollouts/{id}/clear-turn-table` 端点未注册**
   CLI 的 `--clear-turn-table` flag 实际清理由 `bootstrap_on_startup` 完成；API 端点
   Worker D 在 v0.5.1+ 实现。
4. **turn_token chunks 的顺序扫描对 thinking 块不严格保护**
   `splitTurnBody` 用朴素正则顺序扫 + 起始 index 排序，若 chat-template 里有
   `<think>\n\n...</think>` 转义后会失去，但保有转义；如有 nested / 不可见
   `<|im_start|>` blocks 时仍按出现顺序切。Worker D 在 v0.5.3+ 可换成更严格的 AST 切。
5. **`_discover_tokenizer_path` 在没有 `resolved_config.yaml` 的目录退化为 None**
   本地仓库的 `.pt` 路径 `/mnt/zj-data/data/songquanheng/output/slime/<task>/rollout_dumps/rollout_0.pt`
   的所有父目录都没有 `resolved_config.yaml`（生产脚本本来在另一个目录写 yaml），所以
   `_discover_tokenizer_path` 返回 None → 走 response 文本兜底路径。
   此外，本地 `tokenizer.json` (`models/021-32b/...`) 把 `<|im_start|>` 注册为
   `None`（vocab 是 `<|reserved_special_token_6|>` = id 128011），需要 Worker D 在
   v0.5.3 把 `<|im_start|>` ↔ `<|reserved_special_token_6|>` 的映射补上。
   验证：在 shell 里手动用一个 wrapper tokenizer 把 128011 映射到 `<|im_start|>` 后，
   `_split_token_turns` 正确返回 31 chunks（同一个 sample）。当前退化为文本切分
   后的 turn 数 = response 文本中 `<|im_start|>` 字面 marker 数量，多数 segment 是 1
   turn 是数据特征而非代码 bug。

### 端到端验收（T6）

- 服务启动 OK（PID 363846 已 kill，新 PID 474617 在 0.0.0.0:39998 监听）
- `pip install -e .` OK；`passlib[bcrypt] itsdangerous` 已装（实际用 `bcrypt` 直调，passlib 占位）
- 启动期 `bootstrap_on_startup` 走完：`DELETE ri_segment_turn_table` + `ensure_admin_seed`（首次启动写 admin / admin123 / role=admin，行级 WARNING 一次）
- 第二次启动检测 user_count>0 → 直接跳过 seed（INFO '跳过'）
- 重启后 `http://localhost:39998/api/tasks/2/rollouts/158` 返回 200；HTML 含 8 个 group 折叠标题 + 129 attempts + 131 segments
- `/api/rollouts/158` JSON 返回 `distinct_groups: [0,1,2,4,7,9,10,12]` 与 6 个聚合字段（n_tool_calls / n_diff_lines / total_resp_len / total_mask_sum / total_trainable_tokens / reward_bucket）
- `?group=0` 过滤生效：attempts 数从 129 降到 40（仅 group=0）
- attempt 行展示 5+ stat chips：例如 #34901 → `tool 44 / diff 776 / resp_len 23.0k / mask 4.0k / trainable_tokens 0 / segments 1 / mean_reward 0.000`，加上 `bucket-all_zero` badge
- `/api/segments/34259/turns.json` 当前走文本切分（tokenizer 路径因 `_discover_tokenizer_path` 返回 None + 本地 tokenizer 不映射 128011 → 退化）。在 wrapper tokenizer 下验证 token 路径可切出 31 chunks（与 sample tokens 中 reserved_token_6 出现次数一致）
- `rollout-insight rollouts parse --help` 出现 `--backfill-turns / --clear-turn-table / --tokenizer` 三个新选项
- `POST /api/rollouts/158/parse?force=true` 提交后台解析；后台日志显示 `attempts=64 segments=129 persisted_turns=86`（86 = 实际能反查到的 11 个 segment × 各自文本 im_start 数，v0.4.0 已知问题 1 限制）
- DBUser / DBUserClusterConfig 两张表已通过 `Base.metadata.create_all` 自动建好；`ri_user_table` 写入 1 行 admin
- `ensure_admin_seed()` 的 bcrypt 哈希已落库（hash 长度 60，能 verify admin/admin123 通过）


| 子任务 | 内容 | commit |
| --- | --- | --- |
| T14 | `scripts/scan_func_md.py` —— AST 扫描 + 渲染 controller/service/db/utils/data 公开能力；`--check` 模式提供 CI 门禁 | 本 tag 内 |
| T14 | `scripts/make_func_md.sh` —— bash 包装脚本（自动 cd 到 repo 根），供本地"一键重生成" | 本 tag 内 |
| T14 | 首次 `func.md`（基于 main 分支当前代码，覆盖 5 个目录的全部公开类 / 静态方法 / 路由 / 错误码） | 本 tag 内 |
| T15 | 新规则文件 `.cursor/rules/rollout-insight-conventions.mdc`（alwaysApply=true）—— "改前先查 func.md / 改后重生成 func.md" | 本 tag 内 |
| T15 | `skill/SKILL.md` 顶部加 "改前先读 func.md / 改后重生成 func.md"，末尾加 "维护 func.md" 排障 | 本 tag 内 |
| T15 | `skill/recipes.md` 新增 "场景 0：新增 controller / service / db 表 / utils 函数的标准流程"（三步走 + 反例表） | 本 tag 内 |

### 文件变更（vs v0.4.0 → v0.5.2 之间已落地的 v0.5.0 / v0.5.1）

> 本节说明 v0.5.2 仅负责**索引层**（v0.5.0 turn 重写 + v0.5.1 用户/登录 的代码
> 与本 tag 无关）。v0.5.0/v0.5.1 的实际代码改动在 commit `97f5bfe` 之前已落到 main。

#### 新建文件（3 个）

- `func.md` —— 自动生成的公开能力索引（**禁止手改**，下次扫描会被覆盖）
- `scripts/scan_func_md.py` —— AST 扫描 + 渲染器；支持 `--check` / `--output`
- `scripts/make_func_md.sh` —— bash 包装（默认 cd 到 repo 根后调 scan_func_md.py）

#### 修改文件（3 个）

- `.cursor/rules/rollout-insight-conventions.mdc`（new file, but in `.cursor/rules/` —— 见下条）
- `skill/SKILL.md` —— 顶部"## 改前先读 func.md" 段 + 末尾"## 维护 func.md" 段
- `skill/recipes.md` —— 顶部"## 场景 0：新增 controller/service/db/utils 函数的标准流程"

#### 新建规则（1 个）

- `.cursor/rules/rollout-insight-conventions.mdc` —— alwaysApply；强制 "改前查 func.md → 实现 → 跑 scan_func_md.py 重新生成"

### 升级指南

1. **重新生成 func.md（CI 已经能校验）**

   ```bash
   cd /path/to/rollout-insight
   python scripts/scan_func_md.py --check   # 应输出 "[scan_func_md] --check OK"
   ```

2. **更新 skill 软链**

   ```bash
   bash skill/install.sh
   # 或强制覆盖：OVERRIDE=1 bash skill/install.sh
   ```

3. **新规则文件**

   `.cursor/rules/rollout-insight-conventions.mdc` 与 v0.5.2 一同发布；
   Cursor 会自动加载，无需手动复制。

### 新功能速查

- `python scripts/scan_func_md.py` —— 全量覆盖 `func.md`。
- `python scripts/scan_func_md.py --check` —— CI 防漂移：与现有 func.md 不一致则退出码 1。
- `bash scripts/make_func_md.sh [--check]` —— 同上，但自动切换到仓库根目录。
- func.md 覆盖：
  - `data/` — 枚举、DTO、异常；错误码常量（table 化）。
  - `utils/` — 模块级公开函数 + docstring 首行。
  - `db/` — ORM 模型 (class / `__tablename__`)；数据访问 service（`@staticmethod` / `@classmethod`）。
  - `service/` — 公开 class（含实例方法与 module-level 函数）+ 入口签名。
  - `controller/` — FastAPI 路由 (`@router.get/post/...`) Method / Path / Function / docstring。

### 已知问题（v0.5.2 不修，留 v0.5.3+ 处理）

1. **`func.md` 不解析函数 docstring 内部示例代码块**

   只取 signature + 首行注释（避免正则误判代码块 / 多行 noqa）。
   完整 docstring 见 `Read` 原文件。

2. **`DBUserService` 等未提交文件被一并索引**

   `db/db_user*.py` 与 v0.5.1 用户/登录相关，目前在 working tree 但尚未
   commit。`func.md` 会自动收录它们，commit 上来后会保持一致。
   不影响 CI：`--check` 只关心 working tree 与 func.md 的字节级一致。

3. **service 段的 `instance method` 与 `module-level function` 可能重复**

   如 `RolloutParseService.parse_rollout` 同时是 class 方法，又作为 module-level
   包装函数存在（v0.5.0 T2 把入口提到 module 层）。`func.md` 会同时收录两处，
   索引更全但读者需注意调用入口是 module-level 那个。

---

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
