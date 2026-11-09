# rollout-insight 能力索引 (func.md)

> **用途**：重构或新增业务代码前，必须先查阅本文档，确认是否已有可复用实现。  
> **维护**：新增/修改 `controller`、`service`、`db/*_service`、`data`、`utils` 中的公开能力后，必须重跑 `python scripts/scan_func_md.py` 重新生成。  
> **版本**：与 v0.5.2 同步生成；CI 阶段跑 `python scripts/scan_func_md.py --check` 校验一致性。

## 查阅顺序（自上而下）

| 顺序 | 目录 | 何时查 |
| --- | --- | --- |
| 1 | `data/` | 枚举、DTO、异常码、响应封装 |
| 2 | `utils/` | 配置、路径、日志、解析、HTTP 等无状态工具 |
| 3 | `db/db_*_service.py` | 表读写、状态流转、查询组合 |
| 4 | `service/` | 跨表业务流程、外部平台对接 |
| 5 | `db/db_*.py` | ORM 模型与字段 |
| 6 | `controller/` | 已有 REST 路由，避免重复端点 |

## data/ — 枚举、DTO、异常

### `data/response.py`

- **Response** — 继承 BaseModel；统一的 JSON 返回格式。

### `data/rollout_insight_exception.py`

- **RolloutInsightException** — 继承 Exception；RolloutInsight 业务异常，支持 (code, name, message) 三元组作为 error 参数。

  错误码常量：

  | 常量 | 错误码 | 名称 | 默认消息 |
  | --- | --- | --- | --- |
  | `RolloutInsightException.GENERAL_ERROR` | 50000 | `GENERAL_ERROR` | RolloutInsight 业务异常 |
  | `RolloutInsightException.INVALID_ARGUMENT` | 50001 | `INVALID_ARGUMENT` | 参数非法或缺失 |
  | `RolloutInsightException.NOT_IMPLEMENTED` | 50002 | `NOT_IMPLEMENTED` | 功能尚未实现 |
  | `RolloutInsightException.TASK_ERROR` | 51000 | `TASK_ERROR` | 任务处理异常 |
  | `RolloutInsightException.TASK_NOT_FOUND` | 51001 | `TASK_NOT_FOUND` | 任务不存在 |
  | `RolloutInsightException.TASK_VALIDATION` | 51002 | `TASK_VALIDATION` | 任务参数校验失败 |
  | `RolloutInsightException.TASK_STATE_INVALID` | 51003 | `TASK_STATE_INVALID` | 任务状态非法，不允许该操作 |
  | `RolloutInsightException.TASK_DELETE_FORBIDDEN` | 51004 | `TASK_DELETE_FORBIDDEN` | 任务处于运行中，禁止删除 |
  | `RolloutInsightException.TASK_CREATE_FAILED` | 51005 | `TASK_CREATE_FAILED` | 创建任务失败 |
  | `RolloutInsightException.PARSE_ERROR` | 52000 | `PARSE_ERROR` | 解析处理异常 |
  | `RolloutInsightException.PARSE_DUMP_NOT_FOUND` | 52001 | `PARSE_DUMP_NOT_FOUND` | rollout dump 文件不存在 |
  | `RolloutInsightException.PARSE_DUMP_CORRUPTED` | 52002 | `PARSE_DUMP_CORRUPTED` | rollout dump 文件损坏 |
  | `RolloutInsightException.PARSE_TRANSCRIPT_FAILED` | 52003 | `PARSE_TRANSCRIPT_FAILED` | transcript 渲染失败 |
  | `RolloutInsightException.PARSE_SAMPLE_OUT_OF_RANGE` | 52004 | `PARSE_SAMPLE_OUT_OF_RANGE` | sample 索引越界 |
  | `RolloutInsightException.PARSE_TOKENIZER_FAILED` | 52005 | `PARSE_TOKENIZER_FAILED` | tokenizer 加载失败 |
  | `RolloutInsightException.NANHU_ERROR` | 53000 | `NANHU_ERROR` | 南湖调度异常 |
  | `RolloutInsightException.NANHU_VALIDATION` | 53001 | `NANHU_VALIDATION` | 南湖作业参数校验失败 |
  | `RolloutInsightException.NANHU_JOB_CREATE_FAILED` | 53002 | `NANHU_JOB_CREATE_FAILED` | 南湖创建作业失败 |
  | `RolloutInsightException.NANHU_JOB_STOP_FAILED` | 53003 | `NANHU_JOB_STOP_FAILED` | 南湖终止作业失败 |
  | `RolloutInsightException.NANHU_JOB_RESTART_FAILED` | 53004 | `NANHU_JOB_RESTART_FAILED` | 南湖重启作业失败 |
  | `RolloutInsightException.NANHU_RESOURCE_QUERY_FAILED` | 53005 | `NANHU_RESOURCE_QUERY_FAILED` | 南湖资源查询失败 |
  | `RolloutInsightException.NANHU_ALGORITHM_NOT_FOUND` | 53006 | `NANHU_ALGORITHM_NOT_FOUND` | 南湖算法镜像未找到 |
  | `RolloutInsightException.NANHU_JOB_NOT_FOUND` | 53007 | `NANHU_JOB_NOT_FOUND` | 南湖作业未找到 |
  | `RolloutInsightException.SYSTEM_ERROR` | 54000 | `SYSTEM_ERROR` | 系统异常 |
  | `RolloutInsightException.DB_ERROR` | 54001 | `DB_ERROR` | 数据库异常 |
  | `RolloutInsightException.DB_CONNECT_FAILED` | 54002 | `DB_CONNECT_FAILED` | 数据库连接失败 |
  | `RolloutInsightException.IO_ERROR` | 54003 | `IO_ERROR` | IO 异常 |
  | `RolloutInsightException.CONFIG_ERROR` | 54004 | `CONFIG_ERROR` | 配置异常 |
  | `RolloutInsightException.CONFIG_NOT_FOUND` | 54005 | `CONFIG_NOT_FOUND` | 配置项未找到 |

## utils/ — 通用工具（优先复用）

### `utils/analyze_dump.py`

- `def load_dump(path: str | Path) -> Dict[str, Any]` — 加载完整 .pt dump dict（优先 zip/pickle，不依赖可 import 的 torch）。
- `def load_dump_meta(path: str | Path) -> Dict[str, Any]` — 读取 .pt 顶部元数据：rollout_id、n_samples、n_attempts、mean_reward。
- `def summarize_dump(path: str | Path) -> List[Dict[str, Any]]` — 返回每个 sample 的摘要行（与 view_trajectory 表格一致）。
- `def split_turns(text: str) -> List[tuple[str, str]]` — 按 ``<|im_start|>role\n...<|im_end|>`` 切分为 ``(role, content)``。
- `def render_transcript(path: str | Path, sample_idx: int, tokenizer_dir: Optional[str]=None, inc…` — 渲染一个 sample 的完整 transcript（Markdown）。

### `utils/auth_session.py`

- `def get_session_secret(config_path: Optional[str]=None) -> str` — 从 config.json 读 ``session_secret``；首次启动随机生成并写回。
- `def install_session_middleware(app: Any) -> None` — 在 ``app`` 上挂 ``SessionMiddleware``（max_age = 7 天）。
- `def wants_json_response(request: Request) -> bool` — 基于 Accept 头判断 caller 期望 JSON 而不是 HTML。
- `def current_user(request: Request) -> Optional[Any]` — 从 ``request.session["user_id"]`` 查 DBUser；没登录返回 None。
- `def require_login(request: Request) -> Any` — 依赖：未登录 → 302（HTML）或 401 JSON；否则返回当前 DBUser。
- `def require_admin(request: Request) -> Any` — 依赖：未登录 → 同 ``require_login``；非 admin → 403 JSON；否则返回 DBUser。
- `def set_login_session(request: Request, user_id: int) -> None` — 把 user_id 写进 session；FastAPI SessionMiddleware 会在响应时签 cookie。
- `def clear_session(request: Request) -> None` — 清掉 user_id（其它自定义 key 保留；如果要全清用 ``request.session.clear()``）。

### `utils/bootstrap.py`

- `def bootstrap_on_startup() -> Dict[str, Any]` — v0.5.0 启动期入口：

### `utils/config_loader.py`

- `def load_config(config_path: Optional[str]=None) -> Dict` — 加载配置文件，并基于 program_root 计算其他路径。

### `utils/log_setup.py`

- `def setup_rollout_insight_logging(log_base: str='') -> None` — 配置 rollout-insight 的 loguru 输出。
- `def ensure_logging_configured() -> None` — 幂等地初始化 logging，避免被反复执行。

### `utils/nanhu_job_path_resolve.py`

- `def env_map_from_job(job: Mapping[str, Any]) -> Dict[str, str]` — 把 ``envVars`` / ``env_vars`` 列表转成 ``{KEY: value}``。
- `def output_dir_from_job(job: Mapping[str, Any], env: Optional[Mapping[str, str]]=None) -> str` — 解析作业输出根目录（不含 ``rollout_dumps`` 子路径）。
- `def resolve_paths_from_job_detail(job: Mapping[str, Any], *, prefer_existing_dirs: bool=True) -…` — 从作业详情 dict 解析创建任务字段。
- `def extract_job_payload(detail: Mapping[str, Any]) -> Optional[MutableMapping[str, Any]]` — 从 ``get_job_detail`` 返回体取出作业 dict；失败返回 ``None``。

### `utils/rollout_pt_parser.py`

- `def load_pt(path: str) -> Dict[str, Any]` — 加载 .pt dump：优先 zip/pickle，不要求环境能 ``import torch``。
- `def resolve_sample_reward(sample: Dict[str, Any]) -> Optional[float]` — 解析单条 sample 的可测 reward。
- `def sample_abort_reason(sample: Dict[str, Any]) -> Optional[str]` — 若 sample 为 abort，返回聚类用原因字符串；否则 None。
- `def cluster_abort_reasons(dump: Dict[str, Any], *, group_index: Optional[int]=None) -> Dict[str…` — 对 dump 内 abort attempt 按 ``abort_reason`` 聚类（兼容旧调用）。
- `def classify_sample_outcome(sample: Dict[str, Any]) -> Dict[str, str]` — 把单条 sample 归为 pass / abort / timeout / fail / other。
- `def cluster_attempt_outcomes(dump: Dict[str, Any], *, group_index: Optional[int]=None) -> Dict[…` — 跨组 / 组内失败情形聚类（abort / timeout / 正常失败 / 通过）。
- `def samples_index_for(_sample: Dict[str, Any]) -> int` — 占位：见 _segment_payload 中的 try/except；这里返回 0 让 except 兜底。
- `def parse_attempts_and_segments(dump: Dict[str, Any]) -> Dict[str, Any]` — 把 dump 拆成 attempts + segments + 整文 response。
- `def split_response_to_turns(response_text: str) -> List[Dict[str, Any]]` — 按 ``<|im_start|>role\n...<|im_end|>`` 切分整文 response。
- `def split_markdown_headed_turns(text: str) -> List[Dict[str, Any]]` — 把 ``### turn N · role`` 标题切成多轮。
- `def token_estimate(text: str, chars_per_token: float=3.5) -> int` — 粗估 token 数：默认按字符数 / 3.5。
- `def split_response_to_turns_with_tokenizer(response_text: str='', *, tokens: Optional[Sequence[…` — v0.5.0 T1：优先用 tokens + tokenizer 切分；否则 fallback 到 response 文本。

## db/ — ORM 模型

| ORM 类 | 文件 | 物理表 |
| --- | --- | --- |
| `DBAttempt` | `db/db_attempt.py` | `ri_attempt_table` |
| `DBNanhuJob` | `db/db_nanhu_job.py` | `ri_nanhu_job_table` |
| `DBNanhuRestart` | `db/db_nanhu_restart.py` | `ri_nanhu_restart_table` |
| `DBRollout` | `db/db_rollout.py` | `ri_rollout_table` |
| `DBSegment` | `db/db_segment.py` | `ri_segment_table` |
| `DBSegmentTranscript` | `db/db_segment_transcript.py` | `ri_segment_transcript_table` |
| `DBSegmentTurn` | `db/db_segment_turn.py` | `ri_segment_turn_table` |
| `DBTask` | `db/db_task.py` | `ri_task_table` |
| `DBUser` | `db/db_user.py` | `ri_user_table` |
| `DBUserClusterConfig` | `db/db_user_cluster_config.py` | `ri_user_cluster_config_table` |

## db/ — 数据访问服务

#### `DBAttemptService` (`db/db_attempt_service.py`)

- `classmethod` `def upsert(**fields: Any) -> DBAttempt`
- `staticmethod` `def list_by_rollout(rollout_id: int, limit: int=500, offset: int=0) -> List[DBAttempt]`
- `staticmethod` `def list_by_task(limit: int=1000, offset: int=0) -> List[DBAttempt]`
- `staticmethod` `def get_by_id() -> Optional[DBAttempt]`
- `staticmethod` `def delete_by_rollout(rollout_id: int) -> int`
- `staticmethod` `def delete_by_task() -> int`
- `classmethod` `def summarize_reward_metrics_by_rollout(task_id: int) -> Dict[int, Dict[str, int]]`
- `classmethod` `def summarize_reward_metrics_by_group(task_id: int, rollout_id: int) -> Dict[int, Dict[str, Any…`

#### `DBNanhuJobService` (`db/db_nanhu_job_service.py`)

- `staticmethod` `def upsert(**fields: Any) -> DBNanhuJob`
- `staticmethod` `def get_by_job_id() -> Optional[DBNanhuJob]`
- `staticmethod` `def get_by_id() -> Optional[DBNanhuJob]`
- `staticmethod` `def list_jobs(status: Optional[str]=None, keyword: Optional[str]=None, limit: int=200, offset: …`
- `staticmethod` `def update_status(status: str) -> bool`
- `staticmethod` `def delete_by_job_id() -> bool`

#### `DBNanhuRestartService` (`db/db_nanhu_restart_service.py`)

- `staticmethod` `def next_restart_index() -> int`
- `classmethod` `def create_record(nanhu_job_id: str, restart_status: str='REQUESTED', related_task_id: Optional…`
- `staticmethod` `def list_by_job(limit: int=100, offset: int=0) -> List[DBNanhuRestart]`
- `staticmethod` `def latest_for_job() -> Optional[DBNanhuRestart]`
- `staticmethod` `def update_status(restart_index: int, restart_status: str) -> bool`
- `staticmethod` `def count_by_job() -> int`

#### `DBRolloutService` (`db/db_rollout_service.py`)

- `staticmethod` `def upsert_with_fingerprint(rollout_id: int, pt_path: str, pt_size_bytes: int, pt_mtime: float,…`
- `staticmethod` `def list_by_task(limit: int=200, offset: int=0) -> List[DBRollout]`
- `staticmethod` `def get_by_pt_path() -> Optional[DBRollout]`
- `staticmethod` `def get_by_id() -> Optional[DBRollout]`
- `staticmethod` `def delete_by_task() -> int`

#### `DBSegmentService` (`db/db_segment_service.py`)

- `staticmethod` `def upsert(**fields: Any) -> DBSegment`
- `staticmethod` `def list_by_attempt(limit: int=500, offset: int=0) -> List[DBSegment]`
- `staticmethod` `def get_by_id() -> Optional[DBSegment]`
- `staticmethod` `def list_by_task(limit: int=2000, offset: int=0) -> List[DBSegment]`
- `staticmethod` `def count_by_rollout_for_task() -> Dict[int, int]`
- `staticmethod` `def delete_by_attempt() -> int`
- `staticmethod` `def delete_by_task() -> int`

#### `DBSegmentTranscriptService` (`db/db_segment_transcript_service.py`)

- `staticmethod` `def get_by_segment_id() -> Optional[DBSegmentTranscript]`
- `staticmethod` `def upsert_transcript(prompt_text: str='', response_text: str='', turns: Optional[List[Any]]=No…`
- `staticmethod` `def mark_fetched(tokenizer_path_used: str='') -> bool`
- `staticmethod` `def search_by_keyword(task_id: Optional[int]=None, limit: int=100, offset: int=0) -> List[DBSeg…`
- `staticmethod` `def delete_by_segment_id() -> bool`

#### `DBSegmentTurnService` (`db/db_segment_turn_service.py`)

- `staticmethod` `def upsert_segment_turns(turns: List[Dict[str, Any]], force: bool=False) -> int`
- `staticmethod` `def list_turns() -> List[Dict[str, Any]]`
- `staticmethod` `def delete_by_segment() -> int`
- `staticmethod` `def list_orm_turns() -> List[DBSegmentTurn]`
- `staticmethod` `def get_tool_call_names() -> List[str]`

#### `DBTaskService` (`db/db_task_service.py`)

- `staticmethod` `def create_task(**fields: Any) -> DBTask`
- `staticmethod` `def get_by_id() -> Optional[DBTask]`
- `staticmethod` `def get_by_name() -> Optional[DBTask]`
- `staticmethod` `def list_tasks(keyword: Optional[str]=None, limit: int=100, offset: int=0) -> List[DBTask]`
- `staticmethod` `def update_status(status: str) -> bool`
- `staticmethod` `def update_fields(**fields: Any) -> bool`
- `staticmethod` `def delete_task() -> bool`

#### `DBUserClusterConfigService` (`db/db_user_cluster_config_service.py`)

- `staticmethod` `def list_by_user() -> List[DBUserClusterConfig]`
- `staticmethod` `def get_by_user_and_key(cluster_key: str) -> Optional[DBUserClusterConfig]`
- `staticmethod` `def get_by_id(user_id: int) -> Optional[DBUserClusterConfig]`
- `staticmethod` `def list_need_token_refresh() -> List[DBUserClusterConfig]`
- `staticmethod` `def list_all() -> List[DBUserClusterConfig]`
- `staticmethod` `def resolve_for_cluster(user_id: Optional[int]=None) -> Optional[DBUserClusterConfig]`
- `staticmethod` `def upsert(cluster_key: str, display_label: str='', api_base_url: str='', user_id_field: str=''…`
- `staticmethod` `def delete_by_id(user_id: int) -> bool`
- `staticmethod` `def delete_by_user_and_key(cluster_key: str) -> bool`

#### `DBUserService` (`db/db_user_service.py`)

- `staticmethod` `def ensure_admin_seed(password: str=DEFAULT_ADMIN_PASSWORD, log_warning: bool=True) -> Dict[str…`
- `staticmethod` `def count_users() -> int`
- `staticmethod` `def is_username_taken() -> bool`
- `staticmethod` `def create_user(password: str, role: str='user', is_active: bool=True) -> Optional[Any]`
- `staticmethod` `def authenticate(password: str) -> Optional[Any]`
- `staticmethod` `def get_by_username() -> Optional[Any]`
- `staticmethod` `def get_by_id() -> Optional[Any]`
- `staticmethod` `def list_users(offset: int=0, role: Optional[str]=None, is_active: Optional[bool]=None) -> List…`
- `staticmethod` `def update_password(new_password: str) -> bool`
- `staticmethod` `def set_active(is_active: bool) -> bool`
- `staticmethod` `def set_last_login() -> bool`

## service/ — 业务流程

### `AnalysisResult` (`service/agent_transcript_analyzer.py`)

- `method` `def to_dict() -> Dict[str, Any]`

### `agent_transcript_analyzer_module` (`service/agent_transcript_analyzer.py`)

模块级公开函数（不在 class 内）：

- `staticmethod` `def load_transcript(path: str) -> List[Dict[str, Any]]`
- `staticmethod` `def extract_text_from_message(msg: Any) -> str`
- `staticmethod` `def count_tool_calls(msg: Any) -> int`
- `staticmethod` `def classify_user_feedback(text: str) -> float`
- `staticmethod` `def analyze_transcript(path: str) -> Dict[str, Any]`
- `staticmethod` `def batch_analyze(paths: Iterable[str]) -> List[Dict[str, Any]]`

### `nanhu_client_module` (`service/nanhu_client.py`)

模块级公开函数（不在 class 内）：

- `staticmethod` `def list_train_jobs(cluster: str='', status: str='', user_id: str='', local_user_id: Optional[i…`
- `staticmethod` `def get_job_detail(cluster: str, job_id: str, local_user_id: Optional[int]=None) -> Dict[str, A…`
- `staticmethod` `def create_train_job(cluster: str, yaml_path: str, submit: bool=False, local_user_id: Optional[…`
- `staticmethod` `def restart_job(cluster: str, job_id: str, submit: bool=False, local_user_id: Optional[int]=Non…`
- `staticmethod` `def module_diagnostics() -> Dict[str, Any]`

### `RolloutParseService` (`service/rollout_parse_service.py`)

- `method` `def parse_rollout(pt_path: str, force: bool=False) -> Dict[str, Any]`

### `rollout_parse_service_module` (`service/rollout_parse_service.py`)

模块级公开函数（不在 class 内）：

- `staticmethod` `def parse_rollout(pt_path: str, force: bool=False) -> Dict[str, Any]`
- `staticmethod` `def parse_attempt_token_stats(attempt_pk: int) -> Dict[str, Any]`

### `RolloutScanService` (`service/rollout_scan_service.py`)

- `method` `def scan_task(task_id: int, force: bool=False) -> Dict[str, Any]`
- `method` `def rescan_task(task_id: int) -> Dict[str, Any]`
- `method` `def get_progress(task_id: int) -> Dict[str, Any]`

### `segment_search_service_module` (`service/segment_search_service.py`)

模块级公开函数（不在 class 内）：

- `staticmethod` `def search_segments(task_id: Optional[int]=None, min_reward: float=0.0, keyword: str='', limit:…`

### `segment_transcript_service_module` (`service/segment_transcript_service.py`)

模块级公开函数（不在 class 内）：

- `staticmethod` `def ensure_transcript(segment_id: int, tokenizer_dir: str='', force: bool=False) -> Dict[str, A…`

## controller/ — REST 路由（新增接口前对照）

| 模块 | Method | Path | Function | 用途 |
| --- | --- | --- | --- | --- |
| `agent_transcript_controller` | GET | `/api/agent-transcripts/page` | `page_agent_transcripts` | 渲染 agent_transcript_report.html。 |
| `agent_transcript_controller` | GET | `/api/agent-transcripts/report` | `api_agent_transcript_report` | 返回单文件分析 JSON。 |
| `agent_transcript_controller` | GET | `/api/agent-transcripts/batch` | `api_agent_transcript_batch` | 批量分析；返回 list[dict]。 |
| `auth_controller` | GET | `/api/auth/login` | `page_login` | 渲染 ``templates/login.html``。 |
| `auth_controller` | POST | `/api/auth/login` | `api_login` | 登录端点（兼容 form + JSON 两种 body）。 |
| `auth_controller` | GET | `/api/auth/register` | `page_register` | 渲染 ``templates/register.html``。 |
| `auth_controller` | POST | `/api/auth/register` | `api_register` | 注册端点。 |
| `auth_controller` | POST | `/api/auth/logout` | `api_logout` | 登出：清 session，返回 /api/tasks。 |
| `auth_controller` | GET | `/api/auth/me` | `api_me` | 返回当前用户 dict；未登录返回 401 JSON。 |
| `nanhu_controller` | GET | `/api/nanhu/jobs/page` | `page_nanhu_jobs` | 渲染 nanhu_jobs.html（页面 + 表 + 过滤）。 |
| `nanhu_controller` | GET | `/api/nanhu/jobs` | `api_list_nanhu_jobs` | JSON 列表；refresh=1 会先调 nanhu_client.list_train_jobs 同步落库。 |
| `nanhu_controller` | GET | `/api/nanhu/jobs/{job_id}/resolve-for-task` | `api_resolve_job_for_task` | 根据南湖 job 详情解析创建 Rollout 任务所需字段。 |
| `nanhu_controller` | POST | `/api/nanhu/jobs/{job_id}/link-task` | `api_link_task` | 关联 task 到南湖 job。 |
| `nanhu_controller` | POST | `/api/nanhu/jobs/{job_id}/unlink-task` | `api_unlink_task` | 解绑 task 与南湖 job。 |
| `nanhu_controller` | POST | `/api/nanhu/jobs/restart` | `api_restart_job` | 重启南湖 job；默认 dry-run。 |
| `rollout_controller` | GET | `/api/tasks/{task_id}/rollouts/{rollout_id}` | `page_rollout_detail` | 渲染 rollout_detail.html。rollout_id 是 ``ri_rollout_table.primary_id``。 |
| `rollout_controller` | GET | `/api/rollouts/{rollout_id}` | `api_get_rollout_json` | JSON 端点：返回 rollout 概览 + attempts 列表。 |
| `rollout_controller` | POST | `/api/rollouts/{rollout_id}/parse` | `api_parse_rollout` | 触发 T3 重解析（默认 force=True）。 |
| `segment_controller` | GET | `/api/segments/search` | `api_segments_search` | Segment 检索：调 SegmentSearchService.search_segments。 |
| `segment_controller` | GET | `/api/segments/{segment_id}` | `page_segment_transcript` | 渲染 segment_transcript.html（v0.4.0 T5：注入 turns 列表 + 兄弟 segment 导航）。 |
| `segment_controller` | GET | `/api/segments/{segment_id}/transcript.json` | `api_segment_transcript_json` | 调 SegmentService.ensure_transcript 后返回 JSON。 |
| `segment_controller` | GET | `/api/segments/{segment_id}/turns.json` | `api_segment_turns_json` | 按需返回某个 segment 的 turn 列表（供 rollout 详情页懒加载）。 |
| `task_controller` | GET | `/` | `page_task_list` | 根路径 → task_list.html（直接注入任务列表到模板上下文）。 |
| `task_controller` | GET | `/api/tasks` | `api_list_tasks` | 任务列表端点。 |
| `task_controller` | GET | `/api/tasks/create` | `page_task_create` | 渲染 task_create.html。 |
| `task_controller` | POST | `/api/tasks` | `api_create_task` | 创建任务（form-encoded）。写库后由前端跳到 /api/tasks/{id}。 |
| `task_controller` | GET | `/api/tasks/{task_id}` | `page_task_detail` | 渲染 task_detail.html；概览 + Rollout 列表（含 n_segments）+ 南湖绑定。 |
| `task_controller` | POST | `/api/tasks/{task_id}/rescan` | `api_rescan_task` | 触发 RolloutScanService.rescan_task（异步后台跑，请求立即返回）。 |
| `task_controller` | POST | `/api/tasks/{task_id}/delete` | `api_delete_task` | 删除任务（级联删除其 rollout/attempt/segment）。 |
| `user_cluster_config_controller` | GET | `/api/me/cluster-configs` | `page_cluster_configs` | 渲染列表页（HTML）或返回 JSON 列表。 |
| `user_cluster_config_controller` | GET | `/api/me/cluster-configs/new` | `page_cluster_config_new` | 渲染新建表单（cluster_key 为空 → 新建模式）。 |
| `user_cluster_config_controller` | GET | `/api/me/cluster-configs/{cluster_key}/edit` | `page_cluster_config_edit` | 渲染编辑表单（带预填数据）。 |
| `user_cluster_config_controller` | POST | `/api/me/cluster-configs` | `api_cluster_config_upsert` | upsert：新建 / 更新都走这里。 |
| `user_cluster_config_controller` | POST | `/api/me/cluster-configs/{cluster_key}/delete` | `api_cluster_config_delete` | 删除（仅自己的）。 |

## 重构检查清单

开始改动前确认：

- [ ] 已阅读本文档相关章节（及 `.cursor/rules/rollout-insight-conventions.mdc`）
- [ ] 已 Grep 关键词（如 `parse_rollout`、`DBTaskService`）避免漏掉复用点
- [ ] 数据库变更只通过对应 `DBXxxService`，未在 controller/service 写裸 SQL
- [ ] 新 REST 接口使用 `data.response.Response`，业务错误用 `RolloutInsightException`
- [ ] 完成后已重跑 `python scripts/scan_func_md.py` 并跟随本次改动一并 commit

*文档版本：v0.5.2；自动生成，请勿手改。*
