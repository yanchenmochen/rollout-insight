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
  | `RolloutInsightException.TASK_NANHU_JOB_ALREADY_LINKED` | 51006 | `TASK_NANHU_JOB_ALREADY_LINKED` | 该南湖 job_id 已关联过任务 |
  | `RolloutInsightException.TASK_NAME_ALREADY_EXISTS` | 51007 | `TASK_NAME_ALREADY_EXISTS` | 任务名已存在 |
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

### `utils/attempt_artifacts.py`

- `def clear_artifact_cache_for_tests() -> None` — 仅供单测清空进程内索引缓存。
- `def locate_artifacts_dir(rollout_root: Optional[str]) -> Optional[Path]` — 从 ``task.rollout_root``（通常是 ``{run}/generated/rollout_dumps``）向上找 ``attempt_artifacts``。
- `def summarize_entry(summary: Dict[str, Any], stem: str) -> Dict[str, Any]` — 把单条 summary JSON 压成索引行（列表 / 聚类所需字段）。
- `def load_summary_index(artifacts_dir: Path, *, use_cache: bool=True) -> List[Dict[str, Any]]` — 返回 summaries 的索引行列表。
- `def load_summary(artifacts_dir: Path, stem: str) -> Optional[Dict[str, Any]]` — 读取单条完整 summary JSON；不存在返回 None。
- `def load_patch(artifacts_dir: Path, stem: str, max_chars: int=_PATCH_MAX) -> Dict[str, Any]` — 读取 unified diff；超长截断并标记 ``truncated``。
- `def find_trajectory_path(artifacts_dir: Path, stem: str) -> Optional[Path]` — 在 failed_ / passed_trajectories 中查找同名 history.jsonl。
- `def strip_ansi(text: Any) -> str` — 去掉终端颜色控制符（pytest / bash 输出常带）。
- `def load_trajectory_events(path: Path, *, offset: int=0, limit: int=200, errors_only: bool=Fals…` — 分页读取并规整轨迹事件；同时返回全量统计（总数 / 错误数 / 工具分布 / 末尾错误）。

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

### `utils/eval_report_md.py`

- `def iter_score_kinds(scores: Optional[Dict[str, Any]]) -> List[str]` — 列出该次评测里**实际有分数**的口径，按固定顺序返回。
- `def extract_scores(result_text: Any) -> Dict[str, Any]` — 从 ``Result`` 单元格里抽出各个分数。
- `def parse_evaluations_md(text: Optional[str]) -> Dict[str, Any]` — 解析 ``evaluations.md`` 全文。
- `def read_evaluations_md(path: Optional[Path | str]) -> Dict[str, Any]` — 读取并解析指定的 ``evaluations.md``；文件不存在返回 ``exists=False``。
- `def read_evaluations_for_rollout_root(rollout_root: Optional[str]) -> Dict[str, Any]` — 从 ``task.rollout_root`` 解析 run 目录并读取其 ``evaluations.md``。

### `utils/eval_tb_scores.py`

- `def is_primary_eval_tag(tag: str) -> bool` — 是否为评测主分数 tag（排除子指标与 truncated_ratio 等）。
- `def read_eval_scores_from_event_dir(event_dir: Optional[Path | str]) -> List[Dict[str, Any]]` — 读取事件目录下全部评测主分数（取每个 tag 最新一步）。
- `def read_eval_scores_for_rollout_root(rollout_root: Optional[str]) -> Dict[str, Any]` — 从 ``task.rollout_root`` 解析 run → tbs/tf_logs → 评测分数。

### `utils/failure_insight.py`

- `def empty_failure_insights() -> Dict[str, Any]` — 空的失败深挖结构（fail / timeout / abort clusters + by_group）。
- `def parse_pytest_failure(stdout_tail: str) -> Tuple[Optional[str], Optional[str]]` — 从 pytest stdout_tail 抽 ``(首个失败用例, 错误摘要)``；无法识别时为 None。
- `def extract_sample_failure_detail(sample: Dict[str, Any], *, sample_index: int=0) -> Optional[D…` — 抽取单条非通过 sample 的深挖细节；通过样本返回 None。
- `def cluster_failure_insights(dump: Dict[str, Any], *, group_index: Optional[int]=None) -> Dict[…` — 跨组 / 组内失败深挖聚类。

### `utils/failure_reason_taxonomy.py`

- `def agent_outcome_label(agent_outcome: Any) -> str` — agent 结束态的中文标签；未知值原样返回。
- `def is_agent_error_outcome(agent_outcome: Any) -> bool` — agent 结束态是否为异常（非 ``finished`` / 空）。
- `def agent_outcome_reason(agent_outcome: Any) -> str` — agent 结束态 → 统一原因码（与 .pt 侧口径一致）。
- `def normalize_reason(*, status: Any=None, reward: Any=None, diagnosis: Any=None, abort_reason: …` — 把散落的失败字段收敛为 ``(category, reason)``。
- `def classify_from_fields(fields: Dict[str, Any]) -> Tuple[str, str]` — 从 summary / annotation 字典收敛出 ``(category, reason)``。
- `def reason_label(category: str, reason: str) -> str` — ``(category, reason)`` → 中文短标签。
- `def split_reason_prefix(reason: str) -> Tuple[str, str]` — 把 ``evaluator:tests_failed`` 拆成 ``("evaluator", "tests_failed")``。

### `utils/icon_catalog.py`

- `def icon_path(name: str) -> str` — 返回图标在 static 下的访问路径。
- `def nav_icon(key: str) -> Dict[str, str]` — 按 key 取图标信息；未知返回空 dict。
- `def icon_svg(name: str, size: int=16) -> str` — 读出图标 SVG 文本，内联到 HTML（可随 ``currentColor`` 变色）。

### `utils/log_setup.py`

- `def setup_rollout_insight_logging(log_base: str='') -> None` — 配置 rollout-insight 的 loguru 输出。
- `def ensure_logging_configured() -> None` — 幂等地初始化 logging，避免被反复执行。

### `utils/nanhu_job_kind.py`

- `def classify_nanhu_job_kind(job: Optional[Mapping[str, Any]]) -> dict` — 按 algorithm_uuid / 路径 / 名称判定算法库。
- `def attach_nanhu_job_kind(job: MutableMapping[str, Any]) -> MutableMapping[str, Any]` — 就地写入 ``algo_kind`` / ``algo_label``，并返回同一 dict。

### `utils/nanhu_job_path_resolve.py`

- `def env_map_from_job(job: Mapping[str, Any]) -> Dict[str, str]` — 把 ``envVars`` / ``env_vars`` 列表转成 ``{KEY: value}``。
- `def output_dir_from_job(job: Mapping[str, Any], env: Optional[Mapping[str, str]]=None) -> str` — 解析作业输出根目录（不含 ``rollout_dumps`` 子路径）。
- `def resolve_paths_from_job_detail(job: Mapping[str, Any], *, prefer_existing_dirs: bool=True) -…` — 从作业详情 dict 解析创建任务字段。
- `def extract_job_payload(detail: Mapping[str, Any]) -> Optional[MutableMapping[str, Any]]` — 从 ``get_job_detail`` 返回体取出作业 dict；失败返回 ``None``。

### `utils/nanhu_job_schema.py`

- `def ensure_nanhu_job_columns(extra_columns: Iterable[Tuple[str, str]]=_NANHU_JOB_EXTRA_COLUMNS)…` — 若表已存在但缺列，则 ``ALTER TABLE ... ADD COLUMN``。返回新加列名。

### `utils/rollout_pt_parser.py`

- `def load_pt(path: str) -> Dict[str, Any]` — 加载 .pt dump：优先 zip/pickle，不要求环境能 ``import torch``。
- `def resolve_sample_reward(sample: Dict[str, Any]) -> Optional[float]` — 解析单条 sample 的可测 reward。
- `def sample_abort_reason(sample: Dict[str, Any]) -> Optional[str]` — 若 sample 为 abort，返回聚类用原因字符串；否则 None。
- `def index_attempt_outcomes(dump: Dict[str, Any]) -> Dict[Tuple[int, int], Dict[str, Any]]` — 按扫描入库键 ``(group_index, attempt_id)`` 索引每条 sample 的结局。
- `def cluster_abort_reasons(dump: Dict[str, Any], *, group_index: Optional[int]=None) -> Dict[str…` — 对 dump 内 abort attempt 按 ``abort_reason`` 聚类（兼容旧调用）。
- `def classify_sample_outcome(sample: Dict[str, Any]) -> Dict[str, str]` — 把单条 sample 归为 pass / abort / timeout / fail / other。
- `def cluster_attempt_outcomes(dump: Dict[str, Any], *, group_index: Optional[int]=None) -> Dict[…` — 跨组 / 组内失败情形聚类（abort / timeout / 正常失败 / 通过）。
- `def summarize_stack_type_breakdown(dump: Dict[str, Any]) -> Dict[str, Any]` — 为任务页「类型柱」汇总单次 rollout 的细分堆叠段。
- `def summarize_train_sample_stats(dump: Dict[str, Any]) -> Dict[str, Any]` — 统计进入训练的样本数（无 ``metadata.train_drop_reason`` 视为保留）。
- `def samples_index_for(_sample: Dict[str, Any]) -> int` — 占位：见 _segment_payload 中的 try/except；这里返回 0 让 except 兜底。
- `def parse_attempts_and_segments(dump: Dict[str, Any]) -> Dict[str, Any]` — 把 dump 拆成 attempts + segments + 整文 response。
- `def split_response_to_turns(response_text: str) -> List[Dict[str, Any]]` — 按 ``<|im_start|>role\n...<|im_end|>`` 切分整文 response。
- `def split_markdown_headed_turns(text: str) -> List[Dict[str, Any]]` — 把 ``### turn N · role`` 标题切成多轮。
- `def token_estimate(text: str, chars_per_token: float=3.5) -> int` — 粗估 token 数：默认按字符数 / 3.5。
- `def split_response_to_turns_with_tokenizer(response_text: str='', *, tokens: Optional[Sequence[…` — v0.5.0 T1：优先用 tokens + tokenizer 切分；否则 fallback 到 response 文本。

### `utils/slime_job_type.py`

- `def normalize_env_map(raw: Optional[Mapping[str, Any]]) -> Dict[str, str]` — 把任意 mapping 规整为 ``{KEY: str}``（跳过空键）。
- `def classify_slime_job_type(env: Optional[Mapping[str, Any]]=None, *, launcher: str='', trainin…` — 按 env / launcher 判定作业类型。
- `def attach_slime_job_type(target: MutableMapping[str, Any], env: Optional[Mapping[str, Any]]=No…` — 就地写入 ``job_type`` / ``job_type_label`` 等字段。
- `def load_env_from_run_dir(run_dir: Optional[Path | str]) -> Tuple[Dict[str, str], str, str]` — 从 run 目录读取 env 与 launcher。

### `utils/tb_deep_link.py`

- `def installed_tensorboard_version() -> str` — 已安装的 tensorboard 版本；导入失败时返回空串。
- `def tag_filter_supported(version: Optional[str]=None) -> bool` — 该版本是否能用 ``?tagFilter=`` 定位到单个 scalar tag。
- `def tensorboard_tag_filter(tag: str) -> str` — 精确匹配一个 tag 的正则（``tagFilter`` 按正则解释）。
- `def append_tag_filter(base_url: str, tag: str) -> str` — 在 TensorBoard 根 URL 上追加 ``tagFilter``，并丢掉 hash。

### `utils/tb_metric_catalog.py`

- `def classify_tb_tag(tag: str) -> str` — 返回 ``primary`` / ``truncated_ratio`` / ``attempt_count`` / ``unknown``。
- `def describe_tb_tag(tag: str) -> str` — 已知评测 tag 的中文说明；对不上的返回「未收录」。

### `utils/tb_run_paths.py`

- `def resolve_run_dir(rollout_root: Optional[str]) -> Optional[Path]` — 从 ``task.rollout_root``（通常指向 ``.../generated/rollout_dumps``）推 run 根目录。
- `def discover_tb_event_dir(run_dir: Path) -> Optional[Path]` — 在 run 根下按 ``TB_EVENT_DIR_NAMES`` 找可用的 TensorBoard 事件目录。
- `def discover_tbs_dir(run_dir: Path) -> Optional[Path]` — 兼容旧名：等同 ``discover_tb_event_dir``。

### `utils/tb_series_reader.py`

- `def list_scalar_tags(event_dir: Optional[Path | str]) -> List[str]` — 列出事件目录下的全部 scalar tag（已排序）。
- `def read_scalar_series(event_dir: Optional[Path | str], *, tags: Optional[Sequence[str]]=None, …` — 读取事件目录下若干 tag 的完整 scalar 序列。
- `def series_points(series: Dict[str, List[Dict[str, Any]]], tag: str) -> List[Dict[str, Any]]` — 从序列字典里取某个 tag 的点列表（不存在返回空 list）。

### `utils/template_factory.py`

- `def create_templates() -> Jinja2Templates` — 创建带全局模板变量的 Jinja2Templates 实例。

### `utils/train_curve_catalog.py`

- `def curve_metric_keys() -> List[str]` — 返回全部内置指标键。
- `def curve_metric(key: str) -> Optional[Dict[str, Any]]` — 按 key 取指标元信息；未知返回 None。
- `def available_curve_metrics() -> List[Dict[str, Any]]` — 列出内置指标，供页面渲染勾选框。
- `def normalize_curve_keys(keys: Optional[Sequence[str]]) -> List[str]` — 规整指标 key 列表：去空白、去重、保持顺序、过滤未知内置键。
- `def match_tag(key: str, tag: str) -> bool` — 判断某个 TB tag 是否命中指标 key 的匹配规则。
- `def resolve_metric_tags(key: str, available_tags: Sequence[str]) -> List[str]` — 在可用 tag 里找出命中该指标的所有 tag。
- `def is_custom_key(key: str) -> bool` — 是否用户自定义指标键。

### `utils/train_curve_epoch.py`

- `def resolve_steps_per_epoch(n_problems: Optional[int], batch_size: Optional[int]) -> Optional[i…` — 算出「一个 epoch 覆盖多少个 rollout step」。
- `def aggregate_by_epoch(points: Sequence[Dict[str, Any]], steps_per_epoch: Optional[int], *, max…` — 把逐 step 的点按 epoch 聚合（epoch 内所有 step 取平均）。
- `def epoch_view(points: Sequence[Dict[str, Any]], steps_per_epoch: Optional[int]) -> Dict[str, A…` — 聚合出 epoch 曲线，并附带口径说明。

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
| `DBTrainCurvePreference` | `db/db_train_curve_preference.py` | `ri_train_curve_preference_table` |
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
- `classmethod` `def list_outcome_rows_by_task(task_id: int, rollout_id: Optional[int]=None) -> List[Dict[str, A…`
- `classmethod` `def summarize_reward_metrics_by_rollout(task_id: int) -> Dict[int, Dict[str, int]]`
- `classmethod` `def summarize_reward_metrics_by_group(task_id: int, rollout_id: int) -> Dict[int, Dict[str, Any…`

#### `DBNanhuJobService` (`db/db_nanhu_job_service.py`)

- `staticmethod` `def upsert(**fields: Any) -> DBNanhuJob`
- `staticmethod` `def get_by_job_id() -> Optional[DBNanhuJob]`
- `staticmethod` `def get_by_id() -> Optional[DBNanhuJob]`
- `staticmethod` `def list_jobs(status: Optional[str]=None, keyword: Optional[str]=None, creator: Optional[str]=N…`
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
- `staticmethod` `def map_first_segment_id() -> Dict[int, int]`
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
- `staticmethod` `def search_by_keyword(task_id: Optional[int]=None, limit: int=100, offset: int=0, group_index: …`
- `staticmethod` `def delete_by_segment_id() -> bool`

#### `DBSegmentTurnService` (`db/db_segment_turn_service.py`)

- `staticmethod` `def upsert_segment_turns(turns: List[Dict[str, Any]], force: bool=False) -> int`
- `staticmethod` `def list_turns() -> List[Dict[str, Any]]`
- `staticmethod` `def delete_by_segment() -> int`
- `staticmethod` `def list_orm_turns() -> List[DBSegmentTurn]`
- `staticmethod` `def segment_turn_coverage() -> Dict[str, int]`
- `staticmethod` `def list_assistant_turn_flags() -> List[Dict[str, Any]]`
- `staticmethod` `def get_tool_call_names() -> List[str]`

#### `DBTaskService` (`db/db_task_service.py`)

- `staticmethod` `def create_task(**fields: Any) -> DBTask`
- `staticmethod` `def get_by_id() -> Optional[DBTask]`
- `staticmethod` `def get_by_name() -> Optional[DBTask]`
- `staticmethod` `def list_tasks(keyword: Optional[str]=None, limit: int=100, offset: int=0) -> List[DBTask]`
- `staticmethod` `def list_by_nanhu_job_ids() -> Dict[str, Dict[str, Any]]`
- `staticmethod` `def update_status(status: str) -> bool`
- `staticmethod` `def update_fields(**fields: Any) -> bool`
- `staticmethod` `def delete_task() -> bool`

#### `DBTrainCurvePreferenceService` (`db/db_train_curve_preference_service.py`)

- `staticmethod` `def get_global() -> Optional[DBTrainCurvePreference]`
- `staticmethod` `def get_for_task(task_id: int) -> Optional[DBTrainCurvePreference]`
- `staticmethod` `def upsert_global(*, keys: Optional[List[str]]=None, custom_keys: Optional[List[str]]=None) -> …`
- `staticmethod` `def upsert_for_task(task_id: int, *, keys: Optional[List[str]]=None, custom_keys: Optional[List…`
- `staticmethod` `def delete_global() -> bool`
- `staticmethod` `def delete_for_task(task_id: int) -> bool`
- `staticmethod` `def list_all_for_user() -> List[Dict[str, Any]]`

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

### `cross_task_compare_service_module` (`service/cross_task_compare_service.py`)

模块级公开函数（不在 class 内）：

- `staticmethod` `def list_compareable_tasks() -> List[Dict[str, Any]]`
- `staticmethod` `def summarize_reward_guidance(task_id: int) -> Dict[str, Any]`
- `staticmethod` `def build_cross_task_compare(task_ids: Sequence[int], *, instance_id: str='', baseline_task_id:…`
- `staticmethod` `def build_instance_compare(task_ids: Sequence[int], instance_id: str, baseline_task_id: Optiona…`

### `curve_compare_service_module` (`service/curve_compare_service.py`)

模块级公开函数（不在 class 内）：

- `staticmethod` `def list_comparable_tasks() -> List[Dict[str, Any]]`
- `staticmethod` `def list_curve_compare_brief() -> Dict[str, Any]`
- `staticmethod` `def build_curve_compare(task_ids: Optional[Sequence[int]]=None, keys: Optional[Sequence[str]]=N…`

### `failure_analysis_service_module` (`service/failure_analysis_service.py`)

模块级公开函数（不在 class 内）：

- `staticmethod` `def link_attempts_to_entries(rows: List[Dict[str, Any]], entries: List[Dict[str, Any]]) -> Dict…`
- `staticmethod` `def build_failure_clusters(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]`
- `staticmethod` `def build_group_failure_types(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]`
- `staticmethod` `def build_agent_outcome_clusters(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]`
- `staticmethod` `def list_tasks_with_failure_counts(limit: int=50) -> List[Dict[str, Any]]`
- `staticmethod` `def get_task_failure_overview(task_id: int, rollout_id: Optional[int]=None) -> Dict[str, Any]`
- `staticmethod` `def get_attempt_failure_detail(attempt_pk: int) -> Dict[str, Any]`
- `staticmethod` `def get_attempt_trajectory(attempt_pk: int, offset: int=0, limit: int=200, errors_only: bool=Fa…`

### `insight_page_service_module` (`service/insight_page_service.py`)

模块级公开函数（不在 class 内）：

- `staticmethod` `def build_insight_page(task_id: Optional[int]) -> Dict[str, Any]`
- `staticmethod` `def read_latest_scalars(event_dir: Optional[Path | str]) -> Tuple[List[Dict[str, Any]], Optiona…`

### `nanhu_client_module` (`service/nanhu_client.py`)

模块级公开函数（不在 class 内）：

- `staticmethod` `def list_train_jobs(cluster: str='', status: str='', user_id: str='', local_user_id: Optional[i…`
- `staticmethod` `def list_shared_algorithms(cluster: str='', local_user_id: Optional[int]=None, *, page_size: in…`
- `staticmethod` `def resolve_nanhu_user_id(cluster: str='', local_user_id: Optional[int]=None) -> str`
- `staticmethod` `def get_job_detail(cluster: str, job_id: str, local_user_id: Optional[int]=None) -> Dict[str, A…`
- `staticmethod` `def create_train_job(cluster: str, yaml_path: str, submit: bool=False, local_user_id: Optional[…`
- `staticmethod` `def restart_job(cluster: str, job_id: str, submit: bool=False, local_user_id: Optional[int]=Non…`
- `staticmethod` `def module_diagnostics() -> Dict[str, Any]`

### `problem_analysis_service_module` (`service/problem_analysis_service.py`)

模块级公开函数（不在 class 内）：

- `staticmethod` `def summarize_pass_visits(rows: Sequence[Dict[str, Any]], *, batch_size: Optional[int]=None, ba…`
- `staticmethod` `def group_instance_appearances(instance_id: str, rows: Sequence[Dict[str, Any]], rollout_pks: O…`
- `staticmethod` `def build_instance_appearances(task_id: int, instance_id: str) -> Dict[str, Any]`
- `staticmethod` `def build_pass_trend(task_id: int) -> Dict[str, Any]`
- `staticmethod` `def summarize_response_attempts(attempts: Sequence[Dict[str, Any]]) -> Dict[str, Any]`
- `staticmethod` `def merge_response_stats(parts: Iterable[Dict[str, Any]]) -> Dict[str, Any]`
- `staticmethod` `def public_response_stats(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]`
- `staticmethod` `def build_response_tool_stats(task_id: int) -> Dict[str, Any]`
- `staticmethod` `def advance_response_tool_stats(task_id: int, rollout_limit: int=1) -> Dict[str, Any]`
- `staticmethod` `def count_rollout_file(pt_path: str) -> Dict[str, Any]`

### `reward_diagnosis_service_module` (`service/reward_diagnosis_service.py`)

模块级公开函数（不在 class 内）：

- `staticmethod` `def classify_group_kind(categories: Sequence[str]) -> str`
- `staticmethod` `def summarize_gradient_groups(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]`
- `staticmethod` `def summarize_reward_split(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]`
- `staticmethod` `def aggregate_attempt_turn_flags(flags: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]`
- `staticmethod` `def classify_behavior_primary(row: Dict[str, Any], turn_summary: Optional[Dict[str, Any]]=None)…`
- `staticmethod` `def summarize_behavior_cards(rows: Sequence[Dict[str, Any]], turn_flags: Sequence[Dict[str, Any…`
- `staticmethod` `def build_reward_diagnosis(task_id: int) -> Dict[str, Any]`
- `staticmethod` `def summarize_pool_health(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]`
- `staticmethod` `def build_reward_levers(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]`

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

- `staticmethod` `def search_segments(task_id: Optional[int]=None, min_reward: Optional[float]=None, keyword: str…`

### `segment_transcript_service_module` (`service/segment_transcript_service.py`)

模块级公开函数（不在 class 内）：

- `staticmethod` `def ensure_transcript(segment_id: int, tokenizer_dir: str='', force: bool=False) -> Dict[str, A…`

### `task_annotation_service_module` (`service/task_annotation_service.py`)

模块级公开函数（不在 class 内）：

- `staticmethod` `def empty_failure_clusters() -> Dict[str, Any]`
- `staticmethod` `def empty_type_breakdown() -> Dict[str, Any]`
- `staticmethod` `def empty_train_sample_stats() -> Dict[str, Any]`
- `staticmethod` `def empty_failure_insights() -> Dict[str, Any]`
- `staticmethod` `def empty_annotations() -> Dict[str, Any]`
- `staticmethod` `def sidecar_path_for(pt_path: str) -> str`
- `staticmethod` `def load_pt_annotations(pt_path: Optional[str]) -> Dict[str, Any]`
- `staticmethod` `def warm_pt_annotation_sidecar(pt_path: Optional[str]) -> bool`
- `staticmethod` `def read_pt_annotation_sidecar(pt_path: Optional[str]) -> Optional[Dict[str, Any]]`
- `staticmethod` `def assign_stack_type_colors(key_totals: Counter) -> Dict[str, Dict[str, str]]`
- `staticmethod` `def list_task_annotation_meta(task_id: int) -> Dict[str, Any]`
- `staticmethod` `def get_rollout_annotation_payload(task_id: int, rollout_pk: int) -> Dict[str, Any]`
- `staticmethod` `def clear_annotation_cache_for_tests() -> None`

### `task_job_context_service_module` (`service/task_job_context_service.py`)

模块级公开函数（不在 class 内）：

- `staticmethod` `def build_task_job_context(*, rollout_root: Optional[str]=None, nanhu_job_id: Optional[str]=Non…`

### `tensorboard_service_module` (`service/tensorboard_service.py`)

模块级公开函数（不在 class 内）：

- `staticmethod` `def get_tensorboard_port() -> int`
- `staticmethod` `def build_runs_for_tasks(tasks: Sequence[Any]) -> Tuple[List[TensorBoardRun], List[Dict[str, An…`
- `staticmethod` `def build_logdir_spec(runs: Sequence[TensorBoardRun]) -> str`
- `staticmethod` `def public_tensorboard_url(request_host: Optional[str], port: int) -> str`
- `staticmethod` `def stop_tensorboard() -> Dict[str, Any]`
- `staticmethod` `def tensorboard_status() -> Dict[str, Any]`
- `staticmethod` `def start_tensorboard(runs: Sequence[TensorBoardRun], *, request_host: Optional[str]=None, port…`
- `staticmethod` `def collect_tasks_for_scope(scope: str, task_id: Optional[int]=None) -> Tuple[List[Any], List[D…`

### `train_curve_service_module` (`service/train_curve_service.py`)

模块级公开函数（不在 class 内）：

- `staticmethod` `def resolve_event_dir(rollout_root: Optional[str]) -> Optional[str]`
- `staticmethod` `def resolve_curve_keys(user_id: Optional[int], task_id: int) -> Dict[str, Any]`
- `staticmethod` `def build_train_curve(task_id: int, user_id: Optional[int]=None) -> Dict[str, Any]`
- `staticmethod` `def save_curve_preference(task_id: int, user_id: Optional[int], *, keys: Optional[List[str]]=No…`

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
| `cross_task_compare_controller` | GET | `/api/cross-task-compare` | `page_cross_task_compare` | 渲染跨任务对比页。 |
| `cross_task_compare_controller` | GET | `/api/cross-task-compare.json` | `api_cross_task_compare` | 返回跨任务对比数据：共同题目的 epoch 趋势 + 变好/变坏清单 + 奖励指引。 |
| `cross_task_compare_controller` | GET | `/api/cross-task-compare/instance.json` | `api_cross_task_instance` | 返回某一道题在多个任务上的逐次出现与通过率。 |
| `failure_controller` | GET | `/api/tasks/{task_id}/failures` | `page_task_failures` | 任务详情下的失败分析。复用 failure_overview.html 与 overview.json，不重写聚类。 |
| `failure_controller` | GET | `/api/failures` | `page_failure_overview` | 旧地址。有 task_id 时与任务失败分析相同；没有 task_id 时只给回任务列表的说明。 |
| `failure_controller` | GET | `/api/failures/attempts/{attempt_pk}` | `page_failure_attempt` | 渲染 failure_attempt.html：单 attempt 怎么失败的。 |
| `failure_controller` | GET | `/api/failures/tasks/{task_id}/overview.json` | `api_failure_overview` | 任务失败总览：每 rollout 结局分布 + 非通过 attempt 列表 + 原因聚类。 |
| `failure_controller` | GET | `/api/failures/attempts/{attempt_pk}/detail.json` | `api_failure_attempt_detail` | 单 attempt 失败详情（结论 / 评测证据 / 耗时 / patch / 回跳链接）。 |
| `failure_controller` | GET | `/api/failures/attempts/{attempt_pk}/trajectory.json` | `api_failure_attempt_trajectory` | 分页返回 OpenHands 轨迹事件（errors_only=true 只看报错步）。 |
| `insight_controller` | GET | `/api/insights` | `page_insights` | 渲染洞察页。没有事件目录时页面给出说明，不返回 500。 |
| `nanhu_controller` | GET | `/api/nanhu/jobs/page` | `page_nanhu_jobs` | 渲染 nanhu_jobs.html（页面 + 表 + 过滤）。 |
| `nanhu_controller` | GET | `/api/nanhu/jobs` | `api_list_nanhu_jobs` | JSON 列表；refresh=1 会先调 nanhu_client.list_train_jobs 同步落库。 |
| `nanhu_controller` | POST | `/api/nanhu/jobs/import` | `api_import_nanhu_jobs` | 按 job_id 导入作业（绕过 jobList 只返回本人作业的限制）。 |
| `nanhu_controller` | GET | `/api/nanhu/jobs/{job_id}/resolve-for-task` | `api_resolve_job_for_task` | 根据南湖 job 详情解析创建 Rollout 任务所需字段。 |
| `nanhu_controller` | POST | `/api/nanhu/jobs/{job_id}/link-task` | `api_link_task` | 关联 task 到南湖 job。 |
| `nanhu_controller` | POST | `/api/nanhu/jobs/{job_id}/unlink-task` | `api_unlink_task` | 解绑 task 与南湖 job。 |
| `nanhu_controller` | POST | `/api/nanhu/jobs/restart` | `api_restart_job` | 重启南湖 job；默认 dry-run。 |
| `problem_analysis_controller` | GET | `/api/tasks/{task_id}/problem-analysis` | `page_problem_analysis` | 任务详情下的问题分析：通过率曲线与 sglang 响应统计。 |
| `problem_analysis_controller` | GET | `/api/tasks/{task_id}/problem-analysis/instance` | `page_instance_appearances` | 按题目名列出每次出现的 rollout 与轨迹入口。 |
| `problem_analysis_controller` | GET | `/api/tasks/{task_id}/problem-analysis/pass.json` | `api_problem_pass_trend` | 按题目列出历次出现的 rollout 与通过率，不读响应统计。 |
| `problem_analysis_controller` | GET | `/api/tasks/{task_id}/problem-analysis.json` | `api_problem_analysis` | 通过率（按题目出现次序）以及当前已能给出的响应统计。 |
| `problem_analysis_controller` | POST | `/api/tasks/{task_id}/response-stats` | `api_advance_response_stats` | 补统计下一批 rollout 的 assistant 回复。不写入 turn 正文。 |
| `reward_diagnosis_controller` | GET | `/api/tasks/{task_id}/reward-diagnosis` | `page_reward_diagnosis` | 任务详情旁的奖励诊断：梯度组、奖励分解、行为失败卡片。 |
| `reward_diagnosis_controller` | GET | `/api/tasks/{task_id}/reward-diagnosis.json` | `api_reward_diagnosis` | 返回有效梯度组曲线、奖励分解曲线与行为失败卡片。 |
| `rollout_controller` | GET | `/api/tasks/{task_id}/rollouts/{rollout_id}` | `page_rollout_detail` | 渲染 rollout_detail.html。rollout_id 是 ``ri_rollout_table.primary_id``。 |
| `rollout_controller` | GET | `/api/rollouts/{rollout_id}` | `api_get_rollout_json` | JSON 端点：返回 rollout 概览 + attempts 列表。 |
| `rollout_controller` | POST | `/api/rollouts/{rollout_id}/parse` | `api_parse_rollout` | 触发 T3 重解析（默认 force=True）。 |
| `segment_controller` | GET | `/api/segments/search` | `api_segments_search` | Segment 检索 JSON：调 SegmentSearchService.search_segments。 |
| `segment_controller` | GET | `/api/segments/search/page` | `page_segments_search` | Segment 检索页。JSON 接口仍是 GET /api/segments/search，留给 CLI。 |
| `segment_controller` | GET | `/api/segments/{segment_id}` | `page_segment_transcript` | 渲染 segment_transcript.html（v0.4.0 T5：注入 turns 列表 + 兄弟 segment 导航）。 |
| `segment_controller` | GET | `/api/segments/{segment_id}/transcript.json` | `api_segment_transcript_json` | 调 SegmentService.ensure_transcript 后返回 JSON。 |
| `segment_controller` | GET | `/api/segments/{segment_id}/turns.json` | `api_segment_turns_json` | 按需返回某个 segment 的 turn 列表（供 rollout 详情页懒加载）。 |
| `site_controller` | GET | `/favicon.ico` | `get_favicon` | 返回多尺寸 favicon.ico（浏览器标签页 / 书签栏图标）。 |
| `site_controller` | GET | `/apple-touch-icon.png` | `get_apple_touch_icon` | 返回 180×180 apple-touch-icon.png（iOS 添加到主屏幕）。 |
| `site_controller` | GET | `/site.webmanifest` | `get_site_manifest` | 返回 PWA manifest（声明名称、主题色与各尺寸图标）。 |
| `task_controller` | GET | `/` | `page_task_list` | 根路径 → task_list.html（直接注入任务列表到模板上下文）。 |
| `task_controller` | GET | `/api/tasks` | `api_list_tasks` | 任务列表端点。 |
| `task_controller` | GET | `/api/tasks/create` | `page_task_create` | 渲染 task_create.html。 |
| `task_controller` | POST | `/api/tasks` | `api_create_task` | 创建任务（form-encoded）。写库后由前端跳到 /api/tasks/{id}。 |
| `task_controller` | GET | `/api/tasks/{task_id}` | `page_task_detail` | 渲染 task_detail.html；概览 + Rollout 列表（含 n_segments）+ 南湖绑定。 |
| `task_controller` | GET | `/api/tasks/{task_id}/annotations/meta.json` | `api_task_annotations_meta` | 返回待渐进加载的 rollout 列表（rollout_pk / rollout_id / pt_path）。 |
| `task_controller` | GET | `/api/tasks/{task_id}/rollouts/{rollout_pk}/annotations.json` | `api_rollout_annotations` | 单 rollout 类型柱 / 进入训练 / 失败摘要（读 sidecar 或 .pt）。 |
| `task_controller` | POST | `/api/tasks/{task_id}/rescan` | `api_rescan_task` | 触发 RolloutScanService.rescan_task（异步后台跑，请求立即返回）。 |
| `task_controller` | POST | `/api/tasks/{task_id}/delete` | `api_delete_task` | 删除任务（级联删除其 rollout/attempt/segment）。 |
| `tensorboard_controller` | GET | `/api/tensorboard` | `page_tensorboard_compare` | 渲染多任务曲线对比页（勾选任务 + 指标，叠图）。 |
| `tensorboard_controller` | GET | `/api/tensorboard/curve-compare.json` | `api_curve_compare` | 返回多任务曲线叠加数据：勾选的任务 × 勾选的指标。 |
| `tensorboard_controller` | POST | `/api/tensorboard/start` | `api_tensorboard_start` | 启动或复用 TensorBoard，返回可新标签打开的 URL。 |
| `tensorboard_controller` | GET | `/api/tensorboard/status` | `api_tensorboard_status` | — |
| `tensorboard_controller` | POST | `/api/tensorboard/stop` | `api_tensorboard_stop` | — |
| `train_curve_controller` | GET | `/api/tasks/{task_id}/train-curve` | `page_train_curve` | 渲染训练曲线页：指标按钮组 + 折线，可保存指标配置。 |
| `train_curve_controller` | GET | `/api/tasks/{task_id}/train-curve.json` | `api_train_curve` | 返回训练曲线数据：指标按钮组 + 每个指标的折线点。 |
| `train_curve_controller` | POST | `/api/tasks/{task_id}/train-curve/preference` | `api_save_train_curve_preference` | 保存指标偏好。scope=global 写用户默认，scope=task 写该任务追加项。 |
| `user_cluster_config_controller` | GET | `/api/settings` | `redirect_legacy_settings` | 兼容旧顶栏「系统设置」死链 ``/api/settings`` → 每人自己的集群凭据页。 |
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
