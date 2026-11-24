# 奖励提升方案 v2 · 配套设计规格

> 配套文档：[PLAN_reward_uplift.md](PLAN_reward_uplift.md)（v2 主方案）。
> 本文件收录 v2 主方案引用的**详细模块设计规格**：函数签名、数据契约、判定阈值与测试清单。
>
> 生成日期：2026-11-23
> 事实依据：线上 11 任务 / 398 rollout / 51314 attempt 的真实解析（DB + summary JSON + TB 458 tag + `.pt` dump）
>
> **只做设计规格，不含实现代码。**

---

## 目录

- [第一部分 · 时间预算退化与题目池健康度](#第一部分--时间预算退化与题目池健康度)
  - A. `utils/agent_budget_profile.py`
  - B. `utils/failure_reason_taxonomy.py` 扩展
  - C. `utils/difficulty_profile.py`
- [第二部分 · 口径修正与训练侧接入](#第二部分--口径修正与训练侧接入)
  - D. `utils/slime_grpo_semantics.py`
  - E. `service/reward_diagnosis_service.py` 扩展
  - F. `utils/tb_metric_catalog.py` 扩展
  - G. `service/train_health_service.py`
  - H. `service/problem_analysis_service.py` 三口径
- [第三部分 · 页面规格与测试清单](#第三部分--页面规格与测试清单)
  - I. 奖励诊断页「时间预算健康度」卡片
  - J. 测试用例清单
  - K. `func.md` 同步清单

---


---

# 第一部分 · 时间预算退化与题目池健康度

> **【前置发现 · 2026-11-23 实测】扫描完整性缺口**（影响所有基于 DB 的统计）：
>
> | task | DB 已扫 rollout | `.pt` 文件数 | 未入库 rollout |
> | --- | --- | --- | --- |
> | 1 | 6 | 9 | **[6, 7, 8]** |
> | **4** | 63 | **65** | **[63, 64]** |
> | **7** | 47 | **50** | **[47, 48, 49]** |
> | 其余 7 个任务 | 全部 | 全部 | — |
>
> **【实测】** task4 的 rollout 63/64 在 `ri_attempt_table` 中是 **0 行**，`.pt` 里却有 128/130 个样本。
>
> **【实测】差异的三类来源（必须分开，初判曾把它们混在一起）**：
>
> | 来源 | task4 | task9 | task8 | 性质 |
> | --- | --- | --- | --- | --- |
> | `.pt` samples | 8373 | 6176 | 8213 | — |
> | DB attempt 行 | 8064 | 6144 | 8192 | — |
> | ① key collision（同 attempt 多 segment，DB 按 attempt 去重） | 53 | 32 | 21 | **正常行为** |
> | ② **键空间混用导致错位丢失** | **5** | **0** | **0** | **真 bug** |
> | ③ 未扫描的 rollout | 2 个（258 样本） | 0 | 0 | 扫描未完成 |
>
> **【实测】机制 ②**：扫描键 `(group_index, attempt_id)`，其中
> `attempt_id = sample.rollout_id`，缺失时回落到 `sample["index"]`。
> 同组内混用两者时落在**不同键空间** → 错位。
> 实例：task4 rollout 2 的 group 41 —— DB 存的是 attempt_id **72–79**（全记为 aborted/reward=0），
> 而**真实通过的 5 个样本（rollout_id 329/330/331/332/335）没入库**。
> 影响面：task4 **93 个组**混用、丢 **5 个通过样本**；task9 **60 个组**混用、丢 **0 个**。
>
> **【设计 · P1 代码修复】** attempt 键改为**组内统一序号**（或组内检测到混用时统一用 idx 空间），
> 避免真实通过样本被误记为 aborted（**会系统性低估通过率**）。修复后需回填历史数据。
>
> **影响**：task4 的 `removed_ratio` 末值 **1.000** 与 `trainable_attempt_count` 末值 **0**
> 含「扫描未完成」成分 → **不能直接读作「这一步一个样本都不剩」**。
> 但**结论方向不变**：已扫完的 0–62 段内，`removed_ratio` 也已升到 0.731、trainable 已降到个位数。
>
> **【设计】** `utils/agent_budget_profile.py` 与 `service/reward_diagnosis_service` 的所有
> 「末值 / 峰值」统计**必须先校验扫描完整性**，并在输出中携带：
>
> **【设计 · P1】扫描器须校验 `.pt` 指纹**：读 `.pt` 前先比对 `ri_rollout_table` 的
> `pt_size_bytes` / `pt_mtime`，不一致则**跳过并标记 stale**，而不是拿旧数据糊在新文件上。
> （现状：这两个字段被记录但**从未用于判定**）
>
> ```python
> "scan_coverage": {
>     "n_pt_files": int,            # .pt 文件总数
>     "n_scanned_rollouts": int,   # DB 已入库的 rollout 数
>     "missing_rollout_ids": [int, ...],
>     "is_complete": bool,         # n_scanned == n_pt_files
>     "last_scanned_rollout_id": int|None,
>     "stale_rollout_ids": [int, ...],   # .pt 已被重跑覆盖（size/mtime 与 DB 不符）
> }
> ```
>
> 页面必须显示「当前统计截至 rollout N（共 M 个）」，末值高估时给出提示。
> **新增** `service/task_scan_coverage_service.py` 专门计算该口径（复用
> `utils/rollout_scan_service` 的目录扫描逻辑，不新建扫描器）。

> 交付物 A/B/C 的设计规格 + 对应测试清单。
> 事实依据：线上训练产出的真实解析（详见主方案 §1.5）。

# 奖励提升方案 v2 · 设计规格（§1 时间预算退化 / §4 题目池健康度）

> 事实依据：`/tmp/ri_reward_evidence.md`（下称「简报」）；全局方案：`/root/.cursor/plans/奖励提升方案修订_a1a73b8b.plan.md`（下称「计划」）。

---

## 0. 前置：func.md 查阅结论与复用映射

### 0.1 已确认「无同类实现」的三个目标模块

| 目标模块 | func.md 检索结论 | 判定 |
| --- | --- | --- |
| `utils/agent_budget_profile.py` | `utils/` 全表无「预算 / 耗时画像 / 退化判定」条目；`train_curve_catalog.py` 只做 **TB tag** 目录，`train_curve_epoch.py` 只做 **epoch 聚合**，均不涉及 attempt 级 `phase_time` 画像 | **P4 新建** |
| `utils/failure_reason_taxonomy.py` 扩展 | 已存在（`normalize_reason` / `classify_from_fields` / `reason_label` / `agent_outcome_reason` / `CATEGORY_LABELS` / `AGENT_OUTCOME_LABELS` / `AGENT_OUTCOME_REASON_CODES`） | **P2 就地扩展** |
| `utils/difficulty_profile.py` | `utils/` 无难度三态实现；`service/problem_analysis_service.summarize_pass_visits` 有「按题出现次序的通过率」雏形；`service/reward_diagnosis_service.summarize_pool_health` 有**失效的**「末次/最好一次通过率」口径 | **P4 新建纯函数 + P2 扩展上游**（详见 C.4） |

### 0.2 必须复用的既有能力（不得重写）

| 复用点 | 位置 | 在本方案中的用途 |
| --- | --- | --- |
| `classify_sample_outcome` | `utils/rollout_pt_parser.py` | attempt → `pass/abort/timeout/fail/other` 基础分类，**不得**在 A/C 中另写一套 |
| `resolve_sample_reward` | 同上 | abort 的 reward 兜底 0.0 不可当失败 |
| `cluster_attempt_outcomes` | 同上 | 组内结局聚类（A 的 `n_time_budget` 交叉校验） |
| `summarize_pass_visits` | `service/problem_analysis_service.py` | **C 的原始素材**：每题每次出现的 `n_pass/n_attempts/rate` + 已有的 `trend/delta` |
| `classify_group_kind` | `service/reward_diagnosis_service.py` | `mixed/all_pass/all_fail/infra_only` 组语义，A/C 的杠杆口径 |
| `aggregate_attempt_turn_flags` | 同上 | assistant turn 标记聚合 |
| `normalize_reason` / `classify_from_fields` | `utils/failure_reason_taxonomy.py` | **B 的唯一分类入口**，A 的 `agent_outcome` → reason 映射 |
| `summarize_entry` | `utils/attempt_artifacts.py` | artifacts summary → 索引行（A 的行构造器扩展点） |
| `load_pt` / `parse_attempts_and_segments` | `utils/rollout_pt_parser.py` | 无 artifacts 时的 `.pt` metadata 回退取数 |
| `token_estimate` | 同上 | 缺 tokenizer 时的粗估兜底（仅作 C 的说明，不作 A 的主口径） |

### 0.3 简报数字复核（4 处需在设计中修正，否则会把结论带偏）

> 这一节是**设计前提**，不是对简报的否定；目的是让阈值与后续实现口径不被笔误污染。

1. **task8 abort 分类占比**。简报 §2.3 表把 `openhands_infra_exit_2`（219）标为「基建/SDK」，但结论句写「沙箱类合计仅 14 例（1.1%)」。按表重算：abort 合计 `1425+219+7+5+2 = 1658`；**agent 侧 `agent_time_budget_exceeded` = 1425/1658 = 85.9%**；含 `openhands_infra_exit_2` 的基建/SDK = `219+7+5 = 231/1658 = 13.9%`；纯沙箱异常 = `12/1658 = 0.72%`。
   → **结论不变（agent 侧是主因），但「基建 < 1.5%」的说法要改成「纯沙箱异常 0.7%，基建/SDK 合计 13.9%」**。本设计因此把「基建变慢」的判定阈值设为**结构化判据**（见 A.6），而不是「abort 里有没有基建字样」。

2. **「梯度密度」是两种指标混用**。简报 §3 表的「梯度密度」列（task9 50%、task8/11 25%、task4 2.3%、task1 16.7%、task2 4.7%、task3 2.5%、task6/7/10 12.5%）**精确等于 `groups_per_step / n_problems`**（16/32、16/64、16/688、16/96、16/338、16/640、16/128…），逐行可复算；而 §3.1 的 mixed 组占比（task9 77.6%、task4 48.1%）是**实测值**，二者不是同一量。
   → 设计必须把两者**拆成两个字段**（`sampled_density` 理论覆盖 vs `mixed_ratio` 实测），并指出二者**非单调**（task4 密度最低 2.3% 但 mixed 48.1% 高于 task8 的 25% 密度/50.9% mixed）——**「密度 → 奖励」不是单因果**，这是简报里最容易被过度解读的一处。

3. **`stable_pass` 阈值 0.9 与单次出现的量化粒度**。计划 §4.1 写 `stable_pass: 累计通过率 >= 0.9`，但每组 8 attempts 时单次出现的通过率只能取 `k/8`；`7/8 = 0.875 < 0.9`，于是「单次 7/8」被判成 `gradient`。这在语义上没错，但**必须在输出里带上 `n_appearances` 与量化粒度**，否则用户会把「只出现 1 次的题」当成真结论。见 C.5 置信度设计。

4. **计划 §4.3「从 `stable_fail` 里挑最接近 0.2–0.5 的」自相矛盾**（`stable_fail` 定义为 ≤ 0.1，不可能出现 0.2–0.5）。见 C.6 的修正：改用 **best-visit rate** 与 **mixed 概率**排序。

---

# 交付物 A：`utils/agent_budget_profile.py` 设计规格

## A.1 目标与依据

**目标**：把「agent 时间预算耗尽」这条本批任务第一主因，从「读日志猜」变成「一条可复算、可判退化模式、可区分模型慢/基建慢」的曲线与判定。

**依据（简报 §2.2 逐 rollout 全量轨迹，task8 64 rollout / 8213 attempt）**：

| 指标 | rollout 0 | rollout 30 | rollout 55 | rollout 59 | rollout 63 | 变化 |
| --- | --- | --- | --- | --- | --- | --- |
| `turns` p50 | 57 | 56 | 41 | 36 | 42 | **57 → 36（−37%）** |
| `peak_context_tokens` p50 | ~88K | ~83K | ~64K | ~64K | ~67K | **88K → 66K（−25%）** |
| 单轮生成耗时 | 18.7s | 31.3s | 63.1s | 64.3s | 47.1s | **18.7 → 64.3s（+244%）** |
| 单轮生成 token | 375 | 534 | 935 | 945 | 886 | **375 → 945~966（+152%~158%）** |
| `attempt_total − agent_run` | ~278s | ~281s | ~335s | ~693s | **1103s** | **278 → 1103s（+297%）** |
| `time_budget_exceeded` 占比 | 0% | 12.5% | 62.5% | 48.1% | 22.5% | 峰值 62.5%（ro55） |

**对照（简报 §2.5，task9 唯一正例）**：单轮耗时 22.2s → 31.2s（**+40.5%**），轮数 54 → 57（**+5.6%**），**从不撞墙**。

**预算锚点（简报 §2.1）**：1363 个 `time_budget_exceeded` 样本的 `phase_time/agent_run` = p10 **3610s** / p50 **3614s** / p90 **3620s** / max **3903s**，**100% ≥ 3600s** → 精确的 60 分钟墙钟截断，**且截断计在 `agent_run` 上**（不是 `attempt_total`）。这一条决定了 A.6 的「模型慢 vs 基建慢」判据与 A.5 的预算挤占口径。

**为什么必须是纯函数模块**：`phase_time/*` 在 `.pt` `sample.metadata` 与 `attempt_artifacts/summaries/*.json` 里都有，但**`ri_attempt_table` 里没有**任何耗时/上下文字段**（`db/db_attempt.py` 仅 `agent_turns/tool_calls/response_length/...`）。所以画像只能从 `.pt`/artifacts 现算，且必须可被单测直接喂数据 → 纯函数是唯一合理形态。

## A.2 输入行契约（Attempt Budget Row）

### A.2.1 行字段（**归一化后的规范行**，模块内部唯一认这套装 key）

| 字段 | 类型 | 必填 | 来源（`.pt` sample.metadata / artifacts summary / DB） | 说明 |
| --- | --- | --- | --- | --- |
| `rollout_id` | `int` | 是 | `ri_rollout_table.rollout_id` / summary `rollout_id` | 画像的时间轴主键 |
| `group_index` | `int \| None` | 否 | `sample.group_index` / summary `group_index` | 与 `classify_group_kind` 对齐用 |
| `attempt_id` | `int \| None` | 否 | `sample.rollout_id` / summary `attempt_id` | |
| `instance_id` | `str` | 否 | `metadata.instance_id` / `ri_attempt_table.instance_id` | |
| `status` | `str` | 否 | `sample.status` / summary `status` | `completed/truncated/aborted/...` |
| `agent_outcome` | `str` | 否 | `metadata.agent_outcome` / summary `agent_outcome` | `finished/time_budget_exceeded/context_exhausted/sdk_error/...` |
| `reward` | `float \| None` | 否 | `ri_attempt_table.reward` | 用 `resolve_sample_reward` 口径 |
| `category` | `str` | 否 | 由 `classify_sample_outcome` 得出 | `pass/abort/timeout/fail/other` |
| `attempt_total_s` | `float \| None` | 否 | `metadata["phase_time/attempt_total"]` 或 `metadata["phase_time"]["attempt_total"]` 或 summary `phase_times["phase_time/attempt_total"]` | 整次 attempt 墙钟 |
| `agent_run_s` | `float \| None` | 否 | `metadata["phase_time/agent_run"]`（或嵌套 `phase_time["agent_run"]`） | **预算截断计在这上面** |
| `model_generation_s` | `float \| None` | 否 | `metadata["phase_time/model_generation"]` | 模型侧生成耗时 |
| `tool_execution_s` | `float \| None` | 否 | `metadata["phase_time/tool_execution"]` | 工具执行 |
| `cli_install_s` | `float \| None` | 否 | `metadata["phase_time/cli_install"]` | 预算外·准备类 |
| `sandbox_boot_s` | `float \| None` | 否 | `metadata["phase_time/sandbox_boot"]` | 预算外·准备类 |
| `evaluator_total_s` | `float \| None` | 否 | `metadata["phase_time/evaluator_total"]` | 预算外·评测类 |
| `sandbox_boot_queue_s` | `float \| None` | 否 | `metadata["phase_time/sandbox_boot_queue"]` | 预算外·**纯排队** |
| `group_sibling_wait_s` | `float \| None` | 否 | `metadata["phase_time/group_sibling_wait"]` | 预算外·**纯排队** |
| `rollout_batch_wait_s` | `float \| None` | 否 | `metadata["phase_time/rollout_batch_wait"]` | 预算外·**纯排队** |
| `turns` | `int \| None` | 否 | `metadata.agent_turns`（优先）→ `metadata.attempt_completed_turns`（兜底）→ `ri_attempt_table.agent_turns` | 轮数 |
| `generated_response_tokens` | `int \| None` | 否 | `metadata.attempt_generated_response_tokens` | 整次 attempt 生成 token 总数 |
| `peak_context_tokens` | `int \| None` | 否 | `metadata.attempt_peak_context_tokens` | 上下文峰值 |
| `context_limit_tokens` | `int \| None` | 否 | `metadata.context_limit_tokens` | 上下文上限（缺失时用 `default_context_limit`） |
| `cached_prompt_tokens` | `int \| None` | 否 | `metadata.prefix_cache_info.cached_tokens` | 前缀缓存命中（未使用字段，本次启用） |
| `total_prompt_tokens` | `int \| None` | 否 | `metadata.prefix_cache_info.total_prompt_tokens` | 前缀缓存分母 |

### A.2.2 「phase_time 三种写法」的归一化（**必须处理，否则一半数据取不到**）

仓库里 `phase_time` 有三种并存写法：

1. `.pt` `sample.metadata` **扁平 key**：`metadata["phase_time/agent_run"]`（见 `utils/failure_insight.py:230`）；
2. `.pt` `sample.metadata` **嵌套 dict**：`metadata["phase_time"]["agent_run"]`（见 `utils/rollout_pt_parser.py:694`、`controller/segment_controller.py:830`）；
3. artifacts summary 的 `phase_times` dict，**key 带前缀**：`summary["phase_times"]["phase_time/attempt_total"]`（见 `utils/attempt_artifacts.py:115`）。

→ 模块内提供 `normalize_attempt_row(row) -> dict` 统一成 A.2.1 的规范行，**同时接受三种写法**；缺失一律 `None`，**不做推断填充**（唯一例外：`context_limit_tokens` 缺失时用参数 `default_context_limit`，默认 `131072`，标注为**设计假设**，依据 `tests/test_failure_insight.py:64` 的实测 `context_limit_tokens=131071`）。

### A.2.3 行构造器（**不属于本模块**，但接口必须写死）

本模块**不读文件、不查 DB**。行由调用方构造，落点建议：

- **扩展 `utils/attempt_artifacts.summarize_entry`**（P2，就地扩展）：在既有索引行上补 `phase_times`（原始 dict）、`attempt_generated_response_tokens`、`context_limit_tokens`；这是 artifacts 侧唯一取数点。
- 新增一个**行拼装**步骤（建议放 `service/reward_diagnosis_service`，或 `utils/attempt_artifacts` 内新增 `build_budget_rows(entries, pt_fallback)`）：优先 artifacts（有 `phase_times`），缺失时用 `utils/rollout_pt_parser.load_pt` 读 `sample.metadata` 回填。
- **不新建 service**（计划 §6「不新建数据表 / 就地扩展」）。

## A.3 输出 Schema

```python
{
  "budget_seconds": 3600,                    # int，预算墙钟（默认 3600）
  "budget_clock": "agent_run",               # str: "agent_run" | "attempt_total"
                                            #   依据简报 §2.1：截断计在 agent_run 上
  "context_limit_tokens": 131072,            # int，缺省上下文上限（设计假设）
  "n_attempts": int,                         # 总行数
  "n_rollouts": int,                         # 出现的 rollout 数
  "n_rollouts_with_data": int,               # 有 ≥min_samples 有效样本的 rollout 数

  "series": [RolloutPoint, ...],            # 按 rollout_id 升序

  "degradation": {
    "verdict": "per_turn_slowdown" | "turn_count_blowup"
             | "context_exhaustion" | "healthy",
    "verdict_reasons": [str, ...],          # 人类可读的判定依据（每条带数字）
    "confidence": "high" | "medium" | "low",
    "slowdown_attribution": "model_output_growth" | "engine_or_infra"
                          | "mixed" | "unresolved" | "not_applicable",

    "first_rollout_id": int, "last_rollout_id": int,
    "first_window_rollout_ids": [int, ...],  # 首窗口
    "last_window_rollout_ids": [int, ...],   # 末窗口

    "per_turn_seconds_first": float|None, "per_turn_seconds_last": float|None,
    "per_turn_growth": float|None,           # (last-first)/first
    "tokens_per_turn_first": float|None, "tokens_per_turn_last": float|None,
    "tokens_per_turn_growth": float|None,
    "turns_first": float|None, "turns_last": float|None, "turns_growth": float|None,
    "peak_context_ratio_first": float|None, "peak_context_ratio_last": float|None,
    "time_budget_ratio_first": float|None, "time_budget_ratio_last": float|None,
    # 【v2 二次修订】三类开销分列，取代笼统的 queue_stolen/budget_stolen
    "sandbox_boot_queue_first": float|None, "sandbox_boot_queue_last": float|None,
    "queue_wait_ratio_first": float|None, "queue_wait_ratio_last": float|None,
    "prep_overhead_first": float|None, "prep_overhead_last": float|None,
    "infra_overhead_ratio_first": float|None, "infra_overhead_ratio_last": float|None,
  },

  "data_quality": {
    "min_samples": int,                      # 判定可信所需最小样本
    "low_confidence_rollouts": [int, ...],
    "missing_phase_time_rollouts": [int, ...],
    "per_turn_source": "model_generation" | "agent_run" | "mixed" | "missing",
    "notes": [str, ...],
  },

  "note": str,                               # 口径说明（页面直接展示）
}
```

### A.3.1 `RolloutPoint` 字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `rollout_id` | `int` | 时间轴 |
| `n_attempts` | `int` | 该 rollout 的行数 |
| `n_valid` | `int` | 同时具备 `agent_run_s` 与 `turns` 的行数（参与分位数计算） |
| `n_pass` / `n_fail` / `n_timeout` / `n_abort` / `n_other` | `int` | 来自 `classify_sample_outcome` |
| `n_time_budget` | `int` | `agent_outcome == "time_budget_exceeded"` 的行数 |
| `n_context_exhausted` | `int` | `agent_outcome == "context_exhausted"` 的行数 |
| `time_budget_ratio` | `float\|None` | `n_time_budget / n_attempts` |
| `context_exhausted_ratio` | `float\|None` | `n_context_exhausted / n_attempts` |
| `agent_run_p50` / `agent_run_p90` | `float\|None` | `agent_run_s` 分位数 |
| `model_generation_p50` | `float\|None` | `model_generation_s` 分位数 |
| `per_turn_seconds_p50` | `float\|None` | **单轮生成耗时**（见 A.4 定义） |
| `tokens_per_turn_p50` | `float\|None` | 单轮生成 token |
| `turns_p50` | `float\|None` | 轮数 p50 |
| `peak_context_p50` | `float\|None` | 上下文峰值 p50 |
| `peak_context_ratio_p50` | `float\|None` | `peak_context_p50 / context_limit_tokens` |
| `overhead_seconds_p50` | `float\|None` | `attempt_total_s − agent_run_s` 的 p50 |
| `queue_wait_seconds_p50` | `float\|None` | 纯排队（sandbox_boot_queue + group_sibling_wait + rollout_batch_wait）p50 |
| `sandbox_boot_queue_p50` | `float\|None` | **【实测】** `sandbox_boot_queue_s` 的 p50（task8 159.4s → 311.8s）。**注意与 `sandbox_boot_s` 区分** |
| `queue_wait_ratio_p50` | `float\|None` | `sandbox_boot_queue_p50 / attempt_total_p50`。**【实测】** 排队加剧撞墙（分桶撞墙率 20.7%/33.9%/80.0%）→ **吞吐杠杆（辅）** |
| `prep_overhead_p50` | `float\|None` | `cli_install + sandbox_boot + evaluator_total` 的 p50。**不竞争 agent 墙钟** → 成本/吞吐 |
| `infra_overhead_ratio_p50` | `float\|None` | `(attempt_total − agent_run) / attempt_total`。**【实测】** ro63 = 25.0% → 成本/吞吐 |
| `budget_hit_ratio` | `float\|None` | `agent_run_s ≥ budget × 0.98` 的占比（计划 §1.4(a) 原定义） |
| `cache_hit_ratio_p50` | `float\|None` | `cached_prompt_tokens / total_prompt_tokens` |
| `low_confidence` | `bool` | `n_valid < min_samples` |

## A.4 单轮指标的定义（**口径必须写死，否则曲线不可复算**）

设一次 attempt 的规范行为 `turns`（>0）、`agent_run_s`、`model_generation_s`、`generated_response_tokens`：

| 指标 | 定义 | 说明 |
| --- | --- | --- |
| `per_turn_seconds` | `model_generation_s / turns` | **主口径**：纯模型侧每轮生成耗时。若 `model_generation_s` 缺失，退化为 `agent_run_s / turns`，并在 `per_turn_source` 标记 `agent_run` |
| `per_turn_agent_seconds` | `agent_run_s / turns` | 含工具执行的每轮耗时（辅口径） |
| `tokens_per_turn` | `generated_response_tokens / turns` | 单轮生成 token |
| `overhead_seconds` | `attempt_total_s − agent_run_s` | 预算外总开销（简报 §2.2 的 278→1103s） |
| `queue_wait_seconds` | `sandbox_boot_queue_s + group_sibling_wait_s + rollout_batch_wait_s` | 纯排队（简报 §4 列出但从未被使用的三个字段） |
| `model_share` | `model_generation_s / agent_run_s` | 模型时间占 agent 预算比例（A.6 判据用） |

**口径校验（简报 §4.2，task4 phase 平均值）**：`agent_run 1524.1`、`model_generation 1359.7` → `model_share ≈ 0.89`，说明 `model_generation` 是 `agent_run` 的**子集**（不含 `cli_install/sandbox_boot/evaluator_total`），与 `attempt_total 2017.9` 的关系为 `agent_run + 准备/评测类 − 重叠 ≈ attempt_total`（四个非 agent phase 之和 615.6s > 差值 493.8s，存在 ~122s 重叠）。→ **`overhead_seconds` 不是纯排队**，必须与 `queue_wait_seconds` 分列，否则会把「装 CLI 慢」误读成「排队慢」。

**设计假设标注**：简报未指明「单轮生成耗时」= `model_generation/turns` 还是 `agent_run/turns`。本设计取 `model_generation/turns` 为主口径（因为要分离「模型慢」与「工具/基建慢」，必须用模型侧时间），并**同时输出** `per_turn_agent_seconds`；两种口径下 task8 的方向一致（+244% / 更大），不改变结论。

## A.5 三类开销的分离度量（**v2 二次修订 · 已定稿**）

> **【实测修正 · 2026-11-23 · 二次】** 本节经过两轮错误修订，最终结论如下。
> **两版都错在同一个地方：把 `phase_time/sandbox_boot_queue` 误读成了 `phase_time/sandbox_boot`。**
>
> - v2 初稿：把 `attempt_total − agent_run` 命名为 `queue_stolen_ratio` / `budget_stolen_ratio`，
>   并建议「把排队移出 agent 预算」；
> - 第一轮复核：因误读字段（读到 p50 仅 5.5s），错误地**整个删除**了该建议；
> - **本轮定稿（正确）**：见下方实测。

### A.5.1 三个必须分清的字段（**【实测】** task8 逐 rollout）

| 字段 | ro0 p50 | ro60 p50 | 语义 |
| --- | --- | --- | --- |
| `phase_time/sandbox_boot_queue` | **159.4s** | **188.7s** | **排队等待**（沙箱就绪前） |
| `phase_time/sandbox_boot` | 10.4s | 5.2s | 沙箱实际启动（始终极小） |
| `phase_time/agent_run` | 1317.3s | 3360.0s | agent 运行（撞墙上限 3600s） |

**【实测】** 逐 rollout `sandbox_boot_queue` p50：159.4 → 165.0 → 162.8 → 188.4 → **300.2** →
229.8 → 241.0 → **311.8s**（ro0 → ro63）。
TB 独立印证：`perf/rollout_time/attempt/sandbox_boot_queue` = **152.1s → 392.0s**。

> **实现注意**：读 `.pt` 时必须用 `metadata["phase_time/sandbox_boot_queue"]`，
> **不要**误用 `metadata["phase_time/sandbox_boot"]`（两者差一个数量级）。

### A.5.2 排队是否影响撞墙：**有影响，但是辅因**

**【实测】** 撞墙样本 `agent_run` **恒 ≥3600s**（1425 例，p50=3614s，100% ≥3600）
→ 3600s 是独立墙钟上限。

**【实测】** 按 `sandbox_boot_queue` 分桶的撞墙率：

| 分桶 | rollout 55（n=128） | 多 rollout 汇总（ro 20/30/40/50/55/60/63, n=897） |
| --- | --- | --- |
| `q < 100s` | 40.6%（13/32） | **20.7%**（57/275） |
| `100s ≤ q < 250s` | 52.8%（19/36） | **33.9%**（82/242） |
| `q ≥ 250s` | **80.0%**（48/60） | 30.0%（114/380） |

→ **排队确实加剧撞墙概率**（单调上升），但**解释不了**从 0% → 62.5% 的整体跃升。
→ **第一主因仍是「单轮变慢」**：撞墙 100% 落在 `agent_run ≥ 3600s`，说明 agent 自身跑不完。

### A.5.3 三类开销的分离设计（**本节定稿**）

**【设计】** 模块输出**分列三类**，不提供笼统的 `budget_stolen_ratio`：

| 类别 | 字段 | 定义 | 与 agent 墙钟的关系 | 杠杆归属 |
| --- | --- | --- | --- | --- |
| **模型变慢** | `per_turn_seconds_p50` | `model_generation_s / turns` | **竞争**（第一主因） | **奖励杠杆** |
| **排队等待** | `sandbox_boot_queue_p50` | `sandbox_boot_queue_s` 的 p50 | **部分竞争**（加剧撞墙） | **吞吐杠杆（辅）** |
| | `queue_wait_ratio` | `sandbox_boot_queue_p50 / attempt_total_p50` | 同上 | 吞吐杠杆（辅） |
| **准备/评测** | `prep_overhead_p50` | `cli_install + sandbox_boot + evaluator_total` 的 p50 | **不竞争**（在 `agent_run` 之外） | 成本/吞吐 |
| | `infra_overhead_ratio` | `(attempt_total − agent_run) / attempt_total` | **不竞争** | 成本/吞吐 |
| | `throughput_attempts_per_hour` | `3600 / attempt_total_p50` | — | 成本/吞吐 |

**【设计】** 保留 `budget_clock` 字段（默认 `"agent_run"`，依据实测），用于在平台改口径时切换杠杆归属。

**文案纪律**：
- 排队类：「排队等待占 attempt 总时长的比例」/「排队挤占有效推理时间」
- 准备/评测类：「非 agent 时间占总 attempt 的比例」
- **禁止**：「被偷走的预算」「排队吃掉了 agent 预算」

> **【实测】** task4 印证：`cli_install` 平均 **271.9s**，是除 `agent_run` 外的第二大开销，
> 在 agent 墙钟之外 → 纯成本问题。


## A.6 三种退化模式的判定规则与阈值（含推导）

### A.6.0 判定前置：首/末窗口取值

**不用单点**（简报的 rollout 0 / 63 是单点，噪声大），改用**窗口中位数**：

- `first_window`：前 `w` 个 rollout，`w = clamp(round(0.2 × n_rollouts), 3, 10)`
- `last_window`：后 `w` 个 rollout，同上
- 窗口内对**逐 rollout 的 p50** 再取中位数（先按 rollout 聚合，再跨 rollout 聚合，避免 attempt 多的 rollout 主导）
- 同时保留 `first_rollout_id` / `last_rollout_id` 单点值供 UI 展示（对应简报的 rollout 0 / 63）

### A.6.1 规则表（**按优先级从上到下，命中即停**）

| 优先级 | verdict | 触发条件（全部满足） | 阈值来源 |
| --- | --- | --- | --- |
| 1 | `per_turn_slowdown` | ① `per_turn_growth ≥ 0.50`；② `tokens_per_turn_growth ≥ 0.30`；③ `turns_growth ≤ 0.05` | ① task8 **+244%** vs task9 **+40.5%** → 取 0.50（task9 上方 9.5pp，task8 下方远）；③ task8 **−37%**、task9 **+5.6%** → 取 0.05（task9 上方 0.6pp? **偏紧** → 改用 0.10，见下） |
| 2 | `turn_count_blowup` | `turns_growth ≥ 0.15` | task9 健康漂移 **+5.6%** → 阈值取 2.7× 于健康基线（0.15）；task8 为 **−37%**，不触发 |
| 3 | `context_exhaustion` | ① `peak_context_ratio_last ≥ 0.90`；② `context_exhausted_ratio_last ≥ 0.10` | ① task8 峰值上下文 **88K→66K**（对 131072 上限 = 67%→50%），远低于 0.90 → 不触发；task4 `context_exhausted` 1562/8752 = **17.8%** ≥ 0.10 → 可触发 |
| 4 | `healthy` | 以上均不满足 | — |

**阈值修正说明（规则 1 的条件 ③）**：task9 的 `turns` 漂移是 **+5.6%**，若把「不构成轮数爆炸」的上限设为 0.05，则 task9 距阈值只有 0.6pp，太紧、极易误判。→ **取 `turns_growth ≤ 0.10`**（task9 +5.6% 有 4.4pp 余量；task8 −37% 有 37pp 余量）。**设计假设**，依据是 task9 唯一健康基线的实测漂移。

**规则 1 条件 ②（`tokens_per_turn_growth ≥ 0.30`）的定位**：简报**未给 task9 的单轮 token 数据**，只有 task8 的 375→945(+152%)。因此条件 ② 的阈值 0.30 是**设计假设**，作用不是「判定 task9 健康」（task9 由条件 ① 直接判掉），而是**区分「模型输出变长」与「引擎变慢」**（见 A.7）。若 task9 后续数据显示其 token/turn 也增长 > 30%，规则 1 会因条件 ① 不满足（+40.5% < 50%）仍判 `healthy`，不受影响。

### A.6.2 判定优先级为何这样排（依据简报的互斥证据）

简报 §2.2 明确给出三种模式的**互斥证据**：

| 模式 | 支持证据 | 反对证据 |
| --- | --- | --- |
| `per_turn_slowdown` | 单轮耗时 +244%、单轮 token +152%、`time_budget_exceeded` 峰值 62.5% | turns **下降**、peak_context **下降** → 排除轮数爆炸与上下文打满 |
| `turn_count_blowup` | 计划保留项 | task8 实测 turns 57→36 **下降** → 本批任务**不触发** |
| `context_exhaustion` | task4 有 1562 例 `context_exhausted` | task8 peak_context 88K→66K **下降** → 本批任务**不触发** |

因此排序按「**本批实证主因在前**」，与计划 §1.4(a) 一致：单轮变慢（主因） > 轮数爆炸 > 上下文打满 > healthy。

### A.6.3 多信号并存时的表达

主判定唯一（`verdict`），但**次要信号必须一起输出**（否则会出现「turns 也在涨」的漏报）：输出 `secondary_signals: [str]`，取值如 `"turns_also_growing"`、``"peak_context_approaching_limit"`、`"queue_wait_high"`、`"infra_abort_share_high"`。

## A.7 **「模型变慢」vs「基建变慢」的区分（本方案最易误诊点）**

### A.7.1 为什么必须区分

简报 §2.3 的 abort 分布：agent 侧 `agent_time_budget_exceeded` 占 **85.9%**（1425/1658），基建/SDK（`openhands_infra_exit_2` 219 + `SandboxApiException` 7 + `SandboxExecStalled` 5）占 **13.9%**，评测器 2 例。
→ **「有 abort」不等于「基建问题」**。v1 方案「abort 升高 = 基建问题」的推断在此被推翻。

但简报同时给出两个**相反方向**的基建信号：
- `attempt_total − agent_run` 从 278s 涨到 1103s（**+297%**，基建在挤占墙钟）；
- `phase_time/sandbox_boot_queue` 从 ~5s 涨到 274–404s（纯排队在涨）。

**这两者都不是「模型变慢」**，但也**不是「基建坏了」**。真正需要区分的是三类：

| 类别 | 典型成因 | 对策完全不同 |
| --- | --- | --- |
| **模型行为变慢** | 模型学会每轮输出更多 token（task8：375→945，+152%） | 限长 / prompt 约束 / 提高预算 |
| **引擎/服务变慢** | 单 token 生成变慢（prefill、KV、prefix cache 失效） | 查 prefix cache 命中率、推理引擎负载 |
| **基建排队变慢** | 沙箱排队、同组兄弟等待、批次等待 | 扩容 / 错峰 / 预算计时口径 |

### A.7.2 判据：`slowdown_attribution`

在 `verdict == "per_turn_slowdown"`（或 `per_turn_growth ≥ 0.20` 的弱信号）时计算：

| 判据 | 数据来源 | 规则 |
| --- | --- | --- |
| C1 **输出变长** | `tokens_per_turn_growth` | `≥ 0.30` → 输出变长证据成立 |
| C2 **模型时间占比未下降** | `model_share_last − model_share_first` | `≥ −0.05` → 增量确实花在模型上 |
| C3 **工具/排队占比上升** | `tool_execution` 占 `agent_run` 的变化、`queue_wait_seconds` 的变化 | 任一 `≥ +50%` → 基建证据成立 |
| C4 **前缀缓存失效** | `cache_hit_ratio` 变化 | `≤ −0.15`（相对下降 15pp）→ prefill 变慢证据成立（简报 §4 列出但从未使用的字段，本次启用） |

**归因规则（按序）**：

| `slowdown_attribution` | 条件 |
| --- | --- |
| `model_output_growth` | C1 成立 **且** C3 不成立 **且** C2 成立 |
| `engine_or_infra` | C3 或 C4 成立，**且** C1 不成立 |
| `mixed` | C1 与 (C3 或 C4) 同时成立 |
| `unresolved` | 关键字段缺失（无 `model_generation_s` / 无 `tokens_per_turn`）→ **不猜**，明确标「数据不足」 |
| `not_applicable` | `verdict != "per_turn_slowdown"` 且无弱信号 |

**task8 的预期归因（用简报数据反推）**：C1 成立（375→945 = +152%）；C3 **不成立**（简报未报 `tool_execution` 上升，且 `agent_run` 的增量由 `model_generation` 解释）；C2 需 `model_share` 数据，简报 §4.2 的 task4 均值 `model_generation/agent_run = 1359.7/1524.1 = 0.89` 说明模型占主导 → 预期 `model_output_growth`。

**task9 的预期归因**：单轮 +40.5% < 50% 阈值 → `verdict = healthy`，`slowdown_attribution = not_applicable`。

### A.7.3 反向排除：基建「故障」 vs 基建「变慢」

- **基建故障**（沙箱报错、SDK 报错）→ 用 `abort_reason` 分布判定，`openhands_infra_exit_2` / `exception:Sandbox*` 占比；task8 仅 13.9%（含 `openhands_infra_exit_2`），纯沙箱异常 0.72%。
- **基建变慢**（排队、等待）→ 用 `overhead_seconds` / `queue_wait_seconds` 判定，见 A.5。
- 两者的处置完全不同：故障 → 修沙箱/SDK；变慢 → 调预算计时口径 / 扩容 / 错峰。**判据必须分开写，不能用「abort 多」一句话带过。**

## A.8 函数签名（新增 `utils/agent_budget_profile.py`）

```python
# utils/agent_budget_profile.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "DEFAULT_BUDGET_SECONDS",
    "DEFAULT_CONTEXT_LIMIT_TOKENS",
    "DEGRADATION_VERDICTS",
    "normalize_attempt_row",
    "build_budget_profile",
    "summarize_budget_rollouts",
    "classify_degradation",
    "attribute_slowdown",
    "budget_health_flags",
]

DEFAULT_BUDGET_SECONDS: int = 3600
DEFAULT_CONTEXT_LIMIT_TOKENS: int = 131072
DEGRADATION_VERDICTS: tuple[str, ...] = (
    "per_turn_slowdown", "turn_count_blowup", "context_exhaustion", "healthy",
)

def normalize_attempt_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """把一行 attempt（.pt metadata / artifacts summary / DB 行）归一成规范行。

    接受 phase_time 的三种写法：扁平 ``phase_time/agent_run``、
    嵌套 ``phase_time: {"agent_run": ...}``、artifacts ``phase_times``。
    缺失一律 None（``context_limit_tokens`` 由调用方参数补）。
    """

def build_budget_profile(
    rows: Sequence[Dict[str, Any]],
    *,
    budget_seconds: int = DEFAULT_BUDGET_SECONDS,
    budget_clock: str = "agent_run",
    default_context_limit: int = DEFAULT_CONTEXT_LIMIT_TOKENS,
    min_samples: int = 8,
    window_ratio: float = 0.2,
    window_min: int = 3,
    window_max: int = 10,
) -> Dict[str, Any]:
    """画像主入口：attempt 行 → 逐 rollout series + 退化判定 + 预算元信息。"""

def summarize_budget_rollouts(
    rows: Sequence[Dict[str, Any]],
    *,
    budget_seconds: int = DEFAULT_BUDGET_SECONDS,
    default_context_limit: int = DEFAULT_CONTEXT_LIMIT_TOKENS,
    min_samples: int = 8,
) -> List[Dict[str, Any]]:
    """只做逐 rollout 聚合（不做退化判定），返回 series。"""

def classify_degradation(
    series: Sequence[Dict[str, Any]],
    *,
    window_ratio: float = 0.2,
    window_min: int = 3,
    window_max: int = 10,
    per_turn_growth_threshold: float = 0.50,
    tokens_per_turn_growth_threshold: float = 0.30,
    turns_growth_threshold: float = 0.15,
    turns_stable_ceiling: float = 0.10,
    peak_context_ratio_threshold: float = 0.90,
    context_exhausted_ratio_threshold: float = 0.10,
) -> Dict[str, Any]:
    """按优先级判定退化模式；返回 degradation 字典（含 verdict / reasons / confidence）。"""

def attribute_slowdown(
    series: Sequence[Dict[str, Any]],
    *,
    tokens_growth_threshold: float = 0.30,
    model_share_drop_tolerance: float = 0.05,
    infra_growth_threshold: float = 0.50,
    cache_hit_drop_threshold: float = 0.15,
) -> str:
    """区分「模型变慢」与「基建变慢」：model_output_growth / engine_or_infra / mixed / unresolved。"""

def budget_health_flags(
    profile: Dict[str, Any],
    *,
    queue_wait_ratio_warn: float = 0.10,
    queue_wait_ratio_bad: float = 0.20,
    infra_overhead_warn: float = 0.15,
    infra_overhead_bad: float = 0.25,
    sandbox_queue_warn: float = 0.05,
    sandbox_queue_bad: float = 0.15,
) -> List[Dict[str, Any]]:
    """输出预算挤占告警列表（severity: ok/warn/bad + 数值 + 建议动作文案）。"""
```

## A.9 关键实现要点

### A.9.1 分位数计算

- **逐 attempt 先算派生量，再按 rollout 聚合**：`per_turn_seconds` 等是 attempt 级量，必须先算后取分位；不能先对 `model_generation_s` 取分位再除以 turns（会低估尾部）。
- **分位算法**：**线性插值法**（`numpy` 风格的 `(n-1) * p` 位置线性插值），与 TB `p50`/`p95` 的常规口径一致；**禁止**用「取第 k 个元素」的近似法（n 小时会跳变）。
- 只用 `statistics`/纯 Python 实现（仓库 `utils/` 无 numpy 强依赖；`utils/tb_series_reader.py` 也是纯标准库）。

### A.9.2 缺失值处理（**逐字段独立，不做整行丢弃**）

| 情况 | 处理 |
| --- | --- |
| `turns` 缺失或 ≤ 0 | 该 attempt 不参与 `per_turn_seconds` / `tokens_per_turn` 的分位计算，但**仍参与** `agent_run_s` 分位与 `n_attempts` 计数 |
| `model_generation_s` 缺失 | `per_turn_seconds` 退化为 `agent_run_s / turns`，`per_turn_source` 记 `agent_run`；若该 rollout 内 >50% 的样本走了退化口径 → 标 `low_confidence` |
| `agent_run_s` 缺失 | 不参与 `budget_hit_ratio` / `queue_wait_ratio`；不影响 `turns` 统计 |
| `attempt_total_s` 缺失 | `overhead_seconds` / `infra_overhead_ratio` / `queue_wait_ratio` 该样本缺失（**不影响撞墙判定**，撞墙只看 `agent_run`） |
| `context_limit_tokens` 缺失 | 用 `default_context_limit`（131072，设计假设）；`peak_context_ratio` 正常计算 |
| `generated_response_tokens` 缺失 | `tokens_per_turn` 缺失 → `attribute_slowdown` 返回 `unresolved`（**不猜**） |
| `agent_outcome` 缺失 | `n_time_budget` 不计（**不用 `status` 猜**：`status=aborted` 既可能是时间预算也可能是沙箱故障，简报 §2.3 显示 abort 原因高度异质） |

**关键纪律**：**abort 样本的 `reward=0.0` 不能当失败**（`utils/rollout_pt_parser.resolve_sample_reward` 已有该口径，A 必须复用，不能另写）。

### A.9.3 样本量不足时的降级

| 场景 | 降级策略 |
| --- | --- |
| 单 rollout `n_valid < min_samples`（默认 8） | 该点 `low_confidence = True`；**仍进入 series**（让用户看到缺口），但**不参与**退化判定的窗口中位数 |
| 可用 rollout 数 `< 3` | `verdict = "healthy"`，`confidence = "low"`，`verdict_reasons` 写「样本不足，无法判定」 |
| 有效 rollout 数 ≥ 3 但 < 6 | `confidence = "medium"` |
| 有效 rollout 数 ≥ 6 且 `n_valid` 覆盖率 ≥ 80% | `confidence = "high"` |
| `per_turn_source == "agent_run"` 占比 > 50% | `confidence` 降一级，`verdict_reasons` 追加「单轮耗时口径退化（含工具执行）」 |

`min_samples = 8` 的依据：**每组固定 8 attempts**（简报 §3「每步固定 16 组」+ 组内 8 attempts 的 GRPO 配置），即一个 rollout 通常有 16 组 × 8 = 128 attempts；取 8 是「单 rollout 内至少 8 个有效样本才让 p50 稳到 ±1/8 量级」的**设计假设**。

### A.9.4 数值与类型

- 所有比率字段用 `float`，**缺失一律 `None`**（不返回 `0`，避免「没数据」被读成「0%」）。
- 计数用 `int`。
- 序列按 `rollout_id` 升序。

## A.10 与现有代码的复用关系

| 关系 | 说明 |
| --- | --- |
| **复用** `utils.rollout_pt_parser.classify_sample_outcome` | attempt → category，**不重写**分类 |
| **复用** `utils.rollout_pt_parser.resolve_sample_reward` | abort 不可当失败 |
| **复用** `utils.failure_reason_taxonomy.agent_outcome_reason` | `agent_outcome` → reason 码（A 的 `n_time_budget` 判定可与 B 对账） |
| **复用** `utils.attempt_artifacts.summarize_entry` | **P2 扩展**（补 `phase_times` / `attempt_generated_response_tokens` / `context_limit_tokens`），A 不另建 summary 解析 |
| **复用** `utils.rollout_pt_parser.load_pt` | 无 artifacts 时的 `.pt` 回退取数（在**行构造器**里用，不在 A 内） |
| **不就地扩展** `service/reward_diagnosis_service` | A 是纯函数，不碰 service；由 service 调用（A.2.3） |
| **与 `service/problem_analysis_service` 无耦合** | A 不涉及通过率 |
| **新 `func.md` 登记** | `utils/agent_budget_profile.py` 的 8 个公开函数会被 `scripts/scan_func_md.py` 自动登记（`scan_utils` 只收模块级公开函数） |

## A.11 风险与边界

| 风险 | 边界与对策 |
| --- | --- |
| **单轮耗时口径不确定**（简报未指明 `model_generation/turns` vs `agent_run/turns`） | 主口径 + 辅口径同时输出，`per_turn_source` 显式标注；阈值对两者都成立（task8 方向一致） |
| **`phase_time` 覆盖率低** | 逐字段缺失降级；`data_quality.missing_phase_time_rollouts` 显式列出；`n_valid` 低的 rollout 不进判定窗口 |
| **artifacts 与 `.pt` 双源不一致** | 优先 artifacts（有 `phase_times`），缺失回退 `.pt`；行上带 `source` 标记，便于对账 |
| **一个 attempt 多 segment** | `attempt_total_s` 是 attempt 级量，多 segment 必须**取 max 而非 sum**（否则重复计时）；实现约定：行构造器已按 attempt 去重 |
| **`context_limit_tokens` 缺失** | 用 131072 默认值并标「设计假设」；若该任务实际上限不同，`context_exhaustion` 可能漏判 → 输出 `data_quality.notes` 提示 |
| **窗口中位数掩盖突变** | 同时输出单点 `first_rollout_id/last_rollout_id` 值；`verdict_reasons` 里带上两端的数字 |
| **多任务混合** | 模块不做任务级聚合；调用方按 task 分开调用 |
| **不做因果证明** | `per_turn_slowdown` 是**相关性判定**（时间轴共变），不是因果；文案需写「与…共变」，不写「导致」 |

---

# 交付物 B：`utils/failure_reason_taxonomy.py` 扩展设计

## B.1 目标与依据

**目标**：让 `time_budget_exceeded`（时间预算耗尽）与 `context_exhausted`（上下文打满）**在任何字段组合下都能被区分**，同时**不破坏**现有页面与既有 category 取值。

**依据**：

1. 简报 §2.1：task8 有 1363 个 `time_budget_exceeded` 样本，是第一主因。
2. 简报 §4.1：task4 的 `agent_outcome` 分布 `finished 6253 / context_exhausted 1562 / time_budget_exceeded 524 / (None) 302 / sdk_error 82 / max_fake_responses 14 / max_iterations_reached 11`；`diagnosis` 分布 `resolved 5340 / evaluator:tests_failed 2239 / agent:time_budget_exceeded 450 / (None) 384 / evaluator:invalid_evaluator_payload 203 / agent:no_diff 121 / evaluator:apply_failed 15`。
3. 简报 §1.2：三种机制的处置完全相反（提高预算 / 限长 vs 调上下文 vs 换题），混同会给出**错误处方**。

**现状（读代码确认）**：`normalize_reason` 的判定链是 `abort → timeout → pass/fail → other`，其中：

- `_TIMEOUT_TOKENS` 含 `"time_budget"` 与 `"context_exhausted"`，两者都会落进 `timeout`；
- `agent_outcome` **不参与** `_TIMEOUT_TOKENS` 的字符串匹配（只参与 `err or agent_outcome_reason(agent)` 的取值），因此 `status=truncated + agent_outcome=time_budget_exceeded` 与 `status=truncated + agent_outcome=context_exhausted` **返回的 category 完全相同**，只有 reason 码不同；
- 更糟的是**同一现象会落到三个不同 category**：`abort_reason="agent_time_budget_exceeded"` → `("abort", ...)`；`status="truncated"` → `("timeout", ...)`；`status="completed"` 且无 reward → `("other", ...)`（现有测试 `tests/test_failure_reason_taxonomy.py:88` 就断言了后者）。

→ 这不只是「timeout 混同」，而是**同一失败态的 category 不稳定**。扩展必须同时解决这两点。

## B.2 设计原则：**category 不动，新增正交「归因轴」**

### B.2.1 为什么不能新增 category（**消费方盘点**）

| 消费方 | 位置 | 对 category 的依赖 | 新增 category 的后果 |
| --- | --- | --- | --- |
| `service/failure_analysis_service.FAILURE_CATEGORIES` | `("fail","timeout","abort","other")` | 过滤集合 | 需同步改常量 |
| `service/failure_analysis_service._CATEGORY_KEYS` | `frozenset(CATEGORY_LABELS.keys())` | 校验集合 | 跟着 `CATEGORY_LABELS` 变 |
| `utils/failure_insight._build_category_clusters` | `key_fn = {"fail":…, "timeout":…, "abort":…}[category]` | **硬编码字典下标** | 新 category → **KeyError 崩页** |
| `service/task_annotation_service.assign_stack_type_colors` | 按 `key` 前缀 `timeout:` / `fail:` / `abort:` 分桶 | 前缀解析 | 新前缀落 `_STACK_OTHER_COLOR`，颜色/图例错乱 |
| `utils/rollout_pt_parser.summarize_stack_type_breakdown` | `order_cat = {pass,fail,timeout,abort,other}` + `order_flag` | 段排序 | 新 category 排到末尾，类型柱顺序变 |
| `db/db_attempt_service._classify_attempt_row` | 5 分类 | 结局柱口径 | DB 侧不产新 category，`.pt` 侧产了 → **两端不一致** |
| `templates/failure_overview.html` | `CAT_ZH = {pass,fail,timeout,abort,other}` + `<option value="timeout">` 过滤器 | 硬编码 | 过滤器漏选项、颜色 class 缺失 |
| `templates/failure_attempt.html` | `cat_zh` 字典 + `.cat-*` CSS | 硬编码 | 同上 |
| `templates/task_detail.html` | `.lg-timeout` / `.lg-abort` / `.seg-timeout` 等 CSS | 硬编码 | 类型柱颜色缺失 |
| `templates/rollout_detail.html` | `.cat-timeout` / `.trend-seg.cat-timeout` | 硬编码 | 同上 |
| `service/reward_diagnosis_service.classify_group_kind` | `{"abort","timeout"}` 跳过集合 | 语义 | 新 category 会被当成「可测」→ **mixed 判定错误** |
| `service/reward_diagnosis_service.summarize_reward_split` | `pass/fail/timeout/abort/other` 分支 | 计数 | 新 category 落 `n_other` |
| `utils/rollout_pt_parser.classify_sample_outcome` | `{"pass","abort","timeout","fail","other"}` | `.pt` 侧产出 | 新增需同步 |

→ **结论：新增 category 需同步 12+ 处，且其中 2 处是硬崩溃点**。违反「改动不能破坏现有页面」的约束。**采纳方案：category 保持不变，靠 reason 码 + 新增正交归因轴区分。**

### B.2.2 新增「归因轴」`cause_axis`（正交，不参与既有分组）

| `cause_axis` | 语义 | 覆盖的 reason / 字段 |
| --- | --- | --- |
| `agent_policy` | 策略本身没做对（可学） | `evaluator:tests_failed`、`agent:no_diff`、`agent:no_recorded_tool_calls` |
| `agent_budget` | **agent 时间预算耗尽（墙钟截断）** | `agent_time_budget_exceeded`、`agent:time_budget_exceeded`、`status:aborted` + `agent_outcome=time_budget_exceeded` |
| `context_limit` | **上下文打满** | `slime_context_limit_reached`、`agent_outcome=context_exhausted`、`driver_context_limit` |
| `agent_turns` | 轮次/迭代/假响应上限 | `agent:max_iterations_reached`、`agent:max_fake_responses`、`agent:turn_limit_reached` |
| `evaluator` | **评测器结论**（含评测器自身故障） | `evaluator:*` |
| `infra` | 基建 / SDK / 沙箱 | `exception:Sandbox*`、`openhands_infra_exit_2`、`agent:sdk_error` |
| `unknown` | 其它 | — |

**关键：`cause_axis` 不进入任何既有分组逻辑**，它是**附加字段**，供页面在既有 category 之上做二级细分与排序。

## B.3 新增 / 变更的 reason 码与映射

### B.3.1 `AGENT_OUTCOME_REASON_CODES` 现状与增量

现状：

```python
AGENT_OUTCOME_REASON_CODES = {
    "context_exhausted": "slime_context_limit_reached",
    "time_budget_exceeded": "agent_time_budget_exceeded",
    "sdk_error": "agent:sdk_error",
    "max_iterations_reached": "agent:max_iterations_reached",
    "max_fake_responses": "agent:max_fake_responses",
    "turn_limit_reached": "agent:turn_limit_reached",
    "malformed_action": "agent:malformed_action",
}
```

**增量（新增 3 条别名 + 1 条诊断码）**：

| 新增键 | 值 | 依据 |
| --- | --- | --- |
| `"agent:time_budget_exceeded"` | `"agent_time_budget_exceeded"` | 简报 §4.1 的 `diagnosis` 分布里真实存在 `agent:time_budget_exceeded`（450 例），现有映射把 `time_budget_exceeded` 映射成**无冒号**的 `agent_time_budget_exceeded`，两者不是同一个串 → 需要别名 |
| `"context_exhausted"` | 保持 `"slime_context_limit_reached"` | 不动（`.pt` 侧 `conversation_error_code` 用这个值） |
| `"max_response_tokens_exceeded"` | `"agent:max_response_tokens_exceeded"` | **设计假设**：部分 harness 用输出上限而非上下文上限截断，与 `context_exhausted` 语义不同 |
| `"budget_exhausted"` | `"agent_time_budget_exceeded"` | **设计假设**：部分运行把预算耗尽写成 `budget_exhausted` |

> **不变式**：`agent_outcome_reason("context_exhausted")` 必须仍返回 `"slime_context_limit_reached"`、`agent_outcome_reason("time_budget_exceeded")` 必须仍返回 `"agent_time_budget_exceeded"`（现有测试 `tests/test_failure_reason_taxonomy.py:109-113` 断言），**只增不改**。

### B.3.2 `cause_axis` 判定表（**优先级从高到低**）

| 序 | 条件 | `cause_axis` |
| --- | --- | --- |
| 1 | `agent_outcome` ∈ {`time_budget_exceeded`, `budget_exhausted`} **或** `diagnosis` ∈ {`agent:time_budget_exceeded`} **或** `abort_reason` 含 `time_budget` | `agent_budget` |
| 2 | `agent_outcome` ∈ {`context_exhausted`, `max_response_tokens_exceeded`} **或** `error_code` ∈ {`slime_context_limit_reached`} **或** `diagnosis` 含 `context_limit` | `context_limit` |
| 3 | `agent_outcome` ∈ {`sdk_error`} **或** `abort_reason`/`diagnosis` 以 `exception:Sandbox` / `openhands_infra_exit` 开头 | `infra` |
| 4 | `agent_outcome` ∈ {`max_iterations_reached`, `max_fake_responses`, `turn_limit_reached`, `malformed_action`} | `agent_turns` |
| 5 | `diagnosis` 以 `evaluator:` 开头 | `evaluator` |
| 6 | `diagnosis` 以 `agent:` 开头（其它） | `agent_policy` |
| 7 | 其它 / 无法判定 | `unknown` |

> **序 1 高于序 2 的必要性**：一条 attempt 可能同时带 `status=truncated` 与 `agent_outcome=time_budget_exceeded`；也可能是 `abort_reason="agent_time_budget_exceeded"` + `diagnosis="agent:time_budget_exceeded"`。**先判预算，再判上下文**，保证「预算耗尽」不会被上下文轴吞掉。

### B.3.3 `normalize_reason` 的兼容性改造（**不改变已有返回的 category**）

现有 6 个分支的语义全部保留；**只做两处最小改动**：

1. **新增 `cause_axis` 的计算入口**（新函数，不改 `normalize_reason` 的签名与返回）：

```python
def normalize_reason(
    *, status=None, reward=None, diagnosis=None, abort_reason=None,
    error_code=None, agent_outcome=None,
) -> Tuple[str, str]:      # ← 签名与返回类型不变（向后兼容）
    ...
```

2. **`classify_from_fields` 增加一个「三段式」兄弟函数**（**不改** `classify_from_fields` 的签名与返回）：

```python
def classify_with_axis(fields: Dict[str, Any]) -> Dict[str, str]:
    """在 (category, reason) 之上补一条归因轴。

    Returns: {"category": str, "reason": str, "cause_axis": str, "cause_label": str}
    """
```

> **为什么用新函数而不是改 `normalize_reason` 的返回元组**：返回元组从 2 元变 3 元会**破坏所有 `category, reason = normalize_reason(...)` 的调用点**（`utils/failure_reason_taxonomy.classify_from_fields`、`controller/rollout_controller.py:405`、`tests/test_failure_reason_taxonomy.py` 等 6+ 处）。**新增兄弟函数是唯一零破坏方案。**

### B.3.4 `normalize_reason` 内部的两处「稳定化」修复（**必须做，但保持 category 语义**）

| 现状 | 问题 | 修复（不改 category 取值语义） |
| --- | --- | --- |
| `status=truncated + agent_outcome=time_budget_exceeded` → `("timeout", "agent_time_budget_exceeded")` | reason 码正确，但 `category=timeout` 会让 `classify_group_kind` 把它当「不可测但非基建」 | **保持 `timeout` 不变**；由 `cause_axis="agent_budget"` 承载区分。页面在 timeout 段内按 `cause_axis` 再分 |
| `status=completed + agent_outcome=time_budget_exceeded` → `("other", "agent_time_budget_exceeded")` | 与 `status=truncated` 的同因现象落不同 category → **同一失败态 category 不稳定** | **把 `agent_outcome ∈ {time_budget_exceeded, context_exhausted, sdk_error, max_iterations_reached, ...}` 的非空 agent 结束态统一按 `timeout`/`abort` 归类**（见下），使 category **只依赖失败原因**，不依赖 status 措辞 |

**修复后的优先级（category 层）**：

```
1. abort_reason 非空                → ("abort", abort_reason)            # 不变
2. status ∈ _ABORT_STATUSES         → ("abort", f"status:{status}")      # 不变
3. agent_outcome == "sdk_error"     → ("abort", "agent:sdk_error")        # 不变
4. agent_outcome ∈ _BUDGET_OUTCOMES → ("timeout", agent_outcome_reason()) # ← 新增：time_budget_exceeded 等
   （放在 status 分支之前还是之后？→ 放在 status 之后、reward 之前，
     保证 status=truncated 仍返回 timeout 且 reason 优先取 agent 侧码）
5. status ∈ _TIMEOUT_STATUSES       → ("timeout", err or agent_reason or diag or f"status:{status}")  # 不变
6. err/diag 含超时关键词             → ("timeout", ...)                    # 不变
7. reward≈1 / reward≈0              → ("pass"/"fail", ...)                # 不变
8. 其它                             → ("other", ...)                     # 不变
```

**兼容性核查（逐条对现有测试）**：

| 现有测试 | 输入 | 现状返回 | 修复后返回 | 结论 |
| --- | --- | --- | --- | --- |
| `test_abort_reason_wins` | `status=aborted, abort_reason=exception:SandboxApiException` | `("abort","exception:…")` | 不变（规则 1） | ✅ |
| `test_abort_by_status` | `status=aborted` | `("abort","status:aborted")` | 不变（规则 2） | ✅ |
| `test_timeout_by_status` | `status=truncated, reward=0.0` | `("timeout","status:truncated")` | 不变（规则 5） | ✅ |
| `test_timeout_uses_error_code_first` | `status=truncated, error_code=slime_context_limit_reached, agent_outcome=context_exhausted` | `("timeout","slime_context_limit_reached")` | **需校验**：规则 4 若先命中，`context_exhausted` 会给出 `slime_context_limit_reached` → **reason 相同**；但 `error_code` 优先于 `agent_outcome_reason` 的既有语义要保留 | ⚠️ 见下 |
| `test_timeout_falls_back_to_agent_outcome_code` | `status=truncated, agent_outcome=context_exhausted` | `("timeout","slime_context_limit_reached")` | 不变 | ✅ |
| `test_pass_and_fail_by_reward` | reward=1.0/0.0 | `("pass","resolved")` / `("fail","evaluator:tests_failed")` | 不变 | ✅ |
| `test_sdk_error_is_abort` | `agent_outcome=sdk_error` | `("abort","agent:sdk_error")` | 不变（规则 3） | ✅ |
| `test_other_without_reward` | `status=completed, diagnosis=""` | `("other",…)` | 不变 | ✅ |
| `test_agent_outcome_used_when_no_diagnosis` | `status=aborted, agent_outcome=time_budget_exceeded` | `("abort","status:aborted")` | 不变（规则 2 先于规则 4） | ✅ |
| `test_agent_outcome_reason_when_status_is_generic` | `status=completed, reward=None, agent_outcome=time_budget_exceeded` | `("other","agent_time_budget_exceeded")` | **变为 `("timeout","agent_time_budget_exceeded")`** | ❗ **唯一行为变更**，见 B.3.5 |

**关于 `test_timeout_uses_error_code_first` 的处理**：规则 4 只对 **`_BUDGET_OUTCOMES`**（时间预算类）提前，`context_exhausted` **不提前**（保持走规则 5/6，`error_code` 优先）→ 该测试完全不受影响。

### B.3.5 **唯一的行为变更及其理由**（必须写进 CHANGELOG）

`status=completed` + `agent_outcome=time_budget_exceeded` + 无 reward：现状 `("other", …)` → 修复后 `("timeout", "agent_time_budget_exceeded")`。

**理由**：
1. 简报 §4.1 实测 task4 有 **524** 例 `agent_outcome=time_budget_exceeded`；这类 attempt 若落在 `other`，会在「类型柱」「失败聚类」里被归到「其它」，**与「上下文打满」彻底混同**——正是本次要修的核心问题。
2. `other` 在 `classify_group_kind` 中**不算 abort/timeout** → 会被当作**可测**参与 mixed 判定 → **把不可测样本算成策略失败**，这是比混同更严重的口径错误。
3. 变更方向是 `other → timeout`：`timeout` 在既有消费方里统一被当作「不可测」（`classify_group_kind` 跳过、`summarize_reward_split` 单列、类型柱用 `rw1/rw0/none` 三细分）→ **语义更准，且不引入新 category**。

**同步动作**：修改 `tests/test_failure_reason_taxonomy.py::TestClassifyFromFields::test_agent_outcome_reason_when_status_is_generic` 的期望为 `("timeout", "agent_time_budget_exceeded")`，并在 CHANGELOG 记一条「口径修正」。

## B.4 `CATEGORY_LABELS` / `AGENT_OUTCOME_LABELS` 的增量

### B.4.1 `CATEGORY_LABELS` —— **键集不变，值不变**

```python
CATEGORY_LABELS: Dict[str, str] = {
    "pass": "通过",
    "fail": "正常失败",
    "timeout": "超时/截断",
    "abort": "abort",
    "other": "其它",
}
```

**必须保持 5 个键、值也保持原样**。原因：`service/failure_analysis_service._CATEGORY_KEYS = frozenset(CATEGORY_LABELS.keys())` 用它做校验集合，`utils/failure_insight._build_category_clusters` 用它做硬编码分派（见 B.2.1）。**改文案也会改 UI**（`build_group_failure_types` 的 `label`、`_attempt_item` 的 `category_label`），且 `templates/failure_overview.html` 有**自己的硬编码 `CAT_ZH`**——改 `CATEGORY_LABELS` 会造成两处文案不一致。**结论：一个字都不改。**

### B.4.2 新增独立标签表（**不污染既有键集**）

```python
CAUSE_AXIS_LABELS: Dict[str, str] = {
    "agent_policy":   "策略未做对（可学习）",
    "agent_budget":   "时间预算耗尽（墙钟截断）",
    "context_limit":  "上下文打满",
    "agent_turns":    "轮次/迭代上限",
    "evaluator":      "评测器结论",
    "infra":          "基建 / 沙箱 / SDK",
    "unknown":        "未归类",
}

REASON_LABELS: Dict[str, str] = {
    "agent_time_budget_exceeded":       "时间预算耗尽（撞 60 分钟墙钟）",
    "agent:time_budget_exceeded":       "时间预算耗尽（撞 60 分钟墙钟）",
    "slime_context_limit_reached":      "上下文打满",
    "driver_context_limit":             "上下文打满",
    "agent:sdk_error":                  "SDK 报错",
    "exception:SandboxApiException":    "沙箱 API 异常",
    "exception:SandboxExecStalled":     "沙箱执行卡死",
    "openhands_infra_exit_2":           "基建进程异常退出",
    "evaluator:tests_failed":           "评测未通过（策略问题）",
    "evaluator:invalid_evaluator_payload": "评测器载荷非法（评测器问题）",
    "evaluator:apply_failed":           "评测器执行失败（评测器问题）",
    "evaluator_total_timeout":          "评测器超时",
}
```

**`reason_label(category, reason)` 的扩展**：现状只返回 `reason` 原文（B.4.3 现状）。扩展为 **`REASON_LABELS` 命中 → 返回中文标签；未命中 → 保持返回原文**。

> **兼容性核查**：`reason_label` 现有调用方只有 `tests/`（3 处断言）与 `utils/failure_reason_taxonomy` 自身；**生产代码未使用**（已 Grep 确认）。但 `REASON_LABELS` 会改变 `reason_label("timeout","agent_time_budget_exceeded")` 的返回值（原文 → 中文）。为**零破坏**，采用**新增函数**而非改 `reason_label`：

```python
def reason_label(category: str, reason: str) -> str:      # 不变
    """(category, reason) → 中文短标签（保持原行为）。"""

def reason_display(category: str, reason: str) -> str:    # 新增
    """带具体原因码的展示标签：REASON_LABELS 命中 → 中文；否则回落 reason_label。"""
```

### B.4.3 `AGENT_OUTCOME_LABELS` 增量（**只增不改**）

现状已有：`finished / context_exhausted / time_budget_exceeded / sdk_error / max_iterations_reached / max_fake_responses / turn_limit_reached / malformed_action`。

| 新增键 | 值 | 依据 |
| --- | --- | --- |
| `"budget_exhausted"` | `"时间预算耗尽（别名）"` | **设计假设**：跨版本 harness 的别名 |
| `"max_response_tokens_exceeded"` | `"单次输出 token 超限"` | **设计假设**：与上下文打满语义不同，需分开 |
| `"unknown"` | `"未知结束态"` | 兜底 |

**同时建议（可选、向后兼容）**把 `"time_budget_exceeded"` 的文案从 `"时间预算耗尽"` 改为 `"时间预算耗尽（墙钟截断）"`，与 `"context_exhausted" → "上下文打满"` 形成明确对比。**但这是 UI 文案变更** → 需同步 `templates/failure_overview.html` 的硬编码 `CAT_ZH`（该处是 `agent_outcome_label` 的结果，不是硬编码，所以其实不用改）。**建议采纳**，并在 CHANGELOG 记一条。

## B.5 `evaluator:invalid_evaluator_payload` 的处置（**必须与策略失败区分**）

### B.5.1 事实

简报 §4.1：task4 中 `evaluator:invalid_evaluator_payload` **203 例（2.3%）**，`evaluator:apply_failed` 15 例，`evaluator:tests_failed` **2239 例**。前两者是**评测器自身出问题**，第三者是**策略没做对**。工具当前把三者全并入 `fail`（`normalize_reason` 规则 7：reward≈0 → `("fail", diagnosis)`）。

### B.5.2 结论：**必须区分，但不通过改 category 区分**

| 方案 | 评价 |
| --- | --- |
| 新增 `category="evaluator_error"` | ❌ 违反 B.2.1 的 12 处消费方约束 |
| **新增 `cause_axis="evaluator"` + 子标记 `evaluator_fault`** | ✅ **采纳** |

### B.5.3 `evaluator_fault` 判定

```python
EVALUATOR_FAULT_REASONS = frozenset({
    "evaluator:invalid_evaluator_payload",   # 简报 §4.1：203 例
    "evaluator:apply_failed",                 # 简报 §4.1：15 例
    "evaluator:timeout",                      # 评测器超时
    "evaluator_total_timeout",                # abort_reason 形式（简报 §2.3：2 例）
})
```

| 函数 | 语义 |
| --- | --- |
| `is_evaluator_fault(reason: Any) -> bool` | reason ∈ `EVALUATOR_FAULT_REASONS` → True。**注意 `evaluator:tests_failed` 不在其中**（评测器工作正常，是策略没做对） |
| `is_policy_failure(category: str, reason: str) -> bool` | `category == "fail"` 且 **不是** `evaluator_fault` → True（策略失败） |

### B.5.4 落法（**不改动既有的三口径**）

1. **`category` 保持 `"fail"`** → 类型柱 / 结局柱 / `classify_group_kind` / `summarize_reward_split` 全部不受影响。
2. **`cause_axis="evaluator"`**（与 `evaluator:tests_failed` 同轴）+ **`evaluator_fault=True`** 承载区分。
3. **新增统计字段（只加不改）**：
   - 在 `service/reward_diagnosis_service.summarize_reward_split` 的每个 series 点上**追加** `n_evaluator_fault` 与 `evaluator_fault_ratio`（分母 = 该 rollout 的 attempt 总数）。
   - 在 `summarize_reward_split` 的顶层**追加** `policy_failure_rate`（= `(n_fail − n_evaluator_fault) / n_attempts`）与 `evaluator_fault_count`。**现有字段一个都不改** → `tests/test_reward_diagnosis_service.py` 的断言全部不受影响。
4. **杠杆层**：在 `_LEVER_RULES` 追加一条 `evaluator_fault_ratio`（阈值：≥ 0.02 warn / ≥ 0.05 bad；依据 203/8752 = **2.3%** 落在 warn 档），动作文案「先修评测器（载荷非法/执行失败），别把评测器故障当策略失败去调 prompt」。
5. **与 §2.3（三口径）的关系**：计划 §2.3 的 `pass_rate_strict / gradable / slime_effective` 由该子任务负责。**本设计的口径边界**：
   - `pass_rate_strict`（全 attempt 分母）：**不变**；
   - `pass_rate_gradable`（去 abort/timeout）：**不变**；
   - `pass_rate_slime_effective`（仅未被丢弃的组）：**不变**；
   - `evaluator_fault` **不参与**三口径的分子分母调整，**只作为附加字段 + 独立杠杆**出现。
   → **理由**：改三口径的分母会同时改动任务详情页与问题分析页既有曲线（`static/ri_epoch_chart.js` 复用同一口径），风险外溢；把「评测器故障」从「策略失败率」里剥离是**诊断价值最高、破坏面最小**的做法。

## B.6 向后兼容总表（**逐消费方核查**）

| 消费方 | 读取的字段 | 本次变更影响 | 是否需要同步改动 |
| --- | --- | --- | --- |
| `templates/failure_overview.html` 失败分析页 | `category`（5 值）、`reason`、`agent_outcome_label` | `category` 键集不变；`time_budget_exceeded` 文案变更（可选） | ❌ 不需要 |
| 同上 · 过滤器 `<option>` | category 白名单 | 不变 | ❌ |
| `templates/task_detail.html` 类型柱 | `type_segments[].category` / `.key` / `.label` | `summarize_stack_type_breakdown` 的 `order_cat` 不变；新增 `timeout:agent_time_budget_exceeded` 这个 **reason 级细分**（不是新 category） | ❌（细分段自动出现，颜色由 `assign_stack_type_colors` 按前缀 `timeout:` 归族 → **已有配色**） |
| `templates/rollout_detail.html` 组结果列 | `build_group_failure_types` 的 `types[].category/label` | `category` 不变；`label` 由 `CATEGORY_LABELS` 取值 → 不变 | ❌ |
| `templates/failure_attempt.html` | `a.category` + `cat_zh` 硬编码 | `category` 不变 | ❌ |
| `utils/failure_insight._build_category_clusters` | category 硬编码 3 键 | 不变 | ❌（**这正是不能加 category 的原因**） |
| `service/task_annotation_service.assign_stack_type_colors` | `key` 前缀 | 新 reason 走 `timeout:` 前缀族 | ❌ |
| `service/reward_diagnosis_service.classify_group_kind` | `{"abort","timeout"}` | `other → timeout` 的修复让时间预算样本**正确地**不再算作可测 | ✅ 行为修正（正向） |
| `service/reward_diagnosis_service.summarize_reward_split` | category 分支 | 同上 | ✅（修正 `n_other` 归类） |
| `db/db_attempt_service._classify_attempt_row` | status/reward 5 分类 | **不动**（DB 侧无 `agent_outcome`） | ❌ |
| `utils/rollout_pt_parser.classify_sample_outcome` | `.pt` 侧 5 分类 | **建议同步**：`.pt` 侧也读 `metadata.agent_outcome`，应与 `normalize_reason` 同样把 `time_budget_exceeded` 归 `timeout`（否则两端口径漂移） | ✅ 需扩展（见 B.7） |
| `tests/test_failure_reason_taxonomy.py` | 断言 | 1 处期望需更新（见 B.3.5） | ✅ |
| `tests/test_cluster_attempt_outcomes.py` | `.pt` 侧分类 | 若同步扩展 `.pt` 侧，需加用例 | ✅ |

## B.7 `.pt` 侧 `classify_sample_outcome` 的同步扩展（**必须同步，否则两端漂移**）

`utils/rollout_pt_parser.classify_sample_outcome` 目前完全不看 `metadata.agent_outcome`（只看 `status` / `conversation_error_code` / `driver_error` / reward / diagnosis）。而 `.pt` 侧**有** `agent_outcome` 字段（简报 §4 列出，`utils/failure_insight` 已在读）。

**扩展方案（P2，就地扩展 `.pt` 侧）**：

```
在 classify_sample_outcome 中，于「timeout 判定」之前插入：
  agent = metadata.get("agent_outcome")
  if agent in _BUDGET_OUTCOMES:   # time_budget_exceeded / budget_exhausted
      return {"category": "timeout", "reason": "agent_time_budget_exceeded",
              "label": "timeout:agent_time_budget_exceeded"}
```

**兼容性**：`classify_sample_outcome` 的 5 个 category **不变**；现有测试 `tests/test_cluster_attempt_outcomes.py` 的 4 个用例均不涉及 `agent_outcome` → **不受影响**。

**收益**：`.pt` 侧「类型柱」的 timeout 段会自然细分出 `timeout:agent_time_budget_exceeded`（简报 §2.1 的 1363 例），与 artifacts 侧（`abort_reason=agent_time_budget_exceeded` → `abort:agent_time_budget_exceeded`）**存在 category 差异**——需在 B.8 统一。

## B.8 「同一现象两端 category 不一致」的统一方案

| 来源 | 现状 | 统一后 |
| --- | --- | --- |
| artifacts summary（有 `abort_reason="agent_time_budget_exceeded"`） | `("abort", "agent_time_budget_exceeded")`（规则 1 最先命中） | **保持 `abort`**（不改既有语义与页面） |
| `.pt`（`agent_outcome="time_budget_exceeded"`，无 abort_reason） | 修复前 `other` / `truncated` 时 `timeout` | **`("timeout", "agent_time_budget_exceeded")`** |
| DB `ri_attempt_table`（只有 `status`） | `status=aborted` → `abort`；`status=truncated` → `timeout` | 不变 |

**结论：不追求两端 category 相同**，而是**用 `cause_axis` 拉齐**：三种来源在 `cause_axis` 层面**统一为 `agent_budget`**，reason 码统一为 `agent_time_budget_exceeded`。页面按 `cause_axis` 汇总时，三者合一；按 category 展示时保持既有语义不变。**这是「不改 category」路线的必然代价，也是它的收益：零破坏。**

## B.9 函数签名增量（`utils/failure_reason_taxonomy.py`）

```python
__all__ = [
    # 既有（不变）
    "CATEGORY_LABELS", "AGENT_OUTCOME_LABELS", "AGENT_OUTCOME_REASON_CODES",
    "normalize_reason", "classify_from_fields", "reason_label",
    "agent_outcome_label", "agent_outcome_reason", "is_agent_error_outcome",
    "split_reason_prefix",
    # 新增
    "CAUSE_AXIS_LABELS", "REASON_LABELS", "EVALUATOR_FAULT_REASONS",
    "classify_with_axis", "cause_axis_of", "cause_axis_label",
    "reason_display", "is_evaluator_fault", "is_policy_failure",
    "normalize_attempt_row",   # 不放这里 → 属于 agent_budget_profile
]

def cause_axis_of(
    *,
    status: Any = None,
    diagnosis: Any = None,
    abort_reason: Any = None,
    error_code: Any = None,
    agent_outcome: Any = None,
) -> str:
    """按 B.3.2 的优先级表判定归因轴（agent_budget / context_limit / infra / agent_turns / evaluator / agent_policy / unknown）。"""

def classify_with_axis(fields: Dict[str, Any]) -> Dict[str, str]:
    """在既有 (category, reason) 之上补 cause_axis / cause_label。返回 4 键 dict。"""

def cause_axis_label(cause_axis: Any) -> str:
    """归因轴 → 中文标签；未知值原样返回。"""

def reason_display(category: str, reason: str) -> str:
    """带具体原因码的展示标签：REASON_LABELS 命中 → 中文；否则回落 reason_label。"""

def is_evaluator_fault(reason: Any) -> bool:
    """是否为评测器自身故障（非法载荷 / 执行失败 / 评测超时）。evaluator:tests_failed 不算。"""

def is_policy_failure(category: str, reason: str) -> bool:
    """是否属于「策略没做对」（可学习失败）；评测器故障与基建问题不算。"""
```

## B.10 与现有代码的复用关系

| 关系 | 说明 |
| --- | --- |
| **不新建模块** | 全部在 `utils/failure_reason_taxonomy.py` 内扩展（计划 §5「就地扩展」） |
| **复用** `normalize_reason` | `classify_with_axis` 内部调用它取 `(category, reason)`，**不重写分类** |
| **复用** `classify_from_fields` | 同上 |
| **复用** `agent_outcome_reason` / `AGENT_OUTCOME_REASON_CODES` | 新增别名项进同一张表 |
| **复用** `utils.rollout_pt_parser.classify_sample_outcome` | `.pt` 侧同步扩展（B.7），**不新建第二套 `.pt` 分类** |
| **复用** `service.reward_diagnosis_service.summarize_reward_split` | 只**追加** `n_evaluator_fault` / `evaluator_fault_ratio` / `policy_failure_rate` 字段 |
| **消费方零改动** | 见 B.6 总表 |
| **新 `func.md` 登记** | `scan_func_md.py` 会自动登记新增的模块级公开函数与模块级公开常量（`scan_data` 只扫 `data/`，`scan_utils` 只收函数 → **`CAUSE_AXIS_LABELS` 等常量不会进 func.md**，属已知扫描口径，不影响 `--check`） |

## B.11 风险与边界

| 风险 | 边界 |
| --- | --- |
| `other → timeout` 是**行为变更** | 已论证必要性（B.3.5）；需同步改 1 个既有测试 + CHANGELOG；影响面限于「本该不可测却被当可测」的样本 |
| 两端 category 不一致（artifacts=`abort` vs `.pt`=`timeout`） | 不强求统一，用 `cause_axis` 拉齐（B.8）；页面按 axis 汇总时合一 |
| `cause_axis` 不参与既有分组 | 页面若要按 axis 展示，必须**显式**按 axis 聚合；不得改 `classify_group_kind` 的跳过集合 |
| `evaluator_fault` 的 reason 白名单可能不全 | 白名单可扩展；未命中的一律按策略失败处理（保守） |
| `REASON_LABELS` 文案变更 | `reason_display` 是**新函数**，`reason_label` 行为不变 → 零破坏 |
| `context_limit_tokens` 与 `agent_outcome` 冲突（同时存在且互相矛盾） | 优先级表已定：**预算 > 上下文 > 基建 > 轮次 > 评测器 > 策略**；并在校验里记 `note` |

---

# 交付物 C：`utils/difficulty_profile.py` 设计规格

## C.1 目标与依据

**目标**：用「跨全部 rollout 的累计通过率」把题目池分成 `stable_pass / gradient / stable_fail` 三态，**替代 v1 失效的「末次通过率」口径**，并补上 v1 完全没有的**采样密度**维度。

**依据（简报 §3.3，v1 口径失效的直接证据）**：task4 有 **688 题、每题平均只出现 1.5 次、出现间隔 31.1 step** → 「末次通过率 ≈ 首次通过率」，**无法反映学习进展**；v1 的「≥0.875 判饱和、≤0.125 判太难」在此**失效**。

**依据（简报 §3，采样密度实测表）**：每步固定 **16 组**；task9（32 题）50%、task8/task11（64 题）25%、task4（688 题）2.3%、task1（96 题）16.7%、task2（338 题）4.7%、task3（640 题）2.5%、task6/7/10（128 题）12.5%。

**依据（简报 §3.1，全局组类型分布）**：task9 mixed **77.6%** / all_fail **1.8%**；task4 mixed 48.1% / all_fail 9.4%；task8 mixed 50.9% / all_fail 14.7%。

## C.2 三态分类规则

### C.2.1 分类量：`cumulative_pass_rate`

```
对题目 i：
  cumulative_pass_rate(i) = Σ_所有 visit 的 n_pass / Σ_所有 visit 的 n_attempts
```

- **跨全部 rollout 聚合**（不是「末次」，也不是「最好一次」）。
- 分母口径 = **strict**（含 abort/timeout），与 `problem_analysis_service.summarize_pass_visits` 的既有页面口径**一致**（其 `_PAGE_NOTE` 明说「abort 与 timeout 计入分母、不计入分子」）→ **跨页面对账不会打架**。
- 额外输出 `gradable_pass_rate`（去 abort/timeout 的分子分母），由上游 `summarize_pass_visits` 追加的 `n_measurable` 字段提供（C.4）。

### C.2.2 阈值

| state | 条件 | 依据 |
| --- | --- | --- |
| `stable_pass` | `cumulative_pass_rate ≥ 0.90` | 计划 §4.1 阈值；相对 v1 的 0.875 上调，取 0.90 使「≥7/8 持续通过」不再算稳定掌握（见 §0.3-3） |
| `gradient` | `0.10 < cumulative_pass_rate < 0.90` | 计划 §4.1；这是 GRPO 唯一有价值的区段 |
| `stable_fail` | `cumulative_pass_rate ≤ 0.10` | 计划 §4.1 |

**阈值再校准（可解释性）**：GRPO 组内 8 attempts，单次出现的通过率量化粒度 = 1/8 = 0.125。
- `stable_pass` 阈值 0.90 → 单次出现需 **8/8**（1.0）才算；7/8=0.875 落 `gradient`。→ **单次出现的题必须带置信度标记**（C.5）。
- `stable_fail` 阈值 0.10 → 单次出现需 **0/8**（0.0）才算；1/8=0.125 落 `gradient`。→ 同上。

**结论**：三态阈值在**样本充足**时是稳的；在**样本稀少**时必须靠 C.5 的置信度兜底，否则 task4（每题 1.5 次）会产出大量伪三态结论。

## C.3 输出 Schema

```python
{
  "n_problems": int,                 # 题目总数
  "n_rollouts": int,
  "groups_per_step": int|None,       # 每步组数（= 每 rollout 的 group 数众数）
  "pool_size": int,                  # 题目池大小（= n_problems）

  "states": {
    "stable_pass": {"count": int, "ratio": float, "n_low_confidence": int},
    "gradient":    {"count": int, "ratio": float, "n_low_confidence": int},
    "stable_fail": {"count": int, "ratio": float, "n_low_confidence": int},
    "unknown":     {"count": int, "ratio": float},   # 无 visit / 无有效 attempt
  },

  "problems": [ProblemEntry, ...],   # 按 (state 优先级, pass_rate 序) 排序

  "density": {
    "sampled_density": float|None,   # min(1, groups_per_step / n_problems)  ← 简报表的值
    "mixed_ratio": float|None,       # 实测 mixed 组占比（来自 classify_group_kind 聚合）
    "all_pass_ratio": float|None,
    "all_fail_ratio": float|None,
    "infra_only_ratio": float|None,
    "groups_per_step": int|None,
    "pool_size": int,
    "expected_coverage_per_epoch": float|None,   # pool_size / groups_per_step
    "note": str,
  },

  "resample": {
    "remove":     [ResampleItem, ...],
    "downsample": [ResampleItem, ...],
    "keep_boost": [ResampleItem, ...],
    "counts": {"remove": int, "downsample": int, "keep_boost": int, "unknown": int},
    "policy": {"stable_pass_threshold": 0.9, "stable_fail_threshold": 0.1,
               "min_appearances_for_action": 3, "group_size": 8},
  },

  "data_quality": {
    "min_appearances": int,
    "n_problems_below_min": int,
    "coverage": float|None,          # 有 visit 的题目占比
    "notes": [str, ...],
  },

  "note": str,
}
```

### C.3.1 `ProblemEntry`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `instance_id` | `str` | 题目标识 |
| `state` | `str` | `stable_pass / gradient / stable_fail / unknown` |
| `cumulative_pass_rate` | `float\|None` | 累计通过率（strict） |
| `gradable_pass_rate` | `float\|None` | 累计通过率（去 abort/timeout） |
| `n_appearances` | `int` | 出现次数（= visit 数） |
| `n_attempts_total` | `int` | 累计 attempt 数 |
| `n_pass_total` | `int` | 累计通过数 |
| `first_rollout_id` | `int\|None` | 首次出现的 rollout |
| `last_rollout_id` | `int\|None` | 末次出现的 rollout（**排序锚点**） |
| `pass_rate_trend` | `str` | `improved / worsened / unchanged / once`（**复用** `summarize_pass_visits` 的 `trend`） |
| `pass_rate_delta` | `float\|None` | 末次 − 首次（**复用** `delta`） |
| `pass_rate_slope` | `float\|None` | 出现序号 → 通过率的最小二乘斜率（新增，度量「是否在学」） |
| `best_visit_rate` | `float\|None` | 单次出现的最高通过率（**识别「差一点」的关键**，见 C.6） |
| `best_visit_rollout_id` | `int\|None` | 最高通过率那次出现的 rollout |
| `confidence` | `str` | `high / medium / low` |
| `data_insufficient` | `bool` | 样本太少，判定不可靠 |
| `near_boundary` | `bool` | 累计通过率离 0.1 / 0.9 分界不足 1 个量化粒度 |
| `mixed_prob` | `float\|None` | 该题产出 mixed 组（既有对又有挂）的概率估计 `1 − p^g − (1−p)^g` |

### C.3.2 `ResampleItem`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `instance_id` | `str` | |
| `pass_rate` | `float\|None` | 累计通过率 |
| `best_visit_rate` | `float\|None` | 最高单次通过率 |
| `n_appearances` | `int` | 样本量 |
| `last_rollout_id` | `int\|None` | |
| `confidence` | `str` | |
| `mixed_prob` | `float\|None` | 该题的「产出 mixed 组」概率 |
| `reason` | `str` | 一句话理由（供 UI 直接显示） |

## C.4 与 `service/problem_analysis_service.summarize_pass_visits` 的复用关系（**核心设计决策**）

### C.4.1 结论：**扩展上游 + 新建纯函数模块，不新建平行实现**

`summarize_pass_visits` 已经产出本模块所需的 **90% 原始素材**：

```python
# summarize_pass_visits 返回的 tasks[i]
{
  "instance_id": str,
  "visits": [{"rollout_id": int, "n_pass": int, "n_attempts": int, "rate": float}, ...],  # 按 rollout_id 升序
  "first_rate": float,
  "last_rate": float,
  "delta": float|None,
  "trend": "improved" | "worsened" | "unchanged" | "once",
}
```

→ **累计通过率 = `Σ visit.n_pass / Σ visit.n_attempts`**，**不需要重新分桶**。
→ **`pass_rate_trend` = 上游 `trend`**，**`pass_rate_delta` = 上游 `delta`** —— **完全复用，不重算**。
→ `last_rollout_id` = 最后一个 visit 的 `rollout_id`。

### C.4.2 复用方式：**上游做加法（additive），下游做聚合**

**对 `service/problem_analysis_service.summarize_pass_visits` 的唯一改动（P2 扩展）**：在每个 visit 上**追加** `n_measurable`（去 abort/timeout 的 attempt 数）。

```python
# 追加到每个 visit（既有键全部保留，既有断言全部不受影响）
visit["n_measurable"] = n_measurable          # 新增
visit["measurable_rate"] = n_pass / n_measurable if n_measurable else None   # 新增
```

**为什么必须加在这里**：若 `difficulty_profile` 自己重新按 `(rollout_id, instance_id)` 分桶算 `n_measurable`，就是**同一张表的第二次分桶** —— 违反「不新建平行实现」，且一旦 `summarize_pass_visits` 改了分桶规则，两处会静默漂移。

**与计划 §2.3（三口径子任务）的接口约定**：
- §2.3 的 owner 若已在 `summarize_pass_visits` 内实现三口径，则 `n_measurable` 由其提供，本模块直接读；
- 若 §2.3 未加 `n_measurable`，本模块**降级为只用 strict 口径**（`cumulative_pass_rate`），`gradable_pass_rate` 输出 `None` 并在 `data_quality.notes` 标注「未接入可测口径」——**不自己重复分桶**。
- **本模块不负责三口径命名**，避免与 §2.3 冲突。

### C.4.3 `groups_per_step` 的取数（**复用既有能力**）

`groups_per_step` = 每 rollout 的 distinct `group_index` 数（简报实测恒为 16）。

- 既有实现：`service/problem_analysis_service._mode_batch_size(groups_per_rollout)` —— 取各 rollout 组数的**众数**，并区分来源 `"group_mode"` / `"env"`。
- **该函数是私有的（`_` 前缀）**，跨模块 import 私有函数不符合项目约定。
- **方案（P2 扩展，不新建平行实现）**：在 `service/problem_analysis_service` 中把 `_mode_batch_size` **提升为公开函数** `mode_groups_per_rollout(groups_per_rollout) -> Tuple[Optional[int], str]`，保留 `_mode_batch_size` 作为**别名**（`def _mode_batch_size(...): return mode_groups_per_rollout(...)`）以兼容既有调用与测试。
- `difficulty_profile` 通过 `from service.problem_analysis_service import mode_groups_per_rollout` 复用。
- **⚠️ 分层提醒**：`utils/` 不应 import `service/`（utils 是最底层）。→ 因此 `difficulty_profile` **不应自己算 groups_per_step**；应由调用方（`service/reward_diagnosis_service`）把 `groups_per_step` **作为参数传入**：

```python
def build_difficulty_profile(
    rows: Sequence[Dict[str, Any]],
    *,
    groups_per_step: Optional[int] = None,
    group_size: int = 8,
    ...
) -> Dict[str, Any]:
```

`groups_per_step is None` 时 → `density.sampled_density = None`，并在 `data_quality.notes` 注明「未提供每步组数，采样密度不可算」。**调用方**（service 层）用 `mode_groups_per_rollout` / `summarize_pass_visits` 的 `batch_size` 传入。**这条边界必须写死，否则会形成 utils → service 的反向依赖。**

> **同时避免 `env ROLLOUT_BATCH_SIZE` 的坑**：`problem_analysis_service._env_batch_size` 读的是 env 的 `ROLLOUT_BATCH_SIZE`，它可能是**样本数**（16 组 × 8 = 128）而不是**组数**（16）。→ 调用方传入 `groups_per_step` 时**必须优先用 `group_mode`（实测众数）**，只有当 env 值与实测众数一致时才采用 env 值；不一致时以实测为准并在 notes 注明。**这是实测口径，不是拍脑袋。**

## C.5 采样密度 `gradient_density` 的计算口径与阈值

### C.5.1 两个指标必须分开（见 §0.3-2）

| 字段 | 定义 | 与简报的关系 |
| --- | --- | --- |
| `sampled_density` | `min(1.0, groups_per_step / n_problems)` | **精确复现**简报 §3 表的「梯度密度」列（16/32=50%、16/64=25%、16/688=2.3%、16/96=16.7%、16/338=4.7%、16/640=2.5%、16/128=12.5% 全部对上） |
| `mixed_ratio` | 实测：mixed 组数 / 总组数 | 简报 §3.1 的 mixed 占比（task9 77.6%、task4 48.1%） |

**命名建议**：计划 §4.1 的杠杆 key 叫 `gradient_density`。为避免与简报表格口径混淆，**建议**：
- 底层字段名用 `sampled_density`（理论覆盖）与 `mixed_ratio`（实测）；
- 杠杆 key 保留 `gradient_density`（对外契约不变），其**取值口径在杠杆文案里写死**为 `sampled_density`，并在 `note` 里注明「理论覆盖密度，非实测 mixed 比例」。

### C.5.2 密度的物理含义（**这是本方案最有用的一句解释**）

每步采样 `groups_per_step` 个组，每组对应**一道题**（GRPO 同题多次采样）→ **单步能覆盖的题目数 = groups_per_step = 16**。
→ `sampled_density = 16 / 题目池大小` = **单步能刷到的题目池比例**。

- task9（32 题）：单步覆盖 **50%** 的题库 → 4 步刷完全库 → 每题平均出现 24 次 → 反馈快、梯度密度高。
- task4（688 题）：单步覆盖 **2.3%** → **43 步**才刷完一遍题库 → 每题平均 1.5 次 → 「末次通过率 ≈ 首次通过率」→ v1 口径失效的直接原因。

**这两句话是同一个数的两面** —— 这是本次修订最有解释力的一处。

### C.5.3 阈值（**基于简报实测点**）

简报实测的密度点与后果：

| 任务 | 密度 | 后果（简报 §1 横向对比） |
| --- | --- | --- |
| task9 | **50%** | 唯一成功：`removed_ratio ≤ 0.16`、`trainable_attempt_count` 稳定 100+、mixed 77.6%、all_fail 1.8% |
| task8 / task11 | **25%** | `removed_ratio` 0.63 / 0.88，`trainable_attempt_count` 78→47 / 95→15 |
| task4 | **2.3%** | `removed_ratio` → 1.00，`trainable_attempt_count` → 0 |
| task1 | 16.7% | — |
| task6/7/10 | 12.5% | — |

| 档 | `sampled_density` | 依据 |
| --- | --- | --- |
| `bad` | `< 0.05` | task4 = **2.3%**（`removed_ratio` 冲到 1.00 的唯一案例）→ 2.3% 落在 bad 档 |
| `warn` | `0.05 ≤ d < 0.15` | task1 = 16.7%、task6/7/10 = 12.5% → 均在 warn 档 |
| `ok` | `≥ 0.15` | task9 = **50%**（唯一成功）、task8/11 = **25%**（虽失败但 removed 未冲顶） |

> **阈值的诚实说明**：样本只有 3 个（2.3% / 25% / 50%），`0.05` 与 `0.15` 是**在这 3 个实测点之间取的分档边界**（0.05 介于 2.3% 与 12.5% 之间；0.15 介于 12.5% 与 25% 之间），属于**带实证锚点的设计假设**，不是统计拟合。**必须在代码注释与页面文案里写清这一点**，避免后续被当成普适阈值。
>
> **另外**：`ok` 档定在 0.15 而非 0.25，是因为 task8/task11（25%）**实测是失败任务**（removed_ratio 0.63/0.88）→ 25% 不能算「好」。**唯一成功的是 50%。** 因此更准确的表述是：`≥ 0.15` 表示「覆盖密度足够，**不是充分条件**」。

### C.5.4 `mixed_ratio` 的阈值（用于交叉验证）

简报 §3.1 的 mixed 占比：task9 **77.6%**（成功）、task11 58.0%、task10 57.4%、task8 50.9%、task7 54.9%、task4 48.1%、task3 30.8%、task2 41.9%。

| 档 | `mixed_ratio` | 依据 |
| --- | --- | --- |
| `bad` | `< 0.30` | task3 = 30.8% 附近；无低于 30% 的实测点 → **此档为设计假设** |
| `warn` | `0.30 ≤ m < 0.60` | task4 48.1%、task8 50.9%、task2 41.9% |
| `ok` | `≥ 0.60` | task9 = **77.6%**（唯一成功）、task11 58.0%、task10 57.4% 恰在边界 |

### C.5.5 **密度与 mixed 比例非单调（必须写进说明文案）**

| 任务 | 密度 | mixed 比例 |
| --- | --- | --- |
| task9 | 50%（最高） | 77.6%（最高） |
| task11 | 25% | 58.0% |
| task8 | 25% | 50.9% |
| **task4** | **2.3%（最低）** | **48.1%（高于 task8）** |
| task3 | 2.5% | 30.8% |

→ **task4 的密度比 task3 还低，但 mixed 比例反而更高。** 说明「密度」不是 mixed 比例的唯一决定因素（难度分布、题目质量同样决定）。
→ **文案纪律**：杠杆文案只能说「**密度过低会拉低可获得的梯度多样性**」，**不能说「密度低 → mixed 少 → 奖励不涨」**（后者在 task4 上被证伪）。真正的因果链是：**密度低 → 单题出现次数少 → 通过率估计噪声大 → 学习信号被稀释 + 题目池大部分从未被有效采样**。

## C.6 重采样清单的输出结构与排序

### C.6.1 三张清单的语义与排序依据

| 清单 | 成员 | 排序 | 依据 |
| --- | --- | --- | --- |
| `remove` | `stable_pass`（累计通过率 ≥ 0.9） | **通过率降序**（最饱和的先移出） | 已掌握 → 无梯度；`mixed_prob` 最低（p→1 时 `1−p^8−(1−p)^8 → 0`） |
| `downsample` | `stable_fail`（累计通过率 ≤ 0.1） | **best_visit_rate 降序**（最有希望的先保留一点权重） | **修正计划 §4.3 的矛盾**（见 §0.3-4）：`stable_fail` 的累计率 ≤ 0.1，不可能「最接近 0.2–0.5」；真正的「差一点」信号在**单次出现**里 → 用 `best_visit_rate` 排序 |
| `keep_boost` | `gradient`（0.1 < 率 < 0.9） | **\|通过率 − 0.5\| 升序**（最接近 0.5 的最优先） | GRPO 组内 8 次采样，`P(mixed) = 1 − p^8 − (1−p)^8`，**在 p=0.5 处取最大值 0.992**；因此「通过率 ≈ 0.5 的题」产出 mixed 组概率最高，加权采样收益最大 |

**`mixed_prob` 的推导**（可直接写进 docstring）：

```
P(组内既有通过又有失败) = 1 − p^g − (1−p)^g
  p = 该题的通过率估计（用累计通过率，或 best_visit_rate）
  g = 组大小（默认 8，来自 GRPO 配置）
p=0.5 → 1 − 0.0039 − 0.0039 = 0.992
p=0.9 → 1 − 0.4305 − 1e-8 = 0.569
p=0.1 → 1 − 1e-8 − 0.4305 = 0.569
p=1.0 → 0
```
→ 这解释了为什么 `remove` 清单要挑通过率最高的（mixed_prob → 0），而 `keep_boost` 要挑 0.5 附近的。

### C.6.2 `reason` 文案模板

| 清单 | reason 模板 |
| --- | --- |
| `remove` | `"累计通过率 {rate:.0%}（{n} 次出现，末次 rollout {rid}）→ 已掌握，移出训练集"` |
| `downsample` | `"累计通过率 {rate:.0%}，最好一次 {best:.0%} → 基本学不会，降采样；若有 {best:.0%} 的好成绩可优先补中间难度题"` |
| `keep_boost` | `"累计通过率 {rate:.0%}（距 0.5 {dist:.2f}，mixed 概率 {mp:.0%}）→ 有真梯度，优先加权采样"` |

### C.6.3 数据不足的处理（**硬要求**）

- `n_appearances < min_appearances_for_action`（默认 **3**）→ 该题**不进入**任何清单的**动作区**，而是进 `problems` 并标 `data_insufficient=True` / `confidence="low"`。
- **依据**：简报 §3.3 的 task4 每题平均出现 **1.5** 次 → 若不设下限，task4 的重采样清单几乎全是「1 次出现的题」的结论，**完全不可信**。设 3 次 → task9（24 次）与 task8（16 次）不受影响，task4（1.5 次）**全部标为数据不足** → 这正是我们要的效果（task4 的正确结论是「**该做的是缩小题目池 / 提高每步组数，而不是重排这 688 道题**」）。
- **阈值 3 的推导**：累计通过率的量化粒度 = `1 / (n_appearances × group_size)` = `1/24 ≈ 0.042`（3 次 × 8 attempts），**小于 0.1 档位宽度的一半**（0.05）→ 3 次是「能可靠区分 0.1 / 0.9 档」的最小样本量。**设计假设，依据量化粒度推导。**

### C.6.4 `confidence` 判定

| `n_appearances` | `confidence` | 附加条件 |
| --- | --- | --- |
| ≥ 4 | `high` | 且 `n_attempts_total ≥ 16` |
| 2–3 | `medium` | |
| 1 | `low` | `data_insufficient=True` |
| 0 | — | `state="unknown"` |

**`near_boundary` 判定**：`min(|rate − 0.1|, |rate − 0.9|) < 1 / (2 × n_attempts_total)` → 该题离分界不足半个量化粒度 → `near_boundary=True`（判定方向不可靠）。**这是 Wilson 区间思想的简化版**，保证「0.9 恰好卡在 7/8」的题不会被误当作稳定掌握。

## C.7 函数签名（`utils/difficulty_profile.py`）

```python
# utils/difficulty_profile.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "DIFFICULTY_STATES",
    "STABLE_PASS_THRESHOLD",
    "STABLE_FAIL_THRESHOLD",
    "MIN_APPEARANCES_FOR_ACTION",
    "build_difficulty_profile",
    "classify_difficulty_state",
    "compute_pass_rate_trend",
    "compute_mixed_probability",
    "build_resample_lists",
    "compute_sampled_density",
]

DIFFICULTY_STATES: tuple[str, ...] = ("stable_pass", "gradient", "stable_fail", "unknown")
STABLE_PASS_THRESHOLD: float = 0.90
STABLE_FAIL_THRESHOLD: float = 0.10
MIN_APPEARANCES_FOR_ACTION: int = 3

def build_difficulty_profile(
    rows: Sequence[Dict[str, Any]],
    *,
    groups_per_step: Optional[int] = None,
    group_size: int = 8,
    stable_pass_threshold: float = STABLE_PASS_THRESHOLD,
    stable_fail_threshold: float = STABLE_FAIL_THRESHOLD,
    min_appearances: int = MIN_APPEARANCES_FOR_ACTION,
    mixed_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """题目池三态 + 采样密度 + 重采样清单的主入口。

    ``rows`` 每项需含 ``rollout_id`` / ``instance_id`` / ``category``
    （与 ``problem_analysis_service.summarize_pass_visits`` 的输入契约一致）。
    本函数**不直接访问 DB / 文件**，分组口径与 ``summarize_pass_visits`` 对齐。
    """

def classify_difficulty_state(
    pass_rate: Optional[float],
    *,
    stable_pass_threshold: float = STABLE_PASS_THRESHOLD,
    stable_fail_threshold: float = STABLE_FAIL_THRESHOLD,
) -> str:
    """累计通过率 → stable_pass / gradient / stable_fail / unknown。"""

def compute_pass_rate_trend(visits: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """出现次序 → 趋势（复用上游 trend/delta，另给最小二乘斜率 + best visit）。"""

def compute_mixed_probability(pass_rate: Optional[float], group_size: int = 8) -> Optional[float]:
    """P(组内有对有挂) = 1 − p^g − (1−p)^g。"""

def build_resample_lists(
    problems: Sequence[Dict[str, Any]],
    *,
    group_size: int = 8,
    min_appearances: int = MIN_APPEARANCES_FOR_ACTION,
) -> Dict[str, Any]:
    """remove / downsample / keep_boost 三张清单（含排序与 reason 文案）。"""

def compute_sampled_density(
    n_problems: int, groups_per_step: Optional[int]
) -> Optional[float]:
    """min(1, groups_per_step / n_problems)；输入不足返回 None。"""
```

## C.8 关键实现要点

1. **分组口径**：`rows` → 按 `(rollout_id, instance_id)` 分桶 → 每桶算 `n_pass`（`category == "pass"`）/ `n_attempts` / `n_measurable`；**与 `summarize_pass_visits` 完全同口径**（同 key、同分子、同分母）。
2. **空 `instance_id`**：跳过并计入 `n_attempts_without_instance`（对齐上游 `summarize_pass_visits` 的同名字段）。
3. **`reward` vs `category`**：分子用 `category == "pass"`（**不是** `reward >= 1.0`），与上游一致 → 避免 abort 的 `reward=0.0` 与「不可测」混淆。
4. **排序**：`problems` 按 `(state 优先级[gradient > stable_fail > stable_pass > unknown], -pass_rate, instance_id)`；三张清单各自按 C.6.1 的规则。
5. **`groups_per_step` 缺省**：`density.sampled_density = None` + notes 提示；**不自造默认值 16**（16 是 task8/9/11 的实测值，不是普适常量；简报明确「全部任务实测 groups_per_step = 16.0」，但这是**本批实测**，其它任务可能不同 → 作为**可配置默认值 16** 且在 notes 标注「采用实测默认 16」）。
6. **`mixed_ratio`**：由调用方传入（`service/reward_diagnosis_service` 用既有 `classify_group_kind` + `summarize_gradient_groups` 算出）→ **复用既有能力，不重算组语义**。

## C.9 与现有代码的复用关系（**不新建平行实现**）

| 关系 | 说明 |
| --- | --- |
| **复用** `service.problem_analysis_service.summarize_pass_visits` | 产出每题 visits（`n_pass/n_attempts/rate`）与 `trend/delta`；**C 只做跨 visit 的累加与分类** |
| **扩展** `summarize_pass_visits`（加法） | 追加 `n_measurable` / `measurable_rate`（C.4.2）；既有键与既有断言不变 |
| **提升可见性** `service.problem_analysis_service._mode_batch_size` → `mode_groups_per_rollout` | 公开化 + 保留 `_mode_batch_size` 别名（零破坏）；**调用方（service 层）**用它取 `groups_per_step` 并传入，**utils 不反向 import service** |
| **复用** `service.reward_diagnosis_service.classify_group_kind` | `mixed_ratio` 由调用方用既有组语义算出后传入 |
| **不修改** `service.reward_diagnosis_service.summarize_pool_health` | **它已被简报 §3.3 证伪**（用「最好一次通过率」判饱和/太难）→ **新增 `difficulty_profile` 取代它**，`summarize_pool_health` 标记为 **deprecated**（保留函数体不动 + 加一行 docstring 注明「口径已被 `utils/difficulty_profile` 取代，见 PLAN v2 §4」），避免破坏 `tests/test_reward_levers.py` 的 8 条既有断言 |
| **杠杆层** | 在 `service/reward_diagnosis_service._LEVER_RULES` **追加** 3 条：`gradient_density` / `n_gradient_problems` / `pool_coverage_per_epoch`；**既有 7 条不动**（`tests/test_reward_levers.py::test_always_returns_all_levers` 断言的是**集合相等**，会失败 → **必须同步更新该断言**，见 D.3） |
| **新 `func.md` 登记** | `utils/difficulty_profile.py` 的 7 个公开函数自动登记 |

## C.10 风险与边界

| 风险 | 边界与对策 |
| --- | --- |
| **样本少导致伪三态**（task4 每题 1.5 次） | `data_insufficient` + `confidence` + `near_boundary` + `min_appearances_for_action=3`；task4 的正确结论是「缩池 / 提组数」，不是「重排 688 题」 |
| **阈值 0.9/0.1 与量化粒度 1/8 冲突** | 已在 C.2.2 说明；`near_boundary` 兜底 |
| **`sampled_density` 是理论值，可能与实际 mixed 比例脱节** | 同时输出 `mixed_ratio` 实测值；文案明确「非充分条件」；§0.3-2 已记录非单调证据 |
| **分层：utils 不能 import service** | `groups_per_step` / `mixed_ratio` **由调用方传入**（C.4.3 末段） |
| **`summarize_pool_health` 的兼容** | 不删、不改行为，仅加 deprecated 说明；新口径走 `difficulty_profile` |
| **题目池定义** | `n_problems` = 本任务**观测到的** distinct `instance_id` 数，**不等于**配置里的题库大小（可能有些题一次没被抽到）→ `data_quality.notes` 必须注明「观测到的题目数，可能小于配置题库」 |
| **不做因果证明** | 密度 → 奖励的关系是相关性；文案用「与…相关」「会稀释」 |

---

# 交付物 D：测试用例清单

## D.0 仓库现有测试约定（**已核对 `tests/` 目录**）

| 约定 | 实测 |
| --- | --- |
| 风格 | `unittest`（`class TestXxx(unittest.TestCase)` + `assertEqual/assertAlmostEqual/assertTrue/assertIn`） |
| 文件头 | `"""中文模块说明。"""` + `from __future__ import annotations` + `import unittest` |
| 导入 | `from utils.xxx import fn` / `from service.xxx import fn`（绝对导入，非相对） |
| 构造数据 | 模块级 helper 函数（如 `_row(...)` / `_rows(...)`）集中造行 |
| 结尾 | `if __name__ == "__main__": unittest.main()` |
| 命名 | `tests/test_<被测模块>.py`，类名 `Test<被测函数组>`，方法名 `test_<行为>`（**描述行为，不描述实现**） |
| 已有同类 | `tests/test_failure_reason_taxonomy.py`（B）、`tests/test_reward_levers.py`（C 的杠杆层）、`tests/test_reward_diagnosis_service.py`（C 的 service 层）、`tests/test_problem_analysis_service.py`（C 的上游） |
| 空输入 | 每个模块都要有「空输入不崩」用例（既有 `test_empty_returns_none_ratios` / `test_empty_dump` 风格） |

> **本清单只列用例设计，不写实现**；实现时按上表约定落成 `tests/test_*.py`。

---

## D.1 `tests/test_agent_budget_profile.py`（新文件，对应交付物 A）

> 复用 `utils/rollout_pt_parser.classify_sample_outcome` 的口径：构造行时直接给 `category` 字段（A 的规范行含 `category`），避免在测试里重复实现分类。

**构造数据 helper（设计）**：

```python
def _row(rid, *, agent_outcome="finished", agent_run=1200.0, attempt_total=1300.0,
         model_generation=900.0, turns=40, gen_tokens=12000, peak_ctx=60000,
         ctx_limit=131072, status="completed", category="pass", **phase) -> dict
```

### D.1.1 `TestNormalizeAttemptRow`

| # | 测试函数名 | 验证点 | 输入数据 |
| --- | --- | --- | --- |
| 1 | `test_flat_phase_time_keys` | 扁平 `metadata["phase_time/agent_run"]` 能被识别 | `{"phase_time/agent_run": 3614.0, "phase_time/attempt_total": 4200.0}` → `agent_run_s == 3614.0` |
| 2 | `test_nested_phase_time_dict` | 嵌套 `phase_time: {"agent_run": ...}` 能被识别 | `{"phase_time": {"agent_run": 100.0, "attempt_total": 120.0}}` → `agent_run_s == 100.0` |
| 3 | `test_artifacts_phase_times_dict` | artifacts `phase_times: {"phase_time/agent_run": ...}` 能被识别 | → `agent_run_s` 正确 |
| 4 | `test_missing_fields_are_none` | 缺失不报 KeyError，一律 `None` | `{}` → 全部 `None` |
| 5 | `test_turns_fallback_order` | `agent_turns` 优先于 `attempt_completed_turns` | 同时给两个 → 取 `agent_turns` |
| 6 | `test_nested_takes_priority_over_flat` | 扁平与嵌套同时给 → **嵌套优先**（嵌套是 `.pt` 原生形态） | 需在实现里固定优先级，并在测试里钉死 |

### D.1.2 `TestBuildBudgetProfileSeries`

| # | 测试函数名 | 验证点 | 输入数据 |
| --- | --- | --- | --- |
| 7 | `test_empty_rows_returns_empty_profile` | 空输入不崩，`series == []`、`verdict == "healthy"`、`confidence == "low"` | `[]` |
| 8 | `test_series_ordered_by_rollout_id` | series 按 rollout_id 升序 | 造 rollout 3 / 1 / 2 的行 |
| 9 | `test_rollout_counts_and_ratios` | `n_attempts` / `n_time_budget` / `time_budget_ratio` 正确 | 1 个 rollout、8 行，其中 3 行 `agent_outcome="time_budget_exceeded"` → `time_budget_ratio == 3/8` |
| 10 | `test_percentiles_are_interpolated` | p50 用线性插值（n=2 时取均值） | 2 行 `agent_run_s = [10.0, 20.0]` → `agent_run_p50 == 15.0` |
| 11 | `test_low_confidence_when_samples_below_min` | `n_valid < min_samples` → `low_confidence=True` | `min_samples=8`，给 3 行有效 |
| 12 | `test_missing_turns_excluded_from_per_turn_but_counted` | 缺 `turns` 的行不参与 `per_turn_seconds` 分位，但计入 `n_attempts` | 8 行中 3 行 `turns=None` |
| 13 | `test_model_generation_fallback_marks_source` | `model_generation_s` 缺失时退回 `agent_run/turns`，`per_turn_source == "agent_run"` | 行缺 `model_generation_s` |
| 14 | `test_budget_hit_ratio_uses_098_of_budget` | `agent_run >= 3600×0.98 = 3528` 记为 hit | 给 4 行 `agent_run = 3610`（全 hit）+ 4 行 `agent_run = 100`（不 hit）→ `budget_hit_ratio == 0.5` |

### D.1.3 `TestQueueStolenMetrics`

| # | 测试函数名 | 验证点 | 输入数据 |
| --- | --- | --- | --- |
| 15 | `test_overhead_and_infra_overhead_ratio` | `overhead = attempt_total − agent_run`；`infra_overhead_ratio = overhead / attempt_total` | `agent_run=3600, attempt_total=4700` → `overhead=1100`，`infra_overhead_ratio ≈ 0.2340` |
| 16 | `test_sandbox_boot_queue_reads_correct_field` | **回归护栏**：读 `phase_time/sandbox_boot_queue`，**不是** `phase_time/sandbox_boot`（两者差一个数量级，曾导致两次误判） | 同时给两个字段 → 取 queue 的值 |
| 17 | `test_queue_wait_excludes_cli_install` | `queue_wait_seconds` 只含三个排队项，不含 `cli_install` | 给 `cli_install=271.9, sandbox_boot_queue=300` → `queue_wait_seconds == 300`，`overhead` 含 `cli_install` |
| 18 | `test_queue_wait_ratio_uses_attempt_total_denominator` | `queue_wait_ratio = sandbox_boot_queue_p50 / attempt_total_p50` | `queue=300, attempt_total=1500` → `0.2` |
| 19 | `test_missing_attempt_total_gives_none` | 缺 `attempt_total_s` → `queue_wait_ratio is None`（不是 0） | 行缺该字段 |

### D.1.4 `TestClassifyDegradation`（**核心：必须覆盖 task8 与 task9 两条真实轨迹**）

| # | 测试函数名 | 验证点 | 输入数据（**照简报 §2.2 / §2.5 的真实数字构造**） |
| --- | --- | --- | --- |
| 20 | `test_task8_trajectory_verdict_per_turn_slowdown` | task8 全轨迹判 `per_turn_slowdown` | 造 rollout 0（`turns=57, model_generation=1066, gen_tokens=21375` → per-turn 18.7s、token/turn 375）与 rollout 59/63（`turns=36, model_generation=2315, gen_tokens=34020` → per-turn 64.3s、token/turn 945）→ `verdict == "per_turn_slowdown"`，`per_turn_growth ≈ 2.44` |
| 21 | `test_task9_healthy_baseline_is_healthy` | task9 判 `healthy` | rollout 0（`turns=54, per-turn 22.2s`）→ rollout 57（`turns=57, per-turn 31.2s`）→ `verdict == "healthy"`，`per_turn_growth ≈ 0.405 < 0.5`，`turns_growth ≈ 0.056 < 0.15` |
| 22 | `test_turn_count_blowup_verdict` | turns 上升触发 `turn_count_blowup` | 初始 `turns=40`，末次 `turns=60`（+50%），per-turn 不变 → `verdict == "turn_count_blowup"` |
| 23 | `test_context_exhaustion_verdict` | peak_context 逼近上限 + `context_exhausted` 占比达标 → `context_exhaustion` | `peak_ctx=125000`（/131072 = 95.4%）、末次 rollout 中 3/8 行 `agent_outcome="context_exhausted"`（37.5% ≥ 10%）→ `verdict == "context_exhaustion"` |
| 24 | `test_task8_not_context_exhaustion` | task8 的 peak_context **下降** → 不判上下文打满 | peak_ctx 从 88000 → 66000（67% → 50% < 90%）→ `verdict != "context_exhaustion"` |
| 25 | `test_task8_not_turn_count_blowup` | task8 turns 下降 → 不判轮数爆炸 | `turns 57 → 36` → `verdict != "turn_count_blowup"` |
| 26 | `test_verdict_priority_per_turn_first` | 优先级：单轮变慢优先于轮数爆炸 | 同时满足 per-turn 上升（+244%）与 turns 上升（+20%）→ `verdict == "per_turn_slowdown"`，`secondary_signals` 含 `"turns_also_growing"` |
| 27 | `test_insufficient_rollouts_gives_low_confidence` | 可用 rollout < 3 → `confidence == "low"`，`verdict == "healthy"`，reasons 说明「样本不足」 | 只有 2 个 rollout |
| 28 | `test_window_median_used_not_single_point` | 首/末用窗口中位数，单点噪声不改变结论 | 首窗口内 3 个 rollout（per-turn 18/19/18），末窗口 3 个（60/62/64）→ `per_turn_seconds_first ≈ 18.x`，不是某一个具体 rollout 的值 |
| 29 | `test_verdict_reasons_contain_numbers` | `verdict_reasons` 每条都带实测数字 | 断言 `"244" in " ".join(reasons)` 或 `"per-turn" in ...`（**断言含关键数值/关键词，不锁死整串文案**） |

### D.1.5 `TestAttributeSlowdown`（**模型慢 vs 基建慢**）

| # | 测试函数名 | 验证点 | 输入数据 |
| --- | --- | --- | --- |
| 30 | `test_model_output_growth_attribution` | token/turn 涨 + 模型占比未降 → `model_output_growth` | token/turn 375→945（+152%），`model_share` 0.89→0.90，tool/queue 无明显上升 |
| 31 | `test_engine_or_infra_attribution` | token/turn 基本不变但 per-turn 涨 → `engine_or_infra` | token/turn 375→380（+1.3% < 30%），per-turn +60%，`model_share` 0.89→0.70（降 0.19） |
| 32 | `test_mixed_attribution` | 输出变长 **且** 排队上升 → `mixed` | token/turn +152%，`queue_wait` +80% |
| 33 | `test_unresolved_when_tokens_missing` | 缺 `generated_response_tokens` → `unresolved`（**不猜**） | 行缺该字段 |
| 34 | `test_not_applicable_when_healthy` | `verdict == "healthy"` → `not_applicable` | task9 数据 |
| 35 | `test_cache_hit_drop_counts_as_infra` | `cache_hit_ratio` 下降 15pp+ → 计入基建证据 | `cached/total` 从 0.8 → 0.6 |

### D.1.6 `TestBudgetHealthFlags`

| # | 测试函数名 | 验证点 | 输入数据 |
| --- | --- | --- | --- |
| 36 | `test_infra_overhead_278s_is_ok_and_1103s_is_bad` | 阈值锚点：`278/3600 = 7.7%` → ok；`1103/3600 = 30.6%` → bad | 造两组 rollout，`infra_overhead_ratio` 分别为 0.077 / 0.306 |
| 36b | `test_queue_wait_ratio_flagged_as_throughput_not_reward` | 排队高 → 告警归类为**吞吐**，**不进奖励杠杆** | `queue_wait_ratio = 0.25` |
| 37 | `test_threshold_boundaries` | 边界值：`0.10` → warn；`0.25` → bad | 精确构造 0.0999 / 0.10 / 0.2499 / 0.25 |
| 38 | `test_flags_empty_when_no_data` | 空输入 → `[]` | `[]` |

### D.1.7 `TestSeriesWithRealTask8Shape`（**回归护栏：用简报真实数字固化行为**）

| # | 测试函数名 | 验证点 | 输入数据 |
| --- | --- | --- | --- |
| 39 | `test_task8_shape_reproduces_evidence_numbers` | 简报 §2.2 的 5 个 rollout 点（0/30/55/59/63）聚合后，单轮耗时 p50 从 ~18.7 升到 ~47~64、turns 从 57 降到 ~36~42、overhead 从 278 升到 1103 | 按简报表逐点构造 |
| 40 | `test_time_budget_ratio_peak_matches_evidence` | 中段 rollout 的 `time_budget_ratio` 接近 62.5% | rollout 55：8 行中 5 行 `agent_outcome="time_budget_exceeded"` → 0.625 |
| 41 | `test_scan_coverage_reported` | 输出含 `scan_coverage`（n_pt_files / n_scanned / missing / is_complete） | 传入缺失 rollout 的元信息 |
| 41b | `test_mixed_keyspace_group_not_lost` | **回归护栏（P1 修复）**：组内混用 `rollout_id=None`(idx 回落) 与真实 `rollout_id` 时，**真实通过样本不得丢失** | task4 rollout 2 group 41 的 8 样本（idx72 None / idx73-76 rollout_id 329-332 / idx77-78 None / idx79 335）→ 5 个 reward=1.0 全部入库且该组通过率 = 5/8 |
| 42 | `test_incomplete_scan_flags_last_values` | `is_complete=False` 时，末值统计标记为「不可直接比较」 | `missing_rollout_ids=[63,64]` |

---

## D.2 `tests/test_failure_reason_taxonomy.py`（**就地扩展已有文件**，对应交付物 B）

> 现有文件已有 `TestNormalizeReason` / `TestClassifyFromFields` / `TestAgentOutcome` / `TestSplitReasonPrefix` 四个类。**新增 3 个类 + 修改 1 个既有断言**。

### D.2.1 需**修改**的既有用例（1 处）

| 位置 | 现状断言 | 改为 | 理由 |
| --- | --- | --- | --- |
| `TestClassifyFromFields::test_agent_outcome_reason_when_status_is_generic` | `("other", "agent_time_budget_exceeded")` | `("timeout", "agent_time_budget_exceeded")` | B.3.5 的唯一行为变更（`other → timeout`） |

### D.2.2 新增 `TestCauseAxis`

| # | 测试函数名 | 验证点 | 输入数据 |
| --- | --- | --- | --- |
| 1 | `test_time_budget_maps_to_agent_budget_axis` | `agent_outcome="time_budget_exceeded"` → `cause_axis == "agent_budget"` | `classify_with_axis({"status":"truncated","agent_outcome":"time_budget_exceeded"})` |
| 2 | `test_context_exhausted_maps_to_context_limit_axis` | `agent_outcome="context_exhausted"` → `cause_axis == "context_limit"` | 同上，换 agent_outcome |
| 3 | `test_two_axes_are_distinguishable` | **同 category（都是 timeout）下 axis 不同** —— 这是本次的核心目标 | 两条输入的 `category` 相同、`cause_axis` 不同 |
| 4 | `test_evaluator_tests_failed_is_policy_failure` | `diagnosis="evaluator:tests_failed"` → axis `evaluator`，且 `is_policy_failure(...) is True` | |
| 5 | `test_invalid_evaluator_payload_is_evaluator_fault` | `diagnosis="evaluator:invalid_evaluator_payload"` → `is_evaluator_fault(...) is True`，`is_policy_failure(...) is False` | 简报 §4.1 的 203 例 |
| 6 | `test_apply_failed_is_evaluator_fault` | `evaluator:apply_failed` → evaluator fault | 简报 15 例 |
| 7 | `test_sandbox_abort_maps_to_infra_axis` | `abort_reason="exception:SandboxApiException"` → axis `infra` | |
| 8 | `test_openhands_infra_exit_maps_to_infra_axis` | `abort_reason="openhands_infra_exit_2"` → axis `infra` | 简报 §2.3 的 219 例 |
| 9 | `test_budget_axis_wins_over_context_axis` | 同时带 `agent_outcome=time_budget_exceeded` 与 `error_code=slime_context_limit_reached` → `agent_budget` | 优先级表序 1 |
| 10 | `test_agent_turns_axis` | `agent_outcome="max_iterations_reached"` → `agent_turns` | |
| 11 | `test_unknown_axis_for_empty_input` | 全空 → `unknown` | `{}` |
| 12 | `test_unknown_outcome_maps_to_unknown_axis` | `agent_outcome="weird_state"` → `unknown`（不编造含义） | |

### D.2.3 新增 `TestReasonDisplay`（标签增量）

| # | 测试函数名 | 验证点 | 输入数据 |
| --- | --- | --- | --- |
| 13 | `test_reason_display_uses_reason_labels` | `reason_display("timeout","agent_time_budget_exceeded")` → 含「时间预算耗尽」 | |
| 14 | `test_reason_display_falls_back_to_reason_label` | 未收录的 reason → 原样返回（不编造） | `reason_display("fail","evaluator:weird_thing")` → `"evaluator:weird_thing"` |
| 15 | `test_reason_label_behavior_unchanged` | **既有 `reason_label` 行为不变**（向后兼容护栏） | `reason_label("timeout","agent_time_budget_exceeded")` → 仍返回原文 `"agent_time_budget_exceeded"` |
| 16 | `test_category_labels_keyset_unchanged` | **`CATEGORY_LABELS` 键集仍是 5 个**（B.4.1 的护栏） | `set(CATEGORY_LABELS) == {"pass","fail","timeout","abort","other"}` |
| 17 | `test_category_label_values_unchanged` | `CATEGORY_LABELS["timeout"] == "超时/截断"`（防止误改文案导致两处不一致） | |
| 18 | `test_cause_axis_labels_cover_all_axes` | `CAUSE_AXIS_LABELS` 覆盖 7 个轴 | |
| 19 | `test_evaluator_fault_reasons_subset` | `EVALUATOR_FAULT_REASONS` 含 `evaluator:invalid_evaluator_payload`、`evaluator:apply_failed`，**不含** `evaluator:tests_failed` | |

### D.2.4 新增 `TestAgentOutcomeReasonAliases`

| # | 测试函数名 | 验证点 | 输入数据 |
| --- | --- | --- | --- |
| 20 | `test_existing_mappings_unchanged` | **既有 3 个映射不变**（护栏） | `agent_outcome_reason("context_exhausted") == "slime_context_limit_reached"`；`("time_budget_exceeded") == "agent_time_budget_exceeded"`；`("sdk_error") == "agent:sdk_error"` |
| 21 | `test_diagnosis_style_time_budget_alias` | `agent_outcome_reason("agent:time_budget_exceeded")` → `"agent_time_budget_exceeded"` | 简报 §4.1 的 `diagnosis` 原值 |
| 22 | `test_unknown_outcome_gets_prefixed_code` | `agent_outcome_reason("weird") == "agent:weird"` | |

### D.2.5 新增 `TestEvaluateNormalizeReasonCompat`（**向后兼容总表逐条固化**）

| # | 测试函数名 | 验证点 | 输入数据 |
| --- | --- | --- | --- |
| 23 | `test_abort_reason_still_wins` | 有 `abort_reason` → `("abort", reason)` 不变 | `abort_reason="agent_time_budget_exceeded"` → `("abort","agent_time_budget_exceeded")` |
| 24 | `test_status_aborted_still_abort` | `status="aborted"` → `("abort","status:aborted")` 不变 | |
| 25 | `test_status_truncated_still_timeout` | `status="truncated"` → `("timeout", …)` 不变 | |
| 26 | `test_context_exhausted_reason_unchanged` | `status="truncated", agent_outcome="context_exhausted"` → `("timeout","slime_context_limit_reached")` 不变 | |
| 27 | `test_error_code_priority_preserved` | `error_code="slime_context_limit_reached"` 优先于 agent 映射 | |
| 28 | `test_sdk_error_still_abort` | `("abort","agent:sdk_error")` 不变 | |
| 29 | `test_reward_pass_fail_unchanged` | `("pass","resolved")` / `("fail","evaluator:tests_failed")` 不变 | |
| 30 | `test_empty_status_still_other` | `status="completed", diagnosis=""` → `("other", …)` 不变 | |

---

## D.3 `tests/test_difficulty_profile.py`（新文件，对应交付物 C）

**构造数据 helper（设计）**：

```python
def _row(rid, gi, iid, cat, *, reward=None): ...     # 与 test_problem_analysis_service.py 同风格
def _visits(spec): ...                               # 直接构造 visits 以测聚合逻辑
```

### D.3.1 `TestBuildDifficultyProfileStates`

| # | 测试函数名 | 验证点 | 输入数据 |
| --- | --- | --- | --- |
| 1 | `test_empty_rows_returns_empty_profile` | 空输入不崩：`n_problems == 0`、states 全 0、`resample` 三清单为空 | `[]` |
| 2 | `test_stable_pass_state` | 累计通过率 ≥ 0.9 → `stable_pass` | 题 a：2 次出现，每次 8/8 通过 → 累计 1.0 |
| 3 | `test_gradient_state` | 0.1 < 率 < 0.9 → `gradient` | 题 b：一次 4/8（0.5） |
| 4 | `test_stable_fail_state` | 率 ≤ 0.1 → `stable_fail` | 题 c：一次 0/8（0.0） |
| 5 | `test_cumulative_uses_all_appearances_not_last` | **替代 v1「末次通过率」的核心护栏**：题 d 首次 0/8、末次 8/8 → 累计 0.5 → `gradient`（**不是** `stable_pass`） | 简报 §3.3 的失效场景 |
| 6 | `test_legacy_last_rate_would_differ` | 显式构造「末次口径与累计口径结论不同」的样本，断言本模块取累计口径 | 同上 |
| 7 | `test_appears_once_flags_data_insufficient` | `n_appearances == 1` → `data_insufficient=True`、`confidence=="low"` | 题 e 一次出现 |
| 8 | `test_appears_three_times_is_actionable` | `n_appearances == 3` → 进入清单、`confidence` 至少 medium | |
| 9 | `test_near_boundary_flagged` | 率 = 0.875（7/8）离 0.9 边界 < 半个量化粒度 → `near_boundary=True` | |
| 10 | `test_last_rollout_id_is_max` | `last_rollout_id` = 最大 rollout_id | 造 rollout 5 / 2 / 9 |
| 11 | `test_trend_reuses_upstream_direction` | `pass_rate_trend` 与 `summarize_pass_visits` 的 trend 一致 | 首次 0/8 → 末次 8/8 → `improved` |
| 12 | `test_slope_computed` | `pass_rate_slope > 0`（在学） | 同上 |
| 13 | `test_best_visit_rate_tracked` | `best_visit_rate` = 单次最高通过率（跨 visit 取 max） | 首次 4/8、末次 2/8 → `best_visit_rate == 0.5` |

### D.3.2 `TestSampledDensity`

| # | 测试函数名 | 验证点 | 输入数据 |
| --- | --- | --- | --- |
| 14 | `test_task9_density_is_50pct` | **复现简报 §3 表**：32 题、16 组 → `sampled_density == 0.5` | 造 32 个 instance |
| 15 | `test_task8_density_is_25pct` | 64 题、16 组 → 0.25 | 造 64 个 instance |
| 16 | `test_task4_density_is_2_3pct` | 688 题、16 组 → `≈0.02326` | 造 688 个 instance（或直接传 `pool_size` 派生，避免造 688 行——见下方说明） |
| 17 | `test_task2_density_is_4_7pct` | 338 题、16 组 → `≈0.04734` | |
| 18 | `test_density_none_when_groups_unknown` | `groups_per_step=None` → `sampled_density is None` + notes 提示 | |
| 19 | `test_density_capped_at_one` | `groups_per_step > n_problems` → 1.0 | 32 题、64 组 |
| 20 | `test_mixed_ratio_passed_through` | 调用方传入的 `mixed_ratio` 原样出现在输出 | 传 0.776 |

> **实现提示（不是缺陷）**：用例 16/17 若真造 688 行会拖慢单测 → 实现时允许 `build_difficulty_profile` 接受 `pool_size` 覆盖参数，或在测试中用**少量真实 instance + 显式 `pool_size`** 组合测公式。**测试要验证的是公式，不是造数据的速度。**

### D.3.3 `TestResampleLists`

| # | 测试函数名 | 验证点 | 输入数据 |
| --- | --- | --- | --- |
| 21 | `test_remove_list_sorted_by_pass_rate_desc` | `remove` 按累计通过率降序 | 两题：1.0 与 0.95 → 1.0 在前 |
| 22 | `test_downsample_sorted_by_best_visit_rate_desc` | **修正计划 §4.3 矛盾的核心护栏**：`downsample` 按 `best_visit_rate` 降序 | 两题累计率均 0.0，但 best 分别 0.125 与 0.0 → 0.125 在前 |
| 23 | `test_keep_boost_sorted_by_distance_to_half` | `keep_boost` 按 \|率 − 0.5\| 升序 | 三题：0.45 / 0.5 / 0.85 → 顺序 0.5, 0.45, 0.85 |
| 24 | `test_low_appearance_problems_excluded_from_lists` | `n_appearances < 3` → 不进任何清单 | 题 f 出现 1 次且率 1.0 → 不在 `remove` 里 |
| 25 | `test_mixed_probability_values` | `compute_mixed_probability(0.5, 8) ≈ 0.9922`；`compute_mixed_probability(1.0, 8) == 0.0`；`compute_mixed_probability(0.9, 8) ≈ 0.5695` | |
| 26 | `test_item_carries_reason_text` | 每条清单项有非空 `reason` 与 `instance_id` | |
| 27 | `test_counts_match_list_lengths` | `resample.counts` 与三清单长度一致 | |
| 28 | `test_zero_gradients_not_removed` | `gradient` 题不进 `remove`（只有 `stable_pass` 进） | |

### D.3.4 `TestClassifyDifficultyState`（阈值边界）

| # | 测试函数名 | 验证点 | 输入数据 |
| --- | --- | --- | --- |
| 29 | `test_threshold_boundaries` | 0.9 → `stable_pass`；0.8999 → `gradient`；0.1 → `stable_fail`；0.1001 → `gradient` | |
| 30 | `test_none_rate_is_unknown` | `pass_rate=None` → `unknown` | |
| 31 | `test_custom_thresholds_respected` | 传自定义阈值时生效 | `stable_pass_threshold=0.8` |

### D.3.5 `TestDifficultyProfileReuseContract`（**上游契约护栏**）

| # | 测试函数名 | 验证点 | 输入数据 |
| --- | --- | --- | --- |
| 32 | `test_same_bucketing_as_summarize_pass_visits` | 同一批 rows，本模块的每题 `n_attempts_total` 与 `summarize_pass_visits` 的 `Σ visit.n_attempts` **相等** | 同一 fixture 同时喂两个函数 |
| 33 | `test_trend_matches_summarize_pass_visits` | 本模块 `pass_rate_trend` 与上游 `tasks[i]["trend"]` **逐题相等** | 同上 |
| 34 | `test_rows_without_instance_skipped` | `instance_id=""` 的行被跳过并计数 | |

---

## D.4 `tests/test_problem_analysis_service.py`（**就地扩展**，C 的上游改动）

| # | 测试函数名 | 验证点 | 输入数据 |
| --- | --- | --- | --- |
| 1 | `test_visit_includes_n_measurable`（新增） | 每个 visit 含 `n_measurable` / `measurable_rate` | 复用现有 `_rows(rid, gi, iid, n_pass, n_fail, n_abort)`，加 1 个 timeout 行 |
| 2 | `test_n_measurable_excludes_abort_and_timeout`（新增） | `n_measurable = n_attempts − n_abort − n_timeout` | 造 4 pass / 3 fail / 1 abort / 2 timeout → `n_measurable == 7` |
| 3 | `test_existing_visits_unchanged`（新增） | **既有 visit 键与值不变**（向后兼容护栏） | 复用现有断言 + 新增键不影响旧键 |
| 4 | `test_mode_groups_per_rollout_public_alias`（新增） | `mode_groups_per_rollout` 与 `_mode_batch_size` 返回一致 | 构造 `groups_per_rollout = {0: {0..15}, 1: {0..15}}` |

---

## D.5 `tests/test_reward_levers.py`（**就地扩展**，C 的杠杆层）

| # | 测试函数名 | 验证点 | 输入数据 |
| --- | --- | --- | --- |
| 1 | `test_always_returns_all_levers`（**需修改**） | 期望集合从 7 条扩到 **10 条**（新增 `gradient_density` / `n_gradient_problems` / `pool_coverage_per_epoch`） | 空 rows |
| 2 | `test_new_levers_have_phenomenon_and_action`（新增） | 3 条新杠杆都有 `title/phenomenon/action` | |
| 3 | `test_gradient_density_low_is_bad`（新增） | `sampled_density = 0.023`（task4 实测）→ `status == "bad"` | |
| 4 | `test_gradient_density_high_is_ok`（新增） | `sampled_density = 0.5`（task9 实测）→ `status == "ok"` | |
| 5 | `test_n_gradient_problems_low_is_bad`（新增） | `n_gradient_problems` 占比低 → bad | |
| 6 | `test_existing_levers_unchanged`（新增） | 既有 7 条杠杆的 key 与 status 逻辑不变 | 复用既有 fixture |

---

## D.6 `tests/test_reward_diagnosis_service.py`（**就地扩展**，B 的 evaluator_fault 统计）

| # | 测试函数名 | 验证点 | 输入数据 |
| --- | --- | --- | --- |
| 1 | `test_series_includes_evaluator_fault_count`（新增） | series 点含 `n_evaluator_fault` / `evaluator_fault_ratio` | 造 1 行 `category="fail"` + `reason="evaluator:invalid_evaluator_payload"` |
| 2 | `test_policy_failure_rate_excludes_evaluator_fault`（新增） | `policy_failure_rate` 分子不含评测器故障 | 3 pass / 1 fail(策略) / 1 fail(评测器) / 1 abort |
| 3 | `test_existing_series_keys_unchanged`（新增） | **既有 series 键与值不变**（向后兼容护栏） | 复用现有 `test_raw_vs_measurable` 的断言 |

---

## D.7 `tests/test_rollout_pt_parser.py`（**新增文件**，B.7 的 `.pt` 侧同步扩展）

> 现有仓库的 `.pt` 侧测试是**分散**的：`tests/test_cluster_attempt_outcomes.py`、`tests/test_summarize_stack_type_breakdown.py`、`tests/test_resolve_sample_reward.py`。→ 本次扩展**就地加在** `tests/test_cluster_attempt_outcomes.py`（不新建文件，符合「先查是否已有同类测试可扩展」）。

### D.7.1 扩展 `tests/test_cluster_attempt_outcomes.py`

| # | 测试函数名 | 验证点 | 输入数据 |
| --- | --- | --- | --- |
| 1 | `test_time_budget_outcome_classified_as_timeout`（新增） | `.pt` 侧 `agent_outcome="time_budget_exceeded"` → `category == "timeout"`、`reason == "agent_time_budget_exceeded"` | `{"status":"completed","metadata":{"agent_outcome":"time_budget_exceeded"},"reward":0.0}` |
| 2 | `test_context_exhausted_still_timeout`（新增） | `agent_outcome="context_exhausted"` → `category == "timeout"`、`reason == "slime_context_limit_reached"` | |
| 3 | `test_budget_and_context_are_distinguishable_in_pt_side`（新增） | 两条同为 timeout 但 reason 不同 | |
| 4 | `test_existing_classification_unchanged`（新增） | **既有 4 条用例全部保持通过**（不重写，只加新用例） | |

### D.7.2 扩展 `tests/test_summarize_stack_type_breakdown.py`

| # | 测试函数名 | 验证点 | 输入数据 |
| --- | --- | --- | --- |
| 5 | `test_timeout_segment_splits_by_time_budget_reason`（新增） | 类型柱 timeout 段细分出 `timeout:agent_time_budget_exceeded`，且 `order_cat` 仍把 timeout 排在 fail 之后、abort 之前 | |

---

## D.8 其它需同步的测试（**改动清单**）

| 文件 | 改动 | 原因 |
| --- | --- | --- |
| `tests/test_failure_reason_taxonomy.py` | **改 1 个断言** + **新增 4 个测试类 / 30 个用例** | B.3.5 / B.4 |
| `tests/test_cluster_attempt_outcomes.py` | **新增 4 个用例**（不改既有） | B.7 |
| `tests/test_summarize_stack_type_breakdown.py` | **新增 1 个用例** | B.7 |
| `tests/test_problem_analysis_service.py` | **新增 4 个用例**（不改既有） | C.4.2 / C.4.3 |
| `tests/test_reward_levers.py` | **改 1 个集合断言**（7→10）+ **新增 6 个用例** | C.9 |
| `tests/test_reward_diagnosis_service.py` | **新增 3 个用例**（不改既有） | B.5.4 |
| `tests/test_agent_budget_profile.py` | **新文件**，40 个用例 | A |
| `tests/test_difficulty_profile.py` | **新文件**，34 个用例 | C |

**总计**：新增约 **118** 个用例，修改 **2** 个既有断言。

---

# E. 交付顺序、func.md 同步与风险登记

## E.1 建议落地顺序（**先修口径，再加能力**）

| 批次 | 内容 | 理由 |
| --- | --- | --- |
| 1 | **B**（taxonomy 扩展） | 分类口径是 A/C 的地基；且改动面最小（零破坏 + 1 处行为修正） |
| 2 | **A**（预算画像） | 直接对应本批第一主因；纯函数，可独立单测 |
| 3 | **C**（题目池三态 + 密度） | 依赖 B 的 `cause_axis`（`evaluator_fault` 影响分母讨论）与上游 `summarize_pass_visits` 扩展 |

## E.2 func.md 同步要求（**强制**）

新增/修改 `utils/`、`service/` 下的公开能力后**必须**：

```bash
python scripts/scan_func_md.py          # 重新生成 func.md
python scripts/scan_func_md.py --check  # CI 校验必须通过
```

预期 `func.md` 变化：

- `utils/agent_budget_profile.py` → 新增一节，登记 8 个公开函数；
- `utils/difficulty_profile.py` → 新增一节，登记 7 个公开函数；
- `utils/failure_reason_taxonomy.py` → 既有 7 条条目**描述微调** + 新增 8 条；
- `service/problem_analysis_service.py` → `summarize_pass_visits` 签名不变、描述追加；新增 `mode_groups_per_rollout`；
- `service/reward_diagnosis_service.py` → `summarize_reward_split` / `build_reward_levers` 描述追加。

**注意**：`scan_func_md.py` 的 `scan_utils` **只登记模块级公开函数，不登记模块级公开常量** → `CAUSE_AXIS_LABELS` / `REASON_LABELS` / `DEGRADATION_VERDICTS` **不会**出现在 func.md 里（`scan_data` 只扫 `data/` 目录）。**这是既有扫描口径，不影响 `--check` 通过**；若希望常量可查，需另开需求（不在本次范围）。

## E.3 README 同步

用户可见变更 → 更新 `README.md`：
- §5.1 失败分析：说明「时间预算耗尽」与「上下文打满」现在可区分（新增归因轴）；
- §5.3 奖励诊断：新增「时间预算健康度」卡片（A 的输出）；
- §5.2 问题分析：新增题目池三态 + 采样密度。

## E.4 风险登记（**全量**）

| # | 风险 | 影响 | 缓解 |
| --- | --- | --- | --- |
| R1 | **单轮耗时口径不确定**（`model_generation/turns` vs `agent_run/turns`） | 曲线数值可差一倍以上 | 双口径输出 + `per_turn_source`；task8 的**方向与量级**在两种口径下一致，结论不变 |
| R2 | **`phase_time` 覆盖率不足** | 部分 rollout 无法算队列指标 | 逐字段缺失降级；`n_valid` 门限；缺失清单显式输出 |
| R3 | **artifacts 与 `.pt` 双源不一致** | 同一 attempt 两个数 | 优先 artifacts + 行上带 `source`；对账用例锁死优先级 |
| R4 | **一个 attempt 多 segment** | 耗时重复累加，虚高 | 行构造器按 attempt 去重；`attempt_total_s` 取 max 不取 sum |
| R5 | **category 不变导致 UI 仍混同** | 用户仍看到 timeout 大桶 | `cause_axis` 在 timeout 段内做二级细分；杠杆文案显式区分 |
| R6 | **`other → timeout` 行为变更** | 既有用例失败、口径数字变化 | 已论证；同步改 1 个断言 + CHANGELOG；方向为「更准」 |
| R7 | **两端 category 不一致**（artifacts=`abort` vs `.pt`=`timeout`） | 汇总口径分裂 | `cause_axis` 拉齐；category 层不强求一致 |
| R8 | **`evaluator_fault` 白名单不全** | 评测器故障被当策略失败 | 白名单可扩展；未命中按策略失败（保守）；`reason` 码全量可见可事后发现 |
| R9 | **题目三态在低样本下伪结论**（task4 每题 1.5 次） | 重采样清单误导 | `data_insufficient` + `min_appearances=3` + `near_boundary`；task4 的正确动作是「缩池/提组数」 |
| R10 | **`sampled_density` 与 `mixed_ratio` 非单调** | 过度声称因果 | 两字段分开 + 文案纪律；实测证据（task4 2.3% vs 48.1%）写进 docstring 与页面 |
| R11 | **阈值多为「带实证锚点的设计假设」** | 跨任务不通用 | 每条阈值在代码注释与本文档标注依据/假设；阈值全部可配置（函数参数） |
| R12 | **utils 反向 import service** | 分层违规 | `groups_per_step` / `mixed_ratio` **由调用方传入**；`mode_groups_per_rollout` 提升为公开后由 service 层调用 |
| R13 | **`summarize_pool_health` 与新三态并存** | 两套口径并存易混淆 | 新口径取代、旧函数 deprecated（保留行为 + docstring 注明）；`build_reward_levers` 的两条旧杠杆标注「旧口径（末次/最好一次）」，新增三态杠杆并列展示 |
| R14 | **`phase_time` 计时口径随 slime 版本变化** | 阈值失效 | `budget_clock` 显式参数；`data_quality.notes` 记录实际口径；`budget_seconds` 可配置 |

## E.5 本设计**明确不做**的事

- 不新增数据表（`ri_attempt_table` + `.pt` metadata 已够）。
- 不改 slime 源码（`check_no_aborted_samples` 的整组丢弃行为只观测、不推动上游改造）。
- 不新建 service（`utils/` 纯函数 + 就地扩展既有 service）。
- 不新增 category（见 B.2.1 的 12 处消费方约束）。
- 不在 `utils/` 内做文件 IO / DB 访问。
- 不改三口径（`pass_rate_strict / gradable / slime_effective`）的定义（属计划 §2.3 的范围）。

---

# 第二部分 · 口径修正与训练侧接入

> 交付物 §A~§F 的设计规格。

# rollout-insight 奖励提升方案 —— 口径修正与训练侧指标接入设计规格

> 交付物：§A 组语义单源 / §B 奖励诊断扩展 / §C TB tag 目录扩展 / §D TB 读取 + 训练健康度 / §E 通过率三口径 / §F 测试清单


---

## 0. 前置：本次实测对简报的三处重要修正

设计前我在真实 run 目录上跑了一轮交叉验证，发现三件简报未覆盖、且**直接改变设计形状**的事实。必须先写清楚，否则后面的口径会错。

### 0.1 slime 在 TB 里已经把自己的组分类写出来了（简报完全没提）

task8 的 `tf_logs` 里存在 **458 个 tag**（与简报一致），其中包含两套**平行的组分类指标族**：

```
rollout/generated/group_count                     = 16   （每步组数，实测恒为 16）
rollout/generated/group_reward/{mixed,all_one,all_zero,unknown}_count
rollout/generated/group_reward/{mixed,all_one,all_zero,unknown}_fraction
rollout/generated/group_reward/classified_group_count / unclassified_group_count
rollout/generated/group_reward/incomplete_group_count / inconsistent_group_count / nonbinary_group_count
rollout/generated/group_reward/pass_count_{0..8}_of_8_count / _fraction
rollout/generated/shaped_group_reward/…           （与上面同构，31 个 tag）
```

**这是 slime 的组语义真源**。本次实测把简报 §2.4 的组表与之对账（task8）：

| rollout | 简报 §2.4「可测组」 | 简报 §2.4「有对有挂」 | slime `mixed_count` | slime `unknown_group_count` |
| --- | --- | --- | --- | --- |
| 0 | 10 | 8 | **13** | **0** |
| 43 | 4 | 2 | **2** | **12** |
| 52 | 1 | 1 | **1** | **15** |
| **59** | **0** | **0** | **0** | **16** |

> **修正 1**：简报的「可测组 / 有对有挂」是**工具侧口径**（`reward_diagnosis_service.classify_group_kind` 去掉 abort/timeout 后数 reward），与 slime 的 `mixed_count` 在 rollout 0 处为 8 vs 13，**偏差达 38%**。两者不是同一个量，简报 §7 验收标准第 3 条「工具 `zero_gradient_ratio` 与 TB `rollout/zero_std/count_*` 对账误差 < 5%」**不可能达成**，因为 `zero_std/count_*` 数的不是工具现在算的那个东西（见 §0.2）。

### 0.2 `removed/count_zero_std_*` 数的是 attempt，`zero_std/count_*` 数的是 group

本次实测（task8 rollout 0 / 2 / 59）：

| rollout | `.pt` 内 `train_drop_reason` 计数 | TB `removed/count_zero_std_1.0` | TB `zero_std/count_1.0` |
| --- | --- | --- | --- |
| 0 | `zero_std_1.0`×**11** | **11** | **1** |
| 2 | `zero_std_1.0`×**29** | **29** | **3** |
| 59 | `zero_std_1.0`×**7** | **7** | (None) |

rollout 0：11 个 attempt 被 zero_std_1.0 丢弃 → 对应 **1 个组**（8 attempt/组）。

> **修正 2**：简报 §3.2 把 `rollout/removed/count_zero_std_1.0` 与 `rollout/zero_std/count_1.0` 混为一谈。实测二者**单位不同**：`removed/count_*` 是 **attempt 级**，`zero_std/count_*` 是 **group 级**。设计必须把两者分列，否则对账永远对不上。

### 0.3 `unknown_group_count` 不是「零梯度」，是「被丢弃」——简报的 `dropped` 语义获得实证支持

slime 的组分类恰好四类，`unknown` 与 `mixed/all_one/all_zero` 互斥且合计 = `group_count`（实测 64/64 步全部成立）。本次实测把「unknown」与「组内含 abort」对账：

| rollout | 工具侧「组内含任一 abort」的组数 | slime `unknown_group_count` |
| --- | --- | --- |
| 43 | 12 | **12** |
| 52 | 15 | **15** |
| 59 | 16 | **16** |
| 62 | 15 | **15** |
| 63 | 11 | **11** |

**rollout 59 精确命中 16 = 16**，与简报 §2.4「全部 16 组都含 abort → 可测组归零 → 该步零梯度」完全一致。

> **修正 3**：简报把 abort 组叫「被丢弃 · 浪费算力」，方向正确；但它的 `mixed_count` 归因是「组内含 abort」这一**实现细节**。真正的 slime 语义是：`unknown` 组 = **不可分类**（含 abort / 存活数不足），`all_one`/`all_zero` 才是**零梯度**（组内 reward 同号）。这正是 §A 要落地的四分类，也**证明 `dropped` 必须与 `all_pass`/`all_fail` 分开** —— 简报的核心论点成立。

### 0.4 附带发现：`perf/rollout_time/*` 直接给出「排队挤占预算」

简报 §1.3 说「沙箱排队挤占 agent 预算」需要从 `.pt` metadata 现算。**实测 TB 里已有现成指标**：

| tag | task8 首→末 | task9（成功）末 | task11 末 |
| --- | --- | --- | --- |
| `perf/rollout_time/attempt/sandbox_boot_queue` | 152.07s → **392.01s** | 128.39s | — |
| `perf/rollout_time_pct/attempt_total/sandbox_boot_queue` | 8.86% → **10.39%** | 4.97% | 10.77% |
| `perf/rollout_time/barrier/group_sibling_wait` | 622.31s → 702.67s | — | — |
| `perf/rollout_time/barrier/rollout_batch_wait` | 1566.63s → 1804.57s | — | — |
| `perf/wait_time_ratio` | 0.82 → **0.94** | — | — |

这让 `queue_wait_ratio` 杠杆**零成本**可得（不必读 `.pt`），且有 4 个任务横向可对账。

### 0.5 附带发现：`passrate/pass@8` 是现成的对账锚点

```
passrate/pass@1   task8: 0.508 → 0.516
passrate/pass@8   task8: 0.875 → 0.875
```

本次实测验证了两个恒等式（task8 全部 64 步）：

- `passrate/pass@1` ≈ (组内 reward==1 的 attempt 总数) / (全部 attempt 总数) → **64 步中 49 步吻合**（容差 5e-4）
- `passrate/pass@8` = (组内至少一个 attempt reward==1 的组数) / 总组数 → **64/64 步精确吻合**

> `pass@8` 的分母是**总组数**（含被丢弃组），不是「可测组」。这是 §E 三口径设计与 §D 对账规则的重要锚点。

### 0.6 附带发现：`.pt` 的 `sample.metadata.train_drop_reason` 是 slime 的**逐 attempt 丢弃原因**真源

本次实测 task8 rollout 59 的 `train_drop_reason` 分布：

```
zero_std_0.0: 15 | agent_timeout: 62 | too_few_survivors_1: 3
aborted:openhands_infra_exit_2: 8 | None: 26 | zero_std_1.0: 7
too_few_survivors_3: 6 | too_few_survivors_2: 2
```

与 TB `rollout/removed/count_*` 逐项一一对应（`agent_timeout` ↔ `removed/count_agent_timeout`，`aborted:<reason>` ↔ `removed/count_aborted_<reason>`，`zero_std_X` ↔ `removed/count_zero_std_X`，`too_few_survivors_K` ↔ `removed/count_too_few_survivors_K`）。

> **重要约束**：`utils/rollout_pt_parser.summarize_train_sample_stats` 已经解析了这个字段，但 `func.md` 标注它是「**死指标**」（无人调用），且**只做计数、不落库**。本次设计**必须复用它**，而不是新写一套解析。

### 0.7 实测发现的既有口径缺陷（超出简报范围，但影响本设计正确性）

工具侧 `reward_diagnosis_service.classify_group_kind` 与 slime 的 `mixed_count` 在 rollout 0 差 13 vs 8，根因是**两处**：

1. **工具把 `truncated`（timeout）排除在分子分母外**；slime 把 truncated attempt 的 reward 也算进组内（实测 rollout 2 的 g34：7 个 truncated + 1 个 completed，n1=5/n0=3 → slime 判 MIXED，因为 truncated 样本**带 reward**：`('truncated', reward=1.0)` 在 DB 中有 236 条）。
2. **工具的 `all_pass` 判定用了「去掉 abort/timeout 后」的子集**；slime 用**全部 attempt**。

> 这意味着 §A 的 `classify_grpo_group` **必须支持「是否计入 timeout」这一维度**，否则工具永远无法与 slime 对账。

---

## §A `utils/slime_grpo_semantics.py` 设计规格

### A.1 目标与依据

**目标**：把 GRPO 组的语义收敛成**纯函数单一真源**，让「奖励诊断」「跨任务对比」「训练健康度」三处共用同一套组判定，杜绝各处口径漂移。

**依据**：
- slime 的组分类真源已在 TB：`rollout/generated/{group_reward,shaped_group_reward}/*`（本次实测 64/64 步满足 `mixed+all_one+all_zero+unknown == group_count`）。
- 实测 slime 的四分类 = `mixed`（真梯度）/ `all_one`（零梯度·题已掌握）/ `all_zero`（零梯度·题未掌握）/ `unknown`（**组奖励无法分类**）。
  > **【实测澄清 · 重要】** `unknown`（TB）与 `dropped`（工具按 `train_drop_reason` 判定）是**两个不同概念**，
  > **不可互相替代**。实测反例：task8 **rollout 0** 的 TB `unknown_group_count` = **0**，
  > 而按 `.pt` 的 `train_drop_reason` 口径整组丢弃 = **6**。
  > 原因：dropped 的真正判据是「**存活 attempt 数 < `SWE_DAPO_MIN_SURVIVING_ATTEMPTS`（默认 4）**」，
  > 而**不是**「含 abort」——实测 rollout 55 组 884 含 4 个 `agent_timeout` 但有 4 个存活 → **整组保留**。
  > 因此 TB 的 `unknown` 只能解释为「组奖励无法分类」，不能直接当成「被丢弃的组数」。
  > **工具的两个口径都要保留**：TB 侧 `unknown`（权威分类）+ `.pt` 侧 `dropped`（训练实际丢弃）。
- 简报 §2.4：task8 rollout 59 时 16/16 组全含 abort、可测组归零（本次实测 slime `unknown_group_count=16` 吻合）。
- **【实测恒等式】** `mixed + all_zero + all_one + unknown == group_count` 在 task8 **64/64 步**、
  task9 **48/48 步**成立；`classified_group_count + unknown_group_count == group_count` 同样 100% 成立。
  → slime 的四分类是**完备划分**，工具应直接采用，不要自造口径。
- 简报 §3.1 的组类型分布（mixed/all_pass/all_fail/infra_only）与 slime 的四分类**结构同构**，但工具当前把 `infra_only` 单独成类 → 必须重构为 `dropped`。

**与 v1 方案的关系**：v1 §3.1.1 提出 `classify_grpo_group(rewards, statuses)` / `group_gradient_contribution(kind)` / `group_action_hint(kind)`。本次**保留这三个签名**，但：
- 补上 v1 未提的 `dropped` 判定（简报 §2.1 明确要求）；
- 把「是否计入 timeout」做成显式参数（否则无法与 slime 对账，见 §0.7）；
- 修正 v1 未提的**单位问题**：`dropped` 组**既不是零梯度、也不是负贡献**，而是「这 8 个 attempt 的 rollout 预算全部白花」。

### A.2 模块定位与分层

```
utils/slime_grpo_semantics.py     ← 纯函数，无状态、无 DB、无 I/O
  ↑
service/reward_diagnosis_service.py     ← 组合 DB rows → 组 → 汇总
service/train_health_service.py         ← 组合 TB 序列 → 组语义对账
```

**约束**：
- 本模块**只做判定与文案**，不做聚合、不读文件、不查 DB。
- 不引入新异常类型；输入不合法时返回 `unknown` / `"unknown"` 而非抛异常（纯函数契约，见 A.5）。
- 不得复制 `utils/failure_reason_taxonomy.py` 的 `normalize_reason`；abort/timeout 的识别**必须复用**它（见 A.6）。

### A.3 数据契约：组语义分类

#### A.3.1 组类型枚举

```python
# utils/slime_grpo_semantics.py
GROUP_KIND_MIXED     = "mixed"      # 有对有挂 → 真梯度
GROUP_KIND_ALL_PASS  = "all_pass"   # 全过     → 零梯度 · 题已掌握
GROUP_KIND_ALL_FAIL  = "all_fail"   # 全挂     → 零梯度 · 题未掌握
GROUP_KIND_DROPPED   = "dropped"    # 含 abort / 存活不足 → 被丢弃 · 浪费算力
GROUP_KIND_UNKNOWN   = "unknown"    # 无法判定（输入不足）
```

> **命名兼容说明**：沿用 `all_pass` / `all_fail`（与简报 v1 一致，也与工具现有 `_GROUP_KINDS` 前两个一致），**废弃** `infra_only`（语义不准：infra_only 在旧口径下表示「可测样本被清光」，而实测证明「含 abort」与「组内全 timeout」是两回事，前者才是被丢弃）。

#### A.3.2 输入契约（关键：attempt 级 reward，不是 segment 级）

```python
def classify_grpo_group(
    rewards: Sequence[Optional[float]],
    statuses: Sequence[Optional[str]],
    *,
    count_timeout_as_reward: bool = True,
    drop_reasons: Optional[Sequence[str]] = None,
    drop_ungradable: bool = True,
) -> str:
    """把一个 GRPO 组的 attempt 级 reward / status 判定成组类型。

    Args:
        rewards:  组内**每个 attempt** 的 reward（attempt 级，不是 segment 级）。
                   ``None`` 表示该 attempt 无可测 reward（abort / 不可测）。
        statuses: 与 rewards 等长的 status 串（``completed`` / ``truncated`` /
                   ``aborted`` / ...）。允许为 None（此时只看 reward）。
        count_timeout_as_reward: 是否把 timeout/truncated attempt 计入组的
                   reward 统计。默认 True —— 与 slime 一致（实测 truncated 样本
                   带 reward，DB 中 task8 有 236 条 truncated+reward=1.0）。
                   置 False 时得到「工具口径」，用于与历史曲线对比。

    Returns:
        ``mixed`` / ``all_pass`` / ``all_fail`` / ``dropped`` / ``unknown``
    """
```

**为什么必须是 attempt 级**：slime 的 GRPO 组内标准化是 attempt 级（一个 group = 同一 `instance_id` 的 N 个 attempt，N=8 实测）。段级 reward（`ri_segment_table` 一行一个 segment，一个 attempt 可跨多 segment）会把一个 attempt 重复计数，且截断样本的多段会污染组的成功/失败计数。

**长度约束**：`len(rewards)` 必须等于 `len(statuses)`；不等时返回 `"unknown"`（不抛异常）。

#### A.3.2.1 【v2 二次修订 · 用 TB 真值标定】`ungradable` 必须计入 `dropped`

**【实测】** 用「工具自推导 vs TB 权威（`rollout/generated/shaped_group_reward/*`）」的
逐 rollout 全对率反向标定规则：

| 规则变体 | task8 全对 | task9 全对 | task4 全对 | unknown 单独吻合 |
| --- | --- | --- | --- | --- |
| `ungradable` 不算 dropped | 43/64 | 13/48 | 18/65 | 44/64, 15/48, 26/65 |
| **`ungradable` 也算 dropped** | **54/64** | **42/48** | 24/65 | **54/64, 42/48, 25/65** |

→ **`ungradable` 必须计入 `dropped`**（task9 吻合度 13/48 → **42/48**）。

**【实测】** `ungradable` 在两个任务里的分布：

| task | 组总数 | 含 ungradable 的组 | 其中**仅因 ungradable** 被判 dropped | 同时含 abort |
| --- | --- | --- | --- | --- |
| task8 | 1024 | 45 | **27** | 18 |
| **task9** | 768 | **64** | **59** | 5 |

→ task9 有 **59 个组（占 92%）**是「仅因 ungradable 被判 dropped」，
   这正是 task9 吻合度 13/48 → 42/48 的来源。

**【实测】** 对关键指标的影响（task9 全量 768 组）：

| 口径 | mixed 组占比 | dropped 组占比 | all_pass | all_fail |
| --- | --- | --- | --- | --- |
| 旧口径（不含 ungradable） | **75.4%** | 8.9% | 114 | 7 |
| **新口径（计 ungradable→dropped）** | **67.8%** | **16.5%** | 114 | 6 |

→ **旧口径同时高估梯度密度、低估浪费算力**，两个方向都错。
   漏掉这条规则会把这 86 个组误判成 mixed/all_pass，直接污染
   `gradient_density` / `zero_gradient_ratio`。

**【实测 · v2 三次修订 · 穷举定稿】** 在 task8 / task9 / task4 上对
`(too_few_survivors 是否算 dropped, ungradable 是否算 dropped, 是否要求存活数<2, truncated 是否计入 reward)`
四个维度做**组合穷举**，唯一最优解是 **(0, 1, 0, 1)**：

| 规则组合 | task8 全对 | task9 全对 | task4 全对 |
| --- | --- | --- | --- |
| (0,0,1,1) 早期 | 43/64 | 13/48 | 18/65 |
| (0,1,1,1) 二次修订 | 54/64 | 42/48 | 24/65 |
| (1,1,1,1) 加 too_few_survivors | 47/64 | 42/48 | — |
| **(0,1,0,1) ★ 最终定稿** | **64/64** | **48/48** | **62/65** |

→ **最终规则**：`dropped` ⟸ 含 abort / `agent_timeout` / `aborted:*` **或** 含 `ungradable`；
  **不因 `too_few_survivors_*`**、**不设存活数门槛**；`truncated` 计入 reward 统计。

**【实测】** 单组核验（task8 rollout 0）：工具推导 `mixed=13 / all_pass=1 / all_fail=2 / dropped=0`
与 TB `mixed_count=13 / all_one_count=1 / all_zero_count=2 / unknown_count=0` **逐项精确吻合**。

**【实测】** 三口径对关键指标的影响（全量）：

| task | 口径 | mixed 组占比 | dropped 组占比 |
| --- | --- | --- | --- |
| task8（1024 组） | 旧口径 | 37.9% | 43.4% |
| | v2 二次修订 | 34.8% | 47.0% |
| | **★最终定稿** | **35.6%** | **46.0%** |
| task9（768 组） | 旧口径 | 75.9% | 7.9% |
| | v2 二次修订 | 67.8% | 16.5% |
| | **★最终定稿** | **68.4%** | **15.6%** |

→ 旧口径**同时高估梯度密度、低估浪费算力**。

**【设计】** 这条规则是**fallback 的标定依据**（用 TB 真值标定出最接近的规则），
**不替代 TB** —— §A 的 `classify_grpo_group` 仍是「无 TB 场景的 fallback」，
有 TB 时应优先读 `rollout/generated/shaped_group_reward/*`。

**【实测 · 阻塞项】DB 侧字段缺失 → 该规则无法仅凭 DB 复现。**
`ri_attempt_table` 现有列：`primary_id, task_id, rollout_id, group_index, attempt_id, sandbox_id,
instance_id, status, reward, agent_turns, tool_calls, diff_line_count, response_length,
trainable_tokens, finalized` —— **没有** `train_drop_reason` / `abort_reason` / `agent_outcome` / `diagnosis`。

| 口径 | task8 全对 | task9 全对 |
| --- | --- | --- |
| DB-only（现有字段） | 43/64 | 13/48 |
| **`.pt`-based（含 `train_drop_reason`）** | **64/64** | **48/48** |

→ **`train_drop_reason` 必须落库**（`ri_attempt_table` 加列，不建表），
   否则 `ungradable` 这个关键触发条件拿不到，组语义无法精确复现 slime。
→ 落库位置：`service/rollout_scan_service` 解析 `.pt` 时一并写入（与 `agent_turns` 同层）。
→ 老数据该列为空 → 走 DB-only 近似口径，页面须标注「近似值」。

**【实测】** `train_drop_reason` 全部取值（跨 task4/8/9，共 13 种）：
`None` / `zero_std_1.0` / `zero_std_0.0` / `agent_timeout` / `ungradable` /
`aborted:exception:SandboxApiException` / `SandboxExecStalled` / `SandboxInternalException` / `RuntimeError` /
`aborted:openhands_infra_exit_2` / `_137` / `aborted:evaluator_total_timeout` /
`too_few_survivors_1` / `_2` / `_3`。

**【实测】** task4 的 `zero_std_1.0` 高达 **2599** 个 attempt（≈5–11 组/rollout），
样例确认这些组是 **8/8 全过** → **task4 题目池里大量题已被完全掌握**（v1「题目池饱和」的直接证据）。

#### A.3.3 判定规则（判定顺序严格从上到下，短路返回）

设组内 attempt 集合为 `A`，`n = len(A)`。

| 顺序 | 条件 | 返回 | 实证依据 |
| --- | --- | --- | --- |
| **R0** | `n == 0` | `unknown` | 空组无意义 |
| **R1** | 存在 `status == "aborted"`（或 abort 状态集命中） | **`dropped`** | 实测 rollout 43/52/59/62/63：含 abort 组数与 slime `unknown_group_count` **12/15/16/15/11 精确相等** |
| **R1.5** | **存在 `train_drop_reason == "ungradable"`**（无 abort 时） | **`dropped`** | **【v2 二次修订】** 用 TB 真值标定：加此规则后 task9 吻合度 **13/48 → 42/48**、task8 43/64 → **54/64**（A.3.2.1）。task8 有 **27 个组仅因 ungradable 被判 dropped** |
| ~~**R2**~~ | ~~存活 attempt 数 < 2 → dropped~~ | **（已移除）** | **【v2 三次修订】** 穷举标定证明：保留该门槛会破坏逐 rollout 吻合（task8 64/64 → **54/64**、task9 48/48 → **42/48**）。**最终定稿不设存活数门槛** |
| ~~**R2'**~~ | ~~`too_few_survivors_*` → dropped~~ | **（不采用）** | **【v2 三次修订】** 同上：加上它 task8 掉到 47/64。`too_few_survivors_*` 记录的是「部分 attempt 被移除后存活数不足」这一**结果**，不是 slime `unknown` 的触发条件 |
| **R3** | 统计口径内的 attempt 中，同时存在 `reward ≈ 1` 与 `reward ≈ 0` | **`mixed`** | slime `mixed_count` 定义 |
| **R4** | 统计口径内只有 `reward ≈ 1` | **`all_pass`** | slime `all_one_count` |
| **R5** | 统计口径内只有 `reward ≈ 0` | **`all_fail`** | slime `all_zero_count` |
| **R6** | 统计口径内既无 1 也无 0（如全 None） | **`unknown`** | — |

**R2 的「存活数 < 2」阈值来源**：slime 的 `too_few_survivors_K` tag 中 `K` 即存活数（实测 rollout 0 g5 → `too_few_survivors_2`，g11 → `too_few_survivors_3`；rollout 2 g34 → `too_few_survivors_1`）。GRPO 组内标准化在存活数 < 2 时数学上无意义（无法计算组内均值/方差），故阈值取 2。**K 的确切语义（= 存活数 还是 = 阈值）标记为「设计假设」** —— 实测只观察到 K∈{1,2,3} 且与组内非 truncated attempt 数一致，未穷举 K=0..8 全域。

**R3/R4/R5 的统计口径**：
- `count_timeout_as_reward=True`（默认）：`status == "truncated"` 的 attempt **计入** reward 统计（取其 reward 值）。
- `count_timeout_as_reward=False`：`truncated` attempt **不计入**（既不分子也不分母）。
- 两种口径下 `status == "aborted"` 的 attempt **永远不计入**（R1 已短路）。

**reward 判定容差**：`abs(r - 1.0) <= 1e-9` 记为成功；`abs(r) <= 1e-9` 记为失败；其它值（如 0.5）**两侧都不算**，落到 R6 → `unknown`。
> 依据：实测全量 reward 只有 0.0 / 1.0 两个值（简报 §3.2 同）。保持与 `db/db_attempt_service._classify_attempt_row` 的 `_REWARD_EPS = 1e-9` 一致。

#### A.3.4 `all_pass` / `all_fail` 的方向性（本设计最关键的语义修正）

| 组类型 | 梯度贡献 | 业务含义 | 正确动作 |
| --- | --- | --- | --- |
| `mixed` | **1.0** | 题在能力边界上，有学习信号 | **保留**，加大采样 |
| `all_pass` | **0.0** | 题**已掌握** | **移出训练集**（或大幅降采样） |
| `all_fail` | **0.0** | 题**未掌握** | **换题 / 降采样**（补中间难度题） |
| `dropped` | **0.0** | 组内含 abort / 存活不足 → **整组丢弃** | **修预算 / 修基建**，不要动题池 |

> `all_pass` 与 `all_fail` **梯度贡献相同（都是 0）但动作完全相反** —— 这是简报 §2.1 的核心论点，设计必须把它落到 `group_action_hint` 的文案上，不可合并成一句「零梯度」。

### A.4 函数签名全集

```python
__all__ = [
    "GROUP_KINDS",
    "GROUP_KIND_MIXED",
    "GROUP_KIND_ALL_PASS",
    "GROUP_KIND_ALL_FAIL",
    "GROUP_KIND_DROPPED",
    "GROUP_KIND_UNKNOWN",
    "GRADIENT_CONTRIBUTIONS",
    "GROUP_ACTION_HINTS",
    "GROUP_KIND_LABELS",
    "classify_grpo_group",
    "group_gradient_contribution",
    "group_action_hint",
    "group_kind_label",
    "is_zero_gradient",
    "is_wasted",
    "group_zero_gradient_ratio",
    "group_wasted_rollout_ratio",
    "group_gradient_density",
]

GROUP_KINDS: Tuple[str, ...] = ("mixed", "all_pass", "all_fail", "dropped", "unknown")
```

#### A.4.1 `group_gradient_contribution(kind) -> float`

```python
def group_gradient_contribution(kind: str) -> float:
    """该类组对梯度的贡献度：mixed=1.0，其余=0.0。

    ``dropped`` 也是 0.0，但它额外意味着「rollout 预算被浪费」，
    需与零梯度分开看 —— 见 ``is_wasted``。
    """
```

**契约**：
- `mixed` → `1.0`；`all_pass` / `all_fail` / `dropped` / `unknown` → `0.0`。
- 未知 kind → `0.0`（不抛异常）。

> **注意**：本函数**不能**表达「负贡献」。`dropped` 的负贡献是**算力层面**的（8 个 attempt 的 rollout 预算白花 + slime 触发 over-sampling 重试），不是**梯度层面**的（组被丢弃后，该 attempt 的 logprob 不参与更新，既不推正也不推负）。把这个区分写进 docstring，避免下游误用成「负梯度惩罚」。

#### A.4.2 `group_action_hint(kind) -> str`

```python
def group_action_hint(kind: str) -> str:
    """处置建议文案。三类零梯度的动作方向完全相反，不可合并。"""
```

**文案表（中文，直接进 UI）**：

| kind | 返回文案 |
| --- | --- |
| `mixed` | `真梯度：保留该题并加大采样；可优先扩��到同类「差一点就过」的题。` |
| `all_pass` | `零梯度 · 题已掌握：移出训练集或大幅降采样；继续采样只是白花算力。` |
| `all_fail` | `零梯度 · 题未掌握：换题或降采样；补入中间难度题（累计通过率 0.2~0.5）比硬训更有效。` |
| `dropped` | `被丢弃 · 浪费算力：整组 rollout 白花（预算被 abort / 存活不足吃掉）；先修时间预算与基建，不要动题池。` |
| `unknown` | `无法判定：缺少 attempt 级 reward 或状态，暂不计入梯度统计。` |

> 文案必须明确区分「零梯度」（两类，方向相反）与「被丢弃 · 浪费算力」（第三类），这是简报 §2.2 的直接要求。

#### A.4.3 辅助函数

```python
def group_kind_label(kind: str) -> str:
    """组类型的中文短标签。"""
    # mixed     → "真梯度"
    # all_pass  → "零梯度 · 题已掌握"
    # all_fail  → "零梯度 · 题未掌握"
    # dropped   → "被丢弃 · 浪费算力"
    # unknown   → "未判定"

def is_zero_gradient(kind: str) -> bool:
    """是否零梯度（all_pass / all_fail）。dropped 不算 —— 它是浪费算力。"""
    return kind in (GROUP_KIND_ALL_PASS, GROUP_KIND_ALL_FAIL)

def is_wasted(kind: str) -> bool:
    """是否被丢弃（浪费 rollout 预算）。"""
    return kind == GROUP_KIND_DROPPED
```

#### A.4.4 三个一等公民指标（纯函数，输入组类型计数）

```python
def group_zero_gradient_ratio(counts: Mapping[str, int]) -> Optional[float]:
    """零梯度组占比 = (all_pass + all_fail) / n_groups。

    n_groups 为 0 时返回 None（不是 0.0 —— 0.0 会被下游误读成「全部零梯度」）。
    """

def group_wasted_rollout_ratio(counts: Mapping[str, int]) -> Optional[float]:
    """浪费算力组占比 = dropped 组数 / n_groups。"""

def group_gradient_density(counts: Mapping[str, int]) -> Optional[float]:
    """梯度密度 = mixed 组数 / n_groups。"""
```

**分母口径（关键）**：`n_groups` = **全部组数**（含 dropped）。
> 依据：本次实测 slime 的 `passrate/pass@8` 分母是总组数（64/64 步精确吻合 `(至少一个 reward=1 的组数) / 总组数`）。分母若剔除 dropped，会把「预算被吃掉」伪装成「梯度密度高」——正是 task4（`removed_ratio`→1.0）需要暴露的问题。

### A.5 边界与异常行为

| 情形 | 行为 |
| --- | --- |
| `rewards` / `statuses` 长度为 0 | `classify_grpo_group` → `unknown` |
| 两者长度不等 | → `unknown`（**不抛异常**，纯函数不因脏数据炸掉整条诊断链） |
| `reward` 是 `None` | 该 attempt 两侧都不计（既不算 1 也不算 0） |
| `reward` 是 `0.5` 等中间值 | 不计入 1/0 两侧；若组内只有中间值 → `unknown` |
| `status` 是 None | 只看 reward；`reward` 也 None → 该 attempt 不计入 |
| 未知 kind 传入 `group_gradient_contribution` | → `0.0` |
| 未知 kind 传入 `group_action_hint` | → `unknown` 的文案 |

> 设计决策：**不引入 `RolloutInsightException`**。理由：本模块是诊断链路上的纯判定器，脏数据（部分 attempt 缺 reward）在真实数据里必然存在（实测 abort 样本 `reward=None`）；让整个奖励诊断页因一行脏数据 500 不符合项目「不向页面抛内部错误」的约定（见 conventions「全局异常 handler」条）。真正的「任务不存在」等错误仍由 `service` 层抛 `RolloutInsightException`。

### A.6 复用关系（必须遵守，不新建平行实现）

| 需求 | 复用 | 说明 |
| --- | --- | --- |
| abort / timeout 状态识别 | `db/db_attempt_service.DBAttemptService._ABORT_STATUSES` / `_TIMEOUT_STATUSES` | 与 `utils/rollout_pt_parser` 对齐；**不要**在本模块重定义一套状态集合 |
| abort 原因归一 | `utils/failure_reason_taxonomy.normalize_reason` / `classify_from_fields` | 已把 `time_budget_exceeded` 映射为 `agent_time_budget_exceeded` |
| reward 容差 | `db/db_attempt_service._REWARD_EPS`（1e-9） | 保持一致，避免「1.0 vs 0.999999999」在两处判定不同 |
| 单 attempt 结局分类 | `utils/rollout_pt_parser.classify_sample_outcome` | 若调用方已有 `.pt` sample，直接用它的 `category`，不要重新推导 |
| 逐 attempt 丢弃原因 | `utils/rollout_pt_parser.summarize_train_sample_stats` 的 `by_drop_reason` | 该函数目前是「死指标」（func.md 未被调用）；本次由 §B/§D 复活它，**不要**另写一份解析 |

> **注意 `normalize_reason` 的已知缺陷**：`utils/failure_reason_taxonomy.py` 把 `time_budget_exceeded` 归入 `timeout` category（简报 §6 明确指出）。这与 slime 的语义**冲突**：实测 task8 的 `agent_time_budget_exceeded` 会让整组被丢弃（`dropped`），而不是被当作「超时但组仍有效」。因此 **§A 的 R1/R2 判定不能只依赖 `normalize_reason` 的 category**，必须直接看 `status` 原始串与 `abort_reason`。这是 `taxonomy` 扩展（计划 §1.4(b)）需要修的点，但**不在本交付物范围内**，此处只登记为**依赖与风险**。

### A.7 向后兼容影响

- **新模块，无历史消费者** → 零破坏。
- 但 `service/reward_diagnosis_service.classify_group_kind`（现有公开 API，func.md 已登记）**必须保留**（见 §B.7），其返回值域要从 `{mixed, all_pass, all_fail, infra_only}` 扩到 `{mixed, all_pass, all_fail, dropped, unknown}`。
- `tests/test_reward_diagnosis_service.py::TestClassifyGroupKind::test_four_kinds` 现有断言 `classify_group_kind(["abort","timeout"]) == "infra_only"` **会失败** → 必须在 §F 中列明该断言的迁移方式（见 F.2）。

---

## §B `service/reward_diagnosis_service.py` 就地扩展设计

### B.1 目标与依据

**目标**：在**现有模块内**扩展，不新建平行 service（简报 §2.2 / 计划 §5 明令）。

**依据**：
- 简报 §2.2：`zero_gradient_ratio` / `wasted_rollout_ratio` / `gradient_density` 三条一等公民曲线。
- 简报 §2.4：`summarize_pool_health` 的「末次通过率」口径在 task4 失效。
- 简报 §3.1：`removed_ratio` 是第一主因指标；task8 实测 `removed_ratio` 0.391→0.628，`trainable_attempt_count` 78→47。
- 本次实测：`wasted_rollout_ratio`（含 abort 组占比）在 task8 rollout 59 达到 16/16 = **100%**，与 slime `unknown_group_count=16` 精确一致 → 确认它比 `zero_gradient_ratio` 是更好的先行指标（简报 §2.2 论断成立）。
- 本次实测：task8 的 `mixed_count` 在 rollout 0 是 13，而工具当前算出的 `learnable_ratio` 对应 8 → 工具口径与 slime 有系统性偏差（见 §0.7），本次扩展顺带修正。

### B.2 `summarize_gradient_groups` 修订

#### B.2.1 签名（保持向后兼容，只加参数）

```python
def summarize_gradient_groups(
    rows: Sequence[Dict[str, Any]],
    *,
    count_timeout_as_reward: bool = True,
) -> Dict[str, Any]:
    """按 rollout 统计四类组数量与三类一等公民指标曲线。

    Args:
        rows: attempt 结局行（``DBAttemptService.list_outcome_rows_by_task`` 形态），
              每行需含 ``rollout_id`` / ``group_index`` / ``category`` / ``reward``。
        count_timeout_as_reward: 组语义口径，见 ``utils.slime_grpo_semantics``。
    """
```

#### B.2.2 输出 schema（**新增字段**用 `+` 标出）

```jsonc
{
  "n_rollouts": 64,

  "series": [
    {
      "rollout_id": 59,

      "n_groups": 16,          // 已有：总组数（含 dropped）
      "n_mixed": 0,            // 已有
      "n_all_pass": 0,         // 已有
      "n_all_fail": 0,         // 已有
      "n_infra_only": 2,       // 已有（废弃语义，保留字段做兼容）
      "learnable_ratio": 0.0,  // 已有：== gradient_density 的旧名

      // ---- 以下为新增 ----
      "n_dropped": 16,         // 新增：被丢弃组数（含 abort / 存活不足）
      "n_zero_gradient": 0,    // 新增：all_pass + all_fail
      "n_unknown": 0,          // 新增：无法判定组数

      "gradient_density": 0.0,       // 新增：n_mixed / n_groups
      "zero_gradient_ratio": 0.0,    // 新增：(all_pass+all_fail) / n_groups
      "wasted_rollout_ratio": 1.0,   // 新增：n_dropped / n_groups

      "group_kinds": {                // 新增：逐类明细，供 UI 直接画堆叠柱
        "mixed": 0, "all_pass": 0, "all_fail": 0, "dropped": 16, "unknown": 0
      },
      "group_labels": {               // 新增：语义标注（简报 §2.2 要求）
        "mixed": "真梯度",
        "all_pass": "零梯度 · 题已掌握",
        "all_fail": "零梯度 · 题未掌握",
        "dropped": "被丢弃 · 浪费算力"
      }
    }
  ],

  "dips": [ /* 已有：learnable_ratio 骤降点 */ ],

  "latest": { /* 已有：最后一个 series 点 */ },

  "note": "..."   // 已有：文案更新
}
```

#### B.2.3 语义标注文案（简报 §2.2 逐字要求）

| kind | 旧文案 | 新文案 |
| --- | --- | --- |
| `all_pass` | 隐含「已饱和」 | **「零梯度 · 题已掌握」**，动作：**移出训练集** |
| `all_fail` | 「不可学习」 | **「零梯度 · 题未掌握」**，动作：**换题 / 降采样** |
| `dropped`（原 `infra_only`） | 「基建问题」 | **「被丢弃 · 浪费算力」** |

> 关键修正：旧 `note` 写「只剩基建表示可测样本被清光」——把 abort 当基建问题。简报 §2.3 **推翻**了这个判断（task8 的 1425 例 abort 中 1425 例是 `agent_time_budget_exceeded`，沙箱类仅 14 例）。新文案必须写「**整组 rollout 白花（预算被 abort / 存活不足吃掉）；先修时间预算，不要动题池**」。

#### B.2.4 三条新曲线的算法

对每个 rollout：

```
n_groups      = 该 rollout 的组数
n_mixed       = count(kind == "mixed")
n_zero_grad   = count(kind in {"all_pass", "all_fail"})
n_dropped     = count(kind == "dropped")

gradient_density    = n_mixed / n_groups
zero_gradient_ratio = n_zero_grad / n_groups
wasted_rollout_ratio= n_dropped / n_groups
```

分母统一为 `n_groups`（含 dropped），依据见 A.4.4。

#### B.2.5 `dips` 的修订

现有 `dips` 只检测 `learnable_ratio` 骤降（≥0.15）。新增：

```python
"wasted_dips": [
  {
    "rollout_id": 59,
    "prev_rollout_id": 43,
    "prev_ratio": 0.75,
    "ratio": 1.0,
    "drop": 0.25
  }
]
```

- `wasted_dips` 判定：`wasted_rollout_ratio` 环比 **上升** 且绝对增幅 ≥ `_WASTED_DIP_RISE = 0.25`。
- 阈值依据（**实证**）：task8 实测 `wasted_rollout_ratio` 逐 rollout 0% → 25%(r14) → 75%(r43) → 93.8%(r52) → **100%(r59)**，相邻两跳的典型幅度为 0.25~0.50。取 0.25 可捕捉到 r43/r52/r59 三个关键跳变点，又不会被单点抖动触发。
- 现有 `dips` 保留不动（向后兼容）。

### B.3 `summarize_pool_health` 修订

#### B.3.1 问题陈述（简报 §2.4）

旧实现用「**末次（最好一次）**通过率」判定饱和（≥0.875）/ 太难（≤0.125）。
**失效证据（简报 §3.3 + 本次实测）**：task4 有 688 题，本次实测**每题只出现在 1~2 个 rollout**（直方图 `{出现1次: 368题, 出现2次: 320题}`）→ 「末次」≈「首次」→ 无法反映学习进展。

#### B.3.2 新口径：累计通过率三态

```python
def summarize_pool_health(
    rows: Sequence[Dict[str, Any]],
    *,
    difficulty_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """按题目**累计**通过率看题目池健康度（三态）。

    与 ``utils.difficulty_profile`` 的职责划分见 B.3.4。
    """
```

**三态定义（累计口径，跨全部 rollout 聚合）**：

| 态 | 判定 | 业务含义 | 动作 |
| --- | --- | --- | --- |
| `stable_pass` | 累计通过率 **≥ 0.9** | 题已掌握 | 移出训练集 |
| `gradient` | **0.1 <** 累计通过率 **< 0.9** | 可学习（GRPO 最有价值） | 保留 / 加大采样 |
| `stable_fail` | 累计通过率 **≤ 0.1** | 题未掌握 / 太难 | 降采样 / 补中间难度题 |

**阈值依据（实证）**：
- 边界取 0.9 / 0.1 来自计划 §2.4 的既定口径。
- **本次实测验证**：task4（688 题）→ `stable_pass=213, gradient=424, stable_fail=51`；task8（64 题）→ `stable_pass=3, gradient=54, stable_fail=7`。task9（32 题，成功任务）应以 `gradient` 为主 —— 计划 §7 验收标准第 5 条即以此为准。
- 与旧口径的差异：旧 `saturated_pool_ratio` 在 task4 会把「只出现过 1 次且那次恰好全过」的题判为饱和（本次实测 task4 有 368 题只出现 1 次 → 旧口径噪声极大）。新口径在样本不足时**必须标「数据不足」**（见 B.3.3）。

#### B.3.3 样本量置信度标注（新增，必做）

本次实测 task4 每题平均只出现 **1.5 次**（简报 §3.3 一致）。在这种样本量下，单次全过 → 累计通过率 1.0 → 判 `stable_pass`，但**置信度极低**。

**规则**：

```python
MIN_APPEARANCES_FOR_CONFIDENCE = 3
```

- `n_appearances < 3` → 该题标 `confidence = "insufficient_data"`，**仍计入三态统计但单独统计比例**。
- 输出 `n_insufficient_data` 与 `insufficient_data_ratio`。
- **UI 与杠杆不得把 `insufficient_data` 的题当作可靠结论**（见 B.4 的 `n_gradient_problems` 判定）。

> 阈值 3 的依据：累计通过率要能区分「0/3 全挂」与「1/3 偶然通过」，至少需要 3 次观测；少于 3 次时，「2/2 全过」与「2/3」在噪声上不可区分。**标注为设计假设**（无直接实证支撑该具体数字，但简报 §3.3 的 1.5 次样本量明确要求引入此保护）。

#### B.3.4 与 `utils/difficulty_profile.py` 的职责划分（简报 §2.4 明确要求写清）

计划 §4.1 规定 `utils/difficulty_profile.py` 负责「按 `instance_id` 跨全部 rollout 聚合累计通过率 → 三态 + 每题 `n_appearances` / `last_rollout_id` / `pass_rate_trend`」。

**本文档的职责边界（严格遵守，避免重复实现）**：

| 层 | 职责 | 归属 |
| --- | --- | --- |
| **题目级判定** | 单题的三态归属、累计通过率、出现次数、趋势、置信度标注 | `utils/difficulty_profile.py`（题目级纯函数，输入按题聚合的计数） |
| **任务级汇总** | 把题目级结果汇成任务级比例、档位计数、采样密度、梯度密度 | `service/reward_diagnosis_service.summarize_pool_health`（service 层） |

**调用契约**：

```python
# service 层
from utils.difficulty_profile import classify_problem_difficulty, summarize_difficulty_profile
```

- `classify_problem_difficulty(cum_pass_rate, n_appearances) -> str` 返回 `"stable_pass" | "gradient" | "stable_fail"`
- `summarize_difficulty_profile(per_problem) -> Dict` 返回题目级汇总

**`summarize_pool_health` 只做**：调用 `difficulty_profile` 判定 → 汇总为任务级比例 → **不自己实现三态阈值逻辑**。若 `difficulty_profile` 尚未落地，`summarize_pool_health` 可先内联一份**临时**判定，但必须注明「与 `difficulty_profile` 的阈值需保持一致」，并在 `difficulty_profile` 落地后改为调用（TODO 注释 + func.md 登记）。

#### B.3.5 输出 schema（新增 / 变更）

```jsonc
{
  "n_problems": 688,

  // ---- 旧字段：保留（向后兼容），语义改为「累计通过率」口径 ----
  "saturated_pool_ratio": 0.3096,   // == stable_pass_ratio（累计口径）
  "hard_pool_ratio": 0.0741,        // == stable_fail_ratio（累计口径）
  "n_saturated": 213,               // == n_stable_pass
  "n_hard": 51,                     // == n_stable_fail

  // ---- 新增 ----
  "n_stable_pass": 213,
  "n_gradient": 424,
  "n_stable_fail": 51,
  "stable_pass_ratio": 0.3096,
  "gradient_ratio": 0.6163,
  "stable_fail_ratio": 0.0741,

  "n_insufficient_data": 368,        // 出现次数 < 3 的题数
  "insufficient_data_ratio": 0.5349,

  "sampled_density": {
    "n_problems": 688,
    "groups_per_step": 16,
    "mean_appearances": 1.5,
    "median_appearances": 1.0,
    "appearance_interval_steps": 31.1,
    "coverage_per_epoch": 0.0233,     // groups_per_step / n_problems
    "note": "..."
  },

  "pool_health_basis": "cumulative"  // 新增：明确口径，'cumulative' | 'legacy_last'
}
```

> `sampled_density` 的**实证依据**（简报 §3 表，本次实测复核）：
> - task9（32 题）→ 梯度密度 **50%**，唯一成功；
> - task8 / task11（64 题）→ **25%**；
> - task4（688 题）→ **2.3%**，`removed_ratio` 冲到 1.0。
>
> 公式：`coverage_per_epoch = groups_per_step / n_problems`（task4 = 16/688 = 0.0233 = 2.33%，与简报 2.3% 吻合）。
> `groups_per_step` 的来源：本次实测 `rollout/generated/group_count` **恒为 16**（task8 全 64 步、task4 全 65 步、task9 全 48 步、task11 全 50 步、task2 全 33 步），与简报 §3「全部任务实测 `groups_per_step = 16.0`」一致。**优先从 TB 的 `rollout/generated/group_count` 取**；取不到时回退到 `problem_analysis_service.summarize_pass_visits` 的 `batch_size`（它已实现 `_mode_batch_size` / `_env_batch_size` 的 `ROLLOUT_BATCH_SIZE` 解析）。

### B.4 `build_reward_levers` 新增杠杆

#### B.4.1 杠杆表结构（扩展现有 4 元组 → 5 元组）

现有：`(key, title, phenomenon, action)`。新增**状态判定规则**字段：

```python
# 每条杠杆：(key, 标题, 现象判据, 建议动作, 状态判定规则)
_LEVER_RULES: Tuple[Tuple[str, str, str, str, str], ...] = (...)
```

#### B.4.2 新增杠杆条目（6 条）

| key | 标题 | 现象判据 | 建议动作 | 状态判定（ok/warn/bad） |
| --- | --- | --- | --- | --- |
| `time_budget_ratio` | 时间预算耗尽占比 | `time_budget_exceeded` 占 attempt 比例升高 → 60 分钟墙钟把 agent 杀掉 | 提高 agent 时间预算（60→90 分钟）**或**限制单轮输出长度（单轮 token +180% 是根因） | `bad` ≥ 0.30；`warn` ≥ 0.10；else `ok` |
| `per_turn_seconds_growth` | 单轮耗时增幅 | 单轮生成耗时相对训练初期上升 → 模型学会每轮输出更多 → 预算内跑不完 | 对输出长度加惩罚 / 在 prompt 约束单轮输出；检查是否出现长度膨胀 | `bad` ≥ +100%；`warn` ≥ +50%；else `ok` |
| `gradient_density` | 梯度密度 | mixed 组占比低 → 每步真正有梯度的组太少 | 缩小题目池（把 `stable_pass` 题移出）或提高 `groups_per_step`；目标每步 ≥ 50% 组是 mixed | `bad` < 0.15；`warn` < 0.30；else `ok` |
| `n_gradient_problems` | 可学习题数量 | 可学习题过少 → 池里没有「差一点就过」的题 | 补充中间难度题（从 `stable_fail` 里挑累计通过率最接近 0.2~0.5 的） | `bad` < 10；`warn` < 30；else `ok`（并按 `n_problems` 归一，见下） |
| `queue_wait_ratio`（**辅 · 吞吐杠杆**） | 排队挤占有效推理时间 | 沙箱排队时间占 attempt 总时长的比例高 → 排队**加剧**撞墙概率 | 扩容 / 提并发降低排队；**提高同等墙钟内的有效工作时间**，间接降低撞墙率。**【实测】** 分桶撞墙率 20.7% / 33.9% / 80.0% | `bad` ≥ 0.15；`warn` ≥ 0.08；else `ok` |
| `wasted_rollout_ratio` | 浪费算力组占比 | 含 abort 的组被整组丢弃 → rollout 预算白花 | 修时间预算 / 修基建（先看 `time_budget_ratio`）；**不要**动题池 | `bad` ≥ 0.50；`warn` ≥ 0.25；else `ok` |

#### B.4.3 阈值推导依据（逐条，必须可追溯）

**`time_budget_ratio`** — 实证：
- task8 `rollout/outcome/time_budget_exceeded`：0.000 → 0.0938 → **0.2266**（末），峰值 **0.6250**。
- task11：0.000 → **0.406**（末），峰值 0.578。
- task9（成功）：0.016 → 0.125，峰值 0.125。
- task2：全程 0.000 ~ 0.094。
> **阈值推导**：`bad ≥ 0.30` 取在 task11 末值 0.406 与 task8 末值 0.227 之间 —— 即「已接近失败任务末期水平」判 bad；`warn ≥ 0.10` 覆盖 task9 峰值 0.125（成功任务也曾触及），但成功任务的**均值**远低于 0.10。**注意 task9 峰值 0.125 会触发 warn**，这是可接受的保守告警（提示「曾撞过墙」），但不应触发 bad。

**`per_turn_seconds_growth`** — 实证（简报 §2.2）：
- task8：单轮生成耗时 **18.7s → 64.3s = +244%**。
- task9（成功对照）：22.2s → 31.2s = **+40%**。
> **阈值推导**：`bad ≥ +100%` 落在 task8(+244%) 与 task9(+40%) 之间，且有足够余量（task8 是 2.44 倍，bad 阈值取 1 倍即已捕获）；`warn ≥ +50%` 略高于 task9 的 +40%（成功任务不告警），低于 task8 的 +244%。
> **数据来源（本次新发现）**：单轮耗时可用 TB 的 `perf/rollout_time/attempt/agent_run/model_generation` ÷ `rollout/selected/attempt/completed_turns/p50` 估算，**不必**读 `.pt`。简报 §2.2 的 18.7s→64.3s 来自 `.pt` 的 `phase_time/agent_run` ÷ `turns`；TB 版是近似但零成本。

**`gradient_density`** — 实证（简报 §3 表 + 本次实测 slime `mixed_count`）：
- task4：2.3%；task8/task11：25%；task9：50%。
> **阈值推导**：`bad < 0.15` 低于 task8/task11 的 25%（失败任务）；`warn < 0.30` 低于 task9 的 50%（成功任务）但高于 task8/task11。**目标值 50%** 来自计划 §4.2「目标是每步至少 50% 的组是 mixed」。
> **注意口径修正**：简报 §3 的「梯度密度」用的是**工具侧** mixed 占比；本次实测 slime 侧 task8 全程 mixed_count 在 0~13 之间波动（step0=13/16=81%，step59=0）。为与 slime 对账，`gradient_density` 应以 §A 的 `classify_grpo_group` 口径计算（默认 `count_timeout_as_reward=True`）。

**`n_gradient_problems`** — 实证：
- task9：32 题、`gradient` 态为主 → 可学习题 ≈ 24（按 77.6% mixed 比例推）。
- task4：688 题、`gradient=424`（本次实测）。
> **阈值推导**：绝对数随题池规模变化，必须**归一化**。设计：`n_gradient_problems` 判 bad 的条件改为 `n_gradient < max(10, 0.05 * n_problems)`；warn 为 `n_gradient < max(30, 0.15 * n_problems)`。
> - task9：`max(10, 1.6)=10` → 24 > 10 → ok ✓
> - task4：`max(10, 34.4)=34.4` → 424 > 34.4 → ok ✓
> - 一个 32 题且全是 `stable_fail` 的任务：`n_gradient=0 < 10` → bad ✓
> **设计假设标注**：5% / 15% 这两个比例无直接实证支撑（实测样本只有 task4/task8/task9/task11，且都没有「gradient 题过少」的病例），标注为**设计假设**。

**`queue_wait_ratio`** — 实证（本次新发现，§0.4）：
- task8：`perf/rollout_time_pct/attempt_total/sandbox_boot_queue` 8.86% → 10.39%；`perf/rollout_time/attempt/sandbox_boot_queue` 152.07s → **392.01s**。
- task9（成功）：4.97% 末值（峰值 30.02%，单点尖刺）。
- task11：10.77% 末值（峰值 21.01%）。
- task4 / task2：全程 **0.000**（无排队，这些 run 走的是别的执行路径）。
> **阈值推导**：`bad ≥ 0.15` 高于 task8 末值 10.39% 与 task11 末值 10.77%，但 task11 峰值 21.01% 会触发 bad —— 符合直觉（task11 确实失败）。`warn ≥ 0.08` 覆盖 task8 末值 10.39% / task11 末值 10.77%（两个失败任务的末态）且高于 task9 末值 4.97%（成功任务）。
> **口径**：`queue_wait_ratio = perf/rollout_time_pct/attempt_total/sandbox_boot_queue`（TB 已是百分比，无需再除）。
> **单位与文案纪律**：这是**占 attempt 总时长**的比例。**文案必须写「排队挤占有效推理时间」，
> 不写「排队吃掉了 agent 预算」** —— 排队确实加剧撞墙，但**不是第一主因**。
>
> **【实测 · 二次修订补充的因果证据】** 按 `sandbox_boot_queue` 分桶的撞墙率：
>
> | 分桶 | rollout 55（n=128） | 多 rollout 汇总（ro 20/30/40/50/55/60/63, n=897） |
> | --- | --- | --- |
> | `q < 100s` | 40.6% | **20.7%** |
> | `100s ≤ q < 250s` | 52.8% | **33.9%** |
> | `q ≥ 250s` | **80.0%** | 30.0% |
>
> → 排队**单调加剧**撞墙概率，但**解释不了**从 0% → 62.5% 的整体跃升
> （撞墙样本 `agent_run` 100% ≥3600s，说明根因是 agent 自身跑不完）。
> → 因此 `queue_wait_ratio` 是**辅 · 吞吐杠杆**，排在 `time_budget_ratio` 之后。
>
> **【实现护栏 · 重要】** 读 `.pt` 时用 `phase_time/sandbox_boot_queue`，
> **不要**误用 `phase_time/sandbox_boot`（后者 p50 仅 5–10s，差一个数量级）——
> 本次曾因误读该字段两次得出相反结论。


**`wasted_rollout_ratio`** — 实证：
- task8 rollout 59：**100%**（16/16 组含 abort）。
- task9：`unknown_group_count` 1→2（峰值 15），远低于 task8。
> **阈值推导**：`bad ≥ 0.50` 对齐 task8 中后段的 75%/93.8%/100%；`warn ≥ 0.25` 覆盖 task8 早期的 25%(r14)。task9 峰值 15/16=93.75% 会被判 bad —— **这是已知的保守告警**，需在文案中说明「含 abort 组占比高，先看是 agent 侧超时还是基建故障」，避免误导。

#### B.4.4 `values` 字典增量

```python
values = {
    # ---- 已有，保持不变 ----
    "learnable_ratio": ..., "n_mixed": ..., "n_groups": ...,
    "measurable_pass_rate": ..., "timeout_ratio": ..., "abort_ratio": ...,
    "raw_mean_reward": ..., "n_attempts": ...,
    "saturated_pool_ratio": ..., "hard_pool_ratio": ...,
    "n_problems": ..., "n_saturated": ..., "n_hard": ...,
    # ---- 新增 ----
    "gradient_density": ...,          # == learnable_ratio（别名，新名）
    "zero_gradient_ratio": ...,
    "wasted_rollout_ratio": ...,
    "n_dropped": ...,
    "n_gradient_problems": ...,
    "n_stable_pass": ..., "n_stable_fail": ...,
    "insufficient_data_ratio": ...,
    "time_budget_ratio": ...,         # 来自 TB / .pt
    "per_turn_seconds_growth": ...,   # 来自 TB
    "queue_wait_ratio": ...,           # 来自 TB（辅 · 吞吐杠杆）
    "infra_overhead_ratio": ...,       # 来自 .pt（成本/吞吐，不进奖励杠杆）
}
```

> **别名策略**：`learnable_ratio` 与 `gradient_density` 同值，`saturated_pool_ratio` 与 `stable_pass_ratio` 同值。保留旧名 → 旧 UI / 旧测试 / `cross_task_compare_service.summarize_reward_guidance` **零改动**即可继续工作。

#### B.4.5 `_lever_status` 扩展

现有 `_lever_status(key, value, gradient, split, pool)` 签名保持，扩展为：

```python
def _lever_status(
    key: str,
    value: Optional[float],
    gradient: Dict[str, Any],
    split: Dict[str, Any],
    pool: Optional[Dict[str, Any]] = None,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
```

- `extra` 承载新杠杆所需的附加上下文（如 `n_problems` 用于 `n_gradient_problems` 归一化）。
- 新 key 的判定分支见 B.4.2 表格。
- `value is None` → `"unknown"`（现有行为保持）。

#### B.4.6 杠杆数据来源与分层（关键：哪些能纯 DB 算，哪些必须读 TB）

| 杠杆 | 数据来源 | 计算位置 |
| --- | --- | --- |
| `gradient_density` / `zero_gradient_ratio` / `wasted_rollout_ratio` | DB attempt rows | `reward_diagnosis_service`（纯 DB） |
| `n_gradient_problems` / 题目池三态 | DB attempt rows | `reward_diagnosis_service` + `utils/difficulty_profile`（纯 DB） |
| `time_budget_ratio` | TB `rollout/outcome/time_budget_exceeded` **或** `.pt` `agent_outcome` | 优先 TB；无 TB 时回退 `.pt` |
| `per_turn_seconds_growth` | TB `perf/rollout_time/attempt/agent_run/model_generation` + `rollout/selected/attempt/completed_turns/p50` | 需 TB |
| `queue_wait_ratio` | TB `perf/rollout_time_pct/attempt_total/sandbox_boot_queue` | 需 TB（辅 · 吞吐杠杆） |

**分层决策**：
- `build_reward_levers(rows)` 当前是**纯函数**（rows → levers），被 `cross_task_compare_service` 与 `reward_diagnosis_controller` 共用。
- **不破坏纯函数性**的做法：新增一个**可选**参数 `extra: Optional[Dict[str, Any]] = None`，由调用方（`build_reward_diagnosis`）注入 TB 派生的值。`build_reward_levers(rows)` 在无 `extra` 时，TB 类杠杆的 `value` 为 `None` → `status = "unknown"` → **不破坏现有行为、不破坏现有测试**（`tests/test_reward_levers.py` 的 `test_empty_rows_gives_unknown_status` 依赖这个语义）。
- `build_reward_diagnosis(task_id)` 内部负责读 TB（复用 `service/train_curve_service.resolve_event_dir`）并组装 `extra`。
- **`cross_task_compare_service.summarize_reward_guidance`** 保持不读 TB（避免跨任务对比 N 次读 TB）→ 其杠杆里 TB 类为 `unknown`，可接受（并可在文案里标注「需单任务页查看」）。

### B.5 `build_reward_diagnosis` 输出增量

```jsonc
{
  "task": {...},                    // 已有
  "n_attempts": 8192,               // 已有
  "turn_coverage": {...},           // 已有
  "gradient_groups": {...},         // 已有（内容按 B.2 扩展）
  "reward_split": {...},            // 已有
  "behavior": {...},                // 已有
  "levers": {...},                  // 已有（内容按 B.4 扩展）

  // ---- 新增 ----
  "train_health": {                 // 来自 service/train_health_service.build_train_health
    "removed": {...}, "training": {...}, "behavior": {...},
    "verdicts": [...], "exclusions": [...]
  },
  "budget_profile": {               // 来自 utils/agent_budget_profile（其它子任务交付）
    "verdict": "per_turn_slowdown", "per_turn_growth": 2.44, ...
  },
  "pool_health": {...},             // 已有（在 levers 内，此处提升为顶层，便于 UI 直取）
  "reconciliation": {               // 新增：与 slime TB 的对账
    "available": true,
    "tool_gradient_density": 0.1875,
    "slime_mixed_fraction": 0.8125,
    "abs_diff": 0.625,
    "note": "工具侧与 slime 侧口径差异见 §A.3.2 count_timeout_as_reward"
  }
}
```

> `reconciliation` 是**新功能**，直接回应计划 §7 验收标准第 2、3 条，且用本次实测发现的两组真源（`shaped_group_reward/*` 与 `passrate/pass@8`）做锚点。

### B.6 `summarize_reward_split` 的小修订（不改口径，只加字段）

现有 `measurable_pass_rate` 分母排除 abort/timeout —— **保持不变**（向后兼容，且这是 §E `pass_rate_gradable` 的基础）。
新增（可选，不影响旧字段）：
- `n_dropped_groups` / `wasted_rollout_ratio`（与 §B.2 同源，但放在 split 里便于对齐 TB `removed_ratio`）。
- `removed_ratio_estimate` = 1 - (trainable_attempts / total_attempts)，用于与 TB `rollout/removed_ratio` 对账。
> **实测对账锚点**：task8 rollout 59 的 DB 侧 `removed_attempt_count` ≈ 81（TB `rollout/population/removed_attempt_count` 末值 81），`removed_attempt_ratio` 0.633 → 与 TB `rollout/removed_ratio` 0.628 **一致**（差 0.005）。可作对账规则。

### B.7 向后兼容影响（逐项）

| 项 | 影响 | 处理 |
| --- | --- | --- |
| `classify_group_kind` 返回值域 | `infra_only` → `dropped` | **保留函数名与位置**；返回 `dropped`；`unknown` 新增。`infra_only` 字面量在返回值中消失 → 需同步 §F 测试 |
| `summarize_gradient_groups` 输出 | 新增字段 | 纯增量，旧字段全保留；`learnable_ratio` 语义不变 |
| `summarize_pool_health` 输出 | `saturated_pool_ratio` / `hard_pool_ratio` 语义**从「末次口径」变为「累计口径」** | **这是行为变更**（数值会变）。字段名保留 → 旧 UI 不用改，但数值含义变了。必须在 `note` 里写明，并在 §F 列出「口径变更」用例 |
| `build_reward_levers` 输出 | 新增杠杆 | 纯增量；`values` 新增键；旧 7 个 lever 的 `key` 不变 |
| `_LEVER_RULES` 元组长度 4→5 | 内部结构变化 | 私有常量，外部不可见 → 安全 |
| `build_reward_levers(rows)` 签名 | 加 `extra` 关键字参数（**默认值**） | 旧调用 `build_reward_levers(rows)` 仍有效 |
| `cross_task_compare_service.summarize_reward_guidance` | 读 `levers["values"]` 的旧键 | 旧键保留 → 零改动。若要加新键，追加即可 |
| `templates/reward_diagnosis.html` | 杠杆卡片数增加 | 需在 `RATIO_LEVER_KEYS` 集合里加新比例型 key（`gradient_density` / `zero_gradient_ratio` / `wasted_rollout_ratio` / `time_budget_ratio` / `queue_wait_ratio`），否则会按 `num()` 渲染成小数 |

### B.8 风险与边界

1. **口径分歧风险（最大）**：工具侧组判定与 slime 侧必然有偏差（本次实测 rollout 0 为 8 vs 13）。设计对策：(a) 暴露 `count_timeout_as_reward` 开关；(b) 输出 `reconciliation` 显式对账；(c) 绝不把工具侧 `gradient_density` 说成「slime 的梯度密度」。
2. **组归属依赖 `group_index` 正确性**：本次实测 task8 的 `group_index` 是**全局递增**（rollout 0 → 0~15，rollout 1 → 16~31，rollout 2 → 32~47）。所有按 `(rollout_id, group_index)` 分组的逻辑都**必须同时用 rollout_id**，只按 group_index 会跨 rollout 串组。现有 `_groups_by_rollout` 已是 `(rollout_id, group_index)` 双键 ✓。
3. **abort 样本的 reward**：实测 DB 中 `aborted` 的 reward 被写成 `0.0`（1589 条），而 `.pt` 中对应 `episode_reward` 是 `None`。**必须用 `status` 判定 abort，不能用 reward==0 判定**（否则 abort 会被误判为 all_fail）。现有 `DBAttemptService._classify_attempt_row` 优先看 status ✓。
4. **`truncated` 带 reward**：实测 236 条 truncated+reward=1.0、346 条 truncated+reward=0.0（task8）。这是 `count_timeout_as_reward=True` 的实证依据，也说明**旧工具口径把 truncated 全排除会丢掉真实信号**。
5. **阈值是 task8/task9/task4/task11 四任务的经验值**，样本量小；新任务可能需要调。设计要求：所有阈值集中在模块级常量 + 注释里写「实证来源任务」，便于后续调参。

---

## §C `utils/tb_metric_catalog.py` 扩展设计

### C.1 目标与依据

**目标**：把 TB 里实际存在的 458 个 tag 纳入目录，让工具能解释「训练侧在发生什么」。

**依据**：
- 简报 §3.1：扩充 tag 目录，「每条标注 slime 源码出处，保持『不编造含义』原则」。
- 现有 docstring：「只收录代码里能对上的评测 tag，其余一律『未收录』。**不给 loss / kl 等训练曲线编造含义**。」
- 本次实测：task8 的 `tf_logs` 有 **458** 个 tag，与简报一致；其中包含简报**未提及**的三大族：
  - `rollout/generated/group_reward/*` 与 `rollout/generated/shaped_group_reward/*`（**31 × 2 = 62 个**，slime 组分类真源）
  - `rollout/generated/attempt/*`（**55 个**，attempt 级行为画像，与 `rollout/selected/attempt/*` 同构）
  - `perf/rollout_time*/**`（**约 100 个**，含 `attempt/sandbox_boot_queue`、`barrier/group_sibling_wait`）
  - `passrate/pass@{1,2,4,8}`（4 个）
- 本次实测确认简报 §5 的 tag 清单**不完整**（缺 `rollout/generated/*`、`rollout/selected/*` 的 `raw_reward_*` 家族、`perf/*`、`passrate/*`、`train/global_batch_size` 等）。

### C.2 目录结构设计

#### C.2.1 目录对象

```python
@dataclass(frozen=True)
class TbTagInfo:
    """一个 TB scalar tag 的目录条目。"""
    tag: str
    kind: str            # 分类 kind（见 C.3）
    label: str           # 中文短名（≤ 12 字，用于 UI 标签）
    desc: str            # 中文说明（一句话）
    usage: str           # 诊断用途（这条指标能回答什么问题）
    source: str          # 出处：slime 源码路径，或 "推断：<依据>"
    inferred: bool       # True 表示「靠前缀/语义推断，无源码对应」
```

> `source` 与 `inferred` 两个字段共同满足简报「能对上源码的注明出处；靠前缀推断的**明确标注「推断」**」。

#### C.2.2 目录内容物

```python
TB_TAG_CATALOG: Dict[str, TbTagInfo] = { ... }   # 精确 tag → 条目
TB_TAG_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = ...  # 动态 tag（/{p50,p95} 等）→ kind
```

**动态 tag 处理**：实测 `rollout/selected/attempt/{completed_turns,response_tokens,final_context_tokens,final_prompt_tokens,peak_context_tokens}/{p50,p95,mean,max,coverage,attempt_count,measured_count,missing_count,incomplete_count,incomplete_lower_bound_sum,order_ambiguous_count}` 共 11 个后缀 × 5 个指标 × 3 个族（`generated` / `selected`）= **110+ 个 tag**。这些**不可能**逐个硬编码 → 必须用**正则 + 模板**生成，并在 `describe_tb_tag` 时**动态构造**条目。

```python
_ATTEMPT_METRIC_SUFFIXES = {
    "completed_turns": "完成轮数",
    "response_tokens": "响应 token 数",
    "final_context_tokens": "最终上下文 token",
    "final_prompt_tokens": "最终 prompt token",
    "peak_context_tokens": "峰值上下文 token",
}
_ATTEMPT_STAT_SUFFIXES = {
    "p50": "中位数", "p95": "95 分位", "mean": "均值", "max": "最大值",
    "coverage": "覆盖率", "attempt_count": "attempt 数", "measured_count": "已测量数",
    "missing_count": "缺失数", "incomplete_count": "不完整数",
    "incomplete_lower_bound_sum": "不完整下界和", "order_ambiguous_count": "顺序模糊数",
}
```

#### C.2.3 完整 tag 目录（按简报 §5 分组 + 本次实测补充）

标注约定：`[源码]` = 有 slime 语义依据；`[推断]` = 靠前缀/命名推断。

##### C.2.3.1 `rollout/removed/*` —— 丢弃原因全集（**第一主因指标区**）

| tag | kind | 中文说明 | 诊断用途 | source |
| --- | --- | --- | --- | --- |
| `rollout/removed_ratio` | `removed_ratio` | **被移除样本占比** | **第一主因指标**。task4 末值 1.0 = 整步无样本；task8 0.391→0.628 | [源码] slime 的移除统计汇总 |
| `rollout/removed/count_zero_std_0.0` | `removed_reason` | 全挂（零方差 0.0）被移除的 **attempt** 数 | 零梯度·题未掌握的规模 | [源码] |
| `rollout/removed/count_zero_std_1.0` | `removed_reason` | 全过（零方差 1.0）被移除的 **attempt** 数 | 零梯度·题已掌握的规模 | [源码] |
| `rollout/removed/count_ungradable` | `removed_reason` | 不可评分被移除数 | 评测器侧问题（实测 task4 有 203 例 `evaluator:invalid_evaluator_payload`） | [源码] |
| `rollout/removed/count_truncated` | `removed_reason` | 截断被移除数 | 上下文打满 / 截断的规模 | [源码] |
| `rollout/removed/count_agent_timeout` | `removed_reason` | **agent 超时被移除数** | **本批任务第一主因**（task8 2→29、task11 →52） | [源码] |
| `rollout/removed/count_too_few_survivors_{1,2,3}` | `removed_reason` | 组内存活数不足 K 被移除数 | 组污染放大（abort 拖垮整组） | [源码] |
| `rollout/removed/count_aborted_openhands_infra_exit_2` | `removed_reason` | OpenHands 退出码 2 被移除数 | 基建故障 | [源码] |
| `rollout/removed/count_aborted_openhands_infra_exit_137` | `removed_reason` | OpenHands 退出码 137（SIGKILL）被移除数 | 常为 OOM / 被强杀 | [源码] |
| `rollout/removed/count_aborted_exception_SandboxApiException` | `removed_reason` | 沙箱 API 异常被移除数 | 基建（实测 task8 仅 7 例） | [源码] |
| `rollout/removed/count_aborted_exception_SandboxExecStalled` | `removed_reason` | 沙箱执行卡死被移除数 | 基建（实测 task8 仅 5 例） | [源码] |
| `rollout/removed/count_aborted_exception_RuntimeError` | `removed_reason` | 运行时异常被移除数 | 基建 | [源码] |
| `rollout/removed/count_aborted_evaluator_total_timeout` | `removed_reason` | 评测器总超时被移除数 | 评测器侧（实测 task8 仅 2 例） | [源码] |

> **单位警示（本次实测发现）**：`removed/count_*` 是 **attempt 级**；`rollout/zero_std/count_*` 是 **group 级**。描述文案必须写明，否则必然误用（简报 §3.2 曾混淆）。

##### C.2.3.2 `rollout/zero_std/*` —— 组级零梯度计数

| tag | kind | 中文说明 | 诊断用途 | source |
| --- | --- | --- | --- | --- |
| `rollout/zero_std/count_0.0` | `group_count` | **组级**全挂组数 | 与 `removed/count_zero_std_0.0`（attempt 级）区分 | [源码] |
| `rollout/zero_std/count_1.0` | `group_count` | **组级**全过组数 | 同上 | [源码] |

##### C.2.3.3 `rollout/generated/{group_reward,shaped_group_reward}/*` —— **slime 组分类真源**（本次实测新增，简报未列）

两族同构（各 31 个 tag），`shaped_` 前缀为**经奖励整形后**的口径。

| tag（族内相对路径） | kind | 中文说明 | 诊断用途 | source |
| --- | --- | --- | --- | --- |
| `{group_reward,shaped_group_reward}/group_count` | `group_count` | 本步组数 | 分母（实测恒为 16） | [源码] |
| `.../mixed_count` | `group_count` | 真梯度组数（有对有挂） | **梯度密度分子** | [源码] |
| `.../all_one_count` | `group_count` | 全过组数 | 零梯度·题已掌握 | [源码] |
| `.../all_zero_count` | `group_count` | 全挂组数 | 零梯度·题未掌握 | [源码] |
| `.../unknown_count` | `group_count` | **不可分类组数（含 abort / 存活不足）** | **浪费算力组数** | [源码] |
| `.../mixed_fraction` | `group_ratio` | 真梯度组占比 | 与工具 `gradient_density` 对账 | [源码] |
| `.../all_one_fraction` / `all_zero_fraction` | `group_ratio` | 对应占比 | 同上 | [源码] |
| `.../classified_group_count` | `group_count` | 已分类组数 | 分类覆盖率 | [源码] |
| `.../unclassified_group_count` | `group_count` | 未分类组数 | 同 `unknown` | [源码] |
| `.../classified_fraction` | `group_ratio` | 分类覆盖率 | 100% → 无不可分类组 | [源码] |
| `.../incomplete_group_count` | `group_count` | 不完整组数 | 实测 task8 全程 0 | [源码] |
| `.../inconsistent_group_count` | `group_count` | 不一致组数 | 实测 task8 全程 0 | [源码] |
| `.../nonbinary_group_count` | `group_count` | 非二值组数 | 实测 task8 全程 0（reward 只 0/1） | [源码] |
| `.../pass_count_{0..8}_of_8_count` / `_fraction` | `group_histogram` | 组内通过数分布（0/8…8/8） | **直方图**：一眼看出组的通过数分布 | [源码] |

> `pass_count_K_of_8` 的分母 8 = 实测组大小（`groups_per_step=16` 组，每组 8 attempt，共 128 attempt/步，与 `rollout/outcome/attempt_count`=128 吻合）。**8 是本批配置**，其它 run 可能不同 → 描述应写「组内通过数分布」而非硬编码 8。

##### C.2.3.4 `rollout/population/*` —— 训练样本供给（样本枯竭链条）

| tag | kind | 中文说明 | 诊断用途 | source |
| --- | --- | --- | --- | --- |
| `rollout/population/sample_count` | `population` | 样本数 | 供给总量 | [源码] |
| `rollout/population/attempt_count` | `population` | attempt 数 | 供给总量 | [源码] |
| `rollout/population/trainable_attempt_count` | `population` | **进入训练的 attempt 数** | **枯竭链条核心**（task11 95→15；task4 38→0） | [源码] |
| `rollout/population/trainable_sample_count` | `population` | 进入训练的样本数 | 同上 | [源码] |
| `rollout/population/trainable_token_count` | `population` | 进入训练的 token 数 | 同上 | [源码] |
| `rollout/population/removed_attempt_count` | `population` | 被移除 attempt 数 | 与 `removed_ratio` 对账 | [源码] |
| `rollout/population/removed_attempt_ratio` | `population` | 被移除 attempt 占比 | 与 `rollout/removed_ratio` 对账（实测 0.633 vs 0.628） | [源码] |
| `rollout/population/removed_sample_count` | `population` | 被移除样本数 | — | [源码] |
| `rollout/population/removed_token_ratio` | `population` | 被移除 token 占比 | 算力浪费的 token 视角 | [源码] |
| `rollout/population/shaped_reward_coverage` | `population` | **有可用整形奖励的占比** | 枯竭链条（task4 0.984→**0**；task11 0.96→0.43） | [源码] |
| `rollout/population/shaped_reward_unavailable_attempt_count` | `population` | **奖励不可得的 attempt 数** | 枯竭链条（task11 5→73） | [源码] |
| `rollout/population/shaped_reward_measured_attempt_count` | `population` | 已测得整形奖励的 attempt 数 | — | [源码] |
| `rollout/population/shaped_reward` | `population` | 整形奖励均值 | 实际进训练的奖励 | [源码] |
| `rollout/population/raw_reward` / `raw_reward_coverage` / `raw_reward_measured_attempt_count` / `raw_reward_unavailable_attempt_count` | `population` | 原始奖励及其覆盖率 | 与整形奖励对比，识别整形是否失真 | [源码] |
| `rollout/population/paired_raw_reward` / `paired_reward_attempt_count` / `paired_reward_coverage` / `paired_reward_unavailable_attempt_count` | `population` | 配对去偏后的奖励家族 | 配对基线（去位置偏置） | [源码] |
| `rollout/population/reward_attempt_count` | `population` | 有奖励的 attempt 数 | — | [源码] |
| `rollout/population/token_count` | `population` | 本步总 token | 算力消耗 | [源码] |
| `rollout/population/total_lengths` | `population` | 总长度 | — | [源码] |
| `rollout/population/response_lengths` | `population` | 响应长度 | 输出膨胀 | [源码] |
| `rollout/population/truncated` | `population` | 截断占比 | 上下文打满 | [源码] |

##### C.2.3.5 `rollout/selected/*` 与 `rollout/generated/*` —— 行为画像

| tag 族 | kind | 中文说明 | 诊断用途 | source |
| --- | --- | --- | --- | --- |
| `rollout/selected/attempt/{metric}/{stat}` | `behavior_metric` | **进入训练**的 attempt 的行为分布 | **行为退化可观测量**（本次退化核心） | [源码] |
| `rollout/generated/attempt/{metric}/{stat}` | `behavior_metric` | **全部生成**的 attempt 的行为分布 | 与 selected 对比 → 被保留样本的偏移 | [源码] |
| `rollout/selected/{raw_reward_coverage,raw_reward_measured_attempt_count,raw_reward_unavailable_attempt_count,reward_attempt_count,paired_*,shaped_reward,shaped_reward_coverage,...}` | `population` | 进入训练子集的奖励家族 | — | [源码] |

> **本批实测重点值（简报 §1 + 本次实测）**：
> - `rollout/selected/attempt/completed_turns/p50`：task8 62.5 → **40.5**（下降 → 不是轮数爆炸）
> - `rollout/selected/attempt/response_tokens/p50`：task8 23999 → **31697**（上升 → 每轮输出更多）
> - `rollout/selected/attempt/peak_context_tokens/p50`：task8 84811 → **65418**（下降 → 不是上下文打满）
> - 三者合起来 = **「单轮变慢 → 预算内跑不完」** 的直接 TB 证据（简报 §2.2 的机制判定）。
> - `peak_context_tokens/p95` 末值 97952，而实测上下文上限约 131070（`p95` 的 max = 131070.0）→ 接近但未打满。

##### C.2.3.6 `rollout/其它`

| tag | kind | 中文说明 | 诊断用途 | source |
| --- | --- | --- | --- | --- |
| `rollout/truncated_ratio` | `truncated_ratio` | 截断比例 | 与 `rollout/outcome/context_exhausted` 同值（实测 task8 两者末值都是 0.0000） | [源码] |
| `rollout/repetition_frac` | `quality` | 重复内容占比 | 退化循环 | [源码]（实测 task8 全程 0.0） |
| `rollout/prefix_cache_hit_rate` | `quality` | 前缀缓存命中率 | 推理成本（实测 0.9766→0.9629） | [源码] |
| `rollout/segments_per_attempt` | `quality` | 每 attempt 段数 | 多轮重建质量（实测 1.0→1.0078） | [源码] |
| `rollout/response_len/{max,mean,median,min}` | `quality` | 响应长度分布 | 输出膨胀 | [源码]（实测 mean 25505→27662） |
| `rollout/replay/{dropped_reply_share,faithful_prompt_share,fallback_prompts,segments_per_attempt,split_reply_share,token_coverage,token_coverage_fail,token_coverage_pass}` | `quality` | 多轮重建质量 | 重建是否失真（实测 dropped=0.0、faithful=1.0、coverage=1.0，均正常） | [源码] |

##### C.2.3.7 `rollout/outcome/*` —— agent 结束态分布（**本次实测新增族，简报未列**）

| tag | kind | 中文说明 | 诊断用途 | source |
| --- | --- | --- | --- | --- |
| `rollout/outcome/time_budget_exceeded` | `outcome` | **时间预算耗尽占比** | **本批第一主因**（task8 末 0.2266、峰值 0.625；task11 末 0.406） | [源码] |
| `rollout/outcome/context_exhausted` | `outcome` | 上下文打满占比 | 与 time_budget **必须分开**（简报 §2.3） | [源码] |
| `rollout/outcome/finished` | `outcome` | 正常结束占比 | — | [源码] |
| `rollout/outcome/finished_without_output` | `outcome` | 结束但无产出 | — | [源码] |
| `rollout/outcome/sdk_error` | `outcome` | SDK 报错占比 | 基建 | [源码] |
| `rollout/outcome/max_fake_responses` | `outcome` | 假响应超限占比 | 自欺 | [源码] |
| `rollout/outcome/max_iterations_reached` | `outcome` | 迭代超限占比 | 轮次爆掉 | [源码] |
| `rollout/outcome/turn_limit_reached` | `outcome` | 轮次上限占比 | 同上 | [源码] |
| `rollout/outcome/plain_response` / `stuck` / `unknown` | `outcome` | 其它结束态 | — | [源码] |
| `rollout/outcome/attempt_count` | `outcome` | 本步 attempt 总数 | 分母（实测恒为 128 = 16 组 × 8） | [源码] |

##### C.2.3.8 `train/*` —— 训练侧（全部 15 个）

| tag | kind | 中文说明 | 诊断用途 | source |
| --- | --- | --- | --- | --- |
| `train/entropy_loss` | `train` | 策略熵 | **排除熵坍缩**：实测 task4/task8/task9 **上升** 0.15→0.64 | [源码] |
| `train/grad_norm` | `train` | 梯度范数 | 训练不稳信号（实测 task11 0.181→**0.986**） | [源码] |
| `train/loss` | `train` | 策略损失 | — | [源码]（实测 ~1e-4 量级） |
| `train/pg_loss` | `train` | 策略梯度损失 | — | [源码] |
| `train/ppo_kl` | `train` | PPO KL 散度 | **排除 KL 爆炸**：实测 ~1e-6 | [源码] |
| `train/pg_clipfrac` | `train` | PG 裁剪比例 | 裁剪过多少（实测 task8 0→0.043） | [源码] |
| `train/ois` | `train` | 重要性采样权重（OIS） | 偏离度（实测恒 1.0） | [源码] |
| `train/tis` / `train/tis_abs` / `train/tis_clipfrac` | `train` | 截断重要性采样 | 偏离度（实测 tis≈1.0、tis_abs 0.035→0.034） | [源码] |
| `train/train_rollout_logprob_abs_diff` | `train` | **训练 logprob 与 rollout logprob 的绝对差** | **排除 off-policy**：实测全程 0.017–0.045，远低于 slime CI 断言阈值 0.1 | [源码] |
| `train/lr-pg_{0,1,2}` | `train` | 各 param group 学习率 | — | [源码] |
| `train/global_batch_size` | `train` | 全局批大小 | 实测与 `trainable_attempt_count` **同值**（78→47、末值 47），是样本枯竭的另一视角 | [源码] |

##### C.2.3.9 `perf/*` —— 性能与排队（**本次实测新增族，简报未列，但对「排队偷走预算」至关重要**）

| tag 族 | kind | 中文说明 | 诊断用途 | source |
| --- | --- | --- | --- | --- |
| `perf/rollout_time/attempt/{total,agent_run,agent_run/model_generation,agent_run/tool_execution,cli_install,sandbox_boot,sandbox_boot_queue,evaluator,evaluator/boot_and_setup,evaluator/test_execution,diff_collection,runtime_prepare,workspace_prepare,adapter_finish,engine_affinity_wait}` | `perf_phase` | **attempt 各阶段耗时** | **预算分解**：哪一段吃掉了 60 分钟 | [源码] |
| `perf/rollout_time/barrier/{group_sibling_wait,rollout_batch_wait}` | `perf_phase` | **同组兄弟等待 / 批次等待** | 实测 task8 622→703s / 1567→1805s | [源码] |
| `perf/rollout_time_max/attempt/**` | `perf_phase` | 各阶段**最大**耗时 | 实测 `agent_run` max = **3903.24s**（>3600s 墙钟，与简报 §2.1 的 3903s 精确吻合！） | [源码] |
| `perf/rollout_time_pct/{attempt_total,rollout_time}/**` | `perf_phase` | 各阶段**占比** | **queue_wait_ratio 直接取值** | [源码] |
| `perf/{rollout_time,train_time,save_model_time,update_weights_time,data_preprocess_time,log_probs_time,step_time,train_wait_time}` | `perf_phase` | 各阶段总耗时 | — | [源码] |
| `perf/wait_time_ratio` | `perf_phase` | 等待时间占比 | 实测 task8 0.82→**0.94** | [源码] |
| `perf/{actor_train_tflops,log_probs_tflops,e2e_tokens_per_gpu_per_sec,effective_tokens_per_gpu_per_sec,tokens_per_gpu_per_sec,longest_*,actor_train_tok_per_s,actor_train_tok_per_s_per_gpu}` | `perf_phase` | 吞吐 | — | [源码] |

> **本次实测的重大印证**：简报 §2.1 说「1363 个 `time_budget_exceeded` 样本的 `agent_run` max = 3903s，100% ≥ 3600s，是精确的 60 分钟墙钟截断」。TB 侧的 `perf/rollout_time_max/attempt/agent_run` 末值 = **3903.24** —— 与简报**逐位吻合**。这条独立路径确认了 3600s 预算的存在，也说明 TB 侧足以支撑该判定（无需读 `.pt`）。

##### C.2.3.10 `passrate/*` —— 评测通过率（**本次实测新增族**）

| tag | kind | 中文说明 | 诊断用途 | source |
| --- | --- | --- | --- | --- |
| `passrate/pass@1` | `eval_score` | 单次通过率 | 端到端成功率（实测 task8 0.508→0.516） | [源码] |
| `passrate/pass@2` | `eval_score` | 2 次通过率 | — | [源码] |
| `passrate/pass@4` | `eval_score` | 4 次通过率 | — | [源码] |
| `passrate/pass@8` | `eval_score` | **8 次通过率（组级）** | **对账锚点**：本次实测 64/64 步 = (至少一个 reward=1 的组数) / 总组数 | [源码] |

> **口径警示**：`passrate/pass@8` 的分母是**总组数（含被丢弃组）**。实测 step59：(mixed 0 + all_one 0) / 16 = 0.0，但 TB 的 `passrate/pass@8` = 0.5625 —— 因为**被丢弃组里仍有 reward=1 的 attempt**（实测 rollout 59 的 `episode_reward` 有 25 个 1.0）。这正是 §A 的 `dropped` 组**不等于**零梯度的另一处硬证据。

##### C.2.3.11 `eval/*` —— 评测主分数（**已有，保持不变**）

`eval/{dataset}` 主分数、`eval/{dataset}-truncated_ratio`、`eval/{dataset}/attempt_count`。
> **必须保持现有行为**（见 C.5）。

### C.3 `classify_tb_tag` 的新 kind 集合

现有 4 类 → 扩展为 17 类：

```python
_KIND_PRIMARY        = "primary"          # 已有
_KIND_TRUNCATED      = "truncated_ratio"  # 已有
_KIND_ATTEMPT_COUNT  = "attempt_count"    # 已有
_KIND_UNKNOWN        = "unknown"          # 已有

# ---- 新增 ----
_KIND_REMOVED_RATIO  = "removed_ratio"    # rollout/removed_ratio
_KIND_REMOVED_REASON = "removed_reason"   # rollout/removed/count_*
_KIND_GROUP_COUNT    = "group_count"       # *group_count / *_count（组级）
_KIND_GROUP_RATIO    = "group_ratio"       # *_fraction
_KIND_GROUP_HISTOGRAM= "group_histogram"   # pass_count_K_of_N_*
_KIND_POPULATION     = "population"        # rollout/population/*
_KIND_BEHAVIOR_METRIC= "behavior_metric"   # rollout/{selected,generated}/attempt/*
_KIND_QUALITY        = "quality"           # rollout/replay/*, prefix_cache, repetition_frac...
_KIND_OUTCOME        = "outcome"           # rollout/outcome/*
_KIND_TRAIN          = "train"             # train/*
_KIND_PERF_PHASE     = "perf_phase"        # perf/*
_KIND_EVAL_SCORE     = "eval_score"        # passrate/*
```

**判定顺序（严格按序，先具体后通用）**：

```
1. eval/…-truncated_ratio          → truncated_ratio      （已有，顺序不变）
2. eval/…attempt_count             → attempt_count        （已有）
3. eval/{dataset}                  → primary              （已有，is_primary_eval_tag）
4. passrate/…                      → eval_score
5. train/…                         → train
6. perf/…                          → perf_phase
7. rollout/removed_ratio            → removed_ratio
8. rollout/removed/count_*         → removed_reason
9. rollout/zero_std/count_*        → group_count
10. rollout/{generated,selected}/{group_reward,shaped_group_reward}/… → group_count / group_ratio / group_histogram
11. rollout/{generated,selected,population}/…  → population
12. rollout/{generated,selected}/attempt/{metric}/{stat} → behavior_metric
13. rollout/outcome/*              → outcome
14. rollout/replay/*、rollout/prefix_cache_hit_rate、rollout/repetition_frac、rollout/segments_per_attempt、rollout/response_len/* → quality
15. rollout/truncated_ratio        → truncated_ratio
16. 其余 rollout/*                 → unknown
17. 其它根前缀                     → unknown
```

> **顺序关键点**：规则 3（primary）**必须在规则 4~17 之前**，但**必须在规则 1~2 之后** —— 现有 docstring 已说明「更具体的子指标先于主分数判断，避免 `eval/attempt_count` 被当成主分数」。扩展时**不得打破**这个既有顺序。

### C.4 未知 tag 的启发式兜底（标注「推断」）

C.3 已覆盖实测 458 个 tag，但未来 slime 升级会新增 tag。**设计启发式兜底**（**必须标注「推断」**）：

```python
def _infer_tb_tag(tag: str) -> TbTagInfo:
    """未收录 tag 的启发式兜底；结果一律标注 inferred=True。"""
```

**兜底规则表**：

| 命中条件 | 推断 kind | 推断 label | 推断 desc | 依据 |
| --- | --- | --- | --- | --- |
| 含 `removed` | `removed_reason` | 被移除计数 | 按命名推断为「被移除原因计数」，具体原因需对照 slime 源码 | [推断] 前缀语义 |
| 含 `zero_std` | `group_count` | 零标准组计数 | 按命名推断为组级零方差计数 | [推断] |
| 含 `group` | `group_count` | 组级计数 | — | [推断] |
| 含 `fraction` / `_ratio` / `_rate` | `group_ratio` | 占比 | — | [推断] |
| 含 `p50`/`p95`/`mean`/`max`/`min` | `behavior_metric` | 分布统计 | — | [推断] |
| 含 `count` / `_n` | `population` | 计数 | — | [推断] |
| 含 `time` / `duration` / `_latency` | `perf_phase` | 耗时 | — | [推断] |
| 含 `reward` / `loss` / `kl` | `train` | 训练/奖励指标 | **不给具体语义**，只标类别 | [推断] |
| 其它 | `unknown` | 未收录 | 「未收录」 | — |

> **红线（继承现有 docstring）**：兜底**只推断「类别」**，**不推断「具体含义」**。`desc` 必须写「按命名推断为 …，具体含义需对照 slime 源码确认」。**绝不**给 `train/loss` 编造「策略损失」这类语义 —— 除非它已在精确目录里（`train/loss` 已在 C.2.3.8 精确收录，此处不冲突）。

### C.5 `describe_tb_tag` 的输出格式扩展

**现有契约**：`describe_tb_tag(tag) -> str`，返回**中文说明**；未知返回 `"未收录"`（`DESC_UNKNOWN`）。
> **现有测试** `tests/test_tb_metric_catalog.py` 断言：
> - `describe_tb_tag("eval/gsm8k") == DESC_PRIMARY`
> - `describe_tb_tag("eval/gsm8k-truncated_ratio") == DESC_TRUNCATED`
> - `describe_tb_tag("eval/gsm8k/attempt_count") == DESC_ATTEMPT_COUNT`
> - **对 `train/loss`、`train/kl`、`actor/kl_loss`、`loss`、`eval/foo/bar`、`eval/foo-bar`、`train/gsm8k-truncated_ratio`、`train/attempt_count`、`""` 断言 `== DESC_UNKNOWN`**

**冲突预警（必须处理）**：现有测试把 `train/loss` 断言为「未收录」。本设计要把 `train/loss` 精确收录 → **该断言必然失败**。

**解决方案（二选一，须在实施时明确）**：

**方案 A（推荐）**：保留 `describe_tb_tag` 的**严格语义**（只对**精确目录**里的 tag 返回说明），把**启发式推断**放到**新函数** `describe_tb_tag_extended(tag) -> Dict[str, Any]`。

```python
def describe_tb_tag_extended(tag: str) -> Dict[str, Any]:
    """返回结构化的 tag 说明（含来源与推断标记）。

    Returns:
        {
          "tag": str,
          "kind": str,
          "label": str,
          "desc": str,
          "usage": str,
          "source": str,          # slime 源码路径 或 "推断：<依据>"
          "inferred": bool,
          "group": str,           # 目录分组（removed/population/behavior/training/perf/...）
        }
    """
```

- `describe_tb_tag(tag)` **保持原签名与原行为**（精确目录命中 → 返回其 `desc`；否则 `"未收录"`）。
  - **但**：`train/loss` 现在在精确目录里 → `describe_tb_tag("train/loss")` 会返回新说明 → **现有测试的该条断言会失败**。
- 兼容处理：`describe_tb_tag` 只对 **eval 家族**（`primary` / `truncated_ratio` / `attempt_count`）保持完全不变的行为；其余 kind 一律返回 `DESC_UNKNOWN`（= `"未收录"`）→ **现有测试全部通过**，新增的精确说明走 `describe_tb_tag_extended`。
- `service/insight_page_service._scalar_row` 改用 `describe_tb_tag_extended` 并把 `description` 取 `desc` → 洞察页受益，现有断言不破。

**方案 B**：直接扩展 `describe_tb_tag` 并修改现有测试断言。**不推荐** —— 会让「不给训练曲线编造含义」这条红线看起来被破坏，且影响面更大。

**结论：采用方案 A。** `describe_tb_tag` 契约冻结；新说明走 `describe_tb_tag_extended`。

### C.6 其它新增公开函数

```python
__all__ = [
    # ---- 已有（保留）----
    "DESC_ATTEMPT_COUNT", "DESC_PRIMARY", "DESC_TRUNCATED", "DESC_UNKNOWN",
    "classify_tb_tag", "describe_tb_tag",
    # ---- 新增 ----
    "TbTagInfo",
    "TB_TAG_CATALOG",
    "TB_TAG_PATTERNS",
    "TB_TAG_GROUPS",
    "describe_tb_tag_extended",
    "tb_tag_info",
    "catalog_size",
    "tags_in_group",
    "find_tags_by_prefix",
]

def tb_tag_info(tag: str) -> TbTagInfo:
    """取单个 tag 的目录条目（精确命中优先，其次正则模板，最后启发式推断）。"""

def catalog_size() -> int:
    """精确收录的 tag 条目数（用于测试与自检）。"""

def tags_in_group(group: str) -> List[str]:
    """按目录分组列出 tag（group ∈ TB_TAG_GROUPS）。"""

def find_tags_by_prefix(prefix: str) -> List[str]:
    """列出所有以 prefix 开头的已收录 tag（用于「rollout/removed/* 一把取全」）。"""
```

### C.7 复用关系

| 需求 | 复用 |
| --- | --- |
| eval 主分数识别 | `utils/eval_tb_scores.is_primary_eval_tag`（**不改**） |
| tag → TB 深链 | `utils/tb_deep_link.tensorboard_tag_filter` / `append_tag_filter`（不改） |
| 指标 key → tag 匹配 | `utils/train_curve_catalog.resolve_metric_tags` / `match_tag`（**不改**，新目录不替代它 —— 它管「画哪些曲线」，新目录管「这条曲线什么意思」） |
| 事件目录定位 | `utils/tb_run_paths.discover_tb_event_dir` / `resolve_run_dir`（不改） |

> **重要边界**：`tb_metric_catalog`（含义目录）与 `train_curve_catalog`（展示选择）是**两个正交能力**，**不得合并** —— 前者回答「这是什么」，后者回答「画不画」。

### C.8 向后兼容影响

| 项 | 影响 | 处理 |
| --- | --- | --- |
| `classify_tb_tag` 返回值域 | 从 4 值扩到 17 值 | **必须**检查所有消费方 |
| `service/insight_page_service._GROUP_ORDER` | `("primary","truncated_ratio","attempt_count","unknown")` | **不改** → 新 kind 落到 `unknown` 分组（`_group_rows` 已有 `if kind not in groups: kind = "unknown"` 兜底 ✓）。**可选增强**：把 `_GROUP_ORDER` 扩成完整分组顺序 |
| `utils/eval_tb_scores.is_primary_eval_tag` | 不改 | ✓ |
| `utils/eval_tb_scores.read_eval_scores_from_event_dir` | 不改 | ✓ |
| `utils/train_curve_catalog.match_tag` / `resolve_metric_tags` | 不改 | ✓ |
| `tests/test_tb_metric_catalog.py` | 全部断言仍通过（方案 A） | ✓ |

> **风险点**：若其它代码用 `classify_tb_tag(tag) == "unknown"` 来判断「这条 tag 没被识别」，扩展后语义会变（现在 `train/loss` 会是 `train` 而非 `unknown`）。**必须 Grep 全仓 `classify_tb_tag` 的消费方**再实施。已确认消费方只有 `service/insight_page_service._scalar_row`（`utils/tb_metric_catalog` 内部也用）。

### C.9 风险与边界

1. **458 个 tag 的收录工作量**：动态族（`{selected,generated}/attempt/{metric}/{stat}` 共 110+、`pass_count_K_of_8` 共 18×2、`perf/rollout_time*/**` 约 100）**必须用模板/正则生成**，否则维护不可行。设计强制要求这一点。
2. **出处标注的诚实性**：本次能确证「语义」的 tag 主要靠**数值对账**（如 `removed/count_zero_std_1.0` 与 `.pt` 的 `train_drop_reason` 逐 rollout 完全相等、`perf/rollout_time_max/attempt/agent_run` 末值 3903.24 与简报 §2.1 的 max 3903s 吻合）。**这不是源码级证明**。因此 `source` 字段应写「对 slime rollout 统计的对账」而非杜撰的源码路径；只有能确证的才写具体路径。**若无法确证，必须 `inferred=True`。**
3. **`removed/count_*` 与 `zero_std/count_*` 单位不同** —— 必须在 `desc` 里显式写「attempt 级 / group 级」。
4. **`pass_count_K_of_8` 的 8 是本批配置** —— 不硬编码进 desc。
5. **不同 run 的 tag 集不同**：task9 有 457 个（少 1 个），task4 有 458 个。目录必须容忍缺失（`tb_tag_info` 对不存在的 tag 返回 `unknown`）。

---

## §D `utils/tb_series_reader.py` 扩展 + `service/train_health_service.py` 设计

### D.1 `tb_series_reader` 扩展

#### D.1.1 现状与边界（明确「不要重复设计」）

现有（func.md 已登记）：
- `list_scalar_tags(event_dir) -> List[str]`
- `read_scalar_series(event_dir, *, tags=None, max_points=4000) -> Dict[str, List[{"step","value"}]]` —— **已能读全量序列** ✓（简报 §3.2 曾误以为只有 `read_latest_scalars`，本次已复核确认）
- `series_points(series, tag) -> List[Dict]`

**不新增**「读单点最新值」类函数（`service/insight_page_service.read_latest_scalars` 已存在，且用途不同：给洞察页取每个 tag 的最新值，不保留序列）。

#### D.1.2 增量 1：按前缀批量取 tag

```python
def read_scalar_series_by_prefix(
    event_dir: Optional[Path | str],
    prefix: str,
    *,
    max_points: int = _DEFAULT_MAX_POINTS,
) -> Dict[str, List[Dict[str, Any]]]:
    """读取所有以 ``prefix`` 开头的 tag 的完整序列。

    用途：``rollout/removed/*``（14 个 tag）、``train/*``（15 个）一把取全，
    避免调用方先 ``list_scalar_tags`` 再逐个过滤。
    """
```

**实现要点**：先 `list_scalar_tags(event_dir)` 过滤出前缀命中的，再复用 `read_scalar_series(event_dir, tags=..., max_points=...)`。
> **性能注意**：`read_scalar_series` 内部会 **new 一个 EventAccumulator 并 Reload 整个目录**。前缀批量取如果调用两次 = 两次全量 Reload。因此必须提供**单次调用取多前缀**的版本：

```python
def read_scalar_series_by_prefixes(
    event_dir: Optional[Path | str],
    prefixes: Sequence[str],
    *,
    max_points: int = _DEFAULT_MAX_POINTS,
) -> Dict[str, List[Dict[str, Any]]]:
    """一次读取多个前缀命中的 tag（**只 Reload 一次**事件目录）。"""
```

> 实测：458 个 tag 的目录 Reload 约 **1.3~2.3 秒**（本次实测每次 `read_scalar_series` 调用）。`build_train_health` 若要读 3 组曲线（约 150 个 tag），**必须**用 `read_scalar_series_by_prefixes` 一次读完，**否则会慢 3 倍**。这是硬性性能约束。

#### D.1.3 增量 2：sidecar 缓存（按 event_dir + mtime 失效）

**参考实现**：`utils/attempt_artifacts.py` 的增量刷新机制（本次已读全文）：
- `_SIDECAR_VERSION = 2`，sidecar 文件 `.ri_artifacts_index.json`；
- `_summaries_fingerprint(dir) -> (mtime, count)` 目录级指纹；
- `_read_sidecar` / `_write_sidecar`（**临时文件 + `os.replace` 原子替换**）；
- `_incremental_update` 按 `(stem, 内容指纹)` 比对，只重解析变更文件；
- 进程内 `_INDEX_CACHE` + `clear_artifact_cache_for_tests()`。

**`tb_series_reader` 的 sidecar 设计**：

```python
_SERIES_SIDECAR_NAME = ".ri_tb_series_index.json"
_SERIES_SIDECAR_VERSION = 1
```

- **落盘位置**：**事件目录本身**（`{event_dir}/.ri_tb_series_index.json`）。
  - 理由：与 `attempt_artifacts` 一致（sidecar 与被索引数据同目录，便于清理与拷贝）。
  - **风险**：事件目录可能是只读挂载 → 必须**降级为不缓存**（同 `attempt_artifacts._write_sidecar` 的处理：捕获 `OSError` 后 `logger.info` 并跳过）。
- **失效键**：`event_dir` 绝对路径 + **事件目录指纹**。

**事件目录指纹**（关键：TB 事件目录是**目录**，不能用单文件 mtime）：

```python
def _event_dir_fingerprint(event_dir: Path) -> Optional[Tuple[float, int, int]]:
    """事件目录指纹 = (最新事件文件 mtime, 事件文件数, 总字节数)。

    只 stat，不读内容。训练中 run 会持续追加新的 events.out.tfevents.* 文件，
    文件数与总字节数的变化都能反映「有新数据」。
    """
```

**指纹字段选择的依据**：
- `(mtime_max, n_files, total_bytes)` 三者组合：
  - 训练重启 → 新文件 → `n_files` 变 ✓
  - 同文件追加 → `mtime_max` 与 `total_bytes` 变 ✓
  - mtime 精度不足 → `total_bytes` 兜底 ✓
- **不使用** 目录 mtime 单独作键（目录 mtime 只在**条目增删**时变，**文件内容追加不变**）→ 会漏掉「训练持续写入但没新文件」的常见场景。**这是与 `attempt_artifacts` 的差异点**（那里是目录内文件增删，所以目录 mtime 够用；TB 是文件内容追加，目录 mtime 不够）。

- **缓存内容**：`{tag: [{"step","value"}]}` 全量点 + 元信息 `{"version", "fingerprint", "n_tags", "n_points", "built_at"}`。

```python
def read_scalar_series_cached(
    event_dir: Optional[Path | str],
    *,
    tags: Optional[Sequence[str]] = None,
    max_points: int = _DEFAULT_MAX_POINTS,
    use_cache: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """带 sidecar 缓存的序列读取。

    缓存命中条件：sidecar 存在 **且** 事件目录指纹未变 **且** 请求的 tag 是
    缓存的子集。任一不满足 → 重新读取并回写 sidecar。
    """
```

**缓存语义细则**：
- 缓存的是**全量 tag**（不是子集）→ 请求是子集时直接裁剪，避免「按需缓存」导致的缓存碎片。
- `max_points` 抽稀在**返回时**做，缓存里存**未抽稀**的全量点 → 避免不同 `max_points` 互相污染缓存。
- 缓存文件过大时（实测 458 tag × 65 点 ≈ 3 万点，JSON 约 3~5 MB）→ 超过 `_SERIES_CACHE_MAX_BYTES = 32 * 1024 * 1024` 则**不落盘**，只走进程内缓存。
- **进程内缓存**：复用 `_INDEX_CACHE` 风格的模块级 dict，键 = `str(event_dir)`，值 = `{"fingerprint": ..., "series": ...}`；提供 `clear_series_cache_for_tests()`（对齐 `attempt_artifacts.clear_artifact_cache_for_tests`）。

#### D.1.4 增量 3：基于**趋势 + 阈值**（非单点）的异常检测

```python
def find_anomalies(
    series: Mapping[str, Sequence[Dict[str, Any]]],
    rules: Sequence["AnomalyRule"],
) -> List[Dict[str, Any]]:
    """按「趋势 + 阈值」在序列里找异常。

    单点异常会被噪声淹没（本批实证：removed_ratio 单点最高 0.9453、
    shaped_reward_coverage 单点最低 0.3125，但真正有意义的是**持续趋势**）。
    """
```

**规则对象**：

```python
@dataclass(frozen=True)
class AnomalyRule:
    """一条趋势型异常规则。

    Attributes:
        code:        规则码（如 "removed_ratio_rising"）
        tag:         关注的 tag
        direction:   "rising" | "falling"
        threshold:   触发阈值（如 0.5）
        min_growth:  最小变化幅度（如 0.2），趋势必须超过它
        window:      取首尾各 window 个点做比较（0 = 用全序列）
        severity:    "warn" | "bad"
        title:       中文标题
        desc:        中文说明
    """
```

**判定算法（趋势 + 阈值，非单点）**：

```
pts = 序列点（按 step 升序）
if len(pts) < 2: 跳过
head = mean(pts[:window]) 或 pts[0]
tail = mean(pts[-window:]) 或 pts[-1]
delta = tail - head
若 direction == "rising" 且 delta >= min_growth 且 tail >= threshold → 命中
若 direction == "falling" 且 -delta >= min_growth 且 tail <= threshold → 命中
```

**关键设计点（与「单点检测」的本质区别）**：
1. **必须同时满足「趋势幅度」与「终点水平」两个条件** → 单点尖刺不会误报（task9 的 `sandbox_boot_queue` 峰值 30.02% 是单点尖刺，但末值 4.97% → 趋势不成立）。
2. 用**首尾均值**（`window` 个点的均值）而非单点 → 抗噪。
3. 输出 `{"code","tag","severity","title","desc","head_value","tail_value","delta","n_points","first_step","last_step"}` → 供 §D.2 的 verdicts 直接引用。

**内置规则集**（放在 `utils/tb_series_reader` 还是 `service/train_health_service`？）

> **决策：放 `service/train_health_service.py`。**
> 理由：`utils/` 应保持「纯技术、与业务无关」（conventions 明文）。异常规则含**业务语义**（什么算异常、给什么建议）→ 属 service 层。
> `utils/tb_series_reader` 只提供**通用机制**：`AnomalyRule` 数据类 + `find_anomalies(series, rules)` 纯函数。

### D.2 `service/train_health_service.py` 设计（新建）

> **为什么是新建 service**：简报 §3.3 / 计划 §5 明确要求 `service/train_health_service.py`。
> **为什么不是「平行 service」**：它是**新的数据源域**（TB 训练侧），与现有的「attempt 结局域」（`reward_diagnosis_service`）不重叠。`reward_diagnosis_service` 读 DB attempt 行；`train_health_service` 读 TB 事件目录。职责正交，且计划已批准该落点。

#### D.2.1 目标与依据

**目标**：把 458 个 TB tag 收敛成「三组曲线 + 结论 + 已排除病因」，让排查者**不必**再手工翻 TB。

**依据**（简报 §1 + §3.3 + 本次实测）：
- `exclusions` 是**刻意设计**（计划 §3.3）：把「已排除的可能病因」显式列出，避免每次排查都重新怀疑 KL 爆炸、熵坍缩这类常见但本例并非主因的问题。
- 三条排除性结论的**实测依据**：
  | 排除项 | 判据 | 实测值 | 阈值来源 |
  | --- | --- | --- | --- |
  | **off-policy / train-rollout 不一致** | `train/train_rollout_logprob_abs_diff` 全程 < 0.1 | task4 0.019–0.045；task8 0.017–0.044；task9 0.034 | slime CI 断言阈值 **0.1** |
  | **熵坍缩** | `train/entropy_loss` **上升**而非下降 | task4 0.207→0.640；task8 0.152→0.641；task9 0.147→0.577 | 熵坍缩 = 熵**下降**；实测**上升** → 排除 |
  | **KL 爆炸** | `train/ppo_kl` 量级 | 全部 ~1e-6 | 1e-6 量级远低于「爆炸」阈值 |
- `wasted_rollout_ratio` 比 `zero_gradient_ratio` 是更好的先行指标（本次实测 task8 rollout 59 = 100%）。

#### D.2.2 输出 schema

```python
def build_train_health(task_id: int) -> Dict[str, Any]:
    """组装任务的训练健康度（三组曲线 + 结论 + 已排除病因）。"""
```

```jsonc
{
  "task": {"id": 4, "task_name": "...", "rollout_root": "..."},
  "event_dir": "/.../tf_logs",
  "available": true,
  "notice": null,                 // 无事件目录时的说明，不抛异常

  // ---- 三组曲线 ----
  "removed": {
    "removed_ratio":      [{"step":0,"value":0.703}, ...],   // 第一主因
    "removed_attempt_ratio": [...],
    "removed_attempt_count": [...],
    "by_reason": {                              // 逐原因曲线
      "count_agent_timeout":      [...],
      "count_zero_std_1.0":      [...],
      "count_zero_std_0.0":      [...],
      "count_truncated":          [...],
      "count_ungradable":        [...],
      "count_too_few_survivors_1": [...],
      "count_aborted_openhands_infra_exit_2": [...]
    },
    "trainable_attempt_count":  [{"step":0,"value":38}, ...],
    "shaped_reward_coverage":   [{"step":0,"value":0.984}, ...],
    "shaped_reward_unavailable_attempt_count": [...],
    "removed_ratio_summary": {
      "first": 0.703, "last": 1.0, "delta": 0.297,
      "max": 1.0, "min": 0.703,
      "verdict": "saturated"          // ok | rising | saturated
    }
  },

  "training": {
    "entropy_loss":                [...],
    "grad_norm":                   [...],
    "ppo_kl":                      [...],
    "train_rollout_logprob_abs_diff": [...],
    "pg_clipfrac":                 [...],
    "loss":                        [...],
    "tis_abs":                     [...],
    "global_batch_size":           [...]
  },

  "behavior": {
    "completed_turns_p50":     [...],
    "completed_turns_p95":     [...],
    "response_tokens_p50":     [...],
    "response_tokens_p95":     [...],
    "peak_context_tokens_p50": [...],
    "peak_context_tokens_p95": [...],
    "response_len_mean":       [...],       // rollout/response_len/mean
    "time_budget_exceeded":    [...],       // rollout/outcome/time_budget_exceeded
    "context_exhausted":       [...],       // rollout/outcome/context_exhausted
    "per_turn_seconds":        [...]        // 现算：model_generation / completed_turns_p50
  },

  // ---- 结论（趋势 + 阈值，每条带 evidence 与 actions）----
  "verdicts": [
    {
      "code": "removed_ratio_saturated",
      "severity": "bad",
      "group": "removed",
      "title": "训练样本被移除殆尽，整步无样本可训",
      "evidence": [
        "rollout/removed_ratio 从 0.703 升到 1.000（Δ+0.297，65 步）",
        "rollout/population/trainable_attempt_count 从 38 降到 0",
        "rollout/population/shaped_reward_coverage 从 0.984 降到 0.000",
        "最后 3 步 removed_ratio = 0.457 / 0.731 / 1.000"
      ],
      "actions": [
        "先查 rollout/removed/count_* 的主因（见 removed.by_reason 最大项）",
        "主因是 count_agent_timeout → 提高 agent 时间预算或限制单轮输出长度",
        "主因是 count_zero_std_* → 调整题目池难度配比（见奖励诊断的梯度密度）"
      ],
      "related_tags": ["rollout/removed_ratio", "rollout/population/trainable_attempt_count"]
    },
    { "code": "agent_time_budget_exhausted", "severity": "bad", ... },
    { "code": "gradient_density_low", "severity": "warn", ... },
    { "code": "behavior_response_bloat", "severity": "warn", ... }
  ],

  // ---- 已排除的病因（刻意设计）----
  "exclusions": [
    {
      "code": "off_policy",
      "title": "已排除：off-policy / train-rollout 分布不一致",
      "evidence": [
        "train/train_rollout_logprob_abs_diff 全程 0.019 ~ 0.045（45 个点）",
        "全程低于 slime CI 断言阈值 0.1"
      ],
      "threshold": 0.1,
      "source": "slime CI 断言阈值"
    },
    {
      "code": "entropy_collapse",
      "title": "已排除：熵坍缩",
      "evidence": [
        "train/entropy_loss 从 0.207 上升到 0.640（上升，不是坍缩）",
        "熵坍缩的特征是熵持续下降"
      ],
      "threshold": null,
      "source": "方向性判定"
    },
    {
      "code": "kl_explosion",
      "title": "已排除：KL 爆炸",
      "evidence": [
        "train/ppo_kl 全程量级 ~1e-6（最大 1e-6 附近）",
        "远离 KL 爆炸的量级"
      ],
      "threshold": 1e-4,
      "source": "设计假设：KL 爆炸通常指 ppo_kl > 1e-2 量级"
    }
  ],

  "metrics_available": {"n_tags": 458, "n_curves": 31},
  "page_note": "..."
}
```

#### D.2.3 `verdicts` 的规则（趋势 + 阈值，逐条可追溯）

**通用形式**：`severity` 由「趋势幅度」+「终点水平」双条件决定（用 §D.1.4 的 `AnomalyRule` 机制）。

| code | 指标 | 趋势条件 | 水平条件 | severity | 实证依据 |
| --- | --- | --- | --- | --- | --- |
| `removed_ratio_saturated` | `rollout/removed_ratio` | Δ ≥ +0.2 | 末值 ≥ 0.8 | bad | task4: 0.703→**1.000**（Δ+0.297）；task8 峰值 0.9453；task9 末 0.159（不触发） |
| `removed_ratio_rising` | `rollout/removed_ratio` | Δ ≥ +0.2 | 末值 ≥ 0.4 | warn | task8: 0.391→0.628（Δ+0.237，末 0.628 ≥ 0.4 → warn；不达 bad 的 0.8） |
| `trainable_attempts_dried_up` | `rollout/population/trainable_attempt_count` | Δ ≤ -20 | 末值 ≤ 20 | bad | task4: 38→**0**；task11: 95→**15**；task9: 127→107（不触发）；task8: 78→47（末 47 > 20 → 不触发，但 Δ=-31 ≤ -20 → 需与水平条件同时满足 → 不触发 ✓） |
| `shaped_reward_coverage_collapse` | `rollout/population/shaped_reward_coverage` | Δ ≤ -0.2 | 末值 ≤ 0.7 | bad | task4: 0.984→**0.000**；task11: 0.961→**0.430**；task8: 1.0→0.711（Δ-0.289，末 0.711 > 0.7 → warn 级）；task9: 0.992→0.961（不触发） |
| `agent_time_budget_exhausted` | `rollout/outcome/time_budget_exceeded` | Δ ≥ +0.05 | 末值 ≥ 0.20 | bad | task8: 0→**0.2266**（末 ≥ 0.20 ✓）；task11: 0→**0.406** ✓；task9: 0.016→0.125（末 < 0.20 → warn）；task2: 0→0（不触发） |
| `context_exhaustion` | `rollout/outcome/context_exhausted` | Δ ≥ +0.05 | 末值 ≥ 0.20 | warn | task8: 0.1406→0.0（**下降**，不触发）；task9: 峰值 0.125 |
| `per_turn_slowdown` | 现算 `per_turn_seconds`（`model_generation` ÷ `completed_turns_p50`） | Δ ≥ +50% | 末值 ≥ 2× 首值 | warn | 简报 §2.2：task8 单轮 18.7s→64.3s（**+244%**）；task9 +40%（不触发 warn 的 +50%） |
| `response_token_bloat` | `rollout/selected/attempt/response_tokens/p50` | Δ ≥ +20% | 末值 ≥ 1.2× 首值 | warn | task8: 23999→**31697**（**+32.1%** ✓） |
| `turn_count_shrink` | `rollout/selected/attempt/completed_turns/p50` | Δ ≤ -15% | 末值 ≤ 0.85× 首值 | warn | task8: 62.5→**40.5**（**-35.2%** ✓）；简报 §2.2「turns 57→36 下降 → 不是轮数爆炸」 |
| `peak_context_shrink` | `rollout/selected/attempt/peak_context_tokens/p50` | Δ ≤ -15% | 末值 ≤ 0.85× 首值 | info | task8: 84811→**65418**（-22.9%）；**这是「不是上下文打满」的证据** → severity 特意降为 info |
| `grad_norm_spike` | `train/grad_norm` | Δ ≥ +100% | 末值 ≥ 0.5 | bad | task11: 0.181→**0.986**（+445% ✓）；task8 0.241→0.130（不触发） |
| `pg_clipfrac_high` | `train/pg_clipfrac` | — | 末值 ≥ 0.05 | warn | task8: 0→0.0426（末 < 0.05，不触发）；峰值 0.0938 |
| `train_rollout_logprob_drift` | `train/train_rollout_logprob_abs_diff` | — | **全程 max ≥ 0.1** | bad | 实测 task4/task8/task9 全部 < 0.05 → 不触发（并同时产出 `exclusions` 的 off_policy 条目） |
| `gradient_density_low` | 来自 `reward_diagnosis_service`（DB 侧） | — | 末值 < 0.15 | warn | 简报 §3：task4 2.3%；task9 50% |
| `wasted_rollout_high` | 来自 `reward_diagnosis_service`（DB 侧） | Δ ≥ +0.25 | 末值 ≥ 0.5 | bad | task8 rollout 59 = 1.0 |

**每条 verdict 的 `evidence` 必须引用具体指标名与数值**（格式：`"<tag> 从 <a> <到/升到> <b>（Δ<+/-><c>，<n> 个点）"`），`actions` 必须可执行（引用其它 verdict 或具体配置项）。

> **重要**：`gradient_density_low` / `wasted_rollout_high` 依赖 **DB 侧**（`reward_diagnosis_service`），而其它 verdict 依赖 **TB 侧**。`build_train_health` 必须**组合两者** → 它是 service 层，组合 DB + utils 正是 service 的职责 ✓。

#### D.2.4 `exclusions` 的判定规则（**必须基于数值，不能是硬编码文案**）

```python
_EXCLUSION_RULES = (
    ExclusionRule(
        code="off_policy",
        title="已排除：off-policy / train-rollout 分布不一致",
        tag="train/train_rollout_logprob_abs_diff",
        mode="max_below",
        threshold=0.1,
        threshold_source="slime CI 断言阈值 0.1",
        detail="全程低于 0.1 → 训练样本的 logprob 与采样时一致，不存在分布偏移",
    ),
    ExclusionRule(
        code="entropy_collapse",
        title="已排除：熵坍缩",
        tag="train/entropy_loss",
        mode="not_decreasing",       # 末值 ≥ 首值 → 不是坍缩
        threshold=None,
        threshold_source="方向性判定：熵坍缩 = 熵持续下降",
        detail="熵上升 → 策略没有过早收敛",
    ),
    ExclusionRule(
        code="kl_explosion",
        title="已排除：KL 爆炸",
        tag="train/ppo_kl",
        mode="max_below",
        threshold=1e-4,
        threshold_source="设计假设：KL 爆炸通常指 ppo_kl > 1e-2 量级；此处取 1e-4 作为保守上界",
        detail="ppo_kl 远低于爆炸量级",
    ),
    ExclusionRule(
        code="prefix_cache_ok",     # 可选
        ...
    ),
)
```

**`mode` 取值**：`max_below` / `max_above` / `not_decreasing` / `not_increasing` / `min_above`。
**产出条件**：规则命中 → 在 `exclusions` 里追加一条，**带实际数值 evidence**；不命中 → **不出现在 `exclusions` 里**（即「未排除」保持沉默，不写「未排除 off-policy」这种噪音）。

> **实测校验**：task4 的 `train_rollout_logprob_abs_diff` 全程 0.019–0.045 < 0.1 ✓ 排除；`entropy_loss` 0.207→0.640 上升 ✓ 排除；`ppo_kl` ~1e-6 < 1e-4 ✓ 排除。task8 同理。task11 的 `entropy_loss` 0.151→**0.109**（**下降**！）→ **不排除熵坍缩**，会真实出现在 verdicts 里（需一条 `entropy_collapse` verdict 规则，本次可作为 `warn` 级）。这是 `exclusions` 设计的价值：它**不会**把 task11 误标为「已排除熵坍缩」。

#### D.2.5 分层与复用

| 需求 | 复用 |
| --- | --- |
| 事件目录定位 | `service/train_curve_service.resolve_event_dir`（**直接调用**，不重写） |
| 全量序列读取 | `utils.tb_series_reader.read_scalar_series` / `read_scalar_series_by_prefixes`（§D.1.2） |
| tag 含义 | `utils.tb_metric_catalog.describe_tb_tag_extended` / `tb_tag_info`（§C.6） |
| 曲线 tag 前缀匹配 | `utils.train_curve_catalog.resolve_metric_tags`（用于把「行为画像」指标映射到实际 tag） |
| epoch 聚合 | `utils.train_curve_epoch.aggregate_by_epoch` / `epoch_view`（若需要 epoch 视图） |
| 组语义 | `utils.slime_grpo_semantics`（§A） |
| DB 侧组指标 | `service.reward_diagnosis_service.summarize_gradient_groups`（§B） |
| 异常检测机制 | `utils.tb_series_reader.find_anomalies` + `AnomalyRule`（§D.1.4） |

**不新建的东西**：不新建 TB 读取器、不新建 tag 目录、不新建 group 判定器。

#### D.2.6 向后兼容影响

- **全新 module**，无历史消费者 → 零破坏。
- `service/reward_diagnosis_service.build_reward_diagnosis` **新增** `train_health` 字段（增量）→ 旧 UI 不受影响。
- 需要新增 controller 端点（`GET /api/tasks/{task_id}/train-health.json`）→ 属新增路由，不破坏既有。
- **性能约束**：`build_train_health` 内部**必须一次** `read_scalar_series_by_prefixes` 读全部所需 tag（实测 Reload 一次 ~1.3~2.3s）。**禁止**按组多次调用 `read_scalar_series`。

#### D.2.7 风险与边界

1. **无 TB 目录的任务**：必须返回 `{"available": false, "notice": "..."}` 而**不抛异常**（对齐 `service/insight_page_service.build_insight_page` 的既有约定）。
2. **tag 缺失**：不同 run 的 tag 集不同（task9 457 vs task8 458）→ 每条曲线都要容忍缺失（缺 → 空 list + `available=false` 标记）。
3. **TB step 与 rollout_id 的对齐**：本次实测 `rollout/removed_ratio` 的 step 0..64 与 rollout_id 0..63 **相差 1**（TB 多一个点，且 `rollout/removed/count_agent_timeout` 只有 58 个点而 `removed_ratio` 有 65 个 → 部分 tag 缺失步）。**不能假设 step == rollout_id**，必须按各自序列的 step 对齐，不能按数组下标对齐。
4. **`per_turn_seconds` 是现算派生量**：=`perf/rollout_time/attempt/agent_run/model_generation` ÷ `rollout/selected/attempt/completed_turns/p50`。两者 step 轴可能不齐 → 按 step 字典 join，缺失点跳过（不是插值）。
5. **`exclusions` 的阈值必须带来源**：`threshold_source` 字段强制填写，区分「slime CI 断言阈值」（0.1，实证）与「设计假设」（1e-4）。

---

## §E `service/problem_analysis_service.py` 通过率三口径修正设计

### E.1 目标与依据

**目标**：把「通过率」从一个有歧义的口径，拆成**三个明确命名、各有分工**的口径。

**依据**：
- 简报 §2.3 + 计划 §2.3：现有 `summarize_pass_visits` 把 abort/timeout 计入分母（其 `_PAGE_NOTE` 明说），与 slime 的整组丢弃口径不符。
- 本次实测的三个关键锚点：
  1. `passrate/pass@1` ≈ (reward==1 的 attempt 总数) / (全部 attempt 总数) —— 64 步中 49 步吻合（容差 5e-4）→ **strict 口径的 TB 对应物**。
  2. `passrate/pass@8` = (至少一个 reward=1 的组数) / 总组数 —— **64/64 步精确吻合** → **slime_effective 的组级对应物**。
  3. 简报 §3.2：全局 attempt 结局分布 completed 42754 / truncated 5520 / aborted 3555，aborted 的平均 reward = **0.0** → aborted 计入分母会**系统性压低** strict 口径。

### E.2 三个口径的精确分子分母定义

设某次「出现」= 同一 `(rollout_id, instance_id)` 下的全部 attempt。

#### E.2.1 `pass_rate_strict`（严格口径 / 端到端成功率）

```
分子 = 该次出现中 status 非 aborted 且 reward ≈ 1 的 attempt 数
分母 = 该次出现的全部 attempt 数（含 aborted / truncated）
```

- **等价表述**：`n_pass / n_attempts`（与现有 `rate` 字段**完全一致**）。
- **与 TB 对应**：`passrate/pass@1`（本次实测吻合度最好的一档）。
- **语义**：端到端成功率 —— 用户/客户视角「这题做对了吗」。
- **注意**：现有实现的分子只算 `category == "pass"`，而 `DBAttemptService._classify_attempt_row` 对 truncated 且 reward=1 的 attempt 归为 **timeout**（因为 status 优先）→ 这些 attempt **不计入分子**，但**计入分母**。
  > 实测：task8 有 236 条 truncated+reward=1.0。因此 strict 口径的分子应显式写为「`reward ≈ 1` 的 attempt（含 truncated）」，还是「`category == pass`」？
  > **决策**：`pass_rate_strict` 的分子 = `reward ≈ 1` 的 attempt 数（**不论 status**），因为「奖励是 1」这件事本身与是否截断无关；而 `aborted` 的 reward 在 DB 里被写成 **0.0**（实测 1589 条），所以「reward≈1」天然排除了 aborted。**这与 `passrate/pass@1` 的定义一致**（实测吻合验证）。
  > ⚠️ **口径变更风险**：现有 `rate` 字段用的是 `category == "pass"`（排除 truncated+reward=1）。改为「reward≈1」会**改变数值**。→ 见 E.4。

#### E.2.2 `pass_rate_gradable`（可评分口径 / 有效样本质量）

```
分子 = 该次出现中 reward ≈ 1 的 attempt 数（同 strict 的分子）
分母 = 该次出现中【可评分】的 attempt 数
       = 全部 attempt 数 − aborted 数 − truncated 数
```

- **等价表述**：`n_pass / n_gradable`。
- **语义**：只看「能评出分的样本」的通过率 —— 剔除环境吃掉的样本后的策略质量。
- **与 TB 对应**：`rollout/selected/attempt/*` 的 coverage 类指标（`coverage` = 被保留的占比）。
- **注意**：现有 `summarize_reward_split.measurable_pass_rate` 用的分母是 `n_pass + n_fail + n_other`（**包含** other），且分子是 `category == "pass"`。`pass_rate_gradable` 与它**基本一致但分子口径不同** → 见 E.4。

#### E.2.3 `pass_rate_slime_effective`（slime 有效口径 / 对齐真实训练分布）

**组级口径**（不是 attempt 级）：

```
分子 = 该次出现中【未被丢弃的组】里，reward ≈ 1 的 attempt 数之和
分母 = 该次出现中【未被丢弃的组】里的全部 attempt 数之和
```

其中「未被丢弃的组」= 该组**不含任何 aborted attempt** **且** **存活 attempt 数 ≥ 2**（即 §A 的 `classify_grpo_group` 不返回 `dropped`）。

- **等价表述**：`Σ_{g 未被丢弃} n_pass(g) / Σ_{g 未被丢弃} n_attempts(g)`。
- **语义**：**训练真正用到的分布** —— 如果这题所在的组全被丢了，它对训练毫无贡献，不该进分母。
- **与 TB 对应**：`passrate/pass@8`（组级，但**分母是总组数**，含被丢弃组 —— 见 E.2.4 的差异说明）。
- **注意**：这是**三个口径中唯一需要组信息的**。组信息来自 `(rollout_id, group_index)` —— 现有 `summarize_pass_visits` 的 `rows` 已经带 `group_index` ✓。

#### E.2.4 三个口径与 TB 的对账表（**必须写进 docstring / UI**）

| 工具口径 | 层级 | TB 对应物 | 分母含被丢弃组？ | 实测吻合 |
| --- | --- | --- | --- | --- |
| `pass_rate_strict` | attempt | `passrate/pass@1` | ��� | 64 步中 49 步吻合（容差 5e-4） |
| `pass_rate_gradable` | attempt | `rollout/selected/attempt/*/coverage` 反推 | ❌ | — |
| `pass_rate_slime_effective` | attempt（组内加权） | — | ❌ | — |
| （参考）`pass@8` 组级 | group | `passrate/pass@8` | **✅ 含** | 64/64 精确吻合 |

> **关键差异说明**：`pass_rate_slime_effective` 的分母**不含**被丢弃组；而 TB 的 `passrate/pass@8` 分母**含**被丢弃组（实测 step59：工具侧有效组为 0，TB pass@8 = 0.5625，因为被丢弃组里仍有 reward=1 的 attempt）。
> **这个差异本身就是发现**：被丢弃的组**并非毫无价值**（它们仍贡献 pass@8 的分子），但它们**不产生梯度**。所以：
> - `pass_rate_slime_effective` 回答「训练实际看到的样本质量如何」
> - `pass@8` 回答「采样层面的通过广度」
> - 两者差异大 → 说明**大量样本被丢弃**（正是 `wasted_rollout_ratio` 要暴露的）。

### E.3 签名与输出 schema

#### E.3.1 签名

```python
def summarize_pass_visits(
    rows: Sequence[Dict[str, Any]],
    *,
    batch_size: Optional[int] = None,
    batch_size_source: str = "missing",
    count_timeout_as_reward: bool = True,
) -> Dict[str, Any]:
    """把 attempt 结局行收成按题目出现次序的通过率（**三口径**）。

    一次出现 = 同一个 ``rollout_id`` 下的同一个 ``instance_id``。
    出现次序按 ``rollout_id`` 升序，第 k 次即该题的第 k 个 epoch。
    ``rows`` 每项需含 ``rollout_id`` / ``group_index`` / ``instance_id`` /
    ``category``（pass/fail/timeout/abort/other）/ ``reward``。

    三口径定义见 ``_PAGE_NOTE``。
    """
```

`count_timeout_as_reward` 与 §A 同义（默认 True = 与 slime 一致）。

#### E.3.2 输出 schema

**每次出现（`visits` 数组元素）新增字段**：

```jsonc
{
  "rollout_id": 3,
  "n_pass": 4,              // 已有：category == "pass" 的数量
  "n_attempts": 8,          // 已有：全部 attempt 数
  "rate": 0.5,              // 已有：== pass_rate_strict（保持不变！见 E.4）

  // ---- 新增 ----
  "n_reward_one": 4,          // reward ≈ 1 的 attempt 数（不论 status）
  "n_reward_zero": 3,         // reward ≈ 0 的 attempt 数
  "n_aborted": 1,             // aborted 数
  "n_truncated": 0,           // truncated 数
  "n_gradable": 7,            // 全部 − aborted − truncated
  "n_dropped_groups": 0,      // 本次出现涉及的被丢弃组数
  "n_effective_groups": 1,    // 未被丢弃的组数
  "n_effective_attempts": 7,  // 未被丢弃组里的 attempt 总数

  "pass_rate_strict": 0.5,            // n_reward_one / n_attempts
  "pass_rate_gradable": 0.5714,       // n_reward_one / n_gradable
  "pass_rate_slime_effective": 0.5714, // n_reward_one_eff / n_effective_attempts
  "pass_rate_slime_effective_groups": 0.5714  // 组级等价值（组内平均）
}
```

> **`rate` 字段保持不变** = `pass_rate_strict` 的**旧定义**（`n_pass / n_attempts`）→ **完全向后兼容**。
> 新口径用新字段名。UI 默认展示新三口径，`rate` 保留给旧调用方（`cross_task_compare_service` 用了 `visits[].rate`）。

**顶层新增**：

```jsonc
{
  "n_problems": ..., "n_rollouts": ..., "n_attempts_without_instance": ...,
  "batch_size": ..., "batch_size_source": ...,
  "steps_per_epoch": ..., "steps_per_epoch_exact": ...,
  "n_epochs_observed": ..., "n_epochs_from_steps": ...,
  "epoch_aligned": ..., "alignment_checked": ..., "alignment_violations": [...],
  "summary": {...},          // 已有（基于 trend）
  "histogram": [...],        // 已有
  "epoch_series": [...],     // 已有
  "tasks": [...],            // 已有（visits 元素新增上述字段）

  // ---- 新增 ----
  "rate_definitions": {
    "pass_rate_strict": {
      "label": "严格通过率",
      "numerator": "reward≈1 的 attempt 数",
      "denominator": "该次出现的全部 attempt 数（含 abort / timeout）",
      "tb_anchor": "passrate/pass@1",
      "use": "端到端成功率"
    },
    "pass_rate_gradable": {
      "label": "可评分通过率",
      "numerator": "reward≈1 的 attempt 数",
      "denominator": "全部 attempt 数 − abort 数 − timeout 数",
      "tb_anchor": "rollout/selected/attempt/*/coverage",
      "use": "有效样本质量"
    },
    "pass_rate_slime_effective": {
      "label": "slime 有效通过率",
      "numerator": "未被丢弃组内 reward≈1 的 attempt 数之和",
      "denominator": "未被丢弃组内全部 attempt 数之和",
      "tb_anchor": "（组级对应 passrate/pass@8，但分母不含被丢弃组）",
      "use": "对齐 slime 真实训练分布"
    }
  },
  "pass_rate_summary": {
    "pass_rate_strict": {"first": ..., "last": ..., "delta": ...},
    "pass_rate_gradable": {...},
    "pass_rate_slime_effective": {...}
  },
  "note": "..."   // 更新
}
```

### E.4 向后兼容：现有功能如何适配三口径

| 现有功能 | 现状 | 适配方案 |
| --- | --- | --- |
| **首尾差直方图** `_histogram(deltas)` | `delta = last_rate - first_rate`，`rate = n_pass/n_attempts`（strict 旧定义） | **保持用 `rate`（strict 旧定义）**，直方图不变 → 零破坏。**可选新增** `histogram_by_caliber` 提供 gradable / slime_effective 两套直方图（增量，不替换） |
| **太容易 / 太难筛选** `summarize_pool_health` | 用「末次（最好一次）通过率」 | **改用累计通过率三态**（§B.3）。但注意：这是 `reward_diagnosis_service` 的函数，与 `problem_analysis_service` 的 `summarize_pass_visits` 是两个模块。`summarize_pool_health` 的三态判定由 `utils/difficulty_profile` 提供（§B.3.4） |
| **按题检索历次通过率** `group_instance_appearances` | `visit["rate"] = n_pass / n_attempts` | **保留 `rate`**，**新增** `pass_rate_strict` / `pass_rate_gradable` / `pass_rate_slime_effective` 三个并列字段 → `visit` 层面也一致 |
| **`_instance_trend`** | `points[].rate` | 保留 `rate`；新增 `points[].pass_rate_strict/gradable/slime_effective` |
| **`_epoch_pass_series`** | `rate = Σn_pass / Σn_attempts`（strict） | 保留；新增三个口径的 epoch 曲线 |
| **`cross_task_compare_service`** | 用 `visits[].rate` 与 `epoch_series[].rate` | **零改动**（旧字段仍在） |
| **`train_curve_service._resolve_steps_per_epoch`** | 用 `n_problems` + `batch_size` | 零改动 |
| **`_PAGE_NOTE`** | 说明「abort 与 timeout 计入分母、不计入分子」 | **更新**为三口径说明 |

### E.5 UI 标注（如何展示三者差异）

**问题分析页**（`templates/problem_analysis.html`）：

1. **每个题目的通过率显示三行**，并标注口径：
   ```
   严格  50.0%   (4/8)      ← 端到端；含被 abort/timeout 吃掉的样本
   可评分 57.1%  (4/7)      ← 剔除 abort/timeout 后的样本质量
   slime 有效 57.1% (4/7)   ← 只算真正进入训练的组
   ```
2. **差异高亮**：当三个数**不相等**时，在数值旁显示一个提示图标，hover 显示差异来源：
   - `strict < gradable` → 「有 N 个 abort / timeout 吃掉了分母」
   - `gradable > slime_effective` → 「有 M 个组被丢弃（整组不参与训练）」
3. **任务级汇总卡片**：顶部显示三个口径的任务级平均 + **一条「差异说明」**（例如「本任务 12.5% 的样本因 abort 被吃掉，8.3% 的组被整组丢弃」）。
4. **首尾差直方图**：标题旁加一个口径选择器，默认「严格」（保持现状），可选「可评分」「slime 有效」。

**与 `reward_diagnosis` 页的关系**（避免两页矛盾）：
- `reward_diagnosis` 页的 `measurable_pass_rate` = `pass_rate_gradable` 的**任务级 rollout 加权**版本（现有 `summarize_reward_split`）。
- `problem_analysis` 页的 `pass_rate_gradable` = **按题目出现**聚合（epoch 口径）。
- 两者分母定义相同（都剔除 abort/timeout），但加权方式不同（rollout 加权 vs 题目出现加权）→ **必须在两处 note 里互相说明**，否则用户会认为数据打架。

### E.6 风险与边界

1. **分子口径变更风险**：若把 `pass_rate_strict` 的分子从 `category == "pass"` 改为 `reward ≈ 1`，数值会变（task8 有 236 条 truncated+reward=1）。**决策**：`rate` 字段保持旧定义（`category == "pass"`），`pass_rate_strict` 用新定义（`reward ≈ 1`）→ **同一页面会同时存在两个略有差异的数**。
   > **缓解**：`rate` 标记为 `legacy`（在 `rate_definitions` 里说明「历史字段，等价于 pass_rate_strict 的旧分子口径」），并在 UI 上**默认不展示** `rate`。同时在 `note` 里写明差异来源与量级（实测差异 < 3%）。
   > **替代方案（更干净）**：`rate` 也改为 `reward ≈ 1` 口径，并更新 `tests/test_problem_analysis_service.py` 的相关断言。**推荐采用替代方案**（避免双口径并存），但需评估 `cross_task_compare_service` 的影响 —— 实测该 service 只读 `rate` 做展示，不做阈值判定 → 影响可控。
2. **`pass_rate_slime_effective` 需要组信息**：`rows` 必须带 `group_index`。现有 `summarize_pass_visits` 已把 `group_index` 计入 `groups_per_rollout` ✓。若某次调用缺 `group_index` → 该次出现**退化为 gradable 口径**并标注 `group_known=false`。
3. **小样本陷阱**：某题某次出现只有 1 个 attempt 且是 abort → `gradable` 分母为 0 → 返回 `None` 而非 0.0（与 §A 的约定一致）。
4. **`instance_id` 缺失的行**：`n_attempts_without_instance` 现有逻辑保留；这些行不进入任何口径的分子分母。

---

## §F 测试用例清单

### F.1 现有测试风格（已读 `tests/` 现有用例）

- **框架**：`unittest`（`class TestXxx(unittest.TestCase)` + `if __name__ == "__main__": unittest.main()`）。
- **风格**：模块级 `_row(...)` / `_rows(...)` 工厂函数构造输入；`self.assertEqual` / `assertAlmostEqual`；少量 `assertIn` / `assertIsNone` / `assertTrue`。
- **路径处理**：`sys.path` 修正只在需要时（`test_tb_metric_catalog.py` 有 `sys.path.insert`；其余直接 `from service.x import y`）。
- **可选依赖**：`@unittest.skipUnless(_HAS_TB, "tensorboard 未安装")`（`test_tb_series_reader.py` 模式）。
- **命名**：文件 `test_<模块名>.py`；类 `Test<被测类/函数>`；函数 `test_<行为>`（蛇形，描述行为而非实现）。

### F.2 需要迁移的现有断言（**重要：会因本次改动而失败**）

| 文件 | 断言 | 迁移方式 |
| --- | --- | --- |
| `tests/test_reward_diagnosis_service.py` | `classify_group_kind(["abort","timeout"]) == "infra_only"` | 改为 `== "dropped"`；并新增 `classify_group_kind(["abort","timeout"]) != "mixed"` 的语义断言 |
| `tests/test_reward_diagnosis_service.py` | `out["series"][1]["n_infra_only"] == 1` | 保留字段名（兼容）但**断言值改为 `n_dropped`**；`n_infra_only` 仍输出（同值） |
| `tests/test_reward_levers.py` | `test_always_returns_all_levers` 的 key 集合 == 7 个 | **必须扩到新集合**（见 F.5） |
| `tests/test_reward_levers.py` | `TestSummarizePoolHealth::test_saturated_and_hard_problems`（用「末次」口径构造） | 改为累计口径构造；见 F.5 |
| `tests/test_reward_levers.py` | `test_uses_best_appearance_not_last` | **语义变更**：旧口径取「最好一次」，新口径取「累计」。该用例的构造（rollout1 全挂 → rollout2 全过）在新口径下累计通过率 = 0.5 → `gradient`，不再是 `stable_pass`。**必须重写** |
| `tests/test_problem_analysis_service.py` | `by_id["alpha"]["delta"]` 等基于 `rate` 的断言 | `rate` 保持不变 → **全部通过**（若采用 E.6 的替代方案则需更新） |
| `tests/test_tb_metric_catalog.py` | `describe_tb_tag("train/loss") == DESC_UNKNOWN` | 方案 A 下**仍通过** ✓（`describe_tb_tag` 契约冻结） |

### F.3 `tests/test_slime_grpo_semantics.py`（新建）—— 交付物 A

| 测试函数 | 验证点 | 构造输入 |
| --- | --- | --- |
| `test_mixed_group` | 有对有挂 → `mixed` | `rewards=[1.0, 0.0, 1.0, 0.0]`, `statuses=["completed"]*4` |
| `test_all_pass_group` | 全过 → `all_pass` | `rewards=[1.0]*4`, `statuses=["completed"]*4` |
| `test_all_fail_group` | 全挂 → `all_fail` | `rewards=[0.0]*4`, `statuses=["completed"]*4` |
| `test_aborted_group_is_dropped_not_all_fail` | **含 abort → `dropped`（不是 all_fail）** | `rewards=[0.0, 0.0, 0.0, None]`, `statuses=["completed","completed","completed","aborted"]` |
| `test_aborted_with_reward_one_is_dropped` | abort 且 reward=1 仍是 `dropped`（不被 reward 误导） | `rewards=[1.0, None]`, `statuses=["completed","aborted"]` |
| `test_too_few_survivors_is_dropped` | 存活 < 2 → `dropped` | `rewards=[1.0, None, None]`, `statuses=["completed","truncated","truncated"]` |
| `test_ungradable_group_is_dropped` | **【v2 二次修订】** `train_drop_reason="ungradable"` → `dropped`（不是 mixed） | `drop_reasons=["ungradable"]` + 组内有对有挂 |
| `test_ungradable_can_be_disabled` | `drop_ungradable=False` 时该组判 mixed | 同上 + `drop_ungradable=False` |
| `test_exactly_two_survivors_is_not_dropped` | 存活 == 2 → 不判 dropped | `rewards=[1.0, 0.0, None]`, `statuses=["completed","completed","truncated"]` |
| `test_timeout_counted_as_reward_default` | 默认计入 truncated 的 reward → mixed | `rewards=[1.0, 0.0, 1.0]`, `statuses=["completed","completed","truncated"]` → `mixed` |
| `test_timeout_excluded_when_flag_off` | `count_timeout_as_reward=False` → 存活 2 个都是 1.0 → `all_pass` | 同上 + `count_timeout_as_reward=False` → `all_pass` |
| `test_empty_group_is_unknown` | `[]` → `unknown` | `rewards=[]`, `statuses=[]` |
| `test_length_mismatch_is_unknown` | 长度不等 → `unknown`（不抛异常） | `rewards=[1.0,0.0]`, `statuses=["completed"]` |
| `test_mid_value_reward_is_unknown` | reward=0.5 不算 1 也不算 0 | `rewards=[0.5, 0.5]`, `statuses=["completed","completed"]` → `unknown` |
| `test_none_reward_not_counted` | reward=None 两侧都不计 | `rewards=[1.0, None, 0.0]`, `statuses=["completed"]*3` → `mixed` |
| `test_gradient_contribution_mixed_is_one` | `group_gradient_contribution("mixed") == 1.0` | — |
| `test_gradient_contribution_others_zero` | `all_pass`/`all_fail`/`dropped`/`unknown` 均为 0.0 | — |
| `test_gradient_contribution_unknown_kind_zero` | 未知 kind → 0.0（不抛异常） | `"whatever"` |
| `test_action_hint_distinguishes_all_pass_and_all_fail` | **两类零梯度的文案不同且方向相反** | 断言 `"移出"` in all_pass 文案；`"换题"` 或 `"降采样"` in all_fail 文案；两文案 != |
| `test_action_hint_dropped_mentions_waste` | dropped 文案含「浪费」/「白花」 | — |
| `test_action_hint_unknown_kind` | 未知 kind → unknown 文案（不抛异常） | — |
| `test_is_zero_gradient_excludes_dropped` | `is_zero_gradient("dropped") is False` | — |
| `test_is_wasted_only_dropped` | `is_wasted("dropped") is True`；`is_wasted("all_fail") is False` | — |
| `test_zero_gradient_ratio` | `(all_pass+all_fail)/n_groups` | `{"mixed":5,"all_pass":3,"all_fail":2,"dropped":6}` → 5/16 |
| `test_wasted_rollout_ratio` | `dropped/n_groups` | 同上 → 6/16 |
| `test_gradient_density` | `mixed/n_groups` | 同上 → 5/16 |
| `test_ratios_none_when_no_groups` | 空 counts → 三个指标全 `None`（不是 0.0） | `{}` |
| `test_group_kind_labels` | 四个 kind 的中文标签互不相同且含关键词 | — |
| `test_group_kinds_tuple` | `GROUP_KINDS == ("mixed","all_pass","all_fail","dropped","unknown")` | — |

### F.4 `tests/test_tb_metric_catalog.py` 扩展（交付物 C）—— **追加到现有文件**

| 测试函数 | 验证点 | 构造输入 |
| --- | --- | --- |
| `test_classify_removed_ratio` | `classify_tb_tag("rollout/removed_ratio") == "removed_ratio"` | — |
| `test_classify_removed_reason` | `rollout/removed/count_agent_timeout` → `"removed_reason"` | — |
| `test_classify_removed_reason_dynamic` | `rollout/removed/count_aborted_exception_RuntimeError` → `"removed_reason"`（动态后缀） | — |
| `test_classify_zero_std_is_group_count` | `rollout/zero_std/count_1.0` → `"group_count"`（**不是** removed_reason） | — |
| `test_classify_group_reward_tags` | `rollout/generated/shaped_group_reward/mixed_count` → `"group_count"`；`.../mixed_fraction` → `"group_ratio"` | — |
| `test_classify_population` | `rollout/population/trainable_attempt_count` → `"population"` | — |
| `test_classify_behavior_metric` | `rollout/selected/attempt/peak_context_tokens/p95` → `"behavior_metric"` | — |
| `test_classify_outcome` | `rollout/outcome/time_budget_exceeded` → `"outcome"` | — |
| `test_classify_train` | `train/entropy_loss` → `"train"` | — |
| `test_classify_perf_phase` | `perf/rollout_time/attempt/sandbox_boot_queue` → `"perf_phase"` | — |
| `test_classify_passrate` | `passrate/pass@8` → `"eval_score"` | — |
| `test_classify_primary_still_wins` | **回归**：`classify_tb_tag("eval/gsm8k") == "primary"`；`eval/gsm8k/attempt_count` → `"attempt_count"`；`eval/gsm8k-truncated_ratio` → `"truncated_ratio"` | — |
| `test_describe_extended_gives_source_and_inferred_flag` | `describe_tb_tag_extended("rollout/removed_ratio")` → `inferred is False` 且 `source` 非空 | — |
| `test_describe_extended_marks_inference` | 未收录 tag（如 `"rollout/foo/bar"`) → `inferred is True` 且 `desc` 含「推断」 | — |
| `test_describe_extended_behavior_metric_dynamic` | `rollout/selected/attempt/response_tokens/p50` → `label` 含「响应 token」 | — |
| `test_describe_tb_tag_contract_unchanged` | **回归**：`describe_tb_tag("train/loss") == DESC_UNKNOWN`；`describe_tb_tag("eval/gsm8k") == DESC_PRIMARY` | — |
| `test_catalog_covers_removed_family` | `find_tags_by_prefix("rollout/removed/")` 覆盖全部 12 个已知 count tag + `rollout/removed_ratio` | — |
| `test_catalog_size_sane` | `catalog_size() >= 100`（确保不是空目录） | — |
| `test_find_tags_by_prefix` | `find_tags_by_prefix("train/")` 含 `train/entropy_loss` | — |
| `test_tags_in_group` | `tags_in_group("removed")` 非空且全以 `rollout/removed` 开头 | — |
| `test_tb_tag_info_unknown_returns_unknown_kind` | `tb_tag_info("zzz/yyy")` → `kind == "unknown"` | — |

### F.5 `tests/test_reward_levers.py` 扩展（交付物 B）—— **追加 + 迁移**

| 测试函数 | 验证点 | 构造输入 |
| --- | --- | --- |
| **迁移** `test_always_returns_all_levers` | key 集合 == 旧 7 个 + 新 6 个 | `build_reward_levers([])` |
| **迁移** `test_uses_best_appearance_not_last` → `test_uses_cumulative_rate_not_last` | 累计口径：rollout1 全挂 + rollout2 全过 → `stable_fail`/`gradient` 而非 `stable_pass` | 见下 |
| `test_new_levers_present` | `gradient_density` / `wasted_rollout_ratio` / `zero_gradient_ratio` / `n_gradient_problems` / `time_budget_ratio` / `per_turn_seconds_growth` / `queue_wait_ratio` 都在 levers 里 | — |
| `test_gradient_density_equals_learnable_ratio` | 两者同值（别名一致性） | 2 组：1 mixed + 1 all_pass → 都是 0.5 |
| `test_wasted_rollout_ratio_counts_aborted_groups` | 含 abort 的组计入 wasted | 组0: pass+fail（mixed）；组1: pass+abort → `wasted_rollout_ratio == 0.5` |
| `test_zero_gradient_ratio_excludes_dropped` | zero_gradient 不含 dropped | 组0: pass+pass（all_pass）；组1: pass+abort（dropped）→ `zero_gradient_ratio == 0.5`，`gradient_density == 0.0` |
| `test_all_pass_and_all_fail_are_both_zero_gradient_but_different_hints` | 两类都算零梯度，但 lever 文案方向不同 | — |
| `test_gradient_density_low_is_bad` | `< 0.15` → bad | 1 mixed / 16 组 |
| `test_gradient_density_high_is_ok` | `≥ 0.30` → ok | 8 mixed / 16 组 |
| `test_wasted_rollout_high_is_bad` | `≥ 0.50` → bad | 8 组含 abort / 16 |
| `test_pool_tri_state_counts` | 三态计数正确 | 构造 3 题：全过 / 全挂 / 混合 |
| `test_insufficient_data_flagged` | 出现次数 < 3 → `n_insufficient_data > 0` | 每题只出现 1 次 |
| `test_pool_health_basis_is_cumulative` | `pool_health_basis == "cumulative"` | — |
| `test_saturated_ratio_equals_stable_pass_ratio` | 别名一致性 | — |
| `test_tb_levers_unknown_without_extra` | 无 `extra` 时 TB 类杠杆 `status == "unknown"`、`value is None` | `build_reward_levers([_row(1,0,"pass")])` |
| `test_tb_levers_ok_with_extra` | 传入 `extra` 后 TB 类杠杆正常判定 | `extra={"time_budget_ratio": 0.4}` → bad |
| `test_queue_wait_ratio_thresholds` | 0.15 → bad；0.08 → warn；0.02 → ok | 通过 `extra` 注入 |
| `test_per_turn_growth_thresholds` | 1.5 → bad；0.6 → warn；0.2 → ok | 通过 `extra` 注入 |
| `test_levers_sorted_by_status`（**已有，保留**） | 排序不变 | — |
| `test_empty_rows_gives_unknown_status`（**已有，保留**） | 空输入全 unknown | — |

### F.6 `tests/test_reward_diagnosis_service.py` 扩展（交付物 B）

| 测试函数 | 验证点 | 构造输入 |
| --- | --- | --- |
| **迁移** `test_four_kinds` → `test_five_kinds` | `["pass","fail"]`→mixed；`["pass","pass","abort"]`→**all_pass**（abort 不改变已判定组？见 A.3.3 R1 —— **含 abort 必须 dropped**，故断言改为 `["pass","pass"]`→all_pass）；`["fail","timeout"]`→all_fail；`["abort","timeout"]`→**dropped** | — |
| `test_abort_group_is_dropped` | 含 abort 的组 → dropped | — |
| `test_gradient_series_new_fields` | series 元素含 `n_dropped` / `gradient_density` / `zero_gradient_ratio` / `wasted_rollout_ratio` / `group_kinds` | — |
| `test_dropped_group_does_not_count_as_all_fail` | **关键回归**：含 abort 的组既不是 all_fail 也不是 zero_gradient | 组: pass + abort → `n_dropped == 1`，`n_all_fail == 0`，`zero_gradient_ratio == 0.0` |
| `test_wasted_dips_detected` | wasted_rollout_ratio 环比上升 ≥ 0.25 → 进 `wasted_dips` | rollout0: 0% → rollout1: 100% |
| `test_dips_unchanged` | 旧 `dips` 语义不变 | 复用现有用例构造 |
| `test_group_labels_distinguish_zero_gradient_and_wasted` | `group_labels["all_pass"] != group_labels["dropped"]`，且文案含「零梯度」/「浪费算力」 | — |
| `test_count_timeout_as_reward_flag` | `count_timeout_as_reward=False` 时 truncated 不计入组统计 | 组: completed(1.0) + truncated(0.0) → True→all_pass；False→(只剩 1 个 attempt)→unknown/dropped 视 R2 |
| `test_series_ratios_none_when_no_groups` | 空 rows → series 为空（不产生除零） | `[]` |

### F.7 `tests/test_problem_analysis_service.py` 扩展（交付物 E）

| 测试函数 | 验证点 | 构造输入 |
| --- | --- | --- |
| `test_three_rates_present_in_visits` | 每次出现含三个新字段 | — |
| `test_pass_rate_strict_counts_abort_in_denominator` | 分子 1 / 分母含 abort | 1 pass + 1 abort → strict = 0.5 |
| `test_pass_rate_gradable_excludes_abort` | 分母去掉 abort | 同上 → gradable = 1.0 |
| `test_pass_rate_slime_effective_excludes_dropped_group` | 组含 abort → 该组整体不进分母 | 组0: pass+fail（有效）；组1: pass+abort（丢弃）→ slime_effective = 1.0/2 = 0.5 |
| `test_slime_effective_none_when_all_groups_dropped` | 全部组被丢弃 → `None`（不是 0.0） | 1 组：pass+abort → `None` |
| `test_legacy_rate_field_unchanged` | **回归**：`rate` 仍等于 `n_pass/n_attempts` | — |
| `test_rate_definitions_metadata` | `rate_definitions` 含三个口径的分子分母与 TB 锚点 | — |
| `test_group_index_missing_falls_back_to_gradable` | 缺 `group_index` → slime_effective 退化为 gradable 且 `group_known=False` | rows 去掉 group_index |
| `test_histogram_unchanged` | **回归**：直方图与旧口径一致 | — |
| `test_epoch_series_has_three_calibers` | epoch 曲线新增三个口径字段 | — |
| `test_pass_rate_summary_first_last_delta` | 三口径的 first/last/delta | — |

### F.8 `tests/test_tb_series_reader.py` 扩展（交付物 D）

| 测试函数 | 验证点 | 构造输入 |
| --- | --- | --- |
| `test_read_by_prefix_selects_only_matching` | 只返回前缀命中的 tag | 写 2 个 tag：`rollout/removed_ratio` + `train/loss` |
| `test_read_by_prefixes_single_reload` | 多前缀一次读完，返回合并结果 | — |
| `test_read_by_prefix_empty_match` | 无命中 → `{}` | prefix 不存在 |
| `test_find_anomalies_detects_rising_trend` | 趋势上升 + 末值超阈 → 命中 | 序列 0.2→0.9，rule rising/threshold 0.5/min_growth 0.2 |
| `test_find_anomalies_ignores_single_spike` | **单点尖刺不命中**（趋势 vs 单点的核心区别） | 序列 [0.1,0.1,0.95,0.1,0.1] → 不命中 |
| `test_find_anomalies_requires_both_conditions` | 趋势够但末值不够 → 不命中 | 序列 0.1→0.6，threshold 0.8 → 不命中 |
| `test_find_anomalies_falling_direction` | 下降趋势 + 末值低于阈 → 命中 | 序列 0.9→0.1，falling/threshold 0.3 |
| `test_find_anomalies_window_averages` | `window>1` 时用首尾均值，抗单点噪声 | — |
| `test_find_anomalies_empty_series` | 空 → 空结果（不抛异常） | — |
| `test_find_anomalies_insufficient_points` | 只有 1 个点 → 跳过 | — |
| `test_anomaly_output_fields` | 输出含 `code`/`tag`/`severity`/`head_value`/`tail_value`/`delta` | — |
| `test_sidecar_cache_roundtrip`（`skipUnless _HAS_TB`） | 二次读取命中缓存（同一 fingerprint） | 临时目录写事件 → 读 → 再读 |
| `test_sidecar_invalidated_by_new_event_file` | 新增事件文件 → fingerprint 变 → 重新读 | 写第二个 event 文件后读到新点 |
| `test_sidecar_invalidated_by_content_append` | **同文件追加**（目录 mtime 不变）→ 仍能失效 | 追加写入后读到新点 |
| `test_series_cache_clear_for_tests` | `clear_series_cache_for_tests()` 清空进程内缓存 | — |

### F.9 `tests/test_train_health_service.py`（新建）—— 交付物 D

| 测试函数 | 验证点 | 构造输入 |
| --- | --- | --- |
| `test_missing_event_dir_returns_available_false` | 无事件目录 → `available is False` + `notice` 非空，**不抛异常** | task_id 指向无 TB 的任务（用 mock） |
| `test_three_curve_groups_present` | `removed` / `training` / `behavior` 三键都在 | — |
| `test_exclusions_include_off_policy` | `train_rollout_logprob_abs_diff` 全程 < 0.1 → exclusions 含 `off_policy` | 构造序列 0.02~0.04 |
| `test_exclusions_include_entropy_collapse_when_rising` | entropy 上升 → exclusions 含 `entropy_collapse` | 序列 0.15→0.64 |
| `test_no_entropy_collapse_exclusion_when_falling` | **反向用例**：entropy 下降 → **不**出现在 exclusions | 序列 0.64→0.11（task11 情形） |
| `test_exclusions_include_kl_explosion` | ppo_kl ~1e-6 → exclusions 含 `kl_explosion` | 序列 1e-6 |
| `test_verdict_removed_ratio_saturated` | removed_ratio 末值 1.0 → 产出 bad verdict，evidence 含数值 | — |
| `test_verdict_evidence_contains_tag_and_numbers` | 每条 verdict 的 evidence 至少一条且含数字 | — |
| `test_verdict_actions_non_empty` | 每条 verdict 的 actions 非空 | — |
| `test_no_verdict_when_series_flat` | 平坦序列 → 不产出该 verdict | — |
| `test_behavior_curves_present` | `completed_turns_p50` / `response_tokens_p50` / `peak_context_tokens_p50` 在 behavior 里 | — |
| `test_per_turn_seconds_derived` | `per_turn_seconds` 由 model_generation ÷ completed_turns_p50 现算 | — |
| `test_missing_tag_yields_empty_curve` | 某 tag 不存在 → 空 list，不抛异常 | — |

### F.10 测试执行与 CI

- 全部用 `unittest`；建议 `python -m pytest tests/ -q`（项目已装 pytest，`.pytest_cache` 在扫描黑名单里）。
- `scripts/scan_func_md.py --check` 必须在 CI 通过 → **本次改动后必须重跑 `python scripts/scan_func_md.py`** 并把 `func.md` 一并提交（conventions §6 明令）。
- 新增/修改的公开能力需登记 `func.md`：
  - `utils/slime_grpo_semantics.py`（新章节）
  - `utils/tb_metric_catalog.py`（扩充 `classify_tb_tag` / `describe_tb_tag` 条目 + 新函数）
  - `utils/tb_series_reader.py`（新增函数）
  - `service/reward_diagnosis_service.py`（函数说明更新）
  - `service/problem_analysis_service.py`（函数说明更新）
  - `service/train_health_service.py`（新章节）

---

## 附录 G：设计假设清单（无直接实证支撑，需后续验证）

| # | 假设 | 位置 | 验证方式 |
| --- | --- | --- | --- |
| G1 | slime `too_few_survivors_K` 的 K = 组内存活数（而非阈值） | §A.3.3 R2 | 读 slime 源码；或构造组大小 16 的 run 观察 K 范围 |
| G2 | `MIN_APPEARANCES_FOR_CONFIDENCE = 3` | §B.3.3 | 观察不同题池规模下的判定稳定性 |
| G3 | `n_gradient_problems` 的归一化比例 5% / 15% | §B.4.3 | 需要「gradient 题过少」的失败样本（现有 4 个任务均不满足） |
| G4 | `ppo_kl` 的「爆炸」阈值取 1e-4 | §D.2.4 | 读 slime CI / 训练稳定性文档的标准阈值 |
| G5 | 工具侧 `group_index` 在所有 run 中都是全局递增 | §B.8-2 | 已实测 task8（rollout 0→0~15, 1→16~31, 2→32~47）；其它 run 待查 |
| G6 | `pass_count_K_of_8` 中的 8 = 组大小 | §C.2.3.3 | 读 slime 源码；或观察 groups_per_step≠16 的 run |
| G7 | `removed/count_*` 恒为 attempt 级、`zero_std/count_*` 恒为 group 级 | §C.2.3.1-2 | 已在 task8 三步验证；建议在 task4/task9 复核 |

## 附录 H：与其它交付物的接口（不属本交付物，但需对齐）

| 其它交付物 | 本设计对其的依赖 | 本设计对其提供 |
| --- | --- | --- |
| `utils/agent_budget_profile.py` | — | §B.4.2 的 `time_budget_ratio` / `per_turn_seconds_growth` 可由它算；本设计提供 **TB 零成本版本**作为替代/兜底 |
| `utils/difficulty_profile.py` | **强依赖**：§B.3.4 的三态判定由它提供 | 本设计给出**阈值**（0.9 / 0.1）与**任务级汇总**的边界 |
| `utils/failure_reason_taxonomy.py` 扩展 | `time_budget_exceeded` 需独立于 `context_exhausted` | §A.6 已登记该缺陷；§A 的 R1/R2 不依赖其 category，规避风险 |
| `docs/PLAN_reward_uplift.md` v2 | — | 本设计的 §0 三处修正、§A 的四分类、§C 的 tag 目录、§D 的 exclusions 应写进 v2 |
| `templates/reward_diagnosis.html` | — | §B.4.6 的 `RATIO_LEVER_KEYS` 需加新比例型 key；§D.2.2 的新卡片 |
| `service/train_health_service.py` 的 controller | — | 需新增 `GET /api/tasks/{task_id}/train-health.json` |

---

**本文档到此结束。所有数字均来自 `/tmp/ri_reward_evidence.md` 实证简报、本次在真实 run 目录（task2/4/8/9/11）上的补充实测，或已明确标注为「设计假设」（附录 G）。未创建/修改任何仓库文件。**

---

# 第三部分 · 页面规格与测试清单

# 交付物 2：`templates/reward_diagnosis.html`「时间预算健康度」卡片设计规格

## 2.1 卡片在页面中的位置

**放在现有「奖励还能怎么涨」杠杆区块（`#leverSection`）之前，作为新的第 3 个 section。**

现有页面 section 顺序与调整后：

| 顺序 | section id | 标题 | 变更 |
| --- | --- | --- | --- |
| 1 | `#gradientSection` | 有效梯度组 | 不变（但图例加 `dropped` 色块） |
| 2 | `#splitSection` | 奖励分解 | 不变 |
| **3（新增）** | **`#budgetSection`** | **时间预算健康度** | **插入** |
| 4 | `#leverSection` | 奖励还能怎么涨 | 不变（但内容扩充） |
| 5 | `#behaviorSection` | 行为失败卡片 | 不变（但 `abort` 拆为两张卡） |
| 6（第二批） | `#conclusionSection` | 诊断结论 | 新增（在最顶部） |

**理由**：

1. **因果顺序**：时间预算 → 组污染 → 样本枯竭是一条链。杠杆是「结论/建议」，必须放在「证据」之后，
   否则用户先看到建议再看到证据，说服力弱。
2. **杠杆依赖**：`time_budget_ratio` / `per_turn_seconds_growth` / `dropped_group_ratio` 三条新杠杆的
   `value` 与 `status` 都由本卡片产出。杠杆区块紧跟其后，读起来是「看证据 → 看行动」的连贯动线。
3. **与「有效梯度组」区隔**：现有第 1 块讲「组结构」，第 2 块讲「奖励结果」，第 3 块讲「**过程退化**」——
   三块构成「结构 → 结果 → 过程」的递进，正好是用户排查「奖励为什么不涨」的顺序。
4. **不置于最顶**：最顶是「诊断结论」面板（第二批），它是**结论**，应由下面的证据支撑；
   第一批先落地时没有该面板，卡片在第 3 位是过渡期最合理的位置。

## 2.2 复用共享多序列折线模块

**复用对象**：`static/ri_multi_chart.js`（commit `54df1a4` 引入，已在 `templates/base.html` 第 335 行全局引入，
早于子页 inline script 加载，**无需改 base.html**）。

**对外接口**：`window.RiMultiChart.renderLineChart(opts)` / `renderSparkline(opts)` / `colorAt(i)` / `fmtNum` / `fmtPct` / `esc`。

**复用方式**（照抄 `templates/problem_instance.html` 的 `renderTrend()` 模式）：

```
RiMultiChart.renderLineChart({
    host: document.getElementById("budgetChart"),
    readout: document.getElementById("budgetReadout"),
    series: [ {label, color, dash, points: [{x, y, meta}]} ],
    xLabel: "rollout",
    yLabel: "…",
    ariaLabel: "时间预算健康度",
    yFormat: RiMultiChart.fmtNum, // 或 fmtPct
    yMin / yMax: 可选,
    tipFormat: (s, p) => "…",
 legend: true,                    // 卡片用内置图例
})
```

**关键约束**（来自模块实现）：

| 约束 | 说明 |
| --- | --- |
| `host` 必须是**真实 DOM 节点** | `renderLineChart` 直接 `host.innerHTML = ...`；`curve_compare.html` 用 `setTimeout(..., 0)` 延后渲染正是为此 |
| `points` 需 `{x, y}` 且两者都是**数字** | 模块用 `isNum()` 过滤；`null` 点会被丢弃（自动断线） |
| `x` 必须是**数值** | 本卡片 X 轴 = `rollout_id`（整数）。**不要**传字符串，否则 `xAt` 计算全错 |
| `dash` 支持 `true` / `null` / 自定义字符串 | 与 `curve_compare.html` 一致：`TAG_DASHES = [null, "6 4", "2 3", "9 4 2 4"]` |
| `legend` 默认 `true` | 模块自带 `.ri-mc-legend` 样式；本卡片**用内置图例**，不另写 `.legend`（避免双图例） |
| `readout` 传入后 hover/click 会写入读数 | 模块只绑 `mouseenter` / `click`，**不绑 `mouseleave`**，读数不会自动复位 → 见 §2.6 |
| 无数据时模块渲染 `<div class="muted">没有可绘制的数据。</div>` | 卡片需自行判断是否用这条兜底文案（见 §2.7） |

**色彩分配**（复用模块 `PALETTE` 的前 4 色，与 `curve_compare.html` 一致）：

| 序列 | 颜色 | 取法 |
| --- | --- | --- |
| 单轮耗时（秒） | `#2d6a9f`（蓝） | `RiMultiChart.colorAt(0)` |
| 轮数（turns） | `#c2410c`（橙） | `RiMultiChart.colorAt(1)` |
| 撞墙率 | `#c62828`（红） | `RiMultiChart.colorAt(3)`（**刻意跳过 colorAt(2) 绿**，绿色留给「健康」判定） |
| 排队占比 | `#b45309`（琥珀） | `RiMultiChart.colorAt(4)` |

> **【设计】为什么用 4 个不同量纲画在同一张图**：模块只支持**单一 Y 轴**（`yAt` 用同一个 `yMin/yMax`）。
> 四个量纲分别是「秒 / 轮 / 比例 / 比例」，直接叠加会把「秒」压扁成一条直线。
>
> **解决方案：拆成 2 张图 + 1 组 chip**，而不是硬塞进一张：
> - **图 A「耗时与轮数」**：`per_turn_seconds_p50`（秒，Y 轴）+ `turns_p50`（轮）→ 量纲差 100 倍以内，
>   但仍然会互相压制。**再拆**：图 A 只画 `per_turn_seconds_p50` + `agent_run_p50`（同为秒，可共用轴）；
>   图 B 只画 `turns_p50` + `peak_context_p50`（归一化为「占上限 %」）。
> - **图 C「比例」**：`time_budget_ratio` + `dropped_group_ratio`（都是 0–1 比例，共用轴）。
>   **【实测修正 · 二次】** 初稿此处画的是 `queue_stolen_ratio`，一版复核又因**误读字段**
> （把 `sandbox_boot_queue` 当成 `sandbox_boot`）把它整个删除。**两版都不对**。
> 最终结论：排队**确实**加剧撞墙（分桶撞墙率 20.7% / 33.9% / 80.0%）→ **保留该序列**，
> 改名为 `queue_wait_ratio`，归**辅 · 吞吐**；与 `time_budget_ratio` 同在图 C（都是 0–1 比例，共用轴）。
>
> **最终定稿：3 张图**（见 §2.3），每张图的 X 轴都是 rollout_id，视觉上纵向对齐 →读起来仍是一张「时间轴」。

## 2.3 可视化内容

### 卡片整体结构

```
┌─ 时间预算健康度 ────────────────────────────────────────────────┐
│ [verdict 徽章]  单轮变慢（per_turn_slowdown）                     │
│ 一句话：单轮生成耗时从 18.1s 涨到 86.0s（+375%），60 分钟预算内   │
│         跑不完 → attempt 被杀 → 整组丢弃 → 训练样本枯竭           │
├──────────────────────────────────────────────────────────────────┤
│ [chip] 撞墙率 62.5%  [chip] 单轮耗时 86.0s  [chip] 轮数 35      │
│ [chip] 排队占比 16.4%  [chip] 预算 3600s  [chip] 数据级别 L2    │
├──────────────────────────────────────────────────────────────────┤
│ 图 A ·耗时（秒）    ← 单轮耗时 + agent_run p50                   │
│ [readout]                                                       │
│ 图 B · 规模（归一化）← 轮数占首值% + 峰值上下文占上限%           │
│ [readout]                                                       │
│ 图 C · 比例（%）    ← 撞墙率 + 排队占比                            │
│ [readout]                                                       │
├──────────────────────────────────────────────────────────────────┤
│ 已排除的病因（3 项）                                             │
│  ✓ 已排除 off-policy  ✓ 已排除熵坍缩  ✓ 已排除 KL 爆炸           │
├──────────────────────────────────────────────────────────────────┤
│ 丢弃原因分布：agent_timeout 80 · too_few_survivors_3 6 · …      │
│ 整组丢弃率 93.8%（15/16 组）· 可训练 attempt 7 / 128            │
└──────────────────────────────────────────────────────────────────┘
```

### 图 A · 耗时（秒）——复用折线模块

| 序列 | 字段 | 颜色 | 线型 |
| --- | --- | --- | --- |
| 单轮生成耗时 | `per_turn_seconds_p50` | `#2d6a9f` | 实线（`dash: null`） |
| agent 总耗时 | `agent_run_p50` | `#c2410c` | 虚线（`dash: "6 4"`） |

- **Y 轴**：`yFormat: RiMultiChart.fmtNum`，单位标注在 `yLabel: "秒"`。
- **预算线**：在3600s 处画一条水平参考线。模块**不支持参考线** → **【设计】** 用 `series` 追加一条
  常量序列 `{label: "预算上限 3600s", color: "#c62828", dash: "2 3", points: [{x: min, y: 3600}, {x: max, y: 3600}]}`
  实现。这是最小侵入的做法，且 `tipFormat` 能显示「预算上限」。
 - ⚠️ 注意：该序列会让 Y 轴自适应到 3600+，把「单轮耗时 18–86s」的细节压扁。
 **【设计】** 因此图 A **不画**预算线，预算线改放到**图 C**（撞墙率图）或以 chip 展示。
    **最终决定**：预算线只在图 C 画（用一条 `time_budget_ratio` 上界的参考序列，或干脆用文案）。
    → **图 A 只画 2 条序列**。

### 图 B · 规模（归一化）

| 序列 | 字段 | 归一化 | 颜色 |
| --- | --- | --- | --- |
| 轮数 | `turns_p50` | 除以**首值**（相对倍数），或直接用原值 | `#7c3aed` |
| 峰值上下文 | `peak_context_p50` | 除以 `max_context`（占上限 %） | `#0891b2` |

**【设计】** 两个量纲（轮数35–62、上下文 64K–88K）差异大，**归一化到「相对首值 %」**再同轴：

- `turns_p50 / turns_p50[0] * 100`
- `peak_context_p50 / peak_context_p50[0] * 100`

这样两张图的 Y 轴都变成「相对首值 %」，**且两张图可以用同一套 Y 轴范围** → 图 A（秒）、
图 B（相对 %）、图 C（比例）三图视觉节奏一致。

> **【设计】** 归一化后 Y 轴从 100% 起步，**下降**的线（轮数下降、上下文下降）会掉到 100% 以下，
> 一眼能看出「**轮数和上下文都在降 → 不是轮数爆炸、不是上下文打满**」。
> 这正是 §3.0.3 的核心判定证据，视觉上必须显眼。

### 图 C · 比例

| 序列 | 字段 | 颜色 | 线型 |
| --- | --- | --- | --- |
| 撞墙率 | `time_budget_ratio` | `#c62828` | 实线 |
| 整组丢弃率 | `dropped_group_ratio` | `#b45309` | 虚线 `"6 4"` |

- **Y 轴**：`yFormat: RiMultiChart.fmtPct`，`yMin: 0`，`yMax` 取 `Math.max(0.2, 最大值)`。

### 数据来源与降级

| 级别 | 可用图 | 缺失表现 |
| --- | --- | --- |
| L2（summary） | A / B / C 全部 | — |
| L1（.pt） | A / B / C 全部 | — |
| L3（仅 TB） | 只有 B 的轮数（`selected/attempt/completed_turns/p50`）+ C 的撞墙率（`outcome/time_budget_exceeded`） | **图 A 显示「单轮耗时需要 .pt / summary 数据，此任务不可用」**；图 B 只有1 条线 |

## 2.4 退化模式判定（`verdict`）的展示

### 徽章设计

位置：卡片标题右侧，同一行内。样式复用页面既有的 `.chip` + `.alert` 视觉语言。

| verdict | 中文文案 | 颜色 | 图标 | 说明文案 |
| --- | --- | --- | --- | --- |
| `per_turn_slowdown` | **单轮变慢** | `#c62828`（红，severe） | `ri-icon-trending-up` | 「模型每轮输出变多 → 单轮耗时暴涨 → 60 分钟预算内跑不完」 |
| `turn_count_blowup` | **轮数爆炸** | `#c62828`（红，severe） | `ri-icon-repeat` | 「每轮很快但轮数暴涨 → 撞的是轮次上限而非时间预算」 |
| `context_exhaustion` | **上下文打满** | `#ef6c00`（橙，warn） | `ri-icon-triangle-alert` | 「上下文接近上限 → 撞的是上下文而非时间预算」 |
| `healthy` | **未见退化** | `#2d6a4f`（绿，ok） | `ri-icon-check`（若不新增则用 `ri-icon-trophy`） | 「单轮耗时与轮数均在正常范围，未见撞墙」 |

**图标来源**：`static/icons/` 已有 Lucide 图标，`static/ri_icons.css` 已生成对应 class。
已确认可用：`ri-icon-trending-up` / `ri-icon-trending-down` / `ri-icon-repeat` / `ri-icon-triangle-alert` /
`ri-icon-timer` / `ri-icon-clock-alert` / `ri-icon-activity` / `ri-icon-trophy` / `ri-icon-gauge` / `ri-icon-ruler`。
→ **`ri-icon-check` 不存在**，绿色徽章用 `ri-icon-trophy` 或 `ri-icon-activity`。

> **【设计】** 徽章本身**不做颜色以外的区分**（避免只靠颜色传达信息，兼顾色觉障碍）：
> 同时给出**中文文案**（「单轮变慢」）+ **图标** + **英文 code**（小字 `per_turn_slowdown`）。
>页面既有 `.rel-explains_drop` / `.rel-improving_together` 已有同类「颜色 + 文字」的做法，沿用该风格。

### 判定依据行（徽章下方一行 muted 小字）

```
判定依据：单轮耗时 +375% · 单轮 token +180% · 撞墙率 5.0%→62.5% · 轮数 -44%（未增）
```

**【设计】** 必须把**四个判定条件的实测值**都列出来，因为：
- 让人能**自己判断**判定是否合理（而不是盲信）；
- 四个值里任何一个与结论矛盾时，用户能立刻发现。

### 确定性不足时的降级

若样本不足以判定（如 rollout 数 < 3，或首/末窗口数据点< 2）：

| 情况 | 徽章 | 文案 |
| --- | --- | --- |
| 数据点< 3 | **灰色** `#8a939c` | 「样本不足，无法判定退化模式（N 个 rollout）」 |
| 无撞墙但单轮耗时上升 | 橙色 | 「单轮耗时上升但尚未撞墙（撞墙率 X%），暂不判定退化」 |

## 2.5 新增杠杆条目的展示

### 新增 6 条杠杆（key 与 §3.0.7(f) 一致）

| key | 标题 | 展示格式 | 判据 |
| --- | --- | --- | --- |
| `time_budget_ratio` | 时间预算耗尽占比 | 百分比 | 末值 > 0.05 或 verdict = per_turn_slowdown |
| `per_turn_seconds_growth` | 单轮耗时增幅 | 百分比（带 + 号） | 首末增幅 > 50% |
| `dropped_group_ratio` | 整组丢弃率 | 百分比 | > 0.3 |
| `gradient_density` | 梯度密度 | 百分比 | < 0.5 |
| `n_gradient_problems` | 可学习题数量 | 整数 | 数量过少 |

### 展示一致性要求（必须与现有杠杆卡片完全一致）

现有杠杆卡片结构（`templates/reward_diagnosis.html` 第 358 行）：

```html
<div class="lever-card ${status}">
    <span class="lever-value">…</span>
    <div class="lever-title">…</div>
    <div class="lever-body"><b>现象：</b>…<br/><b>建议：</b>…</div>
</div>
```

**新杠杆必须复用同一套 `.lever-card` / `.lever-title` / `.lever-value` / `.lever-body` class 与同一段
`renderLevers()` 渲染函数**，只在 `RATIO_LEVER_KEYS` 集合里补充新 key：

```
现有 RATIO_LEVER_KEYS = {learnable_ratio, timeout_ratio, abort_ratio, saturated_pool_ratio, hard_pool_ratio}
新增：time_budget_ratio, per_turn_seconds_growth, dropped_group_ratio, gradient_density
```

`n_gradient_problems` 是**整数**，**不加入** `RATIO_LEVER_KEYS`，按 `num()` 展示（与现有 `n_problems` 一致）。

### 新杠杆的卡片内容（与现有卡片同结构）

| key | 现象文案（自动填值） | 建议文案（固定） |
| --- | --- | --- |
| `time_budget_ratio` | 「X% 的 attempt 因撞上时间预算被杀（末值，峰值 Y%）」 | 「提高 agent 时间预算（60 → 90 分钟）或限制单轮输出长度」 |
| `per_turn_seconds_growth` | 「单轮生成耗时从 X s 涨到 Y s（+Z%）」 | 「对输出长度加惩罚 / 在 prompt 约束单轮输出；检查是否出现长度崩塌式膨胀」 |

| `dropped_group_ratio` | 「X% 的 GRPO 组被整组丢弃（可训练 attempt 从 A 降到 B）」 | 「查丢弃原因分布：agent_timeout → 查时间预算；too_few_survivors → 查存活阈值；zero_std → 查题目池」 |
| `gradient_density` | 「每步 16 组中只有 X 组是 mixed（有对有挂）」 | 「缩小题目池（把已掌握题移出）；或提高 groups_per_step；目标是每步至少 50% 的组是 mixed」 |
| `n_gradient_problems` | 「题目池 N 题中只有 M 题可学习」 | 「补充中间难度题（从累计通过率 0.2–0.5 的题里挑）」 |

### 排序

现有 `build_reward_levers` 按 `status` 排序：`bad` → `warn` → `unknown` → `ok`。
**新杠杆沿用同一排序**，不特殊对待。**但**：`time_budget_ratio` 在 task8 上应为 `bad`，
会排在最前 —— 这正是期望效果（第一主因排第一）。

### 「丢弃原因分布」的展示位置

放在杠杆区块**之后**（不是卡片内），作为一个独立的 `.muted` 小表：

```
丢弃原因（末 rollout） agent_timeout 76 · zero_std_1.0 34 · too_few_survivors_3 9 · ungradable 1
```

**【设计】** 数据来自 `.pt` 的 `sample.metadata.train_drop_reason` 直方图（§3.1.4），
或 TB 的 `rollout/removed/count_*`（L3 降级时）。用 chip形式（复用 `.chip` class），
按数量降序，数量最多的 chip 用红色边框。

## 2.6 「已排除的病因」（exclusions）的展示

### 位置

**放在卡片的底部，杠杆区块之前**（卡片内）。

### 视觉形式

用一行 `.chip`，每项一个：

```
✓ 已排除 off-policy    ✓ 已排除熵坍缩    ✓ 已排除 KL 爆炸
```

样式：
- 前景色 `#2d6a4f`（绿），背景 `#f0f7f2`，边框 `#2d6a4f`（浅）
- 前缀用 `ri-icon-trophy`（已有）或纯文本 `✓`
- 文案直接用 service 返回的中文（`exclusions[].title`），**前端不二次加工**

### 不可排除时怎么办

**【设计】** 当某条排除条件不满足（如 task11 的 logprob 差峰值 0.1054 越界）：

| 情况 | 展示 |
| --- | --- |
| 该病因**已排除** | 绿色 chip：「✓ 已排除 off-policy」+ hover 显示证据数值 |
| 该病因**未排除**（有异常信号） | **不放在 exclusions**，而是作为一条 **verdict**（红/橙）出现在卡片的判定区，标题形如「train/rollout 脱节（logprob 差峰值 0.1054 > 0.1）」 |

→ **exclusions 区域只放「已排除」的**。这保证「已排除的病因」标题名实相符，
不会出现「已排除 XXX」但 XXX 其实存在的问题。

### 交互

- chip **可 hover**：显示该排除项的完整证据（`exclusions[].evidence`），例如  「训练 logprob 与 rollout logprob 全程 0.017–0.044（阈值 0.1）」。
- chip **不可点击**（无跳转目标）。

## 2.7 交互细节

### hover 提示

| 元素 | hover 行为 |
| --- | --- |
| 折线上的数据点 | 模块自带 `<title>` tooltip（`tipFormat` 决定文本）。**【设计】** `tipFormat` 格式统一为：`{序列名} · rollout {x} · {格式化值}` |
| 折线上的数据点（点击） | 模块绑定了 `click` → 也会写 readout。**注意**：点击后读数不会自动恢复，**【设计】** 读数区初始显示末点，点击后显示被点位置 |
| 判定徽章 | hover 显示完整判定依据（四条件实测值） |
| exclusions chip | hover 显示证据数值 |
| 杠杆卡片 | 沿用现有行为（无特殊交互） |

**【设计】** `tipFormat` 的具体格式：

```javascript
tipFormat: function (s, p) {
    const m = p.meta || {};
    return s.label + " · rollout " + p.x + " · " + (s.isRatio ? RiMultiChart.fmtPct(p.y) : RiMultiChart.fmtNum(p.y) + " " + (s.unit || ""));
}
```

### 点击跳转

**【设计】** 新增 3 个跳转目标，全部**跳到已有页面**，不新建锚点页：

| 触发元素 | 跳转目标 | URL形态 |
| --- | --- | --- |
| 丢弃原因 chip `agent_timeout` | **失败分析页**，按 `agent_outcome=time_budget_exceeded` 过滤 | `/api/tasks/{id}/failures?agent_outcome=time_budget_exceeded` |
| 杠杆 `time_budget_ratio` | **失败分析页**（同上） | 同上 |
| 杠杆 `gradient_density` / `n_gradient_problems` | **问题分析页** | `/api/tasks/{id}/problem-analysis` |
| 判定徽章「单轮变慢」 | **失败分析页** + **训练曲线页**（自定义 tag） | `/api/tasks/{id}/train-curve?custom=rollout/outcome/time_budget_exceeded` |

**【设计】重要约束**：失败分析页**当前不支持 URL query 过滤**（`templates/failure_overview.html` 的`fAgentOutcome` / `fReason` 是页面内 `<select>`，无 URL 同步）。因此：

> **跳转链接的落地方式**：第一版**只跳到页面并带 `?` 参数 + 在页面 JS 里读取该参数并预置对应 select 值**。
> 这需要在 `failure_overview.html` 里加一小段「从 URL 读初始过滤条件」的逻辑（约 5 行），
> 属于**同一批次**的改动，不能只做链接不做预置（否则点了跳过去还要手动再选一遍，体验断裂）。

若不打算改 `failure_overview.html`（缩小改动面），**降级方案**：跳转链接改为**复制到剪贴板 + 提示**，
或直接跳到失败分析页**不带过滤**，并在链接 title 里说明「请在筛选器中选择『时间预算耗尽』」。

### 空数据 / 样本不足的降级展示

| 情况 | 表现 |
| --- | --- |
| **完全没有 attempt 数据**（`n_attempts == 0`） | 整张卡片替换为 `<div class="muted">该任务还没有 rollout 数据。</div>`，**不显示任何图** |
| **有数据但无 `phase_times`**（L3 级别） | 图 A 显示 `<div class="muted">单轮耗时需要 .pt / summary 数据，此任务不可用。</div>`；图 B 只有轮数一条线；图 C 只有撞墙率一条线。**卡片顶部必须标注当前数据级别**（chip：`数据级别：仅 TensorBoard`） |
| **rollout 数 < 3** | 三张图仍画，但**判定徽章显示灰色**「样本不足，无法判定退化模式（N 个 rollout）」；**不画**图 B 的归一化（分母不稳） |
| **`per_turn_seconds_p50` 全为 None**（turns 缺失） | 图 A 隐藏「单轮耗时」序列，只画 `agent_run_p50`；判定条件「单轮耗时增幅」标记为**不可评估**，不参与判定 |
| **某条序列某点缺失** | 模块自动跳过该点（`isNum` 过滤）→ 折线自然断开。**【设计】** 不做插值，保持「数据缺失可见」 |
| **exclusions 为空**（无 TB 数据） | 整个「已排除的病因」区块隐藏，不显示空标题 |

### 加载与性能

- **【设计】** 卡片数据随 `build_reward_diagnosis` 的 JSON一次返回（新增 `budget_profile` 字段），
  **不额外发请求**。理由：现有页面的 4 个 section 全部来自同一个 `reward-diagnosis.json`，
  新增第 5 个 section 不应引入新请求模式（避免 N+1）。
- **【设计】** `budget_profile` 计算走 L2（summary sidecar 缓存）。若 L2 缺失需要读 `.pt`，
  **不在请求内同步读**——而是：
  - 首次请求返回 `budget_profile.status = "pending"` + `pending_hint`（「正在解析 rollout 数据，首次较慢」）；
  - 后台按 `schedule/` 任务补齐（第三批）；
  - 或第一版**只支持 L2/L3**，L1降级为「不可用」。

  → **决策：第一版只实现 L2 + L3**（summary 与 TB 都足够），L1（读 `.pt`）留到第三批，
  避免首屏被大文件阻塞。卡片需明示「数据级别：summary / 仅 TensorBoard」。

## 2.8 视觉一致性检查清单

| 检查项 | 要求 |
| --- | --- |
| 折线渲染 | **必须**用 `window.RiMultiChart.renderLineChart`，不自写 SVG |
| 折线配色 | 用 `RiMultiChart.colorAt()` 或 `RiMultiChart.PALETTE`，与 `curve_compare.html` 同源 |
| 图例 | 用模块内置 `legend: true`（`.ri-mc-legend`），**不**另写 `.legend` |
| 杠杆卡片 | **必须**复用 `.lever-card` / `.lever-title` / `.lever-value` / `.lever-body` 与现有 `renderLevers()` |
| 杠杆数值格式 | 百分比类 key 加入 `RATIO_LEVER_KEYS`；其余用 `num()` |
| chip | 复用既有 `.chip` class |
| 区块结构 | 复用既有 `.section` / `.section h2` / `h2 small` 模式（新卡片 `<h2>时间预算健康度 <small id="budgetMeta"></small></h2>`） |
| 读数行 | 复用既有 `.readout` class（`font-variant-numeric: tabular-nums`） |
| 空态 | 复用既有 `.muted` class |
| 严重程度色 | 复用既有 `.lever-card.ok/.warn/.bad/.unknown` 的色板：`#2d6a4f` / `#e0a800` / `#c62828` / `#8a939c` |
| 图标 | 只用 `static/icons/` 里已存在、`ri_icons.css` 已生成 class 的图标；**不新增** SVG |
| 响应式 | 图容器用既有 `.line-chart` class（`width:100%; .line-chart svg { width:100%; height:auto }`） |
| 移动端 | 三张图纵向堆叠（`.cards` grid 或直接 block），不并排 |

---



# 交付物 3：测试用例清单

> 风格：`unittest`，文件放 `tests/`，函数名 `test_<行为>`，
> 类名 `Test<被测单元>`，用 `self.assertEqual` / `assertAlmostEqual` / `assertTrue` / `assertIsNone`。
> 现有风格参考：`tests/test_reward_levers.py`（含 `_row()` 构造辅助）、`tests/test_reward_diagnosis_service.py`、
> `tests/test_tb_metric_catalog.py`、`tests/test_failure_reason_taxonomy.py`。
> **注意**：`scan_func_md.py` 的 `SKIP_DIR_NAMES` 包含 `tests`，测试文件**不进 func.md**。

## 3.1 `tests/test_slime_grpo_semantics.py`（新建）

> 覆盖：组语义分类（含 `dropped`）、三条一等公民指标。

| 测试函数名 | 验证点 | 构造输入 |
| --- | --- | --- |
| `test_mixed_group` | 有对有挂 → `mixed` | `rewards=[1.0, 0.0, 1.0, 0.0]`, `statuses=["completed"]*4` |
| `test_all_pass_group` | 全过 → `all_pass` | `rewards=[1.0]*8` |
| `test_all_fail_group` | 全挂 → `all_fail` | `rewards=[0.0]*8` |
| `test_infra_only_group` | 全 abort → `infra_only` | `statuses=["aborted"]*8`, `rewards=[None]*8` |
| `test_dropped_when_surviving_below_threshold` | 存活 3 < 4 → `dropped` | `rewards=[1.0, 0.0, 1.0, None, None, None, None, None]`, `statuses` 前 3 个 completed、后 5 个 aborted |
| `test_dropped_when_containing_infra_abort` | 含基建类 abort → `dropped`（即使存活 ≥4） | `rewards=[1.0,0.0,1.0,0.0,1.0,None,None,None]`, `statuses` 含1 个 `aborted`+`abort_reason="exception:SandboxApiException"` |
| `test_not_dropped_when_surviving_exactly_threshold` | 存活 == 4 → **不** dropped（边界） | `rewards=[1.0,0.0,1.0,0.0,None,None,None,None]`, `statuses` 前 4 completed、后 4 aborted（`abort_reason=agent_time_budget_exceeded`） |
| `test_time_budget_abort_alone_does_not_drop_group` | **关键反例**：仅 time_budget abort、存活 ≥4 → 不是 dropped | 同上，存活 4 |
| `test_dropped_when_zero_std` | 存活 5 但奖励全等 → `dropped` | `rewards=[1.0]*5 + [None]*3` |
| `test_min_surviving_configurable` | `min_surviving=3` 时存活 3 → 不 dropped | `min_surviving=3` |
| `test_group_gradient_contribution` | mixed=1.0，其余=0.0 | 遍历 `GROUP_KINDS` |
| `test_group_action_hint_differs_all_pass_vs_all_fail` | all_pass 与 all_fail 的处置文案**不同**（这是 v2 的核心修正） | `group_action_hint("all_pass") != group_action_hint("all_fail")` |
| `test_group_action_hint_for_dropped_mentions_waste` | dropped 的处置文案含「浪费」 | 断言子串 |
| `test_group_drop_reason_returns_histogram` | `group_drop_reason` 返回 drop reason 直方图 | `drop_reasons=["agent_timeout","agent_timeout","too_few_survivors_1"]` |

## 3.2 `tests/test_agent_budget_profile.py`（新建）

> 覆盖：时间预算三种退化模式 + 数据契约 + 预算推断。

| 测试函数名 | 验证点 | 构造输入 |
| --- | --- | --- |
| `test_per_turn_slowdown_verdict` | 单轮变慢 → `per_turn_slowdown` | 20 个 rollout：turns 62→35（下降）、per_turn 18.1→86.0、tok/turn 349→976、time_budget_ratio 0→0.30 |
| `test_healthy_verdict` | 无退化 → `healthy` | task9 形态：turns 59→60、per_turn 23.5→35.7、tok/turn 384→602、time_budget_ratio 全程 ≤0.02 |
| `test_context_exhaustion_verdict` | 上下文打满 → `context_exhaustion` | peak_context 88.5K→接近 max_context(100K) 且增幅 >10%；`context_exhausted` 占比 0.4 |
| `test_turn_count_blowup_verdict` | 轮数爆炸 → `turn_count_blowup` | turns 30→90（+200%）、per_turn 增幅 <10%、time_budget_ratio 低 |
| `test_context_exhaustion_wins_over_per_turn_slowdown` | **优先级**：上下文打满优先于单轮变慢 | 两个条件同时满足 → 期望 `context_exhaustion` |
| `test_turn_count_blowup_wins_over_per_turn_slowdown` | **优先级**：轮数爆炸优先于单轮变慢 | 两个条件同时满足 → 期望 `turn_count_blowup` |
| `test_per_turn_slowdown_requires_all_four_conditions` | 缺任一条件 → `healthy` | 逐一去掉：turns 增幅、tok/turn 增幅、撞墙率、turns 不增 |
| `test_budget_seconds_inferred_from_wall_clock` | 从撞墙样本推断 3600 | `agent_run` = [3606.9, 3613.8, 3619.7, 3903.2] → `budget_seconds == 3600` |
| `test_budget_hit_ratio_near_one_when_all_hit` | 全部撞墙 → `budget_hit_ratio == 1.0` | `agent_run` 全 ≥ 3600 |
| `test_queue_wait_ratio_computed` | 排队占比 = queue / attempt_total | `sandbox_boot_queue=300`, `attempt_total=1500` → 0.2 |
| `test_infra_overhead_ratio_computed` | 预算外占比 = (total−agent)/total | `total=3764.3`, `agent=2825` → ≈0.25 |
| `test_series_rollout_ids_sorted` | 输出按 rollout_id 升序 | 乱序输入 |
| `test_empty_input_returns_empty_series` | 空输入 → 空 series + `degradation.verdict == "healthy"` | `[]` |
| `test_missing_phase_time_excluded_from_timing` | 缺 `agent_run` 的 attempt 不计入耗时但计入 `n_attempts` | 一条 `agent_run=None` |
| `test_first_last_window_uses_percentile_window` | 首末取前/后 20% 中位数（而非单点） | 首点为极端离群值时不主导判定 |
| `test_degradation_reports_growth_ratios` | `degradation` 含 `per_turn_seconds_growth` 等增幅字段 | — |

## 3.3 `tests/test_difficulty_profile.py`（新建）

> 覆盖：题目池三态与采样密度。

| 测试函数名 | 验证点 | 构造输入 |
| --- | --- | --- |
| `test_empty_returns_zero_counts` | 空输入 → `n_problems == 0`，各 ratio 为 None | `[]` |
| `test_three_states_by_cumulative_pass_rate` | 三态分类正确 | 题 a累计通过率 1.0 → stable_pass；题 b 0.5 → gradient；题 c 0.0 → stable_fail |
| `test_boundary_pass_rate_0_9_is_stable_pass` | 边界：0.9 → stable_pass | 9 pass / 10 attempts |
| `test_boundary_pass_rate_0_1_is_stable_fail` | 边界：0.1 → stable_fail | 1 pass / 10 attempts |
| `test_abort_attempts_excluded_from_denominator` | abort/timeout 不进分母 | 题 a：5 pass + 3 abort → 通过率 1.0（分母 5） |
| `test_appearances_mean_computed` | 每题平均出现次数 | 3 题，出现次数 2/3/4 → mean = 3.0 |
| `test_appearance_gap_mean_computed` | 每题出现间隔均值 | 题在 rollout 0/10/20出现 → gap = 10 |
| `test_sampling_density_alias` | `sampling_density == appearances_mean` | — |
| `test_appears_once_flagged_as_insufficient` | 出现次数 < 2 的题标记「数据不足」 | 单次出现的题 |
| `test_n_gradient_problems_counted` | 可学习题数量 | 三态各 2 题 → `n_gradient_problems == 2` |
| `test_missing_instance_id_ignored` | 无 `instance_id` 的行忽略 | `instance_id=""` |

## 3.4 `tests/test_reward_diagnosis_service.py`（扩展）

> 覆盖：组语义分类（含 dropped）在 service 层的落地 + 三条新指标。

| 测试函数名 | 验证点 | 构造输入 |
| --- | --- | --- |
| `test_gradient_groups_includes_dropped_count` | `series` 每行含 `n_dropped` | 组含 1 个 abort + 4 存活 → dropped 判定 |
| `test_zero_gradient_ratio_computed` | `zero_gradient_ratio` = (all_pass+all_fail)/n_groups | 2 mixed + 1 all_pass + 1 all_fail → 0.5 |
| `test_wasted_rollout_ratio_computed` | `wasted_rollout_ratio` = dropped / n_groups | 1 dropped / 4 组 → 0.25 |
| `test_gradient_density_computed` | `gradient_density` = mixed / n_groups | 2 mixed / 4 → 0.5 |
| `test_note_mentions_new_semantics` | `note` 文案含「零梯度」「被丢弃」 | — |
| `test_build_reward_diagnosis_includes_budget_profile` | 顶层载荷含 `budget_profile` 键 | mock 行 |
| `test_build_reward_diagnosis_includes_pool_profile` | 顶层载荷含 `pool_profile` 键 | — |

## 3.5 `tests/test_reward_levers.py`（扩展）

> 覆盖：新增杠杆条目与状态判定。

| 测试函数名 | 验证点 | 构造输入 |
| --- | --- | --- |
| `test_always_returns_all_levers`（**修改现有**） | lever key 集合扩充为 13 个（原 7 + 新 6） | `build_reward_levers([])` |
| `test_new_levers_have_phenomenon_and_action` | 6 条新杠杆都有 phenomenon / action | — |
| `test_time_budget_ratio_high_is_bad` | `time_budget_ratio` > 0.05 → `bad` | — |
| `test_per_turn_seconds_growth_high_is_bad` | 增幅 > 50% → `bad` | — |
| `test_queue_wait_ratio_high_is_bad` | 占比 > 15% → `bad`（辅 · 吞吐） | — |
| `test_gradient_density_low_is_bad` | 密度 < 0.5 → `bad` | — |
| `test_n_gradient_problems_present` | `n_gradient_problems` 在 values 中 | — |
| `test_levers_sorted_by_status_with_new_keys` | 排序仍按 status，新增 bad杠杆排在最前 | — |

## 3.6 `tests/test_problem_analysis_service.py`（扩展）

> 覆盖：通过率三口径。

| 测试函数名 | 验证点 | 构造输入 |
| --- | --- | --- |
| `test_pass_rate_three_caliber_keys` | 返回含 `pass_rate_strict` / `pass_rate_gradable` / `pass_rate_slime_effective` | — |
| `test_pass_rate_strict_includes_abort_in_denominator` | strict 分母含 abort | 1 pass + 1 abort → 0.5 |
| `test_pass_rate_gradable_excludes_abort` | gradable 分母不含 abort | 同上 → 1.0 |
| `test_pass_rate_slime_effective_excludes_dropped_groups` | slime_effective 只统计未被丢的组 | 组 A 全过（存活 8）、组 B 存活 2 → 只算 A |
| `test_pass_rate_slime_effective_uses_group_average` | 组内平均（先组内算率再平均） | 组 A 4/8=0.5、组 B 2/8=0.25 → 0.375 |
| `test_note_mentions_three_calibers` | `_PAGE_NOTE` 更新为三口径说明 | — |

## 3.7 `tests/test_failure_reason_taxonomy.py`（扩展）

> 覆盖：`time_budget_exceeded` 与 `context_exhausted` 分离。

| 测试函数名 | 验证点 | 构造输入 |
| --- | --- | --- |
| `test_time_budget_exceeded_gets_budget_exhausted_category` | `category == "budget_exhausted"` | `normalize_reason(status="aborted", abort_reason="agent_time_budget_exceeded")` |
| `test_context_exhausted_stays_timeout` | `category == "timeout"` | `normalize_reason(status="truncated", error_code="slime_context_limit_reached", agent_outcome="context_exhausted")` |
| `test_budget_exhausted_from_agent_outcome_without_abort_reason` | 无 abort_reason 时靠 agent_outcome 判定 | `normalize_reason(status="completed", agent_outcome="time_budget_exceeded", reward=0.0)` |
| `test_budget_exhausted_wins_over_timeout_token_match` | `_TIMEOUT_TOKENS` 的 `"time_budget"` 不再抢先 | 同上 → `budget_exhausted` 而非 `timeout` |
| `test_category_labels_cover_budget_exhausted` | `CATEGORY_LABELS` 含 `budget_exhausted` | — |
| `test_reason_label_distinguishes_budget_vs_context` | 两个 reason 的中文标签不同 | `reason_label("budget_exhausted", "agent_time_budget_exceeded")` vs `reason_label("timeout", "slime_context_limit_reached")` |
| `test_ordinary_infra_abort_still_abort` | 非 time_budget 的 abort 仍是 `abort` | `abort_reason="exception:SandboxApiException"` |
| `test_sdk_error_still_abort` | 回归：`sdk_error` → abort | `normalize_reason(agent_outcome="sdk_error")` |

## 3.8 `tests/test_rollout_pt_parser_no_torch.py`（扩展）

> 覆盖：`train_drop_reason` 读取位置修正。

| 测试函数名 | 验证点 | 构造输入 |
| --- | --- | --- |
| `test_train_drop_reason_read_from_metadata` | 从 `sample["metadata"]["train_drop_reason"]` 读到 | dump 中 sample 带 metadata |
| `test_train_drop_reason_falls_back_to_top_level` | 顶层键兼容回退 | sample顶层有 `train_drop_reason` |
| `test_train_sample_stats_returns_drop_reason_histogram` | 返回直方图 | 3 条不同 reason |
| `test_train_sample_stats_no_longer_always_kept` | **回归**：不再是恒 100% 保留 | 全部 sample 带 metadata reason |

## 3.9 `tests/test_tb_metric_catalog.py`（扩展）

> 覆盖：TB tag 目录分类（按实测 458 tag）。

| 测试函数名 | 验证点 | 构造输入 |
| --- | --- | --- |
| `test_removed_prefix_classified` | `rollout/removed_ratio` → `removed` | — |
| `test_population_prefix_classified` | `rollout/population/trainable_attempt_count` → `population` | — |
| `test_funnel_prefix_classified` | `rollout/funnel/loss_eligible_attempts` → `funnel` | — |
| `test_outcome_prefix_classified` | `rollout/outcome/time_budget_exceeded` → `outcome` | — |
| `test_behavior_prefix_classified` | `rollout/selected/attempt/completed_turns/p50` → `behavior` | — |
| `test_zero_std_prefix_classified` | `rollout/zero_std/count_0.0` → `zero_std` | — |
| `test_train_prefix_classified` | `train/entropy_loss` → `training` | — |
| `test_describe_removed_ratio_has_source` | `describe_tb_tag` 返回含 slime 源码出处 | — |
| `test_describe_unknown_still_not_invented` | **保留原测试** | 未知 tag → `DESC_UNKNOWN` |
| `test_describe_funnel_loss_eligible` | 新增 tag 有中文说明 | — |
| `test_zero_std_count_minus_one_is_unknown` | `rollout/zero_std/count_-1.0` → unknown（v1 写错的名字） | — |
| `test_train_kl_loss_is_unknown` | `train/kl_loss` → unknown（该 tag 不存在） | — |

## 3.10 `tests/test_tb_series_reader.py`（扩展）

> 覆盖：前缀批量取 + 异常检测。

| 测试函数名 | 验证点 | 构造输入 |
| --- | --- | --- |
| `test_read_scalar_series_by_prefix` | 按前缀取全部 tag | 写 3 个 `rollout/removed/*` tag |
| `test_read_by_prefix_returns_empty_for_unknown` | 未知前缀 → 空 dict | `"nope/*"` |
| `test_find_anomalies_detects_trend` | 趋势型异常被检出 | 单调上升 0.1→0.9 |
| `test_find_anomalies_ignores_single_point_spike` | 单点尖刺**不**报（规则基于趋势+阈值） | 0.1,0.1,0.9,0.1,0.1 |
| `test_find_anomalies_returns_empty_on_flat_series` | 平坦序列无告警 | 全 0.5 |

## 3.11 `tests/test_train_health_service.py`（新建）

> 覆盖：训练健康度 `verdicts` / `exclusions`。

| 测试函数名 | 验证点 | 构造输入 |
| --- | --- | --- |
| `test_empty_returns_empty_sections` | 空数据 → 各段为空、`verdicts == []` | — |
| `test_verdict_time_budget_exhausted` | `removed_ratio` 末值 0.8 + `count_agent_timeout` 76 → 一条 bad verdict | 合成序列 |
| `test_verdict_training_sample_exhausted` | `loss_eligible_attempts` 末值 7 → bad verdict | — |
| `test_verdict_zero_std_overflow` | `count_zero_std_1.0` 末值 34 → warn verdict | — |
| `test_verdict_entropy_collapse` | `entropy_loss` 首末降幅 > 50% → warn verdict |0.6→0.2 |
| `test_verdict_offpolicy_when_logprob_exceeds` | `logprob_abs_diff` 峰值 0.1054 → bad verdict | — |
| `test_exclusion_offpolicy_when_logprob_normal` | 全程 ≤0.1 → 输出 `off_policy` 排除项 | 0.02–0.045 |
| `test_no_offpolicy_exclusion_when_logprob_exceeds` | **关键**：峰值 0.1054 → **不**输出 off-policy 排除项 | task11 形态 |
| `test_exclusion_entropy_collapse_when_entropy_rises` | 熵上升 → 输出 `entropy_collapse` 排除项 | 0.15→0.64 |
| `test_no_entropy_exclusion_when_entropy_falls` | 熵下降 → 不输出该排除项，输出 verdict | 0.6→0.2 |
| `test_exclusion_kl_explosion` | ppo_kl ~1e-6 → 输出 `kl_explosion` 排除项 | — |
| `test_exclusion_not_infra_abort` | agent_timeout 占 abort 70% → 输出 `not_infra_abort` 排除项 | — |
| `test_funnel_reconcile_error_computed` | `reconcile_error` = \|removed − (1−le/launched)\| 中位数 | 合成序列 |
| `test_alignment_source_is_rollout_id` | `alignment.source == "rollout_id"` | — |
| `test_missing_tags_do_not_crash` | 缺 `removed/count_agent_timeout` 不抛异常 | 稀疏 tag 集 |

## 3.12 `tests/test_action_recommendation_service.py`（新建，若第一批实现反事实估算）

| 测试函数名 | 验证点 | 构造输入 |
| --- | --- | --- |
| `test_verdicts_sorted_by_severity` | bad → warn → info | — |
| `test_dataset_actions_three_lists` | remove / downsample / keep_boost 三清单 | — |
| `test_counterfactual_recomputes_mixed_ratio` | 反事实 mixed 占比 > 当前 | — |
| `test_param_actions_carry_evidence_required` | 每条 action 有 `evidence_required` | — |

---

# 交付物 4：func.md 同步要求

## 4.1 需要重跑的命令

```bash
# 在仓库根目录执行
python scripts/scan_func_md.py
# 或（包装脚本，会自动 cd 到仓库根）
bash scripts/make_func_md.sh
```

**CI 校验**（只读，退出码 1 表示不一致）：

```bash
python scripts/scan_func_md.py --check
```

**约束**（来自 `.cursor/rules/rollout-insight-conventions.mdc`）：
- **禁止**手动编辑 `func.md`——下次扫描会被覆盖；
- 需要排除某些目录 / 函数时，改 `scripts/scan_func_md.py` 的 `SKIP_DIR_NAMES` / `SKIP_DIR_PATTERNS`；
- 本方案**不需要**改扫描脚本（新增模块都在 `utils/` `service/` 五个扫描目标目录内）；
- `schedule/` 在黑名单里 → `schedule/schedule_rollout_align.py` **不会**进 func.md（符合现有约定）。

## 4.2 func.md 需要新增登记的公开能力清单

### `utils/agent_budget_profile.py`（新建）

| 函数名 | 用途 |
| --- | --- |
| `build_budget_profile` | 按 rollout 聚合 attempt 级预算画像（series） |
| `classify_degradation` | 退化模式判定（`per_turn_slowdown` / `turn_count_blowup` / `context_exhaustion` / `healthy`） |
| `infer_budget_seconds` | 从撞墙样本 `agent_run` 分布推断墙钟预算（实测 3600s） |
| `summarize_budget_windows` | 首/末 20% 窗口的中位数汇总（用于趋势判定） |
| `BUDGET_VERDICTS` | 四种 verdict 常量（模块级公开常量） |

### `utils/slime_grpo_semantics.py`（新建）

| 函数名 | 用途 |
| --- | --- |
| `classify_grpo_group` | 两级组语义判定（含 `dropped`） |
| `group_gradient_contribution` | 组对训练的梯度贡献（mixed=1.0，其余 0.0） |
| `group_action_hint` | 组类型 → 处置建议文案 |
| `group_drop_reason` | dropped组的丢弃原因直方图 |
| `GROUP_KINDS` | 五种组类型常量 |

### `utils/difficulty_profile.py`（新建）

| 函数名 | 用途 |
| --- | --- |
| `classify_difficulty` | 题目累计通过率三态（`stable_pass` / `gradient` / `stable_fail`） |
| `summarize_pool_health_v2` | 题目池健康度 + 采样密度 + 梯度密度 |
| `build_resample_lists` | `remove` / `downsample` / `keep_boost` 三张重采样清单 |
| `DIFFICULTY_STATES` | 三态常量 |

### `service/train_health_service.py`（新建）

| 函数名 | 用途 |
| --- | --- |
| `build_train_health` | 训练健康度聚合（removed / funnel / training / behavior / outcome + verdicts + exclusions） |
| `build_verdicts` | 判定规则（趋势 + 阈值） |
| `build_exclusions` | 排除性结论（含「条件不满足则不输出」的反例处理） |

### `service/action_recommendation_service.py`（新建，第二批）

| 函数名 | 用途 |
| --- | --- |
| `build_recommendations` | verdict + 数据集动作 + 参数动作 + 反事实估算 |

### 就地扩展（func.md 中对应条目会被重新生成 / 更新描述）

| 模块 | 新增公开能力 | 变化 |
| --- | --- | --- |
| `utils/tb_metric_catalog.py` | `TAG_CATALOG`（新增常量）、`TB_TAG_GROUPS`（前缀 → 类别映射）、`describe_tb_tag_source`（返回 slime 源码出处） | `classify_tb_tag` / `describe_tb_tag` 的描述更新（新增 6 类） |
| `utils/tb_series_reader.py` | `read_scalar_series_by_prefix`、`find_anomalies` | 新增 2 个函数 |
| `utils/failure_reason_taxonomy.py` | `budget_exhausted_label`、`BUDGET_EXHAUSTED_REASONS` | `CATEGORY_LABELS` / `normalize_reason` 描述更新；新增 2 个公开能力 |
| `utils/rollout_pt_parser.py` | `summarize_drop_reasons`（drop reason 直方图） | `summarize_train_sample_stats` 描述更新（不再恒 100%） |
| `service/reward_diagnosis_service.py` | `summarize_dropped_groups`、`summarize_gradient_density` | `classify_group_kind` 描述更新（五类）；`summarize_gradient_groups` / `build_reward_levers` / `summarize_pool_health` 描述更新 |
| `service/problem_analysis_service.py` | `summarize_pass_rates`（三口径） | `summarize_pass_visits` 描述更新（三口径并存） |
| `service/cross_task_compare_service.py` | — | `summarize_reward_guidance` 描述更新（新增 6 个 value 透出） |
| `templates/reward_diagnosis.html` | — | **不进 func.md**（`templates` 在黑名单里）；但需同步更新 `README.md` §5.3 |

## 4.3 不需要登记的（确认清单）

| 项 | 原因 |
| --- | --- |
| `templates/reward_diagnosis.html` | `SKIP_DIR_NAMES` 含 `templates` |
| `static/ri_multi_chart.js` | `SKIP_DIR_NAMES` 含 `static`；**且该模块已有，无需改动** |
| `schedule/schedule_rollout_align.py` | `SKIP_DIR_NAMES` 含 `schedule` |
| `tests/test_*.py` | `SKIP_DIR_NAMES` 含 `tests` |
| `db/db_attempt.py`（若加 `agent_exit_code` / `agent_tool_calls`） | `db/db_*.py` 只登记 ORM 类名 / `__tablename__` / 文件名，**加字段不产生新条目**；但 `func.md` 的 ORM 表会自动重新生成，需一并 commit |
| `service/step_alignment_service.py` | **v2 取消该模块**，不登记 |
| `utils/slime_attempt_classifier.py` | **v2 取消该模块**，不登记 |

## 4.4 提交要求

按 `.cursor/rules/rollout-insight-conventions.mdc` §6，feat / fix / refactor commit **必须**包含 `func.md`：

```text
feat(reward): 新增时间预算退化画像与组语义修正

- 业务说明：...
- python scripts/scan_func_md.py 已重跑；func.md 同步更新
- 测试：tests/test_agent_budget_profile.py 等12 个文件 / 约 90 个用例
```

---

## 附：执行过程中对 v1 的 4 处关键纠正（额外说明）

1. **`train_drop_reason` 并非死指标** —— 字段在 `.pt` 的 `sample.metadata` 中，v1 的工具读的是顶层 `sample["train_drop_reason"]`，键层级错了。
2. **丢弃是两级机制** —— attempt 级 `remove_sample`（`agent_timeout`）+ 组级 `too_few_survivors_N` / `zero_std_*`；
判据是「**存活 attempt 数 < 4**」，不是「含 abort」（实测组 884 反例）。
3. **不需要 `step_alignment_service`** —— `rollout/funnel/*` 已按 rollout step 记录，`removed_ratio ≈ 1 − loss_eligible/launched`
   在同 step 上可对账（误差 < 1%），`rollout_id` 就是天然 join key。
4. **v1 §2.2 的 tag 名多处有误** —— `train/kl_loss` 不存在、`rollout/zero_std/count_-1.0` 不存在、
   `rollout/dynamic_filter/drop_*` 不存在；且 v1 与 plan 都漏掉了 `rollout/funnel/*`、`rollout/population/*`、
   `rollout/outcome/*`、`rollout/selected/attempt/*` 四组共 119 条 tag。

