# rollout-insight 训练奖励提升增强方案（v2）

> 面向 slime + sglang 的 agentic RL 训练。
> 目标：把 rollout-insight 从「rollout 数据可视化工具」升级为「**奖励不涨时的归因与行动引擎**」。
>
> 调研基线：
> - 本仓库 commit 历史（v0.1.0 → v0.5.2，HEAD = `91557c2`）
> - slime 源码 `/mnt/si003126ndy1/default/shenyichong/code/slime`（本批任务实际使用的版本）
> - sglang 侧经 slime `slime/rollout/sglang_rollout.py` 接入
>
> **v2 说明**：本文是 v1 的**修订版**，不是重写。v1 的骨架、§1.2 强项清单、§5 复用对应关系表、§6 风险与边界全部保留；
> v2 用 **2026-11-23 对线上训练产出的真实数据分析**验证并纠错了 v1 的若干关键结论，
> 新增「真实数据验证」与「agent 时间预算退化」两章，并把落地顺序按实测优先级重排。
>
> 本文只做方案设计，不含代码实现。

**阅读约定（v2 新增）**：全文用以下标记区分「实测事实」与「设计决策」：

| 标记 | 含义 |
| --- | --- |
| **【实测】** | 来自 2026-11-23 真实数据解析（DB / summary JSON / TensorBoard / `.pt` dump），可直接复核 |
| **【设计】** | 本方案提出的口径、阈值、模块划分、交互规格，属待实现的设计决策 |
| **【假设】** | 尚无数据支撑、需在实施时验证的推断 |

---

## 0. 结论速览

| 维度 | 现状 | 核心缺口 |
| --- | --- | --- |
| 诊断能力 | 有「有效梯度组 / 奖励分解 / 行为失败卡片」 | 判定口径与 slime 训练实际**不一致**（见 §2.1），结论会误导 |
| 跨任务对比 | 仅 TensorBoard 多 run 叠曲线 | 没有把 TB 曲线与 rollout 数据**对齐到同一条因果链** |
| 训练进程侧 | 只读 `eval/*` 评测 tag | 完全看不到 `train/*`、`rollout/*`（KL / entropy / zero_std / removed / funnel） |
| 可执行性 | 给「杠杆」文字建议 | 不可直接执行，缺参数建议、样本过滤建议、重采样建议 |
| 数据黑洞 | v1 称 `train_drop_reason` 恒为 0 | **v1 判断错误**：该字段在 `.pt` `sample.metadata` 中**确实存在**，只是工具读的位置不对 |

**最高优先级的一件事（v2 已修正）**：

> v1 判定「当前最大的问题是口径不一致」。**v2 实测证明这个判断只对了一半**。
>
> **真正的第一主因是「agent 时间预算耗尽 → 训练样本枯竭」**：
> task8 的 1425 例 abort **全部**是 `agent_time_budget_exceeded`（agent 侧行为退化），
> 沙箱类仅 14 例（1.1%）。v1 主张的「abort 升高 = 基建问题，先修沙箱/driver」**被推翻**。
>
> 口径不一致（v1 §3.1）依然要修，但它是**次要问题**：它会让归因方向偏，而不是让奖励不涨。
> 详见 §1.5（真实数据验证）与 §3.0（时间预算退化完整方案）。

---

## 1. 现状盘点：已有什么

### 1.1 数据链路

```
slime sglang_rollout.generate_rollout_async
  └─ dynamic_filter (check_surviving_reward_nonzero_std)
       └─ 通过的组 → data → _save_debug_rollout_data → rollout_{id}.pt
                    └─ rollout-insight: RolloutScanService.scan_task
                         ├─ ri_rollout_table   (1 行 = 1 个 .pt)
                         ├─ ri_attempt_table   (1 行 = 1 个 GRPO attempt)
                         ├─ ri_segment_table   (1 行 = 1 个 sample/turn 片段)
                         └─ ri_segment_turn_table (懒加载，正文)
```

现有分析页面对应关系：

| 页面 | 路由 | service | 回答的问题 |
| --- | --- | --- | --- |
| 任务详情 | `/api/tasks/{id}` | `task_controller` | 结局分布、类型柱、进入训练 |
| 问题分析 | `/api/tasks/{id}/problem-analysis` | `problem_analysis_service` | 按 epoch 的通过率、sglang 响应 tool_call 统计 |
| 奖励诊断 | `/api/tasks/{id}/reward-diagnosis` | `reward_diagnosis_service` | 有效梯度组、奖励分解、行为失败卡片 |
| 失败分析 | `/api/tasks/{id}/failures` | `failure_analysis_service` | 失败聚类、逐 attempt 详情、轨迹 |
| 训练曲线 | `/api/tasks/{id}/train-curve` | `train_curve_service` | TB scalar 曲线、epoch 聚合 |
| 洞察 | `/api/insights` | `insight_page_service` | TB scalar 清单 |

### 1.2 已具备的强项（应保留，不要重造）

- **`.pt` 无 torch 依赖加载**（`utils/analyze_dump.py` 的 zip/pickle + `_TorchStub`）——工程上很扎实。
- **sidecar 注解缓存**（`task_annotation_service`）——`.pt` 大文件只解析一次。
- **abort / timeout / reward 三态分离**（`resolve_sample_reward` / `classify_sample_outcome`）——避免把 abort 当失败，这是很多同类工具的通病，本项目做对了。
- **epoch 对齐校验**（`_check_epoch_windows`）——知道「一题一个 epoch 窗口」是否成立。
- **TB 多 run 托管**（`tensorboard_service`）——已用 `--load_fast=false` 修过历史 step 被截断的坑。
- **`utils/tb_series_reader.read_scalar_series`**——**已能读全量 scalar 序列**。v1 §3.2.2 误以为「现有 `read_latest_scalars` 只取最后一个点」，实际上本模块已提供 `list_scalar_tags` / `read_scalar_series` / `series_points`，v1 把已有能力当成了待建能力。
- **共享多序列折线模块 `static/ri_multi_chart.js`**（commit `54df1a4` 引入）——`window.RiMultiChart.renderLineChart` / `renderSparkline`，纯 SVG 无第三方依赖，已在 `base.html` 全局引入。「问题分析 · 单题趋势」与「多任务曲线对比」共用同一份实现。v2 的时间预算卡片**直接复用它**，不再新写折线渲染。

### 1.3 数据边界（必须先认清，否则一切分析都建立在错误前提上）

**边界 1：被 dynamic_filter 丢弃的组不落盘。**

`slime/rollout/sglang_rollout.py` 中：

```python
dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
if not dynamic_filter_output.keep:
    metric_gatherer.on_dynamic_filter_drop(reason=..., phase_seconds=...)
    state.remaining_batch_size -= 1
    continue          # ← 该组直接丢弃，不进 data
...
return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect())
```

而 `_save_debug_rollout_data(data, ...)` 落盘的是**过滤后**的 `data`。
`all_data`（含被丢弃组）只在内存中，不写盘。

当前训练脚本启用的是：

```bash
# musa_run/run_openhands_qwen3_coder_30b_swe_gym.sh
export DYNAMIC_SAMPLING_FILTER_PATH="...check_no_aborted_samples"
```

**v2 更正（重要）**：本批任务实际使用的 filter 路径是 `examples/coding_agent_rl/rollout_filters.check_surviving_reward_nonzero_std`，
其**丢弃粒度是 attempt 级**而不是组级：

```python
# examples/coding_agent_rl/rollout_filters.py :: classify_attempt
_EXCLUDABLE_ABORT_REASONS = ("agent_time_budget_exceeded",)

def classify_attempt(args, member) -> AttemptDecision:
    """Order matters: specific excludable abort reasons are tested BEFORE
    the generic aborted status, because the agent time-budget path sets
    ``status = ABORTED`` and would otherwise be swallowed by the generic test."""
    for sample in samples:
        reason = str((sample.metadata or {}).get("abort_reason") or "")
        if reason in _EXCLUDABLE_ABORT_REASONS and _enabled("SWE_EXCLUDE_TIME_BUDGET"):
            return AttemptDecision(keep=False, reason="agent_timeout")
    ...
```

`check_surviving_reward_nonzero_std` 的组级拒绝只有三种：

| 拒绝原因 | 触发条件 | 组后果 |
| --- | --- | --- |
| `aborted_{reason}` | 组内**存活** attempt 带 `abort_reason` 或 `status=ABORTED`（**基建类**，不含已被 `classify_attempt` 排除的 time_budget） | **整组丢弃** |
| `invalid_evaluator_payload` | 组内存活 attempt 的 `evaluation.reason` 无效 | **整组丢弃** |
| `too_few_survivors_{n}` | 存活 attempt 数 < `SWE_DAPO_MIN_SURVIVING_ATTEMPTS`（默认 4） | **整组丢弃** |
| `zero_std_{v}` | 存活 attempt 奖励全等 | **整组丢弃** |

**关键区分（v2 补充，v1 完全没有）**：
`agent_time_budget_exceeded` 走的是 **attempt 级 `remove_sample` 标记**（`train_drop_reason="agent_timeout"`），
它**不会**单独把整组踢掉；但它会把该组的**存活 attempt 数压到 4 以下**，从而触发 `too_few_survivors_*` 把**整组**丢掉。
所以「时间预算耗尽 → 组污染 → 整组丢弃 → 训练样本枯竭」是一条**间接但确定的**链路。

**边界 2：`train_drop_reason` 在 `.pt` metadata 中确实存在（v1 判断错误）。**

v1 §1.3 边界 2 称「全仓 grep 无任何写入点」「工具侧 `summarize_train_sample_stats` 恒为 100% 保留，是无信息量的死指标」。

**v2 实测推翻**：

```text
【实测】task8 rollout_55.pt 中 sample.metadata 的 train_drop_reason 实际分布：
  agent_timeout                80
  aborted:openhands_infra_exit_2  8
  too_few_survivors_3            6
  zero_std_1.0                  5
  zero_std_0.0                  5
  too_few_survivors_1            4
  too_few_survivors_2            2
```

原因是 v1 找错了位置：`train_drop_reason` 由 `rollout_filters._mark_removed` 写入
`sample.metadata["train_drop_reason"]`，而 `utils/rollout_pt_parser.summarize_train_sample_stats`
读的是**顶层 `sample["train_drop_reason"]`**，键层级不对，所以永远取不到。
**这个指标本来就有信息量，是工具读错了。**

**边界 3：`.pt` 里的 reward 是 raw reward，不是训练用的 advantage。**

- `.pt` 落盘：`sample.reward`（评测原始分）
- 训练用：`normalize_rewards_by_attempt` 做 GRPO 组内减均值 / 除标准差
- `raw_reward` 在 `slime/ray/rollout.py:716` 只是「若 metadata 已有 raw_reward 就优先用」，**默认没人写这个 metadata key**

**边界 4：组级 GRPO 标准化会掩盖组内差异。**

```python
# slime/rollout/reward_utils.py
rewards = rewards - rewards.mean()
if grpo_std_normalization:
    rewards = rewards / (rewards.std() + 1e-6)
```

一个全挂组（8/8 全 0）标准化后 advantage 全为 0 → **零梯度**，但它**占用了 rollout 预算**。
一个全过组（8/8 全 1）同理。

---

## 1.5 【新增章节】真实数据验证 —— v1 结论的逐条裁定

### 1.5.1 验证方式说明

**【实测】** 2026-11-23 对线上训练产出做了全量解析，数据源：

| 数据源 | 规模 | 解析方式 |
| --- | --- | --- |
| `rollout_insight.db` | **11 任务 / 470 rollout / 60610 attempt** | 直连 SQLite 查 `ri_attempt_table` / `ri_rollout_table` |
| `attempt_artifacts/summaries/*.json` | task4 **8752 份**、task8 **8320 份**、task9 **6022 份**、task11 **6404 份** | 逐份 JSON 解析 `agent_outcome` / `abort_reason` / `phase_times` |
| TensorBoard 事件目录 | task8/task4/task11 各 **458 个** scalar tag，task9 **457 个** | `utils/tb_series_reader.read_scalar_series` 读全量序列 |
| `generated/rollout_dumps/rollout_*.pt` | task8 **64 rollout / 8213 sample** | `utils.rollout_pt_parser.load_pt`（无 torch 依赖）全量解析 |

DB 侧 task_id → 任务对应关系（已核对 `ri_task_table.rollout_root`）：

| task_id | 任务名 | 题目池 | rollout 数 | attempt 数 |
| --- | --- | --- | --- | --- |
| 1 | `c021-f1base-sqh-20261104-113329` | 96 | 6 | 768 |
| 2 | `c021-e18-f1-ctx128k` | 338 | 33 | 4224 |
| 3 | `c021-e19-curriculum-lr1e6` | 640 | 40 | 5120 |
| 4 | `c021-mixed688-train-sqh-20261112-164259` | **688** | 63 | 8064 |
| 6 | `c021-v06-rand128-sqh-20261112-173926` | 128 | 8 | 1024 |
| 7 | `c021-v06-rand128-train-sqh-20261113-185340` | 128 | 47 | 6016 |
| 8 | `c021-v06-keep64-train-sqh-20261117-173649` | **64** | 64 | 8192 |
| 9 | `c021-s1b2-lr3e6-20261114-231111` | **32** | 48 | 6144 |
| 10 | `c021-v06-rand128-train-sqh-20261117-171712` | 128 | 111 | 14208 |
| 11 | `c021-v06-band64-finish-train-sqh-20261118-174234` | **64** | 50 | 6400 |

> **【实测】** task5 在 DB 中只有 1 个 rollout / 450 attempt（189 题池），样本量不足以做统计对比，本方案所有横向对比**排除 task5**。

### 1.5.2 v1 结论逐条裁定表（证实 / 推翻 / 降级）

| # | v1 结论（原文位置） | 裁定 | 实证依据 |
| --- | --- | --- | --- |
| 1 | 「abort 升高 = 基建问题，先修沙箱/driver」（§2.4、§3.4.2） | **❌ 被推翻** | **【实测】** task8 全量 8320 份 summary 中 `abort_reason` 分布：`agent_time_budget_exceeded` **1425**、`openhands_infra_exit_2` **219**、`exception:SandboxApiException` **7**、`exception:SandboxExecStalled` **5**、`evaluator_total_timeout` **2**、`openhands_infra_exit_137` **1**。**沙箱类合计仅 14 例（1.1%）**。task11 同样：`agent_time_budget_exceeded` **987** / 总 1158 abort（85%）。 |
| 2 | 「`train_drop_reason` 在当前 slime 版本中不存在，恒为 100% 保留，是死指标」（§1.3 边界 2、§3.1.4） | **❌ 被推翻** | **【实测】** task8 `rollout_55.pt` 的 `sample.metadata.train_drop_reason` 实际有 `agent_timeout` 80 / `too_few_survivors_3` 6 / `zero_std_1.0` 5 / `zero_std_0.0` 5 / `aborted:openhands_infra_exit_2` 8 等 7 种取值。字段存在且信息量高，是**工具读错了键层级**（见 §1.3 边界 2）。 |
| 3 | 「被 dynamic_filter 丢弃的组**整组丢弃**」（§1.3 边界 1、§3.1.1） | **⚠️ 修正（口径细化）** | **【实测】** 丢弃是**两级**的：① attempt 级 `remove_sample`（`agent_timeout` / `ungradable` / `truncated` / `turn_cap`）；② 组级 `DynamicFilterOutput(keep=False)`（`aborted_*` / `invalid_evaluator_payload` / `too_few_survivors_*` / `zero_std_*`）。**【实测】** task8 全量 8213 sample 中 `remove_sample=True` 共 **4851（59.1%）**；1024 个组中**整组被丢 548 个（53.5%）**、部分丢 221 个、完全保留 255 个。 |
| 4 | 「工具口径与 slime 不一致，结论会指向错误方向」（§2.1） | **✅ 完全证实，且更严重** | **【实测】** 工具把 `all_fail` 归为「可学」，slime 实为 `zero_std` 丢弃；且 v1 未发现 **`rollout/removed_ratio` 才是第一主因**（task4 末值 **1.00**、task11 **0.88**）。 |
| 5 | 「训练侧指标完全不可见」（§2.2） | **✅ 证实，且更严重** | **【实测】** 实际有 **458 个** tag（不是 v1 估的几十个），含 `rollout/removed/*`（13 条丢弃原因全集）、`rollout/population/*`（27 条）、`rollout/selected/attempt/*`（67 条）、`rollout/funnel/*`（13 条，v1 完全未提）、`train/*`（16 条）。 |
| 6 | 「丢弃统计前缀是 `rollout/dynamic_filter/drop_*`」（§2.2 表格） | **❌ 被推翻** | **【实测】** 该前缀**不存在**。实际前缀是 `rollout/removed/*`（14 条），另有 `rollout/removed_ratio`。全量 458 tag 中 `rollout/dynamic_filter` 命中数为 **0**。 |
| 7 | 「`rollout/zero_std/count_-1.0` 是全挂组数」（§2.2 表格） | **⚠️ tag 名错误** | **【实测】** 实际 tag 是 `rollout/zero_std/count_0.0` 与 `count_1.0`（**没有 `count_-1.0`**）。task8 末值：`count_0.0`=2、`count_1.0`=2。 |
| 8 | 「`train/kl_loss` 是 actor 与 ref 的 KL」（§2.2 表格） | **⚠️ tag 不存在** | **【实测】** 458 tag 中**没有** `train/kl_loss`。实际有 `train/ppo_kl`（task8 量级 **1e-6 ~ 2.7e-5**）、`train/tis` / `train/tis_abs` / `train/tis_clipfrac`、`train/ois`。 |
| 9 | 「zero_std 是零梯度主因」（§0 / §2.1c） | **⬇️ 降级为次要** | **【实测】** task8 `rollout/removed/count_zero_std_1.0` 首→末 11→**34**、`_0.0` 16→**7**；同期 `count_agent_timeout` 2→**76**（末值峰值 80）。**时间预算才是丢弃首因**。 |
| 10 | 「`pass_rate` 把 abort/timeout 计入分母，与 slime 整组丢弃口径不符」（§2.6c / §3.1.3） | **✅ 证实** | **【实测】** `problem_analysis_service._PAGE_NOTE` 自述「abort 与 timeout 计入分母、不计入分子」，与 slime 的两级丢弃口径确实不符。v2 §2.6 改为三口径并存。 |
| 11 | 「题目池健康度用『末次通过率 ≥0.875 判饱和 / ≤0.125 判太难』」（§3.5） | **❌ 在 task4 失效** | **【实测】** task4 题目池 **688 题**，每题平均只出现 **1.47 次**（最大 2 次），相邻出现间隔均值 **31.1 step**。「末次」≈「首次」，无法反映学习进展。 |
| 12 | 「TB 曲线与 rollout 数据是两条平行线，需 step 对齐」（§2.5） | **✅ 方向正确，但 v1 的对齐方式不必做** | **【实测】** `rollout/funnel/*` 与 `rollout/population/*` 已按 **rollout step** 记录（`rollout/step` = rollout_id），`rollout/funnel/loss_eligible_attempts` 与 `rollout/removed_ratio` 在**同一 step 上**可对账（见 §1.5.4）。**不需要**新建 `step_alignment_service` 做跨源对齐，rollout_id 就是天然 join key。 |
| 13 | 「原文完全未提及」 | **🆕 新增 P0 缺陷** | **agent 时间预算耗尽 → 训练样本枯竭**，是本批任务奖励不涨的**第一主因**。详见 §3.0。 |
| 14 | 「原文完全未提及」 | **🆕 新增 P1 维度** | **题目池大小 → 采样密度 → 梯度密度**。实测 32 题→mixed 77.6%（旧口径；最终定稿口径 **68.4%**，见 §1.5.7）→ task9 唯一成功；688 题→每题只出现 1.47 次。详见 §3.5。 |

### 1.5.3 四个任务横向对比（TB 原始数据，**【实测】**）

| 指标 | task4 (688题) | task8 (64题) | **task9 (32题) 成功** | task11 (64题) |
| --- | --- | --- | --- | --- |
| `rollout/removed_ratio` 首→末 | 0.703 → **1.000**（⚠️ 截至 ro62 实为 0.731） | 0.391 → **0.628**（峰值 0.945） | 0.008 → **0.159** | 0.258 → **0.883** |
| `rollout/population/trainable_attempt_count` 首→末 | 38 → **0**（⚠️ 未扫完，见 §1.5.3.1） | 78 → **47** | 127 → **107** | 95 → **15** |
| `rollout/population/shaped_reward_coverage` 首→末 | 0.984 → **0** | 1.000 → 0.711 | 0.992 → **0.961** | 0.961 → **0.430** |
| `rollout/removed/count_agent_timeout` 末值 | **（该 run 无此 tag）** | **76**（峰值 80） | **（该 run 无此 tag）** | **52**（峰值 74） |
| `rollout/removed/count_zero_std_1.0` 末值 | 23 | 34 | 16 | 34 |
| `rollout/removed/count_zero_std_0.0` 末值 | 4 | 7 | 8 | 7 |
| `rollout/removed/count_ungradable` 末值 | 3 | 1 | 5 | 10 |
| `rollout/funnel/loss_eligible_attempts` 首→末 | 38 → **0** | 78 → **47** | 127 → **107** | 95 → **15** |
| `train/entropy_loss` 首→末 | 0.207 → **0.640** | 0.152 → **0.641** | 0.147 → 0.577 | 0.151 → **0.109** |
| `train/grad_norm` 首→末 | 0.250 → 0.179 | 0.241 → 0.130 | 0.146 → 0.084 | 0.181 → **0.986** |
| `train/train_rollout_logprob_abs_diff` 全程范围 | 0.019 – **0.045** | 0.017 – **0.044** | 0.019 – 0.034 | 0.022 – **0.105** |
| `train/ppo_kl` 量级 | ~1e-6（峰值 2.97e-5） | ~1e-6（峰值 2.70e-5） | ~1e-5 | ~1e-5 |
| `train/pg_clipfrac` 首→末 | 0.053 → 0.029 | 0 → 0.043 | 0.031 → 0.037 | 0 → 0（峰值 0.071） |
| 工具侧 abort 占比（全任务） | 1.9% | **19.4%** | 2.2% | **18.1%** |
| 工具侧 abort 占比（末 rollout） | 3.1% | **28.9%** | 0.0% | **49.2%** |
| 工具侧 truncated 占比（末 rollout） | 8.6% | 0.0% | 5.5% | 1.6% |
| 组类型分布：mixed（旧口径，未计 ungradable） | 48.1% | 50.9% | **77.6%** | 58.0% |
| 组类型分布：all_pass | 37.9% | 28.7% | 20.1% | 28.4% |
| 组类型分布：all_fail | 9.4% | 15.1% | **1.8%** | 8.5% |
| 组类型分布：infra_only（全不可测） | 4.6% | 5.3% | 0.5% | 5.1% |
| **含 abort 的组占比（被丢弃风险）** | 8.6% | **43.4%** | 7.9% | **41.5%** |
| 题目池大小 | 688 | 64 | **32** | 64 |
| 每题平均出现次数 | **1.47** | 16.0 | 24.0 | 12.5 |
| 每题出现间隔（step） | **31.1** | 4.0 | 2.0 | 3.9 |
| **梯度密度（mixed 组占比，旧口径）** | 48.1% | 50.9% | **77.6%** | 58.0% |
| **梯度密度（最终定稿口径）** | **35.6%** | **35.6%** | **68.4%** | — |

> **【实测】** 结论：
> - **task9 是唯一的正例**：`removed_ratio` 全程 ≤0.16、`trainable_attempt_count` 稳定 100+、`shaped_reward_coverage` 0.96、`all_fail` 仅 1.8%、含 abort 组仅 7.9%。
> - **task4 与 task11 的失败模式不同**（见 §1.5.5），不能用同一套结论解释。

### 1.5.3.1 【新增 · 重要】扫描完整性 —— 「末值 removed_ratio=1.0」有两解，必须区分

**【实测】** 用 `.pt` 文件数与 DB 的 `ri_rollout_table` 覆盖对比：

| task | DB 已扫 rollout | `.pt` 文件数 | DB attempt 数 | **未入库的 rollout** |
| --- | --- | --- | --- | --- |
| 1 | 0–5（6 个） | 9 | 768 | **[6, 7, 8]** |
| 2 | 0–32 | 33 | 4224 | — |
| 3 | 0–39 | 40 | 5120 | — |
| **4** | 0–**62**（63 个） | **65** | 8064 | **[63, 64]** |
| 6 | 0–7 | 8 | 1024 | — |
| **7** | 0–**46**（47 个） | **50** | 6016 | **[47, 48, 49]** |
| 8 | 0–63 | 64 | 8192 | — |
| 9 | 0–47 | 48 | 6144 | — |
| 10 | 0–110 | 111 | 14208 | — |
| 11 | 0–49 | 50 | 6400 | — |

**【实测】** task4 的 rollout 63/64 在 `ri_attempt_table` 中是 **0 行**，而 `.pt` 里有 128 / 130 个样本。

**【实测 · 根因】差异的主因是「`.pt` 在扫描后被重跑覆盖」，不是扫描丢数据。**

`ri_rollout_table` 记录了每个 `.pt` 的 `pt_size_bytes` / `pt_mtime`（扫描时的指纹）。
与磁盘实际比对：

| task | DB 记录数 | 指纹一致 | **已被重写** | 文件不存在 |
| --- | --- | --- | --- | --- |
| 1 | 6 | 6 | 0 | 0 |
| 2 | 33 | 33 | 0 | 0 |
| **3** | 40 | 37 | **3** | 0 |
| **4** | 63 | 60 | **3** | 0 |
| 6 | 8 | 8 | 0 | 0 |
| 7 | 47 | 47 | 0 | 0 |
| 8 | 64 | 64 | 0 | 0 |
| 9 | 48 | 48 | 0 | 0 |
| 10 | 111 | 111 | 0 | 0 |
| 11 | 50 | 50 | 0 | 0 |

**实例（task4 rollout 0）**：

| | `.pt` 文件大小 | mtime | DB 记录 |
| --- | --- | --- | --- |
| DB 扫描时 | 141,560,377 B | 2026-11-12 17:46 | — |
| **磁盘现状** | **67,830,457 B** | **2026-11-16 11:44** | 141,560,377 B / 11-12 17:46 |

→ **训练在 11-16 重跑了 rollout 0，`.pt` 被新的产物覆盖**，而 DB 仍是 11-12 的旧快照。

**【实测】直接后果**：同一 `(group_index, attempt_id)` 在 DB 与 `.pt` 里对应**不同的样本**：

| rollout 0 的 idx | `.pt` 现状 | DB 记录（同一键） |
| --- | --- | --- |
| 0 | `aborted` / reward 0.0 | `completed` / reward **1.0** |
| 1 | `aborted` / reward 0.0 | `completed` / reward **1.0** |
| 4 | `aborted` / reward 0.0 | `completed` / reward **1.0** |
| 6 | `completed` / reward 0.0 | `truncated` / reward **1.0** |

→ **DB 里的 43 个 abort（rollout 0）在新版 `.pt` 里根本不存在**，
  反之新版 `.pt` 的 43 个 abort 也没进 DB。这不是「丢数据」，是**两份不同运行的数据被混看**。
  （逐 rollout 复核：ro0 `.pt` 43 abort / DB 0；ro2 `.pt` 3 / DB 8 —— ro2 的「DB 更多」
  正是 group 41 那批被键空间错位误记为 aborted 的通过样本。）

**【实测】abort 保真度（DB abort 数 vs `.pt` abort 数）**：

| task | `.pt` abort | DB abort | 组内 abort 数不一致的组 |
| --- | --- | --- | --- |
| **task4** | **376** | **157** | **36 / 1040** |
| task8 | 1589 | 1589 | **0 / 1024** |
| task9 | 137 | 137 | **0 / 768** |

→ **task4 的 abort 统计被低估**；task8 / task9 完全一致（未被重写）。

**【实测】task4 的 376 vs 157 必须按两段拆开**（否则会把两种原因混为一谈）：

| 段落 | `.pt` abort | DB abort | 差 | 原因 |
| --- | --- | --- | --- | --- |
| **已扫描段（ro0–62）** | **195** | **157** | **38（19.5%）** | **`.pt` 被重跑覆盖**（ro0/ro1/ro2）→ 旧行与新文件混存 |
| **未扫描段（ro63–64）** | **181** | **0** | 181 | **扫描未完成**（不是丢失，是没扫） |
| 合计 | 376 | 157 | 219 | — |

→ **「低估 58%」这个说法不准确**：219 的缺口里，**181 来自「没扫」**（补扫即可），
  只有 **38 来自「被重写覆盖」**（需要先清旧行再入库）。
→ **但对 task4 的结论影响有限**：ro0–62 段内 abort 仍在（195 个里存了 157 个），
  说明 task4 的 abort 问题**在工具侧是可见的**，不因这 38 个而改变结论方向。


**【实测】键空间混用（次要，仅 5 个样本）**：扫描器的入库键是 `(group_index, attempt_id)`，
其中 `attempt_id = sample.rollout_id`，缺失时回落到 `sample["index"]`。
当**同一组内**同时存在 `rollout_id=None` 的样本（aborted）与带真实 `rollout_id` 的样本时，
两者落在**不同的键空间**（idx 空间 vs 全局 attempt id 空间），产生错位。

实例（task4 rollout 2 的 group 41）：

| sample idx | `sample.rollout_id` | 扫描键 | 入库结果 | 真实状态 |
| --- | --- | --- | --- | --- |
| 72 | `None` | (41, **72**) | ✅ IN（记为 aborted/reward=0） | aborted |
| 73 | **329** | (41, 329) | ❌ MISS | completed, **reward=1.0** |
| 74 | **330** | (41, 330) | ❌ MISS | completed, **reward=1.0** |
| 75 | **331** | (41, 331) | ❌ MISS | completed, **reward=1.0** |
| 76 | **332** | (41, 332) | ❌ MISS | completed, **reward=1.0** |
| 77 | `None` | (41, **77**) | ✅ IN | aborted |
| 78 | `None` | (41, **78**) | ✅ IN | aborted |
| 79 | **335** | (41, 335) | ❌ MISS | completed, **reward=1.0** |

→ DB 里 group 41 存的是 attempt_id **72–79**，全部记为 `aborted / reward=0.0`；
而**真实通过的 5 个样本（329/330/331/332/335）根本没入库**。

**【实测】影响面**：
- task4：**93 个组**存在键空间混用，其中 **5 个真实通过样本丢失**
- task9：**60 个组**存在键空间混用，但 **0 个样本丢失**
- task8：**0 个样本丢失**

**【实测】差异的精确分解（task4）**：

| 来源 | 数量 | 性质 |
| --- | --- | --- |
| ① key collision（同 attempt 多 segment，DB 按 attempt 去重） | 53 | **正常行为**，非缺失 |
| ② 键空间混用导致的错位丢失 | **5** | **真 bug** |
| ③ 未扫描的 rollout | 2 个（258 样本） | 扫描未完成 |
| ④ **`.pt` 被重跑覆盖**（3 个 rollout：ro0/ro1/ro2） | 整个 rollout | **数据版本不一致**（主因） |

→ **结论**：task4 的 DB 与磁盘 `.pt` **不是同一次运行的产物**，
   任何「DB vs `.pt`」的逐样本比对在 task3 / task4 上都**不成立**。

**关键澄清（直接影响 §1.5.3 的解读）**：

> task4 的 `removed_ratio` 末值 **1.000** 与 `trainable_attempt_count` 末值 **0**
> **不能直接读作「这一步一个样本都不剩」** —— 其中包含**「扫描未完成」**的成分：
> DB 缺 rollout 63/64，而 TB 的 step 轴覆盖到 64。
>
> **但两者的结论方向一致**：task4 在**已扫完的 0–62 段**里，
> `removed_ratio` 也已从 0.703 升到 **0.73**（ro62），`trainable_attempt_count` 已降到个位数。
> → **「样本枯竭」这个结论依然成立**，只是**幅度被末值高估了**（真实末值应参考 ro62 而非 ro64）。
>
> **另外**：task4 的**工具侧 abort 统计**（末 rollout 3.1%）与 TB 的
> `rollout/removed/count_agent_timeout` 不可直接对照 —— 前者来自**旧版 `.pt` 快照**，
> 后者来自**当前 run**。这正是 v2 §1.5.3 把「工具侧」与「TB 侧」分列展示的原因。

**【设计】** 因此：
1. **工具侧一切「末值 / 峰值」统计必须先校验扫描完整性**，并在页面显式提示
   「本任务还有 N 个 rollout 未入库，当前显示的是截至 rollout M 的数据」。
2. **新增「扫描完整性」卡片**（放任务详情页）：显示 `已扫 / 总数`、未入库的 rollout 列表、
   **以及「`.pt` 已被重跑覆盖」的提示**（用 `pt_size_bytes` / `pt_mtime` 与磁盘比对）、
   以及「补扫」入口（复用已有的 `POST /api/tasks/{task_id}/rescan`）。
3. **对比表里的 task4 末值应标注 `截至 rollout 62`**，避免与 task9/task11（已扫完）直接横比。
4. **【P1 · 代码修复】修正键空间混用**：`utils/rollout_pt_parser` 的 attempt 键应改为
   **组内统一序号**（组内第 N 个 attempt → 键用组内序号，而非全局 `sample.rollout_id` 或全局 idx），
   或在组内检测到混用时**统一改用 idx 空间**。
   现状是「组内混用两个键空间」→ 真实通过样本被误记为 aborted（**会系统性低估通过率**）。
   修复后需**回填**已入库任务的历史数据（task4 至少要修这 5 个样本）。
5. **【P1】扫描时记录并校验 `.pt` 指纹**：`ri_rollout_table` 已有 `pt_size_bytes` / `pt_mtime`，
   `DBRolloutService.upsert_with_fingerprint` 也**按指纹**做 upsert
   （`db/db_rollout_service.py` 的 `and_(pt_path, pt_size_bytes, pt_mtime)` 过滤），
   但**扫描器在读 `.pt` 之前没有用指纹做「文件已变更 → 旧数据需失效」的判定**，
   于是 `.pt` 被重写后，**旧 attempt 行仍留在 DB 里**，形成「新旧混合」。
   → 应补上：**读 `.pt` 前先比对指纹**，不一致则先 `delete_by_rollout` 清理该 rollout 的旧行，
     再重新入库（或标记 `stale` 并跳过）。
   → **好消息**：`RolloutScanService.scan_task(force=True)` 内部会调 `_purge_task_data`
     清空该 task 的 rollout/attempt/segment 后再重扫（`service/rollout_scan_service.py`），
     → **现有的「重扫」按钮（`POST /api/tasks/{task_id}/rescan`）就能修复**，
     只是**用户不知道该点**。所以本项的重点是「**自动检测 + 主动提示**」，不是重写扫描逻辑。
6. 键空间混用样本本身也是**信号**：它们是 slime 侧「连 attempt id 都没分配上」的样本，
   属于**最彻底的失败**，建议在失败分析里单列一类（见 §3.0.4 的丢弃原因分布）。




### 1.5.4 【新增】`rollout/funnel/*` —— v1 与 plan 都漏掉的最关键一组 tag

**【实测】** 458 个 tag 中有 **13 条 `rollout/funnel/*`**，它们是 slime 侧
`compute_rollout_funnel_metrics`（`slime/rollout/reward_utils.py:917`）产出的**训练样本供给漏斗**：

```
rollout/funnel/launched_groups            提交过生成任务的组数
rollout/funnel/observed_groups            截止前返回、进入 all_data 的组数
rollout/funnel/filter_kept_groups         通过 dynamic filter 的组数
rollout/funnel/filter_dropped_groups      被 dynamic filter 丢弃的组数
rollout/funnel/selected_groups            真正进入训练批次的组数
rollout/funnel/filter_kept_not_selected_groups
rollout/funnel/launched_attempts / observed_attempts / filter_kept_attempts
rollout/funnel/filter_dropped_attempts / selected_attempts
rollout/funnel/loss_eligible_attempts     至少有一段仍可算 loss 的 attempt 数
```

**【实测】关键对账关系**：

```text
removed_ratio ≈ 1 − loss_eligible_attempts / launched_attempts
```

| 任务 | 命中步数 | 不匹配步数 | 不匹配幅度 |
| --- | --- | --- | --- |
| task8 | 60 / 64 | 4 | ≤0.0053（浮点/分批口径差） |
| task9 | 44 / 48 | 4 | ≤0.0138 |
| task4 | 60 / 65 | 5 | ≤0.0085 |
| task11 | 50 / 50 | 0 | — |

**【实测】** 逐 step 验证 task8（`launched_attempts` 恒为 128）：

| step | `removed_ratio` | ×128 | `sum(removed/count_*)` | `loss_eligible_attempts` |
| --- | --- | --- | --- | --- |
| 0 | 0.3906 | 50 | 50 | 78 |
| 20 | 0.4141 | 53 | 53 | 75 |
| 30 | 0.5547 | 71 | 71 | 57 |
| 50 | 0.8125 | 104 | 104 | 24 |
| 55 | 0.8594 | 110 | 110 | 18 |
| 62 | 0.9453 | 121 | 121 | **7** |
| 63 | 0.6279 | 80 | 81 | 47 |

→ **`rollout/removed_ratio` 就是「这一步有多少比例的 attempt 被丢弃」**，而 `loss_eligible_attempts` 是「还剩多少能算 loss」。
两者在**同一个 rollout step 上**，工具侧不需要做跨源 step 对齐就能对账。

**【实测】** `filter_dropped_groups` 在四个任务里**恒为 0**（launched = observed = filter_kept = selected = 16）。
这说明本批任务的 dynamic filter **没有整组丢过组**——它是被 **post-rollout 的 attempt 级 `remove_sample`** 与
**`too_few_survivors_*` 的组级拒绝**分别作用的结果（见 §1.5.5）。

### 1.5.5 【新增】关键裁定：v1 的「整组丢弃」叙事需要按两级口径重述

**【实测】** task8 全量 8213 sample（64 rollout × 16 组）逐样本解析 `sample.remove_sample` 与 `sample.metadata.train_drop_reason`：

| 统计项 | 数值 |
| --- | --- |
| attempt 总数 | 8213 |
| `remove_sample=True` | **4851（59.1%）** |
| 组总数 | 1024 |
| **整组被丢（组内全部 attempt 被标记）** | **548（53.5%）** |
| 部分被丢（组内部分 attempt 被标记） | 221（21.6%） |
| 完全保留 | 255（24.9%） |

**【实测】** task8 逐 rollout 的组级污染演化（基于 `.pt` 的真实 `remove_sample` 标记）：

| rollout | 组数 | 整组丢弃 | 部分丢弃 | 完全保留 | 可训练 attempt | 平均存活 attempt/组 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 16 | 6 | 2 | 8 | 78 | 4.9 |
| 14 | 16 | 9 | 3 | 4 | 50 | 3.1 |
| 30 | 16 | 8 | 3 | 5 | 57 | 3.6 |
| 43 | 16 | 12 | 2 | 2 | 27 | 1.7 |
| 52 | 16 | 10 | 5 | 1 | 35 | 2.2 |
| 55 | 16 | **13** | 2 | **1** | **18** | 1.1 |
| 59 | 16 | 12 | 4 | **0** | 26 | 1.6 |
| 62 | 16 | **15** | 1 | **0** | **7** | 0.4 |
| 63 | 16 | 9 | 4 | 3 | 48 | 3.0 |

**【实测】** task8 rollout 55 的逐组 `train_drop_reason` 明细（取前 6 组，展示两级作用）：

| 组 | 组内 8 个 attempt 的 drop reason |
| --- | --- |
| 880 | `zero_std_1.0` ×5、`agent_timeout` ×3 → **整组丢** |
| 881 | `agent_timeout` ×4、`too_few_survivors_1` ×1、`aborted:openhands_infra_exit_2` ×3 → **整组丢** |
| 882 | 全部 `None`（保留） |
| 883 | `agent_timeout` ×6、`too_few_survivors_2` ×2 → **整组丢** |
| 884 | `agent_timeout` ×3、`aborted:openhands_infra_exit_2` ×1，其余 4 个保留 → **部分丢** |
| 885 | `agent_timeout` ×5、`too_few_survivors_3` ×3 → **整组丢** |

**因果链（实测确认）**：

```
模型学会每轮输出更多 token → 单轮变慢 → 60 分钟预算内跑不完
  → attempt 级：train_drop_reason = "agent_timeout"（SWE_EXCLUDE_TIME_BUDGET）
  → 该组存活 attempt 数 < SWE_DAPO_MIN_SURVIVING_ATTEMPTS(默认 4)
  → 组级：train_drop_reason = "too_few_survivors_N" → 整组丢弃
  → 该 rollout 的 loss_eligible_attempts 骤降（rollout 62 只剩 7 / 128）
  → removed_ratio 冲到 0.945 → 训练样本枯竭 → 奖励不涨
```

> **【实测】** 组 884 是关键反例：它含 1 个 `agent_timeout`，但因为还有 4 个存活 attempt（≥ 4），
> **整组保留**。这证明「含 abort 的组必然被丢」是**错的**——真正的判据是**存活 attempt 数 < 4**。
> 工具侧现有的 `infra_only` / `all_fail` 分类完全看不到这个判据。

**【实测】** task4 与 task11 的失败模式**不同**，v1 与 plan 都没区分：

| 维度 | task4 | task8 / task11 |
| --- | --- | --- |
| `time_budget_exceeded` 的 `status` | **521 例是 `completed`**，3 例 `truncated` | **1425 例全部是 `aborted`** |
| 落盘时的 `train_drop_reason` | `agent:time_budget_exceeded`（450 例，走 diagnosis 路径） | `agent_timeout`（走 attempt 级 remove 路径） |
| reward 分布 | **68 例 reward=1.0**（已完成且通过）、456 例 reward=0.0 | **1425 例全部 reward=0.0** |
| 主因 | 题目池过大（688 题，每题只出现 1.47 次） | 时间预算耗尽 |

→ **task4 的主因是题目池/采样密度，task8/task11 的主因是时间预算**。工具必须能区分这两者，不能笼统说「abort 升高」。

### 1.5.6 【新增】已排除的病因（避免以后重复怀疑）

**【实测】** 以下三条在本批任务上**已被数据排除**，工具应显式记录，避免每次排查重新怀疑：

| 候选病因 | 排除依据 | 结论 |
| --- | --- | --- |
| **off-policy / train-rollout 分布不一致** | `train/train_rollout_logprob_abs_diff`：task4 全程 0.019–**0.045**、task8 0.017–**0.044**、task9 0.019–0.034。slime CI 断言阈值是 `<= 0.1` | ✅ **已排除**（三个任务均远低于阈值） |
| **熵坍缩（entropy collapse）** | `train/entropy_loss` 在 task4 **0.207 → 0.640（上升）**、task8 **0.152 → 0.641（上升）**、task9 0.147 → 0.577（上升） | ✅ **已排除**（熵在**上升**，不是坍缩；唯一反向的是 task11 0.151 → 0.109） |
| **KL 爆炸** | `train/ppo_kl`：task4 量级 ~1e-6（峰值 2.97e-5）、task8 ~1e-6（峰值 2.70e-5）、task9 ~1e-5（峰值 2.53e-5） | ✅ **已排除**（远小于任何爆炸阈值） |

**【实测】** 需要保留观察的一项：**task11 的 `train/train_rollout_logprob_abs_diff` 末值 0.0827、峰值 0.1054，超过了 slime CI 的 0.1 阈值**（task4/task8/task9 均未超）。task11 同时 `grad_norm` 末值 **0.986**（前三个任务都是下降）。→ **task11 存在独立的 off-policy / 梯度放大信号**，不能与 task8 混为一谈。这属于「排除性结论的反例」，工具应能把它标出来。

### 1.5.7 【新增 · v2 二次修订】组语义口径的标定 —— 「未把 ungradable 计入 dropped」会系统性高估梯度密度

**【实测】** 全文此前引用的「梯度密度 / mixed 组占比」都是**旧口径**（不含 `ungradable`）。
用 TB 权威口径（`rollout/generated/shaped_group_reward/*`）反向标定后，
**`ungradable` 必须计入 `dropped`**（详见 §2.1(e)）：

| 规则变体 | task8 逐 rollout 全对 | task9 逐 rollout 全对 | task4 逐 rollout 全对 |
| --- | --- | --- | --- |
| `ungradable` 不算 dropped | 43/64 | 13/48 | 18/65 |
| **`ungradable` 也算 dropped** | **54/64** | **42/48** | 24/65 |

**【实测】** `ungradable` 在两个任务里的分布：

| task | 组总数 | 含 ungradable 的组 | 其中**仅因 ungradable** 被判 dropped | 同时含 abort |
| --- | --- | --- | --- | --- |
| task8 | 1024 | 45 | **27** | 18 |
| **task9** | 768 | **64** | **59** | 5 |

→ task9 有 **59 个组（占其含 ungradable 组的 92%）**是「仅因 ungradable 被判 dropped」。
   这正是 task9 吻合度从 13/48 跳到 42/48 的来源。

**【实测】** 对关键指标的影响（三口径全量对照）：

| task | 口径 | mixed 组占比 | dropped 组占比 |
| --- | --- | --- | --- |
| **task8**（1024 组） | 旧口径（只认 abort） | 37.9% | 43.4% |
| | v2 二次修订（+ungradable +存活数<2） | 34.8% | 47.0% |
| | **★最终定稿（+ungradable，无存活数门槛）** | **35.6%** | **46.0%** |
| **task9**（768 组） | 旧口径（只认 abort） | 75.9% | 7.9% |
| | v2 二次修订（+ungradable +存活数<2） | 67.8% | 16.5% |
| | **★最终定稿（+ungradable，无存活数门槛）** | **68.4%** | **15.6%** |

→ **旧口径同时高估梯度密度、低估浪费算力**（task9：75.9% vs 68.4%；7.9% vs 15.6%）——两个方向都错。
→ **最终定稿与「二次修订」差异很小**（task8 34.8%→35.6%、task9 67.8%→68.4%），
   说明**「存活数 < 2」门槛的保留与否对总量影响有限，但会破坏逐 rollout 与 slime 的吻合**
   （task8 54/64 → **64/64**、task9 42/48 → **48/48**）。
   → **取最终定稿**：吻合度优先（它是唯一能逐 rollout 对上 slime 的口径）。

**【设计】** 因此：
1. 全文所有梯度密度数值必须**标明口径**；**跨任务横比必须统一口径**。
2. `utils/slime_grpo_semantics.classify_grpo_group` 增加 `drop_ungradable: bool = True` 参数
   （默认 True，即最终定稿口径），并保留开关以便对照。
3. **移除「存活数 < N」门槛**（标定证明它会破坏与 slime 的逐 rollout 吻合）。
4. **优先读 TB 权威口径**（§3.1.1），自推导仅作 fallback —— 本节标定的是 fallback 的最佳参数。


---

## 2. 关键缺陷诊断

### 2.1 【P0 · 语义错误】「有效梯度组」判据与 slime 训练口径不一致

现有实现（`service/reward_diagnosis_service.py`）：

```python
def classify_group_kind(categories):
    # 去掉 abort/timeout 后，既有过又有挂 → mixed（可学习）
    if has_one and has_zero: return "mixed"
    if has_one: return "all_pass"
    if has_zero: return "all_fail"
    return "infra_only"
```

问题有三层。**v2 已按实测数据修正第 (d) 层并新增 (e) 层**：

**（a）`all_fail` 组被算作「不可学习」，但它对 slime 同样是零梯度。**
slime 的 `check_surviving_reward_nonzero_std` 直接丢全等组（`zero_std_{v}`）；即便不过滤，
`grpo_std_normalization` 下全 0 组的 advantage 也恒为 0。
但工具把 `all_fail` 和 `infra_only` 区分开，把 `all_fail` 归为「能力不足（值得学）」，把 `infra_only` 归为「基建问题（不可学）」。

**（b）`all_fail` 与 `all_pass` 被混为一谈，但两者的训练后果完全不同。**
- `all_fail`：模型**没学会**这道题 → 该题太难 / 数据不足 → 应考虑**换掉或降采样**
- `all_pass`：模型**已经掌握** → 该题无信息量 → 应考虑**从训练集移除**

两者的正确动作**完全相反**，工具把它们放在同一个「非 mixed 即无梯度」的桶里，丢掉了最有价值的区分信息。

**（c）`n_valid_groups` 的口径与 slime 的 `zero_std` 统计重复但不一致。**

- 工具 `DBAttemptService.summarize_reward_metrics_by_rollout` 的 `n_valid_groups` = 去掉 abort/timeout 后「有过有挂」的组数
- slime 的 `rollout/zero_std/count_0.0` / `count_1.0` = 按 `group_index` 分组后**奖励全等**的组数（**不排除 abort**）

**【实测】** 两者在 task8 末步：工具侧 mixed 组 7/16，slime 侧 `count_0.0`=2 + `count_1.0`=2（零梯度 4 组）。
工具与 slime 的零梯度组数在量级上可比，但**口径不同**，必须同页对照。

**（d）【v2 修正】`infra_only` 的语义标注错误 —— 它不是「基建问题」，而是「被丢弃 · 浪费算力」。**

v1 把 `infra_only`（组内全为 abort/timeout）标为「基建问题（不可学）」，杠杆给「修基建」。
**【实测】** 这个假设被推翻：

- task8 组内全不可测的组只占 **5.3%**，但**含 abort 的组占 43.4%**——真正吃掉训练样本的是「部分含 abort」的大多数组，
  它们不是 `infra_only`，工具当前**完全看不见**。
- task8 的 abort 里 **94.7% 是 `agent_time_budget_exceeded`**（agent 侧行为退化），不是基建。
- 「修基建」这个动作对 task8 **基本无效**（沙箱类仅 14 例）。

**正确语义**：

| 组类型 | slime 实际（v2 精确版） | 对训练的贡献 | 工具 v1 归类 | **v2 归类** |
| --- | --- | --- | --- | --- |
| `mixed` | 保留，advantage 非零 | **唯一有正贡献** | mixed | ✅ 一致 |
| `all_pass` | 保留，但 advantage=0 | 零梯度 · **题已掌握** | all_pass | **零梯度 · 题已掌握**（杠杆：移出训练集） |
| `all_fail` | 保留，但 advantage=0 | 零梯度 · **题未掌握** | all_fail（隐含"可学"） | **零梯度 · 题未掌握**（杠杆：换题 / 降采样） |
| `infra_only` | 保留（不必然丢） | 零梯度 · **题未掌握** | infra_only（"基建问题"） | **零梯度 · 题未掌握**（同 all_fail） |
| **`dropped`（新增）** | **整组丢弃** | **被丢弃 · 浪费算力** | ❌ 无此概念 | **被丢弃 · 浪费算力**（杠杆：查丢弃原因） |

**（e）【v2 新增】新增 `dropped` 组类型 —— 判据由 TB 权威口径标定得出。**

**【实测 · v2 三次修订】** TB 的 `rollout/generated/shaped_group_reward/*` 是 slime 自己的组分类真源。
用「工具自推导 vs TB 权威」的**逐 rollout 全对率**做**规则组合穷举**，
在三个任务上锁定唯一最优解：

**穷举维度**：(a) `too_few_survivors_*` 是否算 dropped；
(b) `ungradable` 是否算 dropped；(c) 是否要求「存活数 < 2」；(d) `truncated` 是否计入 reward 统计。

| 规则组合 (a,b,c,d) | task8 全对 | task9 全对 | task4 全对 |
| --- | --- | --- | --- |
| (0,0,1,1) — 早期版本 | 43/64 | 13/48 | 18/65 |
| (0,1,1,1) — v2 二次修订 | 54/64 | 42/48 | 24/65 |
| (1,1,1,1) — 加 too_few_survivors | 47/64 | 42/48 | — |
| **(0,1,0,1) — ★ 最终定稿** | **64/64** | **48/48** | **62/65** |

→ **最终规则**：`dropped` ⟸ 组内含 **abort / `agent_timeout` / `aborted:*`** **或** 含 **`ungradable`**；
  **不因 `too_few_survivors_*` 判定**；**不设「存活数 < 2」门槛**；`truncated` **计入** reward 统计。

**【实测】** 三个任务的一致性：**task8 64/64、task9 48/48 完美吻合，task4 62/65**
（对比：早期规则在 task4 上只有 18/65）。

**【实测】** 关键的反直觉点：**`too_few_survivors_*` 不是组级丢弃的判据**。
它记录的是「组内部分 attempt 被移除后，存活数不足」这一**结果**，
但 slime 的 `unknown`（无法分类）只由「**含 abort / ungradable**」触发。
→ 工具若把 `too_few_survivors_*` 当丢弃判据，会把大量**仍产生梯度的组**误判为 `dropped`
  （实测 task8 从 64/64 掉到 47/64）。

**【实测】** 单组核验（task8 rollout 0）：工具推导 `mixed=13 / all_pass=1 / all_fail=2 / dropped=0`，
与 TB 的 `mixed_count=13 / all_one_count=1 / all_zero_count=2 / unknown_count=0` **逐项精确吻合**。

**【实测】** `ungradable` 这条规则的重要性：task8 有 **45 个组**含 `ungradable`（**27 个仅因此判 dropped**），
task9 有 **64 个组**（**59 个仅因此判 dropped**）。漏掉它会直接污染
`gradient_density` 与 `zero_gradient_ratio`（见 §1.5.7）。

**【设计】** v2 的组语义判定改为**两级**：

```
第一级（组是否被 slime 归为 unknown / 不可用）：
  dropped        ← 组内含 abort（status=aborted 或 aborted:* 或 agent_timeout）→ 整组丢弃
                 或 组内含 ungradable（评测器判不出分 → 整组不可用）★ v2 二次修订标定
                 （⚠️ 不含 too_few_survivors_*、不设存活数门槛 —— 见上方标定表）
第二级（可分类但零梯度的组）：
  mixed          ← 组内 reward 既有过又有挂 → advantage 非零
  all_pass       ← 组内 reward 全过 → 零梯度 · 题已掌握
  all_fail       ← 组内 reward 全挂 → 零梯度 · 题未掌握
  infra_only     ← 组内无可测 reward（理论边界，实际被 dropped 吸收）
```

> **【设计】重要提醒**：这套判定是**自推导**，虽经三任务标定已达 task8 64/64、task9 48/48，
> 仍可能与 slime 权威口径有个别 step 不一致（task4 62/65）。
> → **工具应优先读 TB 的 `rollout/generated/shaped_group_reward/*`**（§3.1.1），
> 本模块降级为「无 TB 场景的 fallback」；两者都可用时**输出差异**并在页面标注当前口径。
> → 上述规则是**fallback 的标定依据**（用 TB 真值标定出最接近的规则），不是替代 TB。

**【实测 · v2 三次修订的硬约束】DB 侧字段缺失 → 组语义无法仅凭 DB 复现。**

`ri_attempt_table` 的实际列：

```
primary_id, task_id, rollout_id, group_index, attempt_id, sandbox_id, instance_id,
status, reward, agent_turns, tool_calls, diff_line_count, response_length,
trainable_tokens, finalized
```

→ **没有** `train_drop_reason` / `abort_reason` / `agent_outcome` / `diagnosis` / `agent_exit_code`。
`ri_segment_table` 同样没有。

**【实测】** 仅用 DB 现有字段实现的规则（记为 R6）与 slime 权威口径的逐 rollout 全对率：

| 口径 | task8 | task9 |
| --- | --- | --- |
| **DB-only（现有字段）** | 43/64 | 13/48 |
| **`.pt`-based（含 `train_drop_reason`）** | **64/64** | **48/48** |

→ **结论**：要精确复现 slime 的组语义，**必须能读到 `train_drop_reason`**；
   只靠 `ri_attempt_table` 的 `status` + `reward` **做不到**
   （因为 `ungradable` 这个关键触发条件只存在于 `train_drop_reason`）。

**【设计】** 因此新增一项 **DB 字段补充**（与 §5「需要新增的 DB 字段」合并）：

| 字段 | 类型 | 来源 | 为什么必须 |
| --- | --- | --- | --- |
| `train_drop_reason` | `String` | `.pt` `sample.metadata.train_drop_reason` | 组语义判定的**关键输入**（`ungradable` / `agent_timeout` / `zero_std_*` / `too_few_survivors_*`） |
| `abort_reason` | `String` | `.pt` `sample.metadata.abort_reason` | 区分「基建 abort」与「agent 侧时间预算」 |
| `agent_outcome` | `String` | `.pt` `sample.metadata.agent_outcome` | 区分「时间预算耗尽」与「上下文打满」 |
| `diagnosis` | `String` | `.pt` `sample.metadata.diagnosis` | 区分「策略失败」与「评测器故障」 |

→ **不新建表**（沿用 `ri_attempt_table`，加列即可；SQLite 走 schema touch）。
→ 落库位置：`service/rollout_scan_service` 解析 `.pt` 时一并写入（与 `agent_turns` / `tool_calls` 同层）。
→ 兼容：老数据这些字段为空 → 组语义走 **DB-only 近似口径**，
   页面须标注「本任务无 `train_drop_reason`，组语义为近似值（与 slime 口径偏差较大）」。

**【实测】** 附：`train_drop_reason` 的**全部取值**（跨 task4/8/9 枚举，共 13 种）：

```
None                                未被移除
zero_std_1.0                        组内 reward 全为 1 → 零方差（题已掌握）
zero_std_0.0                        组内 reward 全为 0 → 零方差（题未掌握）
agent_timeout                       agent 时间预算耗尽（本批第一主因）
ungradable                          评测器判不出分 → 整组不可用
aborted:exception:SandboxApiException / SandboxExecStalled / SandboxInternalException / RuntimeError
aborted:openhands_infra_exit_2 / _137
aborted:evaluator_total_timeout
too_few_survivors_1 / _2 / _3       存活 attempt 数不足（结果标记，非 unknown 触发条件）
```

**【实测】** task4 的 `zero_std_1.0` 高达 **2599** 个 attempt（分布到约 5–11 个组/rollout），
实测样例确认这些组是 **8/8 全过**（`n1=8, n0=0`）→ 即 **task4 题目池里大量题已被完全掌握**。
这与 v1 说的「题目池饱和」一致，且是 `zero_std_1.0` 在 `rollout/removed/count_zero_std_1.0`（34）
之外的**第二个来源**（该 tag 数的是被移除的 attempt 数，不是组数）。



**【设计】** 新增三条一等公民指标（详见 §2.2）：

| 指标 | 定义 | 为什么重要 |
| --- | --- | --- |
| **`zero_gradient_ratio`** | (all_pass + all_fail) / n_groups | 传统零梯度率。**v2 保留但降级**：实测 task8 末步才 ~37.5%，解释力弱 |
| **`wasted_rollout_ratio`** | **dropped 组数** / n_groups | **真正的先行指标**。实测 task8 从 0% 涨到 **93.8%**（rollout 62），且与 `removed_ratio` 强相关 |
| **`gradient_density`** | mixed 组数 / n_groups | 题目池 / 采样参数是否匹配。实测 task9 **68.4%**（最终定稿口径，唯一成功）vs task8 **35.6%** |

**【实测】** task8 逐 rollout 的含 abort 组占比（工具 DB 口径，判定「组内任一 attempt 为 abort」）：

| rollout | 0 | 10 | 20 | 30 | 40 | 50 | 55 | 59 | 63 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 含 abort 组占比 | 0.0% | 12.5% | 12.5% | 43.8% | 50.0% | 87.5% | 93.8% | **100.0%** | 68.8% |

### 2.2 【P0 · 能力缺失】训练侧指标完全不可见

当前 `utils/tb_metric_catalog.py` 只识别 4 类 tag：

```python
"primary" / "truncated_ratio" / "attempt_count" / "unknown"
```

且明确声明「不给 loss / kl 等训练曲线编造含义」——态度正确，但**结果是训练侧完全空白**。

**【实测】** 实际产出的 458 个 tag 按前缀分布：

| 前缀 | 条数 | 语义 |
| --- | --- | --- |
| `rollout/generated/*` | 134 | 生成侧奖励与派生指标 |
| `rollout/filtered/*` | 77 | 过滤后（真正进训练）的对应指标 |
| `rollout/selected/*` | 67 | 被选中的 attempt 行为画像 |
| `rollout/population/*` | 27 | **训练样本供给**（v1 未提） |
| `perf/rollout_time*` | 18+17+17 | 采样耗时分解 |
| `rollout/funnel/*` | 13 | **样本供给漏斗**（v1 未提，见 §1.5.4） |
| `rollout/removed/*` | 13 + `removed_ratio` | **丢弃原因全集**（v1 前缀写错） |
| `rollout/outcome/*` | 12 | **agent 结束态分布**（v1 未提） |
| `rollout/replay/*` | 8 | 多轮重建质量 |
| `rollout/exposure/*` | 5 | **题目曝光/重复度**（v1 未提） |
| `rollout/response_len/*` | 4 | 响应长度分布 |
| `rollout/zero_std/*` | 2 | 零梯度组计数 |
| `train/*` | 16 | 训练侧指标 |

**v1 §2.2 表格的 tag 名与实际不符（实测逐条核对）**：

| v1 写的 tag | 实际存在？ | 实际情况 |
| --- | --- | --- |
| `train/kl_loss` | ❌ | 不存在。有 `train/ppo_kl` / `train/tis` / `train/tis_abs` / `train/tis_clipfrac` / `train/ois` |
| `train/ppo_kl` | ✅ | 存在，量级 1e-6 ~ 2.7e-5 |
| `train/pg_clipfrac` | ✅ | 存在 |
| `train/grad_norm` | ✅ | 存在 |
| `train/entropy_loss` | ✅ | 存在 |
| `train/train_rollout_logprob_abs_diff` | ✅ | 存在，**最关键** |
| `rollout/zero_std/count_1.0` | ✅ | 存在 |
| `rollout/zero_std/count_-1.0` | ❌ | 实际是 `count_0.0` |
| `rollout/dynamic_filter/drop_*` | ❌ | 实际前缀是 `rollout/removed/*` |
| `rollout/repetition_frac` | ✅ | 存在 |
| `rollout/truncated_ratio` | ✅ | 存在 |
| `rollout/segments_per_attempt` | ✅ | 存在 |
| `rollout/prefix_cache_hit_rate` | ✅ | 存在 |
| `perf/rollout_time/*` | ✅ | 存在 |
| `rollout/funnel/*` | ✅ | **v1 未列**，13 条，样本供给漏斗 |
| `rollout/population/*` | ✅ | **v1 未列**，27 条 |
| `rollout/selected/attempt/*` | ✅ | **v1 未列**，67 条 |
| `rollout/outcome/*` | ✅ | **v1 未列**，12 条 agent 结束态 |
| `rollout/exposure/*` | ✅ | **v1 未列**，5 条题目曝光 |

**【实测】`train/train_rollout_logprob_abs_diff` 是这个清单里最关键的一个**。slime 自己在 CI 里断言它 `<= 0.1`：

```python
# slime/backends/megatron_utils/model.py
if args.ci_test and "train/train_rollout_logprob_abs_diff" in log_dict:
    assert log_dict["train/train_rollout_logprob_abs_diff"] <= 0.1
```

它衡量「训练时算出的 logprob」与「rollout 引擎生成时的 logprob」的偏差。偏差大 → 训练分布与生成分布脱节 →
**模型在优化的东西和它实际生成的东西不是一回事**，这正是「奖励不涨但 loss 在降」的典型成因。

**【实测】** 本批任务中该指标 task4/task8/task9 全程 < 0.05，**已排除**（见 §1.5.6）；
但 **task11 峰值 0.1054 已越界**，需单独跟踪。

### 2.3 【P1 · 语义缺失】没有「这批数据值不值得训练」的一等公民指标

`problem_analysis_service` 已有 excellent 的雏形：

- 按 `instance_id` 追踪每次出现的通过率
- 首尾差值直方图
- 「太容易」（每次全过）/「太难」（每次全挂）筛选

但它**只回答了"题目难度"这一个维度**，且存在三个缺口：

1. **没有「可学习度」综合评分**。一道题「有时过有时不过」才是 GRPO 最有价值的样本；「总是过」「总是挂」都是零梯度。工具应输出每题的三态分布（稳定过 / 稳定挂 / 分化），而不只是首尾差。
2. **没有难度分层的数据集视图**。要提升奖励，关键是**持续供充分化样本（既过又挂的题）**。应能按「当前通过率区间」统计题目池构成，识别「题目池已饱和（都太简单）」或「题目池太难（都做不出）」。
3. **「太容易」的判定太严**。「每次出现都全通过（8/8）」才叫太容易，但 7/8 也是接近饱和。真正该关注的是**通过率贴近 0 或贴近 1 的题目占比**。

**【v2 补充】4. 【新增】没有「采样密度」维度 —— 这是本批任务最被忽视的杠杆（详见 §3.5）。**

### 2.4 【P1 · 归因盲区】工具的分类体系与 slime 自己的分类体系不一致

slime 已经有一套成熟的失败分类（`examples/coding_agent_rl/diagnostics.py`）：

```python
def classify_attempt(*, reward, agent_exit_code, diff_line_count,
                     agent_tool_calls, evaluation, abort_reason):
    if abort_reason:              return f"aborted:{abort_reason}"
    if reward > 0:                return "resolved"
    if evaluation:
        if evaluation.get("reason"):       return f"evaluator:{reason}"
        if evaluation.get("timed_out"):    return "evaluator:timeout"
    if agent_exit_code < 0:       return "agent:time_budget_exceeded"
    if not diff_line_count:       return "agent:no_diff"
    if not agent_tool_calls:      return "agent:no_recorded_tool_calls"
    return "evaluator:tests_failed"
```

**【实测】** task4 全量 8752 份 summary 的 `diagnosis` 分布：

| diagnosis | 数量 | 性质 |
| --- | --- | --- |
| `resolved` | 5340 | 通过 |
| `evaluator:tests_failed` | 2239 | 真的测试没过（策略问题） |
| **`agent:time_budget_exceeded`** | **450** | **agent 超预算** |
| `evaluator:invalid_evaluator_payload` | **203** | **评测器自身问题**（2.3%） |
| `agent:no_diff` | 121 | 没产出 diff |
| `evaluator:apply_failed` | 15 | 评测器 apply 失败 |

**【实测】** task4 的 `agent_outcome` 分布：

| agent_outcome | 数量 |
| --- | --- |
| `finished` | 6253 |
| **`context_exhausted`** | **1562** |
| **`time_budget_exceeded`** | **524** |
| `sdk_error` | 82 |
| `max_fake_responses` | 14 |
| `max_iterations_reached` | 11 |

**关键缺失（v2 强化）**：`agent:no_diff`（没产出 diff）和 `evaluator:*`（评测器自身问题）在 rollout-insight 里
**都退化成笼统的 fail**。这两类的正确动作截然不同：

- `no_diff` → 策略问题（agent 摆烂/没理解任务）→ 该题对该模型太难或 prompt 有歧义
- `evaluator:*` → **评测器问题** → 不该算作策略失败，不该污染训练信号

**【v2 关键修正】`time_budget_exceeded` 必须与 `context_exhausted` 分属不同 label。**

**【实测】** 现有 `utils/failure_reason_taxonomy.py` 的 `normalize_reason` 把两者都归入 `timeout` category
（`_TIMEOUT_TOKENS` 里含 `"time_budget"`），但它们是**完全不同的病因**：

| agent_outcome | task4 实测 | task8 实测 | 病因 | 处置 |
| --- | --- | --- | --- | --- |
| `context_exhausted` | **1562** | 583 | 上下文打满 → 该提上下文 / 提 max_turns | 提 `MAX_CONTEXT_LEN` |
| `time_budget_exceeded` | **524** | **1425** | 单轮变慢 → 60 分钟墙钟截断 | 提预算 / 限制单轮输出 / 排队隔离 |

**【实测】** task4 的 `time_budget_exceeded` 落在 `status=completed`（521 例）——意味着**agent 跑完了但没在预算内**，
属于「超时但有产出」；task8 的 1425 例落在 `status=aborted`——「被杀且无产出」。
两者对训练信号的影响完全不同（task4 的 68 例 reward=1.0 仍进了训练）。

### 2.5 【P1 · 断裂】rollout 数据与训练指标是两条平行线

现状：
- rollout 侧：`reward_diagnosis_service` 只读 `ri_attempt_table`（纯 DB）
- 训练侧：`insight_page_service` / `train_curve_service` 只读 TB 事件目录

两者**从未在同一坐标系里对齐**。

**【v2 更正】v1 提出「需要 `step_alignment_service` 做 rollout_id ↔ rollout/step ↔ train/step 的复杂对齐」——
实测证明这个需求大部分不成立。**

**【实测】** slime 的 `rollout/funnel/*` 与 `rollout/population/*` **本身就是按 rollout step 记录的**
（`log_dict["rollout/step"] = compute_rollout_step(args, rollout_id)`），
而 `compute_rollout_step` 对 agentic RL 的 rollout_id 是线性映射。
**【实测】** task8 的 `rollout/removed_ratio` 与 `rollout/funnel/loss_eligible_attempts` 在**同一 step 索引**上
精确对账（§1.5.4，60/64 步误差 ≤0.005）。

→ **【设计】** v2 **不新建** `service/step_alignment_service`。
改为在 `train_health_service` 里直接按 **rollout_id 对齐** TB 与 rollout 数据，
并提供一个可选的 `rollout_step_offset` 参数用于校正 off-by-one。
这比 v1 的方案少一个模块、少一套 join 逻辑。

**仍然需要的一步**：把「TB 曲线」与「rollout 数据质量」画在**同一 X 轴**上（v1 §3.3.3 的交叉视图方向正确，保留但实现路径简化）。

### 2.6 【P2 · 其它】

| # | 问题 | 位置 | v2 处置 |
| --- | --- | --- | --- |
| a | `train_drop_reason` 死指标，恒为 100% | `utils/rollout_pt_parser.py:626` | **❌ v1 判断错误**：字段在 `sample.metadata.train_drop_reason`，工具读错了键层级。**改为真读**（§3.2.4） |
| b | `token_estimate` 用「字符数 / 3.5」粗估 | `utils/rollout_pt_parser.py` | 保留到第三批 |
| c | `summarize_pass_visits` 把 **abort/timeout 计入通过率分母** | `problem_analysis_service.py` | **改为三口径并存**（§3.2.3） |
| d | 响应统计是「按 rollout 手动点按钮渐进补算」 | `advance_response_tool_stats` | 保留到第三批 |
| e | `DBTask.mean_reward` 存 `int(round(mean*1000))` 定点 | `rollout_scan_service.py:678` | 保留到第三批 |
| f | 无「数据集难度分层」视图，无「题目池健康度」指标 | 全缺 | **改为累计通过率三态 + 采样密度**（§3.5） |
| g | CLI 无 reward-diagnosis / train-health 子命令 | `cli/rollout_insight.py` | 保留到第三批 |
| h | `schedule/` 是空目录，无定时任务 | `schedule/__init__.py` | 保留到第三批 |
| i | **`removed/count_agent_timeout` 是 attempt 级，不是组级** | 工具未区分 | **新增**：与 `too_few_survivors_*` 分开计数（§3.0.4） |
| j | **`rollout/zero_std/count_0.0` 不是 `count_-1.0`** | v1 tag 名错 | 扩充 tag 目录时更正（§3.2.1） |

---

## 3. 增强方案

按「杠杆 / 成本」排序。每项标注 P0/P1/P2 与预估工作量。

### 3.0 【P0 · 本次最大新增】agent 时间预算退化 —— 本批任务的第一主因

> **【实测】本章所有数字均来自 task8 全量解析（64 rollout / 8213 sample / 8320 份 summary）与 task9 对照组。**

#### 3.0.1 现象

**【实测】** task8 逐 rollout 的 `rollout/outcome/time_budget_exceeded`（TB 原始）：
首值 0 → 峰值 **0.625**（rollout 55）→ 末值 0.227。

**【实测】** 3600s 硬预算 **100% 确认**：task8 全部 **1425** 个 `time_budget_exceeded` 样本的 `phase_time/agent_run`：

| p10 | p50 | p90 | min | max | ≥3600s 占比 |
| --- | --- | --- | --- | --- | --- |
| **3610.1s** | **3613.8s** | **3619.7s** | 3606.9s | 3903.2s | **100.0%** |

→ **精确的 60 分钟墙钟截断**，不是统计巧合。

#### 3.0.2 退化轨迹（**【实测】**，task8 逐 rollout，全量 sample）

| 指标 | rollout 0 | 30 | 55 | 59 | 63 | 变化 |
| --- | --- | --- | --- | --- | --- | --- |
| `turns` p50 | 62 | 58 | 37 | 35 | 40 | **62 → 35（下降）** |
| `peak_context_tokens` p50 | 88 517 | 86 522 | 63 870 | 64 110 | 70 427 | **88.5K → 70.4K（下降）** |
| `agent_run` p50 | 1313.1s | 2189.5s | 3614.3s | 3589.6s | 2469.0s | 撞上 3600s 墙 |
| `model_generation` p50 | 1201.9s | 1828.1s | 2563.5s | 2273.0s | 1968.8s | **生成耗时膨胀** |
| 单轮生成耗时 p50 | **18.1s** | **33.4s** | **86.0s** | **78.5s** | **54.8s** | **18.1 → 86.0s（+375%）** |
| 单轮生成 token p50 | **349** | **552** | **874** | **834** | **976** | **349 → 976（+180%）** |
| `attempt_total − agent_run`（预算外开销） | **278.0s** | **311.2s** | **476.2s** | **560.9s** | **939.3s** | **278 → 939s（+238%）** |
| `time_budget_exceeded` 占比（TB） | 0% | 12.5% | **62.5%** | 48.1% | 22.7% | 峰值 62.5% |

> **【实测】** TB 侧 `rollout/selected/attempt/completed_turns/p50` 独立印证同一趋势：
> **62.5 → 40.5**；`rollout/selected/attempt/peak_context_tokens/p50` **84 811.5 → 65 418**。
> 工具侧 `.pt` 与 TB 侧 `selected/attempt/*` 两套独立来源互相印证。

#### 3.0.3 机制判定（**【实测】排除三种假设**）

| 候选机制 | 判定 | 依据 |
| --- | --- | --- |
| **轮数爆炸**（turns 激增） | **❌ 排除** | turns p50 **62 → 35（下降）**。TB `completed_turns/p50` 62.5 → 40.5 同向印证 |
| **上下文打满**（context exhausted） | **❌ 排除** | `peak_context_tokens` p50 **88.5K → 70.4K（下降）**；task8 `agent_outcome=context_exhausted` 全量仅 583 例，且 rollout 0 时占 14.1% → 末值 0 |
| **基建故障**（沙箱 / driver） | **❌ 排除** | task8 abort 构成：沙箱类 `exception:SandboxApiException` 7 + `exception:SandboxExecStalled` 5 = **12 例（0.8%）**；`openhands_infra_exit_2` 219 例属 agent 侧 SDK 崩溃（非沙箱排队）；`.pt` 的 `failure_class` 中 `infrastructure` 每 rollout 仅 8 例 |
| **单轮变慢**（per-turn slowdown） | **✅ 确认** | 单轮生成耗时 **18.1 → 86.0s（+375%）**；单轮生成 token **349 → 976（+180%）**；`model_generation` 占 `agent_run` 比例 91.5% → 70.9%（变慢但轮数少了） |

**机制陈述（实测）**：

> 模型在训练中学会了「每轮输出更多 token」（单轮 349 → 976 token，+180%）。
> 单轮生成耗时随之从 18.1s 涨到 86.0s（+375%）。
> 60 分钟墙钟预算内跑不完 → agent 被硬杀 → `agent_outcome=time_budget_exceeded`、
> `status=aborted`、`abort_reason=agent_time_budget_exceeded`
> → attempt 级 `train_drop_reason="agent_timeout"`（`SWE_EXCLUDE_TIME_BUDGET`）
> → 该组存活 attempt 数跌破 4 → 组级 `too_few_survivors_N` → **整组丢弃**
> → `loss_eligible_attempts` 从 78 掉到 7 → `removed_ratio` 冲到 0.945
> → **训练样本枯竭 → 奖励不涨**。

#### 3.0.4 组污染放大效应（**【实测】**，基于 `.pt` 真实 `remove_sample` 标记）

| rollout | 组数 | 整组丢弃 | 部分丢弃 | 完全保留 | 可训练 attempt | 平均存活 attempt/组 | 丢弃判据 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 16 | 6 | 2 | 8 | 78 | 4.9 | `zero_std_*` + `truncated` |
| 14 | 16 | 9 | 3 | 4 | 50 | 3.1 | 同上 + `too_few_survivors_*` |
| 30 | 16 | 8 | 3 | 5 | 57 | 3.6 | `zero_std_1.0` 30 + `agent_timeout` 16 |
| 43 | 16 | 12 | 2 | 2 | 27 | 1.7 | `agent_timeout` 主导 |
| 52 | 16 | 10 | 5 | 1 | 35 | 2.2 | `agent_timeout` + `too_few_survivors_*` |
| 55 | 16 | **13** | 2 | **1** | **18** | 1.1 | `agent_timeout` 80 |
| 59 | 16 | 12 | 4 | **0** | 26 | 1.6 | `agent_timeout` 62 |
| 62 | 16 | **15** | 1 | **0** | **7** | **0.4** | `agent_timeout` 76 |

**【实测】**「整组丢弃率」在 rollout 62 达 **93.8%（15/16）**，可训练 attempt 只剩 **7 / 128**。
这就是「组污染放大效应」的准确表述：**一个 attempt 被杀 → 整组作废 → 8 倍算力浪费**。

**【实测】** 与 TB `rollout/removed/count_agent_timeout` 对账：task8 末值 76、峰值 80，
与 `.pt` 逐 rollout 的 `agent_timeout` 数（rollout 55 = 80、62 = 76）**精确一致**。

#### 3.0.5 附带发现：预算外的基建开销（**不占 agent 预算，不影响撞墙**）

> **【实测修正】** 本节初稿曾写「预算被基建开销挤占 → 资源紧张时排队越久越容易撞墙」。
> **该因果链经复核不成立，已更正如下。**

**【实测】** `attempt_total − agent_run`（agent 预算外的时间：沙箱启动、CLI 安装、评测、排队）确实在膨胀：

| rollout | 0 | 30 | 55 | 59 | 63 |
| --- | --- | --- | --- | --- | --- |
| `attempt_total − agent_run` p50 | 278.0s | 311.2s | 476.2s | 560.9s | **939.3s** |
| 其中 `cli_install` p50 | 79.8s | 93.2s | 131.4s | 170.7s | **353.6s** |
| 其中 `sandbox_boot_queue` p50 | **5.8s** | — | — | — | **5.8s** |
| 其中 `group_sibling_wait` p50 | 529.0s | 669.4s | 48.4s | 319.0s | 454.1s |
| 其中 `rollout_batch_wait` p50 | 1418.5s | 1600.5s | 431.0s | 599.8s | 1453.4s |

**复核结论（v2 二次修订 · 实测更正）**：

> **【实测更正 · 2026-11-23】** 本节初稿曾写「排队挤占 agent 预算」，随后被一版复核误判为
> 「排队不与预算竞争」，**两版都不完全正确**。经直接读 `.pt` 的
> `phase_time/sandbox_boot_queue`（**注意：不是 `phase_time/sandbox_boot`**，
> 两者相差一个数量级）逐 rollout 复核，**准确结论如下**。

#### 3.0.5.1 三个必须分清的字段

**【实测】** task8 逐 rollout 的三个易混字段（rollout 0 vs rollout 60）：

| 字段 | ro0 p50 | ro60 p50 | 说明 |
| --- | --- | --- | --- |
| `phase_time/sandbox_boot_queue` | **159.4s** | **188.7s** | **排队等待**（沙箱就绪前的等待） |
| `phase_time/sandbox_boot` | 10.4s | 5.2s | 沙箱实际启动（始终很小） |
| `phase_time/agent_run` | 1317.3s | 3360.0s | agent 运行（撞墙上限 3600s） |

> **【实测】** 初稿的一版复核把 `sandbox_boot_queue` 误读成了 `sandbox_boot`
> （读到 p50 仅 5.5s），因而错误地得出「排队极小、不占预算」的结论。
> TB 侧独立印证：`perf/rollout_time/attempt/sandbox_boot_queue` = **152.1s → 392.0s**，
> `perf/rollout_time_pct/attempt_total/sandbox_boot_queue` = 8.86% → 10.39%。

**【实测】** 逐 rollout 的 `sandbox_boot_queue` p50 轨迹（task8 全量）：

| rollout | 0 | 10 | 20 | 30 | 40 | 50 | 55 | 63 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sandbox_boot_queue` p50 | 159.4s | 165.0s | 162.8s | 188.4s | 300.2s | 229.8s | 241.0s | **311.8s** |
| `agent_run` p50 | 1317s | 1869s | 2054s | 2193s | 2117s | 3440s | **3614s** | 2469s |

#### 3.0.5.2 排队是否挤占 agent 预算：**有影响，但不是主因**

**【实测】** 决定性判据：撞墙样本的 `agent_run` **恒 ≥ 3600s**（task8 全部 1425 例，
p50=3614s，100% ≥3600）→ 说明 3600s 是**独立的墙钟上限**，而不是 `attempt_total` 的分解。

**【实测】** 按 `sandbox_boot_queue` 分桶看撞墙率（rollout 55，n=128）：

| queue 分桶 | 撞墙率 |
| --- | --- |
| `q < 100s` | 40.6%（13/32） |
| `100s ≤ q < 250s` | 52.8%（19/36） |
| `q ≥ 250s` | **80.0%**（48/60） |

→ **排队时间确实影响撞墙概率**（单调上升）。
**【实测】** 多 rollout 汇总（ro 20/30/40/50/55/60/63，n=897）后趋势仍在但变弱：
`q<100` → **20.7%**、`100≤q<250` → **33.9%**、`q≥250` → **30.0%**。

→ **准确结论（三层）**：
1. **`cli_install`（~100–130s）与 `sandbox_boot`（~5–10s）在 `agent_run` 之外**，
   不占用 3600s 墙钟 → 这两项纯粹是**吞吐/成本**问题，与奖励无关。
2. **`sandbox_boot_queue`（159s → 312s）是排队等待**，它与 agent 的墙钟预算**存在竞争关系**
   （分桶撞墙率 20.7% / 33.9% / 30.0%，rollout 55 上更明显：40.6% / 52.8% / 80.0%）
   → **排队是加剧因素，但不是第一主因**。
3. **第一主因仍是「单轮变慢」**：撞墙样本中 `agent_run` 的 100% ≥3600s 说明是 agent 自身跑不完；
   排队只解释了撞墙概率的**部分**差异，无法解释从 0% → 62.5% 的整体跃升。

→ **【设计】** 处置建议因此**分两条**，不可混为一谈：
- **奖励杠杆（主）**：提高 agent 时间预算（60 → 90 分钟）或**限制单轮输出长度**（单轮 token +180%）
- **吞吐杠杆（辅）**：降低沙箱排队（扩池 / 提并发），可**提高同等墙钟内的有效工作时间**，
  从而**间接**降低撞墙率；文案写「排队等待挤占有效推理时间」，**不写**「排队吃掉了 agent 预算」

#### 3.0.5.3 正确分层

| 层 | 证据字段 | 是否竞争 agent 墙钟预算 | 杠杆归属 |
| --- | --- | --- | --- |
| **模型变慢** | `model_generation` / `agent_run` / 单轮 token / `turns` | **是** —— 撞墙的**第一主因** | 奖励杠杆 |
| **排队等待** | `sandbox_boot_queue` | **部分竞争** —— 加剧撞墙概率（实测 20.7%→33.9%） | 吞吐杠杆（辅） |
| **准备/评测开销** | `cli_install` / `sandbox_boot` / `evaluator_total` | **否** —— 在 `agent_run` 之外 | 成本/吞吐（不进奖励杠杆） |

**【设计】** 模块输出应带显式 **`budget_clock`** 字段（默认 `"agent_run"`，依据实测），
并把三类开销**分列**为 `model_overhead` / `queue_wait_seconds` / `prep_overhead_seconds`，
避免用一个 `budget_stolen_ratio` 把「排队（影响奖励）」与「装 CLI（只影响成本）」混在一起。

文案纪律：
- 排队类指标写「**排队等待占 attempt 总时长的比例**」或「**排队挤占有效推理时间**」
- 准备/评测类指标写「**非 agent 时间占总 attempt 的比例**」
- **不写**「被偷走的预算」

**【实测】** task4 侧印证：task4 的 `phase_time` 平均值中 `cli_install` **271.9s**，
是除 `agent_run` 外的第二大开销，同样在 agent 预算之外 —— 属成本问题，不属奖励问题。




#### 3.0.6 对照组 task9（成功任务）

**【实测】** task9（32 题，梯度密度 68.4%（最终定稿口径）/ 77.6%（旧口径），唯一成功）：

| 指标 | rollout 0 | 中段 | rollout 47 |
| --- | --- | --- | --- |
| `turns` p50 | 59.0 | 60.5 | 60.0（**稳定**） |
| 单轮生成耗时 p50 | 23.5s | 33.5s | 35.7s（**+52%**，远低于 task8 的 +375%） |
| 单轮生成 token p50 | 384 | 602 | 602（+57%） |
| `time_budget_exceeded` 占比 | 0% | ~3% | ~13% |
| 全任务 `time_budget_exceeded` 占比 | — | — | **1.46%（88 / 6007）** |

→ **task9 也会退化，但退化幅度只有 task8 的 1/7，因此从不撞墙。**
这是一个「**同机制、不同幅度**」的强对照：证明退化是**连续的**（不是非黑即白的开关），
工具应该用**趋势幅度**而非布尔阈值来判定。

#### 3.0.7 详细设计：`utils/agent_budget_profile.py`

**【设计】** 本节只给**要做什么 / 判定规则 / 数据契约 / 判定优先级 / 阈值推导依据**；
完整函数签名由并行子任务产出，本节不重复。

##### (a) 要做什么

按 rollout 聚合 attempt 级预算画像，输出三部分：

1. **`series`**：逐 rollout 的预算健康度曲线（4 条主线 + 若干辅助量）
2. **`degradation`**：退化判定（判定优先级见 (d)）
3. **`budget_seconds`**：预算上限（从数据推断，见 (e)）

##### (b) 数据契约

**【设计】** 输入统一为 attempt 级行，每行至少含：

| 字段 | 来源 | 缺失处理 |
| --- | --- | --- |
| `rollout_id` | DB `ri_attempt_table.rollout_id` / `.pt` `sample.metadata` | 必填 |
| `group_index` | DB `ri_attempt_table.group_index` / `.pt` `sample["group_index"]`（**顶层**，不是 metadata） | 必填 |
| `agent_run_seconds` | `.pt` `sample.metadata["phase_time/agent_run"]` | 缺失则该 attempt 不参与耗时统计 |
| `attempt_total_seconds` | `.pt` `sample.metadata["phase_time/attempt_total"]` | 缺失时 `budget_stolen_seconds` 记 None |
| `completed_turns` | `.pt` `sample.metadata["attempt_completed_turns"]` | 缺失则不参与单轮耗时 |
| `generated_response_tokens` | `.pt` `sample.metadata["attempt_generated_response_tokens"]` | 同上 |
| `peak_context_tokens` | `.pt` `sample.metadata["attempt_peak_context_tokens"]` | 同上 |
| `agent_outcome` | `.pt` `sample.metadata["agent_outcome"]` / summary JSON | 缺失按 `finished` 处理 |
| `abort_reason` | `.pt` `sample.metadata["abort_reason"]` / summary JSON | 缺失表示无 abort |
| `sandbox_boot_queue_seconds` | `.pt` `sample.metadata["phase_time/sandbox_boot_queue"]` | 可选 |
| `cli_install_seconds` | `.pt` `sample.metadata["phase_time/cli_install"]` | 可选 |

**【设计】** 每条 `series` 行的字段：

| 字段 | 含义 | 口径 |
| --- | --- | --- |
| `rollout_id` | rollout 序号 | — |
| `n_attempts` | 该 rollout 的 attempt 总数 | — |
| `n_time_budget` | `agent_outcome == "time_budget_exceeded"` 的 attempt 数 | — |
| `time_budget_ratio` | `n_time_budget / n_attempts` | **【实测】** task8 峰值 0.625 |
| `agent_run_p50` / `agent_run_p95` | `agent_run_seconds` 的分位数 | — |
| `budget_hit_ratio` | `agent_run_seconds >= budget * 0.98` 的比例 | 撞墙率 |
| `turns_p50` | `completed_turns` 的 p50 | **【实测】** task8 62 → 35 |
| `per_turn_seconds_p50` | `agent_run_seconds / completed_turns` 的 p50 | **【实测】** task8 18.1 → 86.0s |
| `gen_tokens_per_turn_p50` | `generated_response_tokens / completed_turns` 的 p50 | **【实测】** task8 349 → 976 |
| `peak_context_p50` | `peak_context_tokens` 的 p50 | **【实测】** task8 88.5K → 70.4K |
| `sandbox_boot_queue_p50` | `sandbox_boot_queue_seconds` 的 p50 | **【实测】** task8 **159.4s → 311.8s**（ro0 → ro63）。**【设计】** 排队等待，**部分竞争** agent 墙钟 → 归**吞吐杠杆（辅）** |
| `queue_wait_ratio` | `sandbox_boot_queue_p50 / attempt_total_p50` | **【设计】** 排队占 attempt 总时长比例。**【实测】** 分桶撞墙率 20.7% / 33.9%（q<100 vs 100-250）→ 排队是**加剧因素，非第一主因** |
| `prep_overhead_p50` | `cli_install + sandbox_boot + evaluator_total` 的 p50 | **【设计】** 准备/评测开销，**在 `agent_run` 之外** → 归**成本/吞吐**，**不参与撞墙判定** |
| `infra_overhead_ratio` | `(attempt_total − agent_run) / attempt_total` | **【实测】** task8 rollout 63 = 939.3 / 3764.3 = **25.0%**。**【设计】** 成本视角，**不参与撞墙判定**（3600s 限制作用在 `agent_run` 上） |
| `infra_overhead_p50` | `attempt_total − agent_run` 的 p50 | **【实测】** task8 278s → 939s。**【设计】** 同上，仅成本视角 |
| `n_dropped_groups` | 整组被丢的组数（§2.1(e) 判据） | — |
| `dropped_group_ratio` | `n_dropped_groups / n_groups` | **【实测】** task8 rollout 62 = 93.8% |

> **【设计】重要口径约束（v2 二次修订）**：
> 本模块把**三类开销分列**，**不提供**笼统的 `budget_stolen_ratio`：
>
> | 类别 | 字段 | 与 agent 墙钟预算的关系 | 杠杆归属 |
> | --- | --- | --- | --- |
> | 模型变慢 | `per_turn_seconds_p50` / `gen_tokens_per_turn_p50` | **竞争**（第一主因） | **奖励杠杆** |
> | 排队等待 | `sandbox_boot_queue_p50` / `queue_wait_ratio` | **部分竞争**（加剧撞墙概率） | 吞吐杠杆（辅） |
> | 准备/评测 | `prep_overhead_p50` / `infra_overhead_ratio` | **不竞争** | 成本/吞吐（不进奖励杠杆） |
>
> 把「排队」与「装 CLI」混成一个 `budget_stolen_ratio` 会导致错误建议：
> 前者提高并发即可缓解，后者只能靠扩缓存/预装镜像。
> 详见 §3.0.5 的实测更正（含两次自我纠错的记录）。



##### (c) 数据来源与降级策略

**【设计】** 三级降级（页面必须明示当前用的是哪一级）：

| 级别 | 数据源 | 可产出 | 说明 |
| --- | --- | --- | --- |
| **L1 完整** | `.pt` dump（`utils.rollout_pt_parser.load_pt`）+ sidecar | 全部字段 | 最准，但需要读大文件 |
| **L2 摘要** | `attempt_artifacts/summaries/*.json`（`utils.attempt_artifacts.load_summary_index`） | `agent_run` / `turns` / `tokens` / `peak_context` / `agent_outcome` / `abort_reason` | **【实测】** summary JSON 含 `phase_times` 全量与 `attempt_peak_context_tokens` / `attempt_generated_response_tokens` / `attempt_completed_turns`，**足以支撑 L1 的大部分字段**，且有 sidecar 索引缓存 |
| **L3 仅 TB** | `rollout/selected/attempt/*` + `rollout/outcome/*` | `turns_p50` / `peak_context_p50` / `time_budget_ratio` / `context_exhausted` | 没有 `.pt` 也没有 summaries 时用；**无单轮耗时**，需在页面明示 |

**【设计】** 首选 **L2**（summary 侧车有缓存、字段够用、比重读 `.pt` 快一个数量级），
只有当 L2 缺失时才回落到 L1；L3 作为最终兜底。

##### (d) 退化判定规则与优先级

**【设计】** `verdict` 取值：`per_turn_slowdown` / `turn_count_blowup` / `context_exhaustion` / `healthy`。
判定**按优先级从上到下，第一个命中即返回**，理由是三种模式的**处置动作互斥**，必须给出唯一结论。

**优先级 1 — `context_exhaustion`（上下文打满）**：

| 条件 | 阈值 | 依据 |
| --- | --- | --- |
| `peak_context_p50` 逼近上下文上限且持续上升 | `peak_context_p50 >= 0.9 * max_context` **且** 首末增幅 > 10% | **【实测】** task8 `peak_context` 是**下降**的（88.5K → 70.4K），task4 `context_exhausted` 有 1562 例但 `peak_context` 也在降。→ 本批任务**不会**命中此判定 |
| **或** `agent_outcome == "context_exhausted"` 占比 > 30% | — | **【实测】** task8 rollout 0 峰值 0.1406 → 末值 0；task4 全量 1562/8752 = 17.9% |

**优先级 2 — `turn_count_blowup`（轮数爆炸）**：

| 条件 | 阈值 | 依据 |
| --- | --- | --- |
| `turns_p50` 首末增幅 | `>= +30%` **且** `per_turn_seconds_p50` 首末增幅 < +30% | **【实测】** task8 `turns` 是**下降**的（62 → 35），task9 稳定（59 → 60）→ 本批任务**不会**命中 |
| **或** `agent_outcome == "max_iterations_reached" / "turn_limit_reached"` 占比 > 10% | — | **【实测】** task8 全量 59 + 27 = 86 / 8320 = 1.0% |

**优先级 3 — `per_turn_slowdown`（单轮变慢 · 本批主因）**：

| 条件 | 阈值 | 依据 |
| --- | --- | --- |
| `per_turn_seconds_p50` 首末增幅 | `>= +50%` | **【实测】** task8 **+375%**、task9 **+52%**。取 50% 是为了让 task9（+52%）落在「轻度退化」而非「健康」，从而与 task8 拉开区分度 |
| **且** `gen_tokens_per_turn_p50` 首末增幅 | `>= +30%` | **【实测】** task8 **+180%**、task9 **+57%**。两者同步上升 = 「模型学会每轮说更多」，这是单轮变慢的**机制证据** |
| **且** `time_budget_ratio` 末值 | `>= 0.05` | **【实测】** task8 峰值 0.625；task9 全任务 1.46%。5% 是「已开始撞墙」的保守线 |
| **且** `turns_p50` 首末增幅 | `<= 0`（不上升） | **【实测】** task8 62 → 35（下降），确认「不是轮数问题」 |

> **【设计】** 四个条件是 **AND** 关系。这样 task9（+52% / +57% / 1.46% / 轮数稳定）会因为
> `time_budget_ratio >= 0.05` 不满足而落到 `healthy`；task8 全部满足 → `per_turn_slowdown`。
> **判定门槛由「单轮变慢 + 每轮 token 同步变多 + 已开始撞墙 + 轮数未增」四重证据共同支撑**，
> 避免单一指标误判。

**优先级 4 — `healthy`（无退化）**：

不命中以上任何一条。

##### (e) 阈值推导依据汇总

| 阈值 | 取值 | 推导依据（全部实测） |
| --- | --- | --- |
| `budget_seconds` | **3600**（自动推断） | task8 全部 1425 个撞墙样本 `agent_run` ∈ [3606.9, 3903.2]，**100% ≥ 3600**；p50 = 3613.8s。推断规则：取撞墙样本 `agent_run` 的 **p1 向下取整到百位**，且校验「≥ 该值的样本占比 > 95%」 |
| 单轮变慢阈值 | +50% | task8 **+375%**、task9 **+52%**，取 50% 让两者分居阈值两侧 |
| 单轮 token 阈值 | +30% | task8 **+180%**、task9 **+57%**，取 30% 留出区分空间 |
| 撞墙率阈值 | 5% | task9 全任务 **1.46%**、task8 峰值 **62.5%**，5% 是「已开始影响训练」的保守线 |
| 轮数爆炸阈值 | +30% | task9 轮数首末 59 → 60（+1.7%），30% 留足余量 |
| 上下文打满阈值 | 90% 上限 | task8 `peak_context` 88.5K 起步但持续下降；用「逼近上限 **且** 上升」双条件避免误判 |
| 「首/末」的取法 | 首 = 前 20% rollout 的中位数，末 = 后 20% | **【实测】** task8 rollout 0 单样本噪声大（task8 首次解析只有 1 个 sample），用分位数窗口比单点稳健 |

##### (f) 新的杠杆条目

**【设计】** 在 `service/reward_diagnosis_service._LEVER_RULES` 增加：

| key | 现象判据 | 建议动作 |
| --- | --- | --- |
| `time_budget_ratio` | `time_budget_ratio` 末值 > 0.05 **或** `verdict == "per_turn_slowdown"` | **提高 agent 时间预算**（60 → 90 分钟）**或限制单轮输出长度**（单轮 token 膨胀是根因）；检查单轮输出长度分布是否持续增长 |
| `per_turn_seconds_growth` | `per_turn_seconds_p50` 首末增幅 > 50% | 对输出长度加惩罚 / 在 prompt 约束单轮输出；检查是否出现长度崩塌式膨胀 |
| `dropped_group_ratio` | `dropped_group_ratio` 末值 > 0.3 | 查丢弃原因分布：`agent_timeout` → 查时间预算；`too_few_survivors_*` → 查存活阈值；`zero_std_*` → 查题目池 |
| `gradient_density` | `gradient_density` < 0.5 | 缩小题目池（把 `stable_pass` 题移出）；或提高 `groups_per_step`；目标是每步至少 50% 的组是 mixed |
| `n_gradient_problems` | 可学习题数量过少 | 补充中间难度题（从 `stable_fail` 里挑累计通过率 0.2–0.5 的） |
| `queue_wait_ratio`（**辅 · 吞吐杠杆**） | `sandbox_boot_queue_p50` 首末增幅 > 50% **或** `queue_wait_ratio` > 0.10 | **降低沙箱排队**（扩沙箱池 / 提并发）—— 排队挤占有效推理时间，**提高同等墙钟内的实际工作时间**，间接降低撞墙率。**【实测】** 分桶撞墙率 20.7%（q<100s）→ 33.9%（100–250s）→ 80.0%（≥250s，ro55） |

> **【设计】** `time_budget_ratio` / `per_turn_seconds_growth` / `dropped_group_ratio` / `gradient_density`
> 四个 key **必须同时出现在「奖励诊断页」与「跨任务对比页」**，复用 `build_reward_levers` 同一口径
> （与 v1 §5 的复用原则一致）。
>
> **【实测修正 · 二次】** v2 初稿曾包含 `queue_stolen_ratio` 杠杆，建议「把排队等待移出 agent 预算」；
> 一版复核又因**误读字段**（把 `sandbox_boot_queue` 当成 `sandbox_boot`）把它整个删掉。
> **两次都不对**。最终结论见 §3.0.5.2：
> - 排队**确实**加剧撞墙（分桶撞墙率 20.7% → 33.9% → 80.0%）→ **保留**，但改名为 `queue_wait_ratio`，
>   归为**吞吐杠杆（辅）**，文案写「排队挤占有效推理时间」，**不写**「吃掉 agent 预算」
> - `cli_install` / `sandbox_boot` 确实在 `agent_run` 之外 → 归**成本/吞吐**，不进奖励杠杆
> - **第一主因仍是单轮变慢** → 排在 `queue_wait_ratio` 之前


### 3.1 【P0】口径对齐层 —— 让现有结论先变正确

> **这是所有后续工作的地基。若口径不对，后面所有分析都在放大错误结论。**
> **【设计】** 但相比 v1，它的**优先级下调到第二批**（时间预算是 P0 第一位）。

#### 3.1.1 新增 `utils/slime_grpo_semantics.py`

集中定义组语义。**基于本次真实数据补充两条 v1 未提的判定**：

> v1 说 `all_fail` 与 `all_pass` 都"零梯度"。实测二者**处置方向相反**，且必须区分：
> - `all_pass`（题已掌握）→ 从训练集移除
> - `all_fail`（题未掌握）→ 降采样 / 补中间难度题
> - **含任一 abort 的组** → **整组被 slime 丢弃**，既不是"零梯度"也不是"可学"，而是**浪费的 rollout 预算**
>
> **【v2 更正】** 「含任一 abort 的组 → 整组被 slime 丢弃」这一判据**不成立**，需按**两级口径**重述：
>
> **【实测】** task8 rollout 55 逐组 `train_drop_reason` 明细（直接读 `.pt` 的 `sample.metadata.train_drop_reason`）：
>
> | 组 | 组内 8 个 attempt 的 drop reason | 存活数 | 结局 |
> | --- | --- | --- | --- |
> | 880 | `zero_std_1.0` ×5、`agent_timeout` ×3 | 0 | **整组丢** |
> | 881 | `agent_timeout` ×4、`too_few_survivors_1` ×1、`aborted:openhands_infra_exit_2` ×3 | 0 | **整组丢** |
> | 882 | 全部 `None` | 8 | 保留 |
> | 883 | `agent_timeout` ×6、`too_few_survivors_2` ×2 | 0 | **整组丢** |
> | **884** | `agent_timeout` ×3、`aborted:openhands_infra_exit_2` ×1 | **4** | **保留（部分丢）** |
> | 885 | `agent_timeout` ×5、`too_few_survivors_3` ×3 | 0 | **整组丢** |
>
> → **组 884 是决定性反例**：它含 4 个 `agent_timeout`（即 abort），但因为**还有 4 个存活 attempt（≥ 4）**，
> **整组保留**。
> → **正确判据是「存活 attempt 数 < `SWE_DAPO_MIN_SURVIVING_ATTEMPTS`（默认 4）」**，
> 而**不是**「含 abort 即丢」。这正是 v1 §2.1(c) 说的「口径与 slime 不一致」的具体形态。
>
> **【实测】逐 rollout 验证**（`.pt` 的真实 `remove_sample` 标记）：
>
> | rollout | 整组丢弃 | 部分丢弃 | 完全保留 | 可训练 attempt |
> | --- | --- | --- | --- | --- |
> | 0 | 6 | 2 | 8 | 78 |
> | 43 | 12 | 2 | 2 | 27 |
> | 55 | 13 | 2 | 1 | 18 |
> | 62 | **15** | 1 | **0** | **7** |
>
> **【设计】** 实现时 `min_surviving` 默认 4，但**必须可配置**（来自环境变量
> `SWE_DAPO_MIN_SURVIVING_ATTEMPTS`），读法复用 `utils/slime_job_type.load_env_from_run_dir`，
> 读不到时回落到 4 并在返回里标注 `min_surviving_source`。
>
> **【设计】口径注意**：`dropped` 组的判定依赖 `min_surviving`，
> 而 `unknown_group_count`（TB）是**另一回事** —— 它统计的是「**组奖励无法分类**」的组数
> （≈ 组内含 ≥1 个奖励不可得的 attempt），**不是**「被丢弃的组数」。
> 实测 task8 rollout 0：TB `unknown_group_count` = **0**，而 `.pt` 口径的整组丢弃 = **6**。
> 两者不可互相替代，页面上必须分开呈现。



**【设计】** 公开能力：

```
GROUP_KINDS = ("mixed", "all_pass", "all_fail", "infra_only", "dropped")

def classify_grpo_group(rewards, statuses, *, n_surviving=None,
                         min_surviving=4, drop_reasons=None) -> str
def group_gradient_contribution(kind) -> float   # mixed=1.0, 其余=0.0
def group_action_hint(kind) -> str
def group_drop_reason(kind, *, train_drop_reasons=None) -> str
```

##### 【v2 重大更正】slime 已在 TB 里上报自己的组语义 —— 工具应**优先读 TB**，自推导降级为 fallback

**【实测】** `rollout/generated/shaped_group_reward/*`（31 个 tag）是 **slime 自己的权威组分类**：

```
rollout/generated/shaped_group_reward/group_count
rollout/generated/shaped_group_reward/classified_group_count
rollout/generated/shaped_group_reward/unclassified_group_count
rollout/generated/shaped_group_reward/unknown_group_count
rollout/generated/shaped_group_reward/mixed_count  / mixed_fraction
rollout/generated/shaped_group_reward/all_zero_count  / all_zero_fraction
rollout/generated/shaped_group_reward/all_one_count   / all_one_fraction
rollout/generated/shaped_group_reward/inconsistent_group_count
rollout/generated/shaped_group_reward/nonbinary_group_count
rollout/generated/shaped_group_reward/pass_count_{0..8}_of_8_count / _fraction
rollout/generated/group_reward/{mixed,all_one,all_zero,unknown}_count / _fraction   （同构另一族）
```

**【实测】** task8 / task9 首→末对比：

| tag | task8（失败） | task9（成功） |
| --- | --- | --- |
| `group_count` | 16 → 16 | 16 → 16 |
| `mixed_count` | **13 → 3** | **15 → 12** |
| `all_zero_count` | 2 → 0 | 0 → 0 |
| `all_one_count` | 1 → 2 | 0 → 2 |
| `classified_group_count` | 16 → **5** | 15 → 14 |
| **`unknown_group_count`** | **0 → 11** | **1 → 2** |
| `mixed_fraction` | 0.8125 → 0.60 | 1.0 → 0.857 |

→ `mixed_count` 是「真正产生非零 advantage 的组数」，
   `unknown_group_count` 是「**组奖励无法分类**」的组数（≈ 组内含 ≥1 个奖励不可得的 attempt）。
→ **注意 `unknown_group_count` ≠ 「被丢弃的组数」**：实测 task8 rollout 0，
   TB `unknown_group_count` = **0**，而按 `train_drop_reason` 的 `.pt` 口径整组丢弃 = **6**。
   两者是不同概念，页面上必须分开呈现，不可互相替代。


**【实测】工具自推导与 slime 权威口径存在偏差**：以 task8 rollout 0 为例 ——
TB `mixed_count` = 13，工具侧「纯 pass+fail 混合组」= 12、「abort 计为 reward=0 模拟 slime 式判定」= 12。
根源是「不可评分样本（abort）是否参与组内判定与标准化」的处理方式不同，
且各组 abort 数量不同会让分类在 mixed / all_zero / all_one 之间摆动。

**【设计】结论与调整**：

1. **工具应优先读取 `rollout/generated/shaped_group_reward/*`**（slime 权威口径），
   `utils/slime_grpo_semantics.py` 的自推导**降级为「无 TB 场景的 fallback」**。
   这比 v1 §3.1.1「自己实现一套组语义」更准、更省，且天然与 slime 对账。
2. **UI 必须标注当前用的哪一套口径**（`group_source: "slime_tb" | "tool_fallback"`），
   两者不一致时显示差异值（避免用户误以为工具口径就是训练口径）。
3. `pass_count_{0..8}_of_8_count` 提供**组内通过数的完整分布**，
   粒度远高于工具的三分类，建议作为「组内奖励分布」视图的数据源。
4. **【陷阱】同名 zero_std 有两套口径，差 8 倍（组大小）**：
   - `rollout/removed/count_zero_std_0.0` / `_1.0` = **attempts**（被丢弃组内的 attempt 数）
   - `rollout/zero_std/count_0.0` / `_1.0` = **groups**（零梯度组数）
   - `rollout/generated/shaped_group_reward/all_zero_count` / `all_one_count` = **groups**（与 `zero_std/count_*` 可对账）

   **【实测】** task8 step 0：`removed/count_zero_std_0.0` = 16（= 2 组 × 8 attempt），
   而 `zero_std/count_0.0` = 2（组）、`all_zero_count` = 2（组）✓
   → **tag 目录必须为每个 tag 标注单位（group / attempt / ratio / token / seconds）**，
   否则把 `removed/count_zero_std_1.0` = 34 与 `zero_std/count_1.0` = 2 并排显示会直接误导。
   → v1 §2.1(c) 写的 `rollout/zero_std/count_-1.0` 前缀与取值**均不存在**，应为 `count_0.0` / `count_1.0`。

**【设计】** `classify_grpo_group` 的签名因此调整：保留纯函数形式（供无 TB 场景与单测使用），
但在 service 层增加**口径选择**：

```
# 【设计】service 层口径选择
def resolve_group_semantics(task_id) -> Dict:
    """
    优先读 TB rollout/generated/shaped_group_reward/*；
    读不到才回落到 classify_grpo_group 自推导。
    返回 {"source": "slime_tb"|"tool_fallback", "series": [...],
          "discrepancy": {...}}   # 两者都有时给出差异
    """
```

**关键设计**：判定必须用 **attempt 级奖励**（一个 attempt 一个 reward，不是 segment 级）。
slime 的 `collect_attempt_rewards` 已明确：一个 attempt 的多个 segment 共享同一个 terminal reward。
rollout-insight 的 `ri_attempt_table.reward` 恰好就是 attempt 级，✅ 对齐。

**【设计】** `min_surviving` 默认 4，但**必须可配置**，因为它来自环境变量 `SWE_DAPO_MIN_SURVIVING_ATTEMPTS`。
实现时应优先读 run 目录的 `resolved_config.yaml` / env（复用 `utils/slime_job_type.load_env_from_run_dir`），
读不到时回落到默认值 4，并在返回里标注 `min_surviving_source`。

#### 3.1.2 修正 `summarize_gradient_groups`

**【设计】** 在 v1 §3.1.2 基础上，**新增 `dropped` 第五类 + 三条一等公民指标**：

- `all_fail` 的文案从「不可学习」改为「**零梯度 · 题未掌握**」，杠杆从「学」改为「**换题 / 降采样**」
- `all_pass` 标注为「**零梯度 · 题已掌握**」，杠杆改为「**移出训练集**」
- `infra_only` 标注为「**零梯度 · 题未掌握**」（同 all_fail；v1 的「被丢弃 · 修基建」是错的）
- **`dropped` 标注为「**被丢弃 · 浪费算力**」**，杠杆改为「**查丢弃原因**」，并附上丢弃原因分布
- 新增 `zero_gradient_ratio` 曲线 = (all_pass + all_fail) / n_groups
- 新增 **`wasted_rollout_ratio`** 曲线 = **dropped 组数** / n_groups —— **实测的先行指标**
- 新增 **`gradient_density`** 曲线 = mixed 组数 / n_groups

**【设计】** `dropped` 组还要带一个 `drop_reasons` 直方图（从 `.pt` 的 `train_drop_reason` 或
`rollout/removed/count_*` 聚合），让页面能直接回答「是被时间预算杀的还是被零梯度杀的」。

#### 3.1.3 修正 `summarize_pass_visits` 的分母口径 → 三口径并存

现状注释明说「abort 与 timeout 计入分母、不计入分子」。

**问题**：slime 对存活 attempt 不足的组是**整组丢弃并重采**，不是「保留但算 0 分」。
所以一个「7/8 abort + 1/8 通过」的组，在 slime 眼里**根本不存在**（会被丢弃重采）；在工具眼里却是 12.5% 通过率。

**【设计】** 改为**三口径并存**：

| 指标 | 定义 | 用途 |
| --- | --- | --- |
| `pass_rate_strict` | 分母 = 全组 attempt 数（现状） | 看「用户视角的端到端成功率」 |
| `pass_rate_gradable` | 分母 = 去掉 abort/timeout 后的 attempt 数 | 看「slime 视角的有效样本质量」 |
| `pass_rate_slime_effective` | **仅未被丢弃的组**，组内平均 | **对齐 slime 真实训练分布** |

并在 UI 上明确标注三者差异，避免误读。

**【设计】** `pass_rate_slime_effective` 的「未被丢弃」判据复用 §3.1.1 的 `classify_grpo_group`，
即**存活 attempt 数 ≥ min_surviving 且无基建类 abort**。

#### 3.1.4 `train_drop_reason`：从「死指标」改成「一等指标」

**【v2 更正】** v1 §3.1.4 说这个字段不存在、指标恒为 100%。**实测证明字段存在**（§1.3 边界 2）。

**【设计】** 修正 `utils/rollout_pt_parser.summarize_train_sample_stats`：
从**顶层 `sample["train_drop_reason"]`** 改为读 **`sample["metadata"]["train_drop_reason"]`**，
并输出完整的 drop reason 直方图（`agent_timeout` / `too_few_survivors_*` / `zero_std_*` /
`aborted:*` / `ungradable` / `truncated` / `turn_cap`）。

**【设计】** 同时保留旧键作为兼容回退（历史 sidecar 可能写在顶层）。

### 3.2 【P0】训练侧指标接入 —— 打开「训练进程」黑盒

#### 3.2.1 扩充 `utils/tb_metric_catalog.py`

**【设计】** 按 **【实测】的 458 个 tag** 扩充，每条**必须写明来源**（`slime/ray/rollout.py` /
`slime/rollout/reward_utils.py` / `slime/backends/megatron_utils/model.py`），**不编造含义**。

新增分类维度（保留 v1 的 `primary` / `truncated_ratio` / `attempt_count` / `unknown`）：

**丢弃原因全集（v1 前缀写错，实测为 `rollout/removed/*`）**：

| tag | 语义 | 实测意义 |
| --- | --- | --- |
| `rollout/removed_ratio` | **被移除样本占比** | **第一主因指标**；task4 末值 **1.0** = 整步无样本；实测 ≈ `1 − loss_eligible/launched` |
| `rollout/removed/count_agent_timeout` | agent 超时被移除数（**attempt 级**） | task8 2→**76**、task11 2→**52** |
| `rollout/removed/count_zero_std_0.0` / `_1.0` | 全挂 / 全过被移除数（**组级**） | 零梯度组；task8 `_1.0` 11→**34** |
| `rollout/removed/count_truncated` | 截断被移除数 | task8 18→1（下降） |
| `rollout/removed/count_ungradable` | 不可评分被移除数 | **评测器侧问题**；task11 4→**10** |
| `rollout/removed/count_too_few_survivors_{1,2,3}` | **存活数不足被移除数（组级丢弃的直接证据）** | **v1 未列，是组污染的关键指标** |
| `rollout/removed/count_aborted_*` | 按 abort 原因细分 | 区分基建 vs agent 侧；实测 task8 沙箱类仅 12 例 |

**训练样本供给（v1 未提，**【实测】27 条**）**：

| tag | 语义 | 实测意义 |
| --- | --- | --- |
| `rollout/population/trainable_attempt_count` | **进入训练的 attempt 数** | task4 38→**0**（⚠️ 含扫描未完成分，见 §1.5.3.1）、task11 95→**15** |
| `rollout/population/trainable_sample_count` / `_token_count` | 进入训练的样本 / token 数 | 同上 |
| `rollout/population/shaped_reward_coverage` | 有可用奖励的占比 | task11 0.961→**0.430**、task4 0.984→**0** |
| `rollout/population/shaped_reward_unavailable_attempt_count` | 奖励不可得的 attempt 数 | 奖励缺失的绝对量 |
| `rollout/population/removed_attempt_count` / `_ratio` | 被移除 attempt 数 / 占比 | 与 `removed_ratio` 互证 |
| `rollout/population/removed_token_ratio` | 被移除 token 占比 | 算力浪费的 token 视角 |
| `rollout/population/token_count` | 本步总 token | 算力消耗 |

**样本供给漏斗（v1 与 plan 均未提，**【实测】13 条**）**：

| tag | 语义 |
| --- | --- |
| `rollout/funnel/launched_groups` / `_attempts` | 提交过生成任务的组 / attempt 数 |
| `rollout/funnel/observed_groups` | 截止前返回、进入 `all_data` 的组数 |
| `rollout/funnel/filter_kept_groups` | 通过 dynamic filter 的组数 |
| `rollout/funnel/filter_dropped_groups` | **被 dynamic filter 整组丢弃的组数**（实测本批恒为 0） |
| `rollout/funnel/selected_groups` | 真正进入训练批次的组数 |
| `rollout/funnel/loss_eligible_attempts` | **至少有一段仍可算 loss 的 attempt 数**（= `launched × (1 − removed_ratio)`） |
| `rollout/funnel/filter_kept_not_selected_groups` | 过了 filter 但没被选中的组数 |

**Agent 结束态（v1 未提，**【实测】12 条**）**：

| tag | 语义 |
| --- | --- |
| `rollout/outcome/finished` | 正常结束占比 |
| `rollout/outcome/time_budget_exceeded` | **时间预算耗尽占比**（task8 峰值 **0.625**） |
| `rollout/outcome/context_exhausted` | 上下文打满占比（task4 高，task8 末值为 0） |
| `rollout/outcome/sdk_error` / `max_fake_responses` / `max_iterations_reached` / `turn_limit_reached` / `stuck` | 其它 agent 异常 |

**行为画像（v1 未提，**【实测】67 条**）**：

| tag | 语义 |
| --- | --- |
| `rollout/selected/attempt/completed_turns/{p50,p95,mean,max}` | 轮数分布（**本次退化的核心可观测量**；实测 task8 62.5 → 40.5） |
| `rollout/selected/attempt/response_tokens/{p50,p95,mean,max}` | 响应 token 分布 |
| `rollout/selected/attempt/final_context_tokens/{p50,p95,...}` | 最终上下文 |
| `rollout/selected/attempt/peak_context_tokens/{p50,p95,...}` | 峰值上下文（实测 task8 84 811 → 65 418） |

**训练侧（v1 清单保留，但更正 tag 名）**：

| tag | 语义 | 实测 |
| --- | --- | --- |
| `train/ppo_kl` | PPO 近似 KL（**v1 写的 `train/kl_loss` 不存在**） | ~1e-6，**已排除 KL 爆炸** |
| `train/pg_clipfrac` | PPO clip 触发比例 | 0 → 0.043，未见异常 |
| `train/grad_norm` | 梯度范数 | task4/8/9 下降；**task11 末值 0.986（反向）** |
| `train/entropy_loss` | 熵 | task4/8/9 **上升**（0.15 → 0.64），**已排除熵坍缩** |
| `train/train_rollout_logprob_abs_diff` | 训练 logprob 与 rollout logprob 绝对差 | task4/8/9 < 0.05，**已排除 off-policy**；**task11 峰值 0.1054 越界** |
| `train/tis` / `train/tis_abs` / `train/tis_clipfrac` / `train/ois` / `train/loss` / `train/lr-pg_{0,1,2}` | 其余训练指标 | — |

**【设计】** `classify_tb_tag` 的 `unknown` 兜底改为「**按前缀启发式归类 + 标注『推断』**」，
新增前缀 → 类别映射：`rollout/removed/` → `removed`、`rollout/population/` → `population`、
`rollout/funnel/` → `funnel`、`rollout/outcome/` → `outcome`、`rollout/selected/` → `behavior`、
`rollout/zero_std/` → `zero_std`、`train/` → `training`。

> **注意保持项目已有原则**：`utils/tb_metric_catalog.py` 的 docstring 明确写「不给 loss / kl 等训练曲线编造含义」。
> 本次扩充不是编造——每条都对应 slime 源码里的真实计算逻辑，且要在 `describe_tb_tag` 里
> **同时给出源码出处**，让使用者可核验。

#### 3.2.2 扩充 `utils/tb_series_reader.py`

**【实测】** v1 §3.2.2 误以为「现有 `read_latest_scalars` 只取最后一个点，无法画曲线」，
并建议「新增 `utils/tb_series_reader.py`」——**该模块已存在**（含 `list_scalar_tags` /
`read_scalar_series` / `series_points`），v1 是没查 func.md。

**【设计】** 实际需要补的只有 4 点：

1. **前缀批量取**：`read_scalar_series` 支持按 tag 前缀批量取（如 `rollout/removed/*` 一把取全）
2. **sidecar 缓存**：按 `event_dir + mtime` 失效，避免每次读 458 个 tag 全量解析
   （**【实测】** 单次全量读 458 tag 约 1.3–2.0s，可接受但不理想；`train_curve_service` 已按需只读选中的 tag）
3. **`find_anomalies(series, rules)`**：基于**趋势 + 阈值**（非单点）的异常检测
4. **`resolve_event_dir`** 复用：`service/train_curve_service.resolve_event_dir` 已实现，**不要重写**

**【设计】** 新增 `read_scalar_series_by_prefix(event_dir, prefixes)`，返回 `{prefix: {tag: points}}`。

#### 3.2.3 新增 `service/train_health_service.py`

**【设计】**

```
{
  "removed": {
    "ratio": [...],                       # rollout/removed_ratio
    "by_reason": {                        # rollout/removed/count_* → 序列
      "agent_timeout": [...],
      "zero_std_1.0": [...],
      "too_few_survivors_1": [...],
      ...
    },
    "trainable_attempt_count": [...],     # rollout/population/trainable_attempt_count
    "shaped_reward_coverage": [...],      # rollout/population/shaped_reward_coverage
  },
  "funnel": {
    "launched_attempts": [...],
    "loss_eligible_attempts": [...],
    "removed_ratio": [...],
    "reconcile_error": 0.0042,           # |removed_ratio − (1 − loss_eligible/launched)| 的中位数
  },
  "training": {
    "entropy_loss": [...], "grad_norm": [...], "ppo_kl": [...],
    "logprob_abs_diff": [...], "pg_clipfrac": [...],
  },
  "behavior": {
    "completed_turns_p50": [...], "response_tokens_p50": [...],
    "peak_context_tokens_p95": [...],
  },
  "outcome": {                            # rollout/outcome/*
    "time_budget_exceeded": [...], "context_exhausted": [...], "finished": [...],
  },
  "verdicts": [{severity, code, title, evidence[], actions[]}],
  "exclusions": [{code, title, evidence}],
  "alignment": {"source": "rollout_id", "n_aligned": 65, "max_offset": 0},
}
```

**`verdicts` 判定规则（全部基于趋势 + 阈值，不是单点）**：

| 条件 | severity | 结论 | 建议动作 |
| --- | --- | --- | --- |
| `removed_ratio` 末值 > 0.5 且 `count_agent_timeout` 末值 > 20 | bad | **agent 时间预算耗尽** | 提预算 / 限制单轮输出长度 |
| `count_too_few_survivors_*` 末值 > 5 | bad | **组污染：存活数不足被整组丢** | 查 `agent_timeout` 是否是主因 |
| `count_zero_std_*` 末值 > 30 | warn | 零梯度组过多 | 移出饱和题 / 补中间难度题 |
| `loss_eligible_attempts` 末值 < 30 | bad | **训练样本枯竭** | 优先修上游丢弃原因 |
| `shaped_reward_coverage` 末值 < 0.5 | bad | 奖励不可得比例高 | 查评测器 / `ungradable` |
| `entropy_loss` 首末降幅 > 50% | warn | 策略坍缩 | 加熵正则 / 提 temperature |
| `entropy_loss` 首末**升幅** > 100% | info | 熵在上升（非坍缩） | 记录为排除项 |
| `ppo_kl` 持续 > 0.05 | bad | 策略偏离参考 | 降 lr / 加 KL 系数 |
| `logprob_abs_diff` 峰值 > 0.1 | bad | train/rollout 脱节 | 检查权重同步 / async 配置 |
| `grad_norm` 首末升幅 > 300% | warn | 梯度放大 | 查 lr / 奖励尺度 |
| `pg_clipfrac` > 0.3 | warn | 梯度被裁 | 降 lr / 调 clip |
| `completed_turns_p50` 首末降幅 > 30% | warn | 轮数塌缩 | 与单轮耗时一起判（见 §3.0.7） |

**`exclusions` 字段是刻意设计**：把「已排除的可能病因」显式列出，避免每次排查都重新怀疑 KL 爆炸、熵坍缩这类常见但本例并非主因的问题。

**【设计】** `exclusions` 的判定（全部来自 §1.5.6 的实测结论）：

| code | 排除条件 | 中文文案 |
| --- | --- | --- |
| `off_policy` | `logprob_abs_diff` 全程 ≤ 0.1 | 「训练 logprob 与 rollout logprob 全程 < 0.1（实测 < 0.05），**已排除 off-policy**」 |
| `entropy_collapse` | `entropy_loss` 末值 ≥ 首值 | 「熵未坍缩（实测 {first} → {last}，反向），**已排除熵坍缩**」 |
| `kl_explosion` | `ppo_kl` 全程 < 0.05 | 「PPO KL 量级 1e-6，**已排除 KL 爆炸**」 |
| `not_infra_abort` | `agent_timeout` 占 abort 比例 > 50% | 「abort 主因是 agent 时间预算耗尽（{n} 例），**不是基建问题**」 |

> **【设计】** `exclusions` 同时输出「排除结论」与「**反例**」：
> 若 `logprob_abs_diff` 峰值 > 0.1（task11 实测 0.1054），则**不能**输出 `off_policy` 排除项，
> 而是输出一条 `verdict`（train/rollout 脱节）。**排除性结论必须可被数据推翻。**

#### 3.2.4 【新增】修正 `train_drop_reason` 的读取位置

见 §3.1.4。

#### 3.2.5 【v2 更正】不需要 `step_alignment_service`

见 §2.5。**v1 §3.3.1/§3.3.2 的两个模块取消**，改为在 `train_health_service` 内按 `rollout_id` 直接对齐。
v1 §3.3.3 的「按训练步看数据质量」交叉视图**保留**，但实现路径简化为「TB 曲线 + 工具侧组构成柱，同一 rollout_id 轴」。

### 3.3 【P1】保留 v1 的「按训练步看数据质量」交叉视图（实现简化）

**【设计】** X 轴 = rollout_id（**不是** train step，简化对齐）。每个 rollout 叠两条：

- **上半**：TB 曲线（`removed_ratio` / `loss_eligible_attempts` / `time_budget_exceeded` / `entropy_loss`）
- **下半**：rollout 数据质量柱（mixed / all_pass / all_fail / dropped 四类组占比）

用户一眼能看出：**「rollout 55 开始 `agent_timeout` 暴涨，同步 dropped 组占比从 38% 涨到 81%」** —— 这就是因果。

**【设计】** 实现为 `templates/reward_diagnosis.html` 内的第 5 个 section（见交付物 2），
不新建页面，避免增加导航负担。

### 3.4 【P1】失败分类对齐 slime —— 别丢掉「谁的问题」

#### 3.4.1 扩充 `utils/failure_reason_taxonomy.py`（不新建 `slime_attempt_classifier.py`）

**【v2 更正】** v1 建议新建 `utils/slime_attempt_classifier.py` 复刻 `classify_attempt`。
但仓库已有 `utils/failure_reason_taxonomy.py`（`normalize_reason` / `classify_from_fields` /
`reason_label` / `agent_outcome_reason`），且 `AGENT_OUTCOME_REASON_CODES` **已经**
把 `time_budget_exceeded` 映射为 `agent_time_budget_exceeded`。

**【设计】** **就地扩展**，不新建平行模块：

| 改动 | 内容 |
| --- | --- |
| 拆分 `timeout` category | 新增独立 category **`budget_exhausted`**，与 `timeout`（上下文打满）分属不同 label |
| `CATEGORY_LABELS` 增补 | `"budget_exhausted": "时间预算耗尽"`；`"timeout"` 改为 `"上下文打满 / 截断"` |
| `normalize_reason` 判定顺序 | `agent_outcome == "time_budget_exceeded"` **优先于** `_TIMEOUT_TOKENS` 里的 `"time_budget"` 匹配（当前会落到 `timeout`） |
| `reason_label` | `agent_time_budget_exceeded` → 「时间预算耗尽（60 分钟墙钟）」；`slime_context_limit_reached` → 「上下文打满」 |
| 新增 `budget_exhausted_label(reason)` | 区分「超时但有产出（status=completed）」与「被杀无产出（status=aborted）」——**【实测】** task4 是前者、task8 是后者 |

**【设计】** 判定优先级（`normalize_reason` 内）：

```
1. abort_reason 存在：
     agent_time_budget_exceeded  → ("budget_exhausted", "agent_time_budget_exceeded")
     其它                        → ("abort", abort_reason)
2. status ∈ ABORTED            → ("abort", "status:aborted")
3. agent_outcome == "time_budget_exceeded" → ("budget_exhausted", "agent_time_budget_exceeded")
4. status ∈ TRUNCATED / context 类错误码   → ("timeout", ...)
5. pass / fail / other（按 reward 判定）
```

**【设计】** **保持向后兼容**：`budget_exhausted` 是**新增** category，原 `timeout` 语义不变；
但所有把 `category == "timeout"` 当作「上下文打满」的页面需要同步更新文案。
受影响的既有页面：`templates/failure_overview.html`（结局柱 / agent 结束态聚类）、
`templates/reward_diagnosis.html`（行为卡片）、`templates/task_detail.html`（类型柱）。
**这是本方案中唯一的破坏性变更，必须在同一个 PR 内完成全部文案同步。**

#### 3.4.2 「谁的问题」三分类

| 归属 | slime 分类 | 处置 |
| --- | --- | --- |
| **基建问题**（不该算策略失败） | `aborted:*`（非 time_budget）、`evaluator:*` | 修基建；**建议从训练信号中剔除** |
| **Agent 行为问题**（新增，v2） | `agent:time_budget_exceeded` | **提预算 / 限制单轮输出 / 隔离排队**（**不是**修沙箱） |
| **策略问题**（真该学） | `agent:no_diff`、`evaluator:tests_failed` | 保留为负样本；这才是模型要学的 |
| **中性/成功** | `resolved` | 正样本 |

**【实测】** 现有行为卡片里 `abort` 被放在最高优先级（基建优先修）——**方向对但粒度太粗**，
因为 task8 的 abort 里 **94.7% 是 agent 行为问题**。v2 把 `abort` 拆成
「基建中止」与「时间预算耗尽」两张卡。

#### 3.4.3 行动杠杆从「文字」升级为「可执行参数建议」

当前 `_BEHAVIOR_META` 的 `lever` 是纯文字：

```python
"timeout": {
    "lever": "调 max_turns / 上下文长度，或加长度惩罚；勿把 timeout 当普通 fail。",
}
```

**升级为结构化建议**，带具体参数名与取值来源：

```python
ACTION_CATALOG = {
    "agent_budget_exhausted": {
        "title": "agent 时间预算耗尽",
        "actions": [
            {"kind": "param", "name": "SWE_AGENT_TIME_BUDGET_SEC",
             "current": "从 resolved_config.yaml 读", "suggest": "3600 → 5400（实测 100% 撞 3600s 墙）"},
            {"kind": "param", "name": "SWE_DAPO_MIN_SURVIVING_ATTEMPTS",
             "current": "从 env 读", "suggest": "4 → 3（实测组污染放大：1 个超时即可能整组作废）"},
            {"kind": "filter", "name": "SWE_EXCLUDE_TIME_BUDGET",
             "current": "从 env 读", "suggest": "保持 1，但需配合预算提升，否则样本流失更快"},
            {"kind": "param", "name": "SLIME_MAX_TOKENS_PER_RESPONSE",
             "current": "从 env 读", "suggest": "按实测单轮 token 966 上调 / 或在 prompt 约束单轮输出"},
        ],
        "evidence_required": ["phase_time/agent_run 分布", "attempt_generated_response_tokens / completed_turns", "train_drop_reason 直方图"],
    },
    "queue_wait_pressure": {
        "title": "沙箱排队挤占有效推理时间（辅因）",
        "severity": "warn",
        "actions": [
            {"kind": "infra", "name": "沙箱池大小 / 复用策略",
             "current": "从 env 读", "suggest": "实测 sandbox_boot_queue p50 159s → 312s；扩池或提并发以压缩排队"},
            {"kind": "infra", "name": "并发提交节奏",
             "current": "从 env 读", "suggest": "按 queue 分桶撞墙率（20.7% / 33.9% / 80.0%）反推需要的并发"},
        ],
        "evidence_required": ["phase_time/sandbox_boot_queue", "phase_time/agent_run", "按 queue 分桶的撞墙率"],
        "note": (
            "排队是**加剧因素**而非第一主因（撞墙 100% 落在 agent_run ≥3600s，"
            "根因是 agent 自身跑不完）；降排队只能**间接**降低撞墙率。"
            "文案不得写「排队吃掉了 agent 预算」。"
        ),
    },
    ...
}
```

每条 action 带 `evidence_required`，前端展示「支撑该建议的证据」，避免空口建议。

### 3.5 【P1】题目池健康度 —— 搞清楚「还有多少可学的题」

#### 3.5.1 【v2 修正】把「末次通过率」换成「累计通过率三态」

**【实测】v1 的「末次通过率」口径在 task4 完全失效**：

| 任务 | 题数 | 每题平均出现次数 | 每题出现间隔（step） | 「末次 ≈ 首次」？ |
| --- | --- | --- | --- | --- |
| task4 | **688** | **1.47**（最大 2） | **31.1** | **是**（几乎每题只出现 1 次） |
| task8 | 64 | 16.0 | 4.0 | 部分重叠 |
| task9 | 32 | 24.0 | 2.0 | 否 |
| task11 | 64 | 12.5 | 3.9 | 部分重叠 |

→ task4 的「末次通过率」≈「首次通过率」，**无法反映学习进展**，v1 基于它的「≥0.875 判饱和 / ≤0.125 判太难」判定全部失效。

**【设计】** 改为**按题目累计通过率分档**（跨全部 rollout 聚合）：

```
stable_pass   : 累计通过率 >= 0.9    （已掌握 → 移出训练集）
gradient      : 0.1 < 累计通过率 < 0.9 （可学习，GRPO 最有价值）
stable_fail   : 累计通过率 <= 0.1    （未掌握 / 太难 → 降采样）
```

**【实测】** 按此口径（分母 = 该题的可测 attempt 数）：

| 任务 | 题数 | `stable_pass` | `gradient` | `stable_fail` |
| --- | --- | --- | --- | --- |
| task4 | 688 | 265（**38.5%**） | 349（50.7%） | 51（7.4%） |
| task8 | 64 | 13（20.3%） | 46（71.9%） | 5（7.8%） |
| task9 | **32** | 3（9.4%） | **29（90.6%）** | 0（0.0%） |
| task11 | 64 | 10（15.6%） | 53（82.8%） | 1（1.6%） |

→ **task9 的 90.6% 的题是「可学习」的**（每题出现 24 次，有充分机会分化），这是它成功的结构原因。
→ **task4 有 38.5% 的题一次都没通过**（`stable_fail` 7.4% + 部分 `stable_pass` 的边界抖动），
说明 688 题里有大量**不可学**的题，稀释了梯度。

#### 3.5.2 【v2 新增】采样密度 —— v1 完全遗漏的维度

**【实测】每步固定采样 16 组**（全部任务 `groups_per_step = 16.0`）。
题目池越大，单步能覆盖的题目越少，梯度密度越低：

| 任务 | 题目池 | 每步组数 | 总 step 数 | 每题平均出现次数 | 每题出现间隔（step） | **梯度密度（mixed 组占比）** |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 96 | 16 | 6 | 1.0 | — | 38.5% |
| 2 | 338 | 16 | 33 | 1.6 | 16.3 | 41.9% |
| 3 | 640 | 16 | 40 | 1.0 | — | 30.8% |
| **4** | **688** | 16 | 63 | **1.5** | **31.1** | **48.1%** |
| 6 | 128 | 16 | 8 | 1.0 | — | 53.9% |
| 7 | 128 | 16 | 47 | 5.9 | 7.9 | 54.9% |
| 8 | 64 | 16 | 64 | 16.0 | 4.0 | 50.9% |
| **9** | **32** | 16 | 48 | **24.0** | **2.0** | **77.6%**（旧口径）<br>**68.4%**（最终定稿） |
| 10 | 128 | 16 | 111 | 13.9 | 8.0 | 58.3% |
| 11 | 64 | 16 | 50 | 12.5 | 3.9 | 58.0% |

> **【实测】** 简报 §3 给出的「梯度密度」列（32题→50%、64题→25%、688题→2.3%）是用
> 「混合组数 / 组数」在**早期 rollout** 上算的；全量重算后 task9 的 mixed 组占比是 **77.6%**（简报 §3.1 表一致）。
> **【v2 三次修订】** 该值是**旧口径**（未把 `ungradable` 计入 dropped）；按最终定稿口径（§1.5.7）
> task9 为 **68.4%**、`dropped` 占比 **15.6%**；task8 为 **35.6%** / **46.0%**。
> **跨任务横比必须统一口径。**
> 本方案统一采用**全量 mixed 组占比**作为 `gradient_density` 的口径，并保留「每题平均出现次数」
> 与「每题出现间隔」作为**采样密度**的独立指标（三者一起看才完整）。

**【设计】** 新增 `utils/difficulty_profile.py`，输出：

| 字段 | 含义 |
| --- | --- |
| `n_problems` | 题目池大小 |
| `n_stable_pass` / `n_gradient` / `n_stable_fail` | 三态题数 |
| `stable_pass_ratio` / `gradient_ratio` / `stable_fail_ratio` | 三态占比 |
| `appearances_mean` / `appearances_max` | 每题平均 / 最大出现次数 |
| `appearance_gap_mean` | 每题相邻出现间隔（step）均值 |
| `sampling_density` | `appearances_mean`（**采样密度**，v2 新增） |
| `gradient_density` | mixed 组占比（**由 §3.1.2 提供**） |
| `n_gradient_problems` | 可学习题数量 |

#### 3.5.3 题目池健康度仪表

**【设计】**

```
题目池健康度（task #N）
  总题数         688
  ├ stable_pass 265 (38.5%)   ← 已掌握，建议移出
  ├ gradient    349 (50.7%)   ← 真正在学习
  └ stable_fail  51 (7.4%)    ← 未掌握，建议降采样

  采样密度：每题平均出现 1.47 次 · 间隔 31.1 step   ← ⚠ 密度过低，学习信号无法累积
  梯度密度：48.1%（每步 16 组中 7.7 组是 mixed）    ← 目标 ≥ 50%
```

**【设计】** 「采样密度」这一行在 task4 会显示为**红色警示**（`appearances_mean < 3`），
因为 1.47 次出现不足以让一道题从「全挂」分化到「偶尔通过」。

#### 3.5.4 「太容易 / 太难」筛选放宽

现有实现要求「每次都全通过（8/8）」/「每次都 0/8」，太严。按 §3.5.1 的连续阈值重做，
并支持按**题目池**（全量）而非仅按**已采样**筛选。

#### 3.5.5 建议重采样清单

**【设计】** 输出「建议移出 / 建议降采样 / 建议新增」的题目清单，可导出为新的数据集筛选条件。

**必须标注每题的 `n_appearances`**：样本太少时判定不可靠（task4 每题 1.47 次），
需标「**数据不足**」并**降权**，不能与 task9（每题 24 次）的同口径判定等价看待。

**【设计】** 三张清单：

| 清单 | 来源 | 动作 |
| --- | --- | --- |
| `remove` | `stable_pass`（累计通过率 ≥ 0.9） | 移出训练集 |
| `downsample` | `stable_fail`（累计通过率 ≤ 0.1） | 降采样；若有更接近 0.2–0.5 的替代题则替换 |
| `keep_boost` | `gradient`（0.1 < 通过率 < 0.9） | 提高采样权重 |

### 3.6 【P1】闭环行动面板 —— 从「看」到「做」

在奖励诊断页顶部加一个 **「诊断结论」区块**，把 §3.0.7 / §3.2.3 / §3.5 的输出汇总：

```
┌─ 诊断结论 ────────────────────────────────────────────────────┐
│ ⚠️ 主要瓶颈：agent 时间预算耗尽导致训练样本枯竭                 │
│                                                                  │
│ 证据：                                                            │
│   · time_budget_exceeded 占比 0% → 62.5%（rollout 55 峰值）       │
│   · 单轮生成耗时 18.1s → 86.0s（+375%），单轮 token 349 → 976     │
│   · 轮数 62 → 35（下降）、峰值上下文 88.5K → 70.4K（下降）        │
│     → 不是轮数爆炸、不是上下文打满                                 │
│   · 整组丢弃率 0% → 93.8%，可训练 attempt 78 → 7                │
│   · abort 构成：时间预算 1425 例 vs 沙箱类 14 例（1.1%）           │
│                                                                  │
│ 已排除：                                                          │
│   · off-policy（logprob 差 < 0.05）                              │
│   · 熵坍缩（熵 0.15 → 0.64，上升）                                │
│   · KL 爆炸（KL ~1e-6）                                          │
│                                                                  │
│ 建议动作：                                                        │
│   ① 把 agent 时间预算 60 → 90 分钟                               │
│   ② 限制单轮输出长度（单轮 token 已 +180%，prompt 约束）          │
│   ③ 降低沙箱排队（辅因：挤占有效推理时间，间接降撞墙率）          │
│                                                                  │
│ 预期效果：可训练 attempt 从 7 恢复到 ~100，removed_ratio 从 0.95 │
│           回到 < 0.4                                              │
└──────────────────────────────────────────────────────────────────┘
```

**「预期效果」必须基于历史数据反推**（例如：把历史 rollout 里被 `agent_timeout` 丢掉的组
按历史 mixed 比例重算 mixed 占比），不能拍脑袋。

#### 3.6.1 新增 `service/action_recommendation_service.py`

**【设计】**

```
{
  "verdicts": [ {severity, code, title, evidence[], actions[]} ],
  "dataset_actions": {
    "remove": [instance_id...],      # stable_pass
    "downsample": [instance_id...],  # stable_fail
    "keep_boost": [instance_id...],  # gradient
  },
  "param_actions": [ {name, current, suggest, reason} ],
  "counterfactual": {                # 基于历史重算
    "current_mixed_ratio": 0.24,
    "projected_mixed_ratio": 0.51,
    "method": "把历史 rollout 里被 agent_timeout 丢掉的组按历史 mixed 比例重算"
  }
}
```

### 3.7 【P2】工程化补齐

| 项 | 内容 | 工作量 |
| --- | --- | --- |
| **a. 自动全量补齐** | 新增 `schedule/schedule_rollout_align.py`：定时扫描任务目录 → 增量扫描新 `.pt` → 补 TB 曲线 → 更新注解 sidecar。填上 `schedule/` 空目录 | 中 |
| **b. CLI 子命令** | `rollout-insight reward-diagnosis show --task-id N`、`rollout-insight train-health --task-id N`、`rollout-insight difficulty --task-id N`。让 agent 能直接取数做自动化归因 | 小 |
| **c. tokenizer 精确统计** | 用已有 `_load_tokenizer` / `_discover_tokenizer_path` 替换 `token_estimate` 的 3.5 系数；响应长度、上下文占用改用真实 token | 小 |
| **d. 精度修正** | `DBTask.mean_reward` 从 `int(round(mean*1000))` 改为 `Float`；或明确注释「定点存储，×1000」并在所有读处还原 | 小 |
| **e. 响应统计自动化** | `advance_response_tool_stats` 的手动点按钮 → 改为扫描后自动后台补齐 + 进度显示 | 小 |
| **f. 通过率三口径在所有页面统一** | §3.1.3 的三种口径在所有页面统一 | 中 |
| **g. 跨任务对比** | 同任务内跨 epoch 纵向对比为主，跨 run 横向对比作为 P2：统一指标 schema 后做「实验登记表」 | 中 |

---

## 4. 落地顺序（v2 重排：按实测优先级）

### 第一批：时间预算 + 口径修正（P0）

**目标：抓住第一主因，把已有的错误结论纠正过来。**

| # | 内容 | 落点 |
| --- | --- | --- |
| 1 | `utils/agent_budget_profile.py` — 时间预算退化画像 + 三模式判定 | **新建** |
| 2 | `utils/failure_reason_taxonomy.py` — `budget_exhausted` 独立 category + `time_budget_exceeded` 与 `context_exhausted` 分离 | **就地扩展** |
| 3 | `utils/slime_grpo_semantics.py` — 组语义（含 `dropped`）+ 三条一等公民指标 | **新建** |
| 4 | `service/reward_diagnosis_service.py` — `summarize_gradient_groups` 加 `dropped`、加 `zero_gradient_ratio` / `wasted_rollout_ratio` / `gradient_density`；`build_reward_levers` 加 6 条新杠杆 | **就地扩展** |
| 5 | `utils/rollout_pt_parser.py` — `train_drop_reason` 改读 `sample.metadata` + 输出 drop reason 直方图 | **就地扩展** |
| 5b | **【P1 · 代码修复】attempt 键空间混用**：入库键改为**组内统一序号**，避免同组内 `rollout_id=None`(idx 回落) 与真实 `rollout_id` 落在不同键空间 → 真实通过样本被误记为 aborted（§1.5.3.1 机制 ②） | **就地扩展** |
| 5c | **【P1】`.pt` 指纹校验**：读 `.pt` 前比对 `pt_size_bytes`/`pt_mtime`，不一致则先清理该 rollout 的旧行再入库（§1.5.3.1 机制 ④）。**已有 `rescan` 可修复**（`scan_task(force=True)` 会 purge），重点是自动检测 + 主动提示 | **就地扩展** |
| 5d | **【P1 · 阻塞项】`ri_attempt_table` 补 4 列**：`train_drop_reason` / `abort_reason` / `agent_outcome` / `diagnosis`（+ `agent_exit_code` / `agent_tool_calls`）。**没有 `train_drop_reason` 就无法精确复现 slime 的组语义**（实测 DB-only 口径 task8 43/64、task9 13/48 vs `.pt`-based 64/64、48/48，§2.1(e)） | **就地扩展（加列，非建表）** |
| 6 | `service/problem_analysis_service.py` — 通过率三口径并存 | **就地扩展** |
| 7 | `templates/reward_diagnosis.html` — 新增「时间预算健康度」卡片 + 新杠杆展示 | **就地扩展** |
| 8 | 配套 `tests/test_*.py` + `python scripts/scan_func_md.py` 重生成 func.md | — |

**验收**：

- 奖励诊断页「时间预算健康度」对 task8 判定 `per_turn_slowdown`、对 task9 判定 `healthy`
- `wasted_rollout_ratio`（dropped 组占比）与 TB `rollout/removed/count_too_few_survivors_*` + `count_agent_timeout` 可对账
- `time_budget_ratio` 与 TB `rollout/outcome/time_budget_exceeded` 误差 < 5%
- 失败分类里 `agent_time_budget_exceeded` 与 `slime_context_limit_reached` 落到**不同** category
- 页面明确标注通过率三种口径的差异
- **键空间修复后**：task4 rollout 2 group 41 的 5 个 `reward=1.0` 样本能正确入库，
  且该组通过率从「0/8」修正为「5/8」；新增回归用例锁死该场景
- **历史数据回填**：修复后需重扫 task1/4/7（顺便补齐未入库的 rollout 6-8 / 63-64 / 47-49，
  并把 task3/4 中被重跑的 6 个 `.pt` 的旧行清掉）
- **指纹一致性用例**：构造「`.pt` 被替换（size/mtime 变化）」的场景，
  验证扫描器不会把旧 attempt 行与新 `.pt` 混在一起
- **`train_drop_reason` 落库用例**：验证解析 `.pt` 后该列正确写入，
  且组语义在「有该列」时与 slime 权威口径逐 rollout 全对（task8 64/64、task9 48/48）


### 第二批：训练侧接入 + 题目池（P1）

**目标：让下次排查不必再手工翻 TB。**

| # | 内容 | 落点 |
| --- | --- | --- |
| 1 | `utils/tb_metric_catalog.py` — 按实测 458 tag 扩充（`removed/*` / `population/*` / `funnel/*` / `outcome/*` / `selected/attempt/*` / `train/*`） | **就地扩展** |
| 2 | `utils/tb_series_reader.py` — 前缀批量取 + sidecar 缓存 + `find_anomalies` | **就地扩展** |
| 3 | `service/train_health_service.py` — 训练健康度聚合 + `verdicts` + `exclusions` | **新建** |
| 4 | `utils/difficulty_profile.py` — 题目三态 + 采样密度 + 梯度密度 | **新建** |
| 5 | `service/action_recommendation_service.py` — verdict + 数据集动作 + 参数动作 + 反事实估算 | **新建** |
| 6 | `templates/reward_diagnosis.html` — 「诊断结论」面板 + 「按 rollout 看数据质量」交叉视图 | **就地扩展** |
| 7 | `templates/failure_overview.html` / `templates/task_detail.html` — `budget_exhausted` 文案同步 | **就地扩展** |
| 8 | `service/task_scan_coverage_service.py` — **扫描完整性**：`.pt` 文件数 vs DB 已扫数、未入库 rollout 列表、补扫入口（§1.5.3.1） | **新建** |

**验收**：

- 训练健康度 `exclusions` 能明确写出「已排除 off-policy」「已排除熵坍缩」「已排除 KL 爆炸」
- 训练健康度 `exclusions` 对 task11 **不输出** off-policy 排除项（实测其 logprob 差峰值 0.1054 越界）
- 题目池三态分类能把 task9 的 32 题正确划为 `gradient` 为主（实测 90.6%）
- `loss_eligible_attempts` 与 `removed_ratio` 的对账误差 < 1%
- 反事实估算能基于历史重算 mixed 占比
- **扫描完整性卡片能正确标出 task1/4/7 的未入库 rollout，并提示补扫**


### 第三批：工程化（P2）

1. `schedule/` 定时任务（自动扫描 + 自动补齐响应统计）
2. CLI 子命令（reward-diagnosis / train-health / difficulty）
3. tokenizer 精确 token 统计
4. `mean_reward` 精度修正
5. 跨 run 横向对比（实验登记）

> **【v2 相对 v1 的顺序变化】** v1 是「口径对齐 + 训练侧可见」并列 P0 先行。
> v2 把**口径修正挪到第一批的尾部**、把**时间预算提到第一位**；
> 同时把 v1 第一批的「训练侧接入」整体下移到第二批。理由：实测表明时间预算是第一主因，
> 口径修正虽然重要但只影响归因方向的准确性，不影响「奖励为什么不涨」这个事实。

---

## 5. 与现有代码的对应关系（落地时的复用点）

严格遵守项目 `func.md` 复用约定，**不新建平行实现**：

| 新增/改动 | 复用 | 说明 |
| --- | --- | --- |
| `utils/agent_budget_profile.py` | 新建（确认无同类） | 时间预算画像是纯计算，放 utils 符合分层 |
| `utils/slime_grpo_semantics.py` | 新建（确认无同类） | 组语义是跨 service 的纯逻辑 |
| `utils/difficulty_profile.py` | 新建（确认无同类） | 题目三态 + 采样密度是纯计算 |
| `summarize_gradient_groups` | 就地扩展 `service/reward_diagnosis_service.py` | **不新建 service** |
| `build_reward_levers` | 就地扩展 | 跨任务对比页复用同一份实现 |
| `summarize_pool_health` | 就地扩展（改为累计通过率三态） | 同上 |
| `summarize_pass_visits` | 就地扩展 `service/problem_analysis_service.py` | 三口径并存 |
| `utils/tb_metric_catalog.py` | 就地扩展 | 已有 `classify_tb_tag` / `describe_tb_tag`，加维度不重写 |
| `utils/tb_series_reader.py` | 就地扩展 | **`read_scalar_series` 已存在**，只补前缀批量取 + 缓存 + `find_anomalies` |
| `service/train_health_service.py` | 新建 | 组合 TB reader + 组语义模块 |
| `utils/failure_reason_taxonomy.py` | 就地扩展 | **已有 `normalize_reason` / `AGENT_OUTCOME_REASON_CODES`**，只加 `budget_exhausted` category |
| `utils/rollout_pt_parser.py` | 就地扩展 | `summarize_train_sample_stats` 改读 `sample.metadata` |
| `utils/analyze_dump.py` / `utils/attempt_artifacts.py` | **直接复用** | `.pt` 加载 / summary 索引已有 sidecar 缓存 |
| `service/action_recommendation_service.py` | 新建 | 组合上面三者 |
| `templates/reward_diagnosis.html` | 就地扩展 | 新增卡片 |
| `static/ri_multi_chart.js` | **直接复用** | 共享多序列折线渲染（commit `54df1a4`），新卡片直接用 `window.RiMultiChart.renderLineChart` |
| ~~`utils/slime_attempt_classifier.py`~~ | **取消**（v1 方案） | 已有 `failure_reason_taxonomy.py`，新建会平行 |
| ~~`service/step_alignment_service.py`~~ | **取消**（v1 方案） | 实测 `rollout_id` 就是天然 join key，见 §2.5 |
| 新表 | **不新建表** | 全部可由 `ri_attempt_table` / `ri_rollout_table` + TB + `.pt` 现算 |

**需要新增的 DB 字段（少量，v2 修正 + v2 三次修订补充）**：

**`ri_attempt_table`（P1 · 组语义复现的必要条件，见 §2.1(e)）**：

| 字段 | 类型 | 来源 | 为什么必须 |
| --- | --- | --- | --- |
| **`train_drop_reason`** | `String` | `.pt` `sample.metadata.train_drop_reason` | **关键输入**：组语义判定的 `ungradable` / `agent_timeout` / `zero_std_*` / `too_few_survivors_*` 只存在于这里。**【实测】** 仅用现有 DB 字段的规则与 slime 权威口径逐 rollout 全对率只有 task8 43/64、task9 13/48；补上后达 **64/64、48/48** |
| **`abort_reason`** | `String` | `.pt` `sample.metadata.abort_reason` | 区分「基建 abort」与「agent 侧时间预算耗尽」 |
| **`agent_outcome`** | `String` | `.pt` `sample.metadata.agent_outcome` | 区分「时间预算耗尽」与「上下文打满」 |
| **`diagnosis`** | `String` | `.pt` `sample.metadata.diagnosis` | 区分「策略失败」与「评测器故障」 |
| `agent_exit_code` | `Integer` | `.pt` `sample.metadata.agent_exit_code` | slime 已写入 metadata，工具目前没落库 |
| `agent_tool_calls` | `Integer` | `.pt` `sample.metadata.agent_tool_calls` | 同上（另有 `agent_tool_errors`） |

**`ri_rollout_table`**：`rollout_step`（Integer）—— 训练 step 映射。**v2 降级为可选**：
  实测 `rollout/funnel/*` 已按 rollout step 记录，若要精确对齐 TB 的 `rollout/step` 才需要。

**【设计】** **不新建表**是本方案的硬约束：时间预算画像从 `.pt` / summary 现算，
训练健康度从 TB 现算，题目池三态从 `ri_attempt_table` 现算。
**加列**（而非建表）：SQLite 走 schema touch；老数据这些列为空 → 组语义走 DB-only 近似口径，
页面须标注「本任务无 `train_drop_reason`，组语义为近似值」。


---

## 6. 风险与边界

| 风险 | 应对 |
| --- | --- |
| **TB 曲线与 rollout 对不齐**（async 训练、权重同步延迟） | v2 改用 `rollout_id` 直接对齐（实测 `rollout/funnel/*` 与 `removed_ratio` 同 step 可对账，误差 < 1%）；仍保留可选的 `rollout_step_offset` 参数校正；**绝不静默错配**，页面上显示 `alignment.source` 与 `max_offset` |
| **TB 读全量 events 慢** | sidecar 缓存（`event_dir + mtime` 失效）+ 按前缀批量取；只读需要的 tag（实测全量 458 tag 约 1.3–2.0s） |
| **训练侧 tag 可能不存在** | 全部走「存在才展示」，缺失显示「该 run 未上报此指标」。**【实测】** task4/task9 无 `rollout/removed/count_agent_timeout`、无 `train/kl_loss`、无 `rollout/zero_std/count_-1.0` |
| **与 slime 升级脱节** | 语义模块**单一真源** + 每条分类标注 slime 源码路径；slime 升级时只改一处。**【实测】** 本批 filter 路径是 `check_surviving_reward_nonzero_std`（不是 v1 假设的 `check_no_aborted_samples`），且 `SWE_DAPO_MIN_SURVIVING_ATTEMPTS` 可配 → 阈值必须可配 |
| **口径改动导致历史数据不可比** | 版本化：注解 sidecar 有 `version`；口径变更时 bump version 并在 UI 提示「口径已更新」。**【v2 新增】** `budget_exhausted` 是破坏性变更，必须同 PR 完成全部文案同步 |
| **过拟合到单次故障** | verdict 规则必须基于**趋势 + 阈值**，不是单点。**【实测】** task9 也退化（单轮 +52%）但幅度小 7 倍 → 判定必须用连续量而非布尔开关 |
| **`budget_exhausted` 破坏既有页面** | 唯一破坏性变更。落地前跑一遍 `tests/test_failure_reason_taxonomy.py` 全量，并人工核对 `failure_overview.html` / `task_detail.html` / `reward_diagnosis.html` 三处结局柱文案 |
| **排除性结论被后续数据推翻** | **【设计】** `exclusions` 每条都带判定条件，条件不满足时**必须**不输出该排除项（实测 task11 的 logprob 差峰值 0.1054 已越界，工具必须能把它标成 verdict 而不是排除项） |
| **大 `.pt` 文件拖慢页面** | 时间预算画像优先走 L2（summary sidecar 缓存），仅在缺失时回落 L1；页面加载不阻塞首屏 |

---

## 7. 一页纸总结

**当前最大的问题不是「功能不够」，也不是 v1 说的「口径不一致」。**

**【实测】** 真正的第一主因是：**agent 时间预算耗尽 → 训练样本枯竭**。

证据链（task8 全量 8213 sample / 8320 份 summary / 458 个 TB tag）：

1. task8 的 **1425 例 abort 全部是 `agent_time_budget_exceeded`**；沙箱类仅 **14 例（1.1%）**
   → **v1「abort 升高 = 基建问题，先修沙箱/driver」的判断被推翻**。
2. 全部 1425 例撞墙样本的 `agent_run` ∈ [3606.9s, 3903.2s]，**100% ≥ 3600s** → 精确的 60 分钟墙钟。
3. 退化机制是**单轮变慢**：单轮生成耗时 **18.1s → 86.0s（+375%）**、单轮 token **349 → 976（+180%）**；
   **不是**轮数爆炸（turns 62 → 35 **下降**）、**不是**上下文打满（peak ctx 88.5K → 70.4K **下降**）、**不是**基建故障。
4. 放大机制是**组污染**：一个 attempt 被杀 → 存活数跌破 4 → `too_few_survivors_N` → **整组丢弃**。
   整组丢弃率从 0% 涨到 **93.8%**，可训练 attempt 从 **78 掉到 7**。
5. 对照组 **task9 单轮只 +52%**、撞墙率 1.46% → 机制成立、幅度是关键。

**口径修正（v1 §3.1）依然要做，但它是次要问题**：它会让归因方向偏，不会让奖励不涨。
同样重要的还有 v1 完全遗漏的 **题目池 → 采样密度 → 梯度密度** 维度（task9 32 题 → mixed 68.4%（最终定稿口径）唯一成功；
task4 688 题、每题只出现 1.47 次 → 「末次通过率」口径直接失效）。

因此三件事按顺序做：

1. **抓第一主因**（§3.0）——时间预算画像与三模式判定（`per_turn_slowdown` / `turn_count_blowup` /
   `context_exhaustion` / `healthy`）、失败分类里把「时间预算耗尽」与「上下文打满」分开、
   `wasted_rollout_ratio` 变成一等公民指标。
2. **让结论变正确**（§3.1）——组语义加 `dropped`（真正的判据是「存活数 < 4」而不是「含 abort」）、
   通过率三口径并存、`train_drop_reason` 从「死指标」改成「一等指标」（v1 说它不存在是错的）。
3. **打开训练侧 + 补上采样密度**（§3.2 / §3.5）——按实测 458 tag 扩充目录（含 v1 与 plan 都漏掉的
   `rollout/funnel/*` / `rollout/population/*` / `rollout/outcome/*`）、
   训练健康度带 `verdicts` + **`exclusions`（已排除 off-policy / 熵坍缩 / KL 爆炸）**、
   题目池改用累计通过率三态 + 采样密度。

最关键的新视图：**奖励诊断页的「时间预算健康度」卡片**——四条曲线叠加 + 退化模式判定 +
「已排除的病因」清单 + 新杠杆。它把「训练奖励为什么不涨」从一个需要脑内拼接的问题，变成一眼可见。

---

