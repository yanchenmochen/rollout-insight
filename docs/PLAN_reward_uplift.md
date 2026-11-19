# rollout-insight 训练奖励提升增强方案

> 面向 slime + sglang 的 agentic RL 训练。
> 目标：把 rollout-insight 从「rollout 数据可视化工具」升级为「**奖励不涨时的归因与行动引擎**」。
>
> 调研基线：
> - 本仓库 commit 历史（v0.1.0 → v0.5.2，HEAD = `f5b38c3`）
> - slime 源码 `/home/nanhu-agentic-rl-stack/projects/nanhu-slime`
> - sglang 侧经 slime `slime/rollout/sglang_rollout.py` 接入
>
> 本文只做方案设计，不含代码实现。

---

## 0. 结论速览

| 维度 | 现状 | 核心缺口 |
| --- | --- | --- |
| 诊断能力 | 有「有效梯度组 / 奖励分解 / 行为失败卡片」 | 判定口径与 slime 训练实际**不一致**（见 §2.1），结论会误导 |
| 跨任务对比 | 仅 TensorBoard 多 run 叠曲线 | 没有把 TB 曲线与 rollout 数据**对齐到同一条因果链** |
| 训练进程侧 | 只读 `eval/*` 评测 tag | 完全看不到 `train/*`、`rollout/*`（KL / entropy / zero_std） |
| 可执行性 | 给「杠杆」文字建议 | 不可直接执行，缺参数建议、样本过滤建议、重采样建议 |
| 数据黑洞 | `train_drop_reason` 统计恒为 0 | 被 dynamic_filter 丢弃的组**不落盘**，工具侧看不到 |

**最高优先级的一件事**：现有「有效梯度组」判据与 slime 的 GRPO 组内标准化口径存在系统性偏差，会让「奖励不涨」的归因结论**指向错误方向**。详见 §2.1、§3.1。

---

## 1. 现状盘点：已有什么

### 1.1 数据链路

```
slime sglang_rollout.generate_rollout_async
  └─ dynamic_filter (check_no_aborted_samples / check_reward_nonzero_std)
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
| 洞察 | `/api/insights` | `insight_page_service` | TB scalar 清单 |

### 1.2 已具备的强项（应保留，不要重造）

- **`.pt` 无 torch 依赖加载**（`utils/analyze_dump.py` 的 zip/pickle + `_TorchStub`）——工程上很扎实。
- **sidecar 注解缓存**（`task_annotation_service`）——`.pt` 大文件只解析一次。
- **abort / timeout / reward 三态分离**（`resolve_sample_reward` / `classify_sample_outcome`）——避免把 abort 当失败，这是很多同类工具的通病，本项目做对了。
- **epoch 对齐校验**（`_check_epoch_windows`）——知道「一题一个 epoch 窗口」是否成立。
- **TB 多 run 托管**（`tensorboard_service`）——已用 `--load_fast=false` 修过历史 step 被截断的坑。

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

**后果**：一个「全挂组」被丢弃后，rollout-insight 看到的通过率是**被筛选后的乐观值**，真实生成质量被系统性高估。工具当前的「可测通过率」分母已排除 abort/timeout，但**无法反映「有多少组是被丢掉的」**。

> 你已确认选择「tb_only」路线，因此本方案通过 **TB 的 `rollout/zero_std/count_*` 与 `rollout/dynamic_filter/drop_*`** 补足这块信息，不推动 slime 改 dump 格式。这是可行的：slime 的 `MetricGatherer.collect()` 会把丢弃原因写成 `rollout/dynamic_filter/drop_{reason}` 标量。

**边界 2：`train_drop_reason` 在当前 slime 版本中不存在。**

全仓 grep 无任何写入点。rollout-insight 的 `summarize_train_sample_stats` 按「无 `train_drop_reason` 即保留」判定，因此该指标**恒为 100% 保留**，是无信息量的死指标。

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

一个全挂组（8/8 全 0）标准化后 advantage 全为 0 → **零梯度**，对训练毫无贡献，但它**占用了 rollout 预算**。
一个全过组（8/8 全 1）同理。

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

问题有三层：

**（a）`all_fail` 组被算作「不可学习」，但它对 slime 同样是零梯度。**
slime 的 `check_reward_nonzero_std` 直接丢全等组；即便不过滤，`grpo_std_normalization` 下全 0 组的 advantage 也恒为 0。
但工具把 `all_fail` 和 `infra_only` 区分开，把 `all_fail` 归为「能力不足（值得学）」，把 `infra_only` 归为「基建问题（不可学）」。

从「提升奖励」视角，真正该分三类：

| 组类型 | slime 实际 | 对训练的贡献 | 工具当前归类 | 应归类 |
| --- | --- | --- | --- | --- |
| mixed（有对有错） | 保留，advantage 非零 | **唯一有正贡献** | mixed（可学习） | ✅ 一致 |
| all_pass（全对） | 保留，但 advantage=0 | 零梯度，白占预算 | all_pass | ✅ 一致 |
| all_fail（全错） | 保留，但 advantage=0 | 零梯度，白占预算 | all_fail（隐含"可学"） | ⚠️ 应明确标注「零梯度」 |
| infra_only（全 abort/timeout） | **被 dynamic_filter 整组丢弃** | 负贡献（浪费 rollout + 触发 over-sampling 重试） | infra_only | ⚠️ 应标注「被丢弃 / 重试开销」 |

**（b）`all_fail` 与 `all_pass` 被混为一谈，但两者的训练后果完全不同。**
- `all_fail`：模型**没学会**这道题 → 该题太难 / 数据不足 → 应考虑**换掉或降采样**
- `all_pass`：模型**已经掌握** → 该题无信息量 → 应考虑**从训练集移除**

两者的正确动作**完全相反**，工具把它们放在同一个「非 mixed 即无梯度」的桶里，丢掉了最有价值的区分信息。

**（c）`n_valid_groups` 的口径与 slime 的 `zero_std` 统计重复但不一致。**

- 工具 `DBAttemptService.summarize_reward_metrics_by_rollout` 的 `n_valid_groups` = 去掉 abort/timeout 后「有过有挂」的组数
- slime 的 `rollout/zero_std/count_1.0` / `count_-1.0` = 按 `group_index` 分组后**奖励全等**的组数（**不排除 abort**，abort 的 reward 被写成 0.0，会混进 zero_std 组）

两个数口径不同、含义重叠，却**从未在同一个页面上对照过**。这是「奖励不涨」最常见的直接解释——zero_std 组占比高 → 有效梯度样本少 → 奖励涨不动——而工具目前**看不到 slime 自己算出来的这个指标**。

### 2.2 【P0 · 能力缺失】训练侧指标完全不可见

当前 `utils/tb_metric_catalog.py` 只识别 4 类 tag：

```python
"primary" / "truncated_ratio" / "attempt_count" / "unknown"
```

且明确声明「不给 loss / kl 等训练曲线编造含义」——态度正确，但**结果是训练侧完全空白**。

而 slime 实际产出的 tag 远不止这些：

| tag | 含义 | 对「奖励不涨」的诊断价值 |
| --- | --- | --- |
| `train/kl_loss` | actor 与 ref 的 KL | **KL 爆炸 → 策略跑偏，奖励崩** |
| `train/ppo_kl` | PPO 近似 KL | 同上 |
| `train/pg_clipfrac` | clip 触发比例 | **过高 → 梯度被裁，样本浪费** |
| `train/grad_norm` | 梯度范数 | **趋零 → 无学习信号** |
| `train/entropy_loss` | 熵 | **骤降 → 坍缩，策略僵化** |
| `train/train_rollout_logprob_abs_diff` | **训练 logprob 与 rollout logprob 差** | **>0.1 → train/rollout 不一致（off-policy 程度），奖励与行为脱节** |
| `rollout/zero_std/count_1.0` | 全过组数 | **零梯度组规模** |
| `rollout/zero_std/count_-1.0` | 全挂组数 | 同上 |
| `rollout/repetition_frac` | 重复退化比例 | 复读退化 |
| `rollout/truncated_ratio` | 截断比例 | 上下文打满 |
| `rollout/segments_per_attempt` | 每 attempt 段数 | 多轮 agent 的分段特征 |
| `rollout/response_len/*` | 响应长度分布 | 长度崩塌 / 爆炸 |
| `rollout/dynamic_filter/drop_*` | **动态过滤丢弃统计** | **被丢弃组规模与原因** |
| `rollout/prefix_cache_hit_rate` | 前缀缓存命中 | 性能 |
| `perf/rollout_time/*` | 各阶段耗时 | 慢阶段导致 rollout 积压 |

**`train/train_rollout_logprob_abs_diff` 是这个清单里最关键的一个**。slime 自己在 CI 里断言它 `<= 0.1`：

```python
# slime/backends/megatron_utils/model.py
if args.ci_test and "train/train_rollout_logprob_abs_diff" in log_dict:
    assert log_dict["train/train_rollout_logprob_abs_diff"] <= 0.1
```

它衡量「训练时算出的 logprob」与「rollout 引擎生成时的 logprob」的偏差。偏差大 → 训练分布与生成分布脱节 → **模型在优化的东西和它实际生成的东西不是一回事**，这正是「奖励不涨但 loss 在降」的典型成因。工具目前完全看不到这个信号。

### 2.3 【P1 · 语义缺失】没有「这批数据值不值得训练」的一等公民指标

`problem_analysis_service` 已有 excellent 的雏形：

- 按 `instance_id` 追踪每次出现的通过率
- 首尾差值直方图
- 「太容易」（每次全过）/「太难」（每次全挂）筛选

但它**只回答了"题目难度"这一个维度**，且存在三个缺口：

1. **没有「可学习度」综合评分**。一道题「有时过有时不过」才是 GRPO 最有价值的样本；「总是过」「总是挂」都是零梯度。工具应输出每题的三态分布（稳定过 / 稳定挂 / 分化），而不只是首尾差。
2. **没有难度分层的数据集视图**。要提升奖励，关键是**持续供充分化样本（既过又挂的题）**。应能按「当前通过率区间」统计题目池构成，识别「题目池已饱和（都太简单）」或「题目池太难（都做不出）」。
3. **「太容易」的判定太严**。「每次出现都全通过（8/8）」才叫太容易，但 7/8 也是接近饱和。真正该关注的是**通过率贴近 0 或贴近 1 的题目占比**。

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

这个分类**按优先级排序**，信息量比 rollout-insight 的 `classify_sample_outcome` 更细：

| slime 分类 | 语义 | rollout-insight 对应 |
| --- | --- | --- |
| `aborted:{reason}` | 基建中止 | ✅ abort |
| `resolved` | 通过 | ✅ pass |
| `evaluator:{reason}` | **评测器失败**（含 payload 无效） | ❌ 归入 `fail`，丢失「评测器挂了」这一维 |
| `evaluator:timeout` | 评测超时 | ⚠️ 归入 timeout，但与 agent 侧超时混同 |
| `agent:time_budget_exceeded` | **agent 超预算** | ⚠️ 归入 timeout |
| `agent:no_diff` | **没产出 diff** | ❌ 归入 fail（丢失「空改动」这一关键失败模式） |
| `agent:no_recorded_tool_calls` | **没调工具** | ⚠️ 行为卡片里有类似项，但主类判定优先级不同 |
| `evaluator:tests_failed` | 真的测试没过 | ✅ fail |

**关键缺失**：`agent:no_diff`（agent 没做出任何改动）和 `evaluator:*`（评测器自身问题）在 rollout-insight 里**都退化成笼统的 fail**。这两类的正确动作截然不同：
- `no_diff` → 策略问题（agent 摆烂/没理解任务）→ 该题对该模型太难或 prompt 有歧义
- `evaluator:*` → **基建问题** → 不该算作策略失败，不该污染训练信号

而 `metadata` 里这些字段（`agent_exit_code`、`agent_tool_calls`、`evaluation.reason`、`evaluation.timed_out`）**slime 全都写了**，rollout-insight 却大部分没读。

### 2.5 【P1 · 断裂】rollout 数据与训练指标是两条平行线

现状：
- rollout 侧：`reward_diagnosis_service` 只读 `ri_attempt_table`（纯 DB）
- 训练侧：`insight_page_service` 只读 TB 事件目录

两者**从未在同一坐标系里对齐**。典型症状：用户看到 `train/kl_loss` 飙升，但不知道是哪个 rollout、哪道题、哪类失败导致的；看到通过率下降，也不知道同期 `grad_norm` / `entropy` 是否同时异常。

slime 的 TB step 语义提供了天然的 join key：

```python
# slime/ray/rollout.py
log_dict["rollout/step"] = step          # rollout 侧
log_dict["train/step"] = accumulated_step_id   # train 侧
step = compute_rollout_step(args, rollout_id)
```

`rollout/step` 与 `train/step` 存在**确定的偏移关系**（rollout N 对应 train step M）。只要把「rollout_id → rollout/step」建立起来，就能把 TB 曲线与 `.pt` 文件**逐步对齐**。

**这是整个方案里最高杠杆的一步**：它把「训练进程」和「rollout 数据」从两页纸变成一条因果链。

### 2.6 【P2 · 其它】

| # | 问题 | 位置 |
| --- | --- | --- |
| a | `train_drop_reason` 死指标，恒为 100% | `utils/rollout_pt_parser.py:626` |
| b | `token_estimate` 用「字符数 / 3.5」粗估，与真实 tokenizer 偏差大；已有 tokenizer 能力却未用于统计 | `utils/rollout_pt_parser.py` |
| c | `summarize_pass_visits` 把 **abort/timeout 计入通过率分母**（注释明说），但 slime 的 GRPO 是**整组丢弃**——分母口径与训练实际不符 | `problem_analysis_service.py` |
| d | 响应统计是「按 rollout 手动点按钮渐进补算」，无后台全量补齐 | `advance_response_tool_stats` |
| e | `DBTask.mean_reward` 存 `int(round(mean*1000))` 定点，精度与语义都可疑 | `rollout_scan_service.py:678` |
| f | 无「数据集难度分层」视图，无「题目池健康度」指标 | 全缺 |
| g | CLI 无 reward-diagnosis / problem-analysis 子命令，agent 无法自动化取数 | `cli/rollout_insight.py` |
| h | `schedule/` 是空目录，无定时任务；TB 曲线、rollout 数据无自动对齐刷新 | `schedule/__init__.py` |

---

## 3. 增强方案

按「杠杆 / 成本」排序。每项标注 P0/P1/P2 与预估工作量。

### 3.1 【P0】口径对齐层 —— 让现有结论先变正确

> **这是所有后续工作的地基。若口径不对，后面所有分析都在放大错误结论。**

#### 3.1.1 新增 `utils/slime_grpo_semantics.py`

集中定义「slime 视角」的组语义，与 slime 源码一一对应：

```python
"""slime GRPO 组语义：与 slime/rollout/reward_utils.py + filter_hub 对齐。

组类型（对训练的实际贡献）：
  - mixed       : 组内奖励不全等 → advantage 非零 → 唯一有正贡献
  - all_pass    : 全过 → advantage=0 → 零梯度，但题已被掌握
  - all_fail    : 全挂 → advantage=0 → 零梯度，题未被掌握
  - zero_std    : 奖励全等（含全 0 / 全 1）→ 与 all_pass/all_fail 同义，
                  显式建模以便与 slime 的 zero_std/count_* 对账
  - infra_only  : 组内全为 abort/timeout → 被 check_no_aborted_samples 整组丢弃
"""

GROUP_KINDS = ("mixed", "all_pass", "all_fail", "infra_only")

def classify_grpo_group(rewards, statuses) -> str
def group_gradient_contribution(kind) -> float   # mixed=1.0, 其余=0.0
def group_action_hint(kind) -> str                # 该类组对应的处置建议
```

**关键设计**：判定必须用 **attempt 级奖励**（一个 attempt 一个 reward，不是 segment 级）。slime 的 `collect_attempt_rewards` 已明确：一个 attempt 的多个 segment 共享同一个 terminal reward。rollout-insight 的 `ri_attempt_table.reward` 恰好就是 attempt 级，✅ 对齐。

#### 3.1.2 修正 `summarize_gradient_groups`

- 保留 mixed / all_pass / all_fail / infra_only 四分类，**但改语义标注**：
  - `all_fail` 的文案从「不可学习」改为「**零梯度 · 题未掌握**」，杠杆从「学」改为「**换题 / 降采样**」
  - `all_pass` 标注为「**零梯度 · 题已掌握**」，杠杆改为「**移出训练集**」
  - `infra_only` 标注为「**被 dynamic_filter 丢弃 · 触发 over-sampling 重试**」，杠杆改为「**修基建**」
- 新增 `zero_gradient_ratio` 曲线 = (all_pass + all_fail) / n_groups —— 这是**真正该盯的指标**
- 新增 `wasted_rollout_ratio` = infra_only 组数占比 —— 直接对应「rollout 预算被基建问题吃掉多少」

#### 3.1.3 修正 `summarize_pass_visits` 的分母口径

现状注释明说「abort 与 timeout 计入分母、不计入分子」。

**问题**：slime 对含 abort 的组是**整组丢弃并重采**，不是「保留但算 0 分」。
所以一个「7/8 abort + 1/8 通过」的组，在 slime 眼里**根本不存在**（会被丢弃重采）；在工具眼里却是 12.5% 通过率。

**修正方案**：给通过率**同时输出两种口径**：

| 指标 | 定义 | 用途 |
| --- | --- | --- |
| `pass_rate_strict` | 分母 = 全组 attempt 数（现状） | 看「用户视角的端到端成功率」 |
| `pass_rate_gradable` | 分母 = 去掉 abort/timeout 后的 attempt 数 | 看「slime 视角的有效样本质量」 |
| `pass_rate_slime_effective` | 只统计**未被丢弃的组**，组内平均 | **对齐 slime 真实训练分布** |

并在 UI 上明确标注三者差异，避免误读。

#### 3.1.4 删除或标注 `train_drop_reason` 统计

当前 `summarize_train_sample_stats` 恒为「100% 保留」，是死指标。

**处理**：
- 短期：在 UI 上标注「**当前 slime 版本无 `train_drop_reason` 字段，该指标不生效**」
- 长期：改为统计 **slime 真正使用的丢弃口径** —— 即「该组是否会被 `check_no_aborted_samples` 丢弃」，与 §3.1.1 的 `infra_only` 判定复用同一套逻辑

---

### 3.2 【P0】训练侧指标接入 —— 打开「训练进程」黑盒

#### 3.2.1 扩充 `utils/tb_metric_catalog.py`

新增分类维度，**每个 tag 必须写明来源**（`slime/ray/rollout.py` 或 `slime/backends/megatron_utils/model.py`），不编造含义：

```python
TAG_CATALOG = {
    # ── 训练侧（来源: slime/backends/megatron_utils/model.py）──
    "train/kl_loss":                     ("kl",       "actor 与参考模型的 KL 散度；持续升高 → 策略偏离参考，奖励易崩"),
    "train/ppo_kl":                      ("kl",       "PPO 近似 KL"),
    "train/pg_clipfrac":                 ("clip",     "PPO clip 触发比例；过高 → 有效梯度被裁掉"),
    "train/grad_norm":                   ("grad",     "梯度范数；持续趋零 → 无学习信号"),
    "train/entropy_loss":                ("entropy",  "策略熵；骤降 → 坍缩，多样性丧失"),
    "train/train_rollout_logprob_abs_diff": ("offpolicy", "训练 logprob 与 rollout logprob 绝对差；>0.1 → train/rollout 分布不一致"),

    # ── rollout 侧（来源: slime/ray/rollout.py）──
    "rollout/zero_std/count_1.0":        ("zero_std", "全过组数（零梯度）；过高 → 题目池已饱和"),
    "rollout/zero_std/count_-1.0":       ("zero_std", "全挂组数（零梯度）；过高 → 题目太难或基建差"),
    "rollout/repetition_frac":           ("repetition","响应重复退化比例"),
    "rollout/truncated_ratio":           ("truncated", "截断比例"),
    "rollout/segments_per_attempt":      ("segments", "每 attempt 的平均段数"),
    "rollout/dynamic_filter/drop_*":     ("dropped",  "动态过滤丢弃统计（按原因）"),
    "rollout/prefix_cache_hit_rate":     ("perf",     "前缀缓存命中率"),
    "rollout/response_len/*":            ("length",   "响应长度分布"),
}
```

同时把 `classify_tb_tag` 的 `unknown` 兜底改为「按前缀启发式归类 + 标注『推断』」，并保留「未收录」提示。

**注意保持项目已有原则**：`utils/tb_metric_catalog.py` 的 docstring 明确写「不给 loss / kl 等训练曲线编造含义」。本次扩充不是编造——每条都对应 slime 源码里的真实计算逻辑，且要在 `describe_tb_tag` 里**同时给出源码出处**，让使用者可核验。

#### 3.2.2 新增 `utils/tb_series_reader.py`

现有 `read_latest_scalars` 只取**最后一个点**（`events[-1]`），无法画曲线。

**新模块**能力：

```python
def read_scalar_series(event_dir, tag) -> List[Dict]      # [{step, value, wall_time}]
def read_scalar_history(event_dir, tags=None) -> Dict[str, List[Dict]]   # 多 tag 全量
def find_anomalies(series, rules) -> List[Dict]            # 规则式异常检测
```

复用现有 `EventAccumulator`（`size_guidance={"scalars": 100}` 太小，改为按需放大或用 `event_file_loader` 增量读）。

**性能注意**：`EventAccumulator` 读全量 events 对长 run 会慢。建议缓存到 sidecar（同 `.pt` 的 `{pt}.ri_ann.json` 模式），按 `event_dir + mtime` 失效。

#### 3.2.3 新增「训练健康度」聚合

```python
# service/train_health_service.py
def build_train_health(task_id) -> Dict:
    """
    返回:
      - rollout_side: {zero_std_ratio, all_pass_ratio, all_fail_ratio,
                       dropped_ratio, truncated_ratio, repetition_frac,
                       raw_reward_trend}
      - train_side:   {kl_trend, grad_norm_trend, entropy_trend,
                       clipfrac_trend, logprob_abs_diff_max}
      - alignment:    {rollout_step_to_train_step, n_aligned_steps}
      - verdict:      [{severity, code, title, evidence, action}]
    """
```

**verdict 是本模块的核心价值**——把指标翻译成「结论 + 行动」，而不是一堆曲线。

判定规则示例（全部可追溯到 slime 的失败模式）：

| 条件 | 结论 | 建议动作 |
| --- | --- | --- |
| `zero_std_ratio` > 0.6 且 `all_fail` 为主 | **题目池过难 / 基建不稳** | 降采样全挂题；检查 abort 占比；考虑缩小题目池 |
| `all_pass` 占比 > 0.6 | **题目池已饱和** | 移除饱和题，补充新题 |
| `train/kl_loss` 持续上升 | **策略偏离参考** | 降 lr / 加 KL 系数 / 检查 reward hacking |
| `train/entropy_loss` 骤降 > 50% | **策略坍缩** | 加熵正则 / 提 temperature / 加数据多样性 |
| `train/train_rollout_logprob_abs_diff` > 0.1 | **train/rollout 脱节** | 检查权重同步延迟 / off-policy 程度 / async 配置 |
| `train/pg_clipfrac` > 0.3 | **梯度被裁** | 降 lr / 调 clip 范围 |
| `train/grad_norm` 趋零 + `mixed_ratio` 正常 | **信号被吃掉** | 检查 reward 是否二值化过度、std normalization 是否过度 |
| `repetition_frac` 上升 | **复读退化** | 加重复惩罚 / 调整数据 |
| `truncated_ratio` 上升 | **上下文打满** | 提 max_turns / 提上下文 / 加长度惩罚 |

---

### 3.3 【P0】rollout ↔ train 步骤对齐 —— 把两条线接起来

#### 3.3.1 解决 join key

需要确定 `rollout_id` → `rollout/step` 的映射。方案：

1. **优先**：slime 的 `compute_rollout_step(args, rollout_id)` 逻辑可复刻（读 `args.rollout_global_dataset` / `dataset_size` / `rollout_batch_size`）
2. **兜底**：TB `rollout/step` 的**值序列**与本地 rollout_id 序列**单调对应**——TB 里 `rollout/step` 本身就是 `step`，且 `rollout_id` 递增。可以按「TB 中 rollout 序列的第 N 个点 ↔ 本地第 N 个 rollout」对齐，再逐点校正
3. **再兜底**：`resolved_config.yaml` 里的 `NUM_ROLLOUT` / `ROLLOUT_BATCH_SIZE` / `NUM_STEPS_PER_ROLLOUT` 推算

对应 `train/step`：`train/step` = `accumulated_step_id`，由 `global_batch_size` 与 `rollout_batch_size * n_samples_per_prompt` 决定。slime 里：

```python
# GLOBAL_BATCH_SIZE = ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / NUM_STEPS_PER_ROLLOUT
```

从 env 读这三个值即可推算偏移。

#### 3.3.2 新增对齐服务

```python
# service/step_alignment_service.py
def build_step_alignment(task_id) -> Dict:
    """
    {
      "rollout_to_step": {rollout_id: {"rollout_step": int, "train_step": int}},
      "n_rollouts": int,
      "n_aligned": int,
      "alignment_source": "computed" | "inferred" | "failed",
      "warnings": [...],
    }
    """
```

#### 3.3.3 交叉视图：「按训练步看数据质量」

这是方案里**最直接回答「奖励为什么不涨」**的视图。

**设计**：X 轴 = 训练 step（不是 rollout_id）。每个 step 叠两条：
- **上半**：TB 曲线（`raw_reward`、`kl_loss`、`entropy_loss`、`grad_norm`）
- **下半**：rollout 数据质量柱（mixed / all_pass / all_fail / infra_only 组占比、dropped 组占比）

用户一眼能看出：**「step 45 开始 KL 飙升，同步 all_fail 组占比从 12% 涨到 40%」** —— 这就是因果。

---

### 3.4 【P1】失败分类对齐 slime —— 别丢掉「谁的问题」

#### 3.4.1 新增 `utils/slime_attempt_classifier.py`

**直接复刻 slime 的优先级链**（`examples/coding_agent_rl/diagnostics.py`），但放在工具侧并注明出处：

```python
SLIME_CLASSIFIER_SOURCE = "nanhu-slime/examples/coding_agent_rl/diagnostics.py::classify_attempt"

def classify_attempt_slime(*, reward, metadata) -> str:
    """按 slime 原始优先级分类，输出与 slime 完全一致的 label。

    aborted:{reason} > resolved > evaluator:{reason} > evaluator:timeout
    > agent:time_budget_exceeded > agent:no_diff
    > agent:no_recorded_tool_calls > evaluator:tests_failed
    """
```

**新增 `utils/failure_insight.py` 的 slime 分类**（保留现有分类，并列而非替换）：
- 现有：`classify_sample_outcome`（pass/abort/timeout/fail/other）—— **保留**，它是类型柱的基础
- 新增：`classify_attempt_slime` —— 用于「失败归因」页，明确区分**策略问题 vs 基建问题 vs 评测器问题**

#### 3.4.2 「谁的问题」三分类（本次最重要的语义升级）

| 归属 | slime 分类 | 处置 |
| --- | --- | --- |
| **基建问题**（不该算策略失败） | `aborted:*`、`evaluator:*` | 修基建；**建议从训练信号中剔除** |
| **策略问题**（真该学） | `agent:no_diff`、`evaluator:tests_failed` | 保留为负样本；这才是模型要学的 |
| **中性/成功** | `resolved` | 正样本 |

现有行为卡片（`abort → timeout → no_tool_call → last_not_finish → early_finish`）里，`abort` 被放在最高优先级——这**方向是对的**（基建优先修），但因为没有 `evaluator:*` 分类，**评测器故障仍然被混进「策略失败」**，会误导归因。

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
    "infra_abort": {
        "title": "修复沙箱/driver 基建",
        "actions": [
            {"kind": "param", "name": "MAX_CONTEXT_LEN", "current": "从 env 读", "suggest": "按 attempt_peak_context_tokens 上调"},
            {"kind": "param", "name": "SGLANG_SERVER_CONCURRENCY", "current": "从 env 读", "suggest": "过低会加剧组内等待"},
            {"kind": "filter", "name": "DYNAMIC_SAMPLING_FILTER_PATH", "current": "check_no_aborted_samples", "suggest": "保持；勿把 abort 当失败样本"},
        ],
        "evidence_required": ["abort_reason 分布", "phase_time/sandbox_boot"],
    },
    ...
}
```

每条 action 带 `evidence_required`，前端展示「支撑该建议的证据」，避免空口建议。

---

### 3.5 【P1】题目池健康度 —— 搞清楚「还有多少可学的题」

#### 3.5.1 题目难度三态分类

复用 `problem_analysis_service` 已有的 `instance_id` 追踪，加一层**稳定性分类**：

```python
# utils/difficulty_profile.py
def classify_difficulty(rates: List[float]) -> str:
    """rates = 该题历次出现的通过率序列（按 rollout 升序）
    saturated : 每次都 ≈1（≥ threshold_high）→ 已掌握
    unsolved   : 每次都 ≈0（≤ threshold_low）  → 太难 / 基建差
    gradient  : 有高有低 → 有学习价值
    volatile  : 方差极大 → 需看失败聚类
    """
```

`threshold_low/high` 默认 0.05 / 0.95，可配。

**关键升级点**：不再用「首尾差值」当唯一难度信号（首尾差值会被噪声主导），而用**全序列的分布**。

#### 3.5.2 题目池健康度仪表

```
题目池健康度（task #N）
  总题数         128
  ├ gradient   62  (48%)   ← 真正在学习
  ├ saturated  38  (30%)   ← 已掌握，建议移出
  ├ unsolved   22  (17%)   ← 太难，建议降采样或修基建
  └ volatile    6  (5%)    ← 需人工看失败聚类

  当前 rollout 的 sampled 题目难度构成：
    saturated 45% | gradient 35% | unsolved 20%     ← 理想是 gradient 最高
```

**这是「提升奖励」最直接的抓手**：如果 sampled 题目里 saturated 占 45%，那么大部分 rollout 在做无用功，梯度信号被稀释。

#### 3.5.3 「太容易 / 太难」筛选放宽

现有实现要求「每次都全通过（8/8）」/「每次都 0/8」，太严。按 §3.5.1 的连续阈值重做，并支持按**题目池**（全量）而非仅按**已采样**筛选。

#### 3.5.4 建议重采样清单

输出「建议移出 / 建议降采样 / 建议新增」的题目清单，直接可导出为新的数据集筛选条件。

---

### 3.6 【P1】闭环行动面板 —— 从「看」到「做」

在奖励诊断页顶部加一个 **「诊断结论」区块**，把 §3.2.3 / §3.4.3 / §3.5 的输出汇总：

```
┌─ 诊断结论 ────────────────────────────────────────────────────┐
│ ⚠️ 主要瓶颈：题目池饱和 + 零梯度占比过高                       │
│                                                                  │
│ 证据：                                                            │
│   · all_pass 组占比 38%（step 12 → 45，单调上升）                │
│   · mixed 组占比从 61% 降至 24%                                │
│   · slime 侧 rollout/zero_std/count_1.0 同步上升（n=45 已对账）  │
│                                                                  │
│ 建议动作：                                                        │
│   ① 从训练集移出 38 道 saturated 题（清单 →）                     │
│   ② 补 20 道中难度题（从 unsolved 中选 grader 分 0.3-0.6 的）    │
│   ③ 若 all_fail 占比仍 > 25%，检查沙箱基建（abort_reason 分布 →）│
│                                                                  │
│ 预期效果：mixed 组占比回到 50%+，有效梯度样本 ×2                  │
└──────────────────────────────────────────────────────────────────┘
```

「预期效果」必须**基于历史数据反推**（例如：把历史 rollout 里 saturated 组的算力挪给 gradient 组，重算 mixed 占比），不能拍脑袋。

#### 3.6.1 新增 `service/action_recommendation_service.py`

```python
def build_recommendations(task_id) -> Dict:
    """
    {
      "verdicts": [ {severity, code, title, evidence[], actions[]} ],
      "dataset_actions": {
        "remove": [instance_id...],      # saturated
        "downsample": [instance_id...],  # unsolved
        "keep_boost": [instance_id...],  # gradient
      },
      "param_actions": [ {name, current, suggest, reason} ],
      "counterfactual": {                # 基于历史重算
        "current_mixed_ratio": 0.24,
        "projected_mixed_ratio": 0.51,
        "method": "用历史 rollout 重算：把 saturated 组的算力按题目数比例重分给 gradient 组"
      }
    }
    """
```

---

### 3.7 【P2】工程化补齐

| 项 | 内容 | 工作量 |
| --- | --- | --- |
| **a. 自动全量补齐** | 新增 `schedule/schedule_rollout_align.py`：定时扫描任务目录 → 增量扫描新 `.pt` → 补 TB 曲线 → 更新注解 sidecar。填上 `schedule/` 空目录 | 中 |
| **b. CLI 子命令** | `rollout-insight reward-diagnosis show --task-id N`、`rollout-insight train-health --task-id N`、`rollout-insight difficulty --task-id N`。让 agent 能直接取数做自动化归因 | 小 |
| **c. tokenizer 精确统计** | 用已有 `_load_tokenizer` / `_discover_tokenizer_path` 替换 `token_estimate` 的 3.5 系数；响应长度、上下文占用改用真实 token | 小 |
| **d. 精度修正** | `DBTask.mean_reward` 从 `int(round(mean*1000))` 改为 `Float`；或明确注释「定点存储，×1000」并在所有读处还原 | 小 |
| **e. 响应统计自动化** | `advance_response_tool_stats` 的手动点按钮 → 改为扫描后自动后台补齐 + 进度显示 | 小 |
| **f. 通过率口径双输出** | §3.1.3 的三种口径在所有页面统一 | 中 |
| **g. 跨任务对比** | 你选了「同任务内跨 epoch 纵向对比」为主，跨 run 横向对比作为 P2：统一指标 schema 后做「实验登记表」 | 中 |

---

## 4. 落地顺序（建议分三批）

### 第一批：口径对齐 + 训练侧可见（P0，约 3-5 天）

**目标：让现有结论从「可能错」变成「确定对」，并第一次看到训练侧信号。**

1. `utils/slime_grpo_semantics.py` — 新建，组语义 + 零梯度 + 浪费算力
2. `service/reward_diagnosis_service.py` — 修 `summarize_gradient_groups` 语义、新增 `zero_gradient_ratio` / `wasted_rollout_ratio`
3. `utils/tb_metric_catalog.py` — 扩充训练侧 tag 目录（每条注明 slime 源码出处）
4. `utils/tb_series_reader.py` — 新建，读全量序列（替代只取最后一点）
5. `service/train_health_service.py` — 新建，训练健康度 + verdict 规则
6. `service/problem_analysis_service.py` — 通过率三口径输出
7. `utils/rollout_pt_parser.py` — `train_drop_reason` 死指标标注 / 改为 slime 丢弃口径
8. 配套 `tests/test_*.py` + `python scripts/scan_func_md.py` 重生成 func.md

**验收**：
- 奖励诊断页能显示「零梯度组占比」曲线，与 TB `rollout/zero_std/count_*` 能对账（误差 < 5%）
- 训练健康度卡片能显示 KL / entropy / grad_norm / logprob_abs_diff 曲线与判定
- 页面明确标注通过率三种口径的差异

### 第二批：因果对齐 + 闭环（P0/P1，约 5-8 天）

**目标：把训练曲线与 rollout 数据接到同一条时间轴，并给出可执行动作。**

1. `service/step_alignment_service.py` — rollout_id ↔ rollout/step ↔ train/step
2. 「按训练步看数据质量」交叉视图（新模板）
3. `utils/slime_attempt_classifier.py` — 对齐 slime 失败分类
4. `utils/difficulty_profile.py` — 题目难度三态 + 题目池健康度
5. `service/action_recommendation_service.py` — verdict + 数据集动作 + 参数动作 + 反事实估算
6. 奖励诊断页顶部「诊断结论」面板

**验收**：
- 交叉视图能同时看到 TB 曲线与组构成柱，一眼看出同步变化
- 失败归因能区分基建/策略/评测器三类
- 题目池健康度能给出 remove / downsample 清单
- 反事实估算能基于历史重算 mixed 占比

### 第三批：工程化 + 闭环验证（P2，约 3-5 天）

1. `schedule/` 定时任务（自动扫描 + 自动对齐 + 自动补齐响应统计）
2. CLI 子命令（reward-diagnosis / train-health / difficulty）
3. tokenizer 精确 token 统计
4. `mean_reward` 精度修正
5. 跨 run 横向对比（实验登记）

---

## 5. 与现有代码的对应关系（落地时的复用点）

严格遵守项目 `func.md` 复用约定，**不新建平行实现**：

| 新增/改动 | 复用 | 说明 |
| --- | --- | --- |
| `utils/slime_grpo_semantics.py` | 新建（确认无同类） | 组语义是跨 service 的纯逻辑，放 utils 符合分层 |
| `summarize_gradient_groups` | 就地扩展 | 不新建 service |
| `utils/tb_metric_catalog.py` | 就地扩展 | 已有 `classify_tb_tag` / `describe_tb_tag`，加维度不重写 |
| `utils/tb_series_reader.py` | 新建 | `read_latest_scalars` 保留原样，新模块负责全量序列 |
| `service/train_health_service.py` | 新建 | 组合 `DBAttemptService` + tb reader + 语义模块 |
| `service/step_alignment_service.py` | 新建 | 读 `utils/slime_job_type.load_env_from_run_dir` 拿 env |
| `utils/slime_attempt_classifier.py` | 新建 | 与 `utils/failure_insight.py` 并列，不替换 |
| `utils/difficulty_profile.py` | 新建 | `problem_analysis_service` 复用其 `instance_id` 追踪 |
| `service/action_recommendation_service.py` | 新建 | 组合上面三者 |
| 新表 | **不新建表** | 全部可由 `ri_attempt_table` / `ri_rollout_table` 现有字段算出 |

**需要新增的 DB 字段**（少量）：

- `ri_attempt_table`：`agent_exit_code`（Integer）、`agent_tool_calls`（Integer）—— slime 已写入 metadata，工具目前没落库
- `ri_rollout_table`：`rollout_step`（Integer）—— 训练 step 映射，便于按步聚合

按项目约定，改 `db/` 后需重跑 `python scripts/scan_func_md.py`。

---

## 6. 风险与边界

| 风险 | 应对 |
| --- | --- |
| **TB 曲线与 rollout 对不齐**（async 训练、权重同步延迟） | `step_alignment_service` 返回 `alignment_source` 与 `warnings`；对不齐时**降级为「同 index 假设」**并明确标注，绝不静默错配 |
| **TB 读全量 events 慢** | sidecar 缓存 + 按需 tag；只读需要的 tag |
| **`train_rollout_logprob_abs_diff` 等 tag 可能不存在** | 全部走「存在才展示」，缺失显示「该 run 未上报此指标」 |
| **与 slime 升级脱节** | 语义模块**单一真源** + 每条分类标注 slime 源码路径；slime 升级时只改一处 |
| **口径改动导致历史数据不可比** | 版本化：注解 sidecar 有 `version`；口径变更时 bump version 并在 UI 提示「口径已更新」 |
| **过拟合到单次故障** | verdict 规则必须基于**趋势 + 阈值**，不是单点 |

---

## 7. 一页纸总结

**当前最大的问题不是「功能不够」，而是「现有的『有效梯度组』结论与 slime 的 GRPO 训练口径不一致」。**
这会导致：工具说「有 24% 可学习组，奖励不涨是策略问题」，而 slime 实际看到的是「零梯度组占 60%，其中大半是全挂题」——**归因方向完全反了**。

因此三件事按顺序做：

1. **对齐口径**（§3.1）——让「可学习组」真正等于「slime 会产生非零 advantage 的组」，并区分「已掌握 / 未掌握 / 被丢弃」三类零梯度组，给出**相反**的处置建议。
2. **打开训练侧 + 对齐时间轴**（§3.2 / §3.3）——接入 `kl_loss` / `entropy` / `grad_norm` / `train_rollout_logprob_abs_diff` / `zero_std` / `dynamic_filter drop`，并把 TB 曲线与 rollout 组构成画在**同一 X 轴**上。这一步把「两条平行线」变成「一条因果链」。
3. **给出可执行动作**（§3.5 / §3.6）——题目池健康度（饱和/梯度/未掌握）+ 移除/降采样清单 + 参数级建议 + 基于历史的反事实估算。

最关键的一页新视图：**「按训练步看数据质量」**——TB 曲线在上，组构成柱在下。它把「训练奖励为什么不涨」从一个需要脑内拼接的问题，变成一眼可见。
