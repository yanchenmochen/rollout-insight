# rollout-insight CLI 高频配方（recipes）

> 完整命令树见 `SKILL.md`。本文档是"按场景"的速查手册。

## 全局约定

- 默认 base URL：`http://localhost:39998`（可 `--api-base` 或 `ROLLOUT_INSIGHT_API_BASE`）
- 默认南湖集群：`wuxi`（可 `--cluster x10000`）
- 默认输出格式：`json`（列表场景对人用 `--format table`）
- **写操作默认 dry-run；加 `--submit` 才执行**
- 退出码：`0` 成功；`1` 业务/API 错误；`2` 命令行参数错误；`130` Ctrl-C

## 场景 1：用户刚跑完一次训练，要扫进 rollout-insight

```bash
# 1) 验证 rollout 根目录结构
ls /mnt/zj-data/.../output/slime/<exp>/rollout_dumps/

# 2) 验证 tokenizer 路径
rollout-insight tokenizer test --path /mnt/zj-data/models/.../hf

# 3) 写 tokenizer_dir 到 config.json（dry-run → 确认 → submit）
rollout-insight tokenizer set --path /mnt/zj-data/models/.../hf
rollout-insight tokenizer set --path /mnt/zj-data/models/.../hf --submit

# 4) 建任务（dry-run → 确认 → submit）
rollout-insight tasks create \
  --task-name slime-021-rollout \
  --rollout-root /mnt/zj-data/.../output/slime/<exp>/rollout_dumps \
  --recursive
rollout-insight tasks create \
  --task-name slime-021-rollout \
  --rollout-root /mnt/zj-data/.../output/slime/<exp>/rollout_dumps \
  --recursive --submit

# 5) 等 service 把扫描完，列任务 / 检视 segment
rollout-insight --format table tasks list --task-name slime-021-rollout
rollout-insight --format table segments list --task-id 1
```

## 场景 2：从南湖 job 找到「重跑」对应 rollout

```bash
# 1) 列出用户的所有运行中训练
rollout-insight --cluster wuxi --format table nanhu jobs --job-type AI_TRAIN

# 2) 取 job id 后看 detail（找出 OUTPUT_DIR）
rollout-insight --cluster wuxi nanhu jobs detail --job-id jb-xxx | head -c 4k

# 3) 把南湖 job 绑到本地 task
rollout-insight nanhu link --job-id jb-xxx --task-name slime-021-rollout
rollout-insight nanhu link --job-id jb-xxx --task-name slime-021-rollout --submit

# 4) 如任务失败：重启南湖 job
rollout-insight --cluster wuxi nanhu restart --job-id jb-xxx
rollout-insight --cluster wuxi nanhu restart --job-id jb-xxx --submit
```

## 场景 3：定位一条失败 segment 的 tool_calls

```bash
# 1) 找低 reward 的 segment
rollout-insight --format table segments search --task-id 1 --min-reward 0 --page-size 100

# 2) 按 sandbox_id 模糊匹配（agent 死循环定位常用）
rollout-insight segments list --task-id 1 --keyword sb-abc123

# 3) 查看 transcript
rollout-insight segments show --segment-id 42
```

## 场景 4：批量重扫

```bash
# 列出所有 RUNNING / FAILED 任务，挨个重扫
for tid in $(rollout-insight tasks list --status FAILED | python -c "
import json, sys
d = json.load(sys.stdin)
print(' '.join(str(t['id']) for t in d['data']['tasks']))
"); do
    echo "重扫 task_id=$tid"
    rollout-insight tasks rescan --task-id "$tid" --submit
done
```

## 场景 5：清理废任务

```bash
# 先 dry-run 看目标
rollout-insight tasks delete --task-id 99

# 确认 → submit（级联删除其 rollout/attempt/segment）
rollout-insight tasks delete --task-id 99 --submit
```

## 输出格式差异

- **`--format json`（默认）**：紧凑 JSON，适合 agent 解析。
- **`--format table`**：把 list-of-dict 渲染成 markdown 表，每列限宽 48 字符。
  单 dict 用「(unprintable)」占位，所以非列表场景（如 segments show）table 退化为 JSON。

## 排障

| 症状 | 原因 | 行动 |
|------|------|------|
| `无法连接 http://localhost:39998/...` | Web service 没起 | `python main.py` 后重试 |
| `找不到 nanhu / nanhu-cli 可执行文件` | nanhu-skill 未装 | `pip install -e /path/to/nanhu-skill` |
| `nanhu jobs list: error: --job-type required` | nanhu-skill 的 list 要求 job_type，或**不**走 list 改用 `running` | 留空 `--job-type`，自动走 `running` |
| `segments search: INVALID_ARGUMENT 当前阶段 2C 仅支持 task_id` | controller 限制 | 加 `--task-id <n>` |
| `--format table` 输出 `(empty)` | payload 不是 list-of-dict | 改 `--format json` 看数据结构 |
| `tokenizer test` 报 `未找到任何 tokenizer 关键文件` | 路径里没有 tokenizer.json 等 | 确认是 HF tokenizer 目录（`tokenizer.json` / `vocab.json` 等） |