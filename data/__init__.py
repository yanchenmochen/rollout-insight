"""rollout-insight 数据层。

- response: Pydantic 统一的 JSON 响应封装（与 automate/data/response.py 一致）
- rollout_insight_exception: 业务异常类，错误码段：
  - 50xxx 通用
  - 51xxx 任务
  - 52xxx 解析
  - 53xxx 南湖
  - 54xxx 系统
"""
from data.response import Response
from data.rollout_insight_exception import RolloutInsightException

__all__ = ["Response", "RolloutInsightException"]
