"""
统一的响应封装类（仿 Shell-dsv2/automate/data/response.py）。
"""

from typing import Any, Optional
from pydantic import BaseModel


class Response(BaseModel):
    """
    统一的 JSON 返回格式。

    示例：
        Response.success("查询成功", data={"id": 1})
        Response.error("操作失败", code=51001)
    """

    code: int
    msg: str
    data: Optional[Any] = None

    @classmethod
    def success(cls, msg: str = "success", code: int = 0, data: Any = None):
        return cls(code=code, msg=msg, data=data)

    @classmethod
    def error(cls, msg: str = "error", code: int = -1, data: Any = None):
        return cls(code=code, msg=msg, data=data)

    def is_successful(self) -> bool:
        return self.code == 0
