"""rollout-insight 命令行工具包。

- ``cli.rollout_insight``    argparse 入口（`rollout-insight` 可执行文件）
- ``cli.api_client``         本地 Web API HTTP 客户端
- ``cli.nanhu_bridge``       南湖（nanhu-cli）subprocess 桥接
- ``cli.tokenizer_cmd``      config.json / tokenizer 路径处理
- ``cli.formatters``         table/json 渲染辅助

写操作默认 dry-run；加 ``--submit`` 才执行。
"""

# 注意：避免在 package import 时 eager-import rollout_insight.main，
# 否则 ``python -m cli.rollout_insight`` 会因为 package 已 import 过而
# 出现 RuntimeWarning。让入口直接调用 cli.rollout_insight.main 即可。

__all__ = ["api_client", "formatters", "nanhu_bridge", "rollout_insight", "tokenizer_cmd"]