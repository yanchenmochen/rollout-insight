"""service 包：业务逻辑层。

v0.2 阶段落地：
- rollout_scan_service：扫描 .pt dump 文件并入库
"""
from service.rollout_scan_service import RolloutScanService

__all__ = ["RolloutScanService"]