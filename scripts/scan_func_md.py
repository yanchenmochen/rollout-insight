#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
scan_func_md.py
================

扫描 ``controller/``、``service/``、``db/``、``utils/``、``data/`` 五个目录，
把"对外公开能力"汇总到一份 ``func.md`` 索引。

仿 Shell-dsv2/automate/func.md 的写作风格：每个目录一节、用要点列表 / 表格
呈现。**全量覆盖**（不增量追加），保证在 ``--check`` 模式下能字节级对比。

扫描口径（与 plan v0.5.2 对齐）
------------------------------

- **controller/\*.py** —— ``@router.get/post/put/delete`` 装饰器下的 path /
  function name / docstring 首行（其它 ``@app.*`` 也识别为路由）。
- **service/\*.py** 与 **db/db_\*\_service.py** —— 公开 class（不以下划线开头）
  名字 + 公开 ``@staticmethod``（或 ``@classmethod``）方法签名 + 所在文件名。
- **db/db_\*.py** —— ORM class 名 / ``__tablename__`` / 文件名（不含
  ``db_element`` / ``__init__``）。
- **utils/\*.py** —— 模块级公开函数签名（不以下划线开头）。
- **data/\*.py** —— public class / Exception 类 + 错误码常量三元组
  ``(code, name, message)``。

黑名单（plan p18_a 决定）
------------------------

- 跳过目录：``__pycache__/ log/ tests/ .git/ *.pt *.egg-info schedule/
  schedule_* static/ skill/``。
- 跳过根目录文件：``CHANGELOG.md / func.md / scripts/scan_func_md.py`` 自身。

CLI
---

- ``python scripts/scan_func_md.py`` —— 渲染并**覆盖** ``func.md``。
- ``python scripts/scan_func_md.py --check`` —— 与现有 ``func.md`` 不一致则
  exit code 1（CI 防漂移）。输出一致则 exit code 0。

退出码
------

- ``0`` 一致 / 正常生成。
- ``1`` ``--check`` 不一致；或任何 IO 错误。
- ``2`` 命令行参数错误。
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 路径与黑名单
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()

# plan p18_a 黑名单（目录 basename / glob 模式）
SKIP_DIR_NAMES: Tuple[str, ...] = (
    "__pycache__",
    "log",
    "tests",
    ".git",
    "schedule",        # plan 原话: schedule/
    "static",
    "skill",
    ".pytest_cache",
    ".cursor",
    "templates",
)

SKIP_DIR_PATTERNS: Tuple[str, ...] = (
    "*.egg-info",
    "schedule_*",      # plan 原话: schedule_*
)

# 根目录里要跳过的文件 basename
SKIP_ROOT_FILES: Tuple[str, ...] = (
    "CHANGELOG.md",
    "func.md",
    "scripts/scan_func_md.py",  # 自身（走相对路径判断）
)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class RouteInfo:
    method: str           # GET / POST / PUT / DELETE ...
    path: str
    func_name: str
    doc_first_line: str


@dataclass
class OrmInfo:
    class_name: str
    tablename: str
    file: str            # 相对仓库根


@dataclass
class ClassMethodInfo:
    class_name: str
    method_name: str
    signature: str       # 方法签名（不含 self/cls）
    is_static: bool


@dataclass
class ModuleFunctionInfo:
    name: str
    signature: str
    doc_first_line: str


@dataclass
class DataSymbolInfo:
    """data/ 下公开 class 或 Exception + 错误码常量。"""

    name: str
    kind: str            # "class" / "exception" / "error_code"
    detail: str          # 描述或 (code, name, message)


@dataclass
class ScanResult:
    controllers: Dict[str, List[RouteInfo]] = field(default_factory=dict)
    orm: Dict[str, OrmInfo] = field(default_factory=dict)
    service_classes: Dict[str, List[ClassMethodInfo]] = field(default_factory=dict)
    db_service_classes: Dict[str, List[ClassMethodInfo]] = field(default_factory=dict)
    utils_functions: Dict[str, List[ModuleFunctionInfo]] = field(default_factory=dict)
    data_symbols: Dict[str, List[DataSymbolInfo]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    """脚本固定位于 ``<repo_root>/scripts/scan_func_md.py``。"""
    return THIS_FILE.parent.parent


def _first_doc_line(node: ast.AST) -> str:
    """取 ast 节点的 docstring 首行；空返回 ''。"""
    doc = ast.get_docstring(node) or ""
    if not doc:
        return ""
    first = doc.split("\n", 1)[0].strip()
    return first


def _arg_to_str(arg: ast.arg) -> str:
    """ast.arg → 源码风格字符串（含 annotation）。"""
    if arg.annotation is None:
        return arg.arg
    return f"{arg.arg}: {ast.unparse(arg.annotation)}"


def _format_arguments(args: ast.arguments, *, skip_first: bool = False) -> str:
    """把 ast.arguments 渲染成 ``(a, b=1, *args, **kwargs)`` 字符串。"""
    parts: List[str] = []

    pos = list(args.posonlyargs) + list(args.args)
    if skip_first and pos:
        pos = pos[1:]

    defaults = list(args.defaults)
    # 位置参数默认值对齐（ast 的 defaults 跟 args 末尾对齐）
    n_defaults = len(defaults)
    n_pos = len(pos)
    for i, arg in enumerate(pos):
        s = _arg_to_str(arg)
        default_index = i - (n_pos - n_defaults)
        if 0 <= default_index < n_defaults:
            s += f"={ast.unparse(defaults[default_index])}"
        parts.append(s)

    if args.vararg is not None:
        parts.append(f"*{_arg_to_str(args.vararg)}")
    elif args.kwonlyargs:
        parts.append("*")

    for i, arg in enumerate(args.kwonlyargs):
        s = _arg_to_str(arg)
        if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
            s += f"={ast.unparse(args.kw_defaults[i])}"
        parts.append(s)

    if args.kwarg is not None:
        parts.append(f"**{_arg_to_str(args.kwarg)}")

    rendered = ", ".join(parts)
    if rendered:
        return f"({rendered})"
    return "()"


def _method_signature(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """渲染 ``def name(...) -> ReturnType``。"""
    prefix = "async def" if isinstance(func, ast.AsyncFunctionDef) else "def"
    name = func.name
    args = _format_arguments(func.args, skip_first=True)
    ret = ""
    if func.returns is not None:
        ret = f" -> {ast.unparse(func.returns)}"
    return f"{prefix} {name}{args}{ret}"


def _function_signature(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """模块级函数签名（不带 self/cls）。"""
    prefix = "async def" if isinstance(func, ast.AsyncFunctionDef) else "def"
    name = func.name
    args = _format_arguments(func.args)
    ret = ""
    if func.returns is not None:
        ret = f" -> {ast.unparse(func.returns)}"
    return f"{prefix} {name}{args}{ret}"


def _is_public_name(name: str) -> bool:
    return bool(name) and not name.startswith("_")


def _is_route_decorator(dec: ast.expr) -> Optional[str]:
    """识别 ``@router.get/post/put/delete`` / ``@app.get/post/...``。

    返回 HTTP 方法（get/post/...）；非路由返回 None。
    """
    if not isinstance(dec, ast.Call):
        return None
    func = dec.func
    if not isinstance(func, ast.Attribute):
        return None
    method = func.attr.lower()
    if method in {"get", "post", "put", "delete", "patch", "head", "options"}:
        # 排除非 FastAPI 路由装饰器（粗判：func.value 必须是 Name）
        # 例：@app.get、@router.post 都接受；@some_func.get 视作非路由
        if isinstance(func.value, ast.Name):
            return method.upper()
    return None


def _route_path(dec: ast.Call) -> str:
    """取装饰器第一个位置参数作为 path。"""
    if not dec.args:
        return ""
    first = dec.args[0]
    try:
        return ast.literal_eval(first)
    except Exception:
        return ast.unparse(first)


def _parse_source(path: Path) -> Optional[ast.Module]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError:
        return None


# ---------------------------------------------------------------------------
# 目录枚举 & 黑名单判定
# ---------------------------------------------------------------------------

def _is_skipped_dir(dirname: str) -> bool:
    if dirname in SKIP_DIR_NAMES:
        return True
    for pat in SKIP_DIR_PATTERNS:
        if fnmatch.fnmatch(dirname, pat):
            return True
    return False


def _iter_python_files(root: Path, subdirs: Sequence[str]) -> Iterable[Path]:
    """生成 ``root/<sub>/*.py`` 的待扫描 python 文件（应用黑名单）。

    subdirs: 顶层目录名列表，如 ``('controller', 'service')``。
    """
    for sub in subdirs:
        base = root / sub
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            # 检查路径中是否含被黑名单命名的目录
            rel = path.relative_to(root)
            skip = False
            for part in rel.parts[:-1]:
                if _is_skipped_dir(part):
                    skip = True
                    break
            if not skip and path.name != "__pycache__":
                # 再次防御：文件名黑名单
                if path.name in {"__init__.py"}:
                    # __init__.py 我们也要扫，只是里面的内容比较空
                    pass
                yield path


def _iter_db_files(root: Path) -> Iterable[Path]:
    """db/db_<领域>.py 与 db/db_<领域>_service.py。"""
    db_root = root / "db"
    if not db_root.is_dir():
        return
    for path in sorted(db_root.glob("db_*.py")):
        if path.name.startswith("__"):
            continue
        yield path


# ---------------------------------------------------------------------------
# 扫描各目录
# ---------------------------------------------------------------------------

def scan_controller(root: Path, files: Iterable[Path]) -> Dict[str, List[RouteInfo]]:
    out: Dict[str, List[RouteInfo]] = {}
    for path in files:
        module = _parse_source(path)
        if module is None:
            continue
        rel = path.relative_to(root).as_posix()
        routes: List[RouteInfo] = []
        for node in ast.walk(module):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _is_public_name(node.name):
                continue
            for dec in node.decorator_list:
                method = _is_route_decorator(dec)
                if method is not None and isinstance(dec, ast.Call):
                    routes.append(
                        RouteInfo(
                            method=method,
                            path=_route_path(dec),
                            func_name=node.name,
                            doc_first_line=_first_doc_line(node),
                        )
                    )
        if routes:
            out[rel] = routes
    return out


def _scan_db_classes_from_module(
    module: ast.Module, *, want_orm: bool
) -> Tuple[List[OrmInfo], List[ClassMethodInfo]]:
    """从一个 db 文件里抽 ORM class 名 + Service class 的 @staticmethod。

    want_orm=True 时把继承 Base 的类当作 ORM；want_orm=False（service 文件）
    时把所有公开类当作 Service class。
    """
    orms: List[OrmInfo] = []
    services: List[ClassMethodInfo] = []

    for node in module.body:
        if isinstance(node, ast.ClassDef) and _is_public_name(node.name):
            # ORM 判定：继承 Base（db_element.Base）；不递归基类
            is_orm = False
            if want_orm:
                for base in node.bases:
                    base_name = ast.unparse(base)
                    if base_name == "Base":
                        is_orm = True
                        break
            if is_orm:
                tablename = ""
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for tgt in item.targets:
                            if (
                                isinstance(tgt, ast.Name)
                                and tgt.id == "__tablename__"
                                and isinstance(item.value, ast.Constant)
                                and isinstance(item.value.value, str)
                            ):
                                tablename = item.value.value
                                break
                orms.append(
                    OrmInfo(
                        class_name=node.name,
                        tablename=tablename,
                        file="",  # 调用方补充
                    )
                )
            else:
                # Service class：抽 @staticmethod / @classmethod 方法签名
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and _is_public_name(item.name):
                        is_static = False
                        is_classmethod = False
                        for dec in item.decorator_list:
                            if isinstance(dec, ast.Name) and dec.id == "staticmethod":
                                is_static = True
                            elif isinstance(dec, ast.Name) and dec.id == "classmethod":
                                is_classmethod = True
                        # Service 风格：只登记 staticmethod（plan 表述）
                        # 但 classmethod 也算公开方法，补一份以避免漏掉
                        if is_static or is_classmethod:
                            kind = "staticmethod" if is_static else "classmethod"
                            info = ClassMethodInfo(
                                class_name=node.name,
                                method_name=item.name,
                                signature=_method_signature(item),
                                is_static=is_static,
                            )
                            # 额外带个 _decorator_kind（属性由渲染层判断）
                            setattr(info, "_decorator_kind", kind)
                            services.append(info)
    return orms, services


def scan_db(root: Path) -> Tuple[Dict[str, OrmInfo], Dict[str, List[ClassMethodInfo]]]:
    orms: Dict[str, OrmInfo] = {}
    services: Dict[str, List[ClassMethodInfo]] = {}
    for path in _iter_db_files(root):
        module = _parse_source(path)
        if module is None:
            continue
        rel = path.relative_to(root).as_posix()
        is_service_file = path.name.endswith("_service.py")
        orm_list, svc_list = _scan_db_classes_from_module(
            module, want_orm=not is_service_file
        )
        for o in orm_list:
            o.file = rel
            orms[o.class_name] = o
        if svc_list:
            services[rel] = svc_list
    return orms, services


def scan_service(root: Path, files: Iterable[Path]) -> Dict[str, List[ClassMethodInfo]]:
    """service/ 下公开 class 的全部公开方法（不限 static）。

    与 plan 表述"公开 @staticmethod 方法签名"略有差异：本仓库 service 风格混合
    （`RolloutParseService.parse_rollout` 是实例方法；`RolloutScanService` 全
    ``@staticmethod``），统一收录所有公开方法（含实例方法）以保证
    ``RolloutParseService`` 也出现在索引里。

    同时把"模块级不带下划线的函数"也合并到一个虚拟类 ``<module>_module`` 里，
    这样 v0.5.0 T2 之后部分 service 把入口函数写在 module 而不是 class 的情况
    （如 ``service/agent_transcript_analyzer.load_transcript``）也能被索引。
    """
    out: Dict[str, List[ClassMethodInfo]] = {}
    for path in files:
        module = _parse_source(path)
        if module is None:
            continue
        rel = path.relative_to(root).as_posix()
        methods: List[ClassMethodInfo] = []
        module_basename = path.stem  # e.g. "rollout_parse_service"
        for node in module.body:
            if isinstance(node, ast.ClassDef) and _is_public_name(node.name):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and _is_public_name(item.name):
                        is_static = any(
                            isinstance(dec, ast.Name) and dec.id == "staticmethod"
                            for dec in item.decorator_list
                        )
                        is_classmethod = any(
                            isinstance(dec, ast.Name) and dec.id == "classmethod"
                            for dec in item.decorator_list
                        )
                        # 跳过 dunder；其余一律收录
                        if item.name.startswith("__") and item.name.endswith("__"):
                            continue
                        methods.append(
                            ClassMethodInfo(
                                class_name=node.name,
                                method_name=item.name,
                                signature=_method_signature(item),
                                is_static=is_static or is_classmethod,
                            )
                        )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public_name(node.name):
                # 模块级公开函数：登记成虚拟类的"static"方法
                methods.append(
                    ClassMethodInfo(
                        class_name=f"{module_basename}_module",
                        method_name=node.name,
                        signature=_function_signature(node),
                        is_static=True,
                    )
                )
        if methods:
            out[rel] = methods
    return out


def scan_utils(root: Path, files: Iterable[Path]) -> Dict[str, List[ModuleFunctionInfo]]:
    out: Dict[str, List[ModuleFunctionInfo]] = {}
    for path in files:
        module = _parse_source(path)
        if module is None:
            continue
        rel = path.relative_to(root).as_posix()
        funcs: List[ModuleFunctionInfo] = []
        for node in module.body:
            # 模块级公开函数（不在 ClassDef 里）
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public_name(node.name):
                funcs.append(
                    ModuleFunctionInfo(
                        name=node.name,
                        signature=_function_signature(node),
                        doc_first_line=_first_doc_line(node),
                    )
                )
        if funcs:
            out[rel] = funcs
    return out


def _classify_data_class(node: ast.ClassDef) -> str:
    """data/ 下 class 分类：exception / class。"""
    for base in node.bases:
        base_name = ast.unparse(base)
        # Exception / 自定义业务异常基类
        if "Exception" in base_name or "Error" in base_name.split(".")[-1]:
            return "exception"
    return "class"


def _extract_error_codes(module: ast.Module, exception_name: str) -> List[Tuple[str, str, str, str]]:
    """从给定 Exception 类里抽 ``(常量名, code, name, message)``。

    识别条件：类属性赋值为长度为 3 的 tuple，且第一项是 int。
    """
    codes: List[Tuple[str, str, str, str]] = []
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == exception_name:
            for item in node.body:
                if isinstance(item, ast.Assign) and len(item.targets) == 1:
                    tgt = item.targets[0]
                    if not isinstance(tgt, ast.Name):
                        continue
                    # value 必须是 tuple(list-of-constants)
                    if isinstance(item.value, ast.Tuple) and len(item.value.elts) == 3:
                        elts = item.value.elts
                        if all(
                            isinstance(e, ast.Constant) for e in elts
                        ):
                            code_val = elts[0].value
                            if isinstance(code_val, int):
                                codes.append(
                                    (
                                        tgt.id,
                                        str(code_val),
                                        str(elts[1].value),
                                        str(elts[2].value),
                                    )
                                )
    return codes


def scan_data(root: Path, files: Iterable[Path]) -> Dict[str, List[DataSymbolInfo]]:
    out: Dict[str, List[DataSymbolInfo]] = {}
    for path in files:
        module = _parse_source(path)
        if module is None:
            continue
        rel = path.relative_to(root).as_posix()
        symbols: List[DataSymbolInfo] = []

        # 顶层公开 class
        for node in module.body:
            if isinstance(node, ast.ClassDef) and _is_public_name(node.name):
                kind = _classify_data_class(node)
                base_desc = ""
                if node.bases:
                    base_desc = ast.unparse(node.bases[0])
                symbols.append(
                    DataSymbolInfo(
                        name=node.name,
                        kind=kind,
                        detail=(
                            f"继承 {base_desc}" if base_desc else ""
                        ) + (
                            f"；{_first_doc_line(node)}" if _first_doc_line(node) else ""
                        ),
                    )
                )
                # 若识别为 Exception，额外抽错误码常量
                if kind == "exception":
                    codes = _extract_error_codes(module, node.name)
                    for const_name, code, name, message in codes:
                        symbols.append(
                            DataSymbolInfo(
                                name=f"{node.name}.{const_name}",
                                kind="error_code",
                                detail=f"({code}, \"{name}\", \"{message}\")",
                            )
                        )

        # 顶层模块级"非 ClassDef"公开变量（罕见，主要给 data_eval_host 等用）
        for node in module.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                tgt = node.targets[0]
                if isinstance(tgt, ast.Name) and _is_public_name(tgt.id):
                    # 仅登记 module-level 公开名字（不重复类内部已经登记的）
                    name = tgt.id
                    if any(s.name == name for s in symbols):
                        continue
                    symbols.append(
                        DataSymbolInfo(
                            name=name,
                            kind="module",
                            detail=ast.unparse(node.value)[:120],
                        )
                    )

        if symbols:
            out[rel] = symbols
    return out


# ---------------------------------------------------------------------------
# 渲染 markdown
# ---------------------------------------------------------------------------

def _rel_link(rel_path: str) -> str:
    """把相对仓库根的路径包成行内代码。"""
    return f"`{rel_path}`"


def _short_signature(sig: str, max_len: int = 96) -> str:
    """防止太长的签名撑爆表格。"""
    if len(sig) <= max_len:
        return sig
    return sig[: max_len - 1] + "…"


def render_markdown(
    result: ScanResult,
    *,
    version: str = "v0.5.2",
    repo_basename: str = "rollout-insight",
) -> str:
    """全量渲染成 markdown 字符串。"""
    lines: List[str] = []
    lines.append(f"# {repo_basename} 能力索引 (func.md)")
    lines.append("")
    lines.append(
        "> **用途**：重构或新增业务代码前，必须先查阅本文档，确认是否已有可复用实现。  "
    )
    lines.append(
        f"> **维护**：新增/修改 `controller`、`service`、`db/*_service`、`data`、`utils` "
        "中的公开能力后，必须重跑 `python scripts/scan_func_md.py` 重新生成。  "
    )
    lines.append(
        f"> **版本**：与 {version} 同步生成；CI 阶段跑 `python scripts/scan_func_md.py --check` 校验一致性。"
    )
    lines.append("")

    # ------- 查阅顺序 -------
    lines.append("## 查阅顺序（自上而下）")
    lines.append("")
    lines.append("| 顺序 | 目录 | 何时查 |")
    lines.append("| --- | --- | --- |")
    lines.append("| 1 | `data/` | 枚举、DTO、异常码、响应封装 |")
    lines.append("| 2 | `utils/` | 配置、路径、日志、解析、HTTP 等无状态工具 |")
    lines.append("| 3 | `db/db_*_service.py` | 表读写、状态流转、查询组合 |")
    lines.append("| 4 | `service/` | 跨表业务流程、外部平台对接 |")
    lines.append("| 5 | `db/db_*.py` | ORM 模型与字段 |")
    lines.append("| 6 | `controller/` | 已有 REST 路由，避免重复端点 |")
    lines.append("")

    # ------- data -------
    lines.append("## data/ — 枚举、DTO、异常")
    lines.append("")
    if not result.data_symbols:
        lines.append("（未发现公开符号）")
        lines.append("")
    else:
        for rel in sorted(result.data_symbols):
            lines.append(f"### {_rel_link(rel)}")
            lines.append("")
            symbols = result.data_symbols[rel]
            # 按类别归档
            classes = [s for s in symbols if s.kind in {"class", "exception"}]
            codes = [s for s in symbols if s.kind == "error_code"]
            modules = [s for s in symbols if s.kind == "module"]
            if classes:
                for s in classes:
                    desc = s.detail or "（无 docstring）"
                    lines.append(f"- **{s.name}** — {desc}")
            if modules:
                lines.append("")
                for s in modules:
                    lines.append(f"- **{s.name}** = `{s.detail}`")
            if codes:
                lines.append("")
                lines.append("  错误码常量：")
                lines.append("")
                lines.append("  | 常量 | 错误码 | 名称 | 默认消息 |")
                lines.append("  | --- | --- | --- | --- |")
                for s in codes:
                    m = _parse_code_tuple(s.detail)
                    if m is None:
                        continue
                    code, name, message = m
                    lines.append(
                        f"  | `{s.name}` | {code} | `{name}` | {message} |"
                    )
            lines.append("")

    # ------- utils -------
    lines.append("## utils/ — 通用工具（优先复用）")
    lines.append("")
    if not result.utils_functions:
        lines.append("（未发现公开函数）")
        lines.append("")
    else:
        for rel in sorted(result.utils_functions):
            lines.append(f"### {_rel_link(rel)}")
            lines.append("")
            for fn in result.utils_functions[rel]:
                lines.append(
                    f"- `{_short_signature(fn.signature)}` — {fn.doc_first_line or '（无 docstring）'}"
                )
            lines.append("")

    # ------- db ORM -------
    lines.append("## db/ — ORM 模型")
    lines.append("")
    if not result.orm:
        lines.append("（未发现 ORM 模型）")
        lines.append("")
    else:
        lines.append("| ORM 类 | 文件 | 物理表 |")
        lines.append("| --- | --- | --- |")
        for name in sorted(result.orm):
            o = result.orm[name]
            lines.append(
                f"| `{o.class_name}` | {_rel_link(o.file)} | `{o.tablename or '?'}` |"
            )
        lines.append("")

    # ------- db services -------
    lines.append("## db/ — 数据访问服务")
    lines.append("")
    if not result.db_service_classes:
        lines.append("（未发现数据访问 service）")
        lines.append("")
    else:
        for rel in sorted(result.db_service_classes):
            methods = result.db_service_classes[rel]
            # 按 class 分组
            by_class: Dict[str, List[ClassMethodInfo]] = {}
            for m in methods:
                by_class.setdefault(m.class_name, []).append(m)
            for class_name in sorted(by_class):
                lines.append(f"#### `{class_name}` ({_rel_link(rel)})")
                lines.append("")
                for m in by_class[class_name]:
                    # 区分装饰器类型：
                    # - db/*_service.py 只收录 staticmethod / classmethod（plan）
                    # - service/*.py 收录所有公开方法，is_static=False 即实例方法
                    if hasattr(m, "_decorator_kind"):
                        kind = m._decorator_kind
                    elif m.is_static:
                        kind = "staticmethod"
                    else:
                        kind = "method"
                    lines.append(f"- `{kind}` `{_short_signature(m.signature)}`")
                lines.append("")

    # ------- service -------
    lines.append("## service/ — 业务流程")
    lines.append("")
    if not result.service_classes:
        lines.append("（未发现 service）")
        lines.append("")
    else:
        for rel in sorted(result.service_classes):
            methods = result.service_classes[rel]
            # 按 class 分组
            by_class: Dict[str, List[ClassMethodInfo]] = {}
            for m in methods:
                by_class.setdefault(m.class_name, []).append(m)
            for class_name in sorted(by_class):
                lines.append(f"### `{class_name}` ({_rel_link(rel)})")
                lines.append("")
                if class_name.endswith("_module"):
                    lines.append("模块级公开函数（不在 class 内）：")
                    lines.append("")
                for m in by_class[class_name]:
                    # 区分装饰器类型：
                    # - db/*_service.py 只收录 staticmethod / classmethod（plan）
                    # - service/*.py 收录所有公开方法，is_static=False 即实例方法
                    if hasattr(m, "_decorator_kind"):
                        kind = m._decorator_kind
                    elif m.is_static:
                        kind = "staticmethod"
                    else:
                        kind = "method"
                    lines.append(f"- `{kind}` `{_short_signature(m.signature)}`")
                lines.append("")

    # ------- controller -------
    lines.append("## controller/ — REST 路由（新增接口前对照）")
    lines.append("")
    if not result.controllers:
        lines.append("（未发现路由）")
        lines.append("")
    else:
        lines.append("| 模块 | Method | Path | Function | 用途 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for rel in sorted(result.controllers):
            module_basename = Path(rel).stem
            for route in result.controllers[rel]:
                lines.append(
                    f"| `{module_basename}` | {route.method} | `{route.path}` | "
                    f"`{route.func_name}` | {route.doc_first_line or '—'} |"
                )
        lines.append("")

    # ------- 重构检查清单 -------
    lines.append("## 重构检查清单")
    lines.append("")
    lines.append("开始改动前确认：")
    lines.append("")
    lines.append("- [ ] 已阅读本文档相关章节（及 `.cursor/rules/rollout-insight-conventions.mdc`）")
    lines.append("- [ ] 已 Grep 关键词（如 `parse_rollout`、`DBTaskService`）避免漏掉复用点")
    lines.append("- [ ] 数据库变更只通过对应 `DBXxxService`，未在 controller/service 写裸 SQL")
    lines.append("- [ ] 新 REST 接口使用 `data.response.Response`，业务错误用 `RolloutInsightException`")
    lines.append("- [ ] 完成后已重跑 `python scripts/scan_func_md.py` 并跟随本次改动一并 commit")
    lines.append("")
    lines.append(f"*文档版本：{version}；自动生成，请勿手改。*")
    lines.append("")
    return "\n".join(lines)


def _parse_code_tuple(detail: str) -> Optional[Tuple[str, str, str]]:
    """解析 ``(50000, "NAME", "message")``。"""
    detail = detail.strip()
    if not (detail.startswith("(") and detail.endswith(")")):
        return None
    body = detail[1:-1]
    parts: List[str] = []
    cur: List[str] = []
    in_str = False
    quote = ""
    for ch in body:
        if in_str:
            if ch == quote and (not cur or cur[-1] != "\\"):
                in_str = False
                cur.append(ch)
                continue
            cur.append(ch)
        else:
            if ch in ("'", '"'):
                in_str = True
                quote = ch
                cur.append(ch)
            elif ch == ",":
                parts.append("".join(cur).strip())
                cur = []
            else:
                cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    if len(parts) != 3:
        return None
    return parts[0], parts[1].strip("'\""), parts[2].strip("'\"")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def scan(root: Path) -> ScanResult:
    controller_files = list(_iter_python_files(root, ("controller",)))
    service_files = list(_iter_python_files(root, ("service",)))
    utils_files = list(_iter_python_files(root, ("utils",)))
    data_files = list(_iter_python_files(root, ("data",)))

    result = ScanResult()
    result.controllers = scan_controller(root, controller_files)
    orm_map, db_service = scan_db(root)
    result.orm = orm_map
    result.db_service_classes = db_service
    result.service_classes = scan_service(root, service_files)
    result.utils_functions = scan_utils(root, utils_files)
    result.data_symbols = scan_data(root, data_files)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scan_func_md",
        description="扫描 controller/service/db/utils/data 公开能力 → func.md",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只校验：现有 func.md 与本次扫描结果是否一致；不一致返回 exit code 1",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="func.md",
        help="输出的 markdown 文件名（相对仓库根，默认 func.md）",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v0.5.2",
        help="在 markdown 顶部注明的项目版本（默认 v0.5.2）",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    out_path = root / args.output

    try:
        result = scan(root)
        content = render_markdown(
            result, version=args.version, repo_basename=root.name
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[scan_func_md] 扫描/渲染失败: {exc}", file=sys.stderr)
        return 1

    if args.check:
        if not out_path.is_file():
            print(
                f"[scan_func_md] --check 失败：{out_path} 不存在，先跑一次生成。",
                file=sys.stderr,
            )
            return 1
        existing = out_path.read_text(encoding="utf-8")
        if existing != content:
            print(
                f"[scan_func_md] --check 失败：{out_path} 与扫描结果不一致。\n"
                "  修复方式：跑 `python scripts/scan_func_md.py` 重新生成。",
                file=sys.stderr,
            )
            return 1
        print(f"[scan_func_md] --check OK：{out_path} 与代码一致。")
        return 0

    # 默认：覆盖写入
    out_path.write_text(content, encoding="utf-8")
    print(f"[scan_func_md] 已写入 {out_path}（{len(content)} 字节，{args.version}）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
