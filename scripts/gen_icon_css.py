#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""从 ``static/icons/*.svg`` 生成 ``static/ri_icons.css``。

图标用 CSS mask 引入而不是 ``<img>``：mask 会用元素自身的 ``background-color``
（``currentColor``）给 SVG 描边重新上色，因此同一枚图标能在深色顶栏与浅色卡片里
都保持清晰，而不用为每个背景各存一份。

用法::

    python scripts/gen_icon_css.py          # 生成 / 覆盖 static/ri_icons.css
    python scripts/gen_icon_css.py --check  # 只校验是否与图标目录一致（CI 用）

之所以把 SVG vendor 进仓库而不是走 CDN：部署环境为离线内网，CDN 不可达。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
ICON_DIR = REPO_ROOT / "static" / "icons"
CSS_PATH = REPO_ROOT / "static" / "ri_icons.css"

HEADER = """/* 自动生成，请勿手改 —— 由 python scripts/gen_icon_css.py 生成。
 *
 * 图标来源：Lucide（MIT License），见 static/icons/LICENSE。
 * 用法：<span class="ri-icon ri-icon-list-checks"></span>
 * 尺寸由 .ri-icon 的 width/height 控制，颜色跟随 currentColor。
 */
"""

BASE_RULE = """
.ri-icon {
    display: inline-block;
    width: 15px;
    height: 15px;
    flex: 0 0 auto;
    background-color: currentColor;
    -webkit-mask-repeat: no-repeat;
    mask-repeat: no-repeat;
    -webkit-mask-position: center;
    mask-position: center;
    -webkit-mask-size: contain;
    mask-size: contain;
    vertical-align: -2px;
}
"""


def icon_names() -> List[str]:
    """按文件名排序列出图标名（不含 .svg）。"""
    if not ICON_DIR.is_dir():
        return []
    return sorted(p.stem for p in ICON_DIR.glob("*.svg"))


def render_css(names: List[str]) -> str:
    parts = [HEADER, BASE_RULE]
    for name in names:
        parts.append(
            '.ri-icon-%s {\n'
            '    -webkit-mask-image: url("/static/icons/%s.svg");\n'
            '    mask-image: url("/static/icons/%s.svg");\n'
            '}\n' % (name, name, name)
        )
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成图标 CSS")
    parser.add_argument("--check", action="store_true", help="只校验是否最新")
    args = parser.parse_args()

    names = icon_names()
    if not names:
        print("[gen_icon_css] 错误：static/icons/ 下没有图标", file=sys.stderr)
        return 1

    content = render_css(names)

    if args.check:
        current = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.is_file() else ""
        if current != content:
            print("[gen_icon_css] 错误：ri_icons.css 与图标目录不一致，请重跑生成", file=sys.stderr)
            return 1
        print("[gen_icon_css] --check OK：%d 个图标，ri_icons.css 是最新的。" % len(names))
        return 0

    CSS_PATH.write_text(content, encoding="utf-8")
    print("[gen_icon_css] 已写入 %s（%d 个图标）。" % (CSS_PATH, len(names)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
