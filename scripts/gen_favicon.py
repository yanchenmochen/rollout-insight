#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""生成 rollout-insight 的浏览器站点图标（favicon）。

输入
----
``static/branding/ri_icon_source.png`` —— 原始 1024×1024 设计稿（Gemini 生成）。
该文件是**唯一真源**，本脚本从它派生出浏览器需要的全部尺寸。

输出
----
``static/favicon.ico``          —— 多尺寸 ICO（16/32/48）
``static/favicon-16x16.png``    —— 标签页小图标
``static/favicon-32x32.png``    —— 标签页 / 书签
``static/favicon-48x48.png``    —— 书签栏 / 高分屏
``static/apple-touch-icon.png`` —— 180×180（iOS 添加到主屏）
``static/icon-192.png`` / ``static/icon-512.png`` —— PWA / manifest

处理要点
--------
- **裁切**：设计稿四周有一圈氛围辉光，中间的圆角深色 tile 才是主体。
  脚本按固定比例裁出 tile，避免每次手工调参。
- **圆角遮罩**：tile 本身带圆角，再叠一层轻微圆角遮罩让图标在浏览器
  标签页（浅灰 / 深色两种背景）里边缘干净、不糊。
- **小尺寸增强**：16px 下细节会糊成一团，因此统一做
  饱和度 / 对比度 / 亮度微调 + UnsharpMask 锐化，保证标签页仍然可辨识。

用法::

    python scripts/gen_favicon.py           # 生成 / 覆盖全部图标
    python scripts/gen_favicon.py --check   # 只校验是否与源图一致（CI 用）

依赖：Pillow（``PIL``）。若环境未安装 Pillow，会给出明确提示并退出码 1。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
except ImportError:  # pragma: no cover - 环境缺依赖时的兜底
    print(
        "[gen_favicon] 错误：需要 Pillow 才能生成图标。\n"
        "  安装：pip install Pillow",
        file=sys.stderr,
    )
    raise SystemExit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "static"
BRANDING_DIR = STATIC_DIR / "branding"
SOURCE_PNG = BRANDING_DIR / "ri_icon_source.png"

# 设计稿 1024×1024 中，圆角 tile 的实测边界（含 1px 描边余量）。
TILE_BOX: Tuple[int, int, int, int] = (71, 72, 953, 954)

# 圆角遮罩半径占边长的比例。
CORNER_RATIO = 0.115

# 单一尺寸下的增强参数（饱和度 / 对比度 / 亮度）。
SATURATION = 1.22
CONTRAST = 1.14
BRIGHTNESS = 1.04

# 输出清单：文件名 -> 目标边长。ICO 单独处理（多尺寸打包）。
PNG_TARGETS: Dict[str, int] = {
    "favicon-16x16.png": 16,
    "favicon-32x32.png": 32,
    "favicon-48x48.png": 48,
    "apple-touch-icon.png": 180,
    "icon-192.png": 192,
    "icon-512.png": 512,
}

ICO_PATH = STATIC_DIR / "favicon.ico"
ICO_SIZES: Tuple[int, ...] = (16, 32, 48)


def _rounded_mask(size: int) -> "Image.Image":
    """生成 size×size 的圆角矩形遮罩（L 模式）。"""
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=int(size * CORNER_RATIO), fill=255
    )
    return mask


def _load_tile() -> "Image.Image":
    """从源图裁出圆角 tile。"""
    if not SOURCE_PNG.is_file():
        raise FileNotFoundError(
            "[gen_favicon] 缺少源图 %s；该文件是图标的唯一真源。" % SOURCE_PNG
        )
    src = Image.open(SOURCE_PNG).convert("RGB")
    if src.size != (1024, 1024):
        print(
            "[gen_favicon] 警告：源图尺寸为 %s（预期 1024×1024），"
            "裁切框按比例自适应。" % (src.size,),
            file=sys.stderr,
        )
        w, h = src.size
        scale = min(w, h) / 1024.0
        box = tuple(int(v * scale) for v in TILE_BOX)
        return src.crop(box)
    return src.crop(TILE_BOX)


def render_icon(tile: "Image.Image", size: int) -> "Image.Image":
    """把 tile 渲染成指定尺寸的 RGBA 图标（含增强 + 圆角遮罩）。"""
    img = tile.resize((size, size), Image.LANCZOS)
    img = ImageEnhance.Color(img).enhance(SATURATION)
    img = ImageEnhance.Contrast(img).enhance(CONTRAST)
    img = ImageEnhance.Brightness(img).enhance(BRIGHTNESS)
    radius = max(0.6, size / 48.0)
    img = img.filter(ImageFilter.UnsharpMask(radius=radius, percent=125, threshold=2))
    img = img.convert("RGBA")
    img.putalpha(_rounded_mask(size))
    return img


def _save_png(img: "Image.Image", path: Path) -> bytes:
    """保存 PNG 并返回其字节内容（用于 --check 比对）。"""
    img.save(path, format="PNG", optimize=True)
    return path.read_bytes()


def generate() -> Dict[Path, bytes]:
    """生成全部图标，返回 {路径: 字节内容}。"""
    tile = _load_tile()
    outputs: Dict[Path, bytes] = {}

    for name, size in PNG_TARGETS.items():
        path = STATIC_DIR / name
        outputs[path] = _save_png(render_icon(tile, size), path)

    # ICO：把 16/32/48 三档打包进同一个文件，浏览器自动挑最合适的。
    ico_path = ICO_PATH
    ico_img = render_icon(tile, max(ICO_SIZES))
    ico_img.save(
        ico_path,
        format="ICO",
        sizes=[(s, s) for s in ICO_SIZES],
    )
    outputs[ico_path] = ico_path.read_bytes()
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 rollout-insight favicon 资源")
    parser.add_argument(
        "--check", action="store_true", help="只校验图标是否与源图一致（CI 用）"
    )
    args = parser.parse_args()

    try:
        outputs = generate()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.check:
        stale: List[str] = []
        for path, content in outputs.items():
            if not path.is_file() or path.read_bytes() != content:
                stale.append(path.name)
        if stale:
            print(
                "[gen_favicon] --check 失败：以下图标与源图不一致：%s\n"
                "  修复方式：跑 `python scripts/gen_favicon.py` 重新生成。"
                % ", ".join(sorted(stale)),
                file=sys.stderr,
            )
            return 1
        print("[gen_favicon] --check OK：%d 个图标均为最新。" % len(outputs))
        return 0

    for path in sorted(outputs):
        print("[gen_favicon] 已写入 %s（%d 字节）。" % (path.relative_to(REPO_ROOT), len(outputs[path])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
