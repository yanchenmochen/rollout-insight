"""界面图标清单：图标文件存在、清单条目完整、CSS 与图标目录一致。"""
from __future__ import annotations

import unittest
from pathlib import Path

from utils.icon_catalog import ICON_DIR, NAV_ICONS, icon_path, icon_svg, nav_icon

REPO_ROOT = Path(__file__).resolve().parent.parent
ICON_ROOT = REPO_ROOT / ICON_DIR


class TestIconFiles(unittest.TestCase):
    def test_icon_dir_exists(self):
        self.assertTrue(ICON_ROOT.is_dir(), f"{ICON_ROOT} 不存在")

    def test_every_catalog_icon_exists(self):
        missing = [
            key for key, meta in NAV_ICONS.items()
            if not (ICON_ROOT / f"{meta['icon']}.svg").is_file()
        ]
        self.assertEqual(missing, [], f"缺少图标文件: {missing}")

    def test_license_file_present(self):
        """Lucide 是 MIT 许可，必须随图标一起分发许可声明。"""
        self.assertTrue((ICON_ROOT / "LICENSE").is_file())

    def test_svg_files_are_wellformed(self):
        import xml.etree.ElementTree as ET

        for path in sorted(ICON_ROOT.glob("*.svg")):
            with self.subTest(icon=path.name):
                ET.parse(path)  # 解析失败会抛异常


class TestNavIcon(unittest.TestCase):
    def test_unknown_key_returns_empty(self):
        self.assertEqual(nav_icon("nope"), {})

    def test_known_key_returns_icon_and_hint(self):
        meta = nav_icon("tasks")
        self.assertEqual(meta["icon"], "list-checks")
        self.assertTrue(meta["hint"])

    def test_all_entries_have_icon_and_hint(self):
        for key, meta in NAV_ICONS.items():
            with self.subTest(key=key):
                self.assertTrue(meta.get("icon"))
                self.assertTrue(meta.get("hint"))


class TestIconPath(unittest.TestCase):
    def test_path_under_static(self):
        self.assertEqual(icon_path("list-checks"), "/static/icons/list-checks.svg")

    def test_blank_name(self):
        self.assertEqual(icon_path(""), "/static/icons/.svg")


class TestIconSvg(unittest.TestCase):
    def test_returns_svg_text_with_size(self):
        svg = icon_svg("list-checks", 20)
        self.assertIn("<svg", svg)
        self.assertIn('width="20"', svg)

    def test_missing_icon_returns_empty(self):
        self.assertEqual(icon_svg("definitely-not-an-icon"), "")


class TestGeneratedCss(unittest.TestCase):
    def test_css_is_up_to_date(self):
        """static/ri_icons.css 必须与图标目录同步（CI 跑 --check）。"""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "scripts/gen_icon_css.py", "--check"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"ri_icons.css 与图标目录不一致：{result.stdout}{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()