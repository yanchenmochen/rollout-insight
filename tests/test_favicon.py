"""站点图标（favicon）测试：产物齐全、尺寸正确、可被浏览器端点取到。"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from utils.config_loader import config

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "static"

# 与 scripts/gen_favicon.py 的输出清单保持一致。
PNG_TARGETS = {
    "favicon-16x16.png": 16,
    "favicon-32x32.png": 32,
    "favicon-48x48.png": 48,
    "apple-touch-icon.png": 180,
    "icon-192.png": 192,
    "icon-512.png": 512,
}


class TestFaviconAssets(unittest.TestCase):
    def test_source_image_present(self):
        """源图是图标的唯一真源，必须入库。"""
        self.assertTrue((STATIC_DIR / "branding" / "ri_icon_source.png").is_file())

    def test_all_png_targets_exist(self):
        missing = [name for name in PNG_TARGETS if not (STATIC_DIR / name).is_file()]
        self.assertEqual(missing, [], f"缺少图标文件: {missing}")

    def test_ico_exists(self):
        self.assertTrue((STATIC_DIR / "favicon.ico").is_file())

    def test_png_dimensions_match_target(self):
        from PIL import Image

        for name, size in PNG_TARGETS.items():
            with self.subTest(icon=name):
                with Image.open(STATIC_DIR / name) as img:
                    self.assertEqual(img.size, (size, size))

    def test_ico_contains_multiple_sizes(self):
        """ICO 必须打包 16/32/48，浏览器才能自动挑最合适的一档。"""
        from PIL import Image

        with Image.open(STATIC_DIR / "favicon.ico") as img:
            sizes = {tuple(s) for s in img.info.get("sizes", [])}
        for expected in ((16, 16), (32, 32), (48, 48)):
            self.assertIn(expected, sizes)

    def test_generated_assets_are_up_to_date(self):
        """图标必须与源图同步（CI 跑 --check）。"""
        result = subprocess.run(
            [sys.executable, "scripts/gen_favicon.py", "--check"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"favicon 与源图不一致：{result.stdout}{result.stderr}",
        )


class TestFaviconTemplate(unittest.TestCase):
    def test_base_html_declares_icons(self):
        """base.html 必须声明 favicon / apple-touch-icon / manifest。"""
        html = (REPO_ROOT / "templates" / "base.html").read_text(encoding="utf-8")
        for needle in (
            '/static/favicon.ico',
            '/static/favicon-16x16.png',
            '/static/favicon-32x32.png',
            '/static/favicon-48x48.png',
            '/static/apple-touch-icon.png',
            '/site.webmanifest',
        ):
            with self.subTest(tag=needle):
                self.assertIn(needle, html)


class TestSiteController(unittest.TestCase):
    """站点控制器路由：图标端点无需登录即可访问。"""

    def test_routes_registered(self):
        from controller.site_controller import router

        paths = {route.path for route in router.routes}
        for expected in ("/favicon.ico", "/apple-touch-icon.png", "/site.webmanifest"):
            with self.subTest(path=expected):
                self.assertIn(expected, paths)

    def test_router_has_no_login_dependency(self):
        """浏览器在未登录时也要能拿到 favicon，故不能挂 require_login。"""
        from controller.site_controller import router

        self.assertEqual(getattr(router, "dependencies", []), [])

    def test_manifest_payload(self):
        import asyncio
        import json

        from controller.site_controller import get_site_manifest

        manifest = asyncio.run(get_site_manifest())
        # JSONResponse.body 是 bytes；解出来校验关键字段。
        data = json.loads(manifest.body)
        self.assertEqual(data["name"], config.site_brand_name)
        self.assertTrue(data["icons"])
        for icon in data["icons"]:
            self.assertTrue(icon["src"].startswith("/static/"))


class TestSiteBrand(unittest.TestCase):
    """站点品牌名：配置化 + 模板标题统一走它。"""

    def test_config_exposes_brand_name(self):
        self.assertTrue(config.site_brand_name)

    def test_template_factory_injects_brand(self):
        """create_templates 必须把品牌名注入 env.globals，模板才拿得到。"""
        from utils.template_factory import create_templates

        templates = create_templates()
        self.assertEqual(
            templates.env.globals.get("site_brand_name"), config.site_brand_name
        )

    def test_base_html_uses_brand_variable(self):
        html = (REPO_ROOT / "templates" / "base.html").read_text(encoding="utf-8")
        self.assertIn("site_brand_name", html)
        # 顶栏 logo 用品牌名，不再写死英文
        self.assertIn('class="navbar-brand">{{ brand }}', html)

    def test_page_titles_end_with_brand(self):
        """各页面标题统一是「页面名 · 品牌名」，保证标签页短、不被折叠。"""
        template_dir = REPO_ROOT / "templates"
        brand = config.site_brand_name
        checked = 0
        for path in sorted(template_dir.glob("*.html")):
            if path.name in {"base.html", "_sandbox_exec.html", "_task_jump_nav.html"}:
                continue
            text = path.read_text(encoding="utf-8")
            if "{% block title %}" not in text:
                continue
            with self.subTest(template=path.name):
                self.assertIn("{{ brand }}", text)
                self.assertNotIn("Rollout Insight", text)
            checked += 1
        self.assertGreater(checked, 10, "应至少检查到 10 个带 title 的页面模板")


if __name__ == "__main__":
    unittest.main()
