"""真实浏览器验证模型切换器：点开徽章 -> 切换 -> 抓报错 -> 截图。"""
import sys
from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8000/"
SHOT = "switch_ui_check.png"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 900, "height": 760})
        errors = []
        page.on("console", lambda m: errors.append("console." + m.type + ": " + m.text)
                if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append("pageerror: " + str(e)))

        page.goto(BASE, wait_until="networkidle", timeout=30000)
        page.wait_for_selector("#modePill", timeout=10000)

        pill = page.locator("#modePill")
        print("1. 徽章文本:", repr(pill.inner_text()))
        print("   徽章 disabled:", pill.is_disabled())
        print("   徽章 title:", repr(pill.get_attribute("title")))

        pill.click()
        page.wait_for_timeout(400)
        menu = page.locator("#modelMenu")
        print("2. 下拉菜单可见:", menu.is_visible())
        opts = page.locator("#modelMenu .mm-opt")
        print("   选项数量:", opts.count())
        for i in range(opts.count()):
            print("   -", opts.nth(i).get_attribute("data-role"),
                  opts.nth(i).get_attribute("data-model"),
                  repr(opts.nth(i).inner_text()))

        # 切换表达层到 GLM，再切回 Claude，确认切换生效且不残留
        target = page.locator('#modelMenu .mm-opt[data-role="main"][data-model="glm-4-flash-250414"]')
        if target.count():
            target.click()
            page.wait_for_timeout(600)
            print("3. 切到 GLM 后徽章:", repr(pill.inner_text()))
        back = page.locator('#modelMenu .mm-opt[data-role="main"][data-model="claude-haiku-4-5"]')
        if back.count():
            pill.click()
            page.wait_for_timeout(300)
            back.click()
            page.wait_for_timeout(600)
            print("4. 切回 Claude 后徽章:", repr(pill.inner_text()))

        final = page.evaluate("async () => await (await fetch('/api/models')).json()")
        print("5. 服务端 active:", final.get("active"))
        page.screenshot(path=SHOT, full_page=True)
        print("6. 截图已保存:", SHOT)
        print("7. 浏览器报错:", errors if errors else "无")
        browser.close()
        ok = (not errors) and final.get("active", {}).get("main") == "claude-haiku-4-5"
        print("RESULT:", "OK" if ok else "FAIL")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
