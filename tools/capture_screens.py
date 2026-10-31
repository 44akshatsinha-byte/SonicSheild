"""Quick visual verification tool using Playwright."""
import time
from playwright.sync_api import sync_playwright

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1366, 'height': 768})

        # 1. Overview Page
        page.goto('http://localhost:8501', wait_until='networkidle')
        time.sleep(3)
        page.screenshot(path='screen_overview.png', full_page=True)

        # 2. Click Analyze Audio button in sidebar (index 1)
        sidebar_buttons = page.locator('section[data-testid="stSidebar"] button')
        if sidebar_buttons.count() > 1:
            sidebar_buttons.nth(1).click()
            time.sleep(3)
            page.screenshot(path='screen_analyze.png', full_page=True)

            # 3. Test opening a selectbox to check popover colors
            select_box = page.locator('div[data-baseweb="select"]').first
            if select_box.count() > 0:
                select_box.click()
                time.sleep(1)
                page.screenshot(path='screen_analyze_popover.png')

        browser.close()
        print('Visual verification captures complete!')

if __name__ == '__main__':
    main()

