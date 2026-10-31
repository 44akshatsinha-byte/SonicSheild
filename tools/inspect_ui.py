import sys
import time
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding='utf-8')

with sync_playwright() as p:
    browser = p.chromium.launch(channel='msedge', headless=True)
    page = browser.new_page(viewport={'width': 1366, 'height': 768})
    page.goto('http://localhost:8501', wait_until='networkidle')
    time.sleep(2)
    buttons = page.locator('button').all()
    print(f"Total buttons found: {len(buttons)}")
    for i, b in enumerate(buttons):
        txt = b.inner_text().replace('\n', ' ')
        print(f"Button {i}: '{txt}'")

    # Click the Analyze Audio button
    for b in buttons:
        if "Analyze Audio" in b.inner_text():
            print(f"Clicking: {b.inner_text()}")
            b.click()
            time.sleep(3)
            page.screenshot(path="screen_analyze_clicked.png", full_page=True)
            print("Analyze screenshot captured!")
            break

    browser.close()
