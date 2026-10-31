import sys
import time
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding='utf-8')

with sync_playwright() as p:
    browser = p.chromium.launch(channel='msedge', headless=True)
    page = browser.new_page(viewport={'width': 1366, 'height': 768})
    page.goto('http://localhost:8501', wait_until='networkidle')
    time.sleep(2)
    
    # Go to Analyze Audio
    buttons = page.locator('button').all()
    for b in buttons:
        if 'Analyze Audio' in b.inner_text():
            b.click()
            break
    time.sleep(2)
    
    # Click the demo selectbox
    select_box = page.locator('div[data-baseweb="select"]').first
    select_box.click()
    time.sleep(1)
    page.screenshot(path='screen_popover.png')
    print('Popover screenshot captured!')
    browser.close()

