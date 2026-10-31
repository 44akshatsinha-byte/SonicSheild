"""Comprehensive visual verification script across all pages and viewports."""
import sys
import time
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding='utf-8')

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)

        # -------------------------------------------------------------
        # 1. Overview Page at 1366x768 (Standard Laptop Viewport)
        # -------------------------------------------------------------
        page = browser.new_page(viewport={'width': 1366, 'height': 768})
        page.goto('http://localhost:8501', wait_until='networkidle')
        time.sleep(3)
        page.screenshot(path='qa_overview_1366.png')
        page.screenshot(path='qa_overview_full.png', full_page=True)
        print("Captured qa_overview_1366.png and qa_overview_full.png")

        # -------------------------------------------------------------
        # 2. Analyze Audio Page — Idle State
        # -------------------------------------------------------------
        page.locator('section[data-testid="stSidebar"] button:has-text("Analyze Audio")').click()
        time.sleep(3)
        page.screenshot(path='qa_analyze_idle_1366.png')
        page.screenshot(path='qa_analyze_idle_full.png', full_page=True)
        print("Captured qa_analyze_idle_1366.png")

        # -------------------------------------------------------------
        # 3. Test Selectbox Dropdown Popover Colors
        # -------------------------------------------------------------
        selectbox = page.locator('div[data-testid="stSelectbox"]').first
        selectbox.click()
        time.sleep(1)
        page.screenshot(path='qa_dropdown_popover.png')
        print("Captured qa_dropdown_popover.png")
        # Close popover by clicking outside or pressing Escape
        page.keyboard.press("Escape")
        time.sleep(1)

        # -------------------------------------------------------------
        # 4. Run Analysis on Demo Sample to verify Results Presentation
        # -------------------------------------------------------------
        # Select the demo sample
        page.locator('div[data-testid="stSelectbox"]').first.click()
        time.sleep(1)
        # Click the second option (e.g., Demo: Human Like)
        options = page.locator('li[role="option"]').all()
        if len(options) > 1:
            options[1].click()
            time.sleep(1)

        # Click "Run Audio Analysis"
        page.locator('button:has-text("Run Audio Analysis")').first.click()
        time.sleep(6) # allow pipeline to complete
        page.screenshot(path='qa_analyze_results_1366.png')
        page.screenshot(path='qa_analyze_results_full.png', full_page=True)
        print("Captured qa_analyze_results_1366.png and qa_analyze_results_full.png")

        # -------------------------------------------------------------
        # 5. Engine Insights Page
        # -------------------------------------------------------------
        page.locator('section[data-testid="stSidebar"] button:has-text("Engine Insights")').click()
        time.sleep(2)
        page.screenshot(path='qa_insights_1366.png')
        print("Captured qa_insights_1366.png")

        # -------------------------------------------------------------
        # 6. Dataset Evaluation Page
        # -------------------------------------------------------------
        page.locator('section[data-testid="stSidebar"] button:has-text("Dataset Evaluation")').click()
        time.sleep(2)
        page.screenshot(path='qa_evaluation_1366.png')
        print("Captured qa_evaluation_1366.png")

        browser.close()
        print("All visual verification captures completed successfully!")

if __name__ == '__main__':
    main()

