import sys
import time
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding='utf-8')

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1366, 'height': 800})
        page.goto('http://localhost:8501', wait_until='networkidle')
        time.sleep(2)

        # 1. Navigate to Analyze Audio
        page.locator('section[data-testid="stSidebar"] button:has-text("Analyze Audio")').click()
        time.sleep(2)

        # 2. Click the selectbox
        page.locator('div[data-testid="stSelectbox"]').first.click()
        time.sleep(1)

        # 3. Click the Demo option specifically by text
        demo_option = page.locator('li[role="option"]:has-text("Demo: Human Like")')
        if demo_option.count() > 0:
            demo_option.click()
            print("Selected Demo: Human Like")
        else:
            # Fallback to second option
            page.locator('li[role="option"]').nth(1).click()
            print("Selected option index 1")
        time.sleep(2)

        # Capture state after selecting demo
        page.screenshot(path='qa_analyze_selected.png')

        # 4. Click Run Audio Analysis
        page.locator('button:has-text("Run Audio Analysis")').first.click()
        print("Clicked Run Audio Analysis. Waiting 7s for pipeline...")
        time.sleep(7)

        # 5. Capture viewport and scroll down to capture results
        page.screenshot(path='qa_result_top.png')

        # Scroll down in Streamlit's main section
        page.evaluate('document.querySelector("section.main").scrollTo(0, 450)')
        time.sleep(1)
        page.screenshot(path='qa_result_mid.png')

        page.evaluate('document.querySelector("section.main").scrollTo(0, 950)')
        time.sleep(1)
        page.screenshot(path='qa_result_bottom.png')

        browser.close()
        print("Analysis results screenshots saved successfully!")

if __name__ == '__main__':
    main()

