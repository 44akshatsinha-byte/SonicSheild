import sys
import time
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding='utf-8')

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        # Using a taller viewport to capture the full report cleanly!
        page = browser.new_page(viewport={'width': 1366, 'height': 1200})
        page.goto('http://localhost:8501', wait_until='networkidle')
        time.sleep(2)

        # 1. Navigate to Analyze Audio
        page.locator('section[data-testid="stSidebar"] button:has-text("Analyze Audio")').click()
        time.sleep(2)

        # 2. Upload file
        file_input = page.locator('input[type="file"]').first
        file_input.set_input_files("data/smoke/smoke_human_like.wav")
        time.sleep(2)

        # 3. Click Run
        page.locator('button:has-text("Run Audio Analysis")').first.click()
        print("Running pipeline...")
        time.sleep(7)

        # 4. Scroll down to show results
        page.evaluate('''() => {
            const el = document.querySelector('[data-testid="stAppViewContainer"]') || window;
            el.scrollTo(0, 420);
        }''')
        time.sleep(2)
        page.screenshot(path='qa_results_section1.png')

        # 5. Scroll further down to show dark console & gauges & signals
        page.evaluate('''() => {
            const el = document.querySelector('[data-testid="stAppViewContainer"]') || window;
            el.scrollTo(0, 1000);
        }''')
        time.sleep(2)
        page.screenshot(path='qa_results_section2.png')

        browser.close()
        print("Captured qa_results_section1.png and qa_results_section2.png successfully!")

if __name__ == '__main__':
    main()

