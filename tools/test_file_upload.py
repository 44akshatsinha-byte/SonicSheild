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

        # 2. Upload actual WAV recording via native file input
        file_input = page.locator('input[type="file"]').first
        file_input.set_input_files("data/smoke/smoke_human_like.wav")
        print("Uploaded data/smoke/smoke_human_like.wav via file uploader!")
        time.sleep(3)

        # 3. Capture screen showing metadata populated
        page.screenshot(path='qa_uploaded_metadata.png')

        # 4. Click Run Audio Analysis
        page.locator('button:has-text("Run Audio Analysis")').first.click()
        print("Clicked Run Audio Analysis. Waiting 7s for pipeline to execute...")
        time.sleep(7)

        # 5. Capture results at multiple scroll positions
        page.screenshot(path='qa_results_top.png')

        page.evaluate('document.querySelector("section.main").scrollTo(0, 480)')
        time.sleep(1)
        page.screenshot(path='qa_results_mid.png')

        page.evaluate('document.querySelector("section.main").scrollTo(0, 980)')
        time.sleep(1)
        page.screenshot(path='qa_results_bottom.png')

        browser.close()
        print("All file upload and analysis results screenshots captured successfully!")

if __name__ == '__main__':
    main()

