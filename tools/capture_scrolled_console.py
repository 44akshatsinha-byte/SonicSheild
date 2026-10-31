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

        # 2. Upload file
        file_input = page.locator('input[type="file"]').first
        file_input.set_input_files("data/smoke/smoke_human_like.wav")
        time.sleep(2)

        # 3. Click Run
        page.locator('button:has-text("Run Audio Analysis")').first.click()
        time.sleep(7)

        # Find scrolling element
        scroll_info = page.evaluate('''() => {
            const scrollable = Array.from(document.querySelectorAll('*')).filter(el => {
                const style = window.getComputedStyle(el);
                return (style.overflowY === 'auto' || style.overflowY === 'scroll') && el.scrollHeight > el.clientHeight;
            });
            if (scrollable.length > 0) {
                scrollable[0].scrollTop = 800;
                return scrollable[0].tagName + ' testid=' + scrollable[0].getAttribute('data-testid');
            }
            window.scrollTo(0, 800);
            return 'window';
        }''')
        print(f"Scrolled element: {scroll_info}")
        time.sleep(2)
        page.screenshot(path='qa_results_scrolled_console.png')

        # Scroll further down
        page.evaluate('''() => {
            const scrollable = Array.from(document.querySelectorAll('*')).filter(el => {
                const style = window.getComputedStyle(el);
                return (style.overflowY === 'auto' || style.overflowY === 'scroll') && el.scrollHeight > el.clientHeight;
            });
            if (scrollable.length > 0) {
                scrollable[0].scrollTop = 1600;
            }
        }''')
        time.sleep(2)
        page.screenshot(path='qa_results_scrolled_bottom.png')

        browser.close()
        print("Scrolled screenshots captured successfully!")

if __name__ == '__main__':
    main()

