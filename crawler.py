"""
crawler.py — Web page fetcher

fetch_and_parse(url): Sử dụng httpx + BeautifulSoup để trích xuất nội dung HTML thành text.
Dùng bởi luồng tìm kiếm động (dynamic web search) hoặc trực tiếp qua API `/ingest/url`.
"""

import asyncio
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

async def fetch_and_parse_async(url: str) -> str:
    """
    Lấy nội dung trang web bằng Playwright (có render Javascript).
    Dành cho Web động (React/Vue/Angular) và chờ cho đến khi network kết thúc.
    """
    try:
        print(f"[crawler] Đang tải Web bằng Chromium: {url[:70]}...")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()
            
            # Đợi tải tới khi không còn request network nào trong 500ms
            await page.goto(url, wait_until="networkidle", timeout=30000)
            
            html_content = await page.content()
            await browser.close()
            
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Bỏ qua các tag thừa thãi rác
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
            
        text = soup.get_text(separator="\n", strip=True)
        print(f"[crawler] Parse thành công! Độ dài: {len(text)} ký tự.")
        return text
    except Exception as e:
        print(f"[crawler] Lỗi khi cào trang '{url}': {e}")
        return ""


def fetch_and_parse(url: str) -> str:
    """
    Wrapper đồng bộ cho fetch_and_parse_async.
    Dùng bởi các endpoint FastAPI không async.
    """
    try:
        return asyncio.run(fetch_and_parse_async(url))
    except Exception as e:
        print(f"[crawler] Lỗi wrapper fetch_and_parse: {e}")
        return ""
