from transformers import pipeline
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

async def fetch_article_async(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        content = await page.content()
        await browser.close()

        soup = BeautifulSoup(content, "html.parser")

        # Try to extract meaningful text (you can customize this)
        article_tags = soup.find_all(["p", "h1", "h2", "h3"])
        article_text = "\n".join(tag.get_text(strip=True) for tag in article_tags)
        return article_text.strip()


pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Run bias detection
def detect_bias(article_text):
    return pipe(article_text[:512])  # Truncate to fit model input size