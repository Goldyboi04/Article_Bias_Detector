from transformers import pipeline
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

# Load zero-shot classifier with GPU (if available)
pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)


def detect_bias(text):
    if not text or len(text.strip()) < 20:
        return [{"label": "Unknown", "score": 0.0}]

    labels = [
        "This article is from a liberal or progressive political perspective.",
        "This article is from a centrist or neutral political perspective.",
        "This article is from a conservative or right-leaning political perspective."
    ]

    result = pipe(text, candidate_labels=labels)
    top = {
        "label": result["labels"][0],
        "score": result["scores"][0]
    }
    return [top]


async def fetch_article_async(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        content = await page.content()
        await browser.close()

        soup = BeautifulSoup(content, "html.parser")

        article_tag = soup.find("article")
        if article_tag:
            article_text = article_tag.get_text(separator="\n", strip=True)
        else:
            article_tags = soup.find_all(["p", "h1", "h2", "h3"])
            article_text = "\n".join(tag.get_text(strip=True) for tag in article_tags)

        return article_text.strip()
