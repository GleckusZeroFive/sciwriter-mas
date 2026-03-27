"""Content extractor — fetches and extracts article text from URLs.

Uses httpx + simple HTML-to-text extraction.
Falls back gracefully if a URL is unreachable or returns garbage.
"""

import logging
import re
from html.parser import HTMLParser

import httpx

logger = logging.getLogger(__name__)

# Tags whose content we want to extract
CONTENT_TAGS = {"p", "h1", "h2", "h3", "h4", "li", "td", "th", "blockquote", "pre", "code"}
# Tags to skip entirely
SKIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"}

# Max content length to store (chars)
MAX_CONTENT_LENGTH = 8000

# Request timeout
TIMEOUT = 15


class _TextExtractor(HTMLParser):
    """Simple HTML parser that extracts visible text from content tags."""

    def __init__(self):
        super().__init__()
        self.result = []
        self._skip_depth = 0
        self._in_content = False
        self._current_tag = None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self._skip_depth += 1
        elif tag in CONTENT_TAGS:
            self._in_content = True
            self._current_tag = tag
            if tag in ("h1", "h2", "h3", "h4"):
                self.result.append("\n\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in CONTENT_TAGS:
            self._in_content = False
            self._current_tag = None
            self.result.append("\n")

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        text = data.strip()
        if text and self._in_content:
            self.result.append(text + " ")

    def get_text(self) -> str:
        return "".join(self.result).strip()


def extract_text_from_html(html: str) -> str:
    """Extract readable text from HTML content."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    text = parser.get_text()

    # Clean up: collapse whitespace, remove excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def fetch_article_content(url: str) -> str | None:
    """Fetch URL and extract article text. Returns None on failure.

    Tries to get the main content, strips navigation/ads/footers.
    Truncates to MAX_CONTENT_LENGTH.
    """
    if not url:
        return None

    # Skip non-article URLs
    skip_domains = ["github.com", "youtube.com", "twitter.com", "x.com", "reddit.com"]
    for domain in skip_domains:
        if domain in url:
            logger.debug("Skipping non-article URL: %s", url[:80])
            return None

    try:
        resp = httpx.get(
            url,
            timeout=TIMEOUT,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; SciWriter/1.0; +https://github.com/GleckusZeroFive/sciwriter-mas)",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            logger.debug("Non-HTML content type for %s: %s", url[:60], content_type)
            return None

        text = extract_text_from_html(resp.text)

        if len(text) < 200:
            logger.debug("Too little text extracted from %s: %d chars", url[:60], len(text))
            return None

        # Truncate
        if len(text) > MAX_CONTENT_LENGTH:
            text = text[:MAX_CONTENT_LENGTH] + "\n\n[...truncated]"

        logger.info("Extracted %d chars from %s", len(text), url[:60])
        return text

    except httpx.TimeoutException:
        logger.debug("Timeout fetching %s", url[:60])
        return None
    except httpx.HTTPStatusError as e:
        logger.debug("HTTP error %d for %s", e.response.status_code, url[:60])
        return None
    except Exception as e:
        logger.debug("Failed to fetch %s: %s", url[:60], e)
        return None


def enrich_items_with_content(items: list[dict], max_fetch: int = 10) -> int:
    """Fetch full article content for items that have a URL but no content.

    Modifies items in-place. Returns count of successfully enriched items.
    Only fetches up to max_fetch items to avoid hammering servers.
    """
    enriched = 0
    for item in items:
        if enriched >= max_fetch:
            break
        if item.get("content") and len(item["content"]) > 200:
            continue  # Already has content
        url = item.get("url")
        if not url:
            continue

        content = fetch_article_content(url)
        if content:
            item["content"] = content
            enriched += 1

    logger.info("Enriched %d/%d items with full article content", enriched, len(items))
    return enriched
