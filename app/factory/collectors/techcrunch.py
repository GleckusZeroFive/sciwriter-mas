"""TechCrunch RSS collector — fetches recent articles via RSS feed.

Feed URL: https://techcrunch.com/feed/
Standard RSS/Atom, parsed with feedparser. No auth required.
"""

import logging

import feedparser

from app.factory.collectors.base import BaseCollector
from app.factory.models import RawItem

logger = logging.getLogger(__name__)

FEED_URL = "https://techcrunch.com/feed/"
# Additional AI/tech feeds
EXTRA_FEEDS = [
    ("https://www.theverge.com/rss/index.xml", "theverge"),
    ("https://feeds.arstechnica.com/arstechnica/index", "arstechnica"),
]

# Keywords to filter relevant articles
RELEVANCE_KEYWORDS = [
    "ai", "artificial intelligence", "machine learning", "llm", "gpt",
    "robot", "neural", "deep learning", "automation", "quantum",
    "chip", "semiconductor", "gpu", "nvidia", "openai", "google",
    "apple", "microsoft", "tesla", "spacex", "startup", "funding",
    "technology", "tech", "software", "hardware", "cyber",
]


class TechCrunchCollector(BaseCollector):
    name = "techcrunch"

    def __init__(self, include_extra: bool = True):
        self.include_extra = include_extra

    def _parse_feed(self, url: str, source_tag: str) -> list[RawItem]:
        """Parse a single RSS feed into RawItems."""
        feed = feedparser.parse(url)
        items = []

        for entry in feed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            link = entry.get("link", "")

            # Simple relevance filter: check if title or summary contains any keyword
            text = (title + " " + summary).lower()
            relevant = any(kw in text for kw in RELEVANCE_KEYWORDS)
            if not relevant:
                continue

            # Use link as source_id (unique per article)
            source_id = link or title[:100]

            # Extract tags from categories if available
            tags = [source_tag]
            for tag in entry.get("tags", []):
                t = tag.get("term", "")
                if t:
                    tags.append(t.lower())

            items.append(RawItem(
                source="techcrunch",
                source_id=source_id,
                title=title,
                url=link,
                summary=summary[:500] if summary else None,
                score=0.0,  # RSS has no engagement score
                tags=tags,
                language="en",
            ))

        return items

    def collect(self) -> list[RawItem]:
        all_items = []

        # Main TechCrunch feed
        logger.info("[techcrunch] Fetching TechCrunch RSS...")
        all_items.extend(self._parse_feed(FEED_URL, "techcrunch"))

        # Extra feeds
        if self.include_extra:
            for feed_url, tag in EXTRA_FEEDS:
                logger.info("[techcrunch] Fetching %s RSS...", tag)
                all_items.extend(self._parse_feed(feed_url, tag))

        logger.info("[techcrunch] Collected %d relevant articles from RSS feeds", len(all_items))
        return all_items
