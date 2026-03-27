"""arxiv collector — fetches recent papers from AI/ML categories.

arxiv API docs: https://info.arxiv.org/help/api/index.html
Free, no auth. Rate limit: 1 request per 3 seconds (we only make 1).

Categories:
- cs.AI — Artificial Intelligence
- cs.LG — Machine Learning
- cs.CL — Computation and Language (NLP)
- cs.CV — Computer Vision
"""

import logging
from datetime import datetime

import feedparser
import httpx

from app.factory.collectors.base import BaseCollector
from app.factory.models import RawItem

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV"]
MAX_RESULTS = 20


class ArxivCollector(BaseCollector):
    name = "arxiv"

    def __init__(self, categories: list[str] | None = None, max_results: int = MAX_RESULTS):
        self.categories = categories or CATEGORIES
        self.max_results = max_results

    def collect(self) -> list[RawItem]:
        # Build query: cat:cs.AI OR cat:cs.LG OR ...
        cat_query = " OR ".join(f"cat:{c}" for c in self.categories)
        params = {
            "search_query": cat_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": self.max_results,
        }

        logger.info("[arxiv] Querying: %s (max %d)", cat_query, self.max_results)
        resp = httpx.get(ARXIV_API, params=params, timeout=30)
        resp.raise_for_status()

        feed = feedparser.parse(resp.text)
        items = []

        for entry in feed.entries:
            # Extract primary category
            primary_cat = entry.get("arxiv_primary_category", {}).get("term", "")
            tags = [primary_cat] if primary_cat else []
            # Add all categories
            for tag in entry.get("tags", []):
                t = tag.get("term", "")
                if t and t not in tags:
                    tags.append(t)

            # Extract arxiv ID from entry.id (format: http://arxiv.org/abs/XXXX.XXXXX)
            arxiv_id = entry.id.split("/abs/")[-1] if "/abs/" in entry.id else entry.id

            items.append(RawItem(
                source="arxiv",
                source_id=arxiv_id,
                title=entry.title.replace("\n", " ").strip(),
                url=entry.link,
                summary=entry.summary.replace("\n", " ").strip()[:1000],
                content=entry.summary.strip(),  # Full abstract
                score=0.0,  # arxiv has no engagement score; rank by recency
                tags=tags,
                language="en",
            ))

        logger.info("[arxiv] Fetched %d papers", len(items))
        return items
