"""HackerNews collector — fetches top stories via Firebase API.

HN API docs: https://github.com/HackerNews/API
No auth required, no rate limits documented (be reasonable).

Endpoints used:
- GET /v0/topstories.json → list of top story IDs
- GET /v0/item/{id}.json → story details
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from app.factory.collectors.base import BaseCollector
from app.factory.models import RawItem

logger = logging.getLogger(__name__)

HN_API = "https://hacker-news.firebaseio.com/v0"
MIN_SCORE = 30
MAX_STORIES = 30


class HackerNewsCollector(BaseCollector):
    name = "hackernews"

    def __init__(self, max_stories: int = MAX_STORIES, min_score: int = MIN_SCORE):
        self.max_stories = max_stories
        self.min_score = min_score

    def _fetch_item(self, item_id: int) -> dict | None:
        """Fetch a single HN item by id."""
        try:
            resp = httpx.get(f"{HN_API}/item/{item_id}.json", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.debug("Failed to fetch HN item %d: %s", item_id, e)
            return None

    def collect(self) -> list[RawItem]:
        # 1. Get top story IDs
        resp = httpx.get(f"{HN_API}/topstories.json", timeout=10)
        resp.raise_for_status()
        story_ids = resp.json()[:self.max_stories]
        logger.info("[hackernews] Fetching %d top stories...", len(story_ids))

        # 2. Fetch stories in parallel (5 threads)
        stories = []
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(self._fetch_item, sid): sid for sid in story_ids}
            for future in as_completed(futures):
                item = future.result()
                if item and item.get("type") == "story" and item.get("score", 0) >= self.min_score:
                    stories.append(item)

        # 3. Convert to RawItem
        items = []
        for s in stories:
            items.append(RawItem(
                source="hackernews",
                source_id=str(s["id"]),
                title=s.get("title", ""),
                url=s.get("url", f"https://news.ycombinator.com/item?id={s['id']}"),
                summary=f"Score: {s.get('score', 0)}, Comments: {s.get('descendants', 0)}",
                score=float(s.get("score", 0)),
                tags=["hackernews"],
                language="en",
            ))

        # Sort by score descending
        items.sort(key=lambda x: x.score, reverse=True)
        logger.info("[hackernews] %d stories above min_score=%d", len(items), self.min_score)
        return items
