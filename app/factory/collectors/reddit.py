"""Reddit collector — fetches hot posts from tech subreddits via PRAW.

Requires Reddit API credentials:
- FACTORY_REDDIT_CLIENT_ID
- FACTORY_REDDIT_CLIENT_SECRET
- FACTORY_REDDIT_USER_AGENT

Register app at: https://www.reddit.com/prefs/apps
Choose "script" type. Redirect URI: http://localhost:8080
"""

import logging

import praw
from praw.exceptions import PRAWException

from app.config import settings
from app.factory.collectors.base import BaseCollector
from app.factory.models import RawItem

logger = logging.getLogger(__name__)

SUBREDDITS = [
    "technology",
    "MachineLearning",
    "artificial",
    "science",
]
MIN_SCORE = 50
POSTS_PER_SUB = 25


class RedditCollector(BaseCollector):
    name = "reddit"

    def __init__(
        self,
        subreddits: list[str] | None = None,
        min_score: int = MIN_SCORE,
        posts_per_sub: int = POSTS_PER_SUB,
    ):
        self.subreddits = subreddits or SUBREDDITS
        self.min_score = min_score
        self.posts_per_sub = posts_per_sub

    def _get_reddit(self) -> praw.Reddit:
        if not settings.factory_reddit_client_id:
            raise ValueError(
                "Reddit credentials not configured. "
                "Set FACTORY_REDDIT_CLIENT_ID and FACTORY_REDDIT_CLIENT_SECRET in .env. "
                "Register app at https://www.reddit.com/prefs/apps"
            )
        return praw.Reddit(
            client_id=settings.factory_reddit_client_id,
            client_secret=settings.factory_reddit_client_secret,
            user_agent=settings.factory_reddit_user_agent,
        )

    def collect(self) -> list[RawItem]:
        reddit = self._get_reddit()
        items = []

        for sub_name in self.subreddits:
            logger.info("[reddit] Fetching r/%s hot posts...", sub_name)
            try:
                subreddit = reddit.subreddit(sub_name)
                for post in subreddit.hot(limit=self.posts_per_sub):
                    if post.score < self.min_score:
                        continue
                    if post.stickied:
                        continue

                    items.append(RawItem(
                        source="reddit",
                        source_id=post.id,
                        title=post.title,
                        url=post.url if not post.is_self else f"https://reddit.com{post.permalink}",
                        summary=post.selftext[:500] if post.is_self else None,
                        content=post.selftext if post.is_self else None,
                        score=float(post.score),
                        tags=[f"r/{sub_name}"],
                        language="en",
                    ))
            except PRAWException as e:
                logger.error("[reddit] Failed to fetch r/%s: %s", sub_name, e)

        items.sort(key=lambda x: x.score, reverse=True)
        logger.info("[reddit] %d posts above min_score=%d from %d subreddits",
                     len(items), self.min_score, len(self.subreddits))
        return items
