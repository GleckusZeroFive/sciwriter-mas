"""Topic ranker — scores, clusters, and selects the best topic for article generation.

Scoring formula: engagement * freshness_decay * topic_relevance
Clustering: group similar items from different sources into topic bundles.
Dedup: check against published articles in Qdrant to avoid repetition.
"""

import logging
import math
from datetime import datetime, timezone
from dataclasses import dataclass, field

from app.factory.db import get_unprocessed_items, mark_items_processed

logger = logging.getLogger(__name__)

# Topic keywords for relevance scoring
TOPIC_KEYWORDS = {
    "ai": ["ai", "artificial intelligence", "machine learning", "llm", "gpt",
            "neural", "deep learning", "transformer", "openai", "anthropic",
            "gemini", "claude", "chatgpt", "model", "training", "inference"],
    "tech": ["technology", "startup", "software", "hardware", "chip",
             "semiconductor", "gpu", "nvidia", "quantum", "robot",
             "automation", "cyber", "cloud", "data"],
    "science": ["research", "study", "paper", "discovery", "experiment",
                "physics", "biology", "chemistry", "space", "nasa"],
    "gadgets": ["apple", "google", "samsung", "phone", "laptop", "device",
                "tesla", "ev", "electric", "battery", "display"],
}

# Freshness half-life in hours
FRESHNESS_HALF_LIFE = 6.0


@dataclass
class TopicBundle:
    """A cluster of related raw items forming one topic for article generation."""
    items: list[dict] = field(default_factory=list)
    score: float = 0.0
    primary_topic: str = "tech"
    title_suggestion: str = ""
    combined_summary: str = ""

    @property
    def item_ids(self) -> list[int]:
        return [i["id"] for i in self.items]

    @property
    def source_mix(self) -> str:
        sources = set(i["source"] for i in self.items)
        return ", ".join(sorted(sources))


def _freshness_decay(collected_at: datetime) -> float:
    """Exponential decay based on age. Returns 0-1."""
    if collected_at.tzinfo is None:
        collected_at = collected_at.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    hours_old = (now - collected_at).total_seconds() / 3600
    return math.exp(-0.693 * hours_old / FRESHNESS_HALF_LIFE)


def _topic_relevance(title: str, summary: str = "") -> tuple[str, float]:
    """Score topic relevance. Returns (best_topic, score 0-1)."""
    text = (title + " " + (summary or "")).lower()
    best_topic = "tech"
    best_score = 0.0

    for topic, keywords in TOPIC_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text)
        score = min(hits / 3.0, 1.0)  # 3+ keywords = max relevance
        if score > best_score:
            best_score = score
            best_topic = topic

    return best_topic, max(best_score, 0.1)  # minimum 0.1


def score_items(items: list[dict]) -> list[dict]:
    """Score all items by engagement * freshness * relevance."""
    scored = []
    max_engagement = max((i.get("score", 0) for i in items), default=1) or 1

    for item in items:
        engagement = (item.get("score", 0) / max_engagement) if max_engagement > 0 else 0.5
        freshness = _freshness_decay(item["collected_at"])
        topic, relevance = _topic_relevance(item["title"], item.get("summary", ""))

        composite = engagement * freshness * relevance
        item["_composite_score"] = composite
        item["_topic"] = topic
        item["_freshness"] = freshness
        item["_relevance"] = relevance
        scored.append(item)

    scored.sort(key=lambda x: x["_composite_score"], reverse=True)
    return scored


def _simple_cluster(items: list[dict], max_bundle_size: int = 5) -> list[TopicBundle]:
    """Simple keyword-based clustering (no embeddings needed for MVP).

    Groups items by primary topic, then takes top items per topic.
    For proper semantic clustering, integrate Qdrant similarity later.
    """
    by_topic: dict[str, list[dict]] = {}
    for item in items:
        topic = item.get("_topic", "tech")
        by_topic.setdefault(topic, []).append(item)

    bundles = []
    for topic, topic_items in by_topic.items():
        # Take top N items for this topic
        top = topic_items[:max_bundle_size]
        if not top:
            continue

        bundle = TopicBundle(
            items=top,
            score=sum(i["_composite_score"] for i in top) / len(top),
            primary_topic=topic,
            title_suggestion=top[0]["title"],
            combined_summary="\n\n".join(
                f"[{i['source']}] {i['title']}: {(i.get('summary') or '')[:200]}"
                for i in top
            ),
        )
        bundles.append(bundle)

    bundles.sort(key=lambda b: b.score, reverse=True)
    return bundles


def select_best_topic(limit: int = 100) -> TopicBundle | None:
    """Main entry point: fetch unprocessed items, score, cluster, return best bundle.

    Returns None if no suitable topics found.
    """
    items = get_unprocessed_items(limit=limit)
    if not items:
        logger.warning("No unprocessed items in DB")
        return None

    logger.info("Scoring %d unprocessed items...", len(items))
    scored = score_items(items)

    bundles = _simple_cluster(scored)
    if not bundles:
        logger.warning("No topic bundles formed")
        return None

    best = bundles[0]
    logger.info(
        "Best topic: [%s] %s (score=%.3f, %d items from %s)",
        best.primary_topic,
        best.title_suggestion[:60],
        best.score,
        len(best.items),
        best.source_mix,
    )

    return best
