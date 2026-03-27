"""PostgreSQL database operations for the content factory."""

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import psycopg2
import psycopg2.extras

from app.config import settings
from app.factory.models import RawItem, Article, PublishRecord

logger = logging.getLogger(__name__)

# Register JSON adapter for psycopg2
psycopg2.extras.register_default_jsonb()


def get_connection():
    """Create a new database connection."""
    return psycopg2.connect(settings.pg_dsn)


@contextmanager
def get_cursor(commit=True):
    """Context manager for database cursor with auto-commit."""
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


# --- Raw Items ---

def insert_raw_item(item: RawItem) -> Optional[int]:
    """Insert a raw item, skip if duplicate (source + source_id). Returns id or None."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO raw_items (source, source_id, url, title, summary, content, score, tags, language)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source, source_id) DO NOTHING
            RETURNING id
            """,
            (item.source, item.source_id, item.url, item.title, item.summary,
             item.content, item.score, item.tags, item.language),
        )
        row = cur.fetchone()
        if row:
            logger.info("Inserted raw_item: [%s] %s (id=%d)", item.source, item.title[:60], row["id"])
            return row["id"]
        return None


def insert_raw_items_batch(items: list[RawItem]) -> int:
    """Insert multiple raw items, skip duplicates. Returns count of new items."""
    inserted = 0
    with get_cursor() as cur:
        for item in items:
            cur.execute(
                """
                INSERT INTO raw_items (source, source_id, url, title, summary, content, score, tags, language)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source, source_id) DO NOTHING
                RETURNING id
                """,
                (item.source, item.source_id, item.url, item.title, item.summary,
                 item.content, item.score, item.tags, item.language),
            )
            if cur.fetchone():
                inserted += 1
    logger.info("Batch insert: %d/%d new items", inserted, len(items))
    return inserted


def get_unprocessed_items(limit: int = 100, source: Optional[str] = None) -> list[dict]:
    """Fetch unprocessed raw items, ordered by score desc."""
    with get_cursor(commit=False) as cur:
        if source:
            cur.execute(
                "SELECT * FROM raw_items WHERE NOT processed AND source = %s ORDER BY score DESC LIMIT %s",
                (source, limit),
            )
        else:
            cur.execute(
                "SELECT * FROM raw_items WHERE NOT processed ORDER BY score DESC LIMIT %s",
                (limit,),
            )
        return [dict(row) for row in cur.fetchall()]


def mark_items_processed(item_ids: list[int]):
    """Mark raw items as processed."""
    if not item_ids:
        return
    with get_cursor() as cur:
        cur.execute(
            "UPDATE raw_items SET processed = TRUE WHERE id = ANY(%s)",
            (item_ids,),
        )
        logger.info("Marked %d items as processed", len(item_ids))


# --- Articles ---

def create_article(
    source_item_ids: list[int],
    topic_summary: str,
    status: str = "queued",
) -> int:
    """Create a new article record. Returns article id."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO articles (source_item_ids, topic_summary, status)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (source_item_ids, topic_summary, status),
        )
        article_id = cur.fetchone()["id"]
        logger.info("Created article id=%d, sources=%s", article_id, source_item_ids)
        return article_id


def update_article(article_id: int, **kwargs):
    """Update article fields by id. Pass field=value pairs."""
    if not kwargs:
        return
    # Serialize lists/dicts to JSON for JSONB columns
    for key in ("generation_log", "media"):
        if key in kwargs and not isinstance(kwargs[key], str):
            kwargs[key] = json.dumps(kwargs[key], ensure_ascii=False)

    set_parts = [f"{k} = %s" for k in kwargs]
    values = list(kwargs.values())
    values.append(article_id)

    with get_cursor() as cur:
        cur.execute(
            f"UPDATE articles SET {', '.join(set_parts)} WHERE id = %s",
            values,
        )
        logger.info("Updated article id=%d: %s", article_id, list(kwargs.keys()))


def get_article(article_id: int) -> Optional[dict]:
    """Fetch article by id."""
    with get_cursor(commit=False) as cur:
        cur.execute("SELECT * FROM articles WHERE id = %s", (article_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def get_articles_by_status(status: str, limit: int = 10) -> list[dict]:
    """Fetch articles by status, ordered by created_at."""
    with get_cursor(commit=False) as cur:
        cur.execute(
            "SELECT * FROM articles WHERE status = %s ORDER BY created_at LIMIT %s",
            (status, limit),
        )
        return [dict(row) for row in cur.fetchall()]


def get_recent_articles(limit: int = 20) -> list[dict]:
    """Fetch recent articles regardless of status."""
    with get_cursor(commit=False) as cur:
        cur.execute(
            "SELECT * FROM articles ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]


# --- Publish Log ---

def schedule_publish(article_id: int, platform: str, scheduled_at: datetime) -> int:
    """Schedule an article for publishing. Returns publish_log id."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO publish_log (article_id, platform, scheduled_at)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (article_id, platform, scheduled_at),
        )
        pub_id = cur.fetchone()["id"]
        logger.info("Scheduled publish id=%d: article=%d platform=%s at=%s",
                     pub_id, article_id, platform, scheduled_at)
        return pub_id


def get_pending_publishes(limit: int = 5) -> list[dict]:
    """Fetch pending publishes that are due."""
    with get_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT pl.*, a.title_ru, a.content_ru
            FROM publish_log pl
            JOIN articles a ON a.id = pl.article_id
            WHERE pl.status = 'pending' AND pl.scheduled_at <= NOW()
            ORDER BY pl.scheduled_at
            LIMIT %s
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]


def update_publish_log(pub_id: int, **kwargs):
    """Update publish log entry."""
    if not kwargs:
        return
    set_parts = [f"{k} = %s" for k in kwargs]
    values = list(kwargs.values())
    values.append(pub_id)

    with get_cursor() as cur:
        cur.execute(
            f"UPDATE publish_log SET {', '.join(set_parts)} WHERE id = %s",
            values,
        )


# --- Stats ---

def get_factory_stats() -> dict:
    """Get overall factory statistics."""
    with get_cursor(commit=False) as cur:
        stats = {}

        cur.execute("SELECT COUNT(*) as total, COUNT(*) FILTER (WHERE NOT processed) as unprocessed FROM raw_items")
        row = cur.fetchone()
        stats["raw_items_total"] = row["total"]
        stats["raw_items_unprocessed"] = row["unprocessed"]

        cur.execute(
            "SELECT status, COUNT(*) as cnt FROM articles GROUP BY status"
        )
        stats["articles_by_status"] = {row["status"]: row["cnt"] for row in cur.fetchall()}

        cur.execute(
            "SELECT COUNT(*) as total FROM articles WHERE published_at >= NOW() - INTERVAL '24 hours'"
        )
        stats["published_last_24h"] = cur.fetchone()["total"]

        cur.execute(
            "SELECT platform, COUNT(*) as cnt FROM publish_log WHERE status = 'published' GROUP BY platform"
        )
        stats["published_by_platform"] = {row["platform"]: row["cnt"] for row in cur.fetchall()}

        return stats
