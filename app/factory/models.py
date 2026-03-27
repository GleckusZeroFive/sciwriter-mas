"""Data models for the content factory."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class RawItem:
    """A raw item collected from an external source."""
    source: str              # 'reddit', 'hackernews', 'arxiv', 'techcrunch', 'duckduckgo'
    source_id: str           # external ID for dedup
    title: str
    url: Optional[str] = None
    summary: Optional[str] = None
    content: Optional[str] = None
    score: float = 0.0
    tags: list[str] = field(default_factory=list)
    language: str = "en"
    collected_at: Optional[datetime] = None
    id: Optional[int] = None  # DB id after insert


@dataclass
class Article:
    """A generated article in the pipeline."""
    id: Optional[int] = None
    title_ru: Optional[str] = None
    content_ru: Optional[str] = None
    status: str = "queued"   # queued → generating → quality_check → ready → publishing → published → failed
    source_item_ids: list[int] = field(default_factory=list)
    topic_summary: Optional[str] = None
    fact_check_score: Optional[float] = None
    char_count: Optional[int] = None
    revision_count: int = 0
    generation_log: list[dict] = field(default_factory=list)
    media: list[dict] = field(default_factory=list)  # [{path, type, caption}]
    created_at: Optional[datetime] = None
    published_at: Optional[datetime] = None
    habr_url: Optional[str] = None
    dzen_url: Optional[str] = None
    error: Optional[str] = None


@dataclass
class PublishRecord:
    """A publishing log entry."""
    article_id: int
    platform: str            # 'habr' or 'dzen'
    status: str = "pending"  # pending → publishing → published → failed
    scheduled_at: Optional[datetime] = None
    published_at: Optional[datetime] = None
    url: Optional[str] = None
    screenshot_path: Optional[str] = None
    error: Optional[str] = None
    id: Optional[int] = None
