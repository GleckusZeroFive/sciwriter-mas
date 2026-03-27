"""Orchestrator — main loop of the content factory.

Coordinates: collect → rank → generate → quality check → publish.
Can run as one-shot (CLI) or continuous (APScheduler).
"""

import logging
import time
from datetime import datetime, timezone, timedelta

from app.config import settings
from app.factory.collectors.hackernews import HackerNewsCollector
from app.factory.collectors.arxiv import ArxivCollector
from app.factory.collectors.techcrunch import TechCrunchCollector
from app.factory.db import (
    insert_raw_items_batch,
    create_article,
    update_article,
    get_article,
    mark_items_processed,
    get_articles_by_status,
    get_factory_stats,
    schedule_publish,
)
from app.factory.topic_ranker import select_best_topic
from app.factory.quality_gate import check_level1, clean_artifacts, build_llm_verification_prompt

logger = logging.getLogger(__name__)


def collect_all() -> int:
    """Run all available collectors. Returns total new items inserted."""
    collectors = [
        HackerNewsCollector(),
        ArxivCollector(),
        TechCrunchCollector(),
    ]

    # Add Reddit if credentials are configured
    if settings.factory_reddit_client_id:
        from app.factory.collectors.reddit import RedditCollector
        collectors.append(RedditCollector())

    total_new = 0
    for collector in collectors:
        items = collector.run()
        if items:
            new = insert_raw_items_batch(items)
            total_new += new
            logger.info("[COLLECT] %s: %d collected, %d new", collector.name, len(items), new)

    logger.info("[COLLECT] Total new items: %d", total_new)
    return total_new


def generate_one(preset: str = "habr") -> int | None:
    """Generate one article from the best available topic.

    Returns article_db_id on success, None on failure.
    """
    # 1. Select best topic
    bundle = select_best_topic()
    if not bundle:
        logger.warning("[GENERATE] No topics available")
        return None

    logger.info("[GENERATE] Selected topic: %s (%d sources)", bundle.title_suggestion[:60], len(bundle.items))

    # 2. Create article record in DB
    article_id = create_article(
        source_item_ids=bundle.item_ids,
        topic_summary=bundle.combined_summary[:2000],
        status="generating",
    )

    # 3. Mark source items as processed
    mark_items_processed(bundle.item_ids)

    # 4. Enrich sources with full article content
    from app.factory.content_extractor import enrich_items_with_content
    enriched = enrich_items_with_content(bundle.items, max_fetch=5)
    logger.info("[GENERATE] Enriched %d items with full article content", enriched)

    # 5. Build pre_sources for the pipeline
    pre_sources = [
        {
            "source": item["source"],
            "title": item["title"],
            "url": item.get("url", ""),
            "summary": item.get("summary", ""),
            "content": item.get("content", ""),
        }
        for item in bundle.items
    ]

    # 5. Run the pipeline with timeout
    try:
        from app.graph.workflow import create_pipeline
        import threading

        workflow = create_pipeline()
        initial_state = {
            "topic": bundle.title_suggestion,
            "preset": preset,
            "pre_sources": pre_sources,
            "article_db_id": article_id,
            "max_revisions": settings.max_revisions,
        }

        logger.info("[GENERATE] Starting pipeline for article id=%d...", article_id)
        start_time = time.time()

        # Run pipeline with timeout (15 min max)
        result = [None]
        error_box = [None]

        def _run():
            try:
                result[0] = workflow.invoke(initial_state)
            except Exception as e:
                error_box[0] = e

        thread = threading.Thread(target=_run)
        thread.start()
        thread.join(timeout=900)  # 15 min

        if thread.is_alive():
            logger.error("[GENERATE] Pipeline TIMEOUT (15 min) for article id=%d", article_id)
            # Check if publish_node managed to save before timeout
            existing = get_article(article_id)
            if existing and existing.get("content_ru"):
                logger.info("[GENERATE] Partial save found, marking as ready")
                update_article(article_id, status="ready")
                return article_id
            else:
                update_article(article_id, status="failed", error="Pipeline timeout (15 min)")
                return None

        if error_box[0]:
            raise error_box[0]

        elapsed = time.time() - start_time
        logger.info("[GENERATE] Pipeline completed in %.1fs for article id=%d", elapsed, article_id)

        # Check if publish_node already saved the article (factory mode)
        existing = get_article(article_id)
        if existing and existing.get("content_ru") and existing.get("status") in ("ready", "quality_check"):
            logger.info("[GENERATE] Article already saved by publish_node, skipping orchestrator update")
        else:
            # Fallback: save from pipeline result
            final_article = result[0].get("final_article", "") if result[0] else ""
            if final_article:
                cleaned = clean_artifacts(final_article)
                report = check_level1(cleaned, min_length=3000)

                update_article(
                    article_id,
                    content_ru=cleaned,
                    title_ru=cleaned.split("\n")[0].lstrip("# ").strip()[:200],
                    char_count=len(cleaned),
                    fact_check_score=result[0].get("fact_check_score") if result[0] else None,
                    status="ready" if report.passed else "failed",
                    generation_log=result[0].get("log", []) if result[0] else [],
                    error="; ".join(report.issues) if not report.passed else None,
                )

                if report.passed:
                    logger.info("[GENERATE] Article id=%d ready (%d chars)", article_id, len(cleaned))
                else:
                    logger.warning("[GENERATE] Article id=%d failed quality gate: %s",
                                   article_id, "; ".join(report.issues))

        return article_id

    except Exception as e:
        logger.error("[GENERATE] Pipeline failed for article id=%d: %s", article_id, e, exc_info=True)
        update_article(article_id, status="failed", error=str(e)[:500])
        return None


def schedule_ready_articles(platform: str = "both"):
    """Schedule all 'ready' articles for publishing."""
    articles = get_articles_by_status("ready", limit=10)
    if not articles:
        logger.info("[SCHEDULE] No ready articles to schedule")
        return

    now = datetime.now(timezone.utc)
    platforms = ["habr", "dzen"] if platform == "both" else [platform]

    for i, article in enumerate(articles):
        for p in platforms:
            scheduled_at = now + timedelta(minutes=i * 30)  # Stagger by 30 min
            schedule_publish(article["id"], p, scheduled_at)

        update_article(article["id"], status="publishing")

    logger.info("[SCHEDULE] Scheduled %d articles for %s", len(articles), ", ".join(platforms))


def run_cycle(preset: str = "habr"):
    """Run one full cycle: collect → generate → schedule.

    This is the main entry point for both CLI and scheduler.
    """
    logger.info("=" * 60)
    logger.info("[CYCLE] Starting content factory cycle")
    logger.info("=" * 60)

    # 1. Collect
    new_items = collect_all()
    logger.info("[CYCLE] Collected %d new items", new_items)

    # 2. Generate
    article_id = generate_one(preset=preset)
    if article_id:
        logger.info("[CYCLE] Generated article id=%d", article_id)
    else:
        logger.warning("[CYCLE] No article generated")

    # 3. Schedule ready articles
    schedule_ready_articles()

    # 4. Stats
    stats = get_factory_stats()
    logger.info("[CYCLE] Stats: %s", stats)
    logger.info("=" * 60)

    return article_id


def run_continuous():
    """Run the factory in continuous mode with APScheduler.

    Intervals from settings:
    - Collection: every 15 min
    - Generation: every 50 min
    - Publishing: every 60 min
    """
    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler()

    # Collection jobs
    scheduler.add_job(
        collect_all,
        "interval",
        seconds=settings.factory_collect_interval_hn,
        id="collect_all",
        name="Collect from all sources",
        next_run_time=datetime.now(),  # Run immediately on start
    )

    # Generation job
    scheduler.add_job(
        generate_one,
        "interval",
        seconds=settings.factory_generate_interval,
        id="generate_article",
        name="Generate one article",
        kwargs={"preset": settings.article_preset},
    )

    # Schedule ready articles
    scheduler.add_job(
        schedule_ready_articles,
        "interval",
        seconds=settings.factory_publish_interval,
        id="schedule_publish",
        name="Schedule ready articles",
    )

    logger.info("[FACTORY] Starting continuous mode...")
    logger.info("[FACTORY] Collection interval: %ds", settings.factory_collect_interval_hn)
    logger.info("[FACTORY] Generation interval: %ds", settings.factory_generate_interval)
    logger.info("[FACTORY] Publish interval: %ds", settings.factory_publish_interval)

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("[FACTORY] Shutting down...")
        scheduler.shutdown()
