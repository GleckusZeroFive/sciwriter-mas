-- SciWriter MAS — Content Factory Database Schema
-- Created: 2026-03-27

-- Raw items collected from external sources
CREATE TABLE IF NOT EXISTS raw_items (
    id SERIAL PRIMARY KEY,
    source VARCHAR(50) NOT NULL,            -- 'reddit', 'hackernews', 'arxiv', 'techcrunch', 'duckduckgo'
    source_id VARCHAR(255),                 -- external ID for source-level dedup
    url TEXT,
    title TEXT NOT NULL,
    summary TEXT,                            -- snippet / abstract
    content TEXT,                            -- full text if available
    score FLOAT DEFAULT 0,                  -- engagement score (upvotes, points, etc.)
    tags TEXT[] DEFAULT '{}',               -- topic tags
    language VARCHAR(10) DEFAULT 'en',
    collected_at TIMESTAMPTZ DEFAULT NOW(),
    processed BOOLEAN DEFAULT FALSE,
    UNIQUE(source, source_id)
);

-- Generated articles in the pipeline
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    title_ru TEXT,
    content_ru TEXT,                         -- final markdown
    status VARCHAR(30) DEFAULT 'queued',     -- queued → generating → quality_check → ready → publishing → published → failed
    source_item_ids INTEGER[] DEFAULT '{}',  -- FK references to raw_items used
    topic_summary TEXT,                      -- combined topic description for generation
    fact_check_score FLOAT,
    char_count INTEGER,
    revision_count INTEGER DEFAULT 0,
    generation_log JSONB DEFAULT '[]',       -- full pipeline log
    media JSONB DEFAULT '[]',               -- [{path, type, caption}] for images/charts
    created_at TIMESTAMPTZ DEFAULT NOW(),
    published_at TIMESTAMPTZ,
    habr_url TEXT,
    dzen_url TEXT,
    error TEXT
);

-- Publishing schedule and tracking
CREATE TABLE IF NOT EXISTS publish_log (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    platform VARCHAR(20) NOT NULL,           -- 'habr' or 'dzen'
    status VARCHAR(20) DEFAULT 'pending',    -- pending → publishing → published → failed
    scheduled_at TIMESTAMPTZ,
    published_at TIMESTAMPTZ,
    url TEXT,
    screenshot_path TEXT,                    -- path to post-publish screenshot
    error TEXT
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_raw_items_unprocessed ON raw_items(processed, collected_at) WHERE NOT processed;
CREATE INDEX IF NOT EXISTS idx_raw_items_source ON raw_items(source);
CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(status);
CREATE INDEX IF NOT EXISTS idx_articles_created ON articles(created_at);
CREATE INDEX IF NOT EXISTS idx_publish_log_pending ON publish_log(scheduled_at) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_publish_log_article ON publish_log(article_id);
