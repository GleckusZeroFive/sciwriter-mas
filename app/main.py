"""FastAPI application and CLI entry point for SciWriter MAS."""

import argparse
import logging
import sys
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

from app.config import settings

# --- FastAPI App ---

app = FastAPI(
    title="SciWriter MAS",
    description="Multi-agent system for scientific article generation",
    version="1.0.0",
)


class GenerateRequest(BaseModel):
    topic: str
    preset: str = "habr"


class ArticleResponse(BaseModel):
    topic: str
    preset: str
    status: str
    article: str | None = None
    fact_check_score: float | None = None
    revision_count: int = 0
    log: list[str] = []


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate", response_model=ArticleResponse)
async def generate_article(request: GenerateRequest):
    """Generate an article (synchronous — blocks until complete)."""
    from app.graph.workflow import run_pipeline

    result = run_pipeline(topic=request.topic, preset=request.preset)

    return ArticleResponse(
        topic=result.get("topic", request.topic),
        preset=result.get("preset", request.preset),
        status=result.get("status", "unknown"),
        article=result.get("final_article"),
        fact_check_score=result.get("fact_check_score"),
        revision_count=result.get("revision_count", 0),
        log=result.get("log", []),
    )


# --- CLI ---

def cli():
    """Command-line interface for running the pipeline."""
    parser = argparse.ArgumentParser(description="SciWriter MAS — scientific article generator")
    subparsers = parser.add_subparsers(dest="command")

    # Generate
    gen_parser = subparsers.add_parser("generate", help="Generate an article")
    gen_parser.add_argument("topic", help="Article topic")
    gen_parser.add_argument("--preset", default="habr", choices=["habr", "dzen"], help="Format preset")
    gen_parser.add_argument("--output", "-o", help="Output file path (.md)")

    # Index
    idx_parser = subparsers.add_parser("index", help="Index documents into knowledge base")
    idx_parser.add_argument("path", nargs="?", default=str(settings.data_dir / "knowledge_base"),
                           help="Directory with .txt/.md files")
    idx_parser.add_argument("--recreate", action="store_true", help="Recreate collection from scratch")

    # Server
    srv_parser = subparsers.add_parser("serve", help="Start FastAPI server")
    srv_parser.add_argument("--host", default=settings.api_host)
    srv_parser.add_argument("--port", type=int, default=settings.api_port)

    # Factory commands
    factory_parser = subparsers.add_parser("factory", help="Content factory operations")
    factory_sub = factory_parser.add_subparsers(dest="factory_cmd")

    factory_sub.add_parser("collect", help="Run all collectors once")
    factory_sub.add_parser("stats", help="Show factory statistics")

    fgen = factory_sub.add_parser("generate", help="Generate one article from collected data")
    fgen.add_argument("--preset", default="habr", choices=["habr", "dzen"])

    factory_sub.add_parser("cycle", help="Run one full cycle (collect → generate → schedule)")
    factory_sub.add_parser("run", help="Start continuous factory with scheduler")

    args = parser.parse_args()

    if args.command == "generate":
        from app.graph.workflow import run_pipeline

        print(f"Generating article: '{args.topic}' (preset: {args.preset})")
        print("=" * 60)

        result = run_pipeline(topic=args.topic, preset=args.preset)

        article = result.get("final_article", "No article generated")
        score = result.get("fact_check_score", 0)
        revisions = result.get("revision_count", 0)

        print("\n" + "=" * 60)
        print(f"Status: {result.get('status')}")
        print(f"Fact-check score: {score}/10")
        print(f"Revisions: {revisions}")
        print("=" * 60)
        print(article)

        if args.output:
            Path(args.output).write_text(article, encoding="utf-8")
            print(f"\nSaved to: {args.output}")

        # Print log
        print("\n--- Pipeline Log ---")
        for entry in result.get("log", []):
            print(f"  {entry}")

    elif args.command == "index":
        from app.rag.indexer import index_directory
        count = index_directory(args.path, recreate=args.recreate)
        print(f"Indexed {count} chunks")

    elif args.command == "serve":
        import uvicorn
        uvicorn.run("app.main:app", host=args.host, port=args.port, reload=True)

    elif args.command == "factory":
        from app.factory.orchestrator import collect_all, generate_one, run_cycle, run_continuous
        from app.factory.db import get_factory_stats

        if args.factory_cmd == "collect":
            new = collect_all()
            print(f"Collected {new} new items")

        elif args.factory_cmd == "stats":
            stats = get_factory_stats()
            print("=== Factory Stats ===")
            print(f"Raw items: {stats['raw_items_total']} total, {stats['raw_items_unprocessed']} unprocessed")
            print(f"Articles: {stats['articles_by_status']}")
            print(f"Published (24h): {stats['published_last_24h']}")
            print(f"By platform: {stats['published_by_platform']}")

        elif args.factory_cmd == "generate":
            article_id = generate_one(preset=args.preset)
            if article_id:
                print(f"Generated article id={article_id}")
            else:
                print("No article generated (no topics available or pipeline failed)")

        elif args.factory_cmd == "cycle":
            run_cycle()

        elif args.factory_cmd == "run":
            run_continuous()

        else:
            factory_parser.print_help()

    else:
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    cli()
