"""Benchmark: generate 5 articles with sectional writer on different topics.
No reset between runs — each article takes the next best topic.
"""
import sys
import time
import logging

sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")

from app.factory.orchestrator import generate_one, ensure_ollama_alive
from app.factory.db import get_article

NUM = 5
print(f"=== SECTIONAL WRITER BENCHMARK: {NUM} articles ===\n")

for i in range(NUM):
    print(f"--- Article {i+1}/{NUM} ---")
    # Health check before each article
    if not ensure_ollama_alive():
        print("  WARNING: Ollama not responding, waiting 30s...")
        time.sleep(30)
        ensure_ollama_alive()
    start = time.time()
    aid = generate_one(preset="habr")
    elapsed = time.time() - start

    if aid:
        a = get_article(aid)
        title = (a.get("title_ru") or "")[:70]
        chars = a.get("char_count", 0)
        score = a.get("fact_check_score", 0)
        status = a.get("status", "?")
        log = a.get("generation_log") or []
        validate = [l for l in log if "VALIDATE" in str(l) or "ENRICH" in str(l)]

        print(f"  [{status}] id={aid} | {chars} chars | score={score} | {elapsed:.0f}s")
        print(f"  {title}")
        for v in validate:
            print(f"  {v}")
    else:
        print(f"  FAILED ({elapsed:.0f}s)")
    print()

print("=== DONE ===")
