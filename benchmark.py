"""Benchmark: generate multiple articles on different topics, collect stats."""

import sys
import time
import logging

sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")

from app.factory.orchestrator import generate_one
from app.factory.db import get_article, get_factory_stats

NUM_ARTICLES = 5

print(f"=== BENCHMARK: generating {NUM_ARTICLES} articles ===\n")

results = []
for i in range(NUM_ARTICLES):
    print(f"--- Article {i+1}/{NUM_ARTICLES} ---")
    start = time.time()
    article_id = generate_one(preset="habr")
    elapsed = time.time() - start

    if article_id:
        a = get_article(article_id)
        title = (a.get("title_ru") or "")[:80]
        chars = a.get("char_count", 0)
        score = a.get("fact_check_score", 0)
        status = a.get("status", "?")
        log = a.get("generation_log") or []

        # Extract validate stats from log
        validate_line = [l for l in log if "[VALIDATE]" in l]
        publish_line = [l for l in log if "Final validator" in str(l)]

        results.append({
            "id": article_id,
            "title": title,
            "chars": chars,
            "score": score,
            "status": status,
            "time": elapsed,
            "validate": validate_line,
            "publish_validate": publish_line,
        })
        print(f"  ID={article_id} | {status} | {chars} chars | score={score} | {elapsed:.0f}s")
        print(f"  Title: {title}")
        for v in validate_line:
            print(f"  {v}")
    else:
        print(f"  FAILED (no topics or pipeline error)")
        results.append({"id": None, "status": "no_topic", "time": elapsed})

    print()

# Summary
print("=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)
ok = [r for r in results if r.get("status") == "ready"]
failed = [r for r in results if r.get("status") != "ready"]
print(f"Total: {len(results)} | Ready: {len(ok)} | Failed: {len(failed)}")
if ok:
    avg_chars = sum(r["chars"] for r in ok) / len(ok)
    avg_score = sum(r["score"] for r in ok) / len(ok)
    avg_time = sum(r["time"] for r in ok) / len(ok)
    print(f"Avg chars: {avg_chars:.0f} | Avg score: {avg_score:.1f} | Avg time: {avg_time:.0f}s")

for r in results:
    status_icon = "OK" if r.get("status") == "ready" else "FAIL"
    print(f"  [{status_icon}] id={r.get('id')} | {r.get('chars', 0)} chars | score={r.get('score', 0)} | {r.get('time', 0):.0f}s | {r.get('title', '')[:60]}")

stats = get_factory_stats()
print(f"\nDB stats: {stats}")
