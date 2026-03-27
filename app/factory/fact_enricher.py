"""Fact Enricher — verifies and enriches named entities via web search.

When the model writes "ARM Cortex-A55" or "1440x900", this module:
1. Extracts named technical entities (chip names, protocols, specs)
2. Searches the web for "<topic> + <entity>"
3. If search confirms the entity — keeps it (optionally enriches with details)
4. If search contradicts — removes or replaces with verified info
5. If no results — removes the claim

Uses DuckDuckGo search (already in project dependencies).
"""

import logging
import re
import time

logger = logging.getLogger(__name__)

# Patterns for extractable technical entities
ENTITY_PATTERNS = [
    # Processor/chip names: ARM Cortex-A55, NVIDIA Tegra, Intel i7, M2 Ultra
    re.compile(r"\b(ARM\s+Cortex[\w-]*|NVIDIA\s+\w+|Intel\s+\w[\w-]*|Qualcomm\s+\w+|AMD\s+\w+|M\d\s+\w+|Snapdragon\s+\d+|Tegra\s+\w+)", re.IGNORECASE),
    # Specific chip models: MAX16932, LM2596, etc.
    re.compile(r"\b([A-Z]{2,}\d{3,}[A-Z]*(?:/[\w+]+)?)\b"),
    # Protocols/interfaces: MIPI DSI, PCIe, LVDS, JTAG, UART
    re.compile(r"\b(MIPI\s+DSI|PCIe|LVDS|JTAG|UART|USB-C|HDMI|NVMe|OpenOCD)\b", re.IGNORECASE),
    # Display specs: 1440x900, 1920x1080, etc.
    re.compile(r"\b(\d{3,4}\s*[xх×]\s*\d{3,4})\b"),
    # Memory/storage specs with specific values: 64 ГБ NAND, 4 ГБ RAM
    re.compile(r"\b(\d+\s*(?:ГБ|МБ|GB|MB|TB)\s+(?:RAM|NAND|SSD|ROM|DDR\d?|LPDDR\d?))\b", re.IGNORECASE),
]


def extract_entities(text: str) -> list[dict]:
    """Extract technical entities from article text.

    Returns list of {entity, context, line_num} dicts.
    """
    results = []
    seen = set()

    for line_num, line in enumerate(text.split("\n"), 1):
        for pattern in ENTITY_PATTERNS:
            for match in pattern.finditer(line):
                entity = match.group(1).strip()
                if entity.lower() in seen:
                    continue
                seen.add(entity.lower())

                # Get context (surrounding text)
                start = max(0, match.start() - 40)
                end = min(len(line), match.end() + 40)
                context = line[start:end].strip()

                results.append({
                    "entity": entity,
                    "context": context,
                    "line": line_num,
                    "full_line": line.strip(),
                })

    return results


def search_entity(topic: str, entity: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo for topic + entity. Returns list of {title, body, href}."""
    try:
        from duckduckgo_search import DDGS
        query = f"{topic} {entity}"
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as e:
        logger.debug("Search failed for '%s': %s", entity, e)
        return []


def verify_entity(topic: str, entity: str, context: str) -> dict:
    """Verify a single entity via web search.

    Returns {
        entity: str,
        status: "confirmed" | "contradicted" | "unverified",
        evidence: str | None,  # what the search found
        replacement: str | None,  # correct info if contradicted
    }
    """
    results = search_entity(topic, entity)

    if not results:
        return {
            "entity": entity,
            "status": "unverified",
            "evidence": None,
            "replacement": None,
        }

    # Check if entity appears in search results
    entity_lower = entity.lower()
    combined_text = " ".join(
        (r.get("title", "") + " " + r.get("body", "")).lower()
        for r in results
    )

    if entity_lower in combined_text:
        # Entity confirmed by search
        # Try to extract additional details
        evidence = results[0].get("body", "")[:200] if results else None
        return {
            "entity": entity,
            "status": "confirmed",
            "evidence": evidence,
            "replacement": None,
        }
    else:
        # Entity not found in search — likely hallucinated
        # Check if search suggests a different entity
        return {
            "entity": entity,
            "status": "contradicted",
            "evidence": results[0].get("body", "")[:200] if results else None,
            "replacement": None,
        }


def enrich_article(text: str, topic: str, max_lookups: int = 10) -> tuple[str, list[dict]]:
    """Verify and enrich technical entities in article text.

    For each entity:
    - confirmed → keep (optionally add details)
    - contradicted → remove the claim from text
    - unverified → remove the claim from text

    Returns (cleaned_text, verification_log).
    """
    entities = extract_entities(text)
    if not entities:
        logger.info("[ENRICHER] No technical entities found")
        return text, []

    logger.info("[ENRICHER] Found %d entities to verify", len(entities))

    verification_log = []
    lines_to_remove = set()

    for i, ent in enumerate(entities[:max_lookups]):
        logger.info("[ENRICHER] Verifying %d/%d: '%s'", i+1, len(entities), ent["entity"])
        result = verify_entity(topic, ent["entity"], ent["context"])
        result["line"] = ent["line"]
        result["context"] = ent["context"]
        verification_log.append(result)

        if result["status"] == "contradicted":
            # Remove the line with unverified entity
            lines_to_remove.add(ent["line"])
            logger.info("[ENRICHER] CONTRADICTED: '%s' — removing line %d",
                       ent["entity"], ent["line"])
        elif result["status"] == "unverified":
            lines_to_remove.add(ent["line"])
            logger.info("[ENRICHER] UNVERIFIED: '%s' — removing line %d",
                       ent["entity"], ent["line"])
        else:
            logger.info("[ENRICHER] CONFIRMED: '%s'", ent["entity"])

        # Rate limit: don't hammer DuckDuckGo
        time.sleep(1)

    # Remove lines with unverified/contradicted entities
    if lines_to_remove:
        lines = text.split("\n")
        cleaned_lines = []
        for i, line in enumerate(lines, 1):
            if i in lines_to_remove:
                # Don't remove headings or structural elements
                if line.strip().startswith("#") or not line.strip():
                    cleaned_lines.append(line)
                else:
                    logger.debug("[ENRICHER] Removed line %d: %s", i, line[:80])
            else:
                cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)
        # Clean up double empty lines
        text = re.sub(r"\n{3,}", "\n\n", text)

    confirmed = sum(1 for v in verification_log if v["status"] == "confirmed")
    removed = len(lines_to_remove)
    logger.info("[ENRICHER] Results: %d confirmed, %d removed, %d total",
               confirmed, removed, len(verification_log))

    return text, verification_log
