"""Web search tool for CrewAI agents using DuckDuckGo."""

import logging

from crewai.tools import tool

logger = logging.getLogger(__name__)


@tool("web_search")
def web_search(query: str) -> str:
    """Search the web for information on a topic.
    Returns relevant search results with titles, URLs, and snippets.
    Use this to find recent information, statistics, and sources.

    Args:
        query: Search query string.
    """
    from duckduckgo_search import DDGS
    from app.config import settings

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                max_results=settings.web_search_max_results,
            ))
    except Exception as e:
        logger.error("Web search failed: %s", e)
        return f"Web search failed: {e}"

    if not results:
        return "No results found."

    output = []
    for r in results:
        title = r.get("title", "")
        url = r.get("href", "")
        snippet = r.get("body", "")
        output.append(f"**{title}**\n{url}\n{snippet}")

    return "\n\n---\n\n".join(output)
