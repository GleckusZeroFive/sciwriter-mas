"""Conditional edge functions for LangGraph workflow routing."""

from app.graph.state import ArticleState


def route_after_review(state: ArticleState) -> str:
    """Route after review gate: accept → edit, revise → rewrite, reject → end."""
    verdict = state.get("review_verdict", "accept")

    if verdict == "accept":
        return "edit"
    elif verdict == "revise":
        return "write"  # loop back for revision
    else:
        return "end"
