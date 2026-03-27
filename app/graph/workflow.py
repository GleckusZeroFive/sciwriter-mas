"""LangGraph StateGraph assembly — the editorial pipeline."""

import logging

from langgraph.graph import StateGraph, END

from app.graph.state import ArticleState
from app.graph.nodes import (
    research_node,
    write_node,
    fact_check_node,
    review_gate_node,
    edit_node,
    publish_node,
)
from app.graph.edges import route_after_review
from app.config import settings

logger = logging.getLogger(__name__)


def build_workflow() -> StateGraph:
    """Build and compile the editorial workflow graph.

    Flow:
        research → write → fact_check → review_gate
            ↓ accept → edit → publish
            ↓ revise → write (loop, max N times)
    """
    graph = StateGraph(ArticleState)

    # Add nodes
    graph.add_node("research", research_node)
    graph.add_node("write", write_node)
    graph.add_node("fact_check", fact_check_node)
    graph.add_node("review_gate", review_gate_node)
    graph.add_node("edit", edit_node)
    graph.add_node("publish", publish_node)

    # Linear flow: research → write → fact_check → review_gate
    graph.add_edge("research", "write")
    graph.add_edge("write", "fact_check")
    graph.add_edge("fact_check", "review_gate")

    # Conditional: review_gate → edit (accept) or write (revise)
    graph.add_conditional_edges(
        "review_gate",
        route_after_review,
        {
            "edit": "edit",
            "write": "write",
            "end": END,
        },
    )

    # edit → publish → END
    graph.add_edge("edit", "publish")
    graph.add_edge("publish", END)

    # Entry point
    graph.set_entry_point("research")

    return graph


def create_pipeline():
    """Create a compiled, ready-to-run pipeline."""
    graph = build_workflow()
    return graph.compile()


def run_pipeline(topic: str, preset: str = "habr") -> ArticleState:
    """Run the full editorial pipeline synchronously.

    Args:
        topic: Article topic.
        preset: Format preset name ("habr" or "dzen").

    Returns:
        Final state with the published article.
    """
    pipeline = create_pipeline()

    initial_state: ArticleState = {
        "topic": topic,
        "preset": preset,
        "keywords": [],
        "revision_count": 0,
        "max_revisions": settings.max_revisions,
        "draft_version": 0,
        "status": "researching",
        "log": [f"[START] Topic: {topic}, Preset: {preset}"],
    }

    logger.info("Starting pipeline: topic='%s', preset='%s'", topic, preset)
    result = pipeline.invoke(initial_state)
    logger.info("Pipeline complete: status=%s", result.get("status"))

    return result
