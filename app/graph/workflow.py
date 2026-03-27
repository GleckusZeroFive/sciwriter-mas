"""LangGraph StateGraph assembly — the editorial pipeline v3.

Pipeline: research → write → rate → improve → final_rate →(loop or publish)

Each node does ONE thing (important for 8B models):
- Research: extract numbered list of facts
- Writer: write article from facts
- Rater: structured checklist per section
- Improver: rewrite based on checklist
- Final Rater: score 0-100, loop if below threshold (max 2 cycles)
- Publish: save to DB
"""

import logging

from langgraph.graph import StateGraph, END

from app.graph.state import ArticleState
from app.graph.nodes import (
    research_node,
    write_node,
    enrich_node,
    rate_node,
    validate_numbers_node,
    improve_node,
    final_rate_node,
    publish_node,
)
from app.config import settings

logger = logging.getLogger(__name__)


def _route_after_final_rate(state: ArticleState) -> str:
    """Route after final rating: accept → publish, revise → improve again."""
    verdict = state.get("review_verdict", "accept")
    if verdict == "accept":
        return "publish"
    else:
        return "improve"  # loop back


def build_workflow() -> StateGraph:
    """Build the editorial workflow graph.

    Flow:
        research → write → rate → improve → final_rate
            ↓ accept → publish → END
            ↓ revise → improve (loop, max 2 times via revision_count)
    """
    graph = StateGraph(ArticleState)

    # Add nodes
    graph.add_node("research", research_node)
    graph.add_node("write", write_node)
    graph.add_node("enrich", enrich_node)
    graph.add_node("rate", rate_node)
    graph.add_node("validate_numbers", validate_numbers_node)
    graph.add_node("improve", improve_node)
    graph.add_node("final_rate", final_rate_node)
    graph.add_node("publish", publish_node)

    # Linear: research → write → enrich → rate → validate_numbers → improve → final_rate
    graph.add_edge("research", "write")
    graph.add_edge("write", "enrich")
    graph.add_edge("enrich", "rate")
    graph.add_edge("rate", "validate_numbers")
    graph.add_edge("validate_numbers", "improve")
    graph.add_edge("improve", "final_rate")

    # Conditional: final_rate → publish (accept) or improve (revise)
    graph.add_conditional_edges(
        "final_rate",
        _route_after_final_rate,
        {
            "publish": "publish",
            "improve": "improve",
        },
    )

    # publish → END
    graph.add_edge("publish", END)

    # Entry point
    graph.set_entry_point("research")

    return graph


def create_pipeline():
    """Create a compiled, ready-to-run pipeline."""
    graph = build_workflow()
    return graph.compile()


def run_pipeline(topic: str, preset: str = "habr") -> ArticleState:
    """Run the full editorial pipeline synchronously."""
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

    logger.info("Starting pipeline v3: topic='%s', preset='%s'", topic, preset)
    result = pipeline.invoke(initial_state)
    logger.info("Pipeline complete: status=%s", result.get("status"))

    return result
