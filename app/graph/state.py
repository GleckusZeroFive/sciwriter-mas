"""LangGraph state definition for the editorial workflow."""

from typing import TypedDict


class ArticleState(TypedDict, total=False):
    # Input
    topic: str
    preset: str  # "habr" or "dzen"
    keywords: list[str]

    # Research phase
    sources: str  # concatenated research results
    source_count: int

    # Writing phase
    draft: str  # current article text
    draft_version: int

    # Fact-checking phase
    fact_check_report: str
    fact_check_score: float  # 1-10

    # Review gate
    review_verdict: str  # "accept" | "revise" | "reject"
    revision_count: int
    max_revisions: int

    # Editing phase
    edited_text: str
    edit_changelog: str

    # Final output
    final_article: str
    status: str  # "researching" | "writing" | "fact_checking" | "reviewing" | "editing" | "published" | "rejected"

    # Audit trail
    log: list[str]
