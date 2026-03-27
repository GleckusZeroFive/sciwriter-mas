"""Fact-Checker agent — verifies claims in the article against sources."""

from crewai import Agent

from app.tools.web_search import web_search
from app.tools.rag_search import knowledge_base_search


def create_fact_checker(llm) -> Agent:
    return Agent(
        role="Scientific Fact-Checker",
        goal=(
            "Verify every factual claim in the article. "
            "Check numbers, dates, names, and scientific statements "
            "against the knowledge base and web sources. "
            "Produce a structured verification report with scores."
        ),
        backstory=(
            "You are a meticulous fact-checker at a scientific publication. "
            "You never let unverified claims pass. "
            "For each claim you check, you assign a status: "
            "CONFIRMED (found supporting evidence), "
            "UNCONFIRMED (no evidence found), or "
            "CONTRADICTED (found contradicting evidence). "
            "You provide an overall accuracy score from 1 to 10."
        ),
        tools=[web_search, knowledge_base_search],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
