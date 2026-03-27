"""Researcher agent — finds sources on a given topic via web search + RAG."""

from crewai import Agent

from app.tools.web_search import web_search
from app.tools.rag_search import knowledge_base_search


def create_researcher(llm) -> Agent:
    return Agent(
        role="Scientific Researcher",
        goal=(
            "Find reliable, factual sources on the given topic. "
            "Combine web search results with the local knowledge base. "
            "Prioritize peer-reviewed publications, official statistics, "
            "and reputable scientific sources."
        ),
        backstory=(
            "You are a thorough research assistant at a scientific journal. "
            "You know how to distinguish credible sources from unreliable ones. "
            "You always provide specific facts, numbers, and citations. "
            "You work with Russian-language scientific content."
        ),
        tools=[web_search, knowledge_base_search],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
