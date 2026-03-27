"""Reviewer agent — reads the article, rates it, explains the rating, and improves it.

Key design: the prompt is NEUTRAL. It does NOT say "find errors" or "fix problems".
This prevents the model from hallucinating issues that don't exist.
"""

from crewai import Agent

from app.tools.text_analysis import analyze_text


def create_reviewer(llm) -> Agent:
    return Agent(
        role="Article Reviewer",
        goal=(
            "Read the article carefully. "
            "Rate it from 0 to 100. "
            "Explain why you gave that rating. "
            "Then produce an improved version of the article."
        ),
        backstory=(
            "You are an experienced editor at a Russian tech publication. "
            "You read articles and assess their overall quality — "
            "clarity, accuracy, depth of detail, readability, structure. "
            "You give an honest rating and then improve the article where you see room for improvement. "
            "You preserve the author's style and factual content. "
            "You write in Russian."
        ),
        tools=[analyze_text],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
