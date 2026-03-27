"""Writer agent — generates article drafts based on research sources and preset format."""

from crewai import Agent

from app.tools.text_analysis import analyze_text


def create_writer(llm) -> Agent:
    return Agent(
        role="Scientific Article Writer",
        goal=(
            "Write a well-structured, engaging article in Russian "
            "based on the provided research sources. "
            "Follow the specified format (Habr technical or Dzen popular science). "
            "Every claim must be supported by the sources provided."
        ),
        backstory=(
            "You are an experienced science writer for a Russian-language journal. "
            "You can explain complex scientific topics clearly while maintaining accuracy. "
            "You write in Markdown format with proper headings, lists, and emphasis. "
            "You never fabricate facts — only use information from the provided sources."
        ),
        tools=[analyze_text],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
