"""Text analysis tool for CrewAI agents."""

from crewai.tools import tool


@tool("analyze_text")
def analyze_text(text: str) -> str:
    """Analyze text for basic metrics: length, word count, paragraph count, readability.
    Use this to check if the article meets the target format requirements.

    Args:
        text: The text to analyze.
    """
    chars = len(text)
    words = len(text.split())
    paragraphs = len([p for p in text.split("\n\n") if p.strip()])
    sentences = len([s for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()])

    avg_sentence_len = words / max(sentences, 1)
    avg_paragraph_len = words / max(paragraphs, 1)

    return (
        f"Characters: {chars}\n"
        f"Words: {words}\n"
        f"Sentences: {sentences}\n"
        f"Paragraphs: {paragraphs}\n"
        f"Avg sentence length: {avg_sentence_len:.1f} words\n"
        f"Avg paragraph length: {avg_paragraph_len:.1f} words\n"
        f"Estimated reading time: {words // 200} min"
    )
