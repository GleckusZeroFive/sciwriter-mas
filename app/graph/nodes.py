"""LangGraph node functions — each node calls a CrewAI agent or runs logic."""

import logging
import yaml
from pathlib import Path

import os

from crewai import Task, Crew, Process, LLM

from app.config import settings
from app.graph.state import ArticleState
from app.agents.researcher import create_researcher
from app.agents.writer import create_writer
from app.agents.fact_checker import create_fact_checker
from app.agents.editor import create_editor

logger = logging.getLogger(__name__)


def _get_llm() -> LLM:
    # CrewAI natively supports Ollama via 'ollama/' prefix
    return LLM(
        model=f"ollama/{settings.llm_model}",
        base_url=settings.llm_base_url.replace("/v1", ""),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


def _load_preset(name: str) -> dict:
    preset_path = settings.presets_dir / f"{name}.yml"
    if not preset_path.exists():
        logger.warning("Preset %s not found, using defaults", name)
        return {}
    with open(preset_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _add_log(state: dict, message: str) -> list[str]:
    log = list(state.get("log") or [])
    log.append(message)
    return log


# --- Node: Research ---

def research_node(state: ArticleState) -> dict:
    """Researcher agent finds sources on the topic."""
    topic = state["topic"]
    preset = _load_preset(state.get("preset", "habr"))
    llm = _get_llm()

    agent = create_researcher(llm)
    task = Task(
        description=(
            f"Research the topic: '{topic}'\n\n"
            f"Find 5-10 reliable sources with specific facts, numbers, and quotes.\n"
            f"Search both the web and the local knowledge base.\n"
            f"Focus on: recent developments, key statistics, expert opinions.\n"
            f"Language: Russian.\n"
            f"Target audience: {preset.get('audience', 'general readers')}"
        ),
        expected_output=(
            "A structured list of sources with:\n"
            "- Source title and URL (if available)\n"
            "- Key facts and quotes from each source\n"
            "- Relevance to the topic\n"
            "At least 5 sources with specific data points."
        ),
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    result = crew.kickoff()

    return {
        "sources": str(result),
        "source_count": str(result).count("---") + 1,
        "status": "writing",
        "log": _add_log(state, f"[RESEARCH] Found sources on: {topic}"),
    }


# --- Node: Write ---

def write_node(state: ArticleState) -> dict:
    """Writer agent generates article draft from sources."""
    topic = state["topic"]
    sources = state.get("sources", "")
    preset = _load_preset(state.get("preset", "habr"))
    revision = state.get("revision_count", 0)
    fact_report = state.get("fact_check_report", "")
    llm = _get_llm()

    revision_context = ""
    if revision > 0 and fact_report:
        revision_context = (
            f"\n\nThis is revision #{revision}. "
            f"Previous fact-check found these issues:\n{fact_report}\n"
            f"Fix all issues while keeping the article structure."
        )

    agent = create_writer(llm)
    task = Task(
        description=(
            f"Write an article on: '{topic}'\n\n"
            f"Format: {preset.get('format_name', 'Habr technical article')}\n"
            f"Style: {preset.get('style', 'Technical, with code examples where relevant')}\n"
            f"Target length: {preset.get('target_length', '8000-15000 characters')}\n"
            f"Language: Russian\n\n"
            f"Sources to use:\n{sources}\n"
            f"{revision_context}\n\n"
            f"Structure requirements:\n{preset.get('structure', '- Title, Introduction, Main sections, Conclusion')}\n"
        ),
        expected_output=(
            f"A complete article in Markdown format, "
            f"{preset.get('target_length', '8000-15000 characters')}. "
            f"Every claim must reference the provided sources."
        ),
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    result = crew.kickoff()

    version = state.get("draft_version", 0) + 1
    return {
        "draft": str(result),
        "draft_version": version,
        "status": "fact_checking",
        "log": _add_log(state, f"[WRITE] Draft v{version} generated ({len(str(result))} chars)"),
    }


# --- Node: Fact-Check ---

def fact_check_node(state: ArticleState) -> dict:
    """Fact-checker agent verifies claims in the draft."""
    draft = state.get("draft", "")
    sources = state.get("sources", "")
    llm = _get_llm()

    agent = create_fact_checker(llm)
    task = Task(
        description=(
            f"Fact-check this article:\n\n{draft}\n\n"
            f"Original sources used:\n{sources}\n\n"
            f"For each factual claim:\n"
            f"1. Extract the claim\n"
            f"2. Verify against sources and web search\n"
            f"3. Assign status: CONFIRMED / UNCONFIRMED / CONTRADICTED\n"
            f"4. Provide evidence\n\n"
            f"End with an overall accuracy score (1-10) on a separate line: SCORE: X"
        ),
        expected_output=(
            "A structured fact-check report with:\n"
            "- List of claims with verification status\n"
            "- Evidence for each claim\n"
            "- Overall accuracy score (1-10)\n"
            "Format: SCORE: X on the last line"
        ),
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    result = str(crew.kickoff())

    # Extract score
    score = 7.0  # default
    for line in result.split("\n"):
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            try:
                score = float(line.split(":")[1].strip().split("/")[0].strip())
            except (ValueError, IndexError):
                pass

    return {
        "fact_check_report": result,
        "fact_check_score": score,
        "status": "reviewing",
        "log": _add_log(state, f"[FACT-CHECK] Score: {score}/10"),
    }


# --- Node: Review Gate (pure logic, no LLM) ---

def review_gate_node(state: ArticleState) -> dict:
    """Decide: accept, revise, or reject based on fact-check score."""
    score = state.get("fact_check_score", 0)
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", settings.max_revisions)
    threshold = settings.fact_check_pass_threshold

    if score >= threshold:
        verdict = "accept"
    elif revision_count >= max_revisions:
        verdict = "accept"  # accept after max revisions even if imperfect
    else:
        verdict = "revise"

    return {
        "review_verdict": verdict,
        "revision_count": revision_count + (1 if verdict == "revise" else 0),
        "log": _add_log(
            state,
            f"[REVIEW] Verdict: {verdict} (score={score}, revision={revision_count}/{max_revisions})"
        ),
    }


# --- Node: Edit ---

def edit_node(state: ArticleState) -> dict:
    """Editor agent polishes the article."""
    draft = state.get("draft", "")
    fact_report = state.get("fact_check_report", "")
    preset = _load_preset(state.get("preset", "habr"))
    llm = _get_llm()

    agent = create_editor(llm)
    task = Task(
        description=(
            f"Edit this article to publication quality:\n\n{draft}\n\n"
            f"Fact-check report (address all issues):\n{fact_report}\n\n"
            f"Requirements:\n"
            f"- Fix grammar and style issues\n"
            f"- Ensure clear structure with proper headings\n"
            f"- Add SEO elements: meta description, keywords\n"
            f"- Maintain {preset.get('style', 'professional')} tone\n"
            f"- Target length: {preset.get('target_length', '8000-15000 characters')}\n"
            f"- Language: Russian\n\n"
            f"After the article, add a section '## Changelog' listing every edit made."
        ),
        expected_output=(
            "The polished article in Markdown, followed by a Changelog section."
        ),
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    result = str(crew.kickoff())

    # Split article and changelog
    if "## Changelog" in result:
        parts = result.split("## Changelog", 1)
        article = parts[0].strip()
        changelog = parts[1].strip() if len(parts) > 1 else ""
    else:
        article = result
        changelog = "No changelog provided"

    return {
        "edited_text": article,
        "edit_changelog": changelog,
        "status": "editing",
        "log": _add_log(state, f"[EDIT] Article polished ({len(article)} chars)"),
    }


# --- Node: Publish ---

def publish_node(state: ArticleState) -> dict:
    """Finalize and publish the article."""
    article = state.get("edited_text", state.get("draft", ""))

    return {
        "final_article": article,
        "status": "published",
        "log": _add_log(state, f"[PUBLISH] Article published ({len(article)} chars)"),
    }
