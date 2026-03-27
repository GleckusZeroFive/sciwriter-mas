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
from app.agents.reviewer import create_reviewer

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
    """Researcher agent finds sources on the topic.

    Two modes:
    - Factory mode (pre_sources populated): summarize pre-collected data, skip web search
    - Normal mode: search the web and knowledge base
    """
    topic = state["topic"]
    preset = _load_preset(state.get("preset", "habr"))
    pre_sources = state.get("pre_sources", [])
    llm = _get_llm()

    # Factory mode: use pre-collected sources from DB
    if pre_sources:
        logger.info("[RESEARCH] Factory mode: %d pre-collected sources", len(pre_sources))
        source_text = "\n\n---\n\n".join(
            f"Source: [{s.get('source', 'unknown')}] {s.get('title', 'N/A')}\n"
            f"URL: {s.get('url', 'N/A')}\n"
            f"Summary: {s.get('summary', '')}\n"
            f"Content: {(s.get('content') or '')[:2000]}"
            for s in pre_sources
        )

        agent = create_researcher(llm)
        task = Task(
            description=(
                f"You have pre-collected sources about: '{topic}'\n\n"
                f"Extract a numbered list of KEY FACTS from these sources.\n"
                f"Use ONLY information that is EXPLICITLY stated in the source text.\n"
                f"Do NOT infer, guess, or add any numbers/specs not present in sources.\n\n"
                f"For each fact, write one line:\n"
                f"[FACT-N] <specific detail in Russian> (Source: <url>)\n\n"
                f"What counts as a fact:\n"
                f"- Exact numbers from the text: prices, specs, measurements\n"
                f"- Names mentioned: models, chips, tools, people\n"
                f"- Direct quotes (keep in original language)\n"
                f"- Problems described and solutions used\n"
                f"- Concrete steps the author took\n\n"
                f"CRITICAL: If a number or spec is NOT in the source text, do NOT include it.\n"
                f"Better to have 5 real facts than 20 made-up ones.\n\n"
                f"Pre-collected sources:\n{source_text}"
            ),
            expected_output=(
                "A numbered list of facts ONLY from source text.\n"
                "Format: [FACT-1] <detail> (Source: <url>)\n"
                "No invented numbers. No guesses. Only what sources say."
            ),
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()

        return {
            "sources": str(result),
            "source_count": len(pre_sources),
            "status": "writing",
            "log": _add_log(state, f"[RESEARCH] Summarized {len(pre_sources)} pre-collected sources on: {topic}"),
        }

    # Normal mode: web search
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
            f"CRITICAL RULES:\n"
            f"1. Use ONLY [FACT-N] items from the research brief. Reference them as [FACT-N] in text.\n"
            f"2. NEVER invent numbers, specs, prices, or measurements not in [FACT-N] items.\n"
            f"3. If you need a number but no [FACT-N] provides it, write 'по данным автора' without a number.\n"
            f"4. Code examples ONLY if a [FACT-N] contains code. Otherwise no code.\n"
            f"5. Do NOT add sections about topics not covered by [FACT-N] items.\n"
            f"6. Do NOT add Meta Description or Keywords sections.\n"
            f"7. Each fact reference: [FACT-N](source url).\n\n"
            f"Sources to use:\n{sources}\n"
            f"{revision_context}\n\n"
            f"Structure requirements:\n{preset.get('structure', '- Title, Introduction, Main sections, Conclusion')}\n"
        ),
        expected_output=(
            f"A complete article in Markdown format, "
            f"{preset.get('target_length', '8000-15000 characters')}. "
            f"Every claim must be backed by specific data from sources. "
            f"No Meta Description or Keywords sections."
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


# --- Node: Rate ---

def rate_node(state: ArticleState) -> dict:
    """Rater reads the draft and produces a structured checklist per section.

    One job only: evaluate and produce actionable feedback.
    Does NOT rewrite the article — that's Improver's job.
    """
    draft = state.get("draft", "")
    llm = _get_llm()

    agent = create_reviewer(llm)
    task = Task(
        description=(
            f"Here is an article:\n\n{draft}\n\n"
            f"Rate this article from 0 to 100.\n"
            f"Then for each section (heading), write one line of feedback.\n\n"
            f"Format your response EXACTLY like this:\n"
            f"RATING: [number]\n\n"
            f"CHECKLIST:\n"
            f"- Section [name]: [what could be better — be specific]\n"
            f"- Section [name]: [ok / what could be better]\n"
            f"- Duplicates: [list any repeated information across sections]\n"
            f"- Missing: [what important info is absent]\n"
        ),
        expected_output=(
            "RATING: [number]\n\nCHECKLIST:\n- Section ...: ...\n- Section ...: ..."
        ),
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    result = str(crew.kickoff())

    # Extract rating
    import re
    score = 70.0
    rating_match = re.search(r"RATING:\s*(\d+)", result)
    if rating_match:
        score = float(rating_match.group(1))

    return {
        "fact_check_report": result,  # reuse field for checklist
        "fact_check_score": score / 10.0,
        "status": "improving",
        "log": _add_log(state, f"[RATE] Score: {score}/100"),
    }


# --- Node: Validate Numbers (deterministic, no LLM) ---

def validate_numbers_node(state: ArticleState) -> dict:
    """Deterministic fact validation: remove any sentence with numbers but no [FACT-N] tag.

    This is pure Python, no LLM call. Rules:
    - Any line with a number/spec MUST have a [FACT-N] tag
    - If it doesn't → the line is removed (hallucination)
    - Removed lines are logged and added to Rater's checklist for Improver
    """
    draft = state.get("draft", "")
    checklist = state.get("fact_check_report", "")

    from app.factory.quality_gate import validate_facts_deterministic
    cleaned, removed = validate_facts_deterministic(draft)

    if removed:
        logger.info("[VALIDATE] Removed %d untagged lines with numbers", len(removed))
        # Add to checklist so Improver knows what was removed
        addition = (
            "\n\nREMOVED LINES (contained numbers without [FACT-N] source tag):\n"
            + "\n".join(f"- {r[:100]}" for r in removed)
            + "\n\nDo NOT re-add these claims. Only use facts from [FACT-N] tags."
        )
        checklist += addition
    else:
        logger.info("[VALIDATE] All numbered claims are tagged")

    return {
        "draft": cleaned,  # Replace draft with cleaned version
        "fact_check_report": checklist,
        "status": "improving",
        "log": _add_log(state, f"[VALIDATE] {len(removed)} untagged lines removed (deterministic)"),
    }


# --- Node: Improve ---

def improve_node(state: ArticleState) -> dict:
    """Improver takes the draft + checklist and produces an improved version.

    One job only: rewrite based on specific feedback.
    """
    draft = state.get("draft", "")
    checklist = state.get("fact_check_report", "")
    llm = _get_llm()

    agent = create_reviewer(llm)
    task = Task(
        description=(
            f"Here is an article:\n\n{draft}\n\n"
            f"Here is reviewer feedback:\n{checklist}\n\n"
            f"Improve the article based on the feedback above.\n"
            f"- Remove duplicate information across sections\n"
            f"- Integrate quotes into text naturally (no standalone 'quotes' sections)\n"
            f"- Remove sections that have no real content\n"
            f"- Keep all specific facts, numbers, and source links\n"
            f"- Output the full improved article in Markdown\n"
        ),
        expected_output=(
            "The full improved article in Markdown format. Nothing else."
        ),
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    result = str(crew.kickoff())

    return {
        "edited_text": result,
        "revision_count": state.get("revision_count", 0) + 1,
        "status": "final_rating",
        "log": _add_log(state, f"[IMPROVE] Article improved ({len(result)} chars)"),
    }


# --- Node: Final Rate ---

def final_rate_node(state: ArticleState) -> dict:
    """Final rater evaluates the improved article. If below threshold → back to Improver.

    Threshold is configurable. Default 60 (will be calibrated via benchmarks).
    """
    article = state.get("edited_text", state.get("draft", ""))
    llm = _get_llm()

    agent = create_reviewer(llm)
    task = Task(
        description=(
            f"Here is the final version of an article:\n\n{article}\n\n"
            f"Rate it from 0 to 100. Write one sentence explaining the rating."
            f"\n\nFormat:\nRATING: [number]\nEXPLANATION: [one sentence]"
        ),
        expected_output=(
            "RATING: [number]\nEXPLANATION: [one sentence]"
        ),
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    result = str(crew.kickoff())

    import re
    score = 75.0
    rating_match = re.search(r"RATING:\s*(\d+)", result)
    if rating_match:
        score = float(rating_match.group(1))

    revision_count = state.get("revision_count", 0)
    threshold = 60  # TODO: calibrate via benchmarks

    # Decide: accept or loop back
    if score >= threshold or revision_count >= 2:
        verdict = "accept"
    else:
        verdict = "revise"

    return {
        "fact_check_score": score / 10.0,
        "review_verdict": verdict,
        "status": "final_rated",
        "log": _add_log(state, f"[FINAL RATE] Score: {score}/100, verdict: {verdict} (revision {revision_count}/2)"),
    }


# --- Node: Publish ---

def publish_node(state: ArticleState) -> dict:
    """Finalize the article and save to DB if in factory mode."""
    article = state.get("edited_text", state.get("draft", ""))
    article_db_id = state.get("article_db_id")

    # Factory mode: update article in PostgreSQL
    if article_db_id:
        try:
            from app.factory.db import update_article
            from app.factory.quality_gate import check_level1, clean_artifacts, validate_facts_deterministic

            # Clean artifacts before saving
            cleaned = clean_artifacts(article)

            # Final deterministic validation — catch any numbers Improver re-introduced
            cleaned, removed = validate_facts_deterministic(cleaned)
            if removed:
                logger.info("[PUBLISH] Final validator removed %d untagged lines", len(removed))

            # Run quality gate (min 3000 chars for factory articles)
            report = check_level1(cleaned, min_length=3000)
            status = "ready" if report.passed else "quality_check"

            update_article(
                article_db_id,
                content_ru=cleaned,
                title_ru=cleaned.split("\n")[0].lstrip("# ").strip()[:200],
                char_count=len(cleaned),
                fact_check_score=state.get("fact_check_score"),
                revision_count=state.get("revision_count", 0),
                status=status,
                generation_log=state.get("log", []),
            )
            logger.info("[PUBLISH] Article id=%d saved to DB (status=%s, %d chars)",
                        article_db_id, status, len(cleaned))

            if not report.passed:
                logger.warning("[PUBLISH] Quality gate issues: %s", "; ".join(report.issues))

        except Exception as e:
            logger.error("[PUBLISH] Failed to save to DB: %s", e, exc_info=True)

    return {
        "final_article": article,
        "status": "published",
        "log": _add_log(state, f"[PUBLISH] Article finalized ({len(article)} chars)"),
    }
